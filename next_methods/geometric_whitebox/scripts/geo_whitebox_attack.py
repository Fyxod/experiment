#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


@dataclass
class AttackConfig:
    input: str
    prompt: str
    output_dir: str
    seed: int = 7
    max_size: int = 512
    diffusion_steps: int = 4
    guidance_scale: float = 1.0
    max_sequence_length: int = 512
    text_encoder_out_layers: tuple[int, int, int] = (9, 18, 27)
    torch_dtype: str = "bfloat16"
    methods: tuple[str, ...] = ("dct",)
    objective: str = "hybrid"
    attack_iters: int = 40
    lr: float = 0.08
    max_disp_px: float = 1.0
    init_scale_px: float = 0.08
    grid_size: int = 6
    dct_size: int = 4
    rbf_size: int = 5
    rbf_sigma: float = 0.55
    tps_size: int = 5
    piecewise_size: int = 5
    padding_mode: str = "reflection"
    face_mask: str = "none"
    face_center_x: float = 0.5
    face_center_y: float = 0.5
    face_radius_x: float = 0.36
    face_radius_y: float = 0.50
    use_gates: bool = False
    gate_init: float = 2.0
    lambda_visual: float = 30.0
    lambda_disp: float = 0.10
    lambda_smooth: float = 0.20
    lambda_fold: float = 5.0
    lambda_method_l1: float = 0.02
    lambda_gate_l1: float = 0.01
    hybrid_vae_weight: float = 0.20
    objective_scale: float = 1.0
    feature_layers: tuple[int, ...] = (0, 4, 8)
    feature_source: str = "blocks"
    objective_timestep: int = 0
    objective_timesteps: tuple[int, ...] = (0,)
    edge_falloff_px: float = 16.0
    save_theta: bool = True


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resize_for_flux(image: Image.Image, max_size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    width, height = image.size
    scale = min(max_size / max(width, height), 1.0)
    width = max(64, int(width * scale) // 16 * 16)
    height = max(64, int(height * scale) // 16 * 16)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def pil_to_tensor01(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor01_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "RGB")


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        x = a[:, :, channel].astype(np.float64)
        y = b[:, :, channel].astype(np.float64)
        mux = gaussian_filter(x, sigma=1.5)
        muy = gaussian_filter(y, sigma=1.5)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = gaussian_filter(x * x, sigma=1.5) - mux2
        sigy2 = gaussian_filter(y * y, sigma=1.5) - muy2
        sigxy = gaussian_filter(x * y, sigma=1.5) - muxy
        numerator = (2.0 * muxy + c1) * (2.0 * sigxy + c2)
        denominator = (mux2 + muy2 + c1) * (sigx2 + sigy2 + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))


def image_metrics(a: Image.Image, b: Image.Image) -> dict[str, float]:
    aa = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    bb = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    h = min(aa.shape[0], bb.shape[0])
    w = min(aa.shape[1], bb.shape[1])
    aa = aa[:h, :w]
    bb = bb[:h, :w]
    diff = aa - bb
    mse = float(np.mean(diff * diff))
    psnr = math.inf if mse <= 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))
    return {
        "psnr": psnr,
        "ssim": ssim_rgb(aa, bb),
        "l2": float(np.sqrt(mse)),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def load_pipe(dtype: torch.dtype, device: torch.device):
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    for module_name in ("vae", "text_encoder", "transformer"):
        module = getattr(pipe, module_name, None)
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    return pipe


def retrieve_timesteps(pipe: Any, steps: int, device: torch.device, image_seq_len: int):
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    pipe.scheduler.set_begin_index(0)
    return timesteps


def encode_condition(pipe: Any, image01: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    image = (image01 * 2.0 - 1.0).to(dtype=pipe.vae.dtype)
    return pipe.prepare_image_latents(
        images=[image],
        batch_size=1,
        generator=generator,
        device=image.device,
        dtype=pipe.vae.dtype,
    )


def transformer_pred(
    pipe: Any,
    image01: torch.Tensor,
    fixed: dict[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_latents, image_latent_ids = encode_condition(pipe, image01, generator)
    latent_model_input = torch.cat([fixed["latents"], image_latents], dim=1).to(pipe.transformer.dtype)
    latent_image_ids = torch.cat([fixed["latent_ids"], image_latent_ids], dim=1)
    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=fixed["timestep"],
        guidance=None,
        encoder_hidden_states=fixed["prompt_embeds"],
        txt_ids=fixed["text_ids"],
        img_ids=latent_image_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    return noise_pred[:, : fixed["latents"].shape[1] :], image_latents


def summarize_feature(value: Any) -> torch.Tensor | None:
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        value = value[0]
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.float()
    if tensor.ndim >= 3:
        dims = tuple(range(1, tensor.ndim - 1))
        mean = tensor.mean(dim=dims)
        std = tensor.std(dim=dims, unbiased=False)
        return torch.cat([mean, std], dim=-1)
    return tensor.flatten(1)


def select_feature_modules(transformer: Any, layers: tuple[int, ...], source: str) -> list[tuple[str, torch.nn.Module]]:
    source = source.strip().lower()
    selected: list[tuple[str, torch.nn.Module]] = []
    if source == "attention":
        attention = [
            (name, module)
            for name, module in transformer.named_modules()
            if name and ("attn" in name.lower() or "attention" in name.lower())
        ]
        if attention:
            indexes = layers if layers else (0, len(attention) // 2, len(attention) - 1)
            for index in indexes:
                idx = index if index >= 0 else len(attention) + index
                if 0 <= idx < len(attention):
                    selected.append(attention[idx])
            return selected
    for attr in ("transformer_blocks", "single_transformer_blocks"):
        blocks = getattr(transformer, attr, None)
        if blocks is None:
            continue
        total = len(blocks)
        indexes = layers if layers else (0, total // 2, total - 1)
        for index in indexes:
            idx = index if index >= 0 else total + index
            if 0 <= idx < total:
                selected.append((f"{attr}.{idx}", blocks[idx]))
        if selected:
            return selected
    return selected


def transformer_pred_features(
    pipe: Any,
    image01: torch.Tensor,
    fixed: dict[str, Any],
    generator: torch.Generator,
    feature_layers: tuple[int, ...],
    feature_source: str,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    image_latents, image_latent_ids = encode_condition(pipe, image01, generator)
    latent_model_input = torch.cat([fixed["latents"], image_latents], dim=1).to(pipe.transformer.dtype)
    latent_image_ids = torch.cat([fixed["latent_ids"], image_latent_ids], dim=1)
    features: list[torch.Tensor] = []
    handles = []

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        summary = summarize_feature(output)
        if summary is not None and torch.isfinite(summary).all():
            features.append(summary)

    for _name, module in select_feature_modules(pipe.transformer, feature_layers, feature_source):
        handles.append(module.register_forward_hook(hook))
    try:
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=fixed["timestep"],
            guidance=None,
            encoder_hidden_states=fixed["prompt_embeds"],
            txt_ids=fixed["text_ids"],
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    finally:
        for handle in handles:
            handle.remove()
    return noise_pred[:, : fixed["latents"].shape[1] :], image_latents, features


def denoise_latents(
    pipe: Any,
    image01: torch.Tensor,
    fixed: dict[str, Any],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_latents, image_latent_ids = encode_condition(pipe, image01, generator)
    latents = fixed["latents"].clone()
    latent_ids = fixed["latent_ids"]
    pipe.scheduler.set_begin_index(0)
    if hasattr(pipe.scheduler, "_step_index"):
        pipe.scheduler._step_index = None
    for t in fixed["denoise_timesteps_raw"]:
        timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
        latent_model_input = torch.cat([latents, image_latents], dim=1).to(pipe.transformer.dtype)
        latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            guidance=None,
            encoder_hidden_states=fixed["prompt_embeds"],
            txt_ids=fixed["text_ids"],
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.shape[1] :]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents.float(), image_latents


def base_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def dct_basis(size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    basis = []
    for ky in range(size):
        for kx in range(size):
            if ky == 0 and kx == 0:
                continue
            by = torch.cos(math.pi * ky * (yy + 0.5) / height)
            bx = torch.cos(math.pi * kx * (xx + 0.5) / width)
            b = by * bx
            b = b / b.square().mean().sqrt().clamp_min(1e-6)
            basis.append(b)
    return torch.stack(basis, dim=0)


def tps_interpolation_matrix(size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, size, device=device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, size, device=device, dtype=torch.float32)
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    controls = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1)
    n = controls.shape[0]

    diff = controls[:, None, :] - controls[None, :, :]
    r2 = diff.square().sum(dim=-1)
    k = r2 * torch.log(r2.clamp_min(1e-8))
    k = k + torch.eye(n, device=device, dtype=torch.float32) * 1e-5
    p = torch.cat([torch.ones((n, 1), device=device), controls], dim=1)
    upper = torch.cat([k, p], dim=1)
    lower = torch.cat([p.T, torch.zeros((3, 3), device=device)], dim=1)
    system = torch.cat([upper, lower], dim=0)
    inv = torch.linalg.inv(system)

    qy = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    qx = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(qy, qx, indexing="ij")
    query = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    qdiff = query[:, None, :] - controls[None, :, :]
    qr2 = qdiff.square().sum(dim=-1)
    qk = qr2 * torch.log(qr2.clamp_min(1e-8))
    qp = torch.cat([torch.ones((query.shape[0], 1), device=device), query], dim=1)
    eval_matrix = torch.cat([qk, qp], dim=1)
    return eval_matrix @ inv[:, :n]


def piecewise_barycentric(size: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ys = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    xs = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    controls = np.stack([cx.reshape(-1), cy.reshape(-1)], axis=-1)
    tri = Delaunay(controls)
    qy = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    qx = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    yy, xx = np.meshgrid(qy, qx, indexing="ij")
    query = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    simplex = tri.find_simplex(query)
    simplex = np.maximum(simplex, 0)
    transform = tri.transform[simplex]
    delta = query - transform[:, 2, :]
    bary_first = np.einsum("pij,pj->pi", transform[:, :2, :], delta)
    bary = np.concatenate([bary_first, 1.0 - bary_first.sum(axis=1, keepdims=True)], axis=1)
    indices = tri.simplices[simplex].astype(np.int64)
    return (
        torch.from_numpy(indices).to(device=device),
        torch.from_numpy(bary.astype(np.float32)).to(device=device),
    )


def detect_face_fraction(image: Image.Image) -> tuple[float, float, float, float] | None:
    try:
        import cv2

        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None
        faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        width, height = image.size
        cx = (x + 0.5 * w) / max(width, 1)
        cy = (y + 0.48 * h) / max(height, 1)
        rx = min(0.48, max(0.22, 0.70 * w / max(width, 1)))
        ry = min(0.62, max(0.30, 0.82 * h / max(height, 1)))
        return float(cx), float(cy), float(rx), float(ry)
    except Exception:
        return None


class GeometryAttack(torch.nn.Module):
    def __init__(self, cfg: AttackConfig, height: int, width: int, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.height = height
        self.width = width
        self.device_ref = device
        self.methods = tuple(method.strip() for method in cfg.methods if method.strip())
        self.register_buffer("base", base_grid(height, width, device))
        yy = torch.linspace(-1.0, 1.0, height, device=device).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, 1, width)
        self.register_buffer("yy", yy)
        self.register_buffer("xx", xx)
        self.register_buffer("edge_mask", self.make_edge_mask(cfg.edge_falloff_px))
        self.register_buffer("local_mask", self.make_local_mask(cfg.face_mask))

        gen = torch.Generator(device=device).manual_seed(cfg.seed + 991)
        if cfg.use_gates:
            self.method_gate_raw = torch.nn.Parameter(torch.full((len(self.methods),), cfg.gate_init, device=device))
        if "rolling" in self.methods:
            self.rolling_amp = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * cfg.init_scale_px)
            self.rolling_freq_raw = torch.nn.Parameter(torch.zeros((), device=device))
            self.rolling_phase = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * 0.10)
            self.rolling_shear = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * cfg.init_scale_px)
            self.rolling_accel = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * cfg.init_scale_px)
        if "affine" in self.methods:
            self.affine_tx = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * cfg.init_scale_px)
            self.affine_ty = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * cfg.init_scale_px)
            self.affine_rot = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * 0.002)
            self.affine_sx = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * 0.002)
            self.affine_sy = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * 0.002)
            self.affine_shear = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * 0.002)
        if "radial" in self.methods or "lens" in self.methods:
            self.radial_k1 = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * 0.002)
            self.radial_k2 = torch.nn.Parameter(torch.randn((), device=device, generator=gen) * 0.001)
            self.radial_cx = torch.nn.Parameter(torch.zeros((), device=device))
            self.radial_cy = torch.nn.Parameter(torch.zeros((), device=device))
        if "homography" in self.methods or "projective" in self.methods:
            init = torch.randn((4, 2), device=device, generator=gen) * cfg.init_scale_px
            self.homography_corner_raw = torch.nn.Parameter(init)
            corners = torch.tensor(
                [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                device=device,
                dtype=torch.float32,
            )
            self.register_buffer("homography_corners", corners)
        if "grid" in self.methods:
            init = torch.randn((1, 2, cfg.grid_size, cfg.grid_size), device=device, generator=gen) * cfg.init_scale_px
            self.grid_raw = torch.nn.Parameter(init)
        if "dct" in self.methods:
            basis = dct_basis(cfg.dct_size, height, width, device)
            self.register_buffer("dct_basis", basis)
            init = torch.randn((2, basis.shape[0]), device=device, generator=gen) * cfg.init_scale_px
            self.dct_coeff = torch.nn.Parameter(init)
        if "rbf" in self.methods:
            size = cfg.rbf_size
            init = torch.randn((1, 2, size, size), device=device, generator=gen) * cfg.init_scale_px
            border = torch.ones((1, 2, size, size), device=device)
            border[:, :, 0, :] = 0
            border[:, :, -1, :] = 0
            border[:, :, :, 0] = 0
            border[:, :, :, -1] = 0
            self.rbf_raw = torch.nn.Parameter(init * border)
            self.register_buffer("rbf_border_mask", border)
            centers_y = torch.linspace(-1.0, 1.0, size, device=device)
            centers_x = torch.linspace(-1.0, 1.0, size, device=device)
            cy, cx = torch.meshgrid(centers_y, centers_x, indexing="ij")
            centers = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1)
            self.register_buffer("rbf_centers", centers)
        if "tps" in self.methods or "tps_exact" in self.methods:
            size = cfg.tps_size
            init = torch.randn((1, 2, size, size), device=device, generator=gen) * cfg.init_scale_px
            border = torch.ones((1, 2, size, size), device=device)
            border[:, :, 0, :] = 0
            border[:, :, -1, :] = 0
            border[:, :, :, 0] = 0
            border[:, :, :, -1] = 0
            self.tps_raw = torch.nn.Parameter(init * border)
            self.register_buffer("tps_border_mask", border)
            self.register_buffer("tps_matrix", tps_interpolation_matrix(size, height, width, device))
        if "piecewise" in self.methods or "delaunay" in self.methods:
            size = cfg.piecewise_size
            init = torch.randn((1, 2, size, size), device=device, generator=gen) * cfg.init_scale_px
            border = torch.ones((1, 2, size, size), device=device)
            border[:, :, 0, :] = 0
            border[:, :, -1, :] = 0
            border[:, :, :, 0] = 0
            border[:, :, :, -1] = 0
            self.piecewise_raw = torch.nn.Parameter(init * border)
            self.register_buffer("piecewise_border_mask", border)
            indices, weights = piecewise_barycentric(size, height, width, device)
            self.register_buffer("piecewise_indices", indices)
            self.register_buffer("piecewise_weights", weights)

    def make_edge_mask(self, margin_px: float) -> torch.Tensor:
        if margin_px <= 0:
            return torch.ones((1, 1, self.height, self.width), device=self.device_ref)
        y = torch.arange(self.height, device=self.device_ref, dtype=torch.float32)
        x = torch.arange(self.width, device=self.device_ref, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.minimum(torch.minimum(xx, self.width - 1 - xx), torch.minimum(yy, self.height - 1 - yy))
        t = (dist / margin_px).clamp(0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        return smooth.view(1, 1, self.height, self.width)

    def make_local_mask(self, name: str) -> torch.Tensor:
        normalized = name.strip().lower()
        if normalized in {"", "none", "global"}:
            return torch.ones((1, 1, self.height, self.width), device=self.device_ref)
        cx = 2.0 * self.cfg.face_center_x - 1.0
        cy = 2.0 * self.cfg.face_center_y - 1.0
        rx = max(2.0 * self.cfg.face_radius_x, 1e-3)
        ry = max(2.0 * self.cfg.face_radius_y, 1e-3)
        if normalized in {"upper_face", "eyes", "hairline"}:
            cy -= 0.28 * ry
            ry *= 0.58
        elif normalized in {"lower_face", "mouth", "jaw"}:
            cy += 0.24 * ry
            ry *= 0.58
        d = (((self.xx - cx) / rx).square() + ((self.yy - cy) / ry).square()).sqrt()
        inner, outer = 0.78, 1.22
        t = ((d - inner) / max(outer - inner, 1e-6)).clamp(0.0, 1.0)
        smooth = 1.0 - t * t * (3.0 - 2.0 * t)
        return smooth.clamp(0.0, 1.0)

    def bounded(self, value: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        scale = self.cfg.max_disp_px if scale is None else scale
        return scale * torch.tanh(value / max(scale, 1e-6))

    def rolling_field(self) -> torch.Tensor:
        y = self.yy
        amp = self.bounded(self.rolling_amp)
        shear = self.bounded(self.rolling_shear)
        accel = self.bounded(self.rolling_accel)
        freq = 0.5 + 3.0 * torch.sigmoid(self.rolling_freq_raw)
        shift = amp * torch.sin(math.pi * freq * y + self.rolling_phase)
        shift = shift + shear * y + accel * (y.square() - 1.0 / 3.0)
        dx = shift.expand(1, 1, self.height, self.width)
        dy = torch.zeros_like(dx)
        return torch.cat([dx, dy], dim=1)

    def affine_field(self) -> torch.Tensor:
        x = self.base[..., 0]
        y = self.base[..., 1]
        tx = self.bounded(self.affine_tx)
        ty = self.bounded(self.affine_ty)
        rot = 0.015 * torch.tanh(self.affine_rot / 0.015)
        sx = 0.010 * torch.tanh(self.affine_sx / 0.010)
        sy = 0.010 * torch.tanh(self.affine_sy / 0.010)
        shear = 0.010 * torch.tanh(self.affine_shear / 0.010)
        mx = (1.0 + sx) * x - rot * y + shear * y
        my = rot * x + (1.0 + sy) * y
        dx = (mx - x) * max(self.width - 1, 1) / 2.0 + tx
        dy = (my - y) * max(self.height - 1, 1) / 2.0 + ty
        return torch.stack([dx, dy], dim=1)

    def radial_field(self) -> torch.Tensor:
        cx = 0.08 * torch.tanh(self.radial_cx / 0.08)
        cy = 0.08 * torch.tanh(self.radial_cy / 0.08)
        x = self.base[..., 0] - cx
        y = self.base[..., 1] - cy
        r2 = x.square() + y.square()
        k1 = 0.018 * torch.tanh(self.radial_k1 / 0.018)
        k2 = 0.008 * torch.tanh(self.radial_k2 / 0.008)
        scale = k1 * r2 + k2 * r2.square()
        dx = x * scale * max(self.width - 1, 1) / 2.0
        dy = y * scale * max(self.height - 1, 1) / 2.0
        return torch.stack([dx, dy], dim=1)

    def grid_field(self) -> torch.Tensor:
        raw = self.bounded(self.grid_raw)
        return F.interpolate(raw, size=(self.height, self.width), mode="bicubic", align_corners=True)

    def dct_field(self) -> torch.Tensor:
        coeff = self.dct_coeff
        field = torch.einsum("ck,khw->chw", coeff, self.dct_basis).unsqueeze(0)
        return self.bounded(field)

    def homography_field(self) -> torch.Tensor:
        offsets_px = self.bounded(self.homography_corner_raw)
        offsets_norm = torch.empty_like(offsets_px)
        offsets_norm[:, 0] = 2.0 * offsets_px[:, 0] / max(self.width - 1, 1)
        offsets_norm[:, 1] = 2.0 * offsets_px[:, 1] / max(self.height - 1, 1)
        src = self.homography_corners
        dst = self.homography_corners + offsets_norm
        rows = []
        rhs = []
        for idx in range(4):
            x, y = src[idx, 0], src[idx, 1]
            u, v = dst[idx, 0], dst[idx, 1]
            zero = torch.zeros((), device=self.device_ref)
            one = torch.ones((), device=self.device_ref)
            rows.append(torch.stack([x, y, one, zero, zero, zero, -u * x, -u * y]))
            rhs.append(u)
            rows.append(torch.stack([zero, zero, zero, x, y, one, -v * x, -v * y]))
            rhs.append(v)
        matrix = torch.stack(rows, dim=0)
        target = torch.stack(rhs, dim=0)
        coeff = torch.linalg.solve(matrix + torch.eye(8, device=self.device_ref) * 1e-6, target)
        a, b, c, d, e, f, g, h = coeff
        x = self.base[..., 0]
        y = self.base[..., 1]
        denom = (g * x + h * y + 1.0).clamp_min(0.25)
        mx = (a * x + b * y + c) / denom
        my = (d * x + e * y + f) / denom
        dx = (mx - x) * max(self.width - 1, 1) / 2.0
        dy = (my - y) * max(self.height - 1, 1) / 2.0
        return torch.stack([dx, dy], dim=1)

    def rbf_field(self) -> torch.Tensor:
        raw = self.bounded(self.rbf_raw * self.rbf_border_mask)
        values = raw.reshape(1, 2, -1)
        coords = torch.stack(
            [
                self.xx.expand(1, 1, self.height, self.width).reshape(-1),
                self.yy.expand(1, 1, self.height, self.width).reshape(-1),
            ],
            dim=-1,
        )
        diff = coords[:, None, :] - self.rbf_centers[None, :, :]
        weights = torch.exp(-diff.square().sum(dim=-1) / (2.0 * self.cfg.rbf_sigma * self.cfg.rbf_sigma))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        dense = torch.einsum("pn,bcn->bcp", weights, values).reshape(1, 2, self.height, self.width)
        return dense

    def tps_field(self) -> torch.Tensor:
        raw = self.bounded(self.tps_raw * self.tps_border_mask)
        values = raw.reshape(1, 2, -1)
        dense = torch.einsum("pn,bcn->bcp", self.tps_matrix, values).reshape(1, 2, self.height, self.width)
        return self.bounded(dense)

    def piecewise_field(self) -> torch.Tensor:
        raw = self.bounded(self.piecewise_raw * self.piecewise_border_mask)
        values = raw.reshape(1, 2, -1)
        gathered = values[:, :, self.piecewise_indices.reshape(-1)].reshape(1, 2, -1, 3)
        dense = (gathered * self.piecewise_weights.view(1, 1, -1, 3)).sum(dim=-1)
        return dense.reshape(1, 2, self.height, self.width)

    def gate_values(self) -> dict[str, float]:
        if not hasattr(self, "method_gate_raw"):
            return {name: 1.0 for name in self.methods}
        gates = torch.sigmoid(self.method_gate_raw.detach()).cpu().tolist()
        return {name: float(gates[idx]) for idx, name in enumerate(self.methods)}

    def gate_l1(self) -> torch.Tensor:
        if not hasattr(self, "method_gate_raw"):
            return torch.zeros((), device=self.device_ref)
        return torch.sigmoid(self.method_gate_raw).mean()

    def compose_fields(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fields: dict[str, torch.Tensor] = {}
        if "rolling" in self.methods:
            fields["rolling"] = self.rolling_field()
        if "affine" in self.methods:
            fields["affine"] = self.affine_field()
        if "radial" in self.methods or "lens" in self.methods:
            fields["radial"] = self.radial_field()
        if "homography" in self.methods or "projective" in self.methods:
            fields["homography"] = self.homography_field()
        if "grid" in self.methods:
            fields["grid"] = self.grid_field()
        if "dct" in self.methods:
            fields["dct"] = self.dct_field()
        if "rbf" in self.methods:
            fields["rbf"] = self.rbf_field()
        if "tps" in self.methods or "tps_exact" in self.methods:
            fields["tps"] = self.tps_field()
        if "piecewise" in self.methods or "delaunay" in self.methods:
            fields["piecewise"] = self.piecewise_field()
        if not fields:
            raise ValueError("No geometry methods enabled")
        if hasattr(self, "method_gate_raw"):
            gates = torch.sigmoid(self.method_gate_raw)
            gate_lookup = {name: idx for idx, name in enumerate(self.methods)}
            gated_fields = {}
            for name, value in fields.items():
                aliases = {
                    "tps": ("tps", "tps_exact"),
                    "piecewise": ("piecewise", "delaunay"),
                    "homography": ("homography", "projective"),
                    "radial": ("radial", "lens"),
                }.get(name, (name,))
                idx = next((gate_lookup[alias] for alias in aliases if alias in gate_lookup), None)
                gated_fields[name] = value if idx is None else value * gates[idx]
            fields = gated_fields
        masked_fields = {name: value * self.edge_mask * self.local_mask for name, value in fields.items()}
        disp = sum(masked_fields.values())
        magnitude = disp.square().sum(dim=1, keepdim=True).sqrt()
        scale = torch.clamp(self.cfg.max_disp_px / magnitude.clamp_min(1e-6), max=1.0)
        return disp * scale, masked_fields

    def warp(self, image01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        disp, fields = self.compose_fields()
        dx_norm = 2.0 * disp[:, 0] / max(self.width - 1, 1)
        dy_norm = 2.0 * disp[:, 1] / max(self.height - 1, 1)
        grid = self.base.clone()
        grid[..., 0] = grid[..., 0] + dx_norm
        grid[..., 1] = grid[..., 1] + dy_norm
        warped = F.grid_sample(
            image01,
            grid,
            mode="bilinear",
            padding_mode=self.cfg.padding_mode,
            align_corners=True,
        ).clamp(0.0, 1.0)
        return warped, disp, fields

    def rolling_params(self) -> dict[str, float]:
        if "rolling" not in self.methods:
            return {}
        with torch.no_grad():
            return {
                "amp_px": float(self.bounded(self.rolling_amp).detach().cpu()),
                "freq": float((0.5 + 3.0 * torch.sigmoid(self.rolling_freq_raw)).detach().cpu()),
                "phase": float(self.rolling_phase.detach().cpu()),
                "shear_px": float(self.bounded(self.rolling_shear).detach().cpu()),
                "accel_px": float(self.bounded(self.rolling_accel).detach().cpu()),
            }

    def project_parameters(self) -> bool:
        ok = True
        with torch.no_grad():
            for name, param in self.named_parameters():
                finite = torch.isfinite(param).all()
                ok = ok and bool(finite.detach().cpu())
                if not finite:
                    param.nan_to_num_(nan=0.0, posinf=self.cfg.max_disp_px, neginf=-self.cfg.max_disp_px)
                if name == "rolling_phase":
                    param.clamp_(-math.pi, math.pi)
                elif name == "rolling_freq_raw":
                    param.clamp_(-4.0, 4.0)
                else:
                    param.clamp_(-3.0 * self.cfg.max_disp_px, 3.0 * self.cfg.max_disp_px)
        return ok


def tv_loss(disp: torch.Tensor) -> torch.Tensor:
    dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
    dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def jacobian_stats(disp: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    dx = disp[:, 0]
    dy = disp[:, 1]
    ddx_dx = dx[:, :, 2:] - dx[:, :, :-2]
    ddx_dx = F.pad(ddx_dx / 2.0, (1, 1, 0, 0))
    ddx_dy = dx[:, 2:, :] - dx[:, :-2, :]
    ddx_dy = F.pad(ddx_dy / 2.0, (0, 0, 1, 1))
    ddy_dx = dy[:, :, 2:] - dy[:, :, :-2]
    ddy_dx = F.pad(ddy_dx / 2.0, (1, 1, 0, 0))
    ddy_dy = dy[:, 2:, :] - dy[:, :-2, :]
    ddy_dy = F.pad(ddy_dy / 2.0, (0, 0, 1, 1))
    det = (1.0 + ddx_dx) * (1.0 + ddy_dy) - ddx_dy * ddy_dx
    penalty = F.relu(0.2 - det).square().mean()
    with torch.no_grad():
        stats = {
            "jacobian_det_min": float(det.min().detach().cpu()),
            "jacobian_det_mean": float(det.mean().detach().cpu()),
            "foldover_fraction_det_below_0": float((det < 0).float().mean().detach().cpu()),
            "low_det_fraction_below_0p2": float((det < 0.2).float().mean().detach().cpu()),
        }
    return penalty, stats


def displacement_stats(disp: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        value = disp.detach().float().cpu()
        dx = value[:, 0]
        dy = value[:, 1]
        mag = (dx.square() + dy.square()).sqrt()
        return {
            "max_dx_px": float(dx.abs().max()),
            "max_dy_px": float(dy.abs().max()),
            "max_magnitude_px": float(mag.max()),
            "mean_magnitude_px": float(mag.mean()),
            "p95_magnitude_px": float(torch.quantile(mag.reshape(-1), 0.95)),
        }


def method_stats(fields: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return {name: displacement_stats(field) for name, field in fields.items()}


def displacement_visuals(out: Path, disp: torch.Tensor) -> None:
    value = disp.detach().float().cpu().squeeze(0)
    dx, dy = value[0], value[1]
    mag = (dx.square() + dy.square()).sqrt()
    max_mag = max(float(mag.max()), 1e-6)
    rgb = torch.zeros((3, value.shape[1], value.shape[2]), dtype=torch.float32)
    rgb[0] = (dx / max_mag * 0.5 + 0.5).clamp(0, 1)
    rgb[1] = (dy / max_mag * 0.5 + 0.5).clamp(0, 1)
    rgb[2] = (mag / max_mag).clamp(0, 1)
    tensor01_to_pil(rgb.unsqueeze(0)).save(out / "flow.png")
    tensor01_to_pil((mag / 4.0).clamp(0, 1).view(1, 1, value.shape[1], value.shape[2]).repeat(1, 3, 1, 1)).save(
        out / "displacement_x16.png"
    )


def make_sheet(out: Path) -> None:
    items = [
        ("original", out / "original.png"),
        ("perturbed", out / "perturbed.png"),
        ("clean edit", out / "original_diffused.png"),
        ("geo edit", out / "perturbed_diffused.png"),
        ("flow", out / "flow.png"),
    ]
    thumbs = []
    for label, path in items:
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((190, 190), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (206, 226), "white")
        canvas.paste(im, ((206 - im.width) // 2, 8))
        ImageDraw.Draw(canvas).text((8, 204), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    if not thumbs:
        return
    sheet = Image.new("RGB", (206 * len(thumbs), 226), (235, 235, 235))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (idx * 206, 0))
    sheet.save(out / "sheet.jpg", quality=92)


@torch.inference_mode()
def diffuse(pipe: Any, image: Image.Image, cfg: AttackConfig, device: torch.device) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    result = pipe(
        image=image,
        prompt=cfg.prompt,
        height=image.height,
        width=image.width,
        num_inference_steps=cfg.diffusion_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
        max_sequence_length=cfg.max_sequence_length,
        text_encoder_out_layers=cfg.text_encoder_out_layers,
    )
    return result.images[0].convert("RGB")


def selected_timestep_tensors(timesteps: torch.Tensor, cfg: AttackConfig) -> list[torch.Tensor]:
    requested = cfg.objective_timesteps if cfg.objective_timesteps else (cfg.objective_timestep,)
    if any(index < 0 for index in requested):
        indexes = list(range(len(timesteps)))
    else:
        indexes = [max(0, min(int(index), len(timesteps) - 1)) for index in requested]
    selected = []
    for index in dict.fromkeys(indexes):
        selected.append(timesteps[index].expand(1))
    return selected


def prepare_clean_internals(
    pipe: Any,
    clean01: torch.Tensor,
    cfg: AttackConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[torch.Tensor], torch.Tensor, list[list[torch.Tensor]]]:
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=cfg.prompt,
            device=device,
            max_sequence_length=cfg.max_sequence_length,
            text_encoder_out_layers=cfg.text_encoder_out_layers,
        )
        num_channels_latents = pipe.transformer.config.in_channels // 4
        height, width = clean01.shape[-2:]
        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )
        timesteps = retrieve_timesteps(pipe, cfg.diffusion_steps, device, latents.shape[1])
        selected_timesteps = [t.to(prompt_embeds.dtype) / 1000 for t in selected_timestep_tensors(timesteps, cfg)]
        fixed = {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "latents": latents.detach(),
            "latent_ids": latent_ids,
            "selected_timesteps": selected_timesteps,
            "denoise_timesteps_raw": [t.detach() for t in timesteps],
        }
        clean_preds: list[torch.Tensor] = []
        clean_features: list[list[torch.Tensor]] = []
        clean_condition = None
        for timestep in selected_timesteps:
            fixed_t = {**fixed, "timestep": timestep}
            if cfg.objective in {"hidden_feature", "attention_proxy", "hybrid_hidden"}:
                clean_pred, condition, features = transformer_pred_features(
                    pipe,
                    clean01,
                    fixed_t,
                    generator,
                    cfg.feature_layers,
                    "attention" if cfg.objective == "attention_proxy" else cfg.feature_source,
                )
                clean_features.append([feature.detach().float() for feature in features])
            else:
                clean_pred, condition = transformer_pred(pipe, clean01, fixed_t, generator)
                clean_features.append([])
            clean_preds.append(clean_pred.detach().float())
            clean_condition = condition.detach().float()
    assert clean_condition is not None
    return fixed, clean_preds, clean_condition, clean_features


def optimize_geometry(
    pipe: Any,
    clean01: torch.Tensor,
    cfg: AttackConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], GeometryAttack]:
    fixed, clean_preds, clean_condition, clean_features = prepare_clean_internals(pipe, clean01, cfg, device)
    clean_denoised = None
    if cfg.objective == "denoise_latent":
        with torch.no_grad():
            clean_denoised, _ = denoise_latents(pipe, clean01, fixed, torch.Generator(device=device).manual_seed(cfg.seed))
            clean_denoised = clean_denoised.detach().float()
    model = GeometryAttack(cfg, clean01.shape[-2], clean01.shape[-1], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {
        "utility": -float("inf"),
        "score": -float("inf"),
        "state": None,
        "disp": None,
        "image": None,
        "fields": None,
    }

    for iteration in range(1, cfg.attack_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        warped, disp, fields = model.warp(clean01)
        if cfg.objective == "denoise_latent":
            assert clean_denoised is not None
            denoised, condition = denoise_latents(pipe, warped, fixed, generator)
            pred_mse = (denoised.float() - clean_denoised).square().mean()
            vae_mse = (condition.float() - clean_condition).square().mean()
            objective = pred_mse
        else:
            pred_losses = []
            feature_losses = []
            condition = None
            for step_idx, (timestep, clean_pred) in enumerate(zip(fixed["selected_timesteps"], clean_preds)):
                fixed_t = {**fixed, "timestep": timestep}
                if cfg.objective in {"hidden_feature", "attention_proxy", "hybrid_hidden"}:
                    pred, condition, features = transformer_pred_features(
                        pipe,
                        warped,
                        fixed_t,
                        generator,
                        cfg.feature_layers,
                        "attention" if cfg.objective == "attention_proxy" else cfg.feature_source,
                    )
                    pairs = zip(features, clean_features[step_idx])
                    step_feature_losses = [(feature.float() - clean_feature).square().mean() for feature, clean_feature in pairs]
                    if step_feature_losses:
                        feature_losses.append(torch.stack(step_feature_losses).mean())
                else:
                    pred, condition = transformer_pred(pipe, warped, fixed_t, generator)
                pred_losses.append((pred.float() - clean_pred).square().mean())
            pred_mse = torch.stack(pred_losses).mean()
            assert condition is not None
            vae_mse = (condition.float() - clean_condition).square().mean()
            feature_mse = (
                torch.stack(feature_losses).mean()
                if feature_losses
                else torch.zeros((), device=device, dtype=torch.float32)
            )
            if cfg.objective == "vae":
                objective = vae_mse
            elif cfg.objective == "transformer_pred":
                objective = pred_mse
            elif cfg.objective in {"hidden_feature", "attention_proxy"}:
                objective = feature_mse
            elif cfg.objective == "hybrid_hidden":
                objective = feature_mse + 0.25 * pred_mse + cfg.hybrid_vae_weight * vae_mse
            elif cfg.objective == "hybrid":
                objective = pred_mse + cfg.hybrid_vae_weight * vae_mse
            else:
                raise ValueError(f"Unknown objective: {cfg.objective}")
        visual = (warped - clean01).square().mean()
        mag = disp.square().sum(dim=1).sqrt()
        disp_penalty = mag.square().mean() / max(cfg.max_disp_px * cfg.max_disp_px, 1e-6)
        smooth = tv_loss(disp) / max(cfg.max_disp_px, 1e-6)
        fold, jac_stats = jacobian_stats(disp)
        method_l1 = sum(field.abs().mean() for field in fields.values()) / max(len(fields), 1) / max(cfg.max_disp_px, 1e-6)
        gate_l1 = model.gate_l1()
        scaled_objective = cfg.objective_scale * objective
        loss = (
            -scaled_objective
            + cfg.lambda_visual * visual
            + cfg.lambda_disp * disp_penalty
            + cfg.lambda_smooth * smooth
            + cfg.lambda_fold * fold
            + cfg.lambda_method_l1 * method_l1
            + cfg.lambda_gate_l1 * gate_l1
        )
        if not torch.isfinite(loss) or not torch.isfinite(disp).all() or not torch.isfinite(warped).all():
            history.append({"iter": iteration, "stopped": True, "reason": "non_finite_loss_or_geometry"})
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        projected_ok = model.project_parameters()

        score = float(objective.detach().cpu())
        utility = float((-loss).detach().cpu())
        with torch.no_grad():
            finite_geometry = torch.isfinite(disp).all() and torch.isfinite(warped).all()
            if math.isfinite(utility) and finite_geometry and utility > best["utility"]:
                best = {
                    "utility": utility,
                    "score": score,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "disp": disp.detach().clone(),
                    "image": warped.detach().clone(),
                    "fields": {name: value.detach().clone() for name, value in fields.items()},
                }
            if iteration == 1 or iteration == cfg.attack_iters or iteration % max(1, cfg.attack_iters // 8) == 0:
                row = {
                    "iter": iteration,
                    "projected_parameters_finite": projected_ok,
                    "loss": float(loss.detach().cpu()),
                    "objective": score,
                    "scaled_objective": float(scaled_objective.detach().cpu()),
                    "pred_mse": float(pred_mse.detach().cpu()),
                    "vae_mse": float(vae_mse.detach().cpu()),
                    "feature_mse": float(feature_mse.detach().cpu()) if "feature_mse" in locals() else 0.0,
                    "visual_mse": float(visual.detach().cpu()),
                    "disp_penalty": float(disp_penalty.detach().cpu()),
                    "smooth": float(smooth.detach().cpu()),
                    "fold": float(fold.detach().cpu()),
                    "method_l1": float(method_l1.detach().cpu()),
                    "gate_l1": float(gate_l1.detach().cpu()),
                    **displacement_stats(disp),
                    **jac_stats,
                }
                history.append(row)
        for local_name in ("warped", "disp", "pred", "denoised", "condition", "loss"):
            if local_name in locals():
                del locals()[local_name]
        torch.cuda.empty_cache()

    assert best["state"] is not None and best["image"] is not None and best["disp"] is not None and best["fields"] is not None
    model.load_state_dict(best["state"])
    attack_report = {
        "history": history,
        "best_utility": best["utility"],
        "best_score": best["score"],
        "best_displacement": displacement_stats(best["disp"]),
        "best_method_displacement": method_stats(best["fields"]),
        "rolling_params": model.rolling_params(),
        "gate_values": model.gate_values(),
    }
    return best["image"], best["disp"], attack_report, model


def run_with_pipe(cfg: AttackConfig, pipe: Any, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    original = resize_for_flux(Image.open(cfg.input), cfg.max_size)
    detected = None
    if cfg.face_mask.strip().lower() not in {"", "none", "global"}:
        detected = detect_face_fraction(original)
        if detected is not None:
            cfg.face_center_x, cfg.face_center_y, cfg.face_radius_x, cfg.face_radius_y = detected
    write_json(out / "effective_config.json", asdict(cfg))
    original.save(out / "original.png")
    clean01 = pil_to_tensor01(original, device)
    perturbed01, disp, attack_report, model = optimize_geometry(pipe, clean01, cfg, device)
    perturbed = tensor01_to_pil(perturbed01)
    perturbed.save(out / "perturbed.png")
    displacement_visuals(out, disp)
    if cfg.save_theta:
        torch.save({"state_dict": model.state_dict(), "disp": disp.detach().cpu()}, out / "theta.pt")

    original_diffused = diffuse(pipe, original, cfg, device)
    perturbed_diffused = diffuse(pipe, perturbed, cfg, device)
    original_diffused.save(out / "original_diffused.png")
    perturbed_diffused.save(out / "perturbed_diffused.png")
    metrics = {
        "input": image_metrics(original, perturbed),
        "output": image_metrics(original_diffused, perturbed_diffused),
        "displacement": displacement_stats(disp),
    }
    _, jac = jacobian_stats(disp)
    metrics["jacobian"] = jac
    report = {
        "config": asdict(cfg),
        "model_id": MODEL_ID,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "detected_face_fraction": detected,
        "metrics": metrics,
        "attack": attack_report,
        "outputs": {
            "original": str(out / "original.png"),
            "perturbed": str(out / "perturbed.png"),
            "original_diffused": str(out / "original_diffused.png"),
            "perturbed_diffused": str(out / "perturbed_diffused.png"),
            "flow": str(out / "flow.png"),
            "displacement_x16": str(out / "displacement_x16.png"),
            "report": str(out / "report.json"),
            "effective_config": str(out / "effective_config.json"),
            "sheet": str(out / "sheet.jpg"),
        },
    }
    write_json(out / "report.json", report)
    make_sheet(out)
    return report


def parse_args() -> AttackConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--methods", default="dct")
    parser.add_argument(
        "--objective",
        choices=["vae", "transformer_pred", "hidden_feature", "attention_proxy", "hybrid", "hybrid_hidden", "denoise_latent"],
        default="hybrid",
    )
    parser.add_argument("--attack-iters", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--max-disp-px", type=float, default=1.0)
    parser.add_argument("--init-scale-px", type=float, default=0.08)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--dct-size", type=int, default=4)
    parser.add_argument("--rbf-size", type=int, default=5)
    parser.add_argument("--rbf-sigma", type=float, default=0.55)
    parser.add_argument("--tps-size", type=int, default=5)
    parser.add_argument("--piecewise-size", type=int, default=5)
    parser.add_argument("--padding-mode", choices=["reflection", "border", "zeros"], default="reflection")
    parser.add_argument("--face-mask", choices=["none", "global", "face", "upper_face", "eyes", "lower_face", "mouth", "jaw", "hairline"], default="none")
    parser.add_argument("--face-center-x", type=float, default=0.5)
    parser.add_argument("--face-center-y", type=float, default=0.5)
    parser.add_argument("--face-radius-x", type=float, default=0.36)
    parser.add_argument("--face-radius-y", type=float, default=0.50)
    parser.add_argument("--use-gates", action="store_true")
    parser.add_argument("--gate-init", type=float, default=2.0)
    parser.add_argument("--lambda-visual", type=float, default=30.0)
    parser.add_argument("--lambda-disp", type=float, default=0.10)
    parser.add_argument("--lambda-smooth", type=float, default=0.20)
    parser.add_argument("--lambda-fold", type=float, default=5.0)
    parser.add_argument("--lambda-method-l1", type=float, default=0.02)
    parser.add_argument("--lambda-gate-l1", type=float, default=0.01)
    parser.add_argument("--hybrid-vae-weight", type=float, default=0.20)
    parser.add_argument("--objective-scale", type=float, default=1.0)
    parser.add_argument("--feature-layers", default="0,4,8")
    parser.add_argument("--feature-source", choices=["blocks", "attention"], default="blocks")
    parser.add_argument("--objective-timestep", type=int, default=0)
    parser.add_argument("--objective-timesteps", default=None, help="Comma-separated timestep indexes, or -1 for all")
    parser.add_argument("--edge-falloff-px", type=float, default=16.0)
    parser.add_argument("--no-save-theta", action="store_true")
    args = parser.parse_args()
    return AttackConfig(
        input=args.input,
        prompt=args.prompt,
        output_dir=args.output_dir,
        seed=args.seed,
        max_size=args.max_size,
        diffusion_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
        methods=tuple(item.strip() for item in args.methods.split(",") if item.strip()),
        objective=args.objective,
        attack_iters=args.attack_iters,
        lr=args.lr,
        max_disp_px=args.max_disp_px,
        init_scale_px=args.init_scale_px,
        grid_size=args.grid_size,
        dct_size=args.dct_size,
        rbf_size=args.rbf_size,
        rbf_sigma=args.rbf_sigma,
        tps_size=args.tps_size,
        piecewise_size=args.piecewise_size,
        padding_mode=args.padding_mode,
        face_mask=args.face_mask,
        face_center_x=args.face_center_x,
        face_center_y=args.face_center_y,
        face_radius_x=args.face_radius_x,
        face_radius_y=args.face_radius_y,
        use_gates=args.use_gates,
        gate_init=args.gate_init,
        lambda_visual=args.lambda_visual,
        lambda_disp=args.lambda_disp,
        lambda_smooth=args.lambda_smooth,
        lambda_fold=args.lambda_fold,
        lambda_method_l1=args.lambda_method_l1,
        lambda_gate_l1=args.lambda_gate_l1,
        hybrid_vae_weight=args.hybrid_vae_weight,
        objective_scale=args.objective_scale,
        feature_layers=tuple(int(item.strip()) for item in args.feature_layers.split(",") if item.strip()),
        feature_source=args.feature_source,
        objective_timestep=args.objective_timestep,
        objective_timesteps=tuple(
            int(item.strip())
            for item in (args.objective_timesteps.split(",") if args.objective_timesteps is not None else [str(args.objective_timestep)])
            if item.strip()
        ),
        edge_falloff_px=args.edge_falloff_px,
        save_theta=not args.no_save_theta,
    )


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = load_pipe(dtype_from_name(cfg.torch_dtype), device)
    report = run_with_pipe(cfg, pipe, device)
    print(json.dumps({"output_dir": cfg.output_dir, "metrics": report["metrics"], "best_score": report["attack"]["best_score"]}, indent=2))


if __name__ == "__main__":
    main()
