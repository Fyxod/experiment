#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter

REPO = Path("/workspace/geometric-v1")
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


@dataclass
class RunConfig:
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
    epsilon: float = 4.0 / 255.0
    alpha: float = 1.0 / 255.0
    attack_iters: int = 10
    attack: str = "transformer"
    objective_timestep: int = 0
    highpass_sigma: float = 0.0
    random_start: bool = True
    save_delta: bool = False


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def torch_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resize_for_flux(image: Image.Image, max_size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    width, height = image.size
    scale = min(max_size / max(width, height), 1.0)
    width = max(64, int(width * scale) // 16 * 16)
    height = max(64, int(height * scale) // 16 * 16)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def pil_to_01_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor01_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    scores: list[float] = []
    for ch in range(3):
        x = a[:, :, ch].astype(np.float64)
        y = b[:, :, ch].astype(np.float64)
        mux = gaussian_filter(x, sigma=1.5)
        muy = gaussian_filter(y, sigma=1.5)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = gaussian_filter(x * x, sigma=1.5) - mux2
        sigy2 = gaussian_filter(y * y, sigma=1.5) - muy2
        sigxy = gaussian_filter(x * y, sigma=1.5) - muxy
        num = (2.0 * muxy + c1) * (2.0 * sigxy + c2)
        den = (mux2 + muy2 + c1) * (sigx2 + sigy2 + c2)
        scores.append(float(np.mean(num / np.maximum(den, 1e-12))))
    return float(np.mean(scores))


def image_metrics(a: Image.Image, b: Image.Image) -> dict[str, float]:
    aa = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    bb = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    if aa.shape != bb.shape:
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


def make_sheet(out: Path) -> None:
    labels = [
        ("original", out / "original.png"),
        ("perturbed", out / "perturbed.png"),
        ("original diffused", out / "original_diffused.png"),
        ("perturbed diffused", out / "perturbed_diffused.png"),
        ("abs delta x16", out / "delta_x16.png"),
    ]
    thumbs: list[Image.Image] = []
    for label, path in labels:
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((220, 220), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (236, 258), "white")
        canvas.paste(im, ((236 - im.width) // 2, 8))
        ImageDraw.Draw(canvas).text((8, 232), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    if not thumbs:
        return
    sheet = Image.new("RGB", (len(thumbs) * 236, 258), (235, 235, 235))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (idx * 236, 0))
    sheet.save(out / "sheet.jpg", quality=92)


def load_pipe(dtype: torch.dtype, device: torch.device):
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    for module_name in ("vae", "text_encoder", "transformer"):
        module = getattr(pipe, module_name, None)
        if module is not None:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
    return pipe


def retrieve_timesteps(pipe, steps: int, device: torch.device, image_seq_len: int):
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    pipe.scheduler.set_begin_index(0)
    return timesteps


def encode_condition(pipe, image01: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    image = image01 * 2.0 - 1.0
    image = image.to(dtype=pipe.vae.dtype)
    return pipe.prepare_image_latents(
        images=[image],
        batch_size=1,
        generator=generator,
        device=image.device,
        dtype=pipe.vae.dtype,
    )


def transformer_pred(
    pipe,
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


def highpass_delta(delta: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return delta
    arr = delta.detach().float().cpu().numpy()
    low = np.zeros_like(arr)
    for channel in range(arr.shape[1]):
        low[:, channel] = gaussian_filter(arr[:, channel], sigma=(0, sigma, sigma), mode="reflect")
    hp = arr - low
    return torch.from_numpy(hp).to(delta.device, dtype=delta.dtype)


def attack_image(pipe, clean01: torch.Tensor, cfg: RunConfig, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    dtype = torch_dtype(cfg.torch_dtype)
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
        timestep_index = max(0, min(cfg.objective_timestep, len(timesteps) - 1))
        timestep = timesteps[timestep_index].expand(1).to(prompt_embeds.dtype) / 1000
        fixed = {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "latents": latents.detach(),
            "latent_ids": latent_ids,
            "timestep": timestep,
        }
        clean_pred, clean_condition = transformer_pred(pipe, clean01, fixed, generator)
        clean_pred = clean_pred.detach().float()
        clean_condition = clean_condition.detach().float()

    if cfg.random_start:
        delta = torch.empty_like(clean01).uniform_(-cfg.epsilon, cfg.epsilon)
    else:
        delta = torch.zeros_like(clean01)
    delta = highpass_delta(delta, cfg.highpass_sigma)
    delta = delta.clamp(-cfg.epsilon, cfg.epsilon).detach()

    history: list[dict[str, float]] = []
    best_delta = delta.clone()
    best_score = -float("inf")
    for step in range(cfg.attack_iters):
        delta.requires_grad_(True)
        candidate = (clean01 + delta).clamp(0, 1)
        pred, condition = transformer_pred(pipe, candidate, fixed, generator)
        pred = pred.float()
        condition = condition.float()
        pred_score = torch.mean((pred - clean_pred) ** 2)
        latent_score = torch.mean((condition - clean_condition) ** 2)
        if cfg.attack == "vae":
            score = latent_score
        elif cfg.attack == "hybrid":
            score = pred_score + 0.25 * latent_score
        else:
            score = pred_score
        score_value = float(score.detach().cpu())
        with torch.no_grad():
            if score_value > best_score:
                best_score = score_value
                best_delta = delta.detach().clone()
        loss = -score
        loss.backward()
        grad = delta.grad.detach()
        with torch.no_grad():
            delta = delta - cfg.alpha * grad.sign()
            if cfg.highpass_sigma > 0:
                delta = highpass_delta(delta, cfg.highpass_sigma)
            delta = delta.clamp(-cfg.epsilon, cfg.epsilon)
            delta = ((clean01 + delta).clamp(0, 1) - clean01).detach()
            history.append(
                {
                    "iter": float(step + 1),
                    "score": score_value,
                    "pred_mse": float(pred_score.detach().cpu()),
                    "latent_mse": float(latent_score.detach().cpu()),
                    "delta_linf": float(delta.abs().max().detach().cpu()),
                    "delta_mean_abs": float(delta.abs().mean().detach().cpu()),
                }
            )
        del pred, condition, score, loss, grad
        torch.cuda.empty_cache()

    return (clean01 + best_delta).clamp(0, 1).detach(), {"history": history, "best_score": best_score}


@torch.inference_mode()
def diffuse(pipe, image: Image.Image, cfg: RunConfig, device: torch.device) -> Image.Image:
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


def run_with_pipe(cfg: RunConfig, pipe: Any, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", asdict(cfg))

    original = resize_for_flux(Image.open(cfg.input), cfg.max_size)
    original.save(out / "original.png")
    clean01 = pil_to_01_tensor(original, device)

    attack_report: dict[str, Any]
    try:
        perturbed01, attack_report = attack_image(pipe, clean01, cfg, device)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        fallback = RunConfig(**{**asdict(cfg), "attack": "vae"})
        perturbed01, attack_report = attack_image(pipe, clean01, fallback, device)
        attack_report["fallback_from"] = cfg.attack
        attack_report["fallback_error"] = f"{type(exc).__name__}: {exc}"
        cfg.attack = "vae"

    perturbed = tensor01_to_pil(perturbed01)
    perturbed.save(out / "perturbed.png")
    delta = (perturbed01 - clean01).detach().float()
    delta_vis = (delta.abs() / max(cfg.epsilon, 1e-8) * 255.0 * 0.75).clamp(0, 255)
    tensor01_to_pil((delta.abs() * 16.0).clamp(0, 1)).save(out / "delta_x16.png")
    if cfg.save_delta:
        np.save(out / "delta.npy", delta.cpu().numpy())

    original_diffused = diffuse(pipe, original, cfg, device)
    perturbed_diffused = diffuse(pipe, perturbed, cfg, device)
    original_diffused.save(out / "original_diffused.png")
    perturbed_diffused.save(out / "perturbed_diffused.png")

    metrics = {
        "input": image_metrics(original, perturbed),
        "output": image_metrics(original_diffused, perturbed_diffused),
        "delta": {
            "linf": float(delta.abs().max().cpu()),
            "mean_abs": float(delta.abs().mean().cpu()),
            "epsilon": cfg.epsilon,
        },
    }
    report = {
        "config": asdict(cfg),
        "model_id": MODEL_ID,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "attack": attack_report,
        "outputs": {
            "original": str(out / "original.png"),
            "perturbed": str(out / "perturbed.png"),
            "original_diffused": str(out / "original_diffused.png"),
            "perturbed_diffused": str(out / "perturbed_diffused.png"),
            "report": str(out / "report.json"),
            "sheet": str(out / "sheet.jpg"),
        },
    }
    write_json(out / "report.json", report)
    make_sheet(out)
    return report


def run(cfg: RunConfig) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = load_pipe(torch_dtype(cfg.torch_dtype), device)
    report = run_with_pipe(cfg, pipe, device)
    print(json.dumps({"output_dir": cfg.output_dir, "metrics": report["metrics"]}, indent=2))


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=4.0 / 255.0)
    parser.add_argument("--epsilon-255", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--alpha-255", type=float, default=None)
    parser.add_argument("--attack-iters", type=int, default=10)
    parser.add_argument("--attack", choices=["transformer", "vae", "hybrid"], default="transformer")
    parser.add_argument("--objective-timestep", type=int, default=0)
    parser.add_argument("--highpass-sigma", type=float, default=0.0)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--save-delta", action="store_true")
    args = parser.parse_args()
    epsilon = args.epsilon_255 / 255.0 if args.epsilon_255 is not None else args.epsilon
    alpha = args.alpha_255 / 255.0 if args.alpha_255 is not None else args.alpha
    return RunConfig(
        input=args.input,
        prompt=args.prompt,
        output_dir=args.output_dir,
        seed=args.seed,
        max_size=args.max_size,
        diffusion_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
        epsilon=epsilon,
        alpha=alpha,
        attack_iters=args.attack_iters,
        attack=args.attack,
        objective_timestep=args.objective_timestep,
        highpass_sigma=args.highpass_sigma,
        random_start=not args.no_random_start,
        save_delta=args.save_delta,
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(parse_args())
