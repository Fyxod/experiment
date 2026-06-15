#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

ROOT = Path("/workspace/output/geo_whitebox_results_20260614_172707")
ATTACK_SCRIPT = ROOT / "scripts" / "geo_whitebox_attack.py"


def load_attack_module():
    spec = importlib.util.spec_from_file_location("geo_whitebox_attack", ATTACK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {ATTACK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOURCES = {
    "men57": "/workspace/gender_dataset/men/57.jpg",
    "men58": "/workspace/gender_dataset/men/58.jpg",
    "men51": "/workspace/gender_dataset/men/51.jpg",
    "men66": "/workspace/gender_dataset/men/66.jpg",
    "men74": "/workspace/gender_dataset/men/74.jpg",
    "men11": "/workspace/gender_dataset/men/11.jpg",
    "sample": "/workspace/geometric-v1/samples/image.png",
}

PROMPTS = {
    "older": "make the person older",
    "headshot": "turn the image into a professional headshot",
    "sunglasses": "add sunglasses",
    "smile": "make the person smile",
    "studio": "make the person look like a studio portrait",
    "hair": "change the hairstyle",
}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def flatten_params(model: torch.nn.Module) -> tuple[torch.Tensor, list[tuple[str, torch.Size, int]]]:
    pieces = []
    spec = []
    for name, param in model.named_parameters():
        flat = param.detach().flatten()
        pieces.append(flat)
        spec.append((name, param.shape, flat.numel()))
    if not pieces:
        raise RuntimeError("Model has no optimizable parameters")
    return torch.cat(pieces), spec


def assign_params(model: torch.nn.Module, vector: torch.Tensor, spec: list[tuple[str, torch.Size, int]]) -> None:
    params = dict(model.named_parameters())
    offset = 0
    with torch.no_grad():
        for name, shape, count in spec:
            params[name].copy_(vector[offset : offset + count].view(shape))
            offset += count


def make_combined_sheet(rows: list[dict[str, Any]], out_path: Path, limit: int = 16) -> None:
    selected = rows[:limit]
    cell_w, cell_h = 790, 224
    sheet = Image.new("RGB", (cell_w, max(1, len(selected)) * cell_h), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(selected):
        run_dir = Path(row["output_dir"])
        x = 8
        for filename in ["original.png", "perturbed.png", "original_diffused.png", "perturbed_diffused.png", "flow.png"]:
            path = run_dir / filename
            if not path.exists():
                continue
            im = Image.open(path).convert("RGB")
            im.thumbnail((132, 132), Image.Resampling.LANCZOS)
            sheet.paste(im, (x + (144 - im.width) // 2, row_idx * cell_h + 8))
            x += 150
        text = (
            f"{row.get('rank', 0):02d} {row['name']} score={row['score']:.3f} valid={row['valid']}\n"
            f"in psnr={row.get('input_psnr', 0):.1f} ssim={row.get('input_ssim', 0):.3f}; "
            f"disp max={row.get('max_disp', 0):.2f} mean={row.get('mean_disp', 0):.2f} p95={row.get('p95_disp', 0):.2f}\n"
            f"out ssim={row.get('output_ssim', 0):.3f} l2={row.get('output_l2', 0):.3f}; "
            f"clean edit l2={row.get('clean_edit_l2', 0):.3f} pert edit l2={row.get('pert_edit_l2', 0):.3f}"
        )
        draw.text((8, row_idx * cell_h + 148), text, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def score_candidate(
    module: Any,
    original: Image.Image,
    perturbed: Image.Image,
    original_diffused: Image.Image,
    perturbed_diffused: Image.Image,
    disp: torch.Tensor,
    min_psnr: float,
    min_ssim: float,
    max_disp: float,
    max_mean_disp: float,
) -> tuple[float, dict[str, Any], bool]:
    input_metrics = module.image_metrics(original, perturbed)
    output_metrics = module.image_metrics(original_diffused, perturbed_diffused)
    clean_edit = module.image_metrics(original, original_diffused)
    pert_edit = module.image_metrics(perturbed, perturbed_diffused)
    disp_stats = module.displacement_stats(disp)
    valid = (
        input_metrics["psnr"] >= min_psnr
        and input_metrics["ssim"] >= min_ssim
        and disp_stats["max_magnitude_px"] <= max_disp + 1e-5
        and disp_stats["mean_magnitude_px"] <= max_mean_disp
    )
    prompt_failure = max(0.0, clean_edit["l2"] - pert_edit["l2"])
    output_diverge = (1.0 - output_metrics["ssim"]) * 4.0 + output_metrics["l2"] * 5.0
    stealth_penalty = max(0.0, 38.0 - input_metrics["psnr"]) * 0.02 + max(0.0, 0.94 - input_metrics["ssim"]) * 0.4
    score = output_diverge + prompt_failure * 4.0 - stealth_penalty
    if not valid:
        score -= 10.0
    metrics = {
        "input": input_metrics,
        "output": output_metrics,
        "clean_edit": clean_edit,
        "perturbed_edit": pert_edit,
        "displacement": disp_stats,
        "score_terms": {
            "prompt_failure": prompt_failure,
            "output_diverge": output_diverge,
            "stealth_penalty": stealth_penalty,
        },
    }
    return float(score), metrics, valid


def evaluate_vector(
    module: Any,
    model: torch.nn.Module,
    vector: torch.Tensor,
    spec: list[tuple[str, torch.Size, int]],
    clean01: torch.Tensor,
    original: Image.Image,
    original_diffused: Image.Image,
    pipe: Any,
    cfg: Any,
    device: torch.device,
    out_dir: Path,
    min_psnr: float,
    min_ssim: float,
    max_mean_disp: float,
    save_invalid: bool,
) -> dict[str, Any]:
    assign_params(model, vector, spec)
    projected_ok = model.project_parameters()
    with torch.no_grad():
        perturbed01, disp, fields = model.warp(clean01)
    perturbed = module.tensor01_to_pil(perturbed01)
    input_metrics = module.image_metrics(original, perturbed)
    disp_stats = module.displacement_stats(disp)
    prelim_valid = (
        input_metrics["psnr"] >= min_psnr
        and input_metrics["ssim"] >= min_ssim
        and disp_stats["max_magnitude_px"] <= cfg.max_disp_px + 1e-5
        and disp_stats["mean_magnitude_px"] <= max_mean_disp
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    original.save(out_dir / "original.png")
    perturbed.save(out_dir / "perturbed.png")
    original_diffused.save(out_dir / "original_diffused.png")
    module.displacement_visuals(out_dir, disp)
    if prelim_valid or save_invalid:
        perturbed_diffused = module.diffuse(pipe, perturbed, cfg, device)
        perturbed_diffused.save(out_dir / "perturbed_diffused.png")
        score, metrics, valid = score_candidate(
            module,
            original,
            perturbed,
            original_diffused,
            perturbed_diffused,
            disp,
            min_psnr,
            min_ssim,
            cfg.max_disp_px,
            max_mean_disp,
        )
    else:
        score = -10.0
        valid = False
        metrics = {
            "input": input_metrics,
            "displacement": disp_stats,
            "skip_reason": "failed_pre_diffusion_visual_gate",
        }
    report = {
        "config": module.asdict(cfg),
        "projected_parameters_finite": projected_ok,
        "metrics": metrics,
        "score": score,
        "valid": valid,
        "method_displacement": module.method_stats(fields),
        "gate_values": model.gate_values(),
        "outputs": {
            "original": str(out_dir / "original.png"),
            "perturbed": str(out_dir / "perturbed.png"),
            "original_diffused": str(out_dir / "original_diffused.png"),
            "perturbed_diffused": str(out_dir / "perturbed_diffused.png"),
            "flow": str(out_dir / "flow.png"),
            "report": str(out_dir / "report.json"),
        },
    }
    write_json(out_dir / "effective_config.json", module.asdict(cfg))
    write_json(out_dir / "report.json", report)
    torch.save({"state_dict": model.state_dict(), "disp": disp.detach().cpu()}, out_dir / "theta.pt")
    module.make_sheet(out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--prompt-key", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--methods", default="tps,piecewise,dct,rolling")
    parser.add_argument("--face-mask", default="face")
    parser.add_argument("--use-gates", action="store_true")
    parser.add_argument("--max-disp-px", type=float, default=2.0)
    parser.add_argument("--max-mean-disp", type=float, default=0.55)
    parser.add_argument("--min-psnr", type=float, default=36.0)
    parser.add_argument("--min-ssim", type=float, default=0.90)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--sample-std", type=float, default=0.85)
    parser.add_argument("--init-theta", default=None)
    parser.add_argument("--eval-initial", action="store_true")
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--save-invalid", action="store_true")
    args = parser.parse_args()

    module = load_attack_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = module.load_pipe(module.dtype_from_name("bfloat16"), device)
    run_root = ROOT / "blackbox" / args.job_name
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = module.AttackConfig(
        input=SOURCES[args.source],
        prompt=PROMPTS[args.prompt_key],
        output_dir=str(run_root),
        seed=args.seed,
        methods=tuple(item.strip() for item in args.methods.split(",") if item.strip()),
        objective="final_edit_blackbox",
        max_disp_px=args.max_disp_px,
        attack_iters=0,
        init_scale_px=0.0,
        diffusion_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
        max_size=512,
        face_mask=args.face_mask,
        use_gates=args.use_gates,
        edge_falloff_px=16.0,
    )
    original = module.resize_for_flux(Image.open(cfg.input), cfg.max_size)
    detected = module.detect_face_fraction(original) if cfg.face_mask not in {"none", "global"} else None
    if detected is not None:
        cfg.face_center_x, cfg.face_center_y, cfg.face_radius_x, cfg.face_radius_y = detected
    clean01 = module.pil_to_tensor01(original, device)
    original_diffused = module.diffuse(pipe, original, cfg, device)
    original.save(run_root / "original.png")
    original_diffused.save(run_root / "original_diffused.png")

    model = module.GeometryAttack(cfg, clean01.shape[-2], clean01.shape[-1], device)
    if args.init_theta:
        state = torch.load(args.init_theta, map_location=device)
        state_dict = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        manifest_state = {
            "init_theta": args.init_theta,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
    else:
        manifest_state = {"init_theta": None}
    mean, spec = flatten_params(model)
    std = torch.full_like(mean, args.sample_std)
    for idx, (name, _shape, count) in enumerate(spec):
        if "gate" in name:
            start = sum(item[2] for item in spec[:idx])
            std[start : start + count] = 0.45

    gen = torch.Generator(device=device).manual_seed(args.seed + 919)
    started = time.perf_counter()
    manifest = {
        "args": vars(args),
        "config": module.asdict(cfg),
        "detected_face_fraction": detected,
        "parameter_count": int(mean.numel()),
        "parameter_spec": [{"name": name, "shape": list(shape), "count": count} for name, shape, count in spec],
        "state_init": manifest_state,
    }
    write_json(run_root / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    if args.eval_initial:
        out_dir = run_root / "cand_000_initial"
        report = evaluate_vector(
            module,
            model,
            mean.detach().clone(),
            spec,
            clean01,
            original,
            original_diffused,
            pipe,
            cfg,
            device,
            out_dir,
            args.min_psnr,
            args.min_ssim,
            args.max_mean_disp,
            True,
        )
        metrics = report["metrics"]
        rows.append(
            {
                "name": out_dir.name,
                "output_dir": str(out_dir),
                "generation": 0,
                "score": report["score"],
                "valid": report["valid"],
                "input_psnr": metrics.get("input", {}).get("psnr", 0.0),
                "input_ssim": metrics.get("input", {}).get("ssim", 0.0),
                "output_ssim": metrics.get("output", {}).get("ssim", 1.0),
                "output_l2": metrics.get("output", {}).get("l2", 0.0),
                "clean_edit_l2": metrics.get("clean_edit", {}).get("l2", 0.0),
                "pert_edit_l2": metrics.get("perturbed_edit", {}).get("l2", 0.0),
                "max_disp": metrics.get("displacement", {}).get("max_magnitude_px", 0.0),
                "mean_disp": metrics.get("displacement", {}).get("mean_magnitude_px", 0.0),
                "p95_disp": metrics.get("displacement", {}).get("p95_magnitude_px", 0.0),
            }
        )

    for generation in range(1, args.generations + 1):
        candidates: list[tuple[float, torch.Tensor, dict[str, Any]]] = []
        for member in range(1, args.population + 1):
            sample_index = (generation - 1) * args.population + member
            vector = mean + std * torch.randn(mean.shape, generator=gen, device=device)
            out_dir = run_root / f"cand_{sample_index:03d}_g{generation:02d}_m{member:02d}"
            report = evaluate_vector(
                module,
                model,
                vector,
                spec,
                clean01,
                original,
                original_diffused,
                pipe,
                cfg,
                device,
                out_dir,
                args.min_psnr,
                args.min_ssim,
                args.max_mean_disp,
                args.save_invalid,
            )
            metrics = report["metrics"]
            row = {
                "name": out_dir.name,
                "output_dir": str(out_dir),
                "generation": generation,
                "score": report["score"],
                "valid": report["valid"],
                "input_psnr": metrics.get("input", {}).get("psnr", 0.0),
                "input_ssim": metrics.get("input", {}).get("ssim", 0.0),
                "output_ssim": metrics.get("output", {}).get("ssim", 1.0),
                "output_l2": metrics.get("output", {}).get("l2", 0.0),
                "clean_edit_l2": metrics.get("clean_edit", {}).get("l2", 0.0),
                "pert_edit_l2": metrics.get("perturbed_edit", {}).get("l2", 0.0),
                "max_disp": metrics.get("displacement", {}).get("max_magnitude_px", 0.0),
                "mean_disp": metrics.get("displacement", {}).get("mean_magnitude_px", 0.0),
                "p95_disp": metrics.get("displacement", {}).get("p95_magnitude_px", 0.0),
            }
            rows.append(row)
            candidates.append((report["score"], vector.detach().clone(), row))
            ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
            for rank, ranked_row in enumerate(ranked, start=1):
                ranked_row["rank"] = rank
            write_json(run_root / "progress.json", {"elapsed_seconds": time.perf_counter() - started, "rows": ranked})
            torch.cuda.empty_cache()

        elites = sorted(candidates, key=lambda item: item[0], reverse=True)[: max(1, args.elite)]
        elite_vectors = torch.stack([item[1] for item in elites], dim=0)
        mean = elite_vectors.mean(dim=0)
        if elite_vectors.shape[0] > 1:
            std = (0.35 * std + 0.65 * elite_vectors.std(dim=0, unbiased=False).clamp_min(0.08)).clamp(0.04, args.sample_std)
        else:
            std = (std * 0.7).clamp_min(0.06)
        write_json(
            run_root / f"generation_{generation:02d}.json",
            {
                "generation": generation,
                "elite_rows": [item[2] for item in elites],
                "mean_abs": float(mean.abs().mean().detach().cpu()),
                "std_mean": float(std.mean().detach().cpu()),
            },
        )

    ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    write_json(run_root / "ranked.json", {"elapsed_seconds": time.perf_counter() - started, "rows": ranked})
    make_combined_sheet(ranked, run_root / "sheet.jpg")
    print(json.dumps({"run_root": str(run_root), "top": ranked[:8]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
