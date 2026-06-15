#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
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
}

PROMPTS = {
    "older": "make the person older",
    "headshot": "turn the image into a professional headshot",
    "sunglasses": "add sunglasses",
    "smile": "make the person smile",
    "studio": "make the person look like a studio portrait",
    "hair": "change the hairstyle",
}


def jobs_for_batch(batch: str) -> list[dict[str, Any]]:
    if batch == "geo_pass5_scaled":
        base = [
            {
                "source": "men51",
                "prompt": "headshot",
                "methods": "dct,grid,tps,piecewise,rolling,affine,radial",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.0,
                "iters": 90,
                "lr": 0.008,
                "seed": 6201,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
                "dct_size": 10,
                "grid_size": 8,
                "tps_size": 6,
                "piecewise_size": 6,
                "objective_scale": 750.0,
                "lambda_visual": 4.0,
                "lambda_disp": 0.02,
                "lambda_smooth": 0.04,
                "lambda_method_l1": 0.002,
                "lambda_gate_l1": 0.001,
            },
            {
                "source": "men51",
                "prompt": "studio",
                "methods": "dct,grid,tps,piecewise,rolling,affine,radial",
                "objective": "hybrid",
                "max_disp_px": 2.0,
                "iters": 90,
                "lr": 0.008,
                "seed": 6202,
                "face_mask": "face",
                "use_gates": True,
                "dct_size": 10,
                "grid_size": 8,
                "tps_size": 6,
                "piecewise_size": 6,
                "objective_timesteps": (0, 1),
                "objective_scale": 1500.0,
                "lambda_visual": 4.0,
                "lambda_disp": 0.02,
                "lambda_smooth": 0.04,
                "lambda_method_l1": 0.002,
                "lambda_gate_l1": 0.001,
            },
            {
                "source": "men66",
                "prompt": "sunglasses",
                "methods": "dct,grid,tps,piecewise,rolling,affine,radial",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.0,
                "iters": 90,
                "lr": 0.008,
                "seed": 6203,
                "face_mask": "upper_face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
                "dct_size": 10,
                "grid_size": 8,
                "tps_size": 6,
                "piecewise_size": 6,
                "objective_scale": 1000.0,
                "lambda_visual": 4.0,
                "lambda_disp": 0.02,
                "lambda_smooth": 0.04,
                "lambda_method_l1": 0.002,
                "lambda_gate_l1": 0.001,
            },
            {
                "source": "men74",
                "prompt": "smile",
                "methods": "dct,grid,tps,piecewise,rolling,affine,radial",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.0,
                "iters": 90,
                "lr": 0.008,
                "seed": 6204,
                "face_mask": "mouth",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
                "dct_size": 10,
                "grid_size": 8,
                "tps_size": 6,
                "piecewise_size": 6,
                "objective_scale": 1000.0,
                "lambda_visual": 4.0,
                "lambda_disp": 0.02,
                "lambda_smooth": 0.04,
                "lambda_method_l1": 0.002,
                "lambda_gate_l1": 0.001,
            },
            {
                "source": "men57",
                "prompt": "older",
                "methods": "dct,grid,tps,piecewise,rolling,affine,radial",
                "objective": "denoise_latent",
                "max_disp_px": 2.2,
                "iters": 24,
                "lr": 0.005,
                "seed": 6205,
                "face_mask": "face",
                "use_gates": True,
                "dct_size": 10,
                "grid_size": 8,
                "tps_size": 6,
                "piecewise_size": 6,
                "objective_scale": 200.0,
                "lambda_visual": 4.0,
                "lambda_disp": 0.02,
                "lambda_smooth": 0.04,
                "lambda_method_l1": 0.002,
                "lambda_gate_l1": 0.001,
            },
            {
                "source": "men11",
                "prompt": "headshot",
                "methods": "dct,grid,tps,piecewise,rolling,affine,radial",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.2,
                "iters": 90,
                "lr": 0.008,
                "seed": 6206,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
                "dct_size": 10,
                "grid_size": 8,
                "tps_size": 6,
                "piecewise_size": 6,
                "objective_scale": 1000.0,
                "lambda_visual": 4.0,
                "lambda_disp": 0.02,
                "lambda_smooth": 0.04,
                "lambda_method_l1": 0.002,
                "lambda_gate_l1": 0.001,
            },
        ]
    elif batch in {"geo_pass4", "geo_pass4_clean"}:
        base = [
            {
                "source": "men51",
                "prompt": "headshot",
                "methods": "dct,grid,tps,piecewise",
                "objective": "hybrid_hidden",
                "max_disp_px": 1.5,
                "iters": 80,
                "lr": 0.006,
                "seed": 6101,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
                "dct_size": 12,
                "grid_size": 10,
                "tps_size": 6,
                "piecewise_size": 6,
                "lambda_gate_l1": 0.004,
            },
            {
                "source": "men51",
                "prompt": "headshot",
                "methods": "dct,grid,tps,piecewise,rolling",
                "objective": "transformer_pred",
                "max_disp_px": 1.5,
                "iters": 80,
                "lr": 0.006,
                "seed": 6102,
                "face_mask": "face",
                "use_gates": True,
                "dct_size": 12,
                "grid_size": 10,
                "tps_size": 6,
                "piecewise_size": 6,
                "objective_timesteps": (0, 1),
            },
            {
                "source": "men66",
                "prompt": "sunglasses",
                "methods": "dct,grid,tps,piecewise",
                "objective": "hybrid_hidden",
                "max_disp_px": 1.5,
                "iters": 80,
                "lr": 0.006,
                "seed": 6103,
                "face_mask": "upper_face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
                "dct_size": 12,
                "grid_size": 10,
                "tps_size": 6,
                "piecewise_size": 6,
                "lambda_gate_l1": 0.004,
            },
            {
                "source": "men57",
                "prompt": "older",
                "methods": "dct,grid,tps,piecewise",
                "objective": "denoise_latent",
                "max_disp_px": 1.8,
                "iters": 20,
                "lr": 0.004,
                "seed": 6104,
                "face_mask": "face",
                "use_gates": True,
                "dct_size": 12,
                "grid_size": 10,
                "tps_size": 6,
                "piecewise_size": 6,
            },
            {
                "source": "men74",
                "prompt": "smile",
                "methods": "dct,grid,tps,piecewise",
                "objective": "hybrid_hidden",
                "max_disp_px": 1.5,
                "iters": 80,
                "lr": 0.006,
                "seed": 6105,
                "face_mask": "mouth",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
                "dct_size": 12,
                "grid_size": 10,
                "tps_size": 6,
                "piecewise_size": 6,
                "lambda_gate_l1": 0.004,
            },
            {
                "source": "men11",
                "prompt": "headshot",
                "methods": "dct,grid,tps,piecewise",
                "objective": "hybrid_hidden",
                "max_disp_px": 1.8,
                "iters": 80,
                "lr": 0.006,
                "seed": 6106,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
                "dct_size": 12,
                "grid_size": 10,
                "tps_size": 6,
                "piecewise_size": 6,
                "lambda_gate_l1": 0.004,
            },
        ]
    elif batch == "geo_pass3":
        base = [
            {
                "source": "men66",
                "prompt": "sunglasses",
                "methods": "tps,piecewise,dct,rolling",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.0,
                "iters": 32,
                "lr": 0.010,
                "seed": 5201,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
            },
            {
                "source": "men66",
                "prompt": "sunglasses",
                "methods": "tps,piecewise,grid,rolling",
                "objective": "attention_proxy",
                "max_disp_px": 2.0,
                "iters": 26,
                "lr": 0.010,
                "seed": 5202,
                "face_mask": "upper_face",
                "use_gates": True,
                "feature_layers": (0, 4),
            },
            {
                "source": "men51",
                "prompt": "headshot",
                "methods": "tps,piecewise,dct,rolling",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.0,
                "iters": 32,
                "lr": 0.010,
                "seed": 5203,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
            },
            {
                "source": "men51",
                "prompt": "hair",
                "methods": "tps,piecewise,dct,grid",
                "objective": "hidden_feature",
                "max_disp_px": 2.3,
                "iters": 34,
                "lr": 0.009,
                "seed": 5204,
                "face_mask": "hairline",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
            },
            {
                "source": "men57",
                "prompt": "older",
                "methods": "tps,piecewise,dct,rolling",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.3,
                "iters": 34,
                "lr": 0.009,
                "seed": 5205,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
            },
            {
                "source": "men57",
                "prompt": "older",
                "methods": "tps,piecewise,grid,dct",
                "objective": "denoise_latent",
                "max_disp_px": 2.0,
                "iters": 10,
                "lr": 0.006,
                "seed": 5206,
                "face_mask": "face",
                "use_gates": True,
            },
            {
                "source": "men74",
                "prompt": "smile",
                "methods": "tps,piecewise,rolling,dct",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.0,
                "iters": 32,
                "lr": 0.010,
                "seed": 5207,
                "face_mask": "mouth",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
            },
            {
                "source": "men74",
                "prompt": "smile",
                "methods": "tps,piecewise,grid",
                "objective": "attention_proxy",
                "max_disp_px": 2.0,
                "iters": 26,
                "lr": 0.010,
                "seed": 5208,
                "face_mask": "lower_face",
                "use_gates": True,
                "feature_layers": (0, 4),
            },
            {
                "source": "men58",
                "prompt": "headshot",
                "methods": "tps,piecewise,dct,rolling",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.0,
                "iters": 32,
                "lr": 0.010,
                "seed": 5209,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
            },
            {
                "source": "men11",
                "prompt": "headshot",
                "methods": "tps,piecewise,dct,rolling",
                "objective": "hidden_feature",
                "max_disp_px": 2.3,
                "iters": 34,
                "lr": 0.009,
                "seed": 5210,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
            },
            {
                "source": "men66",
                "prompt": "studio",
                "methods": "tps,piecewise,dct,grid,rolling",
                "objective": "hybrid_hidden",
                "max_disp_px": 2.5,
                "iters": 36,
                "lr": 0.008,
                "seed": 5211,
                "face_mask": "face",
                "use_gates": True,
                "feature_layers": (0, 4, 8),
            },
            {
                "source": "men51",
                "prompt": "studio",
                "methods": "tps,piecewise,dct,grid,rolling",
                "objective": "hybrid",
                "max_disp_px": 2.5,
                "iters": 40,
                "lr": 0.008,
                "seed": 5212,
                "face_mask": "face",
                "use_gates": True,
                "objective_timesteps": (0, 1),
            },
        ]
    elif batch == "geo_pass2":
        base = [
            ("men57", "older", "rolling,dct", "hybrid", 2.5, 50, 0.006, 4101),
            ("men57", "older", "grid,dct", "transformer_pred", 2.5, 50, 0.006, 4102),
            ("men58", "headshot", "rolling,dct", "hybrid", 2.5, 50, 0.006, 4103),
            ("men58", "headshot", "grid,dct", "transformer_pred", 2.5, 50, 0.006, 4104),
            ("men51", "headshot", "rolling,dct", "hybrid", 2.5, 50, 0.006, 4105),
            ("men51", "headshot", "grid,dct", "transformer_pred", 2.5, 50, 0.006, 4106),
            ("men66", "sunglasses", "rolling,dct", "hybrid", 2.5, 50, 0.006, 4107),
            ("men66", "sunglasses", "grid,dct", "transformer_pred", 2.5, 50, 0.006, 4108),
        ]
    else:
        base = [
            ("men57", "older", "dct", "hybrid", 1.5, 35, 0.012, 4001),
            ("men57", "older", "rolling,dct", "hybrid", 2.0, 35, 0.010, 4002),
            ("men57", "older", "grid,dct", "vae", 2.0, 35, 0.010, 4003),
            ("men58", "headshot", "dct", "hybrid", 1.5, 35, 0.012, 4004),
            ("men58", "headshot", "rolling,dct", "hybrid", 2.0, 35, 0.010, 4005),
            ("men58", "headshot", "grid,dct", "vae", 2.0, 35, 0.010, 4006),
            ("men51", "headshot", "dct", "hybrid", 1.5, 35, 0.012, 4007),
            ("men51", "headshot", "rolling,dct", "hybrid", 2.0, 35, 0.010, 4008),
            ("men66", "sunglasses", "dct", "hybrid", 1.5, 35, 0.012, 4009),
            ("men66", "sunglasses", "rolling,dct", "hybrid", 2.0, 35, 0.010, 4010),
            ("men74", "smile", "dct", "hybrid", 1.5, 35, 0.012, 4011),
            ("men11", "headshot", "rolling,dct", "hybrid", 2.0, 35, 0.010, 4012),
            ("men57", "older", "rbf,rolling", "hybrid", 2.0, 35, 0.010, 4013),
            ("men58", "headshot", "rbf,rolling", "hybrid", 2.0, 35, 0.010, 4014),
        ]
    jobs = []
    for idx, item in enumerate(base, start=1):
        if isinstance(item, dict):
            job = dict(item)
            max_disp = job["max_disp_px"]
            source = job["source"]
            prompt = job["prompt"]
            methods = job["methods"]
            objective = job["objective"]
        else:
            source, prompt, methods, objective, max_disp, iters, lr, seed = item
            job = {
                "source": source,
                "prompt": prompt,
                "methods": methods,
                "objective": objective,
                "max_disp_px": max_disp,
                "iters": iters,
                "lr": lr,
                "seed": seed,
            }
        methods_name = methods.replace(",", "_")
        job["name"] = f"{idx:02d}_{source}_{prompt}_{methods_name}_{objective}_{max_disp:g}px"
        jobs.append(job)
    return jobs


def score_report(report: dict[str, Any]) -> float:
    metrics = report["metrics"]
    inp = metrics["input"]
    disp = metrics["displacement"]
    out = metrics["output"]
    if inp["psnr"] < 36.0 or inp["ssim"] < 0.90:
        return -999.0
    if disp["max_magnitude_px"] > 3.1 or disp["mean_magnitude_px"] > 0.9:
        return -999.0
    return (1.0 - out["ssim"]) * 4.0 + out["l2"] * 5.0


def make_sheet(rows: list[dict[str, Any]], out_path: Path, limit: int = 18) -> None:
    selected = rows[:limit]
    cell_w, cell_h = 720, 220
    sheet = Image.new("RGB", (cell_w, max(1, len(selected)) * cell_h), (238, 238, 238))
    for row_idx, row in enumerate(selected):
        run_dir = Path(row["output_dir"])
        x = 8
        for filename in ["original.png", "perturbed.png", "original_diffused.png", "perturbed_diffused.png", "flow.png"]:
            path = run_dir / filename
            if not path.exists():
                continue
            im = Image.open(path).convert("RGB")
            im.thumbnail((126, 126), Image.Resampling.LANCZOS)
            sheet.paste(im, (x + (136 - im.width) // 2, row_idx * cell_h + 8))
            x += 140
        text = (
            f"{row['rank']:02d} {row['name']} score={row['score']:.3f}\n"
            f"in psnr={row.get('input_psnr', 0):.1f} ssim={row.get('input_ssim', 0):.3f}; "
            f"disp max={row.get('max_disp', 0):.2f} mean={row.get('mean_disp', 0):.2f}\n"
            f"out ssim={row.get('output_ssim', 0):.3f} l2={row.get('output_l2', 0):.3f}"
        )
        ImageDraw.Draw(sheet).text((8, row_idx * cell_h + 144), text, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="geo_pass1")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    module = load_attack_module()
    jobs = jobs_for_batch(args.batch)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    run_root = ROOT / "search" / args.batch
    (ROOT / "configs" / args.batch).mkdir(parents=True, exist_ok=True)
    (ROOT / "configs" / args.batch / "manifest.json").write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = module.load_pipe(module.dtype_from_name("bfloat16"), device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        run_dir = run_root / job["name"]
        cfg = module.AttackConfig(
            input=SOURCES[job["source"]],
            prompt=PROMPTS[job["prompt"]],
            output_dir=str(run_dir),
            seed=job["seed"],
            methods=tuple(job["methods"].split(",")),
            objective=job["objective"],
            max_disp_px=job["max_disp_px"],
            attack_iters=job["iters"],
            lr=job["lr"],
            lambda_visual=10.0,
            lambda_disp=0.04,
            lambda_smooth=0.08,
            lambda_fold=5.0,
            lambda_method_l1=0.01,
            lambda_gate_l1=job.get("lambda_gate_l1", 0.01),
            max_size=512,
            diffusion_steps=4,
            edge_falloff_px=16.0,
            face_mask=job.get("face_mask", "none"),
            use_gates=job.get("use_gates", False),
            feature_layers=tuple(job.get("feature_layers", (0, 4, 8))),
            feature_source=job.get("feature_source", "blocks"),
            objective_timesteps=tuple(job.get("objective_timesteps", (0,))),
            tps_size=job.get("tps_size", 5),
            piecewise_size=job.get("piecewise_size", 5),
            grid_size=job.get("grid_size", 6),
        )
        cfg.lambda_visual = job.get("lambda_visual", cfg.lambda_visual)
        cfg.lambda_disp = job.get("lambda_disp", cfg.lambda_disp)
        cfg.lambda_smooth = job.get("lambda_smooth", cfg.lambda_smooth)
        cfg.lambda_method_l1 = job.get("lambda_method_l1", cfg.lambda_method_l1)
        cfg.objective_scale = job.get("objective_scale", cfg.objective_scale)
        print(f"[{index}/{len(jobs)}] {job['name']}", flush=True)
        try:
            report = module.run_with_pipe(cfg, pipe, device)
            score = score_report(report)
            row = {
                "name": job["name"],
                "output_dir": str(run_dir),
                "source": cfg.input,
                "prompt": cfg.prompt,
                "methods": job["methods"],
                "objective": job["objective"],
                "score": score,
                "input_psnr": report["metrics"]["input"]["psnr"],
                "input_ssim": report["metrics"]["input"]["ssim"],
                "output_ssim": report["metrics"]["output"]["ssim"],
                "output_l2": report["metrics"]["output"]["l2"],
                "max_disp": report["metrics"]["displacement"]["max_magnitude_px"],
                "mean_disp": report["metrics"]["displacement"]["mean_magnitude_px"],
                "ok": True,
            }
        except Exception as exc:
            row = {
                "name": job["name"],
                "output_dir": str(run_dir),
                "source": SOURCES[job["source"]],
                "prompt": PROMPTS[job["prompt"]],
                "methods": job["methods"],
                "objective": job["objective"],
                "score": -999.0,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        (run_root / "progress.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()

    ranked = sorted(rows, key=lambda row: row.get("score", -999.0), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    (run_root / "ranked.json").write_text(json.dumps({"elapsed_seconds": time.perf_counter() - started, "rows": ranked}, indent=2), encoding="utf-8")
    make_sheet(ranked, ROOT / f"{args.batch}_sheet.jpg")
    print(json.dumps({"sheet": str(ROOT / f"{args.batch}_sheet.jpg"), "top": ranked[:8]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
