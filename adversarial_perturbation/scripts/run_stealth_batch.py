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

ROOT = Path("/workspace/output/mentor_results_20260614_123821")
ATTACK_SCRIPT = ROOT / "scripts" / "stealth_flux_attack.py"


def load_attack_module():
    spec = importlib.util.spec_from_file_location("stealth_flux_attack", ATTACK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {ATTACK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOURCES = {
    "sample": "/workspace/geometric-v1/samples/image.png",
    "men11": "/workspace/gender_dataset/men/11.jpg",
    "men20": "/workspace/gender_dataset/men/20.jpg",
    "men27": "/workspace/gender_dataset/men/27.jpg",
    "men51": "/workspace/gender_dataset/men/51.jpg",
    "men57": "/workspace/gender_dataset/men/57.jpg",
    "men58": "/workspace/gender_dataset/men/58.jpg",
    "men66": "/workspace/gender_dataset/men/66.jpg",
    "men74": "/workspace/gender_dataset/men/74.jpg",
    "men79": "/workspace/gender_dataset/men/79.jpg",
    "men80": "/workspace/gender_dataset/men/80.jpg",
}

PROMPTS = {
    "sunglasses": "add sunglasses",
    "studio": "make the person look like a studio portrait",
    "hair": "change the hairstyle",
    "smile": "make the person smile",
    "older": "make the person older",
    "headshot": "turn the image into a professional headshot",
}


def default_jobs(batch: str) -> list[dict[str, Any]]:
    if batch == "stealth_pass2":
        base = [
            ("men66", "sunglasses", 2, 12, 2234, 0.0),
            ("men66", "sunglasses", 3, 12, 2235, 0.0),
            ("men66", "sunglasses", 3, 16, 2236, 0.8),
            ("men57", "older", 2, 12, 2240, 0.0),
            ("men57", "older", 3, 12, 2241, 0.0),
            ("men57", "older", 4, 12, 2242, 0.0),
            ("men74", "smile", 2, 12, 2250, 0.0),
            ("men74", "smile", 3, 12, 2251, 0.0),
            ("men74", "smile", 4, 12, 2252, 0.0),
            ("men11", "headshot", 3, 12, 2260, 0.0),
            ("men11", "headshot", 4, 12, 2261, 0.0),
            ("men58", "headshot", 3, 12, 2270, 0.0),
            ("men58", "headshot", 4, 12, 2271, 0.0),
            ("men51", "headshot", 3, 12, 2280, 0.0),
            ("men20", "sunglasses", 3, 12, 2290, 0.0),
            ("men80", "older", 3, 12, 2300, 0.0),
        ]
        jobs: list[dict[str, Any]] = []
        for idx, (source, prompt, eps, iters, seed, highpass) in enumerate(base, start=1):
            jobs.append(
                {
                    "name": f"{idx:02d}_{source}_{prompt}_eps{eps}_i{iters}_hp{str(highpass).replace('.', 'p')}",
                    "source": source,
                    "prompt": prompt,
                    "epsilon_255": eps,
                    "alpha_255": 1,
                    "iters": iters,
                    "seed": seed,
                    "attack": "transformer",
                    "max_size": 512,
                    "diffusion_steps": 4,
                    "highpass_sigma": highpass,
                    "batch": batch,
                }
            )
        return jobs

    base = [
        ("men66", "sunglasses", 3, 8, 1234),
        ("men66", "sunglasses", 4, 8, 1235),
        ("men11", "sunglasses", 4, 8, 1240),
        ("men20", "sunglasses", 4, 8, 1241),
        ("men27", "smile", 4, 8, 1242),
        ("men51", "smile", 4, 8, 1243),
        ("men57", "older", 4, 8, 1244),
        ("men58", "headshot", 4, 8, 1245),
        ("men74", "smile", 4, 8, 1246),
        ("men79", "sunglasses", 4, 8, 1247),
        ("men80", "older", 4, 8, 1248),
        ("sample", "hair", 4, 8, 1249),
        ("men66", "sunglasses", 5, 10, 1250),
        ("men11", "headshot", 5, 10, 1251),
        ("men51", "headshot", 5, 10, 1252),
        ("men79", "older", 5, 10, 1253),
    ]
    jobs: list[dict[str, Any]] = []
    for idx, (source, prompt, eps, iters, seed) in enumerate(base, start=1):
        jobs.append(
            {
                "name": f"{idx:02d}_{source}_{prompt}_eps{eps}",
                "source": source,
                "prompt": prompt,
                "epsilon_255": eps,
                "alpha_255": 1,
                "iters": iters,
                "seed": seed,
                "attack": "transformer",
                "max_size": 512,
                "diffusion_steps": 4,
                "highpass_sigma": 0.0,
                "batch": batch,
            }
        )
    return jobs


def score_report(report: dict[str, Any]) -> float:
    inp = report["metrics"]["input"]
    out = report["metrics"]["output"]
    # Hard reject ugly inputs. SSIM is harsh on high-frequency imperceptible noise,
    # so PSNR/mean absolute error carry the primary visual stealth gate.
    if inp["psnr"] < 36.0 or inp["mean_abs"] > 0.014 or inp["max_abs"] > 6.1 / 255.0:
        return -999.0
    return (1.0 - out["ssim"]) * 3.0 + out["l2"] * 4.0


def make_summary_sheet(rows: list[dict[str, Any]], out_path: Path, limit: int = 16) -> None:
    selected = rows[:limit]
    cell_w, cell_h = 520, 244
    sheet = Image.new("RGB", (cell_w, max(1, len(selected)) * cell_h), (235, 235, 235))
    for row_idx, row in enumerate(selected):
        run_dir = Path(row["output_dir"])
        paths = [run_dir / "original.png", run_dir / "perturbed.png", run_dir / "original_diffused.png", run_dir / "perturbed_diffused.png"]
        x = 0
        for path in paths:
            im = Image.open(path).convert("RGB")
            im.thumbnail((120, 160), Image.Resampling.LANCZOS)
            sheet.paste(im, (x + (128 - im.width) // 2, row_idx * cell_h + 8))
            x += 128
        d = ImageDraw.Draw(sheet)
        text = (
            f"{row['rank']:02d} {row['name']} score={row['score']:.3f}\n"
            f"in: psnr={row['input_psnr']:.1f} mean={row['input_mean_abs']:.4f} max={row['input_max_abs']:.4f}\n"
            f"out: ssim={row['output_ssim']:.3f} l2={row['output_l2']:.3f}\n"
            f"{row['prompt']}"
        )
        d.text((8, row_idx * cell_h + 174), text, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="stealth_pass1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    module = load_attack_module()
    jobs = default_jobs(args.batch)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    batch_root = ROOT / "stealth_search" / args.batch
    manifest_path = ROOT / "configs" / args.batch / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"batch": args.batch, "jobs": jobs}, indent=2), encoding="utf-8")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = module.load_pipe(module.torch_dtype("bfloat16"), device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        run_dir = batch_root / job["name"]
        cfg = module.RunConfig(
            input=SOURCES[job["source"]],
            prompt=PROMPTS[job["prompt"]],
            output_dir=str(run_dir),
            seed=job["seed"],
            max_size=job["max_size"],
            diffusion_steps=job["diffusion_steps"],
            epsilon=job["epsilon_255"] / 255.0,
            alpha=job["alpha_255"] / 255.0,
            attack_iters=job["iters"],
            attack=job["attack"],
            highpass_sigma=job["highpass_sigma"],
        )
        print(f"[{index}/{len(jobs)}] {job['name']}", flush=True)
        try:
            report = module.run_with_pipe(cfg, pipe, device)
            score = score_report(report)
            row = {
                "name": job["name"],
                "source": SOURCES[job["source"]],
                "prompt": PROMPTS[job["prompt"]],
                "output_dir": str(run_dir),
                "score": score,
                "input_psnr": report["metrics"]["input"]["psnr"],
                "input_ssim": report["metrics"]["input"]["ssim"],
                "input_mean_abs": report["metrics"]["input"]["mean_abs"],
                "input_max_abs": report["metrics"]["input"]["max_abs"],
                "output_ssim": report["metrics"]["output"]["ssim"],
                "output_l2": report["metrics"]["output"]["l2"],
                "output_psnr": report["metrics"]["output"]["psnr"],
                "epsilon_255": job["epsilon_255"],
                "attack_iters": job["iters"],
                "ok": True,
            }
        except Exception as exc:
            row = {
                "name": job["name"],
                "source": SOURCES[job["source"]],
                "prompt": PROMPTS[job["prompt"]],
                "output_dir": str(run_dir),
                "score": -999.0,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        (batch_root / "progress.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()

    ranked = sorted(rows, key=lambda item: item.get("score", -999.0), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    (batch_root / "ranked.json").write_text(json.dumps({"elapsed_seconds": time.perf_counter() - started, "rows": ranked}, indent=2), encoding="utf-8")
    make_summary_sheet(ranked, ROOT / f"{args.batch}_sheet.jpg")
    print(json.dumps({"ranked": ranked[:8], "sheet": str(ROOT / f"{args.batch}_sheet.jpg")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
