#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch

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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def make_sheet(rows: list[dict[str, Any]], out_path: Path, limit: int = 16) -> None:
    selected = rows[:limit]
    cell_w, cell_h = 760, 220
    sheet = Image.new("RGB", (cell_w, max(1, len(selected)) * cell_h), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(selected):
        run_dir = Path(row["output_dir"])
        x = 8
        for filename in ["original.png", "perturbed.png", "original_diffused.png", "perturbed_diffused.png"]:
            path = run_dir / filename
            if path.exists():
                im = Image.open(path).convert("RGB")
                im.thumbnail((145, 145), Image.Resampling.LANCZOS)
                sheet.paste(im, (x + (155 - im.width) // 2, row_idx * cell_h + 8))
            x += 160
        text = (
            f"{row.get('rank', 0):02d} seed={row['seed']} score={row['score']:.3f}\n"
            f"out ssim={row['output_ssim']:.3f} l2={row['output_l2']:.3f}; "
            f"clean_edit_l2={row['clean_edit_l2']:.3f} pert_edit_l2={row['pert_edit_l2']:.3f}\n"
            f"prompt_failure={row['prompt_failure']:.3f}"
        )
        draw.text((8, row_idx * cell_h + 158), text, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", type=int, default=8000)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    args = parser.parse_args()

    module = load_attack_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidate = Path(args.candidate_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_data = json.loads((candidate / "effective_config.json").read_text(encoding="utf-8"))
    cfg_data["output_dir"] = str(out_root)
    if args.diffusion_steps is not None:
        cfg_data["diffusion_steps"] = args.diffusion_steps
    cfg = module.AttackConfig(**cfg_data)
    original = Image.open(candidate / "original.png").convert("RGB")
    perturbed = Image.open(candidate / "perturbed.png").convert("RGB")
    pipe = module.load_pipe(module.dtype_from_name(cfg.torch_dtype), device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx in range(args.count):
        seed = args.seed_start + idx
        run_dir = out_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.seed = seed
        cfg.output_dir = str(run_dir)
        original.save(run_dir / "original.png")
        perturbed.save(run_dir / "perturbed.png")
        original_diffused = module.diffuse(pipe, original, cfg, device)
        perturbed_diffused = module.diffuse(pipe, perturbed, cfg, device)
        original_diffused.save(run_dir / "original_diffused.png")
        perturbed_diffused.save(run_dir / "perturbed_diffused.png")
        output = module.image_metrics(original_diffused, perturbed_diffused)
        clean_edit = module.image_metrics(original, original_diffused)
        pert_edit = module.image_metrics(perturbed, perturbed_diffused)
        prompt_failure = max(0.0, clean_edit["l2"] - pert_edit["l2"])
        score = (1.0 - output["ssim"]) * 4.0 + output["l2"] * 5.0 + prompt_failure * 4.0
        row = {
            "seed": seed,
            "output_dir": str(run_dir),
            "score": score,
            "output_ssim": output["ssim"],
            "output_l2": output["l2"],
            "clean_edit_l2": clean_edit["l2"],
            "pert_edit_l2": pert_edit["l2"],
            "prompt_failure": prompt_failure,
        }
        rows.append(row)
        ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        write_json(out_root / "progress.json", {"rows": ranked, "elapsed_seconds": time.perf_counter() - started})
        torch.cuda.empty_cache()
    ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    write_json(out_root / "ranked.json", {"rows": ranked, "elapsed_seconds": time.perf_counter() - started})
    write_json(out_root / "effective_config.json", cfg_data)
    make_sheet(ranked, out_root / "sheet.jpg")
    print(json.dumps({"output_dir": str(out_root), "top": ranked[:8]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
