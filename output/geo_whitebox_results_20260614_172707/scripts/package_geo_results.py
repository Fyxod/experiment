#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/workspace/geometric-v1")

from PIL import Image, ImageDraw

from geometric_v1.config import DeepFaceConfig
from geometric_v1.deepface_compare import compare_images

ROOT = Path("/workspace/output/geo_whitebox_results_20260614_172707")

SELECTED = [
    {
        "name": "candidate_01_men51_headshot_rolling_dct",
        "src": ROOT / "search/geo_pass2/05_men51_headshot_rolling_dct_hybrid_2.5px",
        "assessment": "Best geometry-only headshot attempt; output differs slightly but prompt still succeeds.",
        "failure": "weak identity/style drift, not a strong failure",
    },
    {
        "name": "candidate_02_men66_sunglasses_dct",
        "src": ROOT / "search/geo_pass1/09_men66_sunglasses_dct_hybrid_1.5px",
        "assessment": "Best sunglasses geometry attempt; sunglasses still apply, with only small shape/style variation.",
        "failure": "minor output variation, not prompt failure",
    },
    {
        "name": "candidate_03_men57_older_grid_dct",
        "src": ROOT / "search/geo_pass1/03_men57_older_grid_dct_vae_2px",
        "assessment": "Best clean older-prompt geometry attempt; aging still succeeds.",
        "failure": "minor output variation, not prompt failure",
    },
    {
        "name": "candidate_04_men51_headshot_dct",
        "src": ROOT / "search/geo_pass1/07_men51_headshot_dct_hybrid_1.5px",
        "assessment": "Clean DCT-only headshot attempt; edit still succeeds with minor face/style drift.",
        "failure": "minor output variation",
    },
    {
        "name": "candidate_05_men66_denoise_latent_lowinit",
        "src": ROOT / "smoke/denoise_latent_men66_sunglasses_dct_2px_i12_lowinit",
        "assessment": "Cleanest denoise-latent trajectory attempt; edit still succeeds.",
        "failure": "no meaningful failure",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def deepface_summary(report: dict[str, Any]) -> dict[str, Any]:
    vals = []
    models = {}
    for name, result in report.get("models", {}).items():
        if result.get("ok") and result.get("match_percent") is not None:
            vals.append(float(result["match_percent"]))
            models[name] = {
                "match_percent": float(result["match_percent"]),
                "verified": result.get("verified"),
                "distance": result.get("distance"),
                "threshold": result.get("threshold"),
            }
        else:
            models[name] = {"ok": False, "error": result.get("error"), "skipped": result.get("skipped")}
    return {
        "mean_match_percent": sum(vals) / len(vals) if vals else None,
        "min_match_percent": min(vals) if vals else None,
        "max_match_percent": max(vals) if vals else None,
        "models": models,
    }


def run_deepface(dest: Path) -> dict[str, Any]:
    cfg = DeepFaceConfig(workers=1)
    pre = compare_images(dest / "original.png", dest / "perturbed.png", cfg, allow_parallel=False)
    post = compare_images(dest / "original_diffused.png", dest / "perturbed_diffused.png", cfg, allow_parallel=False)
    write_json(dest / "pre_deepface_all.json", pre)
    write_json(dest / "post_deepface_all.json", post)
    return {"pre": deepface_summary(pre), "post": deepface_summary(post)}


def copy_one(item: dict[str, Any], rank: int) -> dict[str, Any]:
    src = item["src"]
    dest = ROOT / "finalists" / item["name"]
    dest.mkdir(parents=True, exist_ok=True)
    for filename in [
        "original.png",
        "perturbed.png",
        "original_diffused.png",
        "perturbed_diffused.png",
        "flow.png",
        "displacement_x16.png",
        "sheet.jpg",
        "report.json",
        "effective_config.json",
    ]:
        if (src / filename).exists():
            shutil.copy2(src / filename, dest / filename)
    report = read_json(dest / "report.json")
    cfg = report["config"]
    row = {
        "rank": rank,
        "name": item["name"],
        "path": str(dest),
        "source": cfg["input"],
        "prompt": cfg["prompt"],
        "methods": ",".join(cfg["methods"]) if isinstance(cfg["methods"], list) else cfg["methods"],
        "objective": cfg["objective"],
        "assessment": item["assessment"],
        "failure": item["failure"],
        "metrics": report["metrics"],
        "settings": {
            "seed": cfg["seed"],
            "max_size": cfg["max_size"],
            "diffusion_steps": cfg["diffusion_steps"],
            "guidance_scale": cfg["guidance_scale"],
            "max_disp_px": cfg["max_disp_px"],
            "attack_iters": cfg["attack_iters"],
            "lr": cfg["lr"],
            "grid_size": cfg["grid_size"],
            "dct_size": cfg["dct_size"],
            "rbf_size": cfg["rbf_size"],
            "padding_mode": cfg["padding_mode"],
            "edge_falloff_px": cfg["edge_falloff_px"],
            "lambda_visual": cfg["lambda_visual"],
            "lambda_disp": cfg["lambda_disp"],
            "lambda_smooth": cfg["lambda_smooth"],
            "lambda_fold": cfg["lambda_fold"],
            "lambda_method_l1": cfg["lambda_method_l1"],
            "objective_timesteps": cfg.get("objective_timesteps"),
        },
    }
    row["deepface"] = run_deepface(dest)
    (dest / "notes.md").write_text(notes_md(row), encoding="utf-8")
    return row


def notes_md(row: dict[str, Any]) -> str:
    m = row["metrics"]
    d = row["deepface"]
    s = row["settings"]
    return f"""# {row['name']}

Assessment: {row['assessment']}

Visible failure type: {row['failure']}

Source: `{row['source']}`

Prompt: `{row['prompt']}`

Geometry methods: `{row['methods']}`

Objective: `{row['objective']}`

Input visual similarity:
- PSNR: {m['input']['psnr']:.2f} dB
- SSIM: {m['input']['ssim']:.4f}

Displacement:
- max magnitude: {m['displacement']['max_magnitude_px']:.3f} px
- mean magnitude: {m['displacement']['mean_magnitude_px']:.3f} px
- p95 magnitude: {m['displacement']['p95_magnitude_px']:.3f} px

Output difference:
- PSNR: {m['output']['psnr']:.2f} dB
- SSIM: {m['output']['ssim']:.4f}
- L2: {m['output']['l2']:.4f}

DeepFace all-model mean match:
- pre: {d['pre']['mean_match_percent']:.2f}%
- post: {d['post']['mean_match_percent']:.2f}%

Settings:
- seed: {s['seed']}
- max_size: {s['max_size']}
- diffusion_steps: {s['diffusion_steps']}
- guidance_scale: {s['guidance_scale']}
- max_disp_px: {s['max_disp_px']}
- attack_iters: {s['attack_iters']}
- lr: {s['lr']}
- padding_mode: {s['padding_mode']}
- edge_falloff_px: {s['edge_falloff_px']}
- objective_timesteps: {s['objective_timesteps']}
"""


def summary_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Geometric White-Box Results Summary",
        "",
        "Branch: `loss3`",
        "",
        "Implemented method: experiment-local differentiable geometry attacks in `scripts/geo_whitebox_attack.py` using `torch.nn.functional.grid_sample`.",
        "",
        "Implemented transforms:",
        "- rolling shutter",
        "- low-resolution grid/free-form displacement",
        "- DCT/Fourier coordinate warp",
        "- RBF/TPS-like sparse control-point warp",
        "- method combinations with L1/L2-style displacement regularization",
        "",
        "Implemented objectives:",
        "- `vae`: Flux.2 Klein VAE image-conditioning latent MSE",
        "- `transformer_pred`: Flux.2 Klein noise-prediction MSE",
        "- `hybrid`: transformer prediction plus small VAE term",
        "- `denoise_latent`: differentiable 4-step denoising latent trajectory MSE",
        "",
        "Baseline reference: previous pixel-PGD retry results remain in `/workspace/output/mentor_results_20260614_123821/finalists`; those are clearly stronger under the same visual standard.",
        "",
        "## Top Geometry-Only Attempts",
        "",
    ]
    for row in rows:
        m = row["metrics"]
        d = row["deepface"]
        s = row["settings"]
        lines.extend(
            [
                f"### {row['rank']}. {row['name']}",
                "",
                f"- Folder: `{row['path']}`",
                f"- Source: `{row['source']}`",
                f"- Prompt: `{row['prompt']}`",
                f"- Geometric method(s): `{row['methods']}`",
                f"- Objective: `{row['objective']}`",
                f"- Visible failure type: {row['failure']}",
                f"- Visual judgment: {row['assessment']}",
                f"- Pre-diffusion visual similarity: PSNR {m['input']['psnr']:.2f} dB, SSIM {m['input']['ssim']:.4f}",
                f"- Displacement stats: max {m['displacement']['max_magnitude_px']:.3f} px, mean {m['displacement']['mean_magnitude_px']:.3f} px, p95 {m['displacement']['p95_magnitude_px']:.3f} px",
                f"- Post-diffusion output difference: PSNR {m['output']['psnr']:.2f} dB, SSIM {m['output']['ssim']:.4f}, L2 {m['output']['l2']:.4f}",
                f"- DeepFace all-model mean match: pre {d['pre']['mean_match_percent']:.2f}%, post {d['post']['mean_match_percent']:.2f}%",
                f"- Optimized settings: seed {s['seed']}, max_disp {s['max_disp_px']} px, iters {s['attack_iters']}, lr {s['lr']}, max_size {s['max_size']}, diffusion_steps {s['diffusion_steps']}, objective_timesteps {s['objective_timesteps']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Honest Assessment",
            "",
            "Under the strict visual gate, geometry-only white-box attacks were not competitive with the pixel-PGD retry. The optimized geometric fields stayed visually presentable, but Flux.2 Klein usually applied the requested edit successfully. Increasing geometry strength produced visible face/crop distortion before producing meaningful edit failures.",
            "",
            "Best mentor demo if a geometry-only result must be shown: `candidate_01_men51_headshot_rolling_dct`, but it should be presented as a weak/negative result rather than a successful attack.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_sheet(rows: list[dict[str, Any]]) -> None:
    cell_w, cell_h = 760, 220
    sheet = Image.new("RGB", (cell_w, cell_h * len(rows)), (238, 238, 238))
    for row_idx, row in enumerate(rows):
        dest = Path(row["path"])
        x = 8
        for filename in ["original.png", "perturbed.png", "original_diffused.png", "perturbed_diffused.png", "flow.png"]:
            im = Image.open(dest / filename).convert("RGB")
            im.thumbnail((138, 138), Image.Resampling.LANCZOS)
            sheet.paste(im, (x + (146 - im.width) // 2, row_idx * cell_h + 8))
            x += 150
        text = f"{row['rank']:02d} {row['name']}\n{row['failure']}\nPSNR {row['metrics']['input']['psnr']:.1f}, out SSIM {row['metrics']['output']['ssim']:.3f}"
        ImageDraw.Draw(sheet).text((8, row_idx * cell_h + 152), text, fill=(0, 0, 0))
    sheet.save(ROOT / "finalists_sheet.jpg", quality=92)


def main() -> None:
    rows = []
    for rank, item in enumerate(SELECTED, start=1):
        row = copy_one(item, rank)
        rows.append(row)
        write_json(ROOT / "finalists" / "progress.json", {"rows": rows})
    write_json(ROOT / "finalists" / "finalists.json", {"rows": rows})
    make_sheet(rows)
    (ROOT / "summary.md").write_text(summary_md(rows), encoding="utf-8")
    print(ROOT / "summary.md")


if __name__ == "__main__":
    main()
