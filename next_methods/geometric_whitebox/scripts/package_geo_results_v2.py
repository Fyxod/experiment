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
        "name": "geo_v2_01_men51_headshot_seed5007",
        "seed_dir": ROOT / "seed_sweeps/men51_headshot_cand055_32/seed_5007",
        "base_dir": ROOT / "blackbox/men51_headshot_local_seed5203_g6p12/cand_055_g05_m07",
        "source": "/workspace/gender_dataset/men/51.jpg",
        "prompt": "turn the image into a professional headshot",
        "methods": "tps,piecewise,dct,rolling",
        "failure": "identity/outfit/style drift; prompt still mostly succeeds",
        "assessment": "Strongest geometry-only metric result. Perturbed input is clean; edited output changes suit/tie styling and face details, but it is not a full prompt failure.",
    },
    {
        "name": "geo_v2_02_men51_headshot_seed5004",
        "seed_dir": ROOT / "seed_sweeps/men51_headshot_cand055_32/seed_5004",
        "base_dir": ROOT / "blackbox/men51_headshot_local_seed5203_g6p12/cand_055_g05_m07",
        "source": "/workspace/gender_dataset/men/51.jpg",
        "prompt": "turn the image into a professional headshot",
        "methods": "tps,piecewise,dct,rolling",
        "failure": "identity/style drift; prompt still succeeds",
        "assessment": "Lowest output SSIM seed for the best fixed geometry; visually it is mostly a style/identity variation rather than a clean failure.",
    },
    {
        "name": "geo_v2_03_men51_studio_seed7029",
        "seed_dir": ROOT / "seed_sweeps/men51_studio_pass5_64/seed_7029",
        "base_dir": ROOT / "search/geo_pass5_scaled/02_men51_studio_dct_grid_tps_piecewise_rolling_affine_radial_hybrid_2px",
        "source": "/workspace/gender_dataset/men/51.jpg",
        "prompt": "make the person look like a studio portrait",
        "methods": "dct,grid,tps,piecewise,rolling,affine,radial",
        "failure": "background/style drift; prompt still mostly succeeds",
        "assessment": "Best studio-portrait seed. Perturbed edit changes lighting/background and face style, but remains a plausible studio portrait.",
    },
    {
        "name": "geo_v2_04_men51_studio_seed7041",
        "seed_dir": ROOT / "seed_sweeps/men51_studio_pass5_64/seed_7041",
        "base_dir": ROOT / "search/geo_pass5_scaled/02_men51_studio_dct_grid_tps_piecewise_rolling_affine_radial_hybrid_2px",
        "source": "/workspace/gender_dataset/men/51.jpg",
        "prompt": "make the person look like a studio portrait",
        "methods": "dct,grid,tps,piecewise,rolling,affine,radial",
        "failure": "mild structure/style drift; prompt still succeeds",
        "assessment": "Second-best studio seed. Useful as supporting evidence, not as a standalone success demo.",
    },
    {
        "name": "geo_v2_05_men66_sunglasses_seed6010",
        "seed_dir": ROOT / "seed_sweeps/men66_sunglasses_cand030_64/seed_6010",
        "base_dir": ROOT / "blackbox/men66_sunglasses_upper_face_g8p12/cand_030_g03_m06",
        "source": "/workspace/gender_dataset/men/66.jpg",
        "prompt": "add sunglasses",
        "methods": "tps,piecewise,dct,grid,rolling",
        "failure": "minor sunglasses/style variation; not a prompt failure",
        "assessment": "Best sunglasses geometry attempt after seed sweep. Included to document that this prompt stayed robust.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def image_metrics(module: Any, a: Path, b: Path) -> dict[str, float]:
    return module.image_metrics(Image.open(a).convert("RGB"), Image.open(b).convert("RGB"))


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


def make_sheet(dest: Path) -> None:
    items = [
        ("original", "original.png"),
        ("perturbed", "perturbed.png"),
        ("clean edit", "original_diffused.png"),
        ("geo edit", "perturbed_diffused.png"),
        ("flow", "flow.png"),
    ]
    thumbs = []
    for label, filename in items:
        path = dest / filename
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((180, 180), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (198, 220), "white")
        canvas.paste(im, ((198 - im.width) // 2, 8))
        ImageDraw.Draw(canvas).text((8, 198), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (198 * len(thumbs), 220), (238, 238, 238))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (198 * i, 0))
    sheet.save(dest / "sheet.jpg", quality=92)


def notes_md(row: dict[str, Any]) -> str:
    pre = row["metrics"]["input"]
    disp = row["metrics"]["displacement"]
    out = row["metrics"]["output"]
    deep = row["deepface"]
    pre_mean = deep["pre"].get("mean_match_percent")
    post_mean = deep["post"].get("mean_match_percent")
    return f"""# {row['name']}

Assessment: {row['assessment']}

Visible failure type: {row['failure']}

Source: `{row['source']}`

Prompt: `{row['prompt']}`

Geometry methods: `{row['methods']}`

Input visual similarity:
- PSNR: {pre['psnr']:.2f} dB
- SSIM: {pre['ssim']:.4f}

Displacement:
- max magnitude: {disp['max_magnitude_px']:.3f} px
- mean magnitude: {disp['mean_magnitude_px']:.3f} px
- p95 magnitude: {disp['p95_magnitude_px']:.3f} px

Post-diffusion output difference:
- PSNR: {out['psnr']:.2f} dB
- SSIM: {out['ssim']:.4f}
- L2: {out['l2']:.4f}

DeepFace all-model mean match:
- pre: {pre_mean:.2f}%
- post: {post_mean:.2f}%
"""


def package_one(item: dict[str, Any], rank: int, module: Any) -> dict[str, Any]:
    dest = ROOT / "finalists_v2" / item["name"]
    dest.mkdir(parents=True, exist_ok=True)
    for filename in ["original.png", "perturbed.png", "original_diffused.png", "perturbed_diffused.png"]:
        shutil.copy2(item["seed_dir"] / filename, dest / filename)
    for filename in ["flow.png", "displacement_x16.png", "report.json", "effective_config.json", "theta.pt"]:
        src = item["base_dir"] / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
    base_report = read_json(item["base_dir"] / "report.json")
    metrics = {
        "input": image_metrics(module, dest / "original.png", dest / "perturbed.png"),
        "output": image_metrics(module, dest / "original_diffused.png", dest / "perturbed_diffused.png"),
        "displacement": base_report["metrics"]["displacement"],
        "jacobian": base_report["metrics"].get("jacobian", {}),
    }
    report = {
        "rank": rank,
        "name": item["name"],
        "path": str(dest),
        "source": item["source"],
        "prompt": item["prompt"],
        "methods": item["methods"],
        "failure": item["failure"],
        "assessment": item["assessment"],
        "metrics": metrics,
        "base_candidate": str(item["base_dir"]),
        "seed_output": str(item["seed_dir"]),
        "config": base_report.get("config", {}),
        "attack": base_report.get("attack", {}),
    }
    write_json(dest / "report.json", report)
    deep = run_deepface(dest)
    report["deepface"] = deep
    write_json(dest / "report.json", report)
    make_sheet(dest)
    (dest / "notes.md").write_text(notes_md(report), encoding="utf-8")
    return report


def make_finalists_sheet(rows: list[dict[str, Any]]) -> None:
    cell_w, cell_h = 820, 222
    sheet = Image.new("RGB", (cell_w, cell_h * len(rows)), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(rows):
        dest = Path(row["path"])
        x = 8
        for filename in ["original.png", "perturbed.png", "original_diffused.png", "perturbed_diffused.png", "flow.png"]:
            path = dest / filename
            if path.exists():
                im = Image.open(path).convert("RGB")
                im.thumbnail((138, 138), Image.Resampling.LANCZOS)
                sheet.paste(im, (x + (146 - im.width) // 2, row_idx * cell_h + 8))
            x += 152
        m = row["metrics"]
        text = (
            f"{row['rank']:02d} {row['name']}\n"
            f"{row['failure']}\n"
            f"in PSNR {m['input']['psnr']:.1f} SSIM {m['input']['ssim']:.3f}; "
            f"disp max {m['displacement']['max_magnitude_px']:.2f}; out SSIM {m['output']['ssim']:.3f}"
        )
        draw.text((8, row_idx * cell_h + 154), text, fill=(0, 0, 0))
    sheet.save(ROOT / "finalists_v2_sheet.jpg", quality=92)


def summary_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Geometric White-Box Results Summary",
        "",
        "Branch: `loss3`",
        "",
        "Implemented methods: differentiable rolling shutter, low-res FFD grid, exact TPS interpolation, fixed-topology Delaunay/piecewise warp, DCT/spectral coordinate warp, affine similarity warp, radial/lens warp, face-local masks, method gates, hidden-feature/attention-proxy/transformer-pred/VAE/denoise-latent objectives, final-edit black-box CEM, and fixed-geometry seed sweeps. Homography was implemented but intentionally kept out of the main searches per user instruction.",
        "",
        "## Top Candidates",
        "",
    ]
    for row in rows:
        m = row["metrics"]
        d = row["deepface"]
        cfg = row.get("config", {})
        lines.extend(
            [
                f"### {row['rank']}. {row['name']}",
                "",
                f"- Folder: `{row['path']}`",
                f"- Prompt: `{row['prompt']}`",
                f"- Source image: `{row['source']}`",
                f"- Base geometry run: `{row['base_candidate']}`",
                f"- Seed output folder: `{row['seed_output']}`",
                f"- Geometric method(s): `{row['methods']}`",
                f"- Visible failure type: {row['failure']}",
                f"- Visual assessment: {row['assessment']}",
                f"- Pre-diffusion visual similarity: PSNR {m['input']['psnr']:.2f} dB, SSIM {m['input']['ssim']:.4f}",
                f"- Final displacement stats: max {m['displacement']['max_magnitude_px']:.3f} px, mean {m['displacement']['mean_magnitude_px']:.3f} px, p95 {m['displacement']['p95_magnitude_px']:.3f} px",
                f"- Post-diffusion output difference: PSNR {m['output']['psnr']:.2f} dB, SSIM {m['output']['ssim']:.4f}, L2 {m['output']['l2']:.4f}",
                f"- DeepFace all-model mean match: pre {d['pre']['mean_match_percent']:.2f}%, post {d['post']['mean_match_percent']:.2f}%",
                f"- Key settings: seed {cfg.get('seed')}, max_disp_px {cfg.get('max_disp_px')}, attack_iters {cfg.get('attack_iters')}, objective {cfg.get('objective')}, diffusion_steps {cfg.get('diffusion_steps')}, guidance_scale {cfg.get('guidance_scale')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            "",
            "Best 1-2 geometry-only examples to show, if required: `geo_v2_01_men51_headshot_seed5007` and `geo_v2_03_men51_studio_seed7029`. Present them as geometry-induced identity/style drift, not as fully successful prompt-prevention attacks.",
            "",
            "## Honest Assessment",
            "",
            "Geometry-only attacks under strict visual constraints remained weaker than the previous pixel-PGD retry. The fixed-geometry seed sweeps found the most visible output drift, but Flux.2 Klein usually still applied the requested edit. This suggests the geometry direction needs either a stronger semantic/objective signal or a less strict geometry budget to become competitive.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("geo_whitebox_attack", ROOT / "scripts" / "geo_whitebox_attack.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rows = [package_one(item, idx, module) for idx, item in enumerate(SELECTED, start=1)]
    make_finalists_sheet(rows)
    write_json(ROOT / "finalists_v2" / "manifest.json", {"rows": rows})
    text = summary_md(rows)
    (ROOT / "summary.md").write_text(text, encoding="utf-8")
    (ROOT / "summary_v2.md").write_text(text, encoding="utf-8")
    print(json.dumps({"summary": str(ROOT / "summary.md"), "finalists": [row["path"] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
