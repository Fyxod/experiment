#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/workspace/geometric-v1")

from geometric_v1.config import DeepFaceConfig
from geometric_v1.deepface_compare import compare_images

ROOT = Path("/workspace/output/mentor_results_20260614_123821")
FINALISTS = [
    {
        "name": "finalist_01_older_prompt_failure_eps3",
        "src": ROOT / "stealth_search/stealth_pass2/05_men57_older_eps3_i12_hp0p0",
        "failure": "prompt failure",
        "why": "Clean Flux makes the person substantially older; the perturbed edit mostly preserves the original age/hair/face instead.",
        "recommendation": "Best overall: very clean input and obvious edit failure.",
    },
    {
        "name": "finalist_02_headshot_prompt_failure_men58_eps4",
        "src": ROOT / "stealth_search/stealth_pass2/13_men58_headshot_eps4_i12_hp0p0",
        "failure": "prompt failure",
        "why": "Clean Flux turns the image into a suit/headshot; the perturbed edit largely stays in the original casual pose/composition.",
        "recommendation": "Best headshot demo; the failure is easy to explain visually.",
    },
    {
        "name": "finalist_03_headshot_prompt_failure_men51_eps3",
        "src": ROOT / "stealth_search/stealth_pass2/14_men51_headshot_eps3_i12_hp0p0",
        "failure": "prompt failure",
        "why": "Clean Flux produces a professional suit headshot; the perturbed edit keeps the plaid shirt and original composition.",
        "recommendation": "Very clean input and intuitive before/after story.",
    },
    {
        "name": "finalist_04_sunglasses_prompt_failure_eps3",
        "src": ROOT / "stealth_search/stealth_pass1/01_men66_sunglasses_eps3",
        "failure": "prompt failure",
        "why": "Clean Flux replaces regular glasses with dark sunglasses; the perturbed edit mostly keeps the original glasses.",
        "recommendation": "Best sunglasses example.",
    },
    {
        "name": "finalist_05_headshot_identity_drift_men11_eps3",
        "src": ROOT / "stealth_search/stealth_pass2/10_men11_headshot_eps3_i12_hp0p0",
        "failure": "identity drift",
        "why": "Both outputs are headshots, but the perturbed edit changes face shape, hair, and perceived age more than the clean edit.",
        "recommendation": "Useful backup for identity-drift discussion.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def copy_candidate(item: dict[str, Any], index: int) -> dict[str, Any]:
    src = item["src"]
    dest = ROOT / "finalists" / item["name"]
    dest.mkdir(parents=True, exist_ok=True)
    for filename in [
        "original.png",
        "perturbed.png",
        "original_diffused.png",
        "perturbed_diffused.png",
        "delta_x16.png",
        "sheet.jpg",
    ]:
        shutil.copy2(src / filename, dest / filename)
    shutil.copy2(src / "report.json", dest / "report.json")
    shutil.copy2(src / "config.json", dest / "effective_config.json")
    report = read_json(dest / "report.json")
    cfg = report["config"]
    return {
        "rank": index,
        "name": item["name"],
        "path": str(dest),
        "source": cfg["input"],
        "prompt": cfg["prompt"],
        "failure": item["failure"],
        "why": item["why"],
        "recommendation": item["recommendation"],
        "metrics": report["metrics"],
        "settings": {
            "seed": cfg["seed"],
            "max_size": cfg["max_size"],
            "diffusion_steps": cfg["diffusion_steps"],
            "guidance_scale": cfg["guidance_scale"],
            "epsilon": cfg["epsilon"],
            "epsilon_255": cfg["epsilon"] * 255.0,
            "alpha": cfg["alpha"],
            "alpha_255": cfg["alpha"] * 255.0,
            "attack_iters": cfg["attack_iters"],
            "attack": cfg["attack"],
            "objective_timestep": cfg["objective_timestep"],
            "highpass_sigma": cfg["highpass_sigma"],
            "random_start": cfg["random_start"],
        },
    }


def deepface_summary(report: dict[str, Any]) -> dict[str, Any]:
    values = []
    models = {}
    for name, result in report.get("models", {}).items():
        if result.get("ok") and result.get("match_percent") is not None:
            values.append(float(result["match_percent"]))
            models[name] = {
                "match_percent": float(result["match_percent"]),
                "verified": result.get("verified"),
                "distance": result.get("distance"),
                "threshold": result.get("threshold"),
            }
        else:
            models[name] = {"ok": False, "error": result.get("error"), "skipped": result.get("skipped")}
    return {
        "mean_match_percent": sum(values) / len(values) if values else None,
        "min_match_percent": min(values) if values else None,
        "max_match_percent": max(values) if values else None,
        "models": models,
    }


def run_deepface(dest: Path) -> dict[str, Any]:
    config = DeepFaceConfig(workers=1)
    pre_path = dest / "pre_deepface_all.json"
    post_path = dest / "post_deepface_all.json"
    pre = compare_images(dest / "original.png", dest / "perturbed.png", config, allow_parallel=False)
    post = compare_images(dest / "original_diffused.png", dest / "perturbed_diffused.png", config, allow_parallel=False)
    write_json(pre_path, pre)
    write_json(post_path, post)
    return {"pre": deepface_summary(pre), "post": deepface_summary(post)}


def notes_md(row: dict[str, Any]) -> str:
    m = row["metrics"]
    s = row["settings"]
    d = row["deepface"]
    return f"""# {row['name']}

Failure type: {row['failure']}

Why it is useful: {row['why']}

Recommendation: {row['recommendation']}

Source: `{row['source']}`

Prompt: `{row['prompt']}`

Input stealth:
- PSNR: {m['input']['psnr']:.2f} dB
- SSIM: {m['input']['ssim']:.4f}
- mean abs: {m['input']['mean_abs']:.5f}
- max abs: {m['input']['max_abs']:.5f} ({s['epsilon_255']:.1f}/255 bound)

Output disruption:
- PSNR: {m['output']['psnr']:.2f} dB
- SSIM: {m['output']['ssim']:.4f}
- L2: {m['output']['l2']:.4f}
- mean abs: {m['output']['mean_abs']:.5f}

DeepFace all-model mean match:
- Pre-diffusion original vs perturbed: {d['pre']['mean_match_percent']:.2f}%
- Post-diffusion original edit vs perturbed edit: {d['post']['mean_match_percent']:.2f}%

Settings:
- seed: {s['seed']}
- max_size: {s['max_size']}
- diffusion_steps: {s['diffusion_steps']}
- guidance_scale: {s['guidance_scale']}
- attack: {s['attack']}
- attack_iters: {s['attack_iters']}
- epsilon: {s['epsilon_255']:.1f}/255
- alpha: {s['alpha_255']:.1f}/255
- objective_timestep: {s['objective_timestep']}
- highpass_sigma: {s['highpass_sigma']}
- random_start: {s['random_start']}
"""


def summary_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Mentor Results Retry Summary",
        "",
        "Branch: `loss3`",
        "",
        "Method: stealth pixel-space PGD against the Flux.2 Klein conditioning path. This replaced the earlier geometric-warp search because those perturbations were too visible.",
        "",
        "External method inspiration used while redesigning the search:",
        "- PhotoGuard / adversarial perturbations for image editing: https://www.eecs.mit.edu/using-ai-to-protect-against-ai-image-manipulation/",
        "- EditShield latent-distribution protection idea: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07991.pdf",
        "- Low-frequency/DCT attack context: https://proceedings.mlr.press/v115/guo20a.html",
        "",
        "## Top Candidates",
        "",
    ]
    for row in rows:
        m = row["metrics"]
        s = row["settings"]
        d = row["deepface"]
        lines.extend(
            [
                f"### {row['rank']}. {row['name']}",
                "",
                f"- Folder: `{row['path']}`",
                f"- Source: `{row['source']}`",
                f"- Prompt: `{row['prompt']}`",
                f"- Visible failure type: {row['failure']}",
                f"- Visual judgment: {row['why']}",
                f"- Pre-diffusion visual similarity: PSNR {m['input']['psnr']:.2f} dB, SSIM {m['input']['ssim']:.4f}, mean abs {m['input']['mean_abs']:.5f}, max abs {m['input']['max_abs']:.5f}",
                f"- Post-diffusion output difference: PSNR {m['output']['psnr']:.2f} dB, SSIM {m['output']['ssim']:.4f}, L2 {m['output']['l2']:.4f}, mean abs {m['output']['mean_abs']:.5f}",
                f"- DeepFace all-model mean match: pre {d['pre']['mean_match_percent']:.2f}%, post {d['post']['mean_match_percent']:.2f}%",
                f"- Settings: seed {s['seed']}, max_size {s['max_size']}, diffusion_steps {s['diffusion_steps']}, guidance {s['guidance_scale']}, attack {s['attack']}, iters {s['attack_iters']}, epsilon {s['epsilon_255']:.1f}/255, alpha {s['alpha_255']:.1f}/255, objective_timestep {s['objective_timestep']}, highpass_sigma {s['highpass_sigma']}, random_start {s['random_start']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            "",
            "Show `finalist_01_older_prompt_failure_eps3` first: it has a clean input and the most obvious requested-edit failure.",
            "",
            "Show `finalist_02_headshot_prompt_failure_men58_eps4` or `finalist_03_headshot_prompt_failure_men51_eps3` second: both demonstrate that a nearly invisible perturbation can stop the professional-headshot transformation.",
            "",
            "`finalist_04_sunglasses_prompt_failure_eps3` is the best compact prompt-specific example: regular glasses remain instead of becoming sunglasses.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = []
    for idx, item in enumerate(FINALISTS, start=1):
        row = copy_candidate(item, idx)
        dest = Path(row["path"])
        row["deepface"] = run_deepface(dest)
        (dest / "notes.md").write_text(notes_md(row), encoding="utf-8")
        rows.append(row)
        write_json(ROOT / "finalists" / "progress.json", {"rows": rows})
    write_json(ROOT / "finalists" / "finalists.json", {"rows": rows})
    (ROOT / "summary.md").write_text(summary_md(rows), encoding="utf-8")
    print(ROOT / "summary.md")


if __name__ == "__main__":
    main()
