#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/output/mentor_results_20260614_092647")
REPO = Path("/workspace/geometric-v1")
PYTHON = REPO / ".venv-linux-gpu/bin/python"

FAST_MODELS = {"SFace": True, "OpenFace": False, "Facenet": False, "Facenet512": True}
ALL_MODELS = {"SFace": True, "OpenFace": True, "Facenet": True, "Facenet512": True}

SOURCES = {
    "sample": REPO / "samples/image.png",
    "men11": Path("/workspace/gender_dataset/men/11.jpg"),
    "men16": Path("/workspace/gender_dataset/men/16.jpg"),
    "men19": Path("/workspace/gender_dataset/men/19.jpg"),
    "men27": Path("/workspace/gender_dataset/men/27.jpg"),
    "men32": Path("/workspace/gender_dataset/men/32.jpg"),
}

PROMPTS = {
    "sunglasses": "add sunglasses",
    "studio": "make the person look like a studio portrait",
    "hair": "change the hairstyle",
    "smile": "make the person smile",
    "older": "make the person older",
    "headshot": "turn the image into a professional headshot",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_pipeline(source: Path, prompt: str, seed: int, max_size: int, steps: int, models: dict[str, bool]) -> dict[str, Any]:
    cfg = read_json(REPO / "pipeline.json")
    cfg["input"] = str(source)
    cfg["output_dir"] = str(ROOT / "unused_pipeline_output")
    cfg["prompt"] = prompt
    cfg["seed"] = seed
    flux = cfg["diffusion"]["models"]["flux2_klein"]
    flux.update(
        {
            "enabled": True,
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
            "max_size": max_size,
            "height": None,
            "width": None,
            "torch_dtype": "bfloat16",
            "cpu_offload": False,
            "seed": seed,
        }
    )
    cfg["diffusion"]["models"]["instruct_pix2pix"]["enabled"] = False
    cfg["deepface"]["enabled"] = True
    cfg["deepface"]["workers"] = 1
    cfg["deepface"]["models"] = models
    return cfg


def perturbation_templates(profile: str) -> list[dict[str, Any]]:
    homography = profile in {"relaxed", "strong"}
    return [
        {"method": "homography", "enabled": homography, "strength": 0.006},
        {"method": "thin-plate-spline", "enabled": True, "strength": 0.0015, "grid": 7},
        {"method": "delaunay", "enabled": True, "strength": 0.0035, "grid": 8},
        {"method": "fft-phase", "enabled": True, "strength": 0.35, "coefficients": 8},
        {"method": "elastic", "enabled": True, "strength": 0.0025, "sigma": 12.0},
        {
            "method": "rolling-shutter",
            "enabled": True,
            "strength": 0.0025,
            "rolling_frequency": 1.4,
            "rolling_phase": 0.0,
            "rolling_shear": 0.012,
            "rolling_acceleration": 0.0,
        },
    ]


def bounds(profile: str) -> dict[str, dict[str, list[float]]]:
    if profile == "strong":
        return {
            "homography": {"strength": [0.0, 0.028]},
            "thin-plate-spline": {"strength": [0.0, 0.009], "grid": [5, 10]},
            "delaunay": {"strength": [0.0, 0.018], "grid": [6, 12]},
            "fft-phase": {"strength": [0.0, 5.0], "coefficients": [4, 18]},
            "elastic": {"strength": [0.0, 0.018], "sigma": [5.0, 22.0]},
            "rolling-shutter": {
                "strength": [0.0, 0.018],
                "rolling_frequency": [0.5, 3.5],
                "rolling_shear": [0.0, 0.08],
                "rolling_acceleration": [-0.05, 0.05],
            },
        }
    if profile == "relaxed":
        return {
            "homography": {"strength": [0.0, 0.018]},
            "thin-plate-spline": {"strength": [0.0, 0.006], "grid": [5, 10]},
            "delaunay": {"strength": [0.0, 0.012], "grid": [6, 12]},
            "fft-phase": {"strength": [0.0, 2.0], "coefficients": [4, 16]},
            "elastic": {"strength": [0.0, 0.012], "sigma": [6.0, 20.0]},
            "rolling-shutter": {
                "strength": [0.0, 0.012],
                "rolling_frequency": [0.5, 3.0],
                "rolling_shear": [0.0, 0.06],
                "rolling_acceleration": [-0.035, 0.035],
            },
        }
    return {
        "thin-plate-spline": {"strength": [0.0, 0.0045], "grid": [5, 9]},
        "delaunay": {"strength": [0.0, 0.009], "grid": [6, 11]},
        "fft-phase": {"strength": [0.0, 1.0], "coefficients": [4, 14]},
        "elastic": {"strength": [0.0, 0.008], "sigma": [7.0, 18.0]},
        "rolling-shutter": {
            "strength": [0.0, 0.008],
            "rolling_frequency": [0.5, 2.6],
            "rolling_shear": [0.0, 0.04],
            "rolling_acceleration": [-0.025, 0.025],
        },
    }


def make_embedding(
    pipeline_path: Path,
    source: Path,
    prompt: str,
    seed: int,
    out_dir: Path,
    profile: str,
    iterations: int,
    optimizer: str,
    flux_features: bool,
) -> dict[str, Any]:
    cfg = read_json(REPO / "embedding_loss.json")
    cfg["pipeline_config"] = str(pipeline_path)
    cfg["input"] = str(source)
    cfg["output_dir"] = str(out_dir)
    cfg["prompt"] = prompt
    cfg["seed"] = seed
    cfg["randomize_seed"] = False
    cfg["perturbations"] = perturbation_templates(profile)
    cfg["optimizer"].update(
        {
            "type": optimizer,
            "iterations": iterations,
            "random_restarts": 1,
            "save_every": 1,
            "patience": 0,
            "learning_rate": 0.05,
            "spsa_delta": 0.07,
            "finite_difference_epsilon": 0.07,
            "stop": {"loss_below": None, "post_identity_distance_above": None, "output_disruption_above": None},
        }
    )
    cfg["identity"]["overrides"]["models"] = FAST_MODELS
    ident = cfg["objective"]["identity"]
    ident["models"] = FAST_MODELS
    ident["pre_identity_target"] = 94.0 if profile == "stealth" else 90.0
    ident["pre_identity_weight"] = 6.0 if profile == "stealth" else 4.0
    ident["post_identity_distance_weight"] = {"stealth": 2.5, "relaxed": 5.0, "strong": 8.0}[profile]
    stealth = cfg["objective"]["input_stealth"]
    if profile == "strong":
        stealth["psnr"] = {"target": 23.0, "weight": 1.2}
        stealth["ssim"] = {"target": 0.76, "weight": 12.0}
    elif profile == "relaxed":
        stealth["psnr"] = {"target": 27.0, "weight": 2.0}
        stealth["ssim"] = {"target": 0.84, "weight": 20.0}
    else:
        stealth["psnr"] = {"target": 30.0, "weight": 3.0}
        stealth["ssim"] = {"target": 0.9, "weight": 30.0}
    cfg["objective"]["vae_latent"].update({"enabled": True, "use_input_vae": False, "use_output_vae": True, "output_distance_weight": 1.0})
    cfg["objective"]["clip_image"]["enabled"] = False
    cfg["objective"]["flux_transformer"]["enabled"] = bool(flux_features)
    cfg["objective"]["flux_transformer"]["weight"] = 1.0
    cfg["objective"]["output_disruption"].update(
        {
            "enabled": True,
            "pixel_l2_distance_weight": {"stealth": 1.5, "relaxed": 2.0, "strong": 3.0}[profile],
            "ssim_drop_weight": {"stealth": 2.0, "relaxed": 3.0, "strong": 4.0}[profile],
        }
    )
    cfg["objective"]["parameter_regularization"].update({"enabled": True, "weight": 0.02})
    cfg["parameters"]["initialization"] = "fixed"
    cfg["parameters"]["initial_values"] = {
        "thin-plate-spline": {"strength": 0.0015, "grid": 7},
        "delaunay": {"strength": 0.0035, "grid": 8},
        "fft-phase": {"strength": 0.35, "coefficients": 8},
        "elastic": {"strength": 0.0025, "sigma": 12.0},
        "rolling-shutter": {
            "strength": 0.0025,
            "rolling_frequency": 1.4,
            "rolling_shear": 0.012,
            "rolling_acceleration": 0.0,
        },
        "homography": {"strength": 0.006},
    }
    cfg["parameters"]["bounds"] = bounds(profile)
    return cfg


def make_configs(args: argparse.Namespace) -> None:
    configs_dir = ROOT / "configs" / args.batch
    run_root = ROOT / "search" / args.batch
    combos = [item.split(":", 1) for item in args.combo]
    manifest: list[dict[str, Any]] = []
    for index, (source_key, prompt_key) in enumerate(combos, start=1):
        source = SOURCES[source_key]
        prompt = PROMPTS[prompt_key]
        name = f"{index:02d}_{source_key}_{prompt_key}_{args.profile}"
        seed = args.seed + index
        pipe_path = configs_dir / f"{name}_pipeline.json"
        emb_path = configs_dir / f"{name}_embedding.json"
        write_json(pipe_path, make_pipeline(source, prompt, seed, args.max_size, args.steps, FAST_MODELS))
        write_json(
            emb_path,
            make_embedding(pipe_path, source, prompt, seed, run_root / name, args.profile, args.iterations, args.optimizer, args.flux_features),
        )
        manifest.append({"name": name, "source": str(source), "prompt": prompt, "config": str(emb_path), "log": str(ROOT / "logs" / f"{args.batch}_{name}.log")})
    write_json(configs_dir / "manifest.json", {"batch": args.batch, "items": manifest})
    print(configs_dir / "manifest.json")


def run_manifest(args: argparse.Namespace) -> None:
    manifest = read_json(ROOT / "configs" / args.batch / "manifest.json")["items"]
    for item in manifest:
        log_path = Path(item["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(PYTHON), "run_embedding_loss_pipeline.py", "--config", item["config"]]
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
        print(item["name"], proc.returncode, log_path)
        if proc.returncode != 0 and args.stop_on_error:
            raise SystemExit(proc.returncode)


def collect_reports() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted((ROOT / "search").glob("**/embedding_loss_run_*/report.json")):
        if report_path.parent.name == "best":
            continue
        try:
            data = read_json(report_path)
        except Exception:
            continue
        metrics = data.get("best_metrics") or {}
        stealth = metrics.get("input_stealth") or {}
        identity = metrics.get("identity") or {}
        pre = identity.get("pre") or {}
        post = identity.get("post") or {}
        disruption = metrics.get("output_disruption") or {}
        rows.append(
            {
                "report": str(report_path),
                "run_dir": str(report_path.parent),
                "prompt": data.get("prompt"),
                "source": data.get("input"),
                "evaluations": data.get("evaluations"),
                "best_iteration": data.get("best_iteration"),
                "loss": data.get("best_loss"),
                "psnr": stealth.get("psnr"),
                "ssim": stealth.get("ssim"),
                "pre_similarity": pre.get("mean_similarity_percent"),
                "post_distance": post.get("mean_distance_percent"),
                "post_similarity": post.get("mean_similarity_percent"),
                "output_ssim_drop": disruption.get("ssim_drop"),
                "output_pixel_l2": disruption.get("pixel_l2"),
                "best_dir": data.get("outputs", {}).get("best"),
                "original": data.get("outputs", {}).get("original"),
                "original_diffused": data.get("outputs", {}).get("original_diffused"),
            }
        )
    def score(row: dict[str, Any]) -> tuple[bool, float]:
        pre_ok = (row.get("pre_similarity") or 0.0) >= 80.0
        post = row.get("post_distance") or 0.0
        drop = row.get("output_ssim_drop") or 0.0
        pix = row.get("output_pixel_l2") or 0.0
        stealth = max(0.0, min(1.0, ((row.get("ssim") or 0.0) - 0.70) / 0.25))
        return pre_ok, post * 1.0 + drop * 50.0 + pix * 20.0 + stealth * 2.0

    rows.sort(key=score, reverse=True)
    return rows


def scan(args: argparse.Namespace) -> None:
    rows = collect_reports()
    out = ROOT / "search" / "candidate_scan.json"
    write_json(out, {"items": rows})
    for i, row in enumerate(rows[: args.top], start=1):
        print(
            f"{i:02d} post={row.get('post_distance'):.2f} pre={row.get('pre_similarity'):.2f} "
            f"psnr={row.get('psnr'):.2f} ssim={row.get('ssim'):.3f} "
            f"drop={row.get('output_ssim_drop'):.3f} {row.get('prompt')} {row.get('run_dir')}"
        )
    print(out)


def package(args: argparse.Namespace) -> None:
    rows = collect_reports()[: args.top]
    for index, row in enumerate(rows, start=1):
        dest = ROOT / "candidates" / f"candidate_{index:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        best = Path(row["best_dir"])
        for src, name in (
            (Path(row["original"]), "original.png"),
            (best / "perturbed.png", "perturbed.png"),
            (Path(row["original_diffused"]), "original_diffused.png"),
            (best / "perturbed_diffused.png", "perturbed_diffused.png"),
            (best / "report.json", "report.json"),
            (Path(row["report"]).parent / "embedding_loss_config.json", "effective_config.json"),
        ):
            if src.exists():
                shutil.copy2(src, dest / name)
        notes = [
            f"# Candidate {index}",
            "",
            f"- Prompt: {row.get('prompt')}",
            f"- Source: {row.get('source')}",
            f"- Run: {row.get('run_dir')}",
            f"- PSNR / SSIM: {row.get('psnr')} / {row.get('ssim')}",
            f"- Pre identity similarity: {row.get('pre_similarity')}",
            f"- Post identity distance: {row.get('post_distance')}",
            f"- Output SSIM drop: {row.get('output_ssim_drop')}",
            "",
            "Visible failure type: TODO after visual inspection.",
        ]
        (dest / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(dest)


def collect_iterations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((ROOT / "search").glob("**/embedding_loss_run_*/iterations/iter_*/metrics.json")):
        try:
            data = read_json(metrics_path)
        except Exception:
            continue
        run_dir = metrics_path.parents[2]
        run_report_path = run_dir / "report.json"
        try:
            run_report = read_json(run_report_path)
        except Exception:
            run_report = {}
        metrics = data.get("metrics") or {}
        stealth = metrics.get("input_stealth") or {}
        identity = metrics.get("identity") or {}
        pre = identity.get("pre") or {}
        post = identity.get("post") or {}
        disruption = metrics.get("output_disruption") or {}
        row = {
            "metrics": str(metrics_path),
            "run_report": str(run_report_path),
            "run_dir": str(run_dir),
            "iteration_dir": str(metrics_path.parent),
            "iteration": data.get("iteration"),
            "label": data.get("label"),
            "prompt": run_report.get("prompt") or data.get("prompt"),
            "source": run_report.get("input"),
            "loss": data.get("loss"),
            "psnr": stealth.get("psnr"),
            "ssim": stealth.get("ssim"),
            "pre_similarity": pre.get("mean_similarity_percent"),
            "post_distance": post.get("mean_distance_percent"),
            "post_similarity": post.get("mean_similarity_percent"),
            "output_ssim_drop": disruption.get("ssim_drop"),
            "output_pixel_l2": disruption.get("pixel_l2"),
            "original": run_report.get("outputs", {}).get("original"),
            "original_diffused": run_report.get("outputs", {}).get("original_diffused"),
            "perturbed": data.get("outputs", {}).get("perturbed"),
            "perturbed_diffused": data.get("outputs", {}).get("perturbed_diffused"),
        }
        pre_sim = row.get("pre_similarity") or 0.0
        ssim = row.get("ssim") or 0.0
        psnr = row.get("psnr") or 0.0
        post = row.get("post_distance") or 0.0
        drop = row.get("output_ssim_drop") or 0.0
        pix = row.get("output_pixel_l2") or 0.0
        # Demo ranking: keep presentability in the score, but let visible output failure matter.
        row["demo_score"] = (
            (2.0 if pre_sim >= 85.0 else 0.0)
            + max(0.0, min(2.0, (ssim - 0.65) * 8.0))
            + max(0.0, min(1.0, (psnr - 18.0) / 8.0))
            + post
            + 55.0 * drop
            + 20.0 * pix
        )
        rows.append(row)
    rows.sort(key=lambda r: r.get("demo_score") or 0.0, reverse=True)
    return rows


def scan_iterations(args: argparse.Namespace) -> None:
    rows = collect_iterations()
    out = ROOT / "search" / "iteration_candidate_scan.json"
    write_json(out, {"items": rows})
    for i, row in enumerate(rows[: args.top], start=1):
        print(
            f"{i:02d} score={row.get('demo_score'):.2f} post={row.get('post_distance'):.2f} "
            f"pre={row.get('pre_similarity'):.2f} psnr={row.get('psnr'):.2f} "
            f"ssim={row.get('ssim'):.3f} drop={row.get('output_ssim_drop'):.3f} "
            f"{row.get('prompt')} {row.get('iteration_dir')}"
        )
    print(out)


def package_iterations(args: argparse.Namespace) -> None:
    rows = collect_iterations()[: args.top]
    dest_root = ROOT / args.dest
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        dest = dest_root / f"candidate_{index:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        for src_value, name in (
            (row.get("original"), "original.png"),
            (row.get("perturbed"), "perturbed.png"),
            (row.get("original_diffused"), "original_diffused.png"),
            (row.get("perturbed_diffused"), "perturbed_diffused.png"),
            (row.get("metrics"), "report.json"),
            (Path(row["run_dir"]) / "embedding_loss_config.json", "effective_config.json"),
        ):
            if not src_value:
                continue
            src = Path(src_value)
            if src.exists():
                shutil.copy2(src, dest / name)
        notes = [
            f"# Iteration Candidate {index}",
            "",
            f"- Prompt: {row.get('prompt')}",
            f"- Source: {row.get('source')}",
            f"- Run: {row.get('run_dir')}",
            f"- Iteration: {row.get('iteration')} / {row.get('label')}",
            f"- Demo score: {row.get('demo_score')}",
            f"- PSNR / SSIM: {row.get('psnr')} / {row.get('ssim')}",
            f"- Pre identity similarity: {row.get('pre_similarity')}",
            f"- Post identity distance: {row.get('post_distance')}",
            f"- Output SSIM drop: {row.get('output_ssim_drop')}",
            "",
            "Visible failure type: TODO after visual inspection.",
        ]
        (dest / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-configs")
    p.add_argument("--batch", required=True)
    p.add_argument("--profile", choices=["stealth", "relaxed", "strong"], default="stealth")
    p.add_argument("--combo", action="append", required=True, help="source_key:prompt_key")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--optimizer", choices=["random_search", "spsa"], default="random_search")
    p.add_argument("--max-size", type=int, default=768)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--flux-features", action="store_true")
    p.set_defaults(func=make_configs)
    p = sub.add_parser("run")
    p.add_argument("--batch", required=True)
    p.add_argument("--stop-on-error", action="store_true")
    p.set_defaults(func=run_manifest)
    p = sub.add_parser("scan")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=scan)
    p = sub.add_parser("package")
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=package)
    p = sub.add_parser("scan-iters")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=scan_iterations)
    p = sub.add_parser("package-iters")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--dest", default="candidates_iterations")
    p.set_defaults(func=package_iterations)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
