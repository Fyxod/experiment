# Adversarial Perturbation

This folder contains the pixel-space adversarial perturbation work from `output/mentor_results_20260614_123821`.

## Concept

These runs optimize small pixel perturbations against Flux.2 Klein internals while keeping the input visually close to the original. This is the stronger baseline compared with the geometry-only methods.

The main goal was to find demo-worthy cases where:

- `perturbed.png` looks close to `original.png`
- Flux editing succeeds on the original
- Flux editing on the perturbed input drifts, fails, or degrades

## Scripts

- `scripts/stealth_flux_attack.py`: main pixel-space attack script.
- `scripts/run_stealth_batch.py`: batch runner for multiple images/prompts/settings.
- `scripts/package_finalists.py`: packages finalist outputs and summaries.

## Outputs

- `out/finalists/`: curated finalist examples.
- `out/summary.md`: final summary with prompts, settings, metrics, and recommendations.
- `out/finalists_sheet_retry.jpg`: visual contact sheet of the finalist set.

## Reproduce

Use the copied configs in `configs/` and the per-finalist `effective_config.json` / `report.json` files under `out/finalists/`.

Run from the original repo environment:

```bash
cd /workspace/geometric-v1
/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11 /path/to/experiment/adversarial_perturbation/scripts/stealth_flux_attack.py --help
```

