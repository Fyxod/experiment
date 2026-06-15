# Experiment Archive

This repository packages the experiment scripts and results from `/workspace/geometric-v1` and `/workspace/output`.

The root `output/` folder is a copy of `/workspace/output` with `*.pt` tensor checkpoint files removed. Those tensor files were excluded because they made the output tree about 17 GB; they are not needed to inspect results or reproduce runs from the saved configs and scripts.

Excluded tensor files are listed in `excluded_pt_files.txt`.

## Layout

- `adversarial_perturbation/`: pixel-space adversarial perturbation scripts and finalist outputs.
- `next_methods/geometric_whitebox/`: differentiable geometric white-box attack scripts and finalist outputs.
- `next_methods/final_edit_blackbox/`: geometry-only final-edit black-box/CEM search scripts and selected outputs.
- `next_methods/seed_sweeps/`: fixed-geometry seed sweep script and ranked seed outputs.
- `output/`: full experiment output tree, excluding `*.pt`.

## Environment

The scripts were run from `/workspace/geometric-v1` on branch `loss3`, using:

- Python env: `/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11`
- Model: `black-forest-labs/FLUX.2-klein-4B`
- GPU: H100 MIG slice

Most scripts expect the original repo to be importable and the dataset to exist at `/workspace/gender_dataset`.

For reproducible setup, this archive now includes:

- `.python-version`: Python 3.11.
- `requirements.txt`: one global dependency set for all archived scripts.
- `linux-gpu/install_a6000.sh`: A6000-oriented installer adapted from `geometric-v1`.
- `linux-gpu/constraints-a6000.txt`: resolver constraints copied from `geometric-v1`.
- `linux-gpu/verification.md`: import/compiler verification notes from this machine.

Install on an A6000 machine with:

```bash
cd /path/to/experiment
bash linux-gpu/install_a6000.sh
```

If `geometric-v1` is not next to this archive, pass `GEOMETRIC_V1_PATH=/path/to/geometric-v1`.

## Reproduction Notes

Use the `effective_config.json`, `report.json`, summaries, and per-candidate notes in each `out/` folder to reproduce a run. The important parameters are preserved there: source image, prompt, seed, method list, max displacement/epsilon, objective, iterations, guidance scale, diffusion steps, and visual metrics.

The `*.pt` files were optimizer state snapshots. To exactly resume from a saved tensor state they would be useful, but for reproducing from config or evaluating existing images they are not required.
