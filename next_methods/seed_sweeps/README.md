# Seed Sweeps

This folder contains fixed-geometry seed sweeps.

## Concept

After finding a presentable geometric perturbation, the geometry is kept fixed while Flux edit seeds are varied. This checks whether some diffusion seeds expose larger clean-vs-perturbed output drift.

This does not change the perturbation. It only reruns Flux editing with different seeds.

## Script

- `scripts/seed_sweep_candidate.py`: loads `original.png` and `perturbed.png` from a candidate folder, reruns clean and perturbed Flux edits across a seed range, ranks by output difference and under-editing, and writes a sheet.

## Outputs

- `out/men51_headshot_cand055_32`: strongest seed-sweep set.
- `out/men51_studio_pass5_64`: studio-portrait seed sweep.
- `out/men66_sunglasses_cand030_64`: sunglasses seed sweep, mostly weak.
- `out/men51_hair_pass3_64`: hairstyle seed sweep, weak.

## Reproduce

Example:

```bash
cd /workspace/geometric-v1
/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11 /path/to/experiment/next_methods/seed_sweeps/scripts/seed_sweep_candidate.py \
  --candidate-dir /workspace/output/geo_whitebox_results_20260614_172707/blackbox/men51_headshot_local_seed5203_g6p12/cand_055_g05_m07 \
  --output-dir /tmp/seed_sweep_repro \
  --seed-start 5000 \
  --count 32
```

