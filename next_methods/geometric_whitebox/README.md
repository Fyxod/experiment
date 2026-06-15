# Geometric White-Box

This folder contains differentiable geometry attacks against Flux.2 Klein internals.

## Concept

The perturbed image is generated as:

```text
x_geo = T_theta(x)
```

where `T_theta` is a differentiable coordinate warp implemented with `torch.nn.functional.grid_sample`. The optimizer updates geometry parameters instead of raw pixels.

Implemented transform families:

- rolling shutter
- low-resolution grid / free-form deformation
- exact TPS interpolation
- fixed-topology Delaunay/piecewise warp
- DCT/spectral coordinate warp
- affine warp
- radial/lens warp
- face-local masks
- method gates

Implemented Flux objectives:

- VAE image-conditioning latent MSE
- transformer/noise prediction MSE
- hidden-feature divergence
- attention-output proxy divergence
- hybrid objectives
- differentiable denoise-latent trajectory objective

Homography/projective geometry was implemented in the script but intentionally kept out of the main searches because it visibly worsened images without helping.

## Scripts

- `scripts/geo_whitebox_attack.py`: main differentiable geometry optimizer.
- `scripts/run_geo_batch.py`: batch runner for white-box geometry sweeps.
- `scripts/package_geo_results.py`: first packaging script for early results.
- `scripts/package_geo_results_v2.py`: final packaging script for the latest finalists.
- `scripts/geo_blackbox_final_edit.py`: also copied here for convenience, but documented in `../final_edit_blackbox`.
- `scripts/seed_sweep_candidate.py`: also copied here for convenience, but documented in `../seed_sweeps`.

## Outputs

- `out/finalists_v2/`: latest geometry finalists.
- `out/summary_v2.md`: final geometry summary.
- `out/finalists_v2_sheet.jpg`: visual sheet for the finalist set.
- `configs/`: copied batch manifests/configs used during sweeps.

## Reproduce

Example:

```bash
cd /workspace/geometric-v1
/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11 /path/to/experiment/next_methods/geometric_whitebox/scripts/geo_whitebox_attack.py \
  --input /workspace/gender_dataset/men/51.jpg \
  --prompt "turn the image into a professional headshot" \
  --output-dir /tmp/geo_repro \
  --methods tps,piecewise,dct,rolling \
  --objective hybrid_hidden \
  --max-disp-px 2.0 \
  --attack-iters 40 \
  --face-mask face \
  --use-gates
```

For exact settings, use the finalist `effective_config.json` and `report.json` files.

