# Final-Edit Black-Box Geometry Search

This folder contains the geometry-only black-box/CEM search that scores actual edited Flux outputs.

## Concept

The differentiable internal objectives were often too weak, so this method searches over geometry parameters and evaluates the final edited images directly:

1. sample geometry parameters
2. reject candidates that fail visual gates
3. run Flux edit on the perturbed image
4. score output divergence / prompt under-editing
5. update the sampling distribution from elite candidates

This remains geometry-only: the image is changed through coordinate warps, not pixel noise.

## Scripts

- `scripts/geo_blackbox_final_edit.py`: CEM-style final-edit scorer.
- `scripts/package_geo_results_v2.py`: packages selected outputs into finalists.

## Outputs

- `out/selected_candidates/`: selected CEM candidates used for finalist/seed-sweep work.
- `out/finalists_v2/`: packaged finalists created from the best geometry candidates and seed sweeps.
- `out/summary_v2.md`: summary of the final geometry-only results.
- `out/finalists_v2_sheet.jpg`: visual contact sheet.

## Reproduce

Example:

```bash
cd /workspace/geometric-v1
/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11 /path/to/experiment/next_methods/final_edit_blackbox/scripts/geo_blackbox_final_edit.py \
  --job-name repro_men51_headshot \
  --source men51 \
  --prompt-key headshot \
  --methods tps,piecewise,dct,rolling \
  --face-mask face \
  --use-gates \
  --max-disp-px 2.5 \
  --generations 6 \
  --population 12 \
  --elite 3 \
  --sample-std 0.18 \
  --seed 5203
```

The strongest candidate came from local CEM around a white-box result, then a seed sweep.

