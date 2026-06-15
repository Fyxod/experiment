# Continuation Checkpoint

- Repo: `/workspace/geometric-v1`
- Branch: `loss3`
- Experiment root: `/workspace/output/mentor_results_20260614_092647`
- Logs: `/workspace/output/mentor_results_20260614_092647/logs`
- Helper script: `/workspace/output/mentor_results_20260614_092647/scripts/mentor_search.py`
- Python env: `/workspace/geometric-v1/.venv-linux-gpu`
- Install adjustment: removed `opencv-python` and force-reinstalled `opencv-python-headless==4.8.1.78` because this headless server lacks `libGL.so.1`.
- Hugging Face token: not needed so far. `black-forest-labs/FLUX.2-klein-4B` is public and not gated.
- Smoke tests passed:
  - Perturb-only: `logs/smoke_perturb.log`
  - DeepFace SFace/Facenet512: `logs/smoke_deepface_fast.log`
  - Flux.2 Klein one-step pipeline: `logs/smoke_flux.log`
  - Embedding-loss tiny run: `logs/smoke_embedding_loss.log`
- Current search:
  - Batch `pass1` completed successfully.
  - Manifest: `configs/pass1/manifest.json`
  - Strategy: loss3 embedding-loss, Flux transformer disabled, SFace and Facenet512 only, aspect-ratio-preserving Flux max_size 768, four random-search iterations per combo.
  - Driver log: `logs/pass1_driver.log`
  - Per-run logs: `logs/pass1_*.log`
  - Scan: `search/candidate_scan.json`
  - Packaged candidates: `candidates/candidate_01` through `candidates/candidate_05`
  - Result: perturbations are mostly presentable, but post-diffusion identity distances are weak so this is not enough for final mentor demos.

Candidate selection rule:

- Do not rely only on DeepFace identity distance.
- Prefer examples where `perturbed.png` remains visually close/presentable and the post-edit result visibly fails compared with `original_diffused.png`.
- Valid visible failures include prompt not applied, image destruction/garbling, structure break, identity drift, or the perturbed edit turning into a substantially different result.

Resume commands:

```bash
cd /workspace/geometric-v1
.venv-linux-gpu/bin/python /workspace/output/mentor_results_20260614_092647/scripts/mentor_search.py scan --top 12
.venv-linux-gpu/bin/python /workspace/output/mentor_results_20260614_092647/scripts/mentor_search.py package --top 5
```

If `pass1` was interrupted, inspect completed folders under `search/pass1/**/report.json`, then either rerun the whole `pass1` manifest or create a new smaller manifest for missing combos.

Next active search:

- Batch `pass2` completed successfully.
- Manifest: `configs/pass2/manifest.json`
- Strategy: relaxed visual stealth, stronger perturbation ranges, Flux transformer disabled, SFace and Facenet512, eight random-search iterations per combo.
- Scan: `logs/pass2_scan.log`
- Packaged candidates: `candidates_pass2/candidate_01` through `candidate_08`
- Visual grid: `pass2_candidates_sheet.jpg`
- Result: still mostly successful edits; not enough visible failure.

Next active search:

- Batch `pass3` completed successfully.
- Manifest: `configs/pass3/manifest.json`
- Strategy: strong perturbation bounds with lower stealth targets, Flux transformer disabled, SFace and Facenet512, sixteen random-search iterations per combo.
- Scan: `logs/pass3_scan.log`
- Packaged candidates: `candidates_pass3/candidate_01` through `candidate_10`
- Visual grid: `pass3_candidates_sheet.jpg`
- Iteration-level scan: `logs/iteration_scan.log`
- Iteration-level packaged candidates: `candidates_iterations/candidate_01` through `candidate_12`
- Iteration-level visual grid: `iteration_candidates_sheet.jpg`

Next active search:

- Batch `pass4_fluxfeat`
- Manifest: `configs/pass4_fluxfeat/manifest.json`
- Strategy: small loss3 strong search with `objective.flux_transformer.enabled=true`, strict false.
- Run command:

```bash
cd /workspace/geometric-v1
.venv-linux-gpu/bin/python /workspace/output/mentor_results_20260614_092647/scripts/mentor_search.py run --batch pass4_fluxfeat > /workspace/output/mentor_results_20260614_092647/logs/pass4_fluxfeat_driver.log 2>&1
```
