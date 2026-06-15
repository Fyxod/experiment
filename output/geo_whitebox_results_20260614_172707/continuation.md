# Continuation Notes

Root: `/workspace/output/geo_whitebox_results_20260614_172707`

Repo: `/workspace/geometric-v1`, branch `loss3`.

Python env: `/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11`.

GPU at start: H100 MIG 3g.40gb, idle.

Goal: white-box differentiable geometric attacks for Flux.2 Klein. Do not overwrite pixel-PGD retry results.

Current status:
- Created experiment root and logs/scripts/finalists directories.
- Read `Agents.md` for `loss3`.
- Implemented and compiled experiment-local `scripts/geo_whitebox_attack.py`.
- Supported methods: `rolling`, `grid`, `dct`, `rbf`/`tps` approximation, and combinations.
- Supported objectives: `vae`, `transformer_pred`, `hybrid`.
- Smoke `smoke/dct_men57_older_hybrid_1px_i8` completed successfully.
  - Input PSNR 40.88 dB, SSIM 0.9885.
  - Displacement max 1.00 px, mean 0.33 px.
  - Flux output barely changed, so the attack implementation works but 1 px DCT/8 iters is too weak.
- Added edge falloff masking to reduce visible frame warping and keep reported per-method fields masked.
- Stronger `grid,dct` 2 px run became non-finite and produced an invalid/ugly image. Rejected.
- Patched `geo_whitebox_attack.py` with finite checks and gradient clipping.
- Rerun `grid,dct` 1.5 px stayed visually clean but stopped after iter 2; output barely changed. Rejected.
- Patched parameter projection and best-selection by regularized utility.
- Next: test VAE-only geometry objective for stability.
- VAE-only `grid,dct` 2 px run completed but output barely changed. Geometry was visually acceptable; not a finalist.
- Added `scripts/run_geo_batch.py` to keep Flux loaded once and sweep method/objective/prompt combinations.
- `geo_pass1` completed. Inputs are generally clean but no demo-worthy failures; best output SSIM remains about 0.98-0.99.
- Next: run more aggressive `geo_pass2` with 2.5 px bounds and transformer-pred/hybrid objectives.
- `geo_pass2` completed. Still no strict, demo-worthy failures; best valid examples have output SSIM around 0.98-0.99.
- Added multi-timestep prediction objective support via `--objective-timesteps`.
- Next: smoke test multi-timestep `0,1` on `men66` sunglasses.
- Multi-timestep `0,1` on `men66` increased difference only slightly, made input visibly warped, and still did not block sunglasses. Rejected.
- Added `denoise_latent` objective that differentiates through the full 4-step denoising latent trajectory.
- Next: 4-iteration memory smoke for `denoise_latent`.
- `denoise_latent` scheduler bug patched and smoke tested.
- Conservative denoise-latent run was very clean but ineffective.
- Packaged top five geometry-only attempts under `finalists/`.
- Created final summary: `/workspace/output/geo_whitebox_results_20260614_172707/summary.md`.
- Created final sheet: `/workspace/output/geo_whitebox_results_20260614_172707/finalists_sheet.jpg`.
- Honest conclusion: geometry-only white-box attacks were not competitive with previous pixel-PGD retry results under strict visual gates.
- User rejected these as not presentable/effective and asked whether all side-chat methods were tried. Honest answer: no.
- Continuing in this same root without overwriting prior results.
- Next additions: exact TPS interpolation, fixed-topology piecewise/Delaunay-style warp, face/local mask weighting, explicit method gates, and a geometry-only black-box final-edit search over low-dimensional warp parameters.
- Available packages checked: `cv2`, `skimage`, `scipy`, `deepface`, `torch`, and `diffusers` are installed; `mediapipe`, `dlib`, and `face_recognition` are not installed. Use OpenCV/ellipse face masks first so landmark setup does not block progress.
- Implemented script changes in progress: exact TPS interpolation, fixed-topology Delaunay/piecewise barycentric warp, homography/projective corner warp, OpenCV/ellipse face masks, method gates, hidden-feature objective, and attention-output proxy objective.
- Next: compile smoke and then run bounded searches using these methods.
- User asked to keep homography/projective last priority because it visibly degrades images without helping. Keep it implemented only for completeness; exclude it from normal search runs.
- Compile passed for `geo_whitebox_attack.py` and `geo_blackbox_final_edit.py`.
- New-method smoke completed: `smoke_extra/tps_piecewise_face_hidden_i2`.
  - Methods: exact TPS + fixed-topology piecewise/Delaunay + DCT + rolling, face mask, gates, hidden-feature objective.
  - Input PSNR 39.03 dB, SSIM 0.991; max displacement 0.66 px, mean 0.20 px.
  - Flux still applied sunglasses successfully, so broader/stronger search is needed.
- Added `geo_pass3` jobs to `run_geo_batch.py`; these prioritize TPS, piecewise/Delaunay, FFD grid, DCT, rolling, face-local masks, gates, hidden-feature/attention/denoise objectives. Homography excluded.
- First black-box random CEM run `blackbox/men51_headshot_tps_piecewise_dct_rolling_face_g5p12` produced 0/60 valid visual-gate candidates. Sampling was too aggressive and saturated displacement at 2.5 px.
- Patched `geo_blackbox_final_edit.py` with `--init-theta` and `--eval-initial` so final-edit CEM can search locally around a clean white-box geometry state.
- `blackbox/men51_headshot_local_seed5203_g6p12` found the strongest metrics so far: top candidate `cand_055_g05_m07`, input PSNR 38.25 dB / SSIM 0.987, max disp 1.07 px / mean 0.12 px, output SSIM 0.925.
- Visual judgment: perturbation is presentable, but Flux still mostly succeeds at professional-headshot edit; keep as candidate/reference, not enough alone.
- Next: final-edit black-box search on visually obvious prompts, starting with `men66` + `add sunglasses` and upper-face local geometry.
- `men66_sunglasses_upper_face_g8p12` completed weakly: top valid output SSIM about 0.985, not a visible edit failure.
- `men74_smile_mouth_g8p12` is running and currently weak.
- Added `geo_pass4`: higher-frequency spectral coordinate warp (`dct_size=12`) plus finer FFD grid (`grid_size=10`) with TPS/piecewise controls and strict max displacement. Homography still excluded.
- Parallel `geo_pass4` failed with PyTorch/NVML allocator assertions while another Flux job was active. These results are invalid and should be ignored.
- GPU is clear again. Rerun as `geo_pass4_clean` in a fresh output batch with no parallel heavy job.
- `geo_pass4_clean` completed but did not improve: best was `men11/headshot` output SSIM 0.980, and several hidden/denoise jobs selected zero warp.
- Patched white-box optimizer with `objective_scale` to let Flux-internal objectives compete with visual/smooth penalties.
- Added geometric methods `affine` and `radial`/lens distortion for non-homography global/local coordinate transforms. Homography remains excluded.
- Added `geo_pass5_scaled`: scaled objectives, lower regularization, DCT/grid/TPS/piecewise/rolling/affine/radial gated combinations.
- `geo_pass5_scaled` completed. Best scaled result was `men51/studio` output SSIM 0.966, but visually Flux still applies the edit. Not stronger than local CEM.
- Added `seed_sweep_candidate.py` to keep a fixed geometry perturbation and search diffusion seeds for cases where clean edit succeeds but perturbed edit drifts/under-edits. This does not change the perturbation.
- Seed sweeps completed:
  - `men51_headshot_cand055_32`: best metric seeds 5007/5004, output SSIM 0.914/0.893; visually mostly identity/outfit/style drift, prompt still succeeds.
  - `men66_sunglasses_cand030_64`: weak; top output SSIM 0.983, sunglasses still apply.
  - `men51_studio_pass5_64`: top output SSIM 0.914, visually background/style drift but prompt still mostly succeeds.
  - `men51_hair_pass3_64`: weak; top output SSIM 0.991.
- Added `package_geo_results_v2.py` to package new finalists under `finalists_v2/`, run DeepFace all-model checks, and write updated `summary.md`/`summary_v2.md`.
