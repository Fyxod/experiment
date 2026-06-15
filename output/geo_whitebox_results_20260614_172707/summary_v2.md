# Geometric White-Box Results Summary

Branch: `loss3`

Implemented methods: differentiable rolling shutter, low-res FFD grid, exact TPS interpolation, fixed-topology Delaunay/piecewise warp, DCT/spectral coordinate warp, affine similarity warp, radial/lens warp, face-local masks, method gates, hidden-feature/attention-proxy/transformer-pred/VAE/denoise-latent objectives, final-edit black-box CEM, and fixed-geometry seed sweeps. Homography was implemented but intentionally kept out of the main searches per user instruction.

## Top Candidates

### 1. geo_v2_01_men51_headshot_seed5007

- Folder: `/workspace/output/geo_whitebox_results_20260614_172707/finalists_v2/geo_v2_01_men51_headshot_seed5007`
- Prompt: `turn the image into a professional headshot`
- Source image: `/workspace/gender_dataset/men/51.jpg`
- Base geometry run: `/workspace/output/geo_whitebox_results_20260614_172707/blackbox/men51_headshot_local_seed5203_g6p12/cand_055_g05_m07`
- Seed output folder: `/workspace/output/geo_whitebox_results_20260614_172707/seed_sweeps/men51_headshot_cand055_32/seed_5007`
- Geometric method(s): `tps,piecewise,dct,rolling`
- Visible failure type: identity/outfit/style drift; prompt still mostly succeeds
- Visual assessment: Strongest geometry-only metric result. Perturbed input is clean; edited output changes suit/tie styling and face details, but it is not a full prompt failure.
- Pre-diffusion visual similarity: PSNR 38.25 dB, SSIM 0.9865
- Final displacement stats: max 1.072 px, mean 0.122 px, p95 0.738 px
- Post-diffusion output difference: PSNR 20.15 dB, SSIM 0.9140, L2 0.0983
- DeepFace all-model mean match: pre 98.99%, post 95.19%
- Key settings: seed 5203, max_disp_px 2.5, attack_iters 0, objective final_edit_blackbox, diffusion_steps 4, guidance_scale 1.0

### 2. geo_v2_02_men51_headshot_seed5004

- Folder: `/workspace/output/geo_whitebox_results_20260614_172707/finalists_v2/geo_v2_02_men51_headshot_seed5004`
- Prompt: `turn the image into a professional headshot`
- Source image: `/workspace/gender_dataset/men/51.jpg`
- Base geometry run: `/workspace/output/geo_whitebox_results_20260614_172707/blackbox/men51_headshot_local_seed5203_g6p12/cand_055_g05_m07`
- Seed output folder: `/workspace/output/geo_whitebox_results_20260614_172707/seed_sweeps/men51_headshot_cand055_32/seed_5004`
- Geometric method(s): `tps,piecewise,dct,rolling`
- Visible failure type: identity/style drift; prompt still succeeds
- Visual assessment: Lowest output SSIM seed for the best fixed geometry; visually it is mostly a style/identity variation rather than a clean failure.
- Pre-diffusion visual similarity: PSNR 38.25 dB, SSIM 0.9865
- Final displacement stats: max 1.072 px, mean 0.122 px, p95 0.738 px
- Post-diffusion output difference: PSNR 21.96 dB, SSIM 0.8930, L2 0.0798
- DeepFace all-model mean match: pre 98.99%, post 97.89%
- Key settings: seed 5203, max_disp_px 2.5, attack_iters 0, objective final_edit_blackbox, diffusion_steps 4, guidance_scale 1.0

### 3. geo_v2_03_men51_studio_seed7029

- Folder: `/workspace/output/geo_whitebox_results_20260614_172707/finalists_v2/geo_v2_03_men51_studio_seed7029`
- Prompt: `make the person look like a studio portrait`
- Source image: `/workspace/gender_dataset/men/51.jpg`
- Base geometry run: `/workspace/output/geo_whitebox_results_20260614_172707/search/geo_pass5_scaled/02_men51_studio_dct_grid_tps_piecewise_rolling_affine_radial_hybrid_2px`
- Seed output folder: `/workspace/output/geo_whitebox_results_20260614_172707/seed_sweeps/men51_studio_pass5_64/seed_7029`
- Geometric method(s): `dct,grid,tps,piecewise,rolling,affine,radial`
- Visible failure type: background/style drift; prompt still mostly succeeds
- Visual assessment: Best studio-portrait seed. Perturbed edit changes lighting/background and face style, but remains a plausible studio portrait.
- Pre-diffusion visual similarity: PSNR 39.21 dB, SSIM 0.9905
- Final displacement stats: max 0.786 px, mean 0.090 px, p95 0.529 px
- Post-diffusion output difference: PSNR 27.81 dB, SSIM 0.9136, L2 0.0407
- DeepFace all-model mean match: pre 99.44%, post 97.64%
- Key settings: seed 6202, max_disp_px 2.0, attack_iters 90, objective hybrid, diffusion_steps 4, guidance_scale 1.0

### 4. geo_v2_04_men51_studio_seed7041

- Folder: `/workspace/output/geo_whitebox_results_20260614_172707/finalists_v2/geo_v2_04_men51_studio_seed7041`
- Prompt: `make the person look like a studio portrait`
- Source image: `/workspace/gender_dataset/men/51.jpg`
- Base geometry run: `/workspace/output/geo_whitebox_results_20260614_172707/search/geo_pass5_scaled/02_men51_studio_dct_grid_tps_piecewise_rolling_affine_radial_hybrid_2px`
- Seed output folder: `/workspace/output/geo_whitebox_results_20260614_172707/seed_sweeps/men51_studio_pass5_64/seed_7041`
- Geometric method(s): `dct,grid,tps,piecewise,rolling,affine,radial`
- Visible failure type: mild structure/style drift; prompt still succeeds
- Visual assessment: Second-best studio seed. Useful as supporting evidence, not as a standalone success demo.
- Pre-diffusion visual similarity: PSNR 39.21 dB, SSIM 0.9905
- Final displacement stats: max 0.786 px, mean 0.090 px, p95 0.529 px
- Post-diffusion output difference: PSNR 27.08 dB, SSIM 0.9245, L2 0.0443
- DeepFace all-model mean match: pre 99.44%, post 98.94%
- Key settings: seed 6202, max_disp_px 2.0, attack_iters 90, objective hybrid, diffusion_steps 4, guidance_scale 1.0

### 5. geo_v2_05_men66_sunglasses_seed6010

- Folder: `/workspace/output/geo_whitebox_results_20260614_172707/finalists_v2/geo_v2_05_men66_sunglasses_seed6010`
- Prompt: `add sunglasses`
- Source image: `/workspace/gender_dataset/men/66.jpg`
- Base geometry run: `/workspace/output/geo_whitebox_results_20260614_172707/blackbox/men66_sunglasses_upper_face_g8p12/cand_030_g03_m06`
- Seed output folder: `/workspace/output/geo_whitebox_results_20260614_172707/seed_sweeps/men66_sunglasses_cand030_64/seed_6010`
- Geometric method(s): `tps,piecewise,dct,grid,rolling`
- Visible failure type: minor sunglasses/style variation; not a prompt failure
- Visual assessment: Best sunglasses geometry attempt after seed sweep. Included to document that this prompt stayed robust.
- Pre-diffusion visual similarity: PSNR 37.48 dB, SSIM 0.9872
- Final displacement stats: max 1.260 px, mean 0.205 px, p95 0.811 px
- Post-diffusion output difference: PSNR 34.08 dB, SSIM 0.9827, L2 0.0198
- DeepFace all-model mean match: pre 99.67%, post 99.53%
- Key settings: seed 7601, max_disp_px 2.2, attack_iters 0, objective final_edit_blackbox, diffusion_steps 4, guidance_scale 1.0

## Recommendation

Best 1-2 geometry-only examples to show, if required: `geo_v2_01_men51_headshot_seed5007` and `geo_v2_03_men51_studio_seed7029`. Present them as geometry-induced identity/style drift, not as fully successful prompt-prevention attacks.

## Honest Assessment

Geometry-only attacks under strict visual constraints remained weaker than the previous pixel-PGD retry. The fixed-geometry seed sweeps found the most visible output drift, but Flux.2 Klein usually still applied the requested edit. This suggests the geometry direction needs either a stronger semantic/objective signal or a less strict geometry budget to become competitive.
