# Mentor Results Summary

- Experiment root: `/workspace/output/mentor_results_20260614_092647`
- Branch used: `loss3`
- Model: `black-forest-labs/FLUX.2-klein-4B` via Diffusers `Flux2KleinPipeline`
- Search path: loss3 embedding-loss pipeline, Flux transformer disabled for passes 1-3; a small pass4 enabled Flux transformer hooks and confirmed they were available, but did not beat pass3.
- DeepFace finalist rerun: `SFace`, `OpenFace`, `Facenet`, `Facenet512` with `detector_backend=skip`, `distance_metric=cosine`, `workers=1`.
- Install note: replaced GUI `opencv-python` with `opencv-python-headless==4.8.1.78` because the server lacks `libGL.so.1`.
- Hugging Face token: not needed; FLUX.2 Klein was public and not gated.

## Recommendation

Show `finalist_01_presentable_smile_degrade` first if visual stealth is the priority. Show `finalist_02_stronger_smile_identity_quality` if the mentor wants the most obvious failure; it is stronger but the perturbation is more visible.

## Top Candidates

### 1. finalist_01_presentable_smile_degrade

- Output folder: `/workspace/output/mentor_results_20260614_092647/finalists/finalist_01_presentable_smile_degrade`
- Prompt: `make the person smile`
- Source image: `/workspace/gender_dataset/men/32.jpg`
- Visible failure type: quality degradation / structure change
- Visual judgment: Best visual-stealth tradeoff; perturbation remains the most presentable, but post-edit failure is moderate.
- Pre-diffusion visual similarity: PSNR 23.79 dB, SSIM 0.840
- Fast search identity metrics: pre similarity 95.80%, post distance 4.42%, output SSIM drop 0.183, output pixel L2 0.077
- All-model DeepFace mean match: pre 91.01%, post 89.20%
- Post all-model scores: SFace match 90.96% dist 0.1072, OpenFace match 87.19% dist 0.0256, Facenet match 90.71% dist 0.0743, Facenet512 match 87.95% dist 0.0723
- Pre all-model scores: SFace match 89.93% dist 0.1195, OpenFace match 89.32% dist 0.0214, Facenet match 92.38% dist 0.0610, Facenet512 match 92.41% dist 0.0456
- Diffusion settings: steps=4, guidance_scale=1.0, max_size=768, dtype=bfloat16, seed=7006, height=None, width=None
- Optimizer/source settings: iteration `16` label `restart_0_random_15`, configured seed `7006`, optimizer `random_search`
- Actual perturbation parameters: homography: strength=0.007565992347528475; thin-plate-spline: strength=0.002575946789071539, grid=8; delaunay: strength=0.0029563328506667795, grid=9; fft-phase: strength=3.1866768322212065, coefficients=17; elastic: strength=0.009056279101090789, sigma=7.555331748443137; rolling-shutter: strength=0.010365192963360558, rolling_frequency=2.3830872765299804, rolling_shear=0.03314602965888757, rolling_acceleration=0.026498139774616153

### 2. finalist_02_stronger_smile_identity_quality

- Output folder: `/workspace/output/mentor_results_20260614_092647/finalists/finalist_02_stronger_smile_identity_quality`
- Prompt: `make the person smile`
- Source image: `/workspace/gender_dataset/men/32.jpg`
- Visible failure type: identity drift / quality degradation
- Visual judgment: Strongest metric failure and obvious post-edit quality/identity degradation, with a visibly stronger perturbation.
- Pre-diffusion visual similarity: PSNR 19.98 dB, SSIM 0.793
- Fast search identity metrics: pre similarity 89.27%, post distance 15.64%, output SSIM drop 0.269, output pixel L2 0.115
- All-model DeepFace mean match: pre 76.14%, post 54.04%
- Post all-model scores: SFace match 65.04% dist 0.4146, OpenFace match 10.97% dist 0.1781, Facenet match 75.54% dist 0.1957, Facenet512 match 64.60% dist 0.2124
- Pre all-model scores: SFace match 72.64% dist 0.3244, OpenFace match 64.42% dist 0.0712, Facenet match 85.21% dist 0.1184, Facenet512 match 82.31% dist 0.1061
- Diffusion settings: steps=4, guidance_scale=1.0, max_size=768, dtype=bfloat16, seed=7006, height=None, width=None
- Optimizer/source settings: iteration `11` label `restart_0_random_10`, configured seed `7006`, optimizer `random_search`
- Actual perturbation parameters: homography: strength=0.01671603903666078; thin-plate-spline: strength=0.00769569060662355, grid=6; delaunay: strength=0.01783081406966942, grid=9; fft-phase: strength=3.9344267270395576, coefficients=16; elastic: strength=0.0012339668381202043, sigma=6.194602195557136; rolling-shutter: strength=0.015623661871733491, rolling_frequency=3.197090154258366, rolling_shear=0.0776793761995335, rolling_acceleration=-0.03272074436430733

### 3. finalist_03_sunglasses_structure

- Output folder: `/workspace/output/mentor_results_20260614_092647/finalists/finalist_03_sunglasses_structure`
- Prompt: `add sunglasses`
- Source image: `/workspace/gender_dataset/men/16.jpg`
- Visible failure type: structure change
- Visual judgment: Good backup for sunglasses: prompt applies, but perturbed edit changes head/background structure.
- Pre-diffusion visual similarity: PSNR 22.70 dB, SSIM 0.796
- Fast search identity metrics: pre similarity 95.43%, post distance 9.38%, output SSIM drop 0.208, output pixel L2 0.087
- All-model DeepFace mean match: pre 84.89%, post 79.62%
- Post all-model scores: SFace match 78.54% dist 0.2545, OpenFace match 65.42% dist 0.0692, Facenet match 94.69% dist 0.0425, Facenet512 match 79.81% dist 0.1211
- Pre all-model scores: SFace match 89.75% dist 0.1216, OpenFace match 79.47% dist 0.0411, Facenet match 80.54% dist 0.1557, Facenet512 match 89.80% dist 0.0612
- Diffusion settings: steps=4, guidance_scale=1.0, max_size=768, dtype=bfloat16, seed=7005, height=None, width=None
- Optimizer/source settings: iteration `5` label `restart_0_random_4`, configured seed `7005`, optimizer `random_search`
- Actual perturbation parameters: homography: strength=0.017244868904112767; thin-plate-spline: strength=0.006208377293789755, grid=9; delaunay: strength=0.005935361398548821, grid=10; fft-phase: strength=4.460433915005045, coefficients=11; elastic: strength=0.014318486848901777, sigma=8.188745623176468; rolling-shutter: strength=0.013709428501258912, rolling_frequency=1.1746684767549467, rolling_shear=0.008806568920020234, rolling_acceleration=0.033112203638551205

### 4. finalist_04_older_color_degrade

- Output folder: `/workspace/output/mentor_results_20260614_092647/finalists/finalist_04_older_color_degrade`
- Prompt: `make the person older`
- Source image: `/workspace/geometric-v1/samples/image.png`
- Visible failure type: image quality degradation
- Visual judgment: Backup for color/quality degradation after older prompt.
- Pre-diffusion visual similarity: PSNR 19.47 dB, SSIM 0.747
- Fast search identity metrics: pre similarity 93.17%, post distance 7.12%, output SSIM drop 0.237, output pixel L2 0.093
- All-model DeepFace mean match: pre 70.43%, post 82.38%
- Post all-model scores: SFace match 81.49% dist 0.2195, OpenFace match 61.08% dist 0.0778, Facenet match 97.70% dist 0.0184, Facenet512 match 89.23% dist 0.0646
- Pre all-model scores: SFace match 84.36% dist 0.1855, OpenFace match 18.49% dist 0.1630, Facenet match 92.90% dist 0.0568, Facenet512 match 85.97% dist 0.0842
- Diffusion settings: steps=4, guidance_scale=1.0, max_size=768, dtype=bfloat16, seed=7003, height=None, width=None
- Optimizer/source settings: iteration `5` label `restart_0_random_4`, configured seed `7003`, optimizer `random_search`
- Actual perturbation parameters: homography: strength=0.01494490122961296; thin-plate-spline: strength=0.00892301172993138, grid=8; delaunay: strength=0.0017179552169534547, grid=10; fft-phase: strength=4.138085750832363, coefficients=18; elastic: strength=0.009929174217141897, sigma=13.438601117942202; rolling-shutter: strength=0.012639847988083142, rolling_frequency=2.0960982856381825, rolling_shear=0.07895711361727957, rolling_acceleration=-0.0412824268503903

### 5. finalist_05_hair_structure_break

- Output folder: `/workspace/output/mentor_results_20260614_092647/finalists/finalist_05_hair_structure_break`
- Prompt: `change the hairstyle`
- Source image: `/workspace/gender_dataset/men/11.jpg`
- Visible failure type: structure change / image degradation
- Visual judgment: Backup for hair/structure break, but pre-model identity metrics are mixed.
- Pre-diffusion visual similarity: PSNR 16.86 dB, SSIM 0.692
- Fast search identity metrics: pre similarity 91.00%, post distance 10.74%, output SSIM drop 0.289, output pixel L2 0.139
- All-model DeepFace mean match: pre 73.87%, post 75.51%
- Post all-model scores: SFace match 88.99% dist 0.1306, OpenFace match 70.89% dist 0.0582, Facenet match 91.52% dist 0.0678, Facenet512 match 50.63% dist 0.2962
- Pre all-model scores: SFace match 90.04% dist 0.1181, OpenFace match 54.81% dist 0.0904, Facenet match 91.02% dist 0.0718, Facenet512 match 59.59% dist 0.2425
- Diffusion settings: steps=4, guidance_scale=1.0, max_size=768, dtype=bfloat16, seed=7004, height=None, width=None
- Optimizer/source settings: iteration `7` label `restart_0_random_6`, configured seed `7004`, optimizer `random_search`
- Actual perturbation parameters: homography: strength=0.021103669252677718; thin-plate-spline: strength=0.007947375404673768, grid=7; delaunay: strength=0.0031573458436512315, grid=12; fft-phase: strength=1.591529529858463, coefficients=4; elastic: strength=0.0037104619831439603, sigma=6.857476079328323; rolling-shutter: strength=0.01561313116516744, rolling_frequency=2.7056326727781963, rolling_shear=0.0047570735095135855, rolling_acceleration=-0.04038424573508203

## Other Saved Artifacts

- Visual sheets: `finalists_sheet.jpg`, `presentable_candidates_sheet.jpg`, `iteration_candidates_sheet.jpg`, `pass3_candidates_sheet.jpg`, `pass2_candidates_sheet.jpg`.
- Candidate scans: `search/candidate_scan.json`, `search/iteration_candidate_scan.json`.
- Logs: `logs/`, including install, smoke tests, pass drivers, and `final_deepface_all.log`.
- Continuation checkpoint: `continuation.md`.

## Caveat

Flux remained robust. The best examples are degradation/structure-change demos rather than clean, invisible perturbations that completely block the prompt. The strongest failure has a more visible perturbation; the most presentable one has a weaker metric gap.
