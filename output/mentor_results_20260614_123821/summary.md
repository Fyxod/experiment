# Mentor Results Retry Summary

Branch: `loss3`

Combined visual sheet: `/workspace/output/mentor_results_20260614_123821/finalists_sheet_retry.jpg`

Method: stealth pixel-space PGD against the Flux.2 Klein conditioning path. This replaced the earlier geometric-warp search because those perturbations were too visible.

External method inspiration used while redesigning the search:
- PhotoGuard / adversarial perturbations for image editing: https://www.eecs.mit.edu/using-ai-to-protect-against-ai-image-manipulation/
- EditShield latent-distribution protection idea: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07991.pdf
- Low-frequency/DCT attack context: https://proceedings.mlr.press/v115/guo20a.html

## Top Candidates

### 1. finalist_01_older_prompt_failure_eps3

- Folder: `/workspace/output/mentor_results_20260614_123821/finalists/finalist_01_older_prompt_failure_eps3`
- Source: `/workspace/gender_dataset/men/57.jpg`
- Prompt: `make the person older`
- Visible failure type: prompt failure
- Visual judgment: Clean Flux makes the person substantially older; the perturbed edit mostly preserves the original age/hair/face instead.
- Pre-diffusion visual similarity: PSNR 40.88 dB, SSIM 0.9552, mean abs 0.00813, max abs 0.01176
- Post-diffusion output difference: PSNR 11.03 dB, SSIM 0.5321, L2 0.2809, mean abs 0.22486
- DeepFace all-model mean match: pre 99.81%, post 29.37%
- Settings: seed 2241, max_size 512, diffusion_steps 4, guidance 1.0, attack transformer, iters 12, epsilon 3.0/255, alpha 1.0/255, objective_timestep 0, highpass_sigma 0.0, random_start True

### 2. finalist_02_headshot_prompt_failure_men58_eps4

- Folder: `/workspace/output/mentor_results_20260614_123821/finalists/finalist_02_headshot_prompt_failure_men58_eps4`
- Source: `/workspace/gender_dataset/men/58.jpg`
- Prompt: `turn the image into a professional headshot`
- Visible failure type: prompt failure
- Visual judgment: Clean Flux turns the image into a suit/headshot; the perturbed edit largely stays in the original casual pose/composition.
- Pre-diffusion visual similarity: PSNR 39.42 dB, SSIM 0.9305, mean abs 0.00929, max abs 0.01569
- Post-diffusion output difference: PSNR 9.86 dB, SSIM 0.5385, L2 0.3212, mean abs 0.27066
- DeepFace all-model mean match: pre 99.68%, post 71.62%
- Settings: seed 2271, max_size 512, diffusion_steps 4, guidance 1.0, attack transformer, iters 12, epsilon 4.0/255, alpha 1.0/255, objective_timestep 0, highpass_sigma 0.0, random_start True

### 3. finalist_03_headshot_prompt_failure_men51_eps3

- Folder: `/workspace/output/mentor_results_20260614_123821/finalists/finalist_03_headshot_prompt_failure_men51_eps3`
- Source: `/workspace/gender_dataset/men/51.jpg`
- Prompt: `turn the image into a professional headshot`
- Visible failure type: prompt failure
- Visual judgment: Clean Flux produces a professional suit headshot; the perturbed edit keeps the plaid shirt and original composition.
- Pre-diffusion visual similarity: PSNR 41.12 dB, SSIM 0.9594, mean abs 0.00783, max abs 0.01176
- Post-diffusion output difference: PSNR 13.05 dB, SSIM 0.6123, L2 0.2225, mean abs 0.16716
- DeepFace all-model mean match: pre 99.74%, post 59.17%
- Settings: seed 2280, max_size 512, diffusion_steps 4, guidance 1.0, attack transformer, iters 12, epsilon 3.0/255, alpha 1.0/255, objective_timestep 0, highpass_sigma 0.0, random_start True

### 4. finalist_04_sunglasses_prompt_failure_eps3

- Folder: `/workspace/output/mentor_results_20260614_123821/finalists/finalist_04_sunglasses_prompt_failure_eps3`
- Source: `/workspace/gender_dataset/men/66.jpg`
- Prompt: `add sunglasses`
- Visible failure type: prompt failure
- Visual judgment: Clean Flux replaces regular glasses with dark sunglasses; the perturbed edit mostly keeps the original glasses.
- Pre-diffusion visual similarity: PSNR 40.63 dB, SSIM 0.9575, mean abs 0.00840, max abs 0.01176
- Post-diffusion output difference: PSNR 17.83 dB, SSIM 0.8254, L2 0.1284, mean abs 0.04561
- DeepFace all-model mean match: pre 99.89%, post 74.78%
- Settings: seed 1234, max_size 512, diffusion_steps 4, guidance 1.0, attack transformer, iters 8, epsilon 3.0/255, alpha 1.0/255, objective_timestep 0, highpass_sigma 0.0, random_start True

### 5. finalist_05_headshot_identity_drift_men11_eps3

- Folder: `/workspace/output/mentor_results_20260614_123821/finalists/finalist_05_headshot_identity_drift_men11_eps3`
- Source: `/workspace/gender_dataset/men/11.jpg`
- Prompt: `turn the image into a professional headshot`
- Visible failure type: identity drift
- Visual judgment: Both outputs are headshots, but the perturbed edit changes face shape, hair, and perceived age more than the clean edit.
- Pre-diffusion visual similarity: PSNR 41.13 dB, SSIM 0.9584, mean abs 0.00783, max abs 0.01176
- Post-diffusion output difference: PSNR 11.88 dB, SSIM 0.5879, L2 0.2548, mean abs 0.15400
- DeepFace all-model mean match: pre 99.81%, post 57.22%
- Settings: seed 2260, max_size 512, diffusion_steps 4, guidance 1.0, attack transformer, iters 12, epsilon 3.0/255, alpha 1.0/255, objective_timestep 0, highpass_sigma 0.0, random_start True

## Recommendation

Show `finalist_01_older_prompt_failure_eps3` first: it has a clean input and the most obvious requested-edit failure.

Show `finalist_02_headshot_prompt_failure_men58_eps4` or `finalist_03_headshot_prompt_failure_men51_eps3` second: both demonstrate that a nearly invisible perturbation can stop the professional-headshot transformation.

`finalist_04_sunglasses_prompt_failure_eps3` is the best compact prompt-specific example: regular glasses remain instead of becoming sunglasses.
