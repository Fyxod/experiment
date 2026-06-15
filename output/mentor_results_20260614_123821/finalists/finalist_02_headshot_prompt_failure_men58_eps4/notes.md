# finalist_02_headshot_prompt_failure_men58_eps4

Failure type: prompt failure

Why it is useful: Clean Flux turns the image into a suit/headshot; the perturbed edit largely stays in the original casual pose/composition.

Recommendation: Best headshot demo; the failure is easy to explain visually.

Source: `/workspace/gender_dataset/men/58.jpg`

Prompt: `turn the image into a professional headshot`

Input stealth:
- PSNR: 39.42 dB
- SSIM: 0.9305
- mean abs: 0.00929
- max abs: 0.01569 (4.0/255 bound)

Output disruption:
- PSNR: 9.86 dB
- SSIM: 0.5385
- L2: 0.3212
- mean abs: 0.27066

DeepFace all-model mean match:
- Pre-diffusion original vs perturbed: 99.68%
- Post-diffusion original edit vs perturbed edit: 71.62%

Settings:
- seed: 2271
- max_size: 512
- diffusion_steps: 4
- guidance_scale: 1.0
- attack: transformer
- attack_iters: 12
- epsilon: 4.0/255
- alpha: 1.0/255
- objective_timestep: 0
- highpass_sigma: 0.0
- random_start: True
