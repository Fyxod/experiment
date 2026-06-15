# finalist_03_headshot_prompt_failure_men51_eps3

Failure type: prompt failure

Why it is useful: Clean Flux produces a professional suit headshot; the perturbed edit keeps the plaid shirt and original composition.

Recommendation: Very clean input and intuitive before/after story.

Source: `/workspace/gender_dataset/men/51.jpg`

Prompt: `turn the image into a professional headshot`

Input stealth:
- PSNR: 41.12 dB
- SSIM: 0.9594
- mean abs: 0.00783
- max abs: 0.01176 (3.0/255 bound)

Output disruption:
- PSNR: 13.05 dB
- SSIM: 0.6123
- L2: 0.2225
- mean abs: 0.16716

DeepFace all-model mean match:
- Pre-diffusion original vs perturbed: 99.74%
- Post-diffusion original edit vs perturbed edit: 59.17%

Settings:
- seed: 2280
- max_size: 512
- diffusion_steps: 4
- guidance_scale: 1.0
- attack: transformer
- attack_iters: 12
- epsilon: 3.0/255
- alpha: 1.0/255
- objective_timestep: 0
- highpass_sigma: 0.0
- random_start: True
