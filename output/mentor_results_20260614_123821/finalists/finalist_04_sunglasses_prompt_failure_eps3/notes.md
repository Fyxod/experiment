# finalist_04_sunglasses_prompt_failure_eps3

Failure type: prompt failure

Why it is useful: Clean Flux replaces regular glasses with dark sunglasses; the perturbed edit mostly keeps the original glasses.

Recommendation: Best sunglasses example.

Source: `/workspace/gender_dataset/men/66.jpg`

Prompt: `add sunglasses`

Input stealth:
- PSNR: 40.63 dB
- SSIM: 0.9575
- mean abs: 0.00840
- max abs: 0.01176 (3.0/255 bound)

Output disruption:
- PSNR: 17.83 dB
- SSIM: 0.8254
- L2: 0.1284
- mean abs: 0.04561

DeepFace all-model mean match:
- Pre-diffusion original vs perturbed: 99.89%
- Post-diffusion original edit vs perturbed edit: 74.78%

Settings:
- seed: 1234
- max_size: 512
- diffusion_steps: 4
- guidance_scale: 1.0
- attack: transformer
- attack_iters: 8
- epsilon: 3.0/255
- alpha: 1.0/255
- objective_timestep: 0
- highpass_sigma: 0.0
- random_start: True
