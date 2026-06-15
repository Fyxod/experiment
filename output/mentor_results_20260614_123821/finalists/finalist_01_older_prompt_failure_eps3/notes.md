# finalist_01_older_prompt_failure_eps3

Failure type: prompt failure

Why it is useful: Clean Flux makes the person substantially older; the perturbed edit mostly preserves the original age/hair/face instead.

Recommendation: Best overall: very clean input and obvious edit failure.

Source: `/workspace/gender_dataset/men/57.jpg`

Prompt: `make the person older`

Input stealth:
- PSNR: 40.88 dB
- SSIM: 0.9552
- mean abs: 0.00813
- max abs: 0.01176 (3.0/255 bound)

Output disruption:
- PSNR: 11.03 dB
- SSIM: 0.5321
- L2: 0.2809
- mean abs: 0.22486

DeepFace all-model mean match:
- Pre-diffusion original vs perturbed: 99.81%
- Post-diffusion original edit vs perturbed edit: 29.37%

Settings:
- seed: 2241
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
