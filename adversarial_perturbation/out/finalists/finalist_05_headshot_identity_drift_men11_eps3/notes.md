# finalist_05_headshot_identity_drift_men11_eps3

Failure type: identity drift

Why it is useful: Both outputs are headshots, but the perturbed edit changes face shape, hair, and perceived age more than the clean edit.

Recommendation: Useful backup for identity-drift discussion.

Source: `/workspace/gender_dataset/men/11.jpg`

Prompt: `turn the image into a professional headshot`

Input stealth:
- PSNR: 41.13 dB
- SSIM: 0.9584
- mean abs: 0.00783
- max abs: 0.01176 (3.0/255 bound)

Output disruption:
- PSNR: 11.88 dB
- SSIM: 0.5879
- L2: 0.2548
- mean abs: 0.15400

DeepFace all-model mean match:
- Pre-diffusion original vs perturbed: 99.81%
- Post-diffusion original edit vs perturbed edit: 57.22%

Settings:
- seed: 2260
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
