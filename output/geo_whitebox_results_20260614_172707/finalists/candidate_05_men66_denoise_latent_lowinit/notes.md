# candidate_05_men66_denoise_latent_lowinit

Assessment: Cleanest denoise-latent trajectory attempt; edit still succeeds.

Visible failure type: no meaningful failure

Source: `/workspace/gender_dataset/men/66.jpg`

Prompt: `add sunglasses`

Geometry methods: `dct`

Objective: `denoise_latent`

Input visual similarity:
- PSNR: 46.23 dB
- SSIM: 0.9980

Displacement:
- max magnitude: 0.398 px
- mean magnitude: 0.115 px
- p95 magnitude: 0.240 px

Output difference:
- PSNR: 42.16 dB
- SSIM: 0.9950
- L2: 0.0078

DeepFace all-model mean match:
- pre: 99.97%
- post: 99.88%

Settings:
- seed: 5202
- max_size: 512
- diffusion_steps: 4
- guidance_scale: 1.0
- max_disp_px: 2.0
- attack_iters: 12
- lr: 0.0005
- padding_mode: reflection
- edge_falloff_px: 16.0
- objective_timesteps: [0]
