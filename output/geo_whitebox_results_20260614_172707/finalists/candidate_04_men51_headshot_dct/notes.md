# candidate_04_men51_headshot_dct

Assessment: Clean DCT-only headshot attempt; edit still succeeds with minor face/style drift.

Visible failure type: minor output variation

Source: `/workspace/gender_dataset/men/51.jpg`

Prompt: `turn the image into a professional headshot`

Geometry methods: `dct`

Objective: `hybrid`

Input visual similarity:
- PSNR: 36.99 dB
- SSIM: 0.9780

Displacement:
- max magnitude: 0.983 px
- mean magnitude: 0.320 px
- p95 magnitude: 0.672 px

Output difference:
- PSNR: 34.89 dB
- SSIM: 0.9843
- L2: 0.0180

DeepFace all-model mean match:
- pre: 99.49%
- post: 98.87%

Settings:
- seed: 4007
- max_size: 512
- diffusion_steps: 4
- guidance_scale: 1.0
- max_disp_px: 1.5
- attack_iters: 35
- lr: 0.012
- padding_mode: reflection
- edge_falloff_px: 16.0
- objective_timesteps: None
