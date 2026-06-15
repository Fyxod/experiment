# candidate_01_men51_headshot_rolling_dct

Assessment: Best geometry-only headshot attempt; output differs slightly but prompt still succeeds.

Visible failure type: weak identity/style drift, not a strong failure

Source: `/workspace/gender_dataset/men/51.jpg`

Prompt: `turn the image into a professional headshot`

Geometry methods: `rolling,dct`

Objective: `hybrid`

Input visual similarity:
- PSNR: 37.21 dB
- SSIM: 0.9787

Displacement:
- max magnitude: 0.751 px
- mean magnitude: 0.295 px
- p95 magnitude: 0.570 px

Output difference:
- PSNR: 35.67 dB
- SSIM: 0.9832
- L2: 0.0165

DeepFace all-model mean match:
- pre: 99.73%
- post: 99.39%

Settings:
- seed: 4105
- max_size: 512
- diffusion_steps: 4
- guidance_scale: 1.0
- max_disp_px: 2.5
- attack_iters: 50
- lr: 0.006
- padding_mode: reflection
- edge_falloff_px: 16.0
- objective_timesteps: None
