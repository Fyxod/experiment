# candidate_03_men57_older_grid_dct

Assessment: Best clean older-prompt geometry attempt; aging still succeeds.

Visible failure type: minor output variation, not prompt failure

Source: `/workspace/gender_dataset/men/57.jpg`

Prompt: `make the person older`

Geometry methods: `grid,dct`

Objective: `vae`

Input visual similarity:
- PSNR: 39.47 dB
- SSIM: 0.9842

Displacement:
- max magnitude: 0.898 px
- mean magnitude: 0.388 px
- p95 magnitude: 0.655 px

Output difference:
- PSNR: 35.61 dB
- SSIM: 0.9901
- L2: 0.0166

DeepFace all-model mean match:
- pre: 99.05%
- post: 99.40%

Settings:
- seed: 4003
- max_size: 512
- diffusion_steps: 4
- guidance_scale: 1.0
- max_disp_px: 2.0
- attack_iters: 35
- lr: 0.01
- padding_mode: reflection
- edge_falloff_px: 16.0
- objective_timesteps: None
