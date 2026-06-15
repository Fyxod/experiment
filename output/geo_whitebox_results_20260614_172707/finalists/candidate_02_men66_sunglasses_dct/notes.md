# candidate_02_men66_sunglasses_dct

Assessment: Best sunglasses geometry attempt; sunglasses still apply, with only small shape/style variation.

Visible failure type: minor output variation, not prompt failure

Source: `/workspace/gender_dataset/men/66.jpg`

Prompt: `add sunglasses`

Geometry methods: `dct`

Objective: `hybrid`

Input visual similarity:
- PSNR: 36.00 dB
- SSIM: 0.9824

Displacement:
- max magnitude: 1.092 px
- mean magnitude: 0.346 px
- p95 magnitude: 0.635 px

Output difference:
- PSNR: 35.94 dB
- SSIM: 0.9800
- L2: 0.0160

DeepFace all-model mean match:
- pre: 99.39%
- post: 99.43%

Settings:
- seed: 4009
- max_size: 512
- diffusion_steps: 4
- guidance_scale: 1.0
- max_disp_px: 1.5
- attack_iters: 35
- lr: 0.012
- padding_mode: reflection
- edge_falloff_px: 16.0
- objective_timesteps: None
