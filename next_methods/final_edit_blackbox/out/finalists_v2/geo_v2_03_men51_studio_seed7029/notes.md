# geo_v2_03_men51_studio_seed7029

Assessment: Best studio-portrait seed. Perturbed edit changes lighting/background and face style, but remains a plausible studio portrait.

Visible failure type: background/style drift; prompt still mostly succeeds

Source: `/workspace/gender_dataset/men/51.jpg`

Prompt: `make the person look like a studio portrait`

Geometry methods: `dct,grid,tps,piecewise,rolling,affine,radial`

Input visual similarity:
- PSNR: 39.21 dB
- SSIM: 0.9905

Displacement:
- max magnitude: 0.786 px
- mean magnitude: 0.090 px
- p95 magnitude: 0.529 px

Post-diffusion output difference:
- PSNR: 27.81 dB
- SSIM: 0.9136
- L2: 0.0407

DeepFace all-model mean match:
- pre: 99.44%
- post: 97.64%
