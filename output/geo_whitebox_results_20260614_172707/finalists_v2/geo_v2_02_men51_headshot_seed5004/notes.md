# geo_v2_02_men51_headshot_seed5004

Assessment: Lowest output SSIM seed for the best fixed geometry; visually it is mostly a style/identity variation rather than a clean failure.

Visible failure type: identity/style drift; prompt still succeeds

Source: `/workspace/gender_dataset/men/51.jpg`

Prompt: `turn the image into a professional headshot`

Geometry methods: `tps,piecewise,dct,rolling`

Input visual similarity:
- PSNR: 38.25 dB
- SSIM: 0.9865

Displacement:
- max magnitude: 1.072 px
- mean magnitude: 0.122 px
- p95 magnitude: 0.738 px

Post-diffusion output difference:
- PSNR: 21.96 dB
- SSIM: 0.8930
- L2: 0.0798

DeepFace all-model mean match:
- pre: 98.99%
- post: 97.89%
