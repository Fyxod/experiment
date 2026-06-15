# geo_v2_01_men51_headshot_seed5007

Assessment: Strongest geometry-only metric result. Perturbed input is clean; edited output changes suit/tie styling and face details, but it is not a full prompt failure.

Visible failure type: identity/outfit/style drift; prompt still mostly succeeds

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
- PSNR: 20.15 dB
- SSIM: 0.9140
- L2: 0.0983

DeepFace all-model mean match:
- pre: 98.99%
- post: 95.19%
