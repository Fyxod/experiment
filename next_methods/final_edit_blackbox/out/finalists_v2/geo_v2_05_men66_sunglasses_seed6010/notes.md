# geo_v2_05_men66_sunglasses_seed6010

Assessment: Best sunglasses geometry attempt after seed sweep. Included to document that this prompt stayed robust.

Visible failure type: minor sunglasses/style variation; not a prompt failure

Source: `/workspace/gender_dataset/men/66.jpg`

Prompt: `add sunglasses`

Geometry methods: `tps,piecewise,dct,grid,rolling`

Input visual similarity:
- PSNR: 37.48 dB
- SSIM: 0.9872

Displacement:
- max magnitude: 1.260 px
- mean magnitude: 0.205 px
- p95 magnitude: 0.811 px

Post-diffusion output difference:
- PSNR: 34.08 dB
- SSIM: 0.9827
- L2: 0.0198

DeepFace all-model mean match:
- pre: 99.67%
- post: 99.53%
