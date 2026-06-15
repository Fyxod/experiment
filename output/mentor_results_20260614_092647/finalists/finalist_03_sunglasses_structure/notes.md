# Candidate 3

- Prompt: add sunglasses
- Source: /workspace/gender_dataset/men/16.jpg
- Run: /workspace/output/mentor_results_20260614_092647/search/pass3/05_men16_sunglasses_strong/embedding_loss_run_20260614_101125
- PSNR / SSIM: 22.69572644599787 / 0.7962121463633026
- Pre identity similarity: 95.42740540620244
- Post identity distance: 9.382876660984003
- Output SSIM drop: 0.20803243275966377

Visible failure type: TODO after visual inspection.

## Visual Judgment

- Failure type: structure change
- Rationale: Sunglasses prompt applies, but perturbed edit bends head/background and changes face structure.

## Finalist Metrics

- Pre visual: PSNR 22.70 dB, SSIM 0.796
- All-model DeepFace mean match: pre 84.89%, post 79.62%
Post all-model scores: SFace 78.54%, OpenFace 65.42%, Facenet 94.69%, Facenet512 79.81%
- Failure type: structure change
