# Iteration Candidate 6

- Prompt: change the hairstyle
- Source: /workspace/gender_dataset/men/11.jpg
- Run: /workspace/output/mentor_results_20260614_092647/search/pass3/04_men11_hair_strong/embedding_loss_run_20260614_101039
- Iteration: 7 / restart_0_random_6
- Demo score: 31.783352405460306
- PSNR / SSIM: 16.861030458390264 / 0.692027641833404
- Pre identity similarity: 91.00066688750023
- Post identity distance: 10.741058917982457
- Output SSIM drop: 0.28938427418420665

Visible failure type: TODO after visual inspection.

## Visual Judgment

- Failure type: structure change / image degradation
- Rationale: Hair edit output is visibly skewed with background/color artifacts.

## Finalist Metrics

- Pre visual: PSNR 16.86 dB, SSIM 0.692
- All-model DeepFace mean match: pre 73.87%, post 75.51%
Post all-model scores: SFace 88.99%, OpenFace 70.89%, Facenet 91.52%, Facenet512 50.63%
- Failure type: structure change / image degradation
