# Presentable Candidate 1

- Prompt: make the person smile
- Source: /workspace/gender_dataset/men/32.jpg
- Run: /workspace/output/mentor_results_20260614_092647/search/pass3/06_men32_smile_strong/embedding_loss_run_20260614_101233
- Iteration: 16 / restart_0_random_15
- PSNR / SSIM: 23.792493270943915 / 0.839684272806128
- Pre identity similarity: 95.8033160515912
- Post identity distance: 4.420732407440205
- Output SSIM drop: 0.1827647349457875

Visible failure type: TODO after visual inspection.

## Visual Judgment

- Failure type: quality degradation / structure change
- Rationale: Presentable input tradeoff; post-edit smile has eye/glasses and facial structure artifacts.

## Finalist Metrics

- Pre visual: PSNR 23.79 dB, SSIM 0.840
- All-model DeepFace mean match: pre 91.01%, post 89.20%
Post all-model scores: SFace 90.96%, OpenFace 87.19%, Facenet 90.71%, Facenet512 87.95%
- Failure type: quality degradation / structure change
