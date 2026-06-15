# Iteration Candidate 2

- Prompt: make the person smile
- Source: /workspace/gender_dataset/men/32.jpg
- Run: /workspace/output/mentor_results_20260614_092647/search/pass3/06_men32_smile_strong/embedding_loss_run_20260614_101233
- Iteration: 11 / restart_0_random_10
- Demo score: 36.135075647443706
- PSNR / SSIM: 19.98337313150854 / 0.7931139753000016
- Pre identity similarity: 89.26868849289119
- Post identity distance: 15.637293551720305
- Output SSIM drop: 0.2693061650950287

Visible failure type: TODO after visual inspection.

## Visual Judgment

- Failure type: identity drift / quality degradation
- Rationale: Stronger perturbation; post-edit face and glasses are visibly more distorted than original edit.

## Finalist Metrics

- Pre visual: PSNR 19.98 dB, SSIM 0.793
- All-model DeepFace mean match: pre 76.14%, post 54.04%
Post all-model scores: SFace 65.04%, OpenFace 10.97%, Facenet 75.54%, Facenet512 64.60%
- Failure type: identity drift / quality degradation
