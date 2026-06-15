# Candidate 4

- Prompt: make the person older
- Source: /workspace/geometric-v1/samples/image.png
- Run: /workspace/output/mentor_results_20260614_092647/search/pass3/03_sample_older_strong/embedding_loss_run_20260614_100941
- PSNR / SSIM: 19.46596514742238 / 0.746974845575706
- Pre identity similarity: 93.1720789127459
- Post identity distance: 7.120576178517759
- Output SSIM drop: 0.23651448076561332

Visible failure type: TODO after visual inspection.

## Visual Judgment

- Failure type: image quality degradation
- Rationale: Older prompt applies, but perturbed edit carries color wash/artifacts and visible quality loss.

## Finalist Metrics

- Pre visual: PSNR 19.47 dB, SSIM 0.747
- All-model DeepFace mean match: pre 70.43%, post 82.38%
Post all-model scores: SFace 81.49%, OpenFace 61.08%, Facenet 97.70%, Facenet512 89.23%
- Failure type: image quality degradation
