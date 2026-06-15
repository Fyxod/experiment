# Continuation Notes

Current retry root: `/workspace/output/mentor_results_20260614_123821`

Repo: `/workspace/geometric-v1`, branch `loss3`.

Python env: `/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11`.

Why previous results were rejected: the original geometric-warp finalists had visibly garbled perturbed inputs.

New method used: experiment-local stealth pixel-space PGD against the Flux.2 Klein conditioning path, implemented in:
- `/workspace/output/mentor_results_20260614_123821/scripts/stealth_flux_attack.py`
- `/workspace/output/mentor_results_20260614_123821/scripts/run_stealth_batch.py`
- `/workspace/output/mentor_results_20260614_123821/scripts/package_finalists.py`

Useful logs:
- `/workspace/output/mentor_results_20260614_123821/logs/smoke_fixed_men66_sunglasses_eps4_i8.log`
- `/workspace/output/mentor_results_20260614_123821/logs/stealth_pass1.log`
- `/workspace/output/mentor_results_20260614_123821/logs/stealth_pass2.log`
- `/workspace/output/mentor_results_20260614_123821/logs/package_finalists.log`

Main outputs:
- Summary: `/workspace/output/mentor_results_20260614_123821/summary.md`
- Combined finalist sheet: `/workspace/output/mentor_results_20260614_123821/finalists_sheet_retry.jpg`
- Finalists root: `/workspace/output/mentor_results_20260614_123821/finalists`

Best examples:
1. `finalist_01_older_prompt_failure_eps3`: clean input, clean edit makes person much older, perturbed edit mostly fails aging.
2. `finalist_02_headshot_prompt_failure_men58_eps4`: clean edit makes suit headshot, perturbed edit stays close to casual original.
3. `finalist_03_headshot_prompt_failure_men51_eps3`: clean edit makes suit headshot, perturbed edit keeps plaid/source composition.
4. `finalist_04_sunglasses_prompt_failure_eps3`: clean edit adds sunglasses, perturbed edit mostly keeps regular glasses.
5. `finalist_05_headshot_identity_drift_men11_eps3`: backup identity-drift example.

Do not rerun the old geometric-warp search unless specifically needed; it does not satisfy the user's visual stealth requirement.
