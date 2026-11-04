# Results log — EMG Ninapro

**Track:** `emg_ninapro` · **Split unit:** subject / repetition · **Primary metric:** macro-F1 · **Evaluation modes:** within-subject + new-subject

## Iteration log

| # | Date | What changed & why (one line) | Primary metric **with spread** | Better than previous? | If not — why it was kept | Commit |
|---|---|---|---|---|---|---|
| 1 | 2026-10-04 | Supplied baseline on real S1–S5, E1; establish the baseline. Synthetic smoke test passed. | Within-subject: mean 0.790 (SD 0.054, range 0.743–0.880). New-subject: mean 0.161 (SD 0.043, range 0.097–0.210). Across 5 subjects. | — (baseline) | — | Base `d910d38`; changes uncommitted |

## Decision log — the choices behind the numbers

| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| Data loading | DB1, E1, S1–S5 | All 27 subjects | Keep the supplied subset for the baseline. | 1 | — |
| Classification, incl. imbalance | Supplied random forest; seed 0; `imbalance="balanced"` | `"none"` | Keep the default weighting for less frequent gestures. | 1 | — |
| Feature selection | none (all 50 features) | anova k=25, lasso C=0.1 | Neither improved either mode: anova lowered new-subject 0.161 → 0.132 (larger spread), lasso kept 46–50 features and matched none exactly. | 1 | — |

**Reproduce:** in Colab, restart and run the notebook. Locally, set `USE_REAL = True`, then run from the repo root: `python -m jupyter nbconvert --to notebook --execute notebooks/track_emg_ninapro.ipynb --output track_emg_ninapro.executed.ipynb --ExecutePreprocessor.timeout=1800`.

## Who did what

| Iteration | Who | Modules / tasks owned | Reviewed by |
|---|---|---|---|
| 1 | Marissa | Real-data setup, signal inspection, baseline validation | |
| 1 | Barnus | Recording-selection fix, evaluation and smoke-test checks | |
| 1 | Haripriya | Baseline feature check (shape, finite values, stats, flat windows, redundancy, subject scale), feature documentation, fold-safe selection A/B | |
