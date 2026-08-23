# Track instructions — <TRACK NAME>

*What a student group receives. Copy this, fill the `<...>`, and attach the track's
dataset card (`<track>_card.md`).*

## 1. The problem
<One paragraph: the clinical/physical question and the prediction task, e.g. "score 30-s
epochs into W/N1/N2/N3/REM from EEG/EOG/EMG.">

## 2. Your data (see the dataset card)
- Dataset: `<name>` (`<license>`, direct download — **no agreement**).
- Signals: `<signals>`. Sampling rate: `<fs>`. Labels: `<classes>`.
- **Split unit (leakage): `<SPLIT_UNIT>`** — never let the same `<unit>` appear in train and test.
- Smoke subset for setup/CI: `<SMOKE_TEST_RECORDS>`. Expected full runtime: `<runtime>`.
- Evaluation mode(s): `<new-subject | within-subject>`.

## 3. What you are given (do not rebuild these)
- The **modular pipeline** (`data_loader → preprocessing → feature_extraction →
  feature_selection → classification → inference → report`).
- A **working baseline model** (`tracks.adapter.default_baseline`) and the track adapter
  (`<track>.py`) with `smoke()`, `preprocess()`, `extract_features()`, and a Colab `load()`.
- The shared, leakage-safe **evaluator** (`TrackAdapter.evaluate`, leave-one-`<unit>`-out).
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across <unit>s** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** on the smoke subset, then on real data; report the honest panel.
2. **Improve the signal processing**: better preprocessing (filtering/denoising) and
   **features** (the DSP is the point — construct them, don't just accept the given ones).
3. **Improve the model / selection**, validated *inside* folds.
4. **Analyse failures** with the confusion matrix; relate errors to the domain.

Those steps are a *menu, not a march*: which preprocessing, which features, whether to select
features at all, and which learner to climb to are decisions the scaffold deliberately leaves
open (see `preprocess()` and `make_selector()` / `default_baseline()` in `adapter.py` for the options and their
trade-offs). The grade is on the quality of the reasoning, not on matching one blessed recipe.

## 5. Deliverables
- A short **report** (design decisions justified; results read against chance/inter-rater ceiling).
- `predictions.csv` on the held-out split (per-track mini-competition), if enabled.
- A slot in the **cross-track showcase**: "same methods, our signal — what was different?"
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Beat the **supplied baseline**, honestly. State the **split unit** and **evaluation mode**
  with every number. Never report smoke/CI numbers as results. Report the metric **with its spread** across <unit>s (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
<From the dataset card — e.g. label merges, class imbalance, motion/artifact, window-overlap
leakage, montage/reference choices.>
