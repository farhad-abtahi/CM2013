# Per-track hold-out evaluation

Each track includes a small **hold-out evaluation** alongside the cross-track showcase: groups
train on the labelled training records, predict on a **hold-out** set, and submit a
`predictions.csv`. The instructor scores that submitted file against the withheld labels on the
track's **default metric** — this is graded as an honest-validation check, not ranked against
other teams. There is no leaderboard: your score is reported back to your team only.

**This is an honesty check, not a blinding mechanism.** Every dataset this repo ships is
public — `load()` always returns fully labelled recordings, there is no separate unlabelled
file the instructor hands out. The hold-out split is enforced by **id discipline**, exactly
like the leakage-safe split everywhere else in this scaffold: `TrackMeta.holdout_ids` names
the `group`s (subjects/records) your team must train on nothing from and predict blind. The
same honesty mechanism the rest of this course already leans on — `RESULTS.md` + the `git log`
cross-check (Criterion 9) — is what makes "predict blind" enforceable on a public dataset; it
is not a cryptographic guarantee, and it does not need to be.

## Submission format (set per track by `TrackMeta.submission_granularity`)

| Track | Granularity | Header | One row per | Default metric |
|---|---|---|---|---|
| Sleep-EDF | `epoch` | `record,epoch,label` | 30-s epoch | Cohen's κ |
| ECG (CinC-2017) | `record` | `record,label` | recording | macro-F1 |
| HAR (UCI, raw IMU) | `epoch` | `record,epoch,label` | 2.56-s window | macro-F1 |
| CTG (CTU-UHB) | `record` | `record,label` | recording (one pH-derived label per trace) | macro-F1 |
| EMG (Ninapro DB1) | `epoch` | `record,epoch,label` | 200-ms sEMG window | macro-F1 |
| BCI (EEGMMIDB) | `epoch` | `record,epoch,label` | imagery trial | macro-F1 |

Granularity is declared per track by `TrackMeta.submission_granularity`, and
`write_submission()` writes the right header for you — never hand-roll the CSV. This
matches the CM2013 sleep convention (`record,epoch,label`) and generalises it.

**Two-mode tracks (EMG, BCI).** The hold-out score is computed on the **new-subject**
hold-out only — the honest deployment claim. Your report must still show both modes
side by side (Criterion 8).

## How a group produces a submission
```python
from ecg_cinc2017 import ECGCinC2017Track
trk = ECGCinC2017Track()
all_recs = trk.load("data")                                     # every labelled record (public dataset)
train   = [r for r in all_recs if r.group not in trk.meta.holdout_ids]
holdout = [r for r in all_recs if r.group in trk.meta.holdout_ids]   # predict blind on these

model = trk.train_baseline(train)        # or the group's improved model
trk.write_submission(holdout, "predictions.csv", model)   # holdout's labels are never read
```
`trk.meta.holdout_ids` is set by the instructor before the assignment goes live (see
`dataset_manifest.json` / each track's card). Training on a `group` in `holdout_ids` — even by
accident, e.g. by loading `"data"` and forgetting to filter — is a leakage violation exactly
like ignoring the split unit, and is caught the same way: `RESULTS.md` + `git log`.

## How the instructor scores it
```python
report = trk.score_submission("predictions.csv", holdout)   # scores the SUBMITTED file
#   -> {accuracy, cohens_kappa, macro_f1, balanced_accuracy, confusion, ...}
```
`score_submission()` reads the team's actual `predictions.csv` and aligns it to the withheld
labels by `(record[, epoch])` — it does not retrain anything, so it is scoring exactly what the
team produced. (`holdout_score()` also exists in `adapter.py`, but it *refits* `clf`/`cfg` from
scratch and — called with no arguments, as it is easy to do — silently scores the shipped
default baseline instead of a team's work. Use `score_submission()` for grading; `holdout_score()`
is a reproducibility check, not a submission scorer.)

Because the metric and the split unit are declared in the dataset card, hold-out scores are
comparable *within* a track and honestly *incomparable* across domains — which is why they
are never pooled into a cross-track ranking. See `CAPSTONE_REPORT_RUBRIC.md` for how this
fits into grading.

## Rules
- `holdout_ids` are never trained on — treat them exactly like a fourth CV fold you must
  never touch, from the first line of code.
- Submissions must come from a model trained with the **declared split unit** (no leakage).
- The baseline is supplied; the hold-out score rewards **honest improvement over the
  baseline**, not a ranking against other teams — see Criterion 7 (beating it is not required).
