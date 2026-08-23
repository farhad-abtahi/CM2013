# Per-track hold-out evaluation

Each track includes a small **hold-out evaluation** alongside the cross-track showcase: groups
train on the labelled training records, predict on a **hold-out** set, and submit a
`predictions.csv`. The instructor scores each submission against the withheld labels on the
track's **default metric** — this is graded as an honest-validation check, not ranked against
other teams. There is no leaderboard: your score is reported back to your team only.

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
train = trk.load("data/train")          # labelled records (Colab) or trk.smoke() offline
model = trk.train_baseline(train)        # or the group's improved model
holdout = trk.load("data/holdout")       # unlabelled records
trk.write_submission(holdout, "predictions.csv", model)
```

## How the instructor scores it
```python
report = trk.holdout_score(train, labelled_holdout)   # fits on train, scores on hold-out
#   -> {accuracy, cohens_kappa, macro_f1, balanced_accuracy, confusion}
```
Because the metric and the split unit are declared in the dataset card, hold-out scores are
comparable *within* a track and honestly *incomparable* across domains — which is why they
are never pooled into a cross-track ranking. See `CAPSTONE_REPORT_RUBRIC.md` for how this
fits into grading.

## Rules
- The hold-out labels are revealed **once**, at grading — never used for tuning.
- Submissions must come from a model trained with the **declared split unit** (no leakage).
- The baseline is supplied; the hold-out score rewards **honest improvement over the
  baseline**, not a ranking against other teams.
