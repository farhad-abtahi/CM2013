# Per-track hold-out competition

Each track runs a small **hold-out competition** alongside the cross-track showcase: groups
train on the labelled training records, predict on a **hold-out** set, and submit a
`predictions.csv`. The instructor scores submissions on the withheld labels and posts a
leaderboard on the track's **default metric**.

## Submission format (set per track by `TrackMeta.submission_granularity`)

| Track | Granularity | Header | One row per | Leaderboard metric |
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

**Two-mode tracks (EMG, BCI).** The leaderboard scores the **new-subject** hold-out only —
the honest deployment claim. Your report must still show both modes side by side
(Criterion 8), but the competition ranks the hard one.

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
Rank on the track's **default metric** (see the table above; Sleep → Cohen's κ, all
others → macro-F1). Because the
metric and the split unit are declared in the dataset card, cross-track leaderboards are
comparable *within* a track and honestly *incomparable* across domains — which is the point
of the capstone rubric (`CAPSTONE_REPORT_RUBRIC.md`).

## Rules
- The hold-out labels are revealed **once**, at grading — never used for tuning.
- Submissions must come from a model trained with the **declared split unit** (no leakage).
- The baseline is supplied; the leaderboard rewards **honest improvement over the baseline**.
