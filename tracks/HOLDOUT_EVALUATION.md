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

## The hold-out sets (HT26)

These ids are fixed for the whole course. They are also in the code as `trk.meta.holdout_ids`.

| Track | Split unit | Hold-out | Size |
|---|---|---|---|
| Sleep-EDF | subject | `SC408, SC416, SC424, SC432, SC440, SC448, SC456, SC464, SC472, SC480` (mne subject indices 8, 16, …, 80; both nights) | 10 subjects |
| ECG CinC-2017 | record | 120 records, 30 per class (N / A / O / ~): `HOLDOUT_RECORDS` in `ecg_cinc2017.py` | 120 records |
| EMG Ninapro DB1 | subject | `S23, S24, S25, S26, S27` | 5 subjects |
| BCI EEGMMIDB (opt-in) | subject | `S015, S025, S035, S045, S055, S065, S075, S085, S095, S105` | 10 subjects |

None of them is a smoke-test record or part of a track's default download, and every `download()` now
**skips hold-out ids automatically**, so they cannot end up in your training data by accident. If you
downloaded data with your own settings before these ids were published, check with
`trk.assert_no_holdout(recs)`, remove any hold-out recordings you find, and say so in `RESULTS.md`.

## How a group produces a submission

**1. Develop on training data only.** All your cross-validation, tuning and design decisions use data
without the hold-out ids:
```python
from sleep_edf import SleepEDFTrack
trk = SleepEDFTrack()
trk.download("data/train", subset=range(0, 30))      # hold-out subjects are skipped automatically
recs = trk.load("data/train")
trk.assert_no_holdout(recs)                          # raises if a hold-out recording slipped in
```

**2. Register your expected score — before you look at the hold-out.** When your final model is
frozen, commit a file `EXPECTED_SCORE.md` to your team repository containing:
- the track's default metric from your cross-validation, as **mean ± SD across folds**;
- the split unit and number of folds;
- the commit of the frozen final model.

The commit time is your registration: it must come **before** the hold-out data appears in your
repository's history. This turns the hold-out into a test of your own estimate. Scoring within or
below your CV range is the normal, honest outcome; a hold-out score far above it is a question we
will ask.

**3. Predict the hold-out once.** Download the hold-out set into a **separate folder** and write
the submission with the frozen model:
```python
trk.download_holdout("data/holdout")                 # only the hold-out ids
holdout = trk.load("data/holdout")
trk.write_submission(holdout, "predictions.csv", model)   # labels are never read
```
Submit `predictions.csv` with your report. Do not score it yourself, and do not change the model
afterwards: the hold-out is used once.

Training, tuning or selecting features on a `group` in `holdout_ids` — even by accident — is a
leakage violation exactly like ignoring the split unit, and is caught the same way: `RESULTS.md`,
`EXPECTED_SCORE.md` and `git log`.

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
