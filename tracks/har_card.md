# Dataset card — Human activity (IMU)

| Field | Value |
|---|---|
| Dataset | UCI HAR (raw inertial signals) (v1.0) |
| License / access | Open (UCI ML Repository) · **open-direct** |
| Signals | tri-axial accelerometer (total_acc) |
| Task | multiclass — classes: static, walk, stairs |
| **Split unit (leakage)** | **subject** |
| Evaluation modes | new-subject |
| Default metrics | macro_f1, balanced_accuracy |
| Smoke-test records | subj_1, subj_2 |
| Expected runtime | ~1–3 min (Colab, CPU) |
| DSP focus | gravity/motion filtering, windowing, dominant-frequency (cadence), magnitude |
| Hold-out submission | epoch-level `predictions.csv` |
| Difficulty (1–5) | 2 |

**Citation.** Anguita D, et al. UCI HAR dataset (2013).

**Source.** https://archive.ics.uci.edu/dataset/341/

## Measured on real data (validated)
Full UCI-HAR raw inertial set: **30 subjects**, 10,299 windows (128 samples @ 50 Hz), coarse map
static/walk/stairs. Supplied RF baseline, **GroupKFold by subject**:

| Metric | Value |
|---|---|
| macro-F1 | **0.89** |
| balanced accuracy | **0.88** |
| Runtime | ~13 s download+extract (60 MB zip) + ~9 s features/CV, CPU |

Honest confusion is **walk ↔ stairs** (static is near-perfect); the 6→3 coarse map makes this the
easiest track (difficulty 2). The accelerometer-only reference loader leaves the **gyroscope** as an
obvious student improvement.

**Yardstick.** The classic UCI-HAR benchmark reaches **~0.96 accuracy on the full 6-class task** using
the provided 561-feature vector + SVM. Our 0.89 is on the coarser 3-class map from **raw** signals with
a handful of self-built features — the gap is what feature construction (and adding the gyroscope) buys.

## Known pitfalls
- **Raw inertial signals**, not the 561-feature file — feature *construction* is the point.
- **Classes:** 3 coarse **static / walk / stairs** (a map of UCI-HAR's 6). Smoke uses the same space.
- **Accelerometer only** in the reference loader; the **gyroscope** is available for an extension.
- **Split by subject;** overlapping windows must not straddle train/test.
