# Dataset card — Sleep staging

| Field | Value |
|---|---|
| Dataset | Sleep-EDF Expanded (v1.0.0) |
| License / access | Open Data Commons Attribution (ODC-BY) · **open-direct** |
| Signals | EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental |
| Task | multiclass — classes: W, N1, N2, N3, REM |
| **Split unit (leakage)** | **subject** |
| Evaluation modes | new-subject |
| Default metrics | cohens_kappa, macro_f1 |
| Smoke-test records | SC4001, SC4011, SC4021 |
| Expected runtime | ~3–6 min on a Sleep-Cassette subset (Colab, CPU) |
| DSP focus | band power, spindle STFT/wavelet, EOG/EMG artifact handling |
| Competition submission | epoch-level `predictions.csv` |
| Difficulty (1–5) | 3 |

**Citation.** Kemp B, et al. (2000); Goldberger AL, et al. PhysioNet (2000).

**Source.** https://physionet.org/content/sleep-edfx/1.0.0/

## Measured on real data (validated)
Sleep-Cassette subset: **4 subjects × 2 nights = 8 nights** (22,240 epochs after the R&K→AASM merge),
resampled to 100 Hz. Supplied RF baseline, **leave-one-subject-out** (leakage-guarded):

| Metric | Value |
|---|---|
| Cohen's κ | **0.835** |
| macro-F1 | **0.741** |
| accuracy | 0.921 |
| Runtime | download ~33 min (PhysioNet, one-time; cached after) · load 23 s · features 14 s · LOSO 28 s |

Per-class: W and N2 strong; **N1 is the hard class** (~0.20 recall) as expected. Note **Wake dominates**
(~69 % of epochs) because Sleep-Cassette records span the full ~20 h day — accuracy is inflated by W,
so **read κ / macro-F1 / the confusion matrix**, not accuracy. (Cropping to lights-off ± 30 min is a
sensible student improvement.)

**Yardstick.** Human inter-rater agreement on sleep staging is **κ ≈ 0.76** (AASM); strong published
models on subject-wise Sleep-EDF reach **κ ≈ 0.79–0.82**. Our baseline's κ = 0.835 *looks superhuman* —
that is a **warning sign, not a triumph**: it is inflated by the long, trivially-classified Wake
stretches and by a tiny 4-subject LOSO (high variance). A number above the human ceiling should always
make you suspect the evaluation, not celebrate. Re-score on a Wake-balanced subset and it drops.

## Known pitfalls
- **R&K labels:** merge `Sleep stage 3` + `Sleep stage 4` → **N3**; drop `Movement time` and `Sleep stage ?`.
- **Split by subject, not night** (two nights per subject).
- **Class imbalance:** N2 dominates, N1 is rare — read κ / macro-F1 / confusion.
- **Loader:** all three channels required and validated per epoch; resampled to `fs_target` (100 Hz).
