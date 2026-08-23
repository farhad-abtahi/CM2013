# Dataset card — EMG gesture

| Field | Value |
|---|---|
| Dataset | Ninapro DB1 (surface EMG, exercise E1) (v1.0) |
| License / access | CC BY-NC-ND — cite Atzori et al. (2014) · **open-direct** |
| Signals | 10-channel surface EMG envelope @ 100 Hz |
| Task | multiclass — classes: g01, g02, g03, g04, g05, g06, g07, g08, g09, g10, g11, g12 |
| **Split unit (leakage)** | **subject** |
| Evaluation modes | within-subject, new-subject |
| Default metrics | macro_f1, cohens_kappa |
| Smoke-test records | S1, S2, S3 |
| Expected runtime | ~3–6 min on a subject subset (Colab, CPU) |
| DSP focus | windowing, MAV/RMS/waveform-length/variance + mean-frequency per channel |
| Hold-out submission | epoch-level `predictions.csv` |
| Difficulty (1–5) | 3 |

**Citation.** Atzori M, et al. Sci Data (2014). Ninapro DB1.

**Source.** https://ninapro.hevs.ch/instructions/DB1.html

## Measured on real data (validated)
5 subjects, exercise E1 (12 finger movements), 200 ms / 150 ms windows → 13,875 windows.
Supplied RF baseline; **both evaluation modes**:

| Mode | macro-F1 | Cohen's κ |
|---|---|---|
| **within-subject** (leave-repetitions-out) | **0.78** | 0.77 |
| **new-subject** (leave-one-subject-out) | **0.19** | 0.12 |

Runtime ~21 s download (5 subjects) + ~90 s features/CV, CPU. **This gap is the lesson:** sEMG
gesture recognition is easy *within* a subject but collapses *across* subjects — electrode
placement, muscle anatomy, and skin impedance differ. Report the mode with every number.

**Yardstick.** Published within-subject accuracy on Ninapro DB1 is **~0.75–0.90** with the classic
Hudgins time-domain feature set + a good classifier — our 0.78 within-subject is right in that band.
Cross-subject has **no strong ceiling**: it stays low without calibration/domain adaptation, so our
0.19 is honest, not broken. The two numbers answer two different deployment questions.

## Known pitfalls
- **DB1 ships a rectified low-pass envelope, not raw bipolar sEMG** — so zero-crossings / slope-sign
  changes (raw-sEMG staples) are meaningless; use amplitude/energy features. Know your signal.
- **Cross-subject is near-useless without adaptation** (calibration, normalisation, domain transfer).
- **Window-overlap leakage:** never let windows from the same repetition span train and test —
  split by **repetition** (within-subject) or **subject** (new-subject).
- **Class imbalance / rest handling:** rest windows are excluded here; decide deliberately.
