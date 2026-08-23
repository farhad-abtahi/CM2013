# Dataset card — BCI motor imagery

| Field | Value |
|---|---|
| Dataset | EEG Motor Movement/Imagery (EEGMMIDB) (v1.0.0) |
| License / access | Open Data Commons Attribution (ODC-BY) · **open-direct** |
| Signals | 64-channel EEG @ 160 Hz |
| Task | binary — classes: L, R |
| **Split unit (leakage)** | **subject** |
| Evaluation modes | within-subject, new-subject |
| Default metrics | macro_f1, cohens_kappa |
| Smoke-test records | S001, S002, S003 |
| Expected runtime | ~3–6 min on a subject subset (Colab, CPU) |
| DSP focus | band-pass, mu/beta ERD at C3/C4/Cz, C3–C4 laterality, CSP (extension) |
| Competition submission | epoch-level `predictions.csv` |
| Difficulty (1–5) | 4 |

**Citation.** Schalk G, et al. (2004); Goldberger AL, et al. PhysioNet (2000).

**Source.** https://physionet.org/content/eegmmidb/1.0.0/

## Measured on real data (validated)
8 subjects, imagery runs R04/R08/R12, 270 L/R trials (4 s epochs, 64 ch). Supplied RF baseline on
log mu/beta band power across a sensorimotor montage; **both modes**:

| Mode | macro-F1 | Cohen's κ |
|---|---|---|
| within-subject (trial CV) | ~0.51 | ~0.02 |
| new-subject (LOSO) | ~0.51 | ~0.04 |

**The supplied baseline is near chance — that is expected and is the point.** EEGMMIDB motor imagery
is hard: separability varies enormously across subjects, and naive band-power + RandomForest barely
beats a coin flip. The known path up is a **CSP spatial filter fit inside each fold + LDA**, plus
subject-specific band tuning. This is the most advanced track (difficulty 4); students are expected
to move the needle with proper spatial filtering, not just more features.

**Yardstick.** With **CSP + LDA**, within-subject decoding reaches **~0.65–0.80 for good subjects**
(and near chance for others — EEGMMIDB has known "BCI-illiterate" subjects); subject-independent stays
much lower. Chance is 0.50. So the honest target is not "beat 0.95" — it is "get **meaningfully above
chance on the decodable subjects** with a leakage-safe CSP," and report which subjects were decodable.

## Known pitfalls
- **Near-chance naive baseline** — do not mistake it for a bug; the signal needs CSP, not just C3/C4.
- **Fit any spatial filter (CSP) INSIDE the CV fold**, never on all data — that is a classic leak.
- **Two modes matter:** within-subject (calibrated) vs new-subject (LOSO) answer different claims;
  cross-subject EEG is brutal (non-stationarity, montage/reference differences).
- **Label semantics:** in runs 4/8/12, annotation T1 = left-fist imagery, T2 = right-fist; T0 = rest.
