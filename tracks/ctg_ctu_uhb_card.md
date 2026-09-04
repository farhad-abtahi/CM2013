# Dataset card — CTG (fetal)

> ⚠️ **Not offered this year.** See [`START_HERE.md`](START_HERE.md) for the current tracks.

| Field | Value |
|---|---|
| Dataset | CTU-UHB Intrapartum Cardiotocography (v1.0.0) |
| License / access | Open Data Commons Attribution (ODC-BY) · **open-direct** |
| Signals | FHR @ 4 Hz, uterine contraction (toco) @ 4 Hz |
| Task | binary — classes: normal, pathological |
| **Split unit (leakage)** | **recording** |
| Evaluation modes | new-recording |
| Default metrics | macro_f1, cohens_kappa |
| Smoke-test records | 1001, 1002, 1004 |
| Expected runtime | ~2–4 min on a subset (Colab, CPU) |
| DSP focus | dropout removal + interpolation, FHR baseline/variability (STV/LTV), accel/decel detection, uterine-contraction features |
| Hold-out submission | record-level `predictions.csv` |
| Difficulty (1–5) | 3 |

**Citation.** Chudáček V, et al. (2014); Goldberger AL, et al. PhysioNet (2000).

**Source.** https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

## Measured on real data (validated)
150-record subset, umbilical-artery **pH < 7.15 = pathological** (29 pathological / 121 normal).
Supplied RF baseline on the **last 30 min** of each trace, **GroupKFold by recording**:

| Metric | Value |
|---|---|
| macro-F1 | **0.53** |
| Cohen's κ | **0.11** |
| Recall (normal / pathological) | 0.98 / **0.10** |
| Runtime | ~44 s download (150 recs) + ~15 s features/CV, CPU |

**This is honestly a hard problem.** Predicting cord-blood acidemia from CTG has low sensitivity
even in the literature; the supplied baseline barely detects the pathological class. That is the
teaching point — the gains are in **better variability/deceleration features and window choice**,
not the classifier. Analysing the full 90-min trace instead of the last 30 min is *worse*
(macro-F1 ≈ 0.44, pathological recall 0.0).

**Yardstick.** There is no strong ceiling here: **clinician interobserver agreement on CTG is only
κ ≈ 0.3–0.5**, and automated cord-pH prediction is an open problem with modest sensitivity. So "good"
is not 0.95 — it is a defensible, honestly-evaluated improvement in **pathological-class sensitivity**
over our 0.10, achieved without leakage. Read your result against that low, noisy human ceiling.

## Known pitfalls
- **Raw FHR is full of signal-loss zeros and spikes** — you *must* clean (remove 0/out-of-range,
  interpolate, median-filter) before any feature, or variability is meaningless.
- **Label = umbilical pH threshold** (default 7.15; 7.05 is severe-acidemia and very imbalanced).
  State your threshold — it changes the whole task. Apgar is an alternative label in the header.
- **Strong class imbalance** (few pathological). Score macro-F1 / κ / per-class recall, never accuracy.
- **Where in the trace matters** — the acidemia signal concentrates near delivery; window the tail.
- **Split by recording.** One label per record (`record,label` hold-out submission).
