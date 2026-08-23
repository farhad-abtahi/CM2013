# Dataset card — ECG rhythm (AF)

| Field | Value |
|---|---|
| Dataset | PhysioNet/CinC Challenge 2017 (single-lead ECG) (v1.0.0) |
| License / access | Open (PhysioNet Challenge) · **open-direct** |
| Signals | single-lead ECG @ 300 Hz |
| Task | multiclass — classes: N, A, O, ~ |
| **Split unit (leakage)** | **record** |
| Evaluation modes | new-record |
| Default metrics | macro_f1, cohens_kappa |
| Smoke-test records | A00001, A00004, A00006 |
| Expected runtime | ~2–5 min on a subset (Colab, CPU) |
| DSP focus | band-pass + notch, Pan–Tompkins QRS, R–R irregularity/HRV, signal quality |
| Competition submission | record-level `predictions.csv` |
| Difficulty (1–5) | 3 |

**Citation.** Clifford GD, et al. PhysioNet/CinC Challenge 2017.

**Source.** https://physionet.org/content/challenge-2017/1.0.0/

## Measured on real data (validated)
Class-stratified subset of **431** training records (N 238 · O 113 · A 40 · ~ 40), 300 Hz,
variable length (median 9000 samples ≈ 30 s; range ~10–61 s). Supplied RF baseline, **GroupKFold(5)
by record** (leakage-guarded):

| Metric | Value |
|---|---|
| macro-F1 | **0.655** |
| Cohen's κ | **0.524** |
| Per-class recall | N 0.88 · A 0.58 · O 0.52 · ~ 0.52 |
| Runtime | ~73 s total (≈70 s download, ~3 s features+CV) on CPU |

Honest gap vs the synthetic smoke (macro-F1 ≈ 0.81): real single-lead ECG rhythm is hard — Other (O)
and AF (A) overlap, and the supplied features are deliberately simple so students can improve them.
The smoke set is *deliberately* imperfect too (each record draws a "typicality", and the class counts
are the challenge's own lopsided ones), so the offline number is a plausible rehearsal for the real
one rather than a separable caricature that scores ≈ 1.0.

**Yardstick.** The CinC-2017 Challenge **winners reached macro-F1 ≈ 0.83** (averaged over N/A/O). Our
0.66 baseline is ~0.17 below that — a concrete, honest target for students to close with better QRS
detection, richer HRV/morphology features, and explicit Noisy-class signal-quality handling.

## Known pitfalls
- **Single lead, variable length (9–60 s), 300 Hz** — pad/segment consistently.
- **AF signature is R–R irregularity**, not a single feature; the `~` (Noisy) class is a *signal-quality* problem, not a rhythm.
- **Strong class imbalance** (Normal ≫ AF ≫ Other ≫ Noisy) — score macro-F1 / κ, never accuracy.
- **Split by record**; the official metric is per-record — the competition submits `record,label`.
- Robust **QRS detection under noise** is half the battle; a bad detector wrecks the R–R features.
