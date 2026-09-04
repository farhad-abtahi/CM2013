# Track instructions — Human activity (IMU), UCI-HAR

> ⚠️ **Not offered this year.** This track isn't one of the options for CM2013 HT26 — see
> [`START_HERE.md`](START_HERE.md) for the tracks actually on offer. Kept here as reference /
> for future years.

*Attach the dataset card (`har_card.md`). Adapter: `har.py`. **The gentle on-ramp track.***

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "Human
> activity / IMU" section): it lists the **book sections** (gravity/motion separation §9.2–9.4, cadence
> FFT §3.5, windowing Ch4 …), the **literature-review anchors** (UCI-HAR paper, HAR surveys), and the
> **guiding questions** your report must answer to *motivate* your design (course outcome **L5**).

## 1. The problem
From a **tri-axial accelerometer**, classify the activity (**static / walk / stairs** — a coarse map of
UCI-HAR's 6 activities) from short overlapping windows.

## 2. Your data (see the dataset card)
- Dataset: **UCI-HAR raw inertial signals** (open, direct zip download — **no agreement**).
- Signals: total_acc x/y/z @ 50 Hz, 2.56 s windows (128 samples, 50 % overlap). Labels: 3 coarse classes.
- **Split unit (leakage): subject** — overlapping windows must not straddle train/test.
- Smoke subset: `subj_1, subj_2`. Evaluation mode: **new-subject**.

## 3. What you are given (do not rebuild these)
- The modular pipeline + a working baseline (`tracks.adapter.default_baseline`).
- The adapter (`har.py`) with `smoke()`, `preprocess()` (identity by default — the gravity/motion
  split is yours to make), `extract_features()` (per-axis mean/std/energy + dominant frequency +
  magnitude), and a `download()/load()` that reads the raw Inertial Signals.
- The shared leakage-safe **evaluator** (GroupKFold by subject).
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across subjects** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline**; report macro-F1 / balanced accuracy and the confusion matrix.
2. **Improve the signal processing / features**: proper **gravity vs body-acceleration** separation
   (low-pass), cadence from the spectrum, jerk, and **add the gyroscope** (available but unused).
3. **Improve the model / selection** (validated inside folds).
4. **Analyse failures** — walking vs stairs is the confusable pair; which feature separates them?

Those steps are a *menu, not a march*: which preprocessing, which features, whether to select
features at all, and which learner to climb to are decisions the scaffold deliberately leaves
open (see `preprocess()` and `make_selector()` / `default_baseline()` in `adapter.py` for the options and their
trade-offs). The grade is on the quality of the reasoning, not on matching one blessed recipe.

## 4c. The design-decision menus (stages 2-5) — what the scaffold offers, and the trade-off

Everything here is a **menu, not a recipe**. `tracks/adapter.py` ships each option *with its
trade-off* and no blessed answer; the rubric grades the reasoning (criteria 2 and 5), not the
choice. The notebook's "Decision points on this track" section runs several of them side by side
so you can watch the numbers move.

### Stage 2 — preprocessing · `cfg["gravity"]`

The decision that defines this track: `"none"` (default) · `"mean"` · `"highpass"`, i.e. what to do
about **gravity**, which sits in the accelerometer at ~1 g pointing wherever the device happens to be
mounted. Orientation is informative (sitting vs standing differ mainly in where gravity points) and
simultaneously pure nuisance across subjects. `adapter.denoise()` is importable and gives you the Chapter 8
per-noise-type menu below if what you see is drift or spikes rather than a gravity problem.

### Stage 2 — the Chapter 8 noise-type menu · **available as functions, not wired into this track's `cfg`**

`HARTrack.preprocess()` reads **only** `cfg["gravity"]`. It does not forward the denoise keys, so `cfg={"preprocess": "denoise", "impulsive": "median"}` is **rejected** with an `UnsupportedCfgKey` error rather than silently ignored — an option that quietly does nothing teaches the opposite of what these menus are for.

The functions themselves are real and importable, and the table below is the decision you should still be making — you just have to **call them yourself** from a `preprocess()` you write (`from adapter import denoise, bandpass_notch, wavelet_denoise`). Once you have, register the knobs so they become first-class config options:

```python
track = <YourTrack>()
track.declare_cfg_keys("preprocess", "impulsive", "baseline", "powerline", "broadband")
```


Chapter 8's rule is that you do not pick a filter, you **identify a corruption and then pick its
remedy** — §8.11: *"the noise's signature chooses the tool … not the reverse."* So
`adapter.denoise()` takes one key per problem, not one key per technique:

| Ch. 8 noise type | key | options | the trade-off |
|---|---|---|---|
| Impulsive / electrode pop (§8.7) | `impulsive` | `"median"` | the only remedy that works — a linear filter "lets the outlier vote" and smears the spike. A window longer than your narrowest feature flattens it |
| Baseline wander (§8.8) | `baseline` | `"highpass"` · `"detrend"` † · `"wavelet"` † | 0.5 Hz is fine for monitoring; a *diagnostic* ECG needs 0.05 Hz — too high a corner "can manufacture artificial ST shifts that mimic ischaemia". `"detrend"` avoids the corner but subtracts genuine slow trends too |
| Powerline (§8.6) | `powerline` | `"notch"` · `"adaptive"` · `"spectral"` † | `q` is the whole decision: too narrow misses a drifting hum, too wide bites real signal. `"adaptive"` (LMS) tracks drift; `"spectral"` avoids ringing but needs frequency resolution |
| Broadband / white (§8.4) | `broadband` | `"movavg"` · `"savgol"` · `"wavelet"` · `"gaussian"` | §8.4: "the only clean weapons against it are averaging and improving the acquisition hardware". `"savgol"` keeps peak height/width; `"wavelet"` keeps sharp transients best; neither rejects outliers |

**The order is fixed:** `impulsive → baseline → powerline → broadband`, because §9.7 requires that
"impulse removal must precede any linear filtering". Pass `broadband="median"` and the harness
moves it and tells you so.

> † `"detrend"`, `"wavelet"` (baseline) and `"spectral"` are **extensions beyond the book** —
> Ch. 8 names only high-pass/detrend, notch, median and averaging/adaptive. Use them if you can
> defend them, but cite something other than the book.

Two single-tool helpers sit underneath: `adapter.bandpass_notch()` (stationary narrow-band
interference) and `adapter.wavelet_denoise()` (non-stationary transients — motion, pops, drift —
removed *without* rounding off sharp landmarks the way a fixed band does). Full tables in their
docstrings.

### Stage 3 — spectral estimation · available, but **not wired into this track's features**

`adapter.make_spectral_estimator` / `spectral_bandpower` offer the Chapter 7 menu (`"periodogram"` ·
`"bartlett"` · `"welch"` · `"multitaper"` · `"ar"`), but this track's shipped `extract_features()`
computes no PSD-integrated band powers, so `cfg["spectral_method"]` has nothing to act on here.

The dominant-frequency (cadence) feature is read off a raw `|rFFT|` magnitude spectrum. That is worth
revisiting in light of §7.8's "common student mistakes" box: a peak located on a raw magnitude
spectrum inherits the periodogram's variance, and §7.9's five-point "Is that spectral peak real?"
checklist applies directly. Swapping in a proper PSD estimate is a legitimate stage-3 improvement.

### Stage 4 — feature selection · `cfg["select"]`, `cfg["select_k"]`, `cfg["select_C"]`

`"none"` (default) · `"variance"` · `"anova"` · `"mutual_info"` · `"tree"` · **`"lasso"`**.

**`"lasso"`** is L1-penalised linear SVM selection: coefficients are driven to **exactly zero**, so
you get genuine sparsity and a short, defensible feature list. It is *embedded* like `"tree"` — both
see interactions a univariate filter cannot — but the two behave oppositely on correlated features.
A forest **splits the credit** between near-duplicates, so both survive; L1 **picks one and zeroes
the other**, because a second copy of an already-used feature buys no likelihood and costs penalty.
Trade-off: the sparsity is interpretable and cheap to report, but it assumes roughly **linear**
separability and is wrong when the real relationship is not; and because the winner inside each
correlated cluster is close to arbitrary, **check the surviving set is stable across folds** before
calling it "the" feature set. `select_C` is the strength (smaller C = fewer features) and it is a
number you must report. The harness prints how many features survived each fold, and falls back
loudly if the penalty erased them all.

### Stage 5 — class imbalance · `cfg["imbalance"]`, `cfg["threshold"]`

`"none"` · `"balanced"` (default) · `"balanced_subsample"` · `"resample"` · **`"smote"`** ·
**`"adasyn"`** · `"threshold"`.

**`"smote"`/`"adasyn"`** synthesise minority rows by **interpolating** between a real sample and one
of its k nearest minority neighbours, rather than duplicating exact rows (`"resample"`) or
reweighting the loss (`"balanced"`). The distinction to state in your report: `"balanced"` never adds
a row; `"resample"` adds exact copies, so the minority region gets heavier but not one millimetre
wider; `"smote"` adds **new points in feature space**, so the region genuinely expands and the
boundary is pushed rather than weighted.

That expansion is the whole benefit *and* the whole danger. A point halfway between two epochs is a
claim about feature space, **not about physiology** — interpolate between two different patients, or
between an N2 and an N3 epoch, and you have manufactured a body that does not exist. On these small,
heterogeneous cohorts that risk is live, not a footnote. If you use it, say what a synthetic minority
sample *means* on this track, and be honest if the answer is "nothing physiological".

Both are fold-safe (`adapter.SMOTEd` fits inside `fit` only, never at predict time) and both need the
optional `imbalanced-learn` package; every other option is sklearn-only. On tiny folds the
`k_neighbors` requirement is adapted downward automatically, and a minority class with fewer than two
members falls back to plain duplication with a loud warning rather than crashing mid-CV.

### Not a menu — the validation scheme

The leakage-safe split (LOSO / GroupKFold on this track's split unit) is **not** a design choice, and
there is deliberately no config key to turn it off. It is the only number that counts for your grade,
and `evaluate()` enforces it on every fold. The notebook's decision-points section contains a
**required one-time demonstration** that scores the data twice — a naive random stratified split and
the honest group-aware split — and prints the gap between them. Run it once, predict the gap first,
and record both numbers in `RESULTS.md`.

## 5. Deliverables
- A short **report** reading your macro-F1 against the **~0.96 on the standard 6-class benchmark** (with
  the provided feature vector) — you are on the harder *raw-signal, feature-construction* path. Justify choices.
- `predictions.csv` (`record,epoch,label`) on the held-out split for hold-out evaluation.
- A cross-track showcase slot: "windowing + spectral features, on IMU."
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Beat the supplied baseline **honestly**. State the **split unit (subject)** with every number. Never
  report smoke/CI numbers as results. Report the metric **with its spread** across subjects (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: use the **raw** inertial signals (not the 561-feature file), separate gravity from motion,
don't let overlapping windows straddle the subject split, and walk↔stairs is the hard pair.
