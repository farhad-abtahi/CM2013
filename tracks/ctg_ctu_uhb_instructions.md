# Track instructions — CTG (fetal), CTU-UHB

*Attach the dataset card (`ctg_ctu_uhb_card.md`). Adapter: `ctg_ctu_uhb.py`.*

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "CTG / fetal"
> section): it lists the **book sections** (dropout/interpolation §12.9–12.11, baseline/variability,
> deceleration detection via the Pan–Tompkins template §9.9 …), the **literature-review anchors** (CTU-UHB
> paper, **FIGO** guidelines defining deceleration types), and the **guiding questions** your report must
> answer to *motivate* your design (course outcome **L5**).

## 1. The problem
From an intrapartum cardiotocogram — **fetal heart rate (FHR)** + **uterine contractions (UC)** —
predict whether the delivery outcome was **normal** or **pathological** (fetal acidemia). One label
per recording, derived from umbilical-artery **pH**.

## 2. Your data (see the dataset card)
- Dataset: **CTU-UHB Intrapartum CTG** (ODC-BY, direct download — **no agreement**).
- Signals: FHR + UC (toco), both **4 Hz**, ~90 min/record. Labels: normal / pathological (pH < 7.15).
- **Split unit (leakage): recording** — one label per record; never leak a record across folds.
- Smoke subset for setup/CI: records `1001, 1002, 1004`. Full subset runtime: ~2–4 min.
- Evaluation mode: **new-recording** (GroupKFold on record id).

## 3. What you are given (do not rebuild these)
- The **modular pipeline** and a **working baseline** (`tracks.adapter.default_baseline`).
- The adapter (`ctg_ctu_uhb.py`) with `smoke()`, `preprocess()` (identity by default — dropout handling
  belongs here once you choose a strategy), `extract_features()` (dropout cleaning + baseline/variability
  + accel/decel + UC features on the **last 30 min**), and a `wfdb` `download()/load()` that parses pH.
- The shared leakage-safe **evaluator** (`TrackAdapter.evaluate`, GroupKFold by record).
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across folds** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** on the smoke subset, then on real records. It barely detects the pathological
   class (recall ≈ 0.1) — that is your starting point, not your result.
2. **Improve the signal processing**: better dropout handling; deceleration morphology (late vs
   variable vs early), deceleration area, short-term variability (STV) done properly, sample entropy;
   try the **last 15 min** or contraction-locked windows.
3. **Improve the model / threshold** (validated inside folds); consider the pH threshold's effect.
4. **Analyse failures** — which pathological records are missed, and what does their trace look like?

Those steps are a *menu, not a march*: which preprocessing, which features, whether to select
features at all, and which learner to climb to are decisions the scaffold deliberately leaves
open (see `preprocess()` and `make_selector()` / `default_baseline()` in `adapter.py` for the options and their
trade-offs). The grade is on the quality of the reasoning, not on matching one blessed recipe.

## 4c. The design-decision menus (stages 2-5) — what the scaffold offers, and the trade-off

Everything here is a **menu, not a recipe**. `tracks/adapter.py` ships each option *with its
trade-off* and no blessed answer; the rubric grades the reasoning (criteria 2 and 5), not the
choice. The notebook's "Decision points on this track" section runs several of them side by side
so you can watch the numbers move.

### Stage 2 — preprocessing · the dropout decision, shipped in the wrong place

`_clean_fhr()` is called from `extract_features()` for convenience, but it is **preprocessing**.
Moving it into `preprocess()` and choosing what to do with the signal-loss gaps — interpolate,
exclude, or keep-and-flag — is one of the decisions this track exists to make you argue for.
`adapter.denoise()` is importable and gives you the Chapter 8 per-noise-type menu below (call it from a `preprocess()` you write — it is not a `cfg` key on this track); note that raw FHR dropout is a
**data-integrity** problem in §8.2's sense ("bad segments should be detected and excluded or flagged,
**not silently filtered**"), not a noise-remedy problem, and conflating the two is the specific
mistake this track punishes.

### Stage 2 — the Chapter 8 noise-type menu · **available as functions, not wired into this track's `cfg`**

`CTGTrack` has no `preprocess()` stage at all — dropout handling (`_clean_fhr`) is called from `extract_features()`, which is exactly the misplacement this track asks you to argue about — so `cfg={"preprocess": "denoise", ...}` is **rejected** with an `UnsupportedCfgKey` error rather than silently ignored.

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

The variability features (STV/LTV) are windowed variances in the time domain, not band powers.

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
- A short **report** (features justified physiologically; results vs. the class-prior baseline).
- `predictions.csv` (`record,label`) on the held-out split for hold-out evaluation.
- A slot in the **cross-track showcase**: "same variability/deceleration ideas, our signal."
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Beat the supplied baseline **honestly**. State the **split unit (recording)**, the **pH threshold**,
  and the evaluation mode with every number. Never report smoke/CI numbers as results. Report the metric **with its spread** across folds (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: raw FHR is full of signal-loss zeros/spikes (clean first), strong class imbalance, the
label is a pH threshold you must state, and the acidemia signal concentrates near delivery (window the tail).
