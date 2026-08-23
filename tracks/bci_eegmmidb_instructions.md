# Track instructions — BCI motor imagery, EEGMMIDB

*Attach the dataset card (`bci_eegmmidb_card.md`). Adapter: `bci_eegmmidb.py`. **Advanced track.***

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "BCI motor
> imagery" section): it lists the **book sections** (SSVEP-BCI averaging §10.4, mu/beta ERD §7.5, spatial
> unmixing PCA/ICA→CSP §10.7 …), the **literature-review anchors** (BCI2000 paper, ERD/ERS, **CSP**), and
> the **guiding questions** your report must answer to *motivate* your design (course outcome **L5**).

## 1. The problem
From **64-channel EEG** during motor imagery, decide whether the subject imagined moving the
**left or right hand** (runs R04/R08/R12). One label per ~4 s imagery trial.

## 2. Your data (see the dataset card)
- Dataset: **EEGMMIDB** (ODC-BY, direct download via `mne.datasets.eegbci` — **no agreement**).
- Signals: 64-ch EEG **@ 160 Hz**. Labels: L / R (annotation T1 = left fist, T2 = right fist).
- **Split unit (leakage): subject** (new-subject) or **trial** (within-subject).
- Smoke subset: `S001, S002, S003`. Two evaluation modes: **within-subject** and **new-subject**.

## 3. What you are given (do not rebuild these)
- The modular pipeline + a working baseline (`tracks.adapter.default_baseline`).
- The adapter (`bci_eegmmidb.py`) with `smoke()`, `preprocess()` (identity by default — a learned
  spatial filter belongs here or in `select_features()`, fit inside the fold), `extract_features()`
  (log mu/beta band power over a sensorimotor montage + C3–C4 laterality), `download()/load()`, and
  **`evaluate_modes()`** — each mode reported with its per-subject spread.
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across subjects** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** in both modes. It is **near chance** — that is your starting point and the
   whole challenge, not a bug.
2. **Add a spatial filter**: **CSP** (Common Spatial Patterns) fit **inside each CV fold**, then LDA —
   this is the single biggest lever. Tune subject-specific mu/beta bands.
3. **Handle non-stationarity** for the new-subject mode (normalisation, transfer).
4. **Analyse failures** — which subjects are decodable at all? (Many EEGMMIDB subjects are not.)

Those steps are a *menu, not a march*: which preprocessing, which features, whether to select
features at all, and which learner to climb to are decisions the scaffold deliberately leaves
open (see `preprocess()` and `make_selector()` / `default_baseline()` in `adapter.py` for the options and their
trade-offs). The grade is on the quality of the reasoning, not on matching one blessed recipe.

## 4c. The design-decision menus (stages 2-5) — what the scaffold offers, and the trade-off

Everything here is a **menu, not a recipe**. `tracks/adapter.py` ships each option *with its
trade-off* and no blessed answer; the rubric grades the reasoning (criteria 2 and 5), not the
choice. The notebook's "Decision points on this track" section runs several of them side by side
so you can watch the numbers move.

### Stage 2 — preprocessing · a stub you fill in

The moves that matter here are a **CSP spatial filter fit inside the training fold**, a
subject-specific mu band, and a Laplacian / common-average re-reference. `adapter.denoise()` is importable (call it from a `preprocess()` you write; it is not a `cfg` key on this track) and gives you
the Chapter 8 per-noise-type menu below if what you actually see in the traces is hum, drift or pops
rather than a spatial-mixing problem.

### Stage 2 — the Chapter 8 noise-type menu · **available as functions, not wired into this track's `cfg`**

`BCIEEGMMIDBTrack.preprocess()` is an identity stub — the stage-2 move with the literature behind it here is a CSP spatial filter you write and fit inside the training fold — so it forwards no denoise keys, and `cfg={"preprocess": "denoise", ...}` is **rejected** with an `UnsupportedCfgKey` error rather than silently ignored.

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

### Stage 3 — spectral estimation · `cfg["spectral_method"]`, `cfg["ar_order"]`, `cfg["mt_bandwidth"]`

Every band-power feature is an integral under a **PSD**, and the PSD is an *estimate*, not a fact —
which estimator produced it has the same standing as the band edges (Chapter 7). `"periodogram"` ·
`"bartlett"` · `"welch"` (default) · `"multitaper"` · `"ar"` (Burg), via
`adapter.make_spectral_estimator` / `spectral_bandpower`.

§7.2 calls the raw periodogram **inconsistent**: more data buys more bins, not less scatter. §7.4
names the law the rest of the family obeys — *"This resolution-variance trade-off is the governing
law of non-parametric spectral estimation."* Welch is §7.16's "long, stationary record — the
workhorse"; **multitaper** averages orthogonal DPSS tapers over the *whole* record instead of
chopping it, so it is the better non-parametric choice on short segments "where Welch runs out of
segments to average"; **AR/Burg** gives a smooth, high-resolution spectrum from a short record but
carries §7.9's model-order risk — too low and real peaks merge, too high and the model sprouts
"spurious peaks that are not in the true spectrum", which your band powers will faithfully integrate.

§7.9's decision framework, worth quoting in your report: *"if you have a long, stationary record, use
non-parametric Welch — the everyday workhorse; if you have a short segment, use parametric AR (Burg
or Yule-Walker) for its resolution — and choose the order carefully."*

**This is the track Chapter 7 §7.6-7.7 were written for.** Every feature here is a band power, and a
trial is 4 s minus a 0.5 s trim at 160 Hz — about 560 samples. Welch has to chop that into a handful
of segments and quickly "runs out of segments to average"; §7.16 recommends multitaper for exactly
this case ("short record, want low-leakage smooth PSD (EEG/neuro)") and §7.7 recommends AR for "short
segment, resonant rhythm", which is a literal description of the mu rhythm. Single-trial band power
is the noisiest quantity in the scaffold, so a better estimator has more room to help here than
anywhere else — and `"ar"` is correspondingly most dangerous: at too high an order one of §7.9's
spurious peaks landing in 8-12 Hz becomes a confident, fictitious ERD.

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

### ⚠️ Causality — this track has a real-time framing

`adapter.bandpass_notch()` defaults to `scipy.signal.filtfilt`: **zero-phase, and therefore
non-causal** — its output at each sample depends on *later* samples. That is correct for the offline,
whole-epoch analysis this scaffold performs, and it is not being taken away from you. But it means an
academically clean pipeline built here **is not deployable as written** in the live system this track
is framed around. A real deployment needs a causal filter — `scipy.signal.lfilter` with an accepted
phase delay, or a causal IIR design — and `bandpass_notch(..., causal=True)` gives you exactly that
single forward pass so you can *measure* what the honesty costs instead of asserting it. The same
applies to `denoise()`'s `"detrend"`, `"spectral"` and `"wavelet"` remedies, which are whole-epoch
operations and inherently non-causal, and to any statistic estimated over the whole recording (a
per-subject normalisation constant implies a **calibration procedure** the user performs before use —
name it). Report which regime each number belongs to, exactly as you state the split unit.

### Not a menu — the validation scheme

The leakage-safe split (LOSO / GroupKFold on this track's split unit) is **not** a design choice, and
there is deliberately no config key to turn it off. It is the only number that counts for your grade,
and `evaluate()` enforces it on every fold. The notebook's decision-points section contains a
**required one-time demonstration** that scores the data twice — a naive random stratified split and
the honest group-aware split — and prints the gap between them. Run it once, predict the gap first,
and record both numbers in `RESULTS.md`.

## 5. Deliverables
- A short **report** with the **evaluation mode** on every number, and an honest account of which
  subjects were decodable.
- `predictions.csv` on the held-out split for hold-out evaluation.
- A cross-track showcase slot: "spatial filtering was the difference."
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Beat the supplied baseline **honestly, per mode**. **Fit CSP/any spatial filter inside the fold** —
  fitting on all data is a leak that fakes high scores. State the split unit with every number. Report the metric **with its spread** across subjects (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: near-chance naive baseline (needs CSP), spatial-filter leakage if fit outside folds,
brutal cross-subject non-stationarity, and the T1/T2 = left/right label semantics.
