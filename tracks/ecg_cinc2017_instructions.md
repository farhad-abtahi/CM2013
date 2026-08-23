# Track instructions — ECG rhythm / AF, PhysioNet CinC-2017

*Attach the dataset card (`ecg_cinc2017_card.md`). Adapter: `ecg_cinc2017.py`.*

> **Before you build — background & literature review.** Refresh the methods and do a short literature
> review of the application domain using **[`BACKGROUND_MAP.md`](BACKGROUND_MAP.md)** (the "ECG rhythm /
> AF" section): it lists the **book sections** (Pan–Tompkins §9.9, HRV §7.11, notch §9.6 …), the
> **literature-review anchors** (Challenge paper, Pan & Tompkins 1985, HRV Task Force), and the
> **guiding questions** your report must answer to *motivate* your design (course outcome **L5**).

## 1. The problem
From a **single-lead ECG** (300 Hz, ~10–60 s), classify the rhythm as **Normal / AF / Other / Noisy** —
the task a wearable rhythm monitor must do. One label per recording.

## 2. Your data (see the dataset card)
- Dataset: **PhysioNet/CinC-2017** (open, direct download via `wfdb` — **no agreement**).
- Signals: single-lead ECG @ 300 Hz, variable length. Labels: N / A / O / ~ (strongly imbalanced).
- **Split unit (leakage): record** — one label per recording; GroupKFold on record id.
- Smoke subset: `A00001, A00004, A00006`. Evaluation mode: **new-record**.

## 3. What you are given (do not rebuild these)
- The modular pipeline + a working baseline (`tracks.adapter.default_baseline`).
- The adapter (`ecg_cinc2017.py`) with `smoke()`, `preprocess()` (identity by default — hoisting the
  band-pass out of `_rpeaks` is one option), `extract_features()` (QRS detection → R–R/HRV + a
  signal-quality index), a stratified `download()` and a `wfdb` `load()`.
- The shared leakage-safe **evaluator** (GroupKFold by record).
- The **seven pipeline stages**, separable on the adapter (rubric Criterion 1): `download`/`load`/`smoke` → `preprocess()` → `extract_features()` → `select_features()` → `baseline()` → `infer()` → `report()`. Selection is fit **inside** every CV fold; `infer()` is the frozen, no-refit path used for `predictions.csv`.
- The **reporting module** (`report.py`): `summarize_results()` leads with the confusion matrix, then the primary metric **with its spread across folds** — the shape §16.3 requires. `evaluate()` returns `per_group` / `per_fold` / `spread` / `summary` for exactly this.

## 4. What you must do (iterations)
1. **Run the baseline** on the smoke subset, then real records; report macro-F1, κ, per-class recall.
2. **Improve the signal processing / features**: robust QRS detection under noise, better R–R
   irregularity/HRV features, and an explicit **signal-quality** feature set for the Noisy class.
3. **Improve the model / selection** (validated inside folds); handle the class imbalance deliberately.
4. **Analyse failures** — where do A↔O confuse, and which records are mislabelled "Noisy"?

Those steps are a *menu, not a march*: which preprocessing, which features, whether to select
features at all, and which learner to climb to are decisions the scaffold deliberately leaves
open (see `preprocess()` and `make_selector()` / `default_baseline()` in `adapter.py` for the options and their
trade-offs). The grade is on the quality of the reasoning, not on matching one blessed recipe.

## 4c. The design-decision menus (stages 2-5) — what the scaffold offers, and the trade-off

Everything here is a **menu, not a recipe**. `tracks/adapter.py` ships each option *with its
trade-off* and no blessed answer; the rubric grades the reasoning (criteria 2 and 5), not the
choice. The notebook's "Decision points on this track" section runs several of them side by side
so you can watch the numbers move.

### Stage 2 — preprocessing · `cfg["preprocess"]`

`"none"` (default) · `"bandpass"` · `"wavelet"` · `"denoise"`. The shipped default lets `_rpeaks()`
own its own band-pass and lets every other feature see the raw strip; `"bandpass"` hoists the filter
into `preprocess()` so it becomes a reportable, swappable stage.

**The catch that makes this track distinctive: denoising can destroy the label.** Two of the eleven
shipped features (the QRS/high-frequency ratio and the 49-51 Hz powerline power) exist to *measure*
corruption, and the `~` class **is** the low-quality class. Any cleaning applied before those
features are computed makes Noisy harder to detect, not easier. Separately, the **QRS complex is a
broadband ~80 ms transient**, so a fixed band-pass rounds its edges, widens its apparent duration and
shifts the peak your R-R intervals are measured from — which is exactly the case `"wavelet"` exists
to handle. Measure it both ways and report which you chose.

### Stage 2 — preprocessing, **indexed by noise type** · `cfg["preprocess"] = "denoise"`

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

Three of the eleven features are band powers. The 49-51 Hz powerline feature is the narrowest band in
the whole scaffold — 2 Hz wide — so it is the one most punished by a coarsened estimator and most
helped by `"multitaper"` on a short strip. Conversely `"ar"` will happily place a pole near 50 Hz and
report a confident, smooth, possibly fictitious hum peak: §7.9's over-fitting warning arriving as a
feature value. A 20-30 s strip at 300 Hz is long enough for Welch to be comfortable, so expect small
movements — say so honestly if that is what you measure.

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
- A short **report** reading your macro-F1 against the **Challenge winners (~0.83)**. Justify choices.
- `predictions.csv` (`record,label`) on the held-out split for hold-out evaluation.
- A cross-track showcase slot: "Pan–Tompkins event detection, on ECG."
- A **results log**: copy `results_log_TEMPLATE.md` into your team repo as `RESULTS.md` and add one row per iteration — what changed and why, the metric **with its spread**, whether it beat the previous iteration (or why you kept it anyway), and the commit. **This file is graded** (rubric Criterion 9, 3 pts) and it asks specifically for at least one decision you went back and **revised because of a downstream result** — the notebook's "Decision points on this track" section has a symptom → stage table to diagnose from, and prints an A/B of several options so you can see the numbers move.

## 6. Rules
- Beat the supplied baseline **honestly**. State the **split unit (record)** with every number. Never
  report smoke/CI numbers as results. Report the metric **with its spread** across folds (`rep["summary"]`), not a lone pooled number. Grading: [`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md) (team) + [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (individual).

## 7. Known pitfalls
See the card: single lead + variable length, the **Noisy class is a signal-quality problem** (not a
rhythm), strong imbalance (score macro-F1/κ), and a bad QRS detector wrecks every downstream feature.
