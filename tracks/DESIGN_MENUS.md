# Design-decision menus — stages 2–5

**Shared reference for every track.** This file describes the options you can choose between and
what each one costs you. Your own track's instructions file covers what these choices mean *on your
signal* — read that first, and come back here when you are deciding between options.

Everything here is a **menu, not a recipe**. `tracks/adapter.py` ships each option *with its
trade-off* and no blessed answer; the rubric grades the reasoning (criteria 2 and 5), not the choice
you land on. Your notebook's "Decision points on this track" section runs several of them side by
side so you can watch the numbers move.

**The menus are shortcuts, not the boundary.** They cover common options so you can compare quickly;
you are free — and expected — to add your own methods at any stage, and feature extraction and the
classifier (which have no menu here) are meant to be your own work. At each stage, compare two or three
approaches, one change at a time. See *The code is a starting point, not a lab instruction* in
`START_HERE.md` for what is expected and what stays fixed (the validation scheme, below).

---

## Stage 2 — preprocessing, indexed by noise type · `cfg["preprocess"] = "denoise"`

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

---

## Stage 3 — spectral estimation · `cfg["spectral_method"]`, `cfg["ar_order"]`, `cfg["mt_bandwidth"]`

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

**What it costs on your track** — which bands are narrow enough to be punished by a coarsened
estimator, and how much movement to expect — is in your track's own instructions.

---

## Stage 4 — feature selection · `cfg["select"]`, `cfg["select_k"]`, `cfg["select_C"]`

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

---

## Stage 5 — class imbalance · `cfg["imbalance"]`, `cfg["threshold"]`

`"none"` · `"balanced"` (default) · `"balanced_subsample"` · `"resample"` · **`"smote"`** ·
**`"adasyn"`** · `"threshold"`.

**`"smote"`/`"adasyn"`** synthesise minority rows by **interpolating** between a real sample and one
of its k nearest minority neighbours, rather than duplicating exact rows (`"resample"`) or
reweighting the loss (`"balanced"`). The distinction to state in your report: `"balanced"` never adds
a row; `"resample"` adds exact copies, so the minority region gets heavier but not one millimetre
wider; `"smote"` adds **new points in feature space**, so the region genuinely expands and the
boundary is pushed rather than weighted.

That expansion is the whole benefit *and* the whole danger. A synthetic point halfway between two
real samples is a claim about feature space, **not about physiology** — interpolate between two
different subjects, or between two adjacent classes that a clinician would never blend, and you have
manufactured a recording that does not exist. On these small, heterogeneous cohorts that risk is
live, not a footnote. If you use it, say what a synthetic minority sample *means* on your track, and
be honest if the answer is "nothing physiological".

Both are fold-safe (`adapter.SMOTEd` fits inside `fit` only, never at predict time) and both need the
optional `imbalanced-learn` package; every other option is sklearn-only. On tiny folds the
`k_neighbors` requirement is adapted downward automatically, and a minority class with fewer than two
members falls back to plain duplication with a loud warning rather than crashing mid-CV.

---

## Not a menu — the validation scheme

The leakage-safe split (LOSO / GroupKFold on your track's split unit) is **not** a design choice, and
there is deliberately no config key to turn it off. It is the only number that counts for your grade,
and `evaluate()` enforces it on every fold. Your notebook's decision-points section contains a
**required one-time demonstration** that scores the data twice — a naive random stratified split and
the honest group-aware split — and prints the gap between them. Run it once, predict the gap first,
and record both numbers in `RESULTS.md`.
