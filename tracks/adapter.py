"""
tracks.adapter — the shared contract for every capstone track.

A "track" = a real public dataset + a small adapter. The adapter owns the
DOMAIN-SPECIFIC work (download, load, preprocessing, feature construction — i.e.
the signal processing the students do) and declares rich metadata (a dataset card
in code). The BASE owns the shared DISCIPLINE (subject-independent validation,
the honest metric panel, leakage guards) so every track is evaluated the same way.

THE SEVEN MODULES (book §16.2, rubric Criterion 1). Each is a separate, callable,
testable stage — never one opaque blob:

  | # | Stage           | Method                          | Who writes it |
  |---|-----------------|---------------------------------|---------------|
  | 1 | data loading    | `download()` / `load()` / `smoke()` | supplied  |
  | 2 | preprocessing   | `preprocess(rec, cfg)`          | **you**       |
  | 3 | feature extract | `extract_features(rec, cfg)`    | **you**       |
  | 4 | feature select  | `select_features(X, y, cfg)`    | **you** (options supplied) |
  | 5 | classification  | `baseline()` / your own `clf`   | **you**       |
  | 6 | inference       | `infer(model, rec, cfg)`        | supplied (frozen) |
  | 7 | reporting       | `report(rep)` -> `report.py`    | supplied + **your** reading |

`evaluate()` threads 2-5 together *inside every CV fold* (so selection and scaling
are never fit on test data, §16.2/Chapter 12); `infer()` is the separate,
train-free path that applies an already-fit pipeline to new data.

Design rules (from the review):
  * synthetic `smoke()` path so CI / offline / Colab-sanity always runs green;
  * a SUPPLIED baseline model — students improve it, never start from a blank page;
  * an explicit SPLIT_UNIT (the leakage unit) enforced on every fold;
  * metadata rich enough to render a dataset card automatically;
  * every result reported WITH ITS SPREAD across groups/folds, never a lone
    pooled number (§16.3 "definition of done", point 2).

NOTE ON CHOICE. This file supplies *options with trade-offs*, not one blessed
answer. `preprocess()` and `select_features()` default to pass-through so the
baseline runs; the alternatives (and when each is the better bet) are documented
on the stage itself. Picking one, and writing down why, is your job — and is what
the rubric's "design decisions are justified" line grades.

**BUT ONLY REAL CHOICES.** The six tracks do genuinely different amounts of
signal processing, so not every menu exists everywhere: Sleep-EDF and ECG run the
full Chapter 8 denoise menu and integrate band powers from a Chapter 7 estimator;
BCI has the spectral decision but an identity `preprocess()`; HAR has one
preprocessing knob (gravity) and no PSD-based feature; EMG and CTG have neither.
Each track therefore declares what it consumes in `SUPPORTED_CFG_KEYS` (on top of
`BASE_CFG_KEYS`, the stage-4/5 and CV keys the base class owns and every track
supports). Two consequences:

  * `tools/build_track_notebooks.py` renders only the menus a track really has,
    so a student is never offered an option with no consequence;
  * a cfg key the track never reads raises `UnsupportedCfgKey` instead of being
    silently ignored — a no-op wearing the costume of a design decision teaches
    the opposite of what these menus are for. Implemented a stub yourself?
    `track.declare_cfg_keys("csp_components", ...)` makes its knobs first-class.
    The check runs on **every** cfg resolution, so editing `track.cfg` in place
    after construction is caught too, not just the constructor argument.

**CONFIG PARITY BETWEEN TRAINING AND INFERENCE IS ENFORCED.** `train_baseline()`
records the resolved cfg on the `FittedModel` as an **immutable snapshot**;
`infer()` and `write_submission()` rebuild test features from *that* cfg when
none is passed, and a cfg you *do* pass is treated as a partial override of it —
the keys you name are applied on top of the training cfg, the rest keep their
training values, and only a named, feature-affecting key that genuinely differs
is refused. Predicting from a different pipeline than the one you validated is a
failure no metric in the report can reveal, because the report was computed on
the other pipeline.
"""
from __future__ import annotations
import copy
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
# sklearn is imported at module level for the two base classes only — the stage-5
# wrappers below MUST inherit from `BaseEstimator` (they are meta-estimators, and
# that is what makes `clone()`, `get_params(deep=True)` and nested `set_params`
# such as `estimator__clf__max_depth=5` work). Every other sklearn import in this
# file stays lazy, inside the function that needs it.
from sklearn.base import BaseEstimator, ClassifierMixin   # noqa: E402

# make the shared bsp package (and this flat tracks/ dir) importable from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "src"), os.path.join(_HERE, "..", "src", "bsp")):
    _p = os.path.normpath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bsp import metrics as M                     # noqa: E402
from bsp import notebook_checks as C             # noqa: E402


# --------------------------------------------------------------- metadata
@dataclass
class TrackMeta:
    name: str
    dataset: str
    dataset_version: str
    license: str
    citation: str
    url: str
    signals: list
    task_type: str                  # multiclass | multilabel | binary | regression
    classes: list
    split_unit: str                 # subject | patient | recording | session
    default_metrics: list           # e.g. ["cohens_kappa", "macro_f1"]
    smoke_test_records: list        # tiny public subset ids for CI/Colab
    expected_runtime: str
    dsp_focus: str
    access: str = "open-direct"     # this repo only ships agreement-free datasets
    eval_modes: tuple = ("new-subject",)   # add "within-subject" where relevant
    difficulty: int = 3             # 1..5, feeds the weighted rubric
    submission_granularity: str = "epoch"  # "epoch" (row per epoch) | "record" (row per record)


@dataclass
class Recording:
    """One subject/patient/recording. `epochs` is a dict of per-epoch signal
    arrays (n_epochs x n_samples); `labels` is per-epoch; `group` is the leakage
    unit id; `fs` the sampling rate."""
    group: str
    fs: float
    epochs: dict                    # {channel: ndarray[n_epochs, n_samples]}
    labels: np.ndarray
    meta: dict = field(default_factory=dict)

    def replace_epochs(self, epochs: dict) -> "Recording":
        """Return a copy carrying new (e.g. cleaned) epoch arrays — the natural
        return value of a `preprocess()` implementation."""
        return Recording(group=self.group, fs=self.fs, epochs=epochs,
                         labels=self.labels, meta=dict(self.meta))


# --------------------------------------------------------------- config
#: The shared, config-driven defaults (§16.6: every knob in one place, no magic
#: numbers scattered through the code). Override per run:
#:     track = SleepEDFTrack(cfg={"select": "anova", "select_k": 15})
#: or per call: ``track.evaluate(X, y, g, cfg={"select": "tree"})``.
DEFAULT_CFG = {
    "seed": 0,
    "preprocess": "none",     # "none" | "bandpass" | "wavelet" | "denoise" — see preprocess()
    # ---- stage 2, indexed by NOISE TYPE (Ch. 8; used when preprocess="denoise") ----
    "impulsive": None,        # None | "median"                       — §8.7 spikes/pops
    "baseline": None,         # None | "highpass" | "detrend" | "wavelet"  — §8.8 wander
    "powerline": None,        # None | "notch" | "adaptive" | "spectral"   — §8.6 mains
    "broadband": None,        # None | "movavg" | "savgol" | "wavelet" | "gaussian" — §8.4
    "hp_fc": 0.5,             # high-pass corner (Hz) when baseline="highpass"
    "f0": 50.0,               # mains frequency (Hz); 60.0 in the Americas
    "q": 30.0,                # notch quality factor
    "win": 5,                 # smoothing / median window in SAMPLES
    "causal": False,          # True -> single forward lfilter pass (deployable, has delay)
    # ---- stage 3, spectral estimation (Ch. 7) ----
    "spectral_method": "welch",   # "periodogram" | "bartlett" | "welch" | "multitaper" | "ar"
    "ar_order": 16,           # AR model order when spectral_method="ar" (§7.9: the risky knob)
    "mt_bandwidth": 4.0,      # time-bandwidth product NW when spectral_method="multitaper"
    "select": "none",         # "none" | "variance" | "anova" | "mutual_info" | "tree" | "lasso"
    "select_k": 20,           # k for the filter selectors (clamped to n_features, with a note)
    "select_C": 1.0,          # inverse L1 strength when select="lasso" (smaller C = sparser)
    "imbalance": "balanced",  # "none" | "balanced" | "balanced_subsample" | "resample" | "smote" | "adasyn" | "threshold"
    "threshold": 0.5,         # decision threshold on the minority class when imbalance="threshold"
    "smote_k": 5,             # k-neighbours for SMOTE/ADASYN (auto-reduced on tiny folds)
    "wavelet": "db4",         # mother wavelet when preprocess="wavelet"
    "wavelet_level": None,    # decomposition depth (None = data-driven, capped at 5)
    "cv_max_splits": 5,       # GroupKFold splits once there are many groups
    "loso_max_groups": 12,    # <= this many groups -> leave-one-group-out
}

#: Config keys the BASE class consumes on every track: the seed, the stage-4
#: selection menu, the stage-5 imbalance menu, and the CV geometry. These work
#: identically on all six tracks because `TrackAdapter` — not the track — owns
#: stages 4, 5 and the validation scheme.
BASE_CFG_KEYS = frozenset({
    "seed",
    "select", "select_k", "select_C",
    "imbalance", "threshold", "smote_k",
    "cv_max_splits", "loso_max_groups",
})

#: The stage-2 keys `denoise()` / `bandpass_notch()` / `wavelet_denoise()` read.
#: A track declares these only if its own `preprocess()` actually forwards them —
#: Sleep-EDF and ECG do; BCI, EMG and CTG ship an identity/absent `preprocess()`,
#: and HAR exposes one knob of its own instead.
DENOISE_CFG_KEYS = frozenset({
    "preprocess",
    "impulsive", "baseline", "powerline", "broadband",
    "hp_fc", "f0", "q", "win", "causal", "order",
    "wavelet", "wavelet_level", "wavelet_mode", "threshold_mode", "level",
    "poly_order", "savgol_order", "harmonics", "mu",
    "notch",
})

#: The stage-3 keys `make_spectral_estimator()` reads. Only declared by tracks
#: whose features are actually integrated from a PSD (sleep, ECG, BCI).
SPECTRAL_CFG_KEYS = frozenset({"spectral_method", "ar_order", "mt_bandwidth"})

#: Of those, the keys that touch ONLY stages 4-5 (selection / classifier / CV)
#: and therefore cannot change the feature matrix. `infer()` uses this to decide
#: whether a caller-supplied cfg genuinely conflicts with the one the model was
#: trained under (see `TrackAdapter.infer`).
NON_FEATURE_CFG_KEYS = frozenset(BASE_CFG_KEYS)


class _ResolvedCfg(dict):
    """A cfg that has already been merged (DEFAULT_CFG + adapter cfg + call cfg)
    and validated against the track's declared capabilities.

    It is a plain `dict` for every practical purpose; the subclass exists only so
    `TrackAdapter._cfg()` can tell "a user handed me three override keys" apart
    from "an internal caller is passing the fully-expanded dict back down the
    stack". Without that distinction the capability check would fire on
    DEFAULT_CFG's own keys the moment one stage forwarded its cfg to the next."""


class _FrozenCfg(_ResolvedCfg):
    """A resolved cfg that **cannot be edited after the fact**.

    This is what `FittedModel.cfg` holds. The train → infer parity guarantee is
    only worth something if the recorded training config is a *historical fact*:
    `infer()` and `write_submission()` rebuild test features from `model.cfg`
    and trust it to describe the pipeline the classifier was actually fit on. A
    plain dict lets `model.cfg["gravity"] = "none"` rewrite that history, and the
    predictions then change silently while every printed number still refers to
    the old pipeline — exactly the invisible failure the guarantee exists to
    prevent. So every mutating dict method raises `TypeError` instead.

    Reading is completely normal: `model.cfg["select_k"]`, `dict(model.cfg)`,
    iteration, `in`, `.get()`, `.items()` all behave as usual, and the object
    pickles and deep-copies (to another frozen cfg) so a model can be trained in
    one process and scored in another. To run under a *different* config, pass it
    explicitly to `infer()` — which then checks it against this one — or train a
    new model; do not edit the record of what happened."""

    _WHY = ("FittedModel.cfg records the config this model was TRAINED under, so "
            "inference can rebuild the same features. Editing it in place would "
            "change what infer() computes while leaving every validated number "
            "describing the old pipeline — a silent train/infer mismatch. Pass an "
            "explicit cfg to infer() (it is checked against this one), or re-train.")

    def _frozen(self, *_a, **_k):
        raise TypeError(f"{type(self).__name__} is read-only. {self._WHY}")

    __setitem__ = _frozen
    __delitem__ = _frozen
    __ior__ = _frozen
    update = _frozen
    setdefault = _frozen
    pop = _frozen
    popitem = _frozen
    clear = _frozen

    # -- copying / pickling: dict.__init__ fills the C-level storage directly, so
    #    it does not go through the blocked __setitem__ above.
    def __copy__(self):
        return type(self)(dict(self))

    def __deepcopy__(self, memo):
        return type(self)({k: copy.deepcopy(v, memo) for k, v in self.items()})

    def __reduce__(self):
        return (type(self), (dict(self),))


def _freeze_cfg(cfg) -> _FrozenCfg:
    """Deep-copy `cfg` and seal it — see `_FrozenCfg`. The deep copy matters as
    much as the seal: without it a nested value (a band written as a list, say)
    would still be reachable and editable through the caller's own dict."""
    if isinstance(cfg, _FrozenCfg):
        return cfg
    return _FrozenCfg(copy.deepcopy(dict(cfg or {})))


#: One line per cfg key saying WHICH stage would have consumed it — so the error
#: a student hits explains the capability gap instead of just refusing.
#: Tracks may add/override entries with a class-level ``CFG_KEY_HINTS`` dict.
_CFG_KEY_HINTS = {
    "preprocess": "selects the stage-2 recipe (bandpass / wavelet / denoise). This track's "
                  "preprocess() ignores it — it is either an identity stub you are meant to fill "
                  "in, or it exposes a different knob (see the track's preprocess docstring).",
    "impulsive": "a Chapter 8 stage-2 remedy, only read when preprocess='denoise'.",
    "baseline": "a Chapter 8 stage-2 remedy, only read when preprocess='denoise'.",
    "powerline": "a Chapter 8 stage-2 remedy, only read when preprocess='denoise'.",
    "broadband": "a Chapter 8 stage-2 remedy, only read when preprocess='denoise'.",
    "hp_fc": "the high-pass corner for the stage-2 baseline='highpass' remedy.",
    "f0": "the mains frequency for the stage-2 powerline remedies.",
    "q": "the notch quality factor for the stage-2 powerline='notch' remedy.",
    "win": "the smoothing / median window for the stage-2 broadband and impulsive remedies.",
    "causal": "switches the stage-2 IIR filters from filtfilt to a single forward lfilter pass; "
              "this track applies no IIR filter of its own.",
    "wavelet": "the mother wavelet for the stage-2 wavelet options.",
    "wavelet_level": "the DWT depth for preprocess='wavelet'.",
    "wavelet_mode": "soft/hard thresholding for preprocess='wavelet'.",
    "notch": "a stage-2 mains notch; this track applies no filter of its own.",
    "spectral_method": "chooses the Chapter 7 PSD estimator that band-power features are "
                       "integrated from. This track computes no PSD-based band-power feature, "
                       "so there is no spectrum for the estimator to change.",
    "ar_order": "the Burg model order, read only when spectral_method='ar'.",
    "mt_bandwidth": "the multitaper time-bandwidth product NW, read only when "
                    "spectral_method='multitaper'.",
    "gravity": "the HAR-specific gravity/body-acceleration split.",
    "gravity_fc": "the HAR gravity high-pass corner, read only when gravity='highpass'.",
    "eeg_band": "the Sleep-EDF band-pass edges.",
    "ecg_band": "the CinC-2017 band-pass edges.",
    "filter_channels": "which channels the Sleep-EDF stage-2 filter touches.",
    "rebuild": "a NOTEBOOK-ONLY flag used by the A/B cell to decide whether stages 2-3 must "
               "re-run; pop it before passing the dict to the adapter.",
}


class UnsupportedCfgKey(ValueError):
    """Raised when a cfg key is set on a track whose stages never read it.

    Silently accepting-and-ignoring an option is worse than failing: the whole
    point of the decision menus is that a choice has a measurable consequence,
    and an option that provably does nothing teaches the opposite lesson. See
    `TrackAdapter.supported_cfg_keys`."""


# --------------------------------------------------------------- stage 4 options
class PassThrough:
    """The identity selector: keep every feature. A perfectly defensible choice
    when the feature count is already small relative to the number of samples —
    but it should be a *choice you made*, not one nobody noticed."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        n = getattr(self, "n_features_in_", 0)
        return np.arange(n) if indices else np.ones(n, bool)


def make_selector(kind: str = "none", k: int = 20, seed: int = 0, threshold="median",
                  C: float = 1.0):
    """Feature-selection stage (module 4) — **an unfitted transformer you must fit
    inside the training fold only** (§16.2: "never before").

    Five honest options, none of them "the right answer":

    | `kind`          | Family   | Picks features by | Good when | Watch out |
    |-----------------|----------|-------------------|-----------|-----------|
    | `"none"`        | —        | keeps everything  | few features vs. many samples; you want the model to arbitrate | irrelevant features dilute distance-based learners (k-NN, SVM) |
    | `"variance"`    | filter   | throwing away near-constant columns | cheap sanity pass before anything else | says nothing about relevance to `y` |
    | `"anova"`       | filter   | univariate F-test vs. the label | fast, stable, tiny folds | blind to feature *interactions*; assumes roughly linear separation per feature |
    | `"mutual_info"` | filter   | non-linear dependence with the label | non-monotonic relations (e.g. a band power good only in a middle range) | noisier estimate; needs more samples than ANOVA |
    | `"tree"`        | embedded | impurity importance from a forest | interactions matter; you already use trees | importance is biased toward high-cardinality/continuous features; correlated features share credit |
    | `"lasso"`       | embedded | L1-penalised linear SVM: coefficients driven to **exactly zero** | you want genuine sparsity and a short, defensible feature list; correlated-but-redundant features should be *eliminated*, not merely down-weighted | judges through a **linear** lens — wrong when the real relationship is non-linear, and among two correlated features L1 keeps one essentially arbitrarily |

    **`"tree"` vs `"lasso"` — the contrast worth understanding.** Both are
    *embedded*: the selection falls out of a model that saw all the features at
    once, so both can notice interactions a univariate filter is blind to. But
    they behave oppositely on correlated features. A forest **splits the credit**:
    two near-duplicate band powers each get roughly half the importance, so
    neither looks decisive and both survive a `"median"` threshold. L1 does the
    opposite — it **picks one and zeroes the other**, because a second copy of an
    already-used feature buys no extra likelihood and costs extra penalty. So
    `"tree"` gives you a ranking with redundancy intact; `"lasso"` gives you a
    genuinely short list, at the price of assuming roughly linear separability
    and of an arbitrary choice inside each correlated cluster (re-fit on another
    fold, it may keep the *other* one — check the stability before you claim the
    list is "the" feature set).

    `C` is the inverse L1 strength for `"lasso"` (`cfg["select_C"]`): smaller C =
    stronger penalty = fewer surviving features. It is a real knob, so report it.
    The L1 model is fit on **standardised** features inside the selector, because
    an L1 penalty is scale-dependent — without that, "selection" would just keep
    whatever features happen to be measured in large units.

    Trade-off in one line: filters are fast and stable but judge each feature
    alone; embedded methods see interactions but inherit the model's own biases.
    Whichever you pick, say in the report *why* — and show what it changed.
    """
    kind = (kind or "none").lower()
    if kind in ("none", "off", "passthrough"):
        return PassThrough()
    if kind == "variance":
        from sklearn.feature_selection import VarianceThreshold
        return VarianceThreshold(threshold=0.0)
    if kind == "anova":
        from sklearn.feature_selection import SelectKBest, f_classif
        return SelectKBest(f_classif, k=k)
    if kind in ("mutual_info", "mi"):
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        return SelectKBest(mutual_info_classif, k=k)
    if kind in ("tree", "model", "importance"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        return SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=seed),
            threshold=threshold)
    if kind in ("lasso", "l1"):
        from sklearn.feature_selection import SelectFromModel
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        # LinearSVC(penalty="l1", dual=False) is sklearn's own documented
        # L1-based feature selector, and unlike LogisticRegression(solver=
        # "liblinear") it handles the multiclass tracks without the deprecated
        # one-vs-rest path (removed in sklearn 1.8). Same liblinear machinery,
        # same exact-zero sparsity, squared-hinge instead of log loss.
        l1 = Pipeline([("scale", StandardScaler()),
                       # max_iter is generous on purpose: at the sklearn default
                       # liblinear does not converge on the wider feature sets
                       # (EMG's 50, BCI's 28) and the harness would surface a
                       # convergence warning that is about the solver budget, not
                       # about your data. Cost is ~1 s per fit.
                       ("l1", LinearSVC(penalty="l1", dual=False, C=float(C),
                                        random_state=seed, max_iter=50000))])
        # threshold=1e-5, not "median": the whole point is to keep exactly the
        # features whose coefficient is not zero, rather than half of them.
        return SelectFromModel(l1, threshold=1e-5,
                               importance_getter="named_steps.l1.coef_")
    raise ValueError(f"unknown selector {kind!r}; see make_selector.__doc__")


# --------------------------------------------------------------- stage 2 helper
def bandpass_notch(rec: "Recording", band=None, notch=None, order=4, channels=None,
                   notch_q=30.0, causal=False) -> "Recording":
    """An OPTIONAL band-pass (+ optional mains notch) you can call from your own
    `preprocess()`. Supplied so the plumbing is not the exercise — the *choices*
    are: which band, what order, whether a notch is warranted at all.

    Trade-offs the book already made you meet (Ch. 8-10): a tighter band removes
    more noise but can amputate the signature you are about to measure (a 0.5-40 Hz
    EEG band keeps beta intact; 0.5-30 Hz would not); a higher order is a sharper
    edge but rings harder around transients; a 50/60 Hz notch is free when the hum
    is narrow-band and harmful when your feature lives near it. In §8.11's words,
    **"the noise's signature chooses the tool"** — so identify before you filter,
    and write down which signature you saw. `denoise()` in this module lays the
    same decision out as a menu indexed by noise type rather than by technique.

    ⚠️ **CAUSALITY — read this before you claim a real-time result.**
    The default (`causal=False`) uses `scipy.signal.filtfilt`, which filters the
    epoch forwards and then backwards. That is what makes it **zero-phase**: no
    group delay, no smeared landmark positions. It is also, unavoidably,
    **non-causal** — the output at sample *n* depends on samples *after* n, i.e.
    on the future. There is no way to run it on a sample as it arrives.

    * **Offline / retrospective analysis — which is what this entire scaffold
      does** (whole epochs are already on disk before anything is computed):
      `filtfilt` is the *right* tool, and staying with it is a defensible,
      reportable choice. It is not removed here and you are not expected to
      avoid it.
    * **A real-time / streaming deployment — the framing of the BCI motor-imagery
      and EMG gesture-control tracks**: a `filtfilt` pipeline is **not
      deployable as written**. A live BCI or a myoelectric prosthesis sees a
      sample stream, not a file. Reproducing these numbers on-line needs a
      **causal** filter — `scipy.signal.lfilter` with a single forward pass (and
      the group delay it costs you), a causal IIR design, or a short linear-phase
      FIR whose fixed delay fits your latency budget.

    Passing `causal=True` here switches to exactly that single forward
    `lfilter` pass, so you can *measure* what the honesty costs instead of
    asserting it: the phase distortion, the added latency, and the metric gap
    between the offline and the deployable pipeline. Reporting both numbers, and
    saying which one your deployment claim rests on, is worth marks; quoting an
    offline zero-phase number as if it were a real-time one is the same class of
    mistake as quoting a leaky split.

    Returns a NEW Recording; never mutates the input.
    """
    from scipy.signal import butter, filtfilt, iirnotch, lfilter
    apply = (lambda b, a, x: lfilter(b, a, x, axis=-1)) if causal else \
            (lambda b, a, x: filtfilt(b, a, x, axis=-1))
    nyq = rec.fs / 2.0
    out = {}
    for ch, arr in rec.epochs.items():
        a2 = np.asarray(arr, float)
        if channels is not None and ch not in channels:
            out[ch] = a2
            continue
        x = a2
        if band is not None:
            lo, hi = band
            wn = [max(lo / nyq, 1e-6), min(hi / nyq, 0.999)]
            b, a = butter(order, wn, "band")
            x = apply(b, a, x)
        if notch is not None:
            b, a = iirnotch(notch / nyq, notch_q)
            x = apply(b, a, x)
        out[ch] = x
    return rec.replace_epochs(out)


def wavelet_denoise(rec: "Recording", wavelet: str = "db4", level=None,
                    threshold_mode: str = "soft", channels=None,
                    threshold=None) -> "Recording":
    """An OPTIONAL **wavelet denoiser** — the second supplied stage-2 recipe, and a
    genuinely different bet from `bandpass_notch()`. Multi-level DWT decomposition
    (`pywt.wavedec`) → thresholding of the detail coefficients → reconstruction
    (`pywt.waverec`), per channel, along the last (time) axis.

    **What it actually teaches (Ch. 5).** A fixed-band IIR filter makes one
    decision — *this frequency range survives, that one does not* — and applies it
    identically to every instant of the recording. That is exactly right when the
    interference is **narrow-band and stationary**: 50/60 Hz mains hum sits in one
    place all night, so a notch removes it and nothing else. It is exactly wrong
    when the interference is **broadband and transient**: a motion artifact, an
    electrode pop, a swallow, a baseline step. Those spread across the whole
    spectrum for a fraction of a second, so no band you can choose contains them —
    and the band you would have to remove takes the signal with it.

    Wavelet thresholding attacks the same problem in the other order. It asks not
    "which frequencies?" but "which *coefficients are too large to be noise*?",
    localised in time as well as scale. A sharp, brief, high-amplitude event keeps
    a few big coefficients and survives; low-level broadband noise is spread thinly
    across many small ones and is thresholded away. The practical consequence for
    these tracks:

    | | `bandpass_notch` (fixed-band IIR) | `wavelet_denoise` (DWT thresholding) |
    |---|---|---|
    | best against | **narrow-band, stationary** interference — 50/60 Hz hum, a defined out-of-band range | **non-stationary transients** — motion artifact, electrode pops, baseline drift/steps |
    | what it costs | rings around transients; a tight band can amputate the signature you came to measure | shrinks *everything*, so a too-high threshold erodes genuine low-amplitude structure |
    | effect on sharp morphology | a fixed band **blurs** the landmarks — an ECG **QRS** complex and an EMG action potential are broadband, brief events, so band-limiting rounds their edges and shifts their apparent width | preserves them: the sharp edge is carried by a few large fine-scale coefficients that survive thresholding |
    | effect on rhythmic content | preserves in-band rhythms exactly | can slightly attenuate genuine low-amplitude rhythms (weak sleep **spindles**, a shallow mu rhythm) |
    | the honest use | you can *name* the interfering band | you can see the artifact in the time series but cannot name a band that holds it |

    On Sleep-EDF that trade is live in both directions: **K-complexes and
    spindles** are precisely the sharp/short structures a hard band-pass smears,
    while over-aggressive thresholding is precisely what erases a weak spindle. On
    CinC-2017 ECG the **QRS** complex is the landmark every R–R and HRV feature
    depends on, and motion artifact is the dominant nuisance — the `~` class *is*
    the signal-quality class. Neither answer is free, which is why both are on the
    menu and neither is the default.

    Parameters
    ----------
    wavelet : str
        Mother wavelet (`"db4"` default — a common ECG/EEG choice; `"sym4"`,
        `"coif3"`, `"db6"` are equally defensible). The choice matters: a wavelet
        whose shape resembles the feature you care about needs fewer coefficients
        to represent it, so that feature survives thresholding better.
    level : int | None
        Decomposition depth. `None` = `pywt.dwt_max_level`, capped at 5 (deeper
        levels on a short epoch are dominated by boundary effects).
    threshold_mode : {"soft", "hard"}
        `"soft"` shrinks every coefficient toward zero (smoother output, slight
        amplitude bias); `"hard"` keeps or kills (preserves peak amplitude, can
        leave visible discontinuities). Say which you used.
    threshold : float | None
        `None` = the universal / VisuShrink rule, `σ·sqrt(2·ln N)` with σ robustly
        estimated per epoch from the finest detail band as `median(|cD1|)/0.6745`.
        Per-epoch estimation is deliberate: it is fit from the epoch's own noise
        floor and uses **no labels and no other recording**, so it cannot leak.
        Pass a number to override it — and then justify the number.
    channels : iterable | None
        Restrict to these channels; the rest pass through untouched.

    Returns a NEW Recording; never mutates the input.
    """
    try:
        import pywt
    except ImportError as e:                                # pragma: no cover
        raise ImportError(
            "wavelet_denoise needs PyWavelets — `pip install 'PyWavelets>=1.6'` "
            "(it is already in requirements.txt; Chapter 5 uses it too).") from e

    def _one(x):
        x = np.asarray(x, float)
        n = x.shape[-1]
        L = level
        if L is None:
            L = min(5, pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len))
        if L < 1:
            return x                                        # epoch too short to decompose
        coeffs = pywt.wavedec(x, wavelet, level=L, axis=-1)
        thr = threshold
        if thr is None:
            d1 = coeffs[-1]
            sigma = np.median(np.abs(d1), axis=-1, keepdims=True) / 0.6745
            thr = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
        det = [pywt.threshold(c, thr, mode=threshold_mode) for c in coeffs[1:]]
        rec_x = pywt.waverec([coeffs[0]] + det, wavelet, axis=-1)
        return rec_x[..., :n]                               # waverec can return one extra sample

    out = {}
    for ch, arr in rec.epochs.items():
        a2 = np.asarray(arr, float)
        if channels is not None and ch not in channels:
            out[ch] = a2
            continue
        out[ch] = _one(a2)
    return rec.replace_epochs(out)


# ------------------------------------------------- stage 2, by NOISE TYPE (Ch. 8)
def _along_last(fn, arr):
    """Apply a 1-D function along the last (time) axis of any-dimensional epochs."""
    a = np.asarray(arr, float)
    flat = a.reshape(-1, a.shape[-1])
    out = np.stack([fn(row) for row in flat])
    return out.reshape(a.shape)


def _remove_baseline(x, fs, how, hp_fc=0.5, order=4, poly_order=3, wavelet="db4",
                     causal=False):
    """Baseline-wander remedies. See `denoise.__doc__` for the trade-offs."""
    how = (how or "none").lower()
    if how in ("none", "off"):
        return x
    if how in ("highpass", "hp", "filter"):
        from scipy.signal import butter, filtfilt, lfilter
        b, a = butter(order, max(hp_fc / (fs / 2.0), 1e-6), "high")
        return lfilter(b, a, x, axis=-1) if causal else filtfilt(b, a, x, axis=-1)
    if how in ("detrend", "poly", "polynomial", "spline"):
        n = x.shape[-1]
        t = np.linspace(-1.0, 1.0, n)

        def _one(row):
            c = np.polyfit(t, row, poly_order)
            return row - np.polyval(c, t)
        return _along_last(_one, x)
    if how in ("wavelet", "dwt"):
        import pywt

        def _one(row):
            L = min(int(np.floor(np.log2(max(fs, 2)))) + 1,
                    pywt.dwt_max_level(len(row), pywt.Wavelet(wavelet).dec_len))
            if L < 1:
                return row
            c = pywt.wavedec(row, wavelet, level=L)
            c[0] = np.zeros_like(c[0])          # kill the coarsest approximation band
            return pywt.waverec(c, wavelet)[:len(row)]
        return _along_last(_one, x)
    raise ValueError(f"unknown baseline remedy {how!r}; see denoise.__doc__")


def _remove_powerline(x, fs, how, f0=50.0, q=30.0, harmonics=(1, 2, 3), mu=0.01,
                      causal=False):
    """Powerline-interference remedies. See `denoise.__doc__` for the trade-offs."""
    how = (how or "none").lower()
    if how in ("none", "off"):
        return x
    nyq = fs / 2.0
    freqs = [f0 * h for h in harmonics if f0 * h < nyq * 0.98]
    if not freqs:
        return x
    if how in ("notch", "iirnotch"):
        from scipy.signal import filtfilt, iirnotch, lfilter
        y = x
        for f in freqs:
            b, a = iirnotch(f / nyq, q)
            y = lfilter(b, a, y, axis=-1) if causal else filtfilt(b, a, y, axis=-1)
        return y
    if how in ("spectral", "interp", "spectral_interpolation"):
        # zero out the offending rFFT bins and linearly interpolate across them,
        # so no time-domain filter (and no ringing) is involved at all.
        def _one(row):
            n = len(row)
            X = np.fft.rfft(row)
            fr = np.fft.rfftfreq(n, 1.0 / fs)
            df = fr[1] - fr[0] if len(fr) > 1 else fs
            half = max(1, int(np.ceil((f0 / q) / max(df, 1e-12))))   # same width as the notch
            for f in freqs:
                j = int(round(f / max(df, 1e-12)))
                lo, hi = max(j - half, 1), min(j + half, len(X) - 1)
                if hi <= lo:
                    continue
                a_, b_ = X[lo - 1], X[hi + 1] if hi + 1 < len(X) else X[hi]
                w = np.linspace(0.0, 1.0, hi - lo + 1)
                X[lo:hi + 1] = (1 - w) * a_ + w * b_        # interpolate magnitude AND phase
            return np.fft.irfft(X, n=n)
        return _along_last(_one, x)
    if how in ("adaptive", "lms", "anc"):
        # LMS adaptive noise cancellation against a SYNTHESISED sin/cos reference
        # at f0 (and harmonics). The two-weight-per-frequency form tracks small
        # drifts in mains frequency and phase that a fixed notch cannot.
        def _one(row):
            n = len(row)
            t = np.arange(n) / fs
            ref = np.stack([g(2 * np.pi * f * t) for f in freqs for g in (np.sin, np.cos)])
            w = np.zeros(ref.shape[0])
            e = np.empty(n)
            p = float(np.mean(ref ** 2) * ref.shape[0]) + 1e-12
            step = mu / p
            for i in range(n):
                r = ref[:, i]
                e[i] = row[i] - float(w @ r)
                w += step * e[i] * r                        # standard LMS update
            return e
        return _along_last(_one, x)
    raise ValueError(f"unknown powerline remedy {how!r}; see denoise.__doc__")


def _remove_broadband(x, fs, how, win=5, savgol_order=3, wavelet="db4",
                      threshold_mode="soft", level=None):
    """Broadband / impulsive-noise remedies. See `denoise.__doc__`."""
    how = (how or "none").lower()
    if how in ("none", "off"):
        return x
    if how in ("movavg", "moving_average", "boxcar", "ma"):
        k = max(1, int(win))
        ker = np.ones(k) / k
        return _along_last(lambda r: np.convolve(r, ker, "same"), x)
    if how in ("gaussian", "gauss"):
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(x, sigma=max(float(win) / 6.0, 1e-6), axis=-1, mode="nearest")
    if how in ("savgol", "savitzky_golay", "sg"):
        from scipy.signal import savgol_filter
        k = max(int(win), int(savgol_order) + 2)
        k = k + 1 if k % 2 == 0 else k                      # savgol needs an odd window
        k = min(k, x.shape[-1] - (1 - x.shape[-1] % 2))
        if k <= savgol_order:
            return x
        return savgol_filter(x, k, savgol_order, axis=-1, mode="nearest")
    if how in ("median", "medfilt"):
        from scipy.ndimage import median_filter
        k = max(1, int(win))
        k = k + 1 if k % 2 == 0 else k
        size = [1] * x.ndim
        size[-1] = k
        return median_filter(x, size=tuple(size), mode="nearest")
    if how in ("wavelet", "dwt"):
        tmp = Recording(group="_", fs=fs, epochs={"_": x}, labels=np.zeros(1))
        return wavelet_denoise(tmp, wavelet=wavelet, level=level,
                               threshold_mode=threshold_mode).epochs["_"]
    raise ValueError(f"unknown broadband remedy {how!r}; see denoise.__doc__")


def denoise(rec: "Recording", impulsive=None, baseline=None, powerline=None,
            broadband=None, channels=None, note=None, **kw) -> "Recording":
    """**The stage-2 menu organised by NOISE TYPE rather than by technique** —
    Chapter 8's own structure. `bandpass_notch()` and `wavelet_denoise()` are
    single tools; this is the decision *table* they sit inside.

    Chapter 8's whole argument is that you do not choose a filter, you **identify
    a corruption and then choose its remedy**. §8.11's box is titled *"Identify
    before you filter"* and closes with the rule this function is shaped around:
    *"the noise's signature chooses the tool. Match a narrow, known-frequency
    artifact to a notch; low-frequency drift to a high-pass; broadband in-band
    noise to averaging or adaptive filtering; and impulsive outliers to a median —
    **not the reverse.**"* The chapter epigraph is blunter: *"Removing noise you
    have not identified is how you accidentally remove signal."*

    So the config is **one key per problem**, not one key per technique. Pick a
    remedy for each corruption you actually observed in your own data (run §8.3's
    three-lens check first — time, frequency, and what changes across epochs) and
    leave the rest off:

        track = ECGCinC2017Track(cfg={"impulsive": "median",
                                      "baseline":  "highpass",
                                      "powerline": "notch",
                                      "broadband": "wavelet"})

    **Order is fixed and it is not arbitrary.** `impulsive → baseline →
    powerline → broadband`:

    1. **Impulses first**, because §9.7 is explicit that "impulse removal must
       precede any linear filtering" — a spike is broadband, so a notch or a
       band-pass cannot delete it, it "only smears the spike across a wide
       neighbourhood". Filter first and you have turned one bad sample into a
       hundred mediocre ones.
    2. **Then the slow trend**, which otherwise dominates every subsequent
       estimate and can saturate an adaptive filter.
    3. **Then the hum**, whose removal is cleanest on a de-trended signal.
    4. **Then whatever broadband residue is left.** Smoothing earlier would blur
       the transients you were trying to protect *and* leave the drift behind.

    ---
    ### 0 · Impulsive / outlier noise — `impulsive=` (§8.7: electrode pops,
    saturated samples, the 1-D twin of salt-and-pepper)

    | option | how it works | buys you | costs you |
    |---|---|---|---|
    | `"median"` | running median over `win` samples (`scipy.ndimage.median_filter`) | §8.7: impulsive outliers are "**not** removable by a linear (averaging) filter" — a mean filter "simply smears the outlier into a blur — **it lets the outlier vote**". "a lone extreme value cannot move a median, so the spike is rejected outright while edges are preserved" | non-linear, so its effect has no clean frequency-domain description, and a window longer than a feature flattens that feature outright. Choose `win` shorter than the narrowest thing you care about (on ECG: shorter than the QRS) |

    §8.2 adds the prior question: is this segment *noisy* or *broken*? "bad
    segments should be detected and excluded or flagged, **not silently
    filtered**" — filtering "quietly manufactures a plausible-looking trace out of
    garbage, and every downstream number inherits the fiction." A flatline or a
    railing electrode is a quality-control exclusion, not a median-filter job.

    ### 1 · Baseline wander — `baseline=` (§8.8: half-cell potential shifts, skin
    stretch, impedance change, cable movement; respiration drift near 0.15-0.3 Hz)

    | option | how it works | buys you | costs you |
    |---|---|---|---|
    | `"highpass"` | Butterworth high-pass at `hp_fc` (default 0.5 Hz) — **the book's own remedy** (§8.8, §8.16 table) | one line, well understood, exactly the right shape when the drift really is band-limited below the signal | §8.8: "the cutoff must be chosen with care, and the right value depends on the task. For **monitoring** (rhythm and heart-rate) a cutoff around 0.5 Hz is acceptable, but a **diagnostic** ECG must preserve content down to about **0.05 Hz**: too aggressive a high-pass distorts the low-frequency ST segment and T wave and **can manufacture artificial ST shifts that mimic ischaemia.**" The corner is a clinical decision wearing numerical clothes. A sharp filter also rings around the QRS |
    | `"detrend"` † | fit a low-order polynomial (`poly_order`, default 3) per epoch and subtract it | no filter: no phase response, no ringing, no corner frequency to defend. Handles drift that is *not* a clean band — a monotone ramp, a step-and-settle | the polynomial has no idea what is signal. A genuine slow physiological trend — a real FHR baseline shift, a slow tonic EMG rise — is fitted and subtracted along with the artifact. Raise the order and it starts eating the signal itself |
    | `"wavelet"` † | DWT, zero the coarsest **approximation** band, reconstruct | a wavelet-domain answer to the same problem: removes drift localised in time as well as slow, without committing to a single corner frequency | which scale counts as "baseline" depends on `fs` and the depth, so it is a corner frequency in disguise — just a less explicit one. Say which level you zeroed |

    ### 2 · Powerline interference — `powerline=` (§8.6: 50/60 Hz mains and its
    harmonics — narrow-band, near-stationary, usually strong)

    §8.6 calls this "**the *friendliest* artifact**: because it is narrow and its
    frequency is known in advance, a **notch filter** … removes it while barely
    touching the signal", and reminds you that "**Prevention helps too: good
    grounding, shielding, and twisted leads reduce pickup at the source, which is
    usually better than filtering after the fact.**"

    | option | how it works | buys you | costs you |
    |---|---|---|---|
    | `"notch"` | IIR notch at `f0` and `harmonics` (default 1st-3rd), quality factor `q` — **the book's remedy** (§8.6, built in §9.6) | simplest and cheapest; if the hum is stable and narrow this is genuinely all you need | **`q` is the whole decision.** Too narrow and it misses a hum whose frequency wobbles, and leaves the harmonics you did not list; too wide and it removes real signal beside 50 Hz — on EMG the 20-150 Hz band *is* the signal, so every notch is a real bite out of it |
    | `"adaptive"` | LMS cancellation against a synthesised sin/cos reference at each mains frequency (§10.5's method, applied here) | **tracks** amplitude, phase and small frequency drift, so it removes a wandering hum a fixed notch cannot; it subtracts only what correlates with the reference, so it is much gentler on nearby signal | §8.9's setup assumes a **real reference channel**; the synthesised reference used here is weaker. Needs `mu` tuned, converges over the first samples of each epoch, and can slowly adapt onto genuine signal that happens to correlate with the reference |
    | `"spectral"` † | rFFT, linearly interpolate across the offending bins, inverse transform | no time-domain filter at all: no ringing, no phase distortion, no edge transient. Surgical when the hum really occupies a few bins | needs **frequency resolution** — the bins must isolate the hum, so a short epoch cannot separate 50 Hz from its surroundings. And it assumes the hum is stationary across the whole epoch, which is exactly what `"adaptive"` does not assume |

    ### 3 · Broadband / in-band noise — `broadband=` (§8.4 white/thermal, §8.9
    EMG contamination of EEG)

    §8.4 is uncomfortably honest about the limit here: "Because white noise
    spreads its energy everywhere, some of it always overlaps the signal band and
    cannot be filtered out without touching the signal; **the only clean weapons
    against it are averaging (Chapter 7) and improving the acquisition
    hardware.**" Everything below is a compromise, not a cure.

    | option | how it works | buys you | costs you |
    |---|---|---|---|
    | `"movavg"` | boxcar moving average over `win` samples | the cheapest possible low-pass; unbiased for a locally constant signal | **blurs sharp features** — it lowers and widens a QRS peak — and its frequency response has large side-lobes, so it is a poor low-pass in the frequency domain too. §8.7: it "lets the outlier vote" |
    | `"savgol"` | Savitzky-Golay local polynomial least-squares (`win`, `savgol_order`) — §9.7's "smoothing that keeps the peak" | "**smooths broadband noise while largely preserving the height, width, and area of peaks** — the property a mean filter destroys — provided the window is kept **shorter than the peak** it must follow" | §9.7: "It is still a *linear* filter, so, like the mean, it does not reject true outliers — **pair it with a median pre-pass**" (which is what `impulsive="median"` gives you, and why it runs first) |
    | `"wavelet"` | DWT coefficient thresholding (`wavelet_denoise`) — §10.6's "wavelet denoising: exploiting sparsity" | §10.6's selection rule: use it "when the signal is *sparse/transient* and the noise is *broadband* (a single noisy ECG with no repetition and no reference)". Time-**and**-frequency localised, so it keeps the brief sharp transients (QRS, K-complexes, action potentials) that every fixed smoother rounds off | shrinks everything, so an over-set threshold erodes genuine low-amplitude structure such as a weak spindle; more parameters to defend |
    | `"gaussian"` | Gaussian-weighted smoothing (`win` ≈ 6σ) | smoother frequency response than a boxcar for the same width | same blurring bargain as `"movavg"`, same blindness to outliers |

    **The distinction this section exists to teach:** "noisy" is not one problem.
    Gaussian broadband noise and impulsive spike noise call for *different*
    remedies, and the linear tools that are correct for the first are actively
    wrong for the second — which is why they are separate keys here, and why the
    median runs first.

    ---
    † **Honesty note about provenance.** The options marked † are *extensions
    beyond the book*, offered because they are standard practice and make the
    trade-off richer: Chapter 8 names only high-pass/detrend for wander, notch for
    mains, median for impulses, and averaging/adaptive for broadband. Polynomial
    and wavelet baseline removal, and spectral interpolation of the mains bins, do
    not appear in the text. Use them if you can defend them — but cite something
    other than the book, and do not claim the chapter recommended them.

    ---
    Extra keyword arguments pass through to the individual remedies: `hp_fc`,
    `order`, `poly_order` (baseline); `f0`, `q`, `harmonics`, `mu` (powerline);
    `win`, `savgol_order`, `threshold_mode`, `level` (broadband and impulsive);
    `wavelet` and `causal` are shared. `channels` restricts the whole recipe to a
    subset of channels; the rest pass through untouched. `note` may be a callable
    (e.g. `self.note`) that the function uses to report re-ordering decisions.

    ⚠️ The filtered options are zero-phase (`filtfilt`) by default and therefore
    **non-causal** — right for the offline analysis this scaffold does, wrong for
    a live system. `causal=True` switches the IIR paths to a single forward
    `lfilter` pass. Note that `"detrend"`, `"spectral"` and `"wavelet"` are
    whole-epoch operations and are *inherently* non-causal whatever you pass,
    while `"median"` and `"adaptive"` are naturally causal-friendly. See
    `bandpass_notch.__doc__`.

    Returns a NEW Recording; never mutates the input.
    """
    fs = rec.fs
    # A median asked for in the broadband slot is still an IMPULSE remedy, and
    # §9.7 says it must run before any linear filtering. Move it rather than
    # silently doing the wrong thing in the wrong order.
    if str(broadband or "").lower() in ("median", "medfilt"):
        if not impulsive:
            impulsive, broadband = "median", None
            if callable(note):
                note("denoise: broadband='median' is an IMPULSE remedy, so it was moved to the "
                     "impulsive stage and now runs BEFORE the linear filters (§9.7: 'impulse "
                     "removal must precede any linear filtering' — a linear filter smears a spike "
                     "instead of deleting it). Pass impulsive='median' directly to say so yourself.")
    shared = {k: kw[k] for k in ("wavelet", "causal") if k in kw}
    b_kw = {k: kw[k] for k in ("hp_fc", "order", "poly_order") if k in kw}
    p_kw = {k: kw[k] for k in ("f0", "q", "harmonics", "mu") if k in kw}
    n_kw = {k: kw[k] for k in ("win", "savgol_order", "threshold_mode", "level") if k in kw}
    out = {}
    for ch, arr in rec.epochs.items():
        x = np.asarray(arr, float)
        if channels is not None and ch not in channels:
            out[ch] = x
            continue
        if impulsive:                                       # 0 — spikes first (§9.7)
            x = _remove_broadband(x, fs, impulsive,
                                  **{k: v for k, v in n_kw.items() if k == "win"})
        if baseline:                                        # 1 — slow trend
            x = _remove_baseline(x, fs, baseline,
                                 **{k: v for k, v in shared.items() if k in ("wavelet", "causal")},
                                 **b_kw)
        if powerline:                                       # 2 — narrow-band hum
            x = _remove_powerline(x, fs, powerline,
                                  **{k: v for k, v in shared.items() if k == "causal"}, **p_kw)
        if broadband:                                       # 3 — what is left
            x = _remove_broadband(x, fs, broadband,
                                  **{k: v for k, v in shared.items() if k == "wavelet"}, **n_kw)
        out[ch] = x
    return rec.replace_epochs(out)


# --------------------------------------------- stage 3 options: SPECTRAL ESTIMATION (Ch. 7)
def _arburg(x, order):
    """Burg's method for AR coefficients — the same implementation Chapter 7's
    figures use (`figures_src/ch07_figures.py`), inlined so `tracks/` needs no
    extra dependency. Returns `(a[order], sigma2)` for x[n] = Σ a_k x[n-k] + e[n]."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n <= order + 1:
        return np.zeros(order), float(np.var(x) + 1e-12)
    f = x.copy(); b = x.copy()
    den = np.dot(f, f) + np.dot(b, b)
    E = np.dot(x, x) / n
    A = np.zeros(order + 1); A[0] = 1.0
    for m in range(order):
        if den == 0 or n - m - 1 <= 0:
            break
        k = (-2.0 * np.dot(b[:n - m - 1], f[m + 1:])) / den
        A_prev = A.copy()
        for i in range(1, m + 2):
            A[i] = A_prev[i] + k * A_prev[m + 1 - i]
        E *= (1 - k * k)
        f_new = f[m + 1:] + k * b[:n - m - 1]
        b_new = b[:n - m - 1] + k * f[m + 1:]
        f[m + 1:], b[:n - m - 1] = f_new, b_new
        if (n - m - 2) > 0:
            den = np.dot(f[m + 2:], f[m + 2:]) + np.dot(b[:n - m - 2], b[:n - m - 2])
    return -A[1:order + 1], float(E)


def make_spectral_estimator(kind: str = "welch", nperseg=None, n_segments: int = 8,
                            nfft: int = 1024, ar_order: int = 16,
                            bandwidth: float = 4.0, n_tapers=None):
    """Spectral-estimation stage (part of module 3) — **the PSD every band-power
    feature is integrated from.** Returns a callable `psd(x, fs) -> (freqs, pxx)`.

    Chapter 7's point is that "the PSD" is not a fact you look up; it is an
    **estimate**, and which estimator you used is a design decision with the same
    standing as the band edges. The raw periodogram is *inconsistent*: its
    variance does not shrink as you collect more data, so a longer recording buys
    you a longer, equally noisy spectrum. Every non-parametric method below is a
    different way of spending resolution to buy variance reduction; the parametric
    method buys resolution back by assuming a model.

    §7.2's word for the periodogram is **inconsistent**: "As N grows, we get
    *more frequency bins*, but the scatter around the true value at each bin
    **does not decrease**." A longer record buys finer bin spacing and no extra
    reliability. §7.4 then names the law every non-parametric method obeys:
    "Bartlett's method converts the useless variance of the periodogram into
    usable smoothness, paid for in resolution. **This resolution-variance
    trade-off is the governing law of non-parametric spectral estimation.**"

    | `kind` | Family | Variance | Resolution | Best when | Watch out |
    |---|---|---|---|---|---|
    | `"periodogram"` | non-parametric | high (**inconsistent**) | bin spacing ≈ 1/T | "never for features; a baseline" (§7.16) — include it to be *beaten* | variance stays ≈ the square of the true PSD at every bin **no matter how long the record**. A band power read off it is mostly estimation noise |
    | `"bartlett"` | non-parametric | ≈ 1/K | ≈ K/T (coarser) | long record, want simple averaging (§7.4) | averaging K segments cuts variance by ≈ 1/K, but each segment is N/K long so bin spacing *and* practical resolution coarsen by the same K. No taper, so **leakage** is untouched |
    | `"welch"` *(default)* | non-parametric | ≈ 1/K | tunable via `nperseg` | "long, stationary record — the workhorse" (§7.16); what `bsp.bandpower` already uses | §7.5: "Longer segments give finer resolution but fewer of them to average (higher variance); shorter segments give more averaging but coarser resolution. **You are always dialing along the same trade-off.**" Overlapping segments are not independent, so you get less variance reduction than the segment count suggests |
    | `"multitaper"` | non-parametric | ≈ 1/K, K ≈ 2NW−1 | full-record, set by NW | "short record, want low-leakage smooth PSD (EEG/neuro)" (§7.16) — a 4 s BCI trial or a 20 s ECG strip, where Welch **runs out of segments to average** | §7.6: it "trades a modest, *controllable* smearing (the half-bandwidth W) for a variance reduction". `bandwidth` (the time-bandwidth product NW, typically 3-4) sets resolution *and* how many tapers are usable — raise it to average more (lower variance, broader main lobe), lower it for sharper resolution. Most expensive to compute and to explain |
    | `"ar"` | **parametric** (Burg) | low, smooth | very high (sub-1/T) | "short segment, resonant rhythm" (§7.16) — a mu peak or an HRV peak you intend to *locate* | §7.9: "Choose it too low and the spectrum is over-smoothed — real peaks merge or vanish (**under-fitting**). Choose it too high and the model starts fitting noise, sprouting **spurious peaks** that are not in the true spectrum (**over-fitting**)." At p=40 on the book's EEG epoch the curve "breaks out in a rash of small peaks that are artifacts of over-fitting, not physiology" |

    §7.9's decision framework, which is the sentence to quote in your report:
    *"if you have a long, stationary record, use non-parametric Welch — the
    everyday workhorse; if you have a short segment, use parametric AR (Burg or
    Yule-Walker) for its resolution — and choose the order carefully."* Multitaper
    sits between them: non-parametric and assumption-light like Welch, but because
    it averages orthogonal tapers over the *whole* record it is often the better
    non-parametric choice on short segments.

    **Why `"ar"` here is Burg and not Yule-Walker.** §7.7: Burg "minimizes forward
    *and* backward prediction error directly on the data (never explicitly forming
    the autocorrelation), which gives it better resolution and more stable
    estimates on short records — **usually the best choice for biosignal epochs**."
    Order selection is yours: AIC = N·ln σ̂²_p + 2p, MDL = N·ln σ̂²_p + p·ln N, and
    §7.9's rule of thumb caps p at about N/3, "often far lower for a stationary
    epoch". `ar_order` defaults to 16, which is the value §7.9 shows working on a
    30 s EEG epoch — it is a *starting point you must justify*, not a constant.

    A concrete reason to care on these tracks: sleep band powers are integrated
    over bands only a few Hz wide (the book's own edges are delta 0.5-4, theta
    4-8, alpha 8-11, sigma/spindle 11-16, beta 16-30 Hz), so an estimator whose
    practical resolution has coarsened past a few Hz is no longer measuring the
    band you named — while an estimator with too much variance turns a real
    spindle into a coin flip. Both failure modes are invisible in the final
    metric unless you go and look.

    Note the §7.8 student-mistake this whole helper is built to avoid: band power
    is a density **integrated** over the band, ∫S(f)df — not a sum of raw FFT
    magnitudes. `spectral_bandpower()` below does the integration for you.

    Parameters mirror the book's: `nperseg`/`n_segments` for Welch and Bartlett,
    `bandwidth` (NW) and `n_tapers` for multitaper (`scipy.signal.windows.dpss`),
    `ar_order` for AR, `nfft` for the parametric frequency grid.
    """
    kind = (kind or "welch").lower()
    from scipy import signal as _sig

    if kind in ("periodogram", "raw"):
        def psd(x, fs):
            return _sig.periodogram(np.asarray(x, float), fs=fs)
        return psd

    if kind in ("bartlett", "segment_average"):
        def psd(x, fs):
            x = np.asarray(x, float)
            nseg = nperseg or max(8, len(x) // max(1, int(n_segments)))
            # noverlap=0 and a boxcar window IS Bartlett's method
            return _sig.welch(x, fs=fs, nperseg=min(nseg, len(x)), noverlap=0,
                              window="boxcar")
        return psd

    if kind in ("welch", "default"):
        def psd(x, fs):
            x = np.asarray(x, float)
            return _sig.welch(x, fs=fs, nperseg=min(nperseg or 256, len(x)))
        return psd

    if kind in ("multitaper", "mtm", "dpss"):
        def psd(x, fs):
            x = np.asarray(x, float)
            n = len(x)
            nw = float(bandwidth)
            k = int(n_tapers) if n_tapers else max(1, int(2 * nw) - 1)
            k = max(1, min(k, n - 1))
            tapers = _sig.windows.dpss(n, nw, k)
            f = np.fft.rfftfreq(n, 1.0 / fs)
            acc = np.zeros(len(f))
            for w in np.atleast_2d(tapers):
                X = np.fft.rfft(x * w)
                p = (np.abs(X) ** 2) / (fs * np.sum(w ** 2))
                if len(p) > 2:
                    p[1:-1] *= 2.0                          # one-sided convention
                acc += p
            return f, acc / len(np.atleast_2d(tapers))
        return psd

    if kind in ("ar", "burg", "parametric", "yule_walker", "yule-walker"):
        def psd(x, fs):
            a, s2 = _arburg(np.asarray(x, float), int(ar_order))
            return bio_ar_psd(a, s2, fs, nfft=int(nfft))
        return psd

    raise ValueError(f"unknown spectral method {kind!r}; see make_spectral_estimator.__doc__")


def spectral_bandpower(x, fs, band, method="welch", **kw) -> float:
    """Integrated power in `band` under the spectral estimator of your choice.

    Same trapezoid-with-interpolated-edges integration as
    `bsp.biosignals.bandpower` (so adjacent bands that tile the spectrum share
    their boundary point rather than each losing half of it), but the PSD comes
    from `make_spectral_estimator(method, **kw)` instead of a hard-coded Welch.
    `method="welch"` reproduces the shipped behaviour."""
    f, pxx = make_spectral_estimator(method, **kw)(x, fs)
    lo, hi = band
    lo, hi = max(lo, f[0]), min(hi, f[-1])
    if hi <= lo:
        return 0.0
    inside = (f > lo) & (f < hi)
    fb = np.concatenate(([lo], f[inside], [hi]))
    pb = np.concatenate(([np.interp(lo, f, pxx)], pxx[inside], [np.interp(hi, f, pxx)]))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(pb, fb))


def bio_ar_psd(a, sigma2, fs, nfft=1024):
    """Thin lazy wrapper over the book's own `bsp.ar_psd(a, sigma2, fs)` helper
    (Appendix H / §7.7), imported on demand so this module loads without bsp."""
    import biosignals as _bio
    return _bio.ar_psd(a, sigma2, fs, nfft=nfft)


# --------------------------------------------------------------- metric helpers
def metrics_for(y_true, y_pred, labels=None) -> dict:
    """The four-number panel for one slice of predictions (a group, a fold, or the
    pooled set). Undefined entries come back as NaN rather than a misleading 0."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    nan = float("nan")
    out = {"n": int(len(yt))}
    if len(yt) == 0:
        return {**out, "accuracy": nan, "cohens_kappa": nan,
                "macro_f1": nan, "balanced_accuracy": nan}
    out["accuracy"] = float(np.mean(yt == yp))
    # kappa / macro-F1 / balanced accuracy are undefined (or degenerate) on a
    # single-class slice — e.g. a record-level track where each "group" holds one
    # label. NaN, not a flattering 0.0 or 1.0, so the spread falls back to folds.
    single_class = len(set(yt.tolist())) < 2
    for name, fn in (("cohens_kappa", M.cohens_kappa),
                     ("macro_f1", M.macro_f1),
                     ("balanced_accuracy", M.balanced_accuracy)):
        if single_class:
            out[name] = nan
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[name] = float(fn(yt, yp))
        except Exception:                                   # noqa: BLE001
            out[name] = nan
    return out


def per_group_metrics(y_true, y_pred, groups, labels=None, key="group") -> list:
    """Score EACH held-out group (subject / record / fold) separately — the raw
    material for the spread the book insists you report (§16.3)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    gs = np.asarray(groups)
    rows = []
    for g in _ordered_unique(gs):
        m = gs == g
        row = {key: g if not isinstance(g, np.generic) else g.item()}
        row.update(metrics_for(yt[m], yp[m], labels))
        rows.append(row)
    return rows


def metric_spread(rows, metric="cohens_kappa") -> dict:
    """mean / sd / min / max / n over per-group (or per-fold) results, ignoring
    the groups where the metric is undefined."""
    vals = np.array([r.get(metric, float("nan")) for r in rows], float)
    ok = vals[np.isfinite(vals)]
    if ok.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "n": 0}
    return {"mean": float(np.mean(ok)), "std": float(np.std(ok, ddof=1)) if ok.size > 1 else 0.0,
            "min": float(np.min(ok)), "max": float(np.max(ok)), "n": int(ok.size)}


def spread_line(metric: str, sp: dict, unit: str = "group") -> str:
    """'mean cohens_kappa 0.61 (sd 0.12, range 0.34-0.73 across 8 subjects)' —
    the sentence shape §16.9's worked results paragraph uses."""
    if not sp or not np.isfinite(sp.get("mean", float("nan"))):
        return f"{metric}: spread unavailable (too few scoreable {unit}s)"
    return (f"mean {metric} {sp['mean']:.3f} (sd {sp['std']:.3f}, "
            f"range {sp['min']:.3f}-{sp['max']:.3f} across {sp['n']} {unit}s)")


def _ordered_unique(a):
    seen, out = set(), []
    for v in np.asarray(a).tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _cfg_values_equal(a, b) -> bool:
    """Compare two cfg values tolerantly: a band written `(0.5, 40)` and one
    written `[0.5, 40.0]` are the same design decision, and neither should be
    reported as a train/infer conflict."""
    if a is b:
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_cfg_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:                                   # noqa: BLE001
            return False
    try:
        return bool(a == b)
    except Exception:                                       # noqa: BLE001
        return False


def _transform(selector, X):
    return X if selector is None else selector.transform(X)


def _n_selected(selector, n_feat):
    """How many features a FITTED selector keeps (None if it will not say)."""
    try:
        return int(np.asarray(selector.get_support()).sum())
    except Exception:                                       # noqa: BLE001
        return None


# --------------------------------------------------------------- fitted bundle
@dataclass
class FittedModel:
    """A FROZEN pipeline: the fitted selector + the fitted classifier + the cfg
    they were fit under. `.predict(X)` applies them in order, so `infer()` can
    never accidentally re-fit anything. Quacks like an sklearn estimator, so
    existing `model.predict(...)` code keeps working.

    **`cfg` is not a souvenir — it is the contract.** It records the *resolved*
    configuration stages 2-3 ran under when this model was fit, and `infer()` /
    `write_submission()` rebuild test features from it by default. That is what
    stops a model trained under `spectral_method="ar"` from later being scored on
    Welch features because the inference call forgot to repeat the cfg.

    Because it is the contract, it is **immutable** (a `_FrozenCfg`): read it,
    print it, `dict()` it — but `model.cfg["gravity"] = "none"` raises
    `TypeError`, and so does rebinding `model.cfg` wholesale. A contract you can
    edit after signing is not a contract, and a *silently* edited one is worse
    than none: `infer()` would quietly start building different features while
    every number you already reported still described the old pipeline. To score
    under a different config, pass it to `infer()` (which checks it against this
    one and makes the mismatch explicit) or train a second model."""
    clf: object
    selector: object = None
    cfg: dict = field(default_factory=dict)

    def __post_init__(self):
        # snapshot at fit time: deep-copied, so nothing the caller still holds a
        # reference to can reach inside it, and sealed, so nobody can edit it here.
        object.__setattr__(self, "cfg", _freeze_cfg(self.cfg))

    def __setattr__(self, name, value):
        if name == "cfg" and isinstance(self.__dict__.get("cfg"), _FrozenCfg):
            raise TypeError(
                "FittedModel.cfg is the record of the config this model was TRAINED "
                f"under and cannot be replaced. {_FrozenCfg._WHY}")
        object.__setattr__(self, name, value)

    def predict(self, X):
        return self.clf.predict(_transform(self.selector, np.asarray(X)))

    def predict_proba(self, X):
        return self.clf.predict_proba(_transform(self.selector, np.asarray(X)))

    def __getattr__(self, name):                            # delegate to the classifier
        if name.startswith("__") or name in ("clf", "selector", "cfg"):
            raise AttributeError(name)
        try:
            clf = self.__dict__["clf"]
        except KeyError:                                    # e.g. during unpickling
            raise AttributeError(name) from None
        return getattr(clf, name)


# --------------------------------------------------------------- base track
class TrackAdapter:
    meta: TrackMeta

    #: **Capability declaration — the stage-2/3 cfg keys THIS track actually
    #: reads**, on top of `BASE_CFG_KEYS` (which every track supports because the
    #: base class owns stages 4-5 and the validation scheme).
    #:
    #: This exists because the six tracks do genuinely different amounts of
    #: signal processing. Sleep-EDF and ECG run the full Chapter 8 denoise menu
    #: and integrate band powers from a Chapter 7 estimator; BCI has a spectral
    #: decision but an identity `preprocess()`; HAR has exactly one preprocessing
    #: knob (gravity) and no PSD-based feature at all; EMG and CTG have neither.
    #: Offering a menu that the track's own code never reads is not a "choice
    #: with a consequence" — it is a no-op wearing the costume of one — so the
    #: notebook generator renders only the menus a track declares here, and
    #: `_cfg()` raises `UnsupportedCfgKey` rather than silently ignoring the rest.
    #:
    #: **If you implement a stage the shipped adapter left as a stub** (a CSP
    #: filter in BCI's `preprocess()`, dropout handling in CTG's), declare the new
    #: keys — either by extending this set in your subclass or, per instance, with
    #: `track.declare_cfg_keys("csp_components", ...)`.
    SUPPORTED_CFG_KEYS: frozenset = frozenset()

    def __init__(self, cfg: dict | None = None):
        #: extra keys declared at runtime via `declare_cfg_keys()` — set BEFORE
        #: the cfg below is validated so a subclass can widen the contract first.
        self._extra_cfg_keys: set = set()
        #: things the harness noticed and refuses to hide (e.g. "select_k was
        #: clamped, so selection is a no-op"). Printed once each, and attached to
        #: every report as `rep["notes"]`.
        self.notes: list = []
        #: per-instance config overrides (see DEFAULT_CFG). Config-driven runs are
        #: rubric Criterion 4; keep your knobs here, not scattered in the code.
        self.cfg = dict(cfg or {})
        self._validate_cfg_keys(self.cfg, where=f"{type(self).__name__}(cfg=...)")

    # ---- capability declaration (which menus are real on this track) ----
    @classmethod
    def supported_cfg_keys(cls) -> frozenset:
        """Every cfg key this track actually consumes: `BASE_CFG_KEYS` (stages
        4-5, owned by the base class) plus the track's own `SUPPORTED_CFG_KEYS`
        (its stage-2/3 knobs). `tools/build_track_notebooks.py` reads this to
        decide which decision menus to render, so a student never sees an option
        that would do nothing."""
        return frozenset(BASE_CFG_KEYS) | frozenset(cls.SUPPORTED_CFG_KEYS)

    def declare_cfg_keys(self, *keys: str) -> "TrackAdapter":
        """Widen this instance's contract — call it when you have implemented a
        stage the shipped adapter left as a stub and want its knobs to be legal
        cfg keys. Returns self, so it chains: ``track.declare_cfg_keys("csp_k")``."""
        self.__dict__.setdefault("_extra_cfg_keys", set()).update(str(k) for k in keys)
        return self

    def supports(self, key: str) -> bool:
        """True if `key` is a cfg option this track's own stages read."""
        return key in self.supported_cfg_keys() or key in getattr(self, "_extra_cfg_keys", set())

    def feature_cfg_keys(self) -> frozenset:
        """The supported keys that can change the FEATURE MATRIX (stages 2-3) —
        i.e. everything except the stage-4/5 and CV keys. `infer()` compares
        exactly these when checking a caller-supplied cfg against the one the
        model was trained under."""
        return frozenset(
            (self.supported_cfg_keys() | frozenset(getattr(self, "_extra_cfg_keys", set())))
            - NON_FEATURE_CFG_KEYS)

    def _validate_cfg_keys(self, cfg: dict, where: str = "cfg") -> None:
        """Fail loudly on a cfg key this track never reads (§2 of the review: an
        advertised option that is silently inert is worse than no option)."""
        if not cfg:
            return
        bad = sorted(k for k in cfg if not self.supports(k))
        if not bad:
            return
        table = dict(_CFG_KEY_HINTS)
        table.update(getattr(self, "CFG_KEY_HINTS", {}) or {})
        hints = []
        for k in bad:
            hint = table.get(k)
            hints.append(f"  * {k!r}{': ' + hint if hint else ''}")
        who = getattr(getattr(self, "meta", None), "name", None) or type(self).__name__
        raise UnsupportedCfgKey(
            f"{where}: {', '.join(repr(k) for k in bad)} "
            f"{'is' if len(bad) == 1 else 'are'} not used by the "
            f"{who} track ({type(self).__name__}), so setting "
            f"{'it' if len(bad) == 1 else 'them'} would have NO EFFECT:\n"
            + "\n".join(hints)
            + f"\n\nThis track reads: {', '.join(sorted(self.supported_cfg_keys() | set(getattr(self, '_extra_cfg_keys', set()))))}.\n"
            f"If you have implemented the stage that would consume "
            f"{'this key' if len(bad) == 1 else 'these keys'}, say so with "
            f"`track.declare_cfg_keys({', '.join(repr(k) for k in bad)})` "
            f"(or extend {type(self).__name__}.SUPPORTED_CFG_KEYS) and re-run.")

    def note(self, msg: str, show: bool = True) -> str:
        """Record a harness observation ONCE (deduped) and print it. Silence is
        how a no-op survives; a printed line is how a student learns something
        true instead of nothing."""
        notes = self.__dict__.setdefault("notes", [])
        if msg not in notes:
            notes.append(msg)
            if show:
                print(f"[{type(self).__name__}] {msg}")
        return msg

    def _cfg(self, cfg: dict | None = None) -> dict:
        """Merge DEFAULT_CFG + the adapter's cfg + this call's overrides, and
        **check the overrides against the track's declared capabilities** before
        anything downstream can silently ignore them.

        Already-resolved dicts (the `_ResolvedCfg` returned by an earlier call,
        which one stage hands to the next) skip the check — otherwise every
        internal hand-off would re-validate DEFAULT_CFG's full key set against a
        track that only supports part of it.

        **`self.cfg` is re-checked on every resolution, not just in `__init__`.**
        `track.cfg` is a plain dict a student can edit after construction
        (`track.cfg["spectral_method"] = "ar"`), and validating only at
        construction time left that path wide open: the unsupported key would then
        be merged in below and silently ignored downstream — the very
        silently-inert-option bug the capability declaration exists to kill, just
        reached through a different door. Re-validating costs one set membership
        test per key per call and closes it."""
        if not isinstance(cfg, _ResolvedCfg):
            self._validate_cfg_keys(cfg or {}, where=f"{type(self).__name__}: cfg=")
        own = getattr(self, "cfg", None) or {}
        if not isinstance(own, _ResolvedCfg):
            self._validate_cfg_keys(own, where=f"{type(self).__name__}.cfg")
        out = dict(DEFAULT_CFG)
        out.update(own)
        out.update(cfg or {})
        return _ResolvedCfg(out)

    # ================= module 1 — DATA LOADING (each track overrides) =========
    def download(self, cache_dir: str, subset=None) -> None:
        raise NotImplementedError

    def load(self, cache_dir: str) -> list:          # -> [Recording]  (REAL data)
        raise NotImplementedError

    def smoke(self) -> list:                          # -> [Recording]  (SYNTHETIC, offline)
        raise NotImplementedError

    # ================= module 2 — PREPROCESSING ==============================
    def preprocess(self, rec: Recording, cfg: dict | None = None) -> Recording:
        """Clean the raw epochs BEFORE any feature is computed. Default: identity —
        the supplied baseline runs on the signal as loaded, which is a starting
        point, not a recommendation.

        There is no single correct recipe here, and the scaffold deliberately does
        not pick one for you. The usual candidates (Ch. 8-10), with what each buys
        and costs:

        * **band-pass** — keeps the frequency range your features live in; too
          narrow and you delete the signature you are about to measure.
        * **mains notch (50/60 Hz)** — cheap and effective for narrow-band hum;
          pointless (or harmful) if your band never reaches it.
        * **wavelet denoising** — thresholds DWT coefficients instead of bands, so
          it removes *non-stationary transients* (motion, pops, drift) that no
          fixed band contains, and does it without rounding off the sharp
          landmarks (QRS, K-complexes, action potentials) a band-pass blurs; the
          cost is that too aggressive a threshold erodes genuine low-amplitude
          structure such as a weak spindle.
        * **re-referencing / channel differencing** — cancels common-mode drift;
          changes what "the channel" means, so features are no longer comparable
          to un-re-referenced literature.
        * **artifact handling** — reject vs. interpolate vs. keep-and-flag. The
          first loses data, the second invents it, the third pushes the problem
          into the features. All three are defensible; say which and why.
        * **resampling / detrending / normalisation per recording** — helps
          cross-subject comparability, can erase amplitude information that
          actually carried the class.

        `bandpass_notch(rec, band=..., notch=...)` and
        `wavelet_denoise(rec, wavelet=..., threshold_mode=...)` in this module are
        ready-made implementations of the first three if you want them — you still
        choose the numbers, and the two are answers to *different* noise
        signatures (stationary narrow-band vs. non-stationary transient); see
        `wavelet_denoise.__doc__` for the side-by-side. Return a NEW `Recording`
        (see `Recording.replace_epochs`).

        ⚠️ **`bandpass_notch` defaults to `filtfilt` — zero-phase, and therefore
        NON-CAUSAL.** That is correct for the offline, whole-epoch analysis this
        scaffold performs, but it is not deployable in a live streaming system;
        `causal=True` switches to a single forward `lfilter` pass so you can
        measure the difference. This matters most on the two tracks with a
        real-time framing (BCI motor imagery, EMG gesture control) — see
        `bandpass_notch.__doc__`.
        """
        return rec

    # ================= module 3 — FEATURE EXTRACTION =========================
    def extract_features(self, rec: Recording, cfg: dict | None = None):
        """Cleaned Recording -> `(X[n_epochs, n_feat], y[n_epochs], group)`.

        This is where the DSP of the course turns into numbers, and where most of
        the marks in "Signal processing rigor" are won. Each track ships a working
        implementation you are expected to *replace or extend*, not admire.
        """
        raise NotImplementedError

    def feature_names(self, cfg: dict | None = None) -> list | None:
        """Optional: names aligned with `extract_features`' columns. Useful for
        reading feature importances and for reporting which features a selector
        kept. Return None if you have not written them down (but do write them
        down — an unnamed feature cannot be justified in a report)."""
        return None

    # ---- backward-compatible shim -------------------------------------------
    def features(self, rec: Recording):
        """DEPRECATED (pre-seven-module API): preprocessing + feature extraction in
        one call. Kept so older adapters/notebooks keep running; new code should
        call `preprocess()` then `extract_features()` — the rubric grades those
        as *separable* stages."""
        return self.extract_features(self.preprocess(rec))

    def _features_for(self, rec: Recording, cfg: dict | None = None):
        """Internal: run modules 2+3 for one recording. If a subclass still
        overrides the legacy `features()`, honour it (once, with a warning) so old
        student code does not break."""
        cfg = self._cfg(cfg)
        if type(self).features is not TrackAdapter.features:
            if not getattr(self, "_legacy_warned", False):
                warnings.warn(
                    f"{type(self).__name__} overrides the legacy features(rec); split it into "
                    "preprocess(rec, cfg) + extract_features(rec, cfg) — the rubric grades the "
                    "seven pipeline stages as separable.", DeprecationWarning, stacklevel=2)
                self._legacy_warned = True
            return self.features(rec)
        return self.extract_features(self.preprocess(rec, cfg), cfg)

    def build_dataset(self, recordings, cfg: dict | None = None):
        """Modules 2+3 over a list of recordings -> (X, y, groups).

        ⚠️ **WHERE THIS SITS RELATIVE TO THE FOLD SPLIT — read before you add a
        learned preprocessing step.** This method runs stages 2-3 over *every*
        recording ONCE, and `evaluate()` then splits the resulting `X` into
        folds. Everything the scaffold ships for stage 2 is **stateless**:
        `bandpass_notch`, `wavelet_denoise` and every `denoise()` remedy are fixed
        transforms, or estimate their parameters from *one epoch's own samples*
        (the VisuShrink σ), and none of them looks at labels or at any other
        recording. A stateless transform commutes with the split: computing it
        before or inside the fold gives bit-identical features, so doing it once
        up front is a speed decision with no statistical content.

        **That safety does NOT extend to a transform that LEARNS from the data**,
        and the menus in this scaffold actively encourage you to try one:

        * a **CSP** spatial filter (the single strongest move on the BCI track),
        * **per-dataset / per-cohort normalisation** — a z-score whose mean and
          scale come from the pooled set rather than from one recording,
        * a PCA/ICA basis, a learned artifact template, a dictionary,
        * anything whose *fit* sees more than the one recording in front of it.

        Fit any of those here and every fold's "held-out" subjects have already
        contributed to the transform that built the training features. The
        leakage guard on the fold (`assert_no_subject_leak`) will still pass —
        it checks group ids, and it cannot see that a filter learned from them
        — so the number will look clean and be wrong.

        **The harness does not enforce this for you.** If you add a learned
        stage, do not put it here: fit it inside the fold. The two supported
        places are `select_features()` (which `evaluate()` already calls on the
        training fold only) and the classifier pipeline you pass as `clf=` (an
        `sklearn.pipeline.Pipeline` step is refit per fold by `clone()`). Say in
        `RESULTS.md` which of the two you used and why — "we fit CSP inside the
        fold" is a sentence Criterion 8 is looking for."""
        cfg = self._cfg(cfg)
        Xs, ys, gs = [], [], []
        for rec in recordings:
            X, y, g = self._features_for(rec, cfg)
            Xs.append(np.asarray(X)); ys.append(np.asarray(y))
            gs.append(np.asarray([g] * len(y)))
        return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)

    # ================= module 4 — FEATURE SELECTION ==========================
    def select_features(self, X, y, cfg: dict | None = None):
        """Fit and return the selection stage **on training data only**.

        `evaluate()` calls this INSIDE each fold, which is the whole point: a
        selector fit on all the data has already seen the test labels, and the
        resulting number is fiction (Ch. 12, §16.2). If you call it yourself,
        call it on the training fold — never on `X` entire.

        The default is pass-through (keep every feature), because several tracks
        have only ~10-30 features and pruning them may cost more than it saves.
        Switch with the config, e.g. ``cfg={"select": "anova", "select_k": 12}`` or
        ``cfg={"select": "lasso", "select_C": 0.1}``; see `make_selector` for the
        menu and the trade-offs. Override this method outright if you want
        something the menu does not offer (RFE, a CSP-style spatial filter, a
        domain-motivated hand-picked subset).

        The two **embedded** options ignore `select_k` entirely — `"tree"` cuts at
        an importance threshold and `"lasso"` keeps whatever L1 did not zero — so
        how many features survive is a *result*, not a setting. This method prints
        that count (and, for `"lasso"`, what to do if the penalty erased
        everything).

        **`select_k` is clamped to the track's feature count, out loud.** Four of
        the six tracks have fewer than the default `select_k = 20` features, so a
        naive `k=20` would silently keep everything and teach you that selection
        "does nothing". Instead of swallowing sklearn's warning, this method says
        so once — `note()` prints it and it lands in `rep["notes"]` — because a
        no-op you know about is a result and a no-op you don't is a bug.
        """
        cfg = self._cfg(cfg)
        X = np.asarray(X)
        kind = str(cfg.get("select", "none") or "none").lower()
        k_req = int(cfg.get("select_k", 20))
        n_feat = X.shape[1]
        k = min(k_req, n_feat)
        if kind in ("anova", "mutual_info", "mi"):
            if k_req >= n_feat:
                self.note(
                    f"select={kind!r}: select_k={k_req} >= n_features={n_feat}, so k was reduced "
                    f"to {n_feat} and EVERY feature is kept — feature selection is a NO-OP at this "
                    f"k on this track. Lower select_k (try {max(2, n_feat // 2)}) if you want this "
                    f"stage to actually choose.")
            else:
                self.note(f"select={kind!r}: keeping {k} of {n_feat} features "
                          f"({n_feat - k} dropped per training fold).")
        sel = make_selector(kind, k=k, seed=int(cfg.get("seed", 0)),
                            C=float(cfg.get("select_C", 1.0)))
        # Surface sklearn's own complaints ONCE each instead of swallowing them:
        # "features [6 7 8] are constant" is a real finding about your feature set.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sel = sel.fit(X, np.asarray(y))
        for w in caught:
            self.note(f"selection warning — {w.message}")

        # How many features actually survived? For the EMBEDDED selectors the
        # answer is data-dependent (nobody set a k), so it is the only way to know
        # whether the stage did anything — and for L1 it is the headline result.
        n_kept = _n_selected(sel, n_feat)
        if kind in ("lasso", "l1"):
            if n_kept == 0:
                # every coefficient shrank to zero: the penalty is too strong for
                # this fold. Falling back loudly beats a confusing zero-column crash.
                self.note(
                    f"select='lasso': C={cfg.get('select_C', 1.0)} drove EVERY coefficient to zero "
                    f"on this training fold, so nothing would be left to classify. Falling back to "
                    f"keeping all {n_feat} features for this fold — raise select_C (try "
                    f"{float(cfg.get('select_C', 1.0)) * 10:g}) if you want L1 to choose rather than erase.")
                return PassThrough().fit(X, y)
            self.note(f"select='lasso' (C={cfg.get('select_C', 1.0)}): L1 kept {n_kept} of {n_feat} "
                      f"features with a non-zero coefficient ({n_feat - n_kept} zeroed per training "
                      f"fold). Re-fit on another fold the surviving set may differ — L1 picks ONE of "
                      f"each correlated pair, so check the stability before calling this 'the' "
                      f"feature set.")
        elif kind in ("tree", "model", "importance") and n_kept is not None:
            self.note(f"select='tree': kept {n_kept} of {n_feat} features above the importance "
                      f"threshold (a forest SPLITS credit between correlated features, so "
                      f"near-duplicates tend to survive together — contrast select='lasso').")
        elif n_kept == 0:
            self.note(f"select={kind!r} selected ZERO features on this training fold; falling back "
                      f"to keeping all {n_feat}. Loosen the selector before reading any number.")
            return PassThrough().fit(X, y)
        return sel

    # ================= module 5 — CLASSIFICATION =============================
    def baseline(self, cfg: dict | None = None):
        """The SUPPLIED starting classifier (see `default_baseline`). Swap in any
        sklearn-compatible estimator via `evaluate(..., clf=...)`.

        Takes the cfg so the **imbalance handling is a visible option**, not a
        hidden constructor argument: `cfg={"imbalance": "none"|"balanced"|
        "balanced_subsample"|"resample"|"smote"|"adasyn"|"threshold"}` — the menu
        and its trade-offs are on `default_baseline`.

        The book's k-NN -> SVM -> random forest -> tuned-RF ladder (§16.3) is an
        *illustrative* progression, not a required route: it shows what a healthy
        iteration history looks like, with explicitly simulated numbers. Your
        ladder may reasonably keep one learner and spend every iteration on
        features instead — as long as each rung is measured under the same honest
        harness and you can say why you climbed the way you did.
        """
        raise NotImplementedError

    def _baseline(self, cfg: dict | None = None):
        """Internal: build the baseline for this cfg, tolerating a student's older
        zero-argument `baseline(self)` override (checked by signature, so a real
        TypeError raised *inside* an override is not silently swallowed)."""
        import inspect
        try:
            takes_cfg = len(inspect.signature(self.baseline).parameters) > 0
        except (TypeError, ValueError):                     # noqa: BLE001
            takes_cfg = True
        return self.baseline(cfg) if takes_cfg else self.baseline()

    # ================= module 6 — INFERENCE ==================================
    def infer(self, model, rec: Recording, cfg: dict | None = None,
              allow_cfg_mismatch: bool = False):
        """Apply an ALREADY-FIT pipeline to one new recording -> predicted labels.

        FROZEN: this path fits nothing — no scaler, no selector, no classifier.
        That is what makes it inference rather than a second training run, and it
        is deliberately a separate method from `evaluate()`'s training path so the
        two can never be confused (§16.2, module 6).

        **TRAIN → INFER CONFIG PARITY (this is enforced, not advised).**
        Preprocessing and feature extraction must run under the *same* cfg the
        model was fit under, or the classifier is handed a feature matrix from a
        different pipeline than the one it learned — a silent, invisible failure
        that no metric in the report can catch, because the report was computed
        on the *other* pipeline. `train_baseline()` therefore stores the resolved
        training cfg on the `FittedModel`, and this method:

        * **`cfg=None` (the normal case)** — resolves the cfg from `model.cfg`,
          i.e. exactly what the model was trained with. It does **not** fall back
          to the adapter's defaults, which is what used to make
          ``m = track.train_baseline(recs, cfg={"spectral_method": "ar"})``
          followed by ``track.infer(m, rec)`` quietly predict from Welch features.
        * **an explicit `cfg`** — is a *partial override of the training cfg*, not
          a fresh config: the keys you name are applied on top of `model.cfg`, and
          every key you omit keeps its training-time value (it does **not** fall
          back to the adapter's defaults). Only the keys you actually named are
          checked, and only the feature-affecting ones (`feature_cfg_keys()`) —
          so `cfg={"imbalance": "balanced"}` is never a conflict, while asking for
          a different `spectral_method` than the model was fit under raises
          `ValueError`. Pass `allow_cfg_mismatch=True` to acknowledge a deliberate
          mismatch (it is then recorded as a harness note, so it lands in the
          report rather than in nobody's memory).
        * **a model with no `.cfg`** (a bare sklearn estimator you fit yourself) —
          there is nothing to check against, so the adapter's own cfg is used and
          the parity guarantee is yours to keep.
        """
        cfg = self._infer_cfg(model, cfg, allow_cfg_mismatch)
        X, _, _ = self._features_for(rec, cfg)
        return np.asarray(model.predict(np.asarray(X)))

    def _infer_cfg(self, model, cfg, allow_cfg_mismatch: bool = False) -> dict:
        """Resolve the cfg an inference call must run under — see `infer`.

        **A partial override is resolved against the TRAINING cfg, not against the
        adapter's defaults.** `infer(model, rec, cfg={"imbalance": "balanced"})`
        means "same pipeline, but this one knob", so every key the caller did not
        mention inherits the value the model was *trained* with. Filling the gaps
        from `self.cfg`/`DEFAULT_CFG` instead would both (a) run the wrong feature
        pipeline and (b) then report the resulting difference as a "conflict" the
        caller never asked for — a false alarm on a key they never touched.

        Only the keys the caller **actually named** are compared against the
        training cfg, and only those in `feature_cfg_keys()`: stage-4/5 keys
        (`select`, `imbalance`, ...) cannot change the feature matrix at inference
        time, so restating or changing one is never a parity conflict."""
        train_cfg = model.__dict__.get("cfg") if hasattr(model, "__dict__") else None
        if not isinstance(train_cfg, dict) or not train_cfg:
            # nothing recorded (a bare sklearn estimator): old behaviour.
            return self._cfg(cfg)
        train_cfg = _ResolvedCfg(train_cfg)
        if cfg is None:
            return train_cfg
        if not isinstance(cfg, _ResolvedCfg):
            self._validate_cfg_keys(cfg or {}, where=f"{type(self).__name__}.infer(): cfg=")
        explicit = dict(cfg or {})
        # the caller's keys on top of the TRAINING cfg (DEFAULT_CFG only fills keys
        # the training cfg never carried, e.g. a hand-built FittedModel).
        merged = dict(DEFAULT_CFG)
        merged.update(train_cfg)
        merged.update(explicit)
        want = _ResolvedCfg(merged)
        feature_keys = self.feature_cfg_keys()
        diff = {k: (train_cfg.get(k), explicit[k]) for k in explicit
                if k in feature_keys and not _cfg_values_equal(train_cfg.get(k), explicit[k])}
        if not diff:
            return want
        detail = "\n".join(f"  * {k!r}: trained with {t!r}, inference asked for {w!r}"
                           for k, (t, w) in sorted(diff.items()))
        if not allow_cfg_mismatch:
            raise ValueError(
                f"{type(self).__name__}.infer(): the cfg you passed does not match the cfg this "
                f"model was TRAINED under, on {len(diff)} feature-affecting key(s):\n{detail}\n\n"
                "Features built under one pipeline and a classifier fit under another is a silent "
                "failure — nothing downstream can detect it, because the validated numbers came "
                "from the other pipeline. Either drop the cfg argument (inference then reuses "
                "`model.cfg`, which is the config you validated), or re-train under the cfg you "
                "actually want. If the mismatch is deliberate — e.g. you are deliberately "
                "measuring what a deployment-time preprocessing change costs — pass "
                "`allow_cfg_mismatch=True` and report that you did.")
        self.note(
            f"infer(): running with a cfg that DIFFERS from the training cfg on "
            f"{', '.join(sorted(diff))} (allow_cfg_mismatch=True). The features this model is "
            f"scoring were not built by the pipeline it was fit on — say so in the report, "
            f"because no metric here can tell you what it cost.")
        return want

    # ================= module 7 — REPORTING ==================================
    def report(self, rep: dict, show: bool = True, **kw) -> dict:
        """Turn an `evaluate()` result into the panel §16.8 says to lead with:
        confusion matrix first, then the primary metric WITH ITS SPREAD, then
        macro-F1 / balanced accuracy. Thin wrapper over `report.summarize_results`
        so every track reports the same way; see `report.py` for the pieces
        (including `plot_hypnogram` for staging-style tracks)."""
        from report import summarize_report
        return summarize_report(rep, show=show, **kw)

    # ---- shared discipline (never overridden) ----
    def _make_report(self, y_true, y_pred, groups=None, folds=None,
                     split_unit=None, n_groups=None, primary_metric=None,
                     group_unit=None) -> dict:
        """Pooled panel + PER-GROUP and PER-FOLD panels + the spread summary.
        One place, so `evaluate`, `holdout_score` and the two-mode tracks all
        report a result the same shape."""
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        rep = M.report(yt, yp, labels=self.meta.classes)
        rep["y_true"], rep["y_pred"] = yt, yp
        rep["split_unit"] = split_unit or self.meta.split_unit
        rep["primary_metric"] = primary_metric or (
            self.meta.default_metrics[0] if self.meta.default_metrics else "cohens_kappa")
        if groups is not None:
            groups = np.asarray(groups)
            rep["y_group"] = groups
            rep["per_group"] = per_group_metrics(yt, yp, groups, self.meta.classes, key="group")
            rep["n_groups"] = int(n_groups if n_groups is not None else len(set(groups.tolist())))
        else:
            rep["per_group"] = []
            rep["n_groups"] = int(n_groups or 0)
        if folds is not None:
            folds = np.asarray(folds)
            rep["y_fold"] = folds
            rep["per_fold"] = per_group_metrics(yt, yp, folds, self.meta.classes, key="fold")
        else:
            rep["per_fold"] = []

        # Which unit carries the spread? Per-group is what §16.3 asks for, but on
        # record-level tracks each group holds ONE label, so per-group metrics are
        # undefined and per-fold is the honest fallback. Decide from the data.
        pm = rep["primary_metric"]
        g_ok = sum(1 for r in rep["per_group"] if np.isfinite(r.get(pm, float("nan"))))
        gu = group_unit or self.meta.split_unit
        if g_ok >= 3:
            rows, unit = rep["per_group"], gu
        elif rep["per_fold"]:
            rows, unit = rep["per_fold"], "fold"
        else:
            rows, unit = rep["per_group"], gu
        rep["spread_unit"] = unit
        rep["spread_rows"] = rows
        rep["spread"] = {m: metric_spread(rows, m) for m in
                         ("accuracy", "cohens_kappa", "macro_f1", "balanced_accuracy")}
        rep["summary"] = spread_line(pm, rep["spread"].get(pm, {}), unit)
        rep["notes"] = list(getattr(self, "notes", []))
        return rep

    def evaluate(self, X, y, groups, clf=None, cfg=None):
        """Subject/record-independent evaluation with a leakage guard on every
        fold, threading modules 4 (selection) and 5 (classification) INSIDE the
        fold. Uses leave-one-group-out when there are few groups (e.g. a handful
        of subjects) and 5-fold GroupKFold when there are many (e.g. thousands of
        ECG records) so it stays fast while remaining group-independent.

        Returns a dict — the honest panel, plus the spread the book requires:

        | key | what it holds |
        |---|---|
        | `accuracy`, `cohens_kappa`, `macro_f1`, `balanced_accuracy` | **pooled** over all folds |
        | `labels`, `confusion` | the confusion matrix, in `labels` order |
        | `y_true`, `y_pred`, `y_group`, `y_fold` | aligned per-prediction arrays |
        | `per_group` | one panel per held-out group: `{group, n, accuracy, cohens_kappa, ...}` |
        | `per_fold` | the same, per CV fold |
        | `spread` | `{metric: {mean, std, min, max, n}}` over `spread_rows` |
        | `spread_unit`, `spread_rows` | which unit the spread was taken over (`subject`/`record`/`fold`) and the rows used |
        | `summary` | the one-line "mean kappa 0.61 (range 0.34-0.73 across 8 subjects)" sentence |
        | `split_unit`, `n_groups`, `primary_metric` | the track's declared discipline |

        **Report the spread, not just the pooled number** (§16.3, definition of
        done #2): a pooled kappa hides that one subject scored 0.34. Pass the
        whole dict to `self.report(rep)` for the §16.8 panel.
        """
        from sklearn.base import clone
        from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
        cfg = self._cfg(cfg)
        clf = clf or self._baseline(cfg)
        groups = np.asarray(groups)
        n_groups = len(set(groups.tolist()))
        if n_groups <= int(cfg.get("loso_max_groups", 12)):
            folds = LeaveOneGroupOut().split(X, y, groups)
        else:
            folds = GroupKFold(n_splits=min(int(cfg.get("cv_max_splits", 5)), n_groups)).split(X, y, groups)
        yt, yp, gg, ff = [], [], [], []
        for k, (tr, te) in enumerate(folds):
            C.assert_no_subject_leak(groups[tr], groups[te])
            # module 4 — selection is fit on the TRAINING fold only (§16.2)
            sel = self.select_features(X[tr], y[tr], cfg)
            Xtr, Xte = _transform(sel, X[tr]), _transform(sel, X[te])
            # module 5 — clone per fold: the shared baseline is stateless (fresh
            # RandomForestClassifier), but a student swapping in a warm-start /
            # partial-fit model here would otherwise silently carry state across folds.
            fold_clf = clone(clf)
            fold_clf.fit(Xtr, y[tr])
            yp.extend(fold_clf.predict(Xte)); yt.extend(y[te])
            gg.extend(groups[te].tolist()); ff.extend([k] * len(te))
        return self._make_report(yt, yp, groups=gg, folds=ff, n_groups=n_groups)

    # ---- per-track hold-out evaluation ----
    def train_baseline(self, train_recs, clf=None, cfg=None):
        """Modules 2-5 on the FULL training set -> a frozen `FittedModel`
        (selector + classifier) ready for `infer()` / `write_submission()`.

        The **resolved** cfg is stored on the returned `FittedModel`, and that is
        the config `infer()` / `write_submission()` reuse by default — so a model
        trained under a custom preprocessing/spectral/selection recipe cannot
        later be asked to predict from default-pipeline features. See
        `infer.__doc__` for what happens if you pass a conflicting cfg anyway."""
        cfg = self._cfg(cfg)
        X, y, g = self.build_dataset(train_recs, cfg)
        sel = self.select_features(X, y, cfg)               # fit on train only — no test data here
        clf = clf or self._baseline(cfg)
        clf.fit(_transform(sel, X), y)
        return FittedModel(clf=clf, selector=sel, cfg=cfg)

    def write_submission(self, test_recs, path, model, cfg=None,
                         allow_cfg_mismatch: bool = False):
        """Write a hold-out `predictions.csv` in the track's granularity:
        `record,epoch,label` (epoch-level) or `record,label` (record-level).
        Uses module 6 (`infer`) — the frozen path, no fitting.

        Like `infer()`, this reuses **the cfg the model was trained under** when
        `cfg` is omitted, and refuses a cfg that conflicts with it. A submission
        generated from a different pipeline than the one you validated is the one
        mistake here that no held-out score can reveal to you."""
        import csv
        gran = getattr(self.meta, "submission_granularity", "epoch")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["record", "epoch", "label"] if gran == "epoch" else ["record", "label"])
            for rec in test_recs:
                preds = self.infer(model, rec, cfg, allow_cfg_mismatch=allow_cfg_mismatch)
                if gran == "epoch":
                    for i, p in enumerate(preds):
                        w.writerow([rec.group, i, p])
                else:
                    w.writerow([rec.group, preds[0]])
        return path

    def holdout_score(self, train_recs, holdout_recs, clf=None, cfg=None):
        """Instructor hold-out scoring: fit on train, score on a LABELED hold-out.
        Reports per-held-out-group results too, so the hold-out number also
        carries its spread rather than one pooled figure.

        On record-level tracks (one label per recording) a per-group metric is
        undefined — each group holds a single label — so `spread` comes back NaN
        and the honest hold-out report is the pooled number plus the confusion
        matrix, stated as such."""
        cfg = self._cfg(cfg)
        model = self.train_baseline(train_recs, clf, cfg)
        yt, yp, gg = [], [], []
        for rec in holdout_recs:
            _, y, g = self._features_for(rec, cfg)
            pred = self.infer(model, rec, cfg)
            yp.extend(pred); yt.extend(y); gg.extend([g] * len(y))
        return self._make_report(yt, yp, groups=gg, folds=None)

    def run_smoke(self, cfg=None):
        """Offline end-to-end check: synthetic recordings -> preprocess -> features
        -> selection+classification inside LOSO folds. Proves the adapter conforms
        to the shared pipeline without any download.

        Note the ordering, which the generated notebooks mirror: stages 2-3 run
        over all recordings **before** `evaluate()` cuts the folds. That is safe
        for every preprocessing option this scaffold ships (all of them are
        stateless — see `build_dataset.__doc__`) and it is **not** safe for a
        learned transform you add yourself. `build_dataset.__doc__` says where
        such a stage has to live instead."""
        cfg = self._cfg(cfg)
        X, y, g = self.build_dataset(self.smoke(), cfg)
        return self.evaluate(X, y, g, cfg=cfg)

    # ---- the dataset card, rendered from metadata ----
    def dataset_card(self) -> str:
        m = self.meta
        return (
            f"# Dataset card — {m.name}\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Dataset | {m.dataset} (v{m.dataset_version}) |\n"
            f"| License / access | {m.license} · **{m.access}** |\n"
            f"| Signals | {', '.join(m.signals)} |\n"
            f"| Task | {m.task_type} — classes: {', '.join(map(str, m.classes))} |\n"
            f"| **Split unit (leakage)** | **{m.split_unit}** |\n"
            f"| Evaluation modes | {', '.join(m.eval_modes)} |\n"
            f"| Default metrics | {', '.join(m.default_metrics)} |\n"
            f"| Smoke-test records | {', '.join(map(str, m.smoke_test_records))} |\n"
            f"| Expected runtime | {m.expected_runtime} |\n"
            f"| DSP focus | {m.dsp_focus} |\n"
            f"| Hold-out submission | {m.submission_granularity}-level `predictions.csv` |\n"
            f"| Difficulty (1–5) | {m.difficulty} |\n\n"
            f"**Citation.** {m.citation}\n\n"
            f"**Source.** {m.url}\n")


# --------------------------------------------------------------- stage 5 options
class OverSampled(ClassifierMixin, BaseEstimator):
    """Random over-sampling of the minority classes, applied **inside `fit` only**
    — so it is re-done per CV fold on training data and never touches a test fold.
    (Resampling before the split is a classic leak: duplicated minority rows land
    on both sides and the score is fiction.)

    Scaling order does not matter for *this* wrapper, unlike `SMOTEd`: duplicating
    a row is the same operation in any coordinate system, so the wrapped
    pipeline's own scaler can stay where it is.

    Inherits `sklearn.base.BaseEstimator`, so `clone()`, `get_params(deep=True)`
    and **nested** `set_params(estimator__clf__max_depth=5)` all reach into the
    wrapped pipeline — which is what makes this usable inside `GridSearchCV`."""

    def __init__(self, estimator=None, random_state=0):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.base import clone
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        labs, counts = np.unique(y, return_counts=True)
        target = int(counts.max())
        idx = [np.arange(len(y))]
        for lab, n in zip(labs, counts):
            if n < target:
                pool = np.flatnonzero(y == lab)
                idx.append(rng.choice(pool, size=target - n, replace=True))
        take = np.concatenate(idx)
        self.classes_ = labs
        self.estimator_ = clone(self.estimator).fit(X[take], y[take])
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))


class SMOTEd(ClassifierMixin, BaseEstimator):
    """**SMOTE / ADASYN synthetic over-sampling, applied inside `fit` only** — the
    same fold-safety contract as `OverSampled`, for the same reason: any
    resampling done before the split puts minority information on both sides of it
    and the resulting score is fiction.

    The difference from `OverSampled` is what gets added. Random over-sampling
    duplicates *exact rows* the model has already seen; SMOTE (Chawla et al.,
    2002) instead **interpolates**: it picks a minority sample, picks one of its
    k nearest minority neighbours, and creates a new point somewhere on the line
    between them. ADASYN does the same but concentrates the synthesis near the
    minority points that are hardest to classify (those with the most
    majority-class neighbours).

    ⚠️ **The biosignal-specific catch.** "A point on the line between two
    neighbours" is a statement about feature space, not about physiology. If the
    two neighbours are epochs from *different patients*, or an N2 epoch and an N3
    epoch that happen to be near each other, the synthetic vector is a hybrid that
    no body could have produced — a heart rate from one patient blended with an
    HRV spread from another, a spectral edge frequency that is not consistent with
    the band powers beside it. With the small, heterogeneous cohorts these tracks
    use (a handful of subjects, a few dozen minority rows), that risk is real and
    the "neighbours" can be genuinely dissimilar. Look at what it synthesises
    before you trust what it earns you.

    ⚠️ **SCALING HAPPENS FIRST, AND IT HAS TO.** "Nearest minority neighbour" is
    a statement about *distance*, and distance in raw biosignal feature space is
    dominated by whichever column happens to carry the largest units. A waveform
    length in the hundreds and a normalised spectral entropy in [0, 1] are not
    comparable magnitudes, so a k-NN search over unscaled features effectively
    ranks neighbours by the big column alone — and the synthetic minority rows
    get interpolated along an axis you never chose. That would quietly undo the
    entire point of offering SMOTE as a considered alternative to duplication.

    So this wrapper does **not** resample and then hand the result to the wrapped
    pipeline. It rebuilds the pipeline as `imblearn.pipeline.Pipeline` with the
    sampler inserted **after the wrapped pipeline's transformers and before its
    final estimator** — i.e. `StandardScaler → SMOTE → RandomForest` for the
    supplied baseline. Everything is still fit strictly inside `fit()`, so the
    fold-safety contract is unchanged: the scaler is fit on the training fold, the
    sampler sees only scaled training rows, and at predict time imblearn bypasses
    the sampler entirely (samplers are training-time-only steps). A practical
    consequence worth knowing: the resampling is now **invariant to the units of
    your features**, so re-scaling a column no longer silently changes which
    synthetic patients get invented.

    On very small folds SMOTE's `k_neighbors` requirement (the minority class must
    have more than k members) is adapted downward automatically, and if the
    minority class has fewer than two members in a fold — where interpolation is
    undefined — this falls back to plain duplication with a loud warning rather
    than crashing mid-CV.

    Inherits `sklearn.base.BaseEstimator`, so `clone()`, `get_params(deep=True)`
    and nested `set_params(estimator__clf__max_depth=5)` reach into the wrapped
    pipeline properly (a shallow `set_params` would have created a top-level
    attribute literally named `"estimator__clf__max_depth"`, and every
    `GridSearchCV` candidate would have been the same model).
    """

    def __init__(self, estimator=None, random_state=0, k_neighbors=5, sampler="smote"):
        self.estimator = estimator
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.sampler = sampler

    def _make_sampler(self, k):
        try:
            from imblearn.over_sampling import ADASYN, SMOTE
        except ImportError as e:                            # pragma: no cover
            raise ImportError(
                "imbalance='smote'/'adasyn' needs imbalanced-learn — install it with "
                "`pip install 'imbalanced-learn>=0.12'` (it is listed in requirements.txt). "
                "Everything else in the scaffold runs without it; only this option needs it."
            ) from e
        if str(self.sampler).lower() in ("adasyn",):
            return ADASYN(random_state=self.random_state, n_neighbors=k)
        return SMOTE(random_state=self.random_state, k_neighbors=k)

    def _resampling_pipeline(self, sampler):
        """Wrap the estimator so the sampler runs **after** its transformers.

        The supplied baseline is `Pipeline([("scale", StandardScaler()),
        ("clf", RandomForestClassifier())])`; splicing the sampler in front of the
        FINAL step gives `scale → resample → clf`, so SMOTE's nearest-neighbour
        search happens in standardised space (see the class docstring for why
        that is not a detail). `imblearn.pipeline.Pipeline` is used because it is
        the only pipeline that knows a sampler is a fit-time-only step — it calls
        `fit_resample` during `fit` and skips the step entirely at `predict`.

        If the wrapped estimator is a bare classifier rather than a pipeline there
        is no scaler to run first, and none is invented: the sampler is simply
        prepended, which honestly reflects the (unscaled) model you built."""
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline as SkPipeline
        est = clone(self.estimator)
        if isinstance(est, SkPipeline) and len(est.steps) > 1:
            steps = list(est.steps)
            return ImbPipeline(steps[:-1] + [("resample", sampler)] + steps[-1:])
        return ImbPipeline([("resample", sampler), ("clf", est)])

    def fit(self, X, y):
        from sklearn.base import clone
        X, y = np.asarray(X), np.asarray(y)
        labs, counts = np.unique(y, return_counts=True)
        self.classes_ = labs
        n_min = int(counts.min())
        self.resample_note_ = None

        if len(labs) < 2 or n_min == int(counts.max()):
            # nothing to balance — no sampler, so the wrapped pipeline stands as-is
            self.estimator_ = clone(self.estimator).fit(X, y)
            return self
        if n_min < 2:
            # interpolation needs at least two points of the class to draw a line
            # between. One is not a neighbourhood — duplicate instead, and say so.
            self.resample_note_ = (
                f"imbalance={self.sampler!r}: the rarest class has {n_min} sample(s) in this "
                f"training fold, which is too few to interpolate between. Fell back to plain "
                f"duplication (imbalance='resample') for this fold. A synthetic minority built "
                f"from one example is not a richer minority — treat this fold's number with care.")
            warnings.warn(self.resample_note_, RuntimeWarning, stacklevel=2)
            os = OverSampled(estimator=self.estimator, random_state=self.random_state).fit(X, y)
            self.estimator_ = os.estimator_
            return self
        k = min(int(self.k_neighbors), n_min - 1)
        if k < int(self.k_neighbors):
            self.resample_note_ = (
                f"imbalance={self.sampler!r}: k_neighbors reduced {self.k_neighbors} -> {k} "
                f"because the rarest class has only {n_min} samples in this training fold. "
                f"Fewer neighbours means each synthetic point is interpolated from a smaller, "
                f"more local set — less diverse, and more likely to sit on top of a real row.")
            warnings.warn(self.resample_note_, RuntimeWarning, stacklevel=2)
        # scaler -> sampler -> classifier, all fit inside this call. The sampler
        # runs on SCALED features, so "nearest minority neighbour" is a distance
        # in standardised space and not a contest between measurement units.
        pipe = self._resampling_pipeline(self._make_sampler(k))
        try:
            self.estimator_ = pipe.fit(X, y)
        except (ValueError, RuntimeError) as e:             # ADASYN refuses outright when a
                                                            # minority point has no majority
                                                            # neighbour to weight against
            self.resample_note_ = (
                f"imbalance={self.sampler!r} refused this training fold ({e}); fell back to "
                f"plain duplication (imbalance='resample') so the CV can finish. Report the "
                f"fallback — a run that silently changed method is not the run you described.")
            warnings.warn(self.resample_note_, RuntimeWarning, stacklevel=2)
            os = OverSampled(estimator=self.estimator, random_state=self.random_state).fit(X, y)
            self.estimator_ = os.estimator_
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))


class ThresholdMoved(ClassifierMixin, BaseEstimator):
    """Leave the class weights alone, then **move the decision threshold** on the
    minority class. Trades precision for recall along the model's own probability
    ranking without retraining — and makes the operating point an explicit,
    reportable choice rather than sklearn's silent 0.5.

    Inherits `sklearn.base.BaseEstimator`, so `clone()`, `get_params(deep=True)`
    and nested `set_params(estimator__clf__n_estimators=400)` delegate into the
    wrapped pipeline properly — a shallow `set_params` would have silently made
    every `GridSearchCV` candidate identical."""

    def __init__(self, estimator=None, threshold=0.5, positive=None):
        self.estimator = estimator
        self.threshold = threshold
        self.positive = positive

    def fit(self, X, y):
        from sklearn.base import clone
        X, y = np.asarray(X), np.asarray(y)
        labs, counts = np.unique(y, return_counts=True)
        self.classes_ = labs
        self.positive_ = self.positive if self.positive is not None else labs[int(np.argmin(counts))]
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        P = self.estimator_.predict_proba(np.asarray(X))
        classes = np.asarray(getattr(self.estimator_, "classes_", self.classes_))
        j = int(np.flatnonzero(classes == self.positive_)[0])
        hit = P[:, j] >= float(self.threshold)
        # below threshold -> the best of the remaining classes (not blindly "the other one",
        # so this stays correct on a 4-class track like ECG)
        other = np.where(np.arange(P.shape[1]) == j, -np.inf, P)
        out = classes[np.argmax(other, axis=1)]
        out[hit] = self.positive_
        return out

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))


# --------------------------------------------------------------- shared baseline
def default_baseline(seed: int = 0, n_estimators: int = 200,
                     imbalance: str = "balanced", threshold: float = 0.5,
                     smote_k: int = 5):
    """The SUPPLIED baseline every track inherits: a leakage-safe pipeline whose
    scaler is refit inside each CV fold. Students improve on this, never a blank
    page — and "improve" may mean a different learner, better features, or both;
    the scaffold does not decide which.

    **CLASS IMBALANCE IS A CHOICE, NOT A DEFAULT.** Four of the six tracks are
    imbalanced (CTG's pathological class, ECG's AF/Noisy, HAR's postures, sleep's
    N1), and the rubric explicitly grades *how* you addressed it — so it must not
    be a hidden constructor argument nobody sees. Pick with the config, e.g.
    ``cfg={"imbalance": "threshold", "threshold": 0.35}``:

    | `imbalance` | What it does | Good when | Watch out |
    |---|---|---|---|
    | `"none"` | plain majority vote; the classifier sees the prior as it is | the imbalance is mild, or the prior is *real* and you want calibrated probabilities | the minority class can be predicted almost never and accuracy still looks fine |
    | `"balanced"` *(default)* | per-class weights ∝ 1/frequency inside the loss | a quick, no-extra-data lever; the default so the scaffold is not silently majority-biased | it inflates minority influence uniformly — noisy minority rows get amplified too, and probabilities stop being calibrated |
    | `"balanced_subsample"` | as above, recomputed per tree (random forest only) | forests on strongly skewed data; slightly steadier than `"balanced"` | forest-specific; means nothing for an SVM or a k-NN |
    | `"resample"` | random over-sampling of the minority **inside `fit`** | you want the *decision boundary* moved, not just the loss reweighted | duplicated rows invite overfitting; done outside the fold it is a **leak** (this one is inside, on purpose) |
    | `"smote"` | **synthetic** minority over-sampling **inside `fit`**: new rows interpolated between a minority sample and one of its k nearest minority neighbours | duplication is over-fitting and re-weighting is too blunt; you want a *denser*, less repetitive minority region for the boundary to sit in | the interpolated vector is only physiologically plausible if the neighbours were — blending two **different patients**, or an N2 and an N3 epoch, manufactures a body that does not exist. On these small biosignal cohorts (a handful of subjects, a few dozen minority rows) that is a live risk, not a footnote |
    | `"adasyn"` | as `"smote"`, but synthesises more where the minority is hardest to classify | the minority is easy in one region and contested in another | it densifies exactly the boundary region, so mislabelled or artifactual minority rows get amplified the most |
    | `"threshold"` | keep the model, move the operating point on the minority class | recall matters more than precision and you can name the cost ratio | you must *report* the threshold — a tuned threshold chosen on test data is as dishonest as a leak |

    **`"resample"` vs `"smote"` vs `"balanced"` — the distinction to state in the
    report.** `"balanced"` never adds a row; it changes only how much each existing
    row *counts* in the loss. `"resample"` adds rows that are exact copies of rows
    already there, so the minority region gets heavier but not one millimetre
    wider — a tree can still isolate each duplicated point. `"smote"` adds rows
    that are **new points in feature space**, so the minority region genuinely
    expands and the boundary is pushed rather than merely weighted. That expansion
    is the whole benefit and the whole danger: SMOTE will happily interpolate
    between two dissimilar epochs and hand the classifier a hybrid feature vector
    no physiology produced. If you use it, say what a synthetic minority sample
    *means* on your track — and be honest if the answer is "nothing physiological".

    Both `"smote"` and `"adasyn"` are fitted **inside the CV fold only** (the
    `SMOTEd` wrapper), never at transform/predict time, for the same reason
    `"resample"` is: synthesising minority points before the split puts
    interpolations of the test rows into the training set. They need the optional
    `imbalanced-learn` package; every other option here is sklearn-only.

    They also resample **after** the pipeline's `StandardScaler`, not before it:
    the wrapper splices the sampler in as `scale → resample → clf` so that
    "k nearest minority neighbours" is a distance in standardised space. Search
    for neighbours in raw units and whichever feature happens to be measured in
    the largest numbers decides who gets interpolated with whom — see
    `SMOTEd.__doc__`.

    Trade-off in one line: weights and resampling change what the model *learns*;
    threshold-moving changes only what you *do* with it. Doing nothing is also a
    defensible answer — but then say in the report that you looked and why you
    left it. That sentence is worth marks; a silent default is worth none.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    kind = (imbalance or "none").lower()
    weights = {"none": None, "off": None, "balanced": "balanced",
               "balanced_subsample": "balanced_subsample",
               "resample": None, "smote": None, "adasyn": None, "threshold": None}
    if kind not in weights:
        raise ValueError(f"unknown imbalance option {imbalance!r}; see default_baseline.__doc__")
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", RandomForestClassifier(
                         n_estimators=n_estimators, class_weight=weights[kind],
                         random_state=seed))])
    if kind == "resample":
        return OverSampled(estimator=pipe, random_state=seed)
    if kind in ("smote", "adasyn"):
        return SMOTEd(estimator=pipe, random_state=seed,
                      k_neighbors=int(smote_k), sampler=kind)
    if kind == "threshold":
        return ThresholdMoved(estimator=pipe, threshold=threshold)
    return pipe
