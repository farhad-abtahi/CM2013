"""
tracks.har — second reference track (Human Activity Recognition, raw IMU).

Proves the "same adapter, different signal" claim: identical interface, rubric,
and leakage discipline as the sleep track — only the windowing/features differ.
REAL data = UCI HAR **raw inertial signals** (not the 561-feature file), open,
direct download. `smoke()` synthesises IMU windows so CI/offline runs green.
"""
from __future__ import annotations
import numpy as np

from adapter import TrackAdapter, TrackMeta, Recording, default_baseline
import biosignals as bio

WIN = 128            # 2.56 s at 50 Hz (UCI-HAR window)
STEP = 64            # 50% overlap
# 3 coarse classes present in BOTH the synthetic smoke and real UCI-HAR:
#   static = still/postures, walk = walking on the level, stairs = up/down stairs.
# (bio.imu's 0/1/2 = rest/walk/run; here "run" stands in synthetically for the
#  higher-intensity "stairs" class so the label space matches the real loader.)
ACTIVITY = {0: "static", 1: "walk", 2: "stairs"}


def _rotation(rng, max_deg):
    """A random 3-D rotation matrix (|angle| <= max_deg per axis) — the synthetic
    stand-in for "every subject mounts the sensor differently"."""
    a = np.deg2rad(rng.uniform(-max_deg, max_deg, size=3))
    cx, cy, cz = np.cos(a); sx, sy, sz = np.sin(a)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _window_features(win3):
    """DSP features from a 3-axis accelerometer window (n_samples x 3)."""
    feats = []
    for a in range(3):
        x = win3[:, a]
        feats += [x.mean(), x.std(), np.mean(x ** 2)]         # mean, std, energy
        X = np.abs(np.fft.rfft(x - x.mean()))
        f = np.fft.rfftfreq(len(x), 1 / 50.0)
        feats.append(f[np.argmax(X)] if X.size > 1 else 0.0)  # dominant frequency
    mag = np.sqrt((win3 ** 2).sum(axis=1))
    feats += [mag.mean(), mag.std()]
    return feats


class HARTrack(TrackAdapter):
    meta = TrackMeta(
        name="Human activity (IMU)",
        dataset="UCI HAR (raw inertial signals)",
        dataset_version="1.0",
        license="Open (UCI ML Repository)",
        citation="Anguita D, et al. UCI HAR dataset (2013).",
        url="https://archive.ics.uci.edu/dataset/341/",
        signals=["tri-axial accelerometer (total_acc)"],   # gyro is available for extension
        task_type="multiclass",
        classes=["static", "walk", "stairs"],   # coarse map of real UCI-HAR's 6 activities
        split_unit="subject",
        default_metrics=["macro_f1", "balanced_accuracy"],
        smoke_test_records=["subj_1", "subj_2"],
        expected_runtime="~1–3 min (Colab, CPU)",
        dsp_focus="gravity/motion filtering, windowing, dominant-frequency (cadence), magnitude",
        difficulty=2,
        eval_modes=("new-subject",),
    )

    #: Capability declaration (see `TrackAdapter.SUPPORTED_CFG_KEYS`). HAR has
    #: exactly **one** stage-2 knob — the gravity/body-acceleration split — and it
    #: is the decision that defines the track. It has **no** stage-3 spectral
    #: option: `_window_features` reads a dominant frequency straight off an rFFT
    #: magnitude spectrum rather than integrating a PSD, so there is no estimator
    #: choice to make, and `cfg["spectral_method"]` is refused instead of ignored.
    #: The Chapter 8 denoise menu is likewise not wired in here.
    SUPPORTED_CFG_KEYS = frozenset({"gravity", "gravity_fc"})

    CFG_KEY_HINTS = {
        "spectral_method": "HAR's features are per-axis statistics plus a dominant frequency read "
                           "off an rFFT magnitude spectrum — no band power is integrated from a "
                           "PSD, so there is no spectral estimator to choose. The stage-2 decision "
                           "that DOES move this track's number is cfg['gravity'].",
    }

    def smoke(self, n_subjects=6, blocks=5, seed0=700):
        """Synthetic IMU cohort with the three nuisances that make real HAR
        subject-dependent — because a cohort without them scores 1.000 and makes
        every design decision look free.

        | Nuisance | How it is synthesised | Why it belongs here |
        |---|---|---|
        | **sensor orientation** | a random ±50° 3-D rotation per subject | a phone in a pocket is never mounted the same way twice; it moves *gravity* off the z axis, so raw per-axis means stop being comparable |
        | **cadence** | a per-subject time warp (0.72–1.32×) | one person's brisk walk has the same step rate as another's stairs, so the dominant-frequency feature alone cannot separate them across subjects |
        | **vigour** | per-subject, per-activity motion gain (0.55–1.75×) | some people take stairs gently and walk hard; amplitude features that work within a subject invert across subjects |

        Plus per-subject sensor noise and windows that straddle an activity
        boundary (kept, majority-labelled — real annotation is not frame-exact).
        LOSO lands around macro-F1 ≈ 0.80 with a wide per-subject spread, and the
        gravity-handling decision in `preprocess()` is worth a real, measurable
        jump. That is the point: the choice has a consequence."""
        recs = []
        for s in range(n_subjects):
            rng = np.random.default_rng(seed0 + s)
            warp = rng.uniform(0.72, 1.32)                  # per-subject cadence
            rot = _rotation(rng, 50.0)                      # per-subject mounting
            noise = rng.uniform(0.05, 0.16)
            vigour = {1: rng.uniform(0.55, 1.75), 2: rng.uniform(0.55, 1.75)}
            xs, ys = [], []
            for b in range(blocks):
                # generate at the subject's own pace, then resample onto a common clock
                t, acc, seg = bio.imu(duration=20.0 * warp, fs=50, seed=100 * s + b)
                n_out = int(20.0 * 50)
                idx = np.arange(n_out) * warp
                src = np.arange(len(acc))
                a = np.stack([np.interp(idx, src, acc[:, k]) for k in range(3)], axis=-1)
                sg = np.round(np.interp(idx, src, seg)).astype(int)
                for cls, gain in vigour.items():            # scale MOTION, keep gravity
                    m = sg == cls
                    if m.any():
                        mu = a[m].mean(axis=0)
                        a[m] = mu + gain * (a[m] - mu)
                a = (a @ rot.T) + noise * rng.standard_normal((n_out, 3))
                for start in range(0, n_out - WIN, STEP):
                    lab = sg[start:start + WIN]
                    vals, cnt = np.unique(lab, return_counts=True)
                    if cnt.max() < 0.7 * WIN:               # too mixed to label honestly
                        continue
                    xs.append(a[start:start + WIN])
                    ys.append(ACTIVITY[int(vals[cnt.argmax()])])
            recs.append(Recording(
                group=f"subj_{s+1}", fs=50.0,
                epochs={"acc": np.array(xs)}, labels=np.array(ys)))
        return recs

    # ---- module 2: PREPROCESSING (the gravity decision, made explicit) ----
    def preprocess(self, rec: Recording, cfg=None) -> Recording:
        """Identity by default — the baseline sees the accelerometer as mounted,
        gravity and all. That is a starting point, not a recommendation.

        `cfg={"gravity": ...}` makes the classic IMU decision a reportable knob:

        | `gravity` | What it does | Buys you | Costs you |
        |---|---|---|---|
        | `"none"` *(default)* | nothing | orientation itself is informative — sitting vs. standing differ mainly in *where gravity points* | that same orientation is a nuisance across subjects, who mount the sensor differently |
        | `"mean"` | subtract each window's per-axis mean | a cheap high-pass; body acceleration only, orientation-invariant | you throw away the posture cue that separates the static classes from each other |
        | `"highpass"` | 4th-order Butterworth above `gravity_fc` (default 0.4 Hz) | the textbook gravity/body split (§9.2–9.4), with a stated cut-off | the cut-off is now yours to justify — too high and you clip slow walking |

        `gravity` (and `gravity_fc`) are the **only** cfg keys this method reads,
        which is why they are the only stage-2 keys in `SUPPORTED_CFG_KEYS`. The
        Chapter 8 denoise menu is not wired in here, so `cfg={"preprocess":
        "denoise", ...}` raises rather than doing nothing quietly;
        `adapter.denoise()` is importable if you want to call it from this method
        yourself, and `track.declare_cfg_keys(...)` then makes its knobs real.

        On this track's synthetic cohort the sensors are deliberately mis-mounted
        per subject, so removing gravity is worth a large jump in cross-subject
        macro-F1. On the real UCI-HAR data it is not automatically the right call
        — the six real activities include postures that gravity distinguishes.
        Measure it, then argue for it."""
        cfg = self._cfg(cfg)
        how = str(cfg.get("gravity", "none") or "none").lower()
        if how in ("none", "off", ""):
            return rec
        acc = np.asarray(rec.epochs["acc"], float)
        if how == "mean":
            out = acc - acc.mean(axis=1, keepdims=True)
        elif how in ("highpass", "hp", "butter"):
            from scipy.signal import butter, filtfilt
            fc = float(cfg.get("gravity_fc", 0.4))
            b, a = butter(4, fc / (rec.fs / 2.0), "high")
            out = filtfilt(b, a, acc, axis=1)
        else:
            raise ValueError(f"unknown gravity option {how!r}; see HARTrack.preprocess.__doc__")
        return rec.replace_epochs({**rec.epochs, "acc": out})

    def extract_features(self, rec: Recording, cfg=None):
        """Module 3 — one feature vector per window (per-axis stats, magnitude,
        dominant frequency). Separating gravity from body acceleration is a
        *preprocessing* decision (module 2, `preprocess`), not a feature one:
        a low-pass/high-pass split, a per-window mean subtraction, or leaving
        gravity in and letting the orientation features use it are all arguable —
        pick one and say which."""
        X = [_window_features(w) for w in rec.epochs["acc"]]
        return np.array(X), np.asarray(rec.labels), rec.group

    def baseline(self, cfg=None):
        """Module 5 — the supplied random forest. `cfg["imbalance"]` selects how
        class imbalance is handled ("none" | "balanced" | "balanced_subsample" |
        "resample" | "smote" | "adasyn" | "threshold"); the trade-off table lives
        on `adapter.default_baseline`. It is a *choice you report*, not a default
        nobody sees — the rubric grades it.

        `"smote"`/`"adasyn"` synthesise minority rows by INTERPOLATING between
        real neighbours instead of duplicating them (`"resample"`) or reweighting
        the loss (`"balanced"`) — fold-safe via `adapter.SMOTEd`, and needing the
        optional `imbalanced-learn` package. Before you use it, ask what a point
        halfway between two of THIS track's epochs would physically be."""
        cfg = self._cfg(cfg)
        return default_baseline(seed=int(cfg.get("seed", 0)),
                                imbalance=cfg.get("imbalance", "balanced"),
                                threshold=float(cfg.get("threshold", 0.5)),
                                smote_k=int(cfg.get("smote_k", 5)))

    _URLS = (
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    )

    def download(self, cache_dir, subset=None):
        """Download + extract the UCI-HAR archive into cache_dir so that
        `cache_dir/UCI HAR Dataset/{train,test}/Inertial Signals/` exists. Needs
        network. The modern UCI zip wraps a nested `UCI HAR Dataset.zip`; both
        layouts are handled."""
        import os, io, zipfile, urllib.request
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.isdir(os.path.join(cache_dir, "UCI HAR Dataset", "train", "Inertial Signals")):
            return cache_dir                                   # already extracted
        blob = None
        for url in self._URLS:
            try:
                blob = urllib.request.urlopen(url, timeout=180).read(); break
            except Exception:                                  # noqa: BLE001 — try the next mirror
                continue
        if blob is None:
            raise RuntimeError("could not download UCI-HAR from any known mirror")

        def _extract(data):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(cache_dir)
                for n in z.namelist():
                    if n.endswith("UCI HAR Dataset.zip"):      # nested archive in the modern zip
                        _extract(z.read(n))
        _extract(blob)
        print(f"[har] extracted UCI-HAR into {cache_dir}")
        return cache_dir

    def load(self, cache_dir):
        """Read UCI-HAR raw inertial signals grouped by SUBJECT."""
        import os
        recs = {}
        for split in ("train", "test"):
            base = os.path.join(cache_dir, "UCI HAR Dataset", split)
            sig = os.path.join(base, "Inertial Signals")
            if not os.path.isdir(sig):
                continue
            ax = [np.loadtxt(os.path.join(sig, f"total_acc_{c}_{split}.txt")) for c in "xyz"]
            acc = np.stack(ax, axis=-1)                                   # [n_win, 128, 3]
            y = np.loadtxt(os.path.join(base, f"y_{split}.txt")).astype(int)
            subj = np.loadtxt(os.path.join(base, f"subject_{split}.txt")).astype(int)
            # UCI-HAR labels: 1 WALKING, 2 UPSTAIRS, 3 DOWNSTAIRS, 4 SITTING, 5 STANDING, 6 LAYING
            # -> coarse {walk, stairs, static}, matching the synthetic smoke's label space.
            coarse = {1: "walk", 2: "stairs", 3: "stairs", 4: "static", 5: "static", 6: "static"}
            for s in np.unique(subj):
                m = subj == s
                key = f"subj_{s}"
                recs.setdefault(key, {"acc": [], "y": []})
                recs[key]["acc"].append(acc[m]); recs[key]["y"] += [coarse[i] for i in y[m]]
        out = []
        for k, v in recs.items():
            out.append(Recording(group=k, fs=50.0,
                                 epochs={"acc": np.vstack(v["acc"])},
                                 labels=np.array(v["y"])))
        return out


if __name__ == "__main__":
    t = HARTrack()
    print(t.dataset_card())
    rep = t.run_smoke()
    print("SMOKE (synthetic) LOSO:", {k: rep[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "n_groups")})
    print("spread:", rep["summary"])          # never the pooled number alone (§16.3)
