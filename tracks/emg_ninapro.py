"""
tracks.emg_ninapro — EMG gesture track: Ninapro DB1 (surface EMG hand gestures).

Real data: Ninapro DB1 (ninapro.hevs.ch) — 27 subjects, **10-channel surface EMG @ 100 Hz**,
52 hand movements over 3 exercises, each repeated 10×. Direct per-subject zip download
(no login/agreement); **cite Atzori et al. 2014** (CC BY-NC-ND — research/teaching use).

Note on the signal: DB1 ships a **rectified, low-pass EMG envelope** (non-negative amplitude),
not raw bipolar sEMG — so classic raw-sEMG features (zero-crossings, slope-sign changes) are
*not* meaningful here. Good teaching point: know your signal. We use amplitude/energy features
(MAV, RMS, waveform length, variance) + a coarse mean-frequency per channel.

Two evaluation modes (electrode placement makes cross-subject sEMG much harder):
  * **within-subject** — leave-repetitions-out inside each subject;
  * **new-subject** — leave-one-subject-out.

`smoke()` synthesises per-gesture channel-activation patterns so CI/offline runs green.
"""
from __future__ import annotations
import numpy as np

from adapter import (TrackAdapter, TrackMeta, Recording, default_baseline,
                     denoise)

FS = 100.0
WIN = 20          # 200 ms window
STEP = 15         # 150 ms step
N_CH = 10


def _win_features(win, fs=FS):
    """Amplitude/energy features per channel from a [win_samples, n_ch] window."""
    feats = []
    for ch in range(win.shape[1]):
        x = win[:, ch]
        mav = float(np.mean(np.abs(x)))
        rms = float(np.sqrt(np.mean(x ** 2)))
        wl = float(np.sum(np.abs(np.diff(x))))                 # waveform length
        var = float(np.var(x))
        X = np.abs(np.fft.rfft(x - x.mean())) + 1e-12          # coarse mean frequency
        f = np.fft.rfftfreq(len(x), 1 / fs)
        mnf = float(np.sum(f * X) / np.sum(X))
        feats += [mav, rms, wl, var, mnf]
    return feats


class EMGNinaproTrack(TrackAdapter):
    meta = TrackMeta(
        name="EMG gesture",
        dataset="Ninapro DB1 (surface EMG, exercise E1)",
        dataset_version="1.0",
        license="CC BY-NC-ND — cite Atzori et al. (2014)",
        citation="Atzori M, et al. Sci Data (2014). Ninapro DB1.",
        url="https://ninapro.hevs.ch/instructions/DB1.html",
        signals=["10-channel surface EMG envelope @ 100 Hz"],
        task_type="multiclass",
        classes=[f"g{i:02d}" for i in range(1, 13)],           # 12 finger movements (E1)
        split_unit="subject",
        default_metrics=["macro_f1", "cohens_kappa"],
        smoke_test_records=["S1", "S2", "S3"],
        expected_runtime="~3–6 min on a subject subset (Colab, CPU)",
        dsp_focus="windowing, MAV/RMS/waveform-length/variance + mean-frequency per channel",
        difficulty=3,
        eval_modes=("within-subject", "new-subject"),
    )

    #: Capability declaration (see `TrackAdapter.SUPPORTED_CFG_KEYS`). EMG ships
    #: an identity `preprocess()` — DB1 already gives you a rectified, low-pass
    #: envelope, so the raw-sEMG moves are meaningless and the stage-2 work that
    #: matters (per-subject normalisation, window length) is code you write, not a
    #: key you set. Stage 3's mean-frequency term is read off an rFFT magnitude
    #: spectrum, not integrated from a PSD, so there is no estimator to choose.
    #: Both menus are therefore refused rather than silently ignored; declare your
    #: own keys once you have implemented the stage that reads them.
    SUPPORTED_CFG_KEYS = frozenset()

    CFG_KEY_HINTS = {
        "preprocess": "EMG's preprocess() is an identity stub. The stage-2 move that attacks this "
                      "track's cross-subject gap is per-channel normalisation WITHIN a subject "
                      "(i.e. a calibration procedure you must then declare) — write it, then "
                      "register its knobs with track.declare_cfg_keys(...).",
        "spectral_method": "EMG's features are MAV / RMS / waveform length / variance plus a mean "
                           "frequency taken from an rFFT magnitude spectrum — nothing here "
                           "integrates a PSD, so there is no spectral estimator to choose.",
    }

    # ---- synthetic smoke (offline / CI) ----
    def smoke(self, n_subjects=5, n_reps=6, win_per_rep=4, electrode_shift=2,
              bleed=0.5, rep_sd=0.40, win_noise=0.20, gain_sd=0.35, seed0=4000):
        """Synthetic sEMG whose difficulty comes from the three things that
        actually make sEMG gesture recognition hard — and that the old separable
        version (everything scored 1.000, including the leakage demo) hid:

        | Effect | How it is synthesised | What it teaches |
        |---|---|---|
        | **electrode shift** | each subject's activation pattern is rolled by up to ±`electrode_shift` channels, plus a per-subject smearing onto neighbours | an armband is never re-donned in the same rotation, so a model trained on subject A reads B's channel 4 as A's channel 6 — this is why **new-subject collapses** |
        | **per-repetition state** | every window of one repetition shares a per-repetition gain vector | consecutive windows of one contraction are NOT independent samples — which is precisely what a random-window split leaks |
        | **per-subject gain** | log-normal per-channel gain | skin impedance, muscle mass, electrode contact |

        The result is the pedagogy this track promises and previously refuted:
        within-subject ≈ 0.83 macro-F1, new-subject ≈ 0.33, and a leaky random
        split ≈ 0.84 — a lie the size of the gap between the last two.
        """
        recs = []
        gestures = self.meta.classes
        # a fixed, distinct 3-channel activation template per gesture (shared by all subjects)
        templates = []
        for gi in range(len(gestures)):
            trng = np.random.default_rng(9000 + gi)
            tv = np.zeros(N_CH)
            tv[trng.choice(N_CH, size=3, replace=False)] = trng.uniform(0.3, 0.6, size=3)
            templates.append(tv)
        for s in range(n_subjects):
            rng = np.random.default_rng(seed0 + s)
            shift = int(rng.integers(-electrode_shift, electrode_shift + 1))
            # crosstalk / volume conduction onto neighbouring electrodes, per subject
            M = np.eye(N_CH)
            b = bleed * rng.uniform(0.5, 1.5)
            for i in range(N_CH):
                M[i, (i + 1) % N_CH] += b * rng.uniform(0.3, 1.0)
                M[i, (i - 1) % N_CH] += b * rng.uniform(0.3, 1.0)
            M /= M.sum(axis=1, keepdims=True)
            ch_gain = np.exp(gain_sd * rng.standard_normal(N_CH))
            wins, labs, reps = [], [], []
            for gi, g in enumerate(gestures):
                base = np.roll(templates[gi] @ M, shift) * ch_gain
                for rep in range(1, n_reps + 1):
                    # one contraction = one shared state; its windows are correlated
                    rep_v = base * np.exp(rep_sd * rng.standard_normal(N_CH))
                    for _ in range(win_per_rep):
                        w = np.abs(rep_v[None, :] * (1 + win_noise * rng.standard_normal((WIN, N_CH)))
                                   + 0.6 * win_noise * rng.standard_normal((WIN, N_CH)))
                        wins.append(w); labs.append(g); reps.append(rep)
            recs.append(Recording(group=f"S{s + 1}", fs=FS,
                                  epochs={"emg": np.array(wins)},
                                  labels=np.array(labs), meta={"rep": np.array(reps)}))
        return recs

    # ---- module 2: PREPROCESSING (a stub you fill in — but read the warning) ----
    def preprocess(self, rec: Recording, cfg=None) -> Recording:
        """Identity by default: DB1 already ships a rectified, low-pass **envelope**,
        so the usual raw-sEMG moves (zero crossings, slope-sign changes, a fresh
        band-pass) are meaningless here — knowing that is half the exercise. What
        belongs in this stage on this track is per-channel normalisation *within a
        subject*, extra envelope smoothing, window length, and channel
        re-ordering; see the notebook's stage-2 menu.

        This method reads **no cfg key**, so none is declared in
        `SUPPORTED_CFG_KEYS` and `cfg={"preprocess": "denoise", ...}` raises here
        rather than being silently ignored. `adapter.denoise()` /
        `adapter.bandpass_notch()` are still importable — call them from this
        method, then `track.declare_cfg_keys(...)` to make the knobs real. Note
        that a per-subject normalisation constant is a **learned** quantity: fit it
        inside the fold, not over the whole cohort here (see
        `TrackAdapter.build_dataset.__doc__`).

        ⚠️ **CAUSALITY — this track has a real-time framing, so read this.**
        A myoelectric prosthesis or gesture controller classifies a **stream**: the
        decision for the window ending now must be made from samples up to now,
        inside a latency budget (typically ~100-300 ms end to end). Nothing in this
        scaffold enforces that, because the evaluation here is offline — every
        window is already on disk before a single feature is computed.

        That makes it very easy to build an academically clean pipeline that could
        never be deployed. The specific traps on this track:

        * `adapter.bandpass_notch()` defaults to `scipy.signal.filtfilt`, which is
          **zero-phase and therefore non-causal** — its output at each sample
          depends on later samples. Excellent offline; impossible on a live
          stream. Pass `causal=True` for the single forward `lfilter` pass a real
          controller would run, and report both numbers if you filter at all.
        * Any smoothing, normalisation or artifact removal computed **over the
          whole recording** (a per-subject max-voluntary scale, a global mean, a
          median filter spanning the window boundary) is the same mistake wearing
          different clothes: at run time that statistic does not exist yet. If you
          normalise per subject, say explicitly that it implies a **calibration
          procedure** the user performs before use — that is a deployment design
          decision, and naming it is worth marks.
        * Trimming gesture-onset transition windows makes the numbers prettier and
          removes exactly the windows a controller must actually classify.

        None of this is forbidden — offline analysis is what this scaffold does,
        and `filtfilt` is the right tool for it. What is forbidden is quoting an
        offline number as a real-time claim without saying so. State which regime
        each number belongs to, the same way you state the split unit.
        """
        return rec

    def extract_features(self, rec: Recording, cfg=None):
        """Module 3 — per-window, per-channel time-domain descriptors (MAV, RMS,
        waveform length, variance) plus a mean-frequency term. Rectification and
        envelope smoothing, per-channel normalisation, and a wider window are all
        candidate improvements that belong partly in `preprocess()` — the window
        length in particular trades responsiveness against feature stability."""
        X = [_win_features(w, rec.fs) for w in rec.epochs["emg"]]
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

    # ---- both evaluation modes ----
    def _build(self, recs, cfg=None):
        Xs, ys, subj, rep = [], [], [], []
        for r in recs:
            X, y, g = self._features_for(r, cfg)
            Xs.append(X); ys.append(y)
            subj.append(np.array([g] * len(y)))
            rep.append(np.asarray(r.meta.get("rep", np.zeros(len(y)))))
        return (np.vstack(Xs), np.concatenate(ys),
                np.concatenate(subj), np.concatenate(rep))

    def evaluate_modes(self, recs, cfg=None):
        """Both required modes, each with its PER-SUBJECT spread (§16.3): the gap
        between them is the finding, and a pooled pair of numbers hides which
        subjects drive it."""
        from sklearn.base import clone
        from sklearn.model_selection import GroupKFold
        cfg = self._cfg(cfg)
        X, y, subj, rep = self._build(recs, cfg)
        new_subject = self.evaluate(X, y, subj, cfg=cfg)       # LOSO/GroupKFold on subject
        yt, yp, gg, ff = [], [], [], []
        k = 0
        for s in sorted(set(subj.tolist())):                   # within subject: leave-reps-out
            m = subj == s
            Xs, ys, rs = X[m], y[m], rep[m]
            n_rep = len(set(rs.tolist()))
            if n_rep < 2 or len(set(ys.tolist())) < 2:
                continue
            for tr, te in GroupKFold(n_splits=min(5, n_rep)).split(Xs, ys, rs):
                sel = self.select_features(Xs[tr], ys[tr], cfg)   # module 4, inside the fold
                clf = clone(self._baseline(cfg)).fit(sel.transform(Xs[tr]), ys[tr])
                yp.extend(clf.predict(sel.transform(Xs[te]))); yt.extend(ys[te])
                gg.extend([s] * len(te)); ff.extend([k] * len(te))
                k += 1
        within = self._make_report(yt, yp, groups=gg, folds=ff,
                                   split_unit="repetition (within subject)",
                                   n_groups=len(set(subj.tolist())), group_unit="subject")
        return {"within_subject": within, "new_subject": new_subject}

    # ---- REAL data: Ninapro DB1 subject zips ----
    _BASE = "https://ninapro.hevs.ch/files/DB1/Preprocessed/"

    def download(self, cache_dir, subset=range(1, 6)):
        """Download + extract Ninapro DB1 subject zips (s1.zip ...) into cache_dir.
        Needs network. ~20 MB/subject. `subset` is 1-based subject ids (1..27)."""
        import os, io, zipfile, urllib.request
        os.makedirs(cache_dir, exist_ok=True)
        for s in subset:
            marker = os.path.join(cache_dir, f"S{s}_A1_E1.mat")
            if os.path.exists(marker):
                continue
            blob = urllib.request.urlopen(f"{self._BASE}s{s}.zip", timeout=180).read()
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                z.extractall(cache_dir)
        print(f"[emg_ninapro] ready in {cache_dir}")
        return cache_dir

    def load(self, cache_dir, exercise="E1"):
        """Read Ninapro DB1 `*_{exercise}.mat` files, window the sEMG within each
        (movement, repetition) segment, label by movement. Groups by SUBJECT."""
        import os, glob, re
        from scipy.io import loadmat
        gest_set = {int(g[1:]) for g in self.meta.classes}
        recs = []
        mats = sorted(glob.glob(os.path.join(cache_dir, "**", f"*_{exercise}.mat"), recursive=True))
        for mat in mats:
            m = re.search(r"S(\d+)_", os.path.basename(mat))
            if not m:
                continue
            subj = f"S{m.group(1)}"
            d = loadmat(mat)
            emg = np.asarray(d["emg"], float)                  # [N, 10]
            lab = np.asarray(d["restimulus"]).ravel()
            rep = np.asarray(d["rerepetition"]).ravel()
            wins, labs, reps = [], [], []
            for s in range(0, len(emg) - WIN + 1, STEP):
                wl = lab[s:s + WIN]; wr = rep[s:s + WIN]
                g = int(wl[0])
                if g in gest_set and (wl == g).all() and (wr == wr[0]).all() and wr[0] > 0:
                    wins.append(emg[s:s + WIN]); labs.append(f"g{g:02d}"); reps.append(int(wr[0]))
            if wins:
                recs.append(Recording(group=subj, fs=FS,
                                      epochs={"emg": np.array(wins)},
                                      labels=np.array(labs), meta={"rep": np.array(reps)}))
        return recs


if __name__ == "__main__":
    t = EMGNinaproTrack()
    print(t.dataset_card())
    modes = t.evaluate_modes(t.smoke())
    from sklearn.metrics import f1_score
    for name, rep in modes.items():
        print("SMOKE %-15s macroF1=%.3f kappa=%.3f (%s)" % (
            name, f1_score(rep["y_true"], rep["y_pred"], average="macro"),
            rep["cohens_kappa"], rep["split_unit"]))
        print("   spread:", rep["summary"])   # the per-subject range, not just the pool
