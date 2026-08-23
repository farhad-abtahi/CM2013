"""
tracks.bci_eegmmidb — BCI motor-imagery track: EEG Motor Movement/Imagery (EEGMMIDB).

Real data: PhysioNet EEGMMIDB (ODC-BY, direct download via `mne.datasets.eegbci`).
109 subjects, 64-channel EEG at 160 Hz. We use the **left- vs right-hand motor-imagery**
runs (R04, R08, R12) and classify each imagery trial **L vs R**.

This track ships **two evaluation modes** (BCI generalisation is mode-dependent):
  * **within-subject** — calibrate and test on the same subject (trial-level CV per subject);
  * **new-subject** — leave-one-subject-out (the honest, much harder deployment claim).

DSP the student does: band-pass, **mu (8–12 Hz) / beta (13–30 Hz) event-related
desynchronisation** at the motor channels C3/C4/Cz (contralateral ERD is the signature),
C3–C4 laterality; a **CSP spatial filter** is the natural supervised extension. `smoke()`
synthesises contralateral-ERD trials so CI/offline runs green.
"""
from __future__ import annotations
import numpy as np

from adapter import (TrackAdapter, TrackMeta, Recording, default_baseline,
                     denoise, spectral_bandpower, SPECTRAL_CFG_KEYS)
import biosignals as bio

RUNS = (4, 8, 12)                 # imagine opening/closing left or right fist
ANNOT_TO_CLASS = {"T1": "L", "T2": "R"}    # in these runs: T1=left fist, T2=right fist
# sensorimotor montage: contralateral mu/beta ERD lives over central electrodes.
MOTOR = ["C3", "C1", "Cz", "C2", "C4", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4", "C5", "C6"]


def _synth_mi_trial(cls, fs, dur, rng, mu_peak=10.0, decodability=0.35, mixing=None):
    """3-channel (C3, C4, Cz) motor-imagery trial with contralateral mu ERD:
    left-hand imagery suppresses mu over the RIGHT cortex (C4), right-hand over C3.

    Three per-subject nuisances are what make this track honest rather than a
    demo of a solved problem:

    * `decodability` (0..1) — how deep the contralateral suppression is. Real
      cohorts contain subjects with essentially none of it ("BCI illiteracy",
      15–30 % of people); a synthetic cohort where everyone is decodable makes
      the naive baseline look far better than the literature.
    * `mu_peak` — the subject's own alpha/mu centre frequency (8.5–12.5 Hz). A
      FIXED 8–12 Hz band, as `extract_features` uses, partly misses the subjects
      whose peak sits at the edge. That is the argument for subject-specific
      bands, and it should be *measurable*, not just asserted.
    * `mixing` — a 3x3 volume-conduction matrix: scalp EEG at C3 is a blend of
      sources, differently per head. It is why a spatial filter learned per
      subject (CSP) beats a fixed C3/C4 pair, and why cross-subject transfer of
      fixed-electrode features is so poor.

    Background is broadband (smoothed noise + white), so single-trial band power
    is genuinely noisy — as it is in real EEG.
    """
    n = int(dur * fs); t = np.arange(n) / fs

    def chan(mu_amp):
        bg = np.convolve(rng.standard_normal(n), np.ones(9) / 9, "same") * 2.2
        bg += 0.9 * rng.standard_normal(n)
        return (mu_amp * np.sin(2 * np.pi * mu_peak * t + rng.uniform(0, 6))
                + 0.35 * np.sin(2 * np.pi * 2 * mu_peak * t + rng.uniform(0, 6))   # beta
                + bg)

    erd = 0.55 * float(np.clip(decodability, 0.0, 1.0))     # depth of the ERD
    base = float(np.exp(0.25 * rng.standard_normal()))       # trial-to-trial amplitude
    if cls == "L":                # left hand  -> ERD (low mu) at C4
        a3, a4 = base, base * (1 - erd)
    else:                         # right hand -> ERD (low mu) at C3
        a3, a4 = base * (1 - erd), base
    src = np.stack([chan(a3), chan(a4), chan(base * 0.9)])                  # [3, n]
    return src if mixing is None else np.asarray(mixing) @ src


class BCIEEGMMIDBTrack(TrackAdapter):
    meta = TrackMeta(
        name="BCI motor imagery",
        dataset="EEG Motor Movement/Imagery (EEGMMIDB)",
        dataset_version="1.0.0",
        license="Open Data Commons Attribution (ODC-BY)",
        citation="Schalk G, et al. (2004); Goldberger AL, et al. PhysioNet (2000).",
        url="https://physionet.org/content/eegmmidb/1.0.0/",
        signals=["64-channel EEG @ 160 Hz"],
        task_type="binary",
        classes=["L", "R"],
        split_unit="subject",
        default_metrics=["macro_f1", "cohens_kappa"],
        smoke_test_records=["S001", "S002", "S003"],
        expected_runtime="~3–6 min on a subject subset (Colab, CPU)",
        dsp_focus="band-pass, mu/beta ERD at C3/C4/Cz, C3–C4 laterality, CSP (extension)",
        difficulty=4,
        eval_modes=("within-subject", "new-subject"),
    )

    #: Capability declaration (see `TrackAdapter.SUPPORTED_CFG_KEYS`). Stage 3 is
    #: real here and is in fact the sharpest spectral decision in the scaffold —
    #: every feature is a band power off a ~560-sample trial, which is exactly
    #: §7.6/§7.7's "short segment". **Stage 2 is not**: the shipped `preprocess()`
    #: is an identity stub (the move that matters is a CSP filter you write and
    #: fit inside the fold), so no denoise/filter key is declared and passing one
    #: raises rather than pretending. When you implement the stage, declare its
    #: knobs — `track.declare_cfg_keys("csp_components")` or by extending this set.
    SUPPORTED_CFG_KEYS = frozenset(SPECTRAL_CFG_KEYS)

    CFG_KEY_HINTS = {
        "preprocess": "BCI's preprocess() is an identity stub on purpose — the stage-2 move with "
                      "the literature behind it is a CSP spatial filter you write yourself (fit "
                      "INSIDE the training fold). Implement it, then declare its cfg keys with "
                      "track.declare_cfg_keys(...).",
    }

    # ---- synthetic smoke (offline / CI) ----
    def smoke(self, n_subjects=6, per_class=20, fs=160, dur=4.0, seed0=3000,
              decodability=(0.08, 0.60), mixing=0.55):
        """Synthetic MI cohort that is **honestly near chance** with the naive
        fixed-C3/C4 band-power baseline (chance = 0.50; expect macro-F1 ≈ 0.60
        pooled, with one clearly decodable subject up near 0.9 and others at the
        coin flip).

        The previous version scored 1.000, which flatly contradicted this track's
        own framing ("an honest near-chance baseline — the advanced track's whole
        point") and the yardstick printed two cells later. Here each subject draws
        its own ERD depth, mu peak frequency and volume-conduction mixing, so the
        between-subject spread *is* the result — and CSP has something real to do
        that a fixed electrode pair cannot."""
        recs = []
        for s in range(n_subjects):
            rng = np.random.default_rng(seed0 + s)
            d = float(rng.uniform(*decodability))
            f0 = float(rng.uniform(8.5, 12.5))
            M = np.eye(3) + mixing * rng.uniform(0.2, 1.0, size=(3, 3)) * (1 - np.eye(3))
            M /= M.sum(axis=1, keepdims=True)
            trials, labels = [], []
            for cls in self.meta.classes:
                for _ in range(per_class):
                    trials.append(_synth_mi_trial(cls, fs, dur, rng, mu_peak=f0,
                                                  decodability=d, mixing=M))
                    labels.append(cls)
            recs.append(Recording(
                group=f"S{s + 1:03d}", fs=float(fs),
                epochs={"eeg": np.array(trials)},          # [n_trials, 3, n_samp]
                labels=np.array(labels),
                meta={"ch_names": ["C3", "C4", "Cz"],
                      "decodability": round(d, 3), "mu_peak_hz": round(f0, 2)}))
        return recs

    # ---- module 2: PREPROCESSING (a stub you fill in — but read the warning) ----
    def preprocess(self, rec: Recording, cfg=None) -> Recording:
        """Identity by default — and identity means **identity**: this method reads
        no cfg key at all, which is why `SUPPORTED_CFG_KEYS` declares none for
        stage 2 and why `cfg={"preprocess": "denoise", ...}` raises here instead of
        being quietly ignored.

        The stage-2 moves that matter on this track are a **CSP spatial filter fit
        inside the training fold**, a subject-specific mu band, and a Laplacian /
        common-average re-reference — see the notebook's stage-2 menu. They are
        code you write, not a knob you set. `adapter.denoise()` is importable and
        gives you the Ch. 8 per-noise-type menu if what you actually see in the
        traces is hum, drift or pops rather than a spatial mixing problem — call it
        from this method, then register the keys you want to drive it with
        (`track.declare_cfg_keys("preprocess", "powerline", ...)`) so the option
        becomes real rather than advertised.

        ⚠️ **FIT IT INSIDE THE FOLD.** CSP *learns* from data. Fitting it here,
        where `build_dataset()` runs it over every recording before `evaluate()`
        cuts the folds, means every held-out subject has already shaped the filter
        that built the training features — and `assert_no_subject_leak` will still
        pass, because it checks group ids and cannot see what a filter was learned
        from. Put a learned transform in the `clf=` pipeline (cloned and refit per
        fold) or in `select_features()`. See `TrackAdapter.build_dataset.__doc__`.

        ⚠️ **CAUSALITY — this track has a real-time framing, so read this.**
        A brain-computer interface is, by definition, an **on-line** system: the
        user imagines a movement and the cursor must move within a second or so.
        Nothing in this scaffold enforces that, because the evaluation here is
        offline — every trial is already complete on disk before a single band
        power is computed. It is therefore very easy to build a clean, honest,
        leakage-free pipeline that could never be deployed. The traps:

        * `adapter.bandpass_notch()` defaults to `scipy.signal.filtfilt`, which is
          **zero-phase and therefore non-causal** — its output at each sample
          depends on later samples. Excellent for retrospective ERD analysis;
          impossible on a live stream. Pass `causal=True` for the single forward
          `lfilter` pass a real BCI would run, and report both numbers. The
          group delay you then have to accept is not a nuisance, it is part of
          the **latency budget** that decides whether the interface feels usable.
        * The same applies to `denoise()`'s `"detrend"`, `"spectral"` and
          `"wavelet"` remedies: all three are whole-epoch operations and are
          *inherently* non-causal, whatever you pass for `causal`.
        * Anything estimated over the whole session — a per-subject mu peak, a
          normalisation constant, a CSP filter — must in deployment come from a
          **calibration block recorded before use**. That is a real design
          decision with a real cost in user time, and naming it is worth marks.
          (Estimating it from the *labelled test trials* is not a causality
          problem, it is a leak; the harness catches that one for you.)

        None of this is forbidden — offline analysis is what this scaffold does,
        and `filtfilt` is the right tool for it. What is forbidden is quoting an
        offline number as a real-time claim without saying so. State which regime
        each number belongs to, exactly as you state the split unit.
        """
        return rec

    # ---- module 3: log mu/beta band power (ERD) across a sensorimotor montage ----
    def extract_features(self, rec: Recording, cfg=None):
        """Module 3 — log mu/beta power at a FIXED C3/C4/Cz montage plus the
        laterality ratio. Deliberately the naive option: a *spatial filter* learned
        from the data (CSP) is the standard alternative and usually much stronger,
        but it must be fit inside the training fold — which is why it belongs in
        `preprocess()`/`select_features()` rather than here. Band edges (8-12 /
        13-30 Hz) and the 0.5-s post-cue trim are choices too; subject-specific
        bands are a documented, defensible variant.

        **Every feature here is a band power, so the Ch. 7 spectral estimator is
        a first-class decision on this track** — `cfg["spectral_method"]` ∈
        `{"periodogram", "bartlett", "welch"(default), "multitaper", "ar"}`.
        This is the track §7.6 and §7.7 were written for: a trial is **4 s minus
        a 0.5 s trim at 160 Hz ≈ 560 samples**, which is short. Welch has to chop
        that into a handful of segments and quickly "runs out of segments to
        average"; §7.16 recommends multitaper for exactly this case ("short
        record, want low-leakage smooth PSD (EEG/neuro)"), and §7.7 recommends AR
        for "short segment, resonant rhythm" — which is a literal description of
        the mu rhythm. Single-trial band power is the noisiest quantity in the
        whole scaffold, so a better estimator has more room to help here than
        anywhere else. It is also where `"ar"` is most dangerous: with too high an
        order the model sprouts §7.9's "rash of small peaks", and one of them
        landing in 8-12 Hz becomes a confident, fictitious ERD."""
        cfg = self._cfg(cfg)
        method = str(cfg.get("spectral_method", "welch") or "welch").lower()
        mkw = {"ar_order": int(cfg.get("ar_order", 16)),
               "bandwidth": float(cfg.get("mt_bandwidth", 4.0))}
        ch = rec.meta.get("ch_names", [])
        fs = rec.fs

        def bp(sig, band):
            if method in ("welch", "default"):
                return bio.bandpower(sig, fs, band)
            return spectral_bandpower(sig, fs, band, method=method, **mkw)

        def idx(name):
            key = name.upper().replace(".", "")
            for i, c in enumerate(ch):
                if c.upper().replace(".", "") == key:
                    return i
            return None
        ids = {name: idx(name) for name in MOTOR}
        trim = int(0.5 * fs)                               # skip the first 0.5 s (post-cue ERD)

        def logbp(sig, band):
            return float(np.log(bp(sig, band) + 1e-12))

        rows = []
        for trial in rec.epochs["eeg"]:                    # [n_ch, n_samp]
            seg = trial[:, trim:] if trial.shape[1] > trim else trial
            row = []
            for name in MOTOR:
                i = ids[name]
                if i is None:
                    row += [0.0, 0.0]
                else:
                    row += [logbp(seg[i], (8, 12)), logbp(seg[i], (13, 30))]   # log mu, log beta
            # contralateral laterality (the core L/R signature)
            for band in ((8, 12), (13, 30)):
                i3, i4 = ids["C3"], ids["C4"]
                p3 = bp(seg[i3], band) if i3 is not None else 1e-12
                p4 = bp(seg[i4], band) if i4 is not None else 1e-12
                row.append(float(np.log((p3 + 1e-12) / (p4 + 1e-12))))
            rows.append(row)
        return np.array(rows), np.asarray(rec.labels), rec.group

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

    # ---- both evaluation modes (report both — BCI generalisation is mode-dependent) ----
    def evaluate_modes(self, recs, cfg=None):
        """Both required modes, each carrying its PER-SUBJECT spread (§16.3) —
        on this track the between-subject variability *is* the story, and a single
        pooled number is exactly what hides it."""
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold
        cfg = self._cfg(cfg)
        X, y, g = self.build_dataset(recs, cfg)
        new_subject = self.evaluate(X, y, g, cfg=cfg)      # LOSO/GroupKFold on subject
        yt, yp, gg, ff = [], [], [], []
        fold_id = 0
        for s in sorted(set(g.tolist())):                  # per-subject trial-level CV
            m = g == s
            Xs, ys = X[m], y[m]
            counts = np.unique(ys, return_counts=True)[1]
            if len(counts) < 2:
                continue
            k = int(min(5, counts.min()))
            if k < 2:
                continue
            for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(Xs, ys):
                sel = self.select_features(Xs[tr], ys[tr], cfg)   # module 4, inside the fold
                clf = clone(self._baseline(cfg)).fit(sel.transform(Xs[tr]), ys[tr])
                yp.extend(clf.predict(sel.transform(Xs[te]))); yt.extend(ys[te])
                gg.extend([s] * len(te)); ff.extend([fold_id] * len(te))
                fold_id += 1
        within_subject = self._make_report(yt, yp, groups=gg, folds=ff,
                                           split_unit="trial (within subject)",
                                           n_groups=len(set(g.tolist())), group_unit="subject")
        return {"within_subject": within_subject, "new_subject": new_subject}

    # ---- REAL data: EEGMMIDB via mne.datasets.eegbci ----
    def download(self, cache_dir, subset=range(1, 11)):
        """Fetch EEGMMIDB imagery runs (R04/R08/R12) for `subset` subjects into
        cache_dir. Needs network + `mne`. Subjects are 1-based (1..109)."""
        import os, glob
        os.makedirs(cache_dir, exist_ok=True)
        subs = list(subset)

        def _present():
            return {os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(os.path.join(cache_dir, "**", "S*R*.edf"), recursive=True)}

        def _cached(s, have):
            return all(f"S{int(s):03d}R{r:02d}" in have for r in RUNS)

        have = _present()
        missing = [s for s in subs if not _cached(s, have)]
        if not missing:
            print(f"[bci_eegmmidb] all requested subjects cached in {cache_dir}; skipping download")
            return cache_dir
        # Fetch only the missing subjects, and tolerate per-subject network failures:
        # download what we can and let load() use whatever is present (graceful degradation).
        import mne
        failed = []
        for s in missing:
            try:
                mne.datasets.eegbci.load_data(int(s), list(RUNS), path=cache_dir,
                                              update_path=False, verbose="ERROR")
            except Exception as exc:                     # noqa: BLE001
                failed.append(int(s))
                print(f"[bci_eegmmidb] could not fetch subject {s} ({type(exc).__name__}); "
                      f"continuing with cached data.")
        if failed:
            print(f"[bci_eegmmidb] proceeding without subjects {failed}; "
                  f"{len(_present())} EDFs available in cache.")
        return cache_dir

    def load(self, cache_dir, tmax=4.0):
        """Read the imagery-run EDFs into one Recording per subject (trials = L/R
        imagery epochs). Groups by SUBJECT (filename `S0xxR0y.edf` -> `S0xx`)."""
        import os, glob, re
        import mne
        edfs = glob.glob(os.path.join(cache_dir, "**", "S*R*.edf"), recursive=True)
        want = {f"R{r:02d}" for r in RUNS}
        by_subj = {}
        for f in sorted(edfs):
            m = re.search(r"(S\d{3})(R\d{2})", os.path.basename(f))
            if not m or m.group(2) not in want:
                continue
            by_subj.setdefault(m.group(1), []).append(f)

        recs = []
        for subj, files in sorted(by_subj.items()):
            trials, labels, ch_names = [], [], None
            for f in sorted(files):
                raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
                fs = raw.info["sfreq"]
                step = int(tmax * fs)
                data = raw.get_data()                       # [n_ch, n_samp]
                ch_names = raw.ch_names
                events, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
                id_to_desc = {v: k for k, v in ev_id.items()}
                for onset, _, code in events:
                    cls = ANNOT_TO_CLASS.get(id_to_desc.get(code, ""))
                    if cls is None:
                        continue
                    seg = data[:, onset:onset + step]
                    if seg.shape[1] < step:
                        continue
                    trials.append(seg); labels.append(cls)
            if trials:
                recs.append(Recording(group=subj, fs=float(fs),
                                      epochs={"eeg": np.array(trials)},
                                      labels=np.array(labels), meta={"ch_names": ch_names}))
        return recs


if __name__ == "__main__":
    t = BCIEEGMMIDBTrack()
    print(t.dataset_card())
    modes = t.evaluate_modes(t.smoke())
    for name, rep in modes.items():
        from sklearn.metrics import f1_score
        print("SMOKE %-15s macroF1=%.3f kappa=%.3f (%s)" % (
            name, f1_score(rep["y_true"], rep["y_pred"], average="macro"),
            rep["cohens_kappa"], rep["split_unit"]))
        print("   spread:", rep["summary"])   # the per-subject range, not just the pool
