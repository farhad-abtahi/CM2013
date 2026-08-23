"""
tracks.sleep_edf — the REFERENCE capstone track (Sleep-EDF Expanded, PhysioNet).

Real data: open, ODC-BY, direct download (no agreement). The `load()` path uses
`mne` and runs in Colab; the `smoke()` path uses the synthetic sleep cohort so
CI / offline always runs green. Every other track conforms to this pattern.
"""
from __future__ import annotations
import numpy as np

from adapter import (TrackAdapter, TrackMeta, Recording, default_baseline,
                     bandpass_notch, wavelet_denoise, denoise, spectral_bandpower,
                     DENOISE_CFG_KEYS, SPECTRAL_CFG_KEYS)
from bsp import sleep_pipeline as sp


# R&K (Sleep-EDF) -> AASM 5-class: merge S3+S4 -> N3; drop MOVEMENT / UNKNOWN.
RK_TO_AASM = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
DROP = {"Sleep stage ?", "Movement time"}


class SleepEDFTrack(TrackAdapter):
    meta = TrackMeta(
        name="Sleep staging",
        dataset="Sleep-EDF Expanded",
        dataset_version="1.0.0",
        license="Open Data Commons Attribution (ODC-BY)",
        citation="Kemp B, et al. (2000); Goldberger AL, et al. PhysioNet (2000).",
        url="https://physionet.org/content/sleep-edfx/1.0.0/",
        signals=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"],
        task_type="multiclass",
        classes=["W", "N1", "N2", "N3", "REM"],
        split_unit="subject",       # NOT night: Sleep-Cassette has 2 nights/subject
        default_metrics=["cohens_kappa", "macro_f1"],
        smoke_test_records=["SC4001", "SC4011", "SC4021"],
        expected_runtime="~3–6 min on a Sleep-Cassette subset (Colab, CPU)",
        dsp_focus="band power, spindle STFT/wavelet, EOG/EMG artifact handling",
        difficulty=3,
        eval_modes=("new-subject",),
    )

    #: Capability declaration (see `TrackAdapter.SUPPORTED_CFG_KEYS`). This is the
    #: **fullest** track: `preprocess()` forwards the whole Chapter 8 denoise menu
    #: and both filter recipes, and the EEG band powers are integrated from a
    #: Chapter 7 estimator — so every stage-2 and stage-3 menu is real here.
    SUPPORTED_CFG_KEYS = (DENOISE_CFG_KEYS | SPECTRAL_CFG_KEYS
                          | {"eeg_band", "filter_channels"})

    # ---- synthetic smoke (offline / CI) ----
    def smoke(self, n_subjects=5, n_epochs=80, seed=0, difficulty="medium"):
        """Synthetic cohort at the **"medium"** rung of `bsp.sleep_pipeline`'s
        difficulty curriculum — deliberately NOT "easy".

        "easy" is strongly separable and every pipeline scores κ ≈ 1.00 on it,
        which makes every design decision on this track look free: change the
        band-pass, change the selector, change the classifier, the number does not
        move. That is the opposite of the lesson. "medium" adds subject gain
        shift, alpha-peak variability, pink/brown noise, baseline wander,
        artifacts and scorer label noise, so LOSO lands around κ ≈ 0.6 with a real
        per-subject spread — and choices start to *cost* or *earn* something you
        can measure. Pass `difficulty="easy"` if you want the old separable cohort
        for a plotting sanity-check, or `"hard"` for a dress rehearsal for real
        Sleep-EDF (κ collapses to ~0.15 there; that is also honest)."""
        recs = []
        for night in sp.cohort(n_subjects=n_subjects, n_epochs=n_epochs, seed=seed,
                               difficulty=difficulty):
            recs.append(Recording(
                group=night["subject"], fs=night["fs"],
                epochs={"eeg": night["eeg"], "eog": night["eog"], "emg": night["emg"]},
                labels=night["stages"]))
        return recs

    # ---- module 2: PREPROCESSING (opt-in; the recipe is your decision) ----
    def preprocess(self, rec: Recording, cfg=None) -> Recording:
        """Identity by default — the supplied baseline scores the epochs as loaded,
        which is a floor to beat, not a recommendation.

        **Three supplied recipes, answering different noise signatures.** Pick one
        with `cfg["preprocess"]`:

            SleepEDFTrack(cfg={"preprocess": "none"})                  # default
            SleepEDFTrack(cfg={"preprocess": "bandpass", "eeg_band": (0.5, 40.0), "notch": 50.0})
            SleepEDFTrack(cfg={"preprocess": "wavelet",  "wavelet": "db4"})
            SleepEDFTrack(cfg={"preprocess": "denoise",  "impulsive": "median",
                               "baseline": "highpass", "powerline": "notch"})

        (Setting `eeg_band` / `notch` alone still turns the band-pass on, as
        before, so older configs keep working.)

        `"denoise"` is the **per-noise-type menu** — Chapter 8's own structure,
        one key per corruption rather than one key per technique, applied in the
        order §8/§9 require (impulses → wander → mains → broadband). It is the
        right shape for this track because the synthetic cohort deliberately
        injects four *different* corruptions (`bsp.sleep_pipeline` adds electrode
        pops, flat-line dropout, clipping, baseline wander and 50 Hz hum), and a
        single band-pass is the correct answer to only one of them. Run §8.3's
        three-lens check on a few epochs, name what you actually see, then pick a
        remedy per problem — see `adapter.denoise.__doc__` for the full table.

        | `preprocess` | attacks | on THIS track that means | costs you |
        |---|---|---|---|
        | `"bandpass"` | **stationary, narrow-band** interference | 50 Hz mains hum sits in one place all night — a notch deletes it and nothing else; a 0.5-40 Hz band removes DC drift and EMG bleed | a fixed band is applied to every instant equally, so it **smears the sharp transients**: K-complexes and spindle onsets are brief broadband events, and band-limiting rounds their edges. Tighten to 0.5-30 Hz and you also delete the beta some Wake/N1 features lean on |
        | `"wavelet"` | **non-stationary transients** | movement arousals, electrode pops and slow baseline steps live in no single band, so no band-pass can remove them without taking signal too. DWT thresholding removes them where they *are* — localised in time and scale — and leaves the sharp K-complex edge intact | it shrinks everything, so too high a threshold erodes **weak spindles**, exactly the low-amplitude structure N2 depends on. Also a per-epoch threshold means neighbouring epochs are cleaned slightly differently |
        | `"none"` *(default)* | nothing | the honest floor to beat | the artifacts are still there, in every feature you compute |

        Both are supplied so the plumbing is not the exercise; the numbers (band
        edges, wavelet, threshold mode) are yours to choose and justify. Other
        defensible moves this stub does *not* make for you: re-referencing,
        EOG-informed artifact rejection vs. interpolation vs. keep-and-flag,
        cropping to lights-off ± 30 min (which changes the Wake prior, and
        therefore every metric). Record which you chose and why.

        Note: `bandpass_notch` is zero-phase (`filtfilt`) and therefore
        **non-causal** — correct here, because sleep scoring is retrospective and
        the whole night is on disk before anything is computed. See
        `adapter.bandpass_notch.__doc__` if you ever port this to a live monitor.
        """
        cfg = self._cfg(cfg)
        how = str(cfg.get("preprocess", "none") or "none").lower()
        chans = cfg.get("filter_channels", ("eeg",))
        if how in ("denoise", "by_noise", "noise"):
            return denoise(rec, impulsive=cfg.get("impulsive"), baseline=cfg.get("baseline"),
                           powerline=cfg.get("powerline"), broadband=cfg.get("broadband"),
                           channels=chans, note=self.note,
                           **{k: cfg[k] for k in ("hp_fc", "f0", "q", "win", "wavelet", "causal",
                                                  "poly_order", "savgol_order", "harmonics", "mu",
                                                  "order", "threshold_mode", "level")
                              if cfg.get(k) is not None})
        if how in ("wavelet", "dwt", "wavelet_denoise"):
            return wavelet_denoise(rec, wavelet=cfg.get("wavelet", "db4"),
                                   level=cfg.get("wavelet_level"),
                                   threshold_mode=cfg.get("wavelet_mode", "soft"),
                                   channels=chans)
        band, notch = cfg.get("eeg_band"), cfg.get("notch")
        if how in ("bandpass", "bandpass_notch", "filter") and band is None and notch is None:
            band = (0.5, 40.0)               # the §9.6 case-study spec, stated out loud
            self.note("preprocess='bandpass' with no eeg_band given — using the §9.6 default "
                      "(0.5-40 Hz). Set cfg['eeg_band'] yourself and justify the edges.")
        if band is None and notch is None:
            return rec
        return bandpass_notch(rec, band=band, notch=notch, channels=chans,
                              causal=bool(cfg.get("causal", False)))

    # ---- module 3: FEATURE EXTRACTION (the DSP the student does) ----
    def extract_features(self, rec: Recording, cfg=None):
        """Cleaned epochs -> one feature vector per 30-s epoch (band powers,
        Hjorth, spectral entropy, EOG movement, EMG RMS — `bsp.sleep_pipeline`).
        This is the supplied starting set; spindle (sigma) detection via STFT or
        wavelets and per-subject normalisation are the obvious next moves, and
        choosing among them is the exercise.

        **The five EEG band powers are integrated from a PSD, and WHICH ESTIMATOR
        produced that PSD is a design decision** (Ch. 7), not an implementation
        detail. `bsp.epoch_features` uses `bsp.bandpower`, i.e. Welch — the
        book's "everyday workhorse". Override it:

            SleepEDFTrack(cfg={"spectral_method": "multitaper", "mt_bandwidth": 4.0})
            SleepEDFTrack(cfg={"spectral_method": "ar", "ar_order": 16})
            SleepEDFTrack(cfg={"spectral_method": "periodogram"})   # the naive baseline

        The menu and its trade-offs are on `adapter.make_spectral_estimator`.
        This track is where the decision bites hardest, for two reasons:

        * **The bands are narrow.** Sigma (the spindle band) is 11-16 Hz and
          alpha is 8-11 Hz. Every non-parametric estimator buys variance
          reduction by spending resolution, and once the practical resolution
          coarsens past a few Hz you are no longer measuring the band you named —
          alpha and sigma start reporting each other's power. §7.4's "governing
          law" is a concrete constraint here, not an abstraction.
        * **The epochs are short-ish and the classes hinge on weak rhythms.**
          A 30 s epoch at 100 Hz is 3000 samples; Welch with `nperseg=256` gets
          ~23 segments, which is comfortable. But a raw `"periodogram"` has ~100 %
          variance on every bin, so a genuine spindle and a noise excursion look
          alike — expect it to *lose* you kappa, and report that it did.
          `"multitaper"` gets Welch-like smoothing without chopping the record,
          and `"ar"` gives a smooth spectrum with a crisp peak — at the cost of
          §7.9's model-order risk (too low merges alpha and sigma into one hump;
          too high sprouts spurious peaks that your band powers will faithfully
          integrate).

        Predict which way each will move kappa *before* you run it, then check.
        """
        cfg = self._cfg(cfg)
        method = str(cfg.get("spectral_method", "welch") or "welch").lower()
        rows = [sp.epoch_features(e, o, m, rec.fs)
                for e, o, m in zip(rec.epochs["eeg"], rec.epochs["eog"], rec.epochs["emg"])]
        if method not in ("welch", "default"):
            # Recompute ONLY the band-power columns under the chosen estimator;
            # everything else (Hjorth, entropy, EOG, EMG) is unchanged, so the
            # comparison isolates the spectral-estimation decision.
            kw = {"ar_order": int(cfg.get("ar_order", 16)),
                  "bandwidth": float(cfg.get("mt_bandwidth", 4.0))}
            self.note(f"spectral_method={method!r}: the {len(sp._BANDS)} EEG band powers are "
                      f"integrated from a {method} PSD instead of Welch (Hjorth / entropy / EOG / "
                      f"EMG features are untouched, so any metric change is the estimator's doing). "
                      f"See adapter.make_spectral_estimator for what you traded away.")
            for row, e in zip(rows, rec.epochs["eeg"]):
                for name, band in sp._BANDS.items():
                    row[f"eeg_{name}"] = spectral_bandpower(e, rec.fs, band, method=method, **kw)
        X = [list(r.values()) for r in rows]
        return np.array(X), np.asarray(rec.labels), rec.group

    def feature_names(self, cfg=None):
        z = np.zeros(64)
        return list(sp.epoch_features(z, z, z, 100.0).keys())

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

    # ---- REAL data (Colab): Sleep-EDF via mne ----
    def download(self, cache_dir, subset=None):
        """Fetch a subset of Sleep-Cassette PSG+Hypnogram EDFs into cache_dir.

        Needs network + `mne`. `subset` is an iterable of subject indices
        (0-based, as mne numbers them); defaults to the first four subjects
        (0..3), both nights -> 8 nights. mne caches the EDFs in its own data
        dir; we link/copy each (PSG, Hypnogram) pair into `cache_dir` under its
        original PhysioNet name so `load()`'s glob finds them.
        """
        import os
        import mne
        subjects = list(subset) if subset is not None else [0, 1, 2, 3]
        os.makedirs(cache_dir, exist_ok=True)
        # Download straight into cache_dir (a single, reusable home — e.g. a
        # Dropbox-backed folder). mne lays the EDFs out under a
        # `physionet-sleep-data/` subfolder; load() globs recursively for them.
        mne.datasets.sleep_physionet.age.fetch_data(
            subjects=subjects, recording=[1, 2], path=cache_dir,
            on_missing="warn", verbose="ERROR")
        return cache_dir

    def load(self, cache_dir, fs_target=100.0, epoch_s=30.0):
        """Read Sleep-EDF PSG + Hypnogram pairs into Recordings.
        Groups by SUBJECT (filename 'SC4ss*'-> subject 'SC4ss'), maps R&K->AASM,
        merges S3+S4->N3, drops MOVEMENT/UNKNOWN epochs."""
        import glob, os
        import mne
        recs = []
        # recursive: finds EDFs whether they sit in cache_dir itself or in mne's
        # `physionet-sleep-data/` subfolder.
        psgs = sorted(glob.glob(os.path.join(cache_dir, "**", "*PSG.edf"), recursive=True))
        for psg in psgs:
            hyp = psg.replace("-PSG.edf", "-Hypnogram.edf")
            if not os.path.exists(hyp):
                cand = glob.glob(psg[:len(psg) - 10] + "*Hypnogram.edf")
                if not cand:
                    continue
                hyp = cand[0]
            raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
            ann = mne.read_annotations(hyp)
            raw.set_annotations(ann, emit_warning=False)
            # resample to a common rate FIRST, so event sample-indices match the data
            if fs_target and abs(raw.info["sfreq"] - fs_target) > 1e-6:
                raw.resample(fs_target, verbose="ERROR")
            fs = raw.info["sfreq"]
            step = int(epoch_s * fs)
            # require all three channels — a record missing one would crash features()
            ch = {"eeg": "EEG Fpz-Cz", "eog": "EOG horizontal", "emg": "EMG submental"}
            missing = [k for k, v in ch.items() if v not in raw.ch_names]
            if missing:
                print(f"[sleep_edf] {os.path.basename(psg)}: missing {missing}; skipping record.")
                continue
            data = {k: raw.get_data(picks=v)[0] for k, v in ch.items()}
            events, ev_id = mne.events_from_annotations(
                raw, event_id={k: i for i, k in enumerate(RK_TO_AASM)},
                chunk_duration=epoch_s, verbose="ERROR")
            id_to_stage = {i: RK_TO_AASM[k] for k, i in ev_id.items()}
            ep = {k: [] for k in data}
            labels = []
            for e in events:
                stage = id_to_stage.get(e[2])
                if stage is None:
                    continue
                onset = e[0]
                # collect ALL channel segments first; append only if every channel is full-length
                segs, ok = {}, True
                for k, sig in data.items():
                    s = sig[onset:onset + step]
                    if len(s) < step:
                        ok = False; break
                    segs[k] = s
                if ok:
                    for k in data:
                        ep[k].append(segs[k])
                    labels.append(stage)
            subj = os.path.basename(psg)[:5]         # 'SC4ss' -> subject id
            recs.append(Recording(group=subj, fs=fs,
                                  epochs={k: np.array(v) for k, v in ep.items()},
                                  labels=np.array(labels)))
        return recs


if __name__ == "__main__":
    t = SleepEDFTrack()
    print(t.dataset_card())
    rep = t.run_smoke()
    print("SMOKE (synthetic) LOSO:", {k: rep[k] for k in ("accuracy", "cohens_kappa", "macro_f1", "n_groups")})
    print("spread:", rep["summary"])          # never the pooled number alone (§16.3)
    t.report(rep)                             # module 7 — confusion matrix first (§16.8)
