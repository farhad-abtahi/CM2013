"""
tracks.ecg_cinc2017 — second reference track: single-lead ECG rhythm (AF).

Real data: PhysioNet/CinC Challenge 2017 — ~8,528 short single-lead ECGs (AliveCor,
300 Hz), labelled Normal (N) / AF (A) / Other rhythm (O) / Noisy (~). Open, direct
download (no agreement). `load()` uses `wfdb` (Colab); `smoke()` synthesises the four
rhythm classes so CI/offline runs green.

DSP the student does: band-pass + 50/60 Hz notch, Pan–Tompkins QRS detection, R–R
irregularity / HRV, and signal-quality features (the AF signature is an *irregularly
irregular* rhythm; Noisy is low signal quality).
"""
from __future__ import annotations
import numpy as np

from adapter import (TrackAdapter, TrackMeta, Recording, default_baseline,
                     bandpass_notch, wavelet_denoise, denoise, spectral_bandpower,
                     DENOISE_CFG_KEYS, SPECTRAL_CFG_KEYS)
import biosignals as bio


def _gauss(t, a, mu, s):
    return a * np.exp(-0.5 * ((t - mu) / s) ** 2)


def _beats(rr_list, fs, dur, rng, p_amp=0.10, fib=0.0):
    n = int(dur * fs); t = np.arange(n) / fs; x = np.zeros(n)
    tc = 0.4
    for rr in rr_list:
        if tc > dur - 0.3:
            break
        x += _gauss(t, 1.0, tc, 0.010) - _gauss(t, 0.22, tc + 0.035, 0.012)      # QRS
        if p_amp > 0:
            x += _gauss(t, p_amp, tc - 0.18, 0.025)                              # P
        x += _gauss(t, 0.28, tc + 0.16, 0.040)                                    # T
        tc += rr
    if fib:                                                                       # fibrillatory baseline (AF)
        x += fib * np.sin(2 * np.pi * rng.uniform(5, 8) * t) * rng.standard_normal(n) * 0.3
    x += 0.05 * np.sin(2 * np.pi * 0.3 * t)                                        # mild baseline wander
    return x


def _synth_record(kind, fs, dur, rng, typicality=1.0):
    """One synthetic record of class `kind`.

    `typicality` in [0, 1] is how *textbook* this example is: 1.0 is the clean
    caricature (unmistakable AF, unmistakable noise), 0.0 is the borderline case a
    real cardiologist argues about. Drawing it uniformly per record is what turns
    a trivially separable smoke set into an honest one — classes now OVERLAP, and
    the N↔A confusion that dominates the real challenge shows up here too.
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    a = float(np.clip(typicality, 0.0, 1.0))
    rr0 = rng.uniform(0.60, 1.05)                    # this subject's resting R-R
    nbeat = int(dur / rr0) + 20
    # every real recording carries some noise, hum and wander — not only the '~' class,
    # otherwise "has 50 Hz hum" is a giveaway label rather than a signal-quality feature.
    base_noise = rng.uniform(0.03, 0.22)
    hum = rng.uniform(0.0, 0.25)
    wander = rng.uniform(0.02, 0.25)

    if kind == "N":                              # normal sinus rhythm + mild HRV
        hrv = rng.uniform(0.015, 0.080)
        rr = rr0 + 0.05 * np.sin(2 * np.pi * 0.25 * np.arange(nbeat)) + hrv * rng.standard_normal(nbeat)
        if rng.random() < 0.4:                   # ectopic beats on an otherwise normal strip
            j = rng.integers(0, nbeat - 1, size=int(rng.integers(1, 5)))
            rr[j] *= 0.55
            rr[np.clip(j + 1, 0, nbeat - 1)] *= 1.4
        x = _beats(rr, fs, dur, rng, p_amp=0.04 + 0.10 * a)
    elif kind == "A":                            # AF: irregularly irregular, no P waves
        scale = 0.05 + 0.17 * a                  # low typicality -> AF that looks almost regular
        rr = np.clip(rr0 * 0.75 + rng.exponential(scale, nbeat), 0.28, 1.7)
        x = _beats(rr, fs, dur, rng, p_amp=0.05 * (1 - a), fib=0.03 + 0.10 * a)
    elif kind == "O":                            # Other: bigeminy, or regular brady/tachycardia
        if rng.random() < 0.5:
            depth = 0.06 + 0.42 * a              # shallow bigeminy is easily read as normal HRV
            rr = (np.tile([rr0 * (1 - depth), rr0 * (1 + depth)], nbeat // 2 + 1)[:nbeat]
                  + 0.03 * rng.standard_normal(nbeat))
        else:
            rr = (np.full(nbeat, rr0 * (0.62 if rng.random() < 0.5 else 1.45))
                  + 0.03 * rng.standard_normal(nbeat))
        x = _beats(rr, fs, dur, rng, p_amp=0.10)
    else:                                        # ~ Noisy: normal rhythm + corruption (low SQI)
        rr = rr0 + 0.04 * rng.standard_normal(nbeat)
        x = _beats(rr, fs, dur, rng, p_amp=0.10)
        nz = 0.10 + 0.55 * a                     # a barely-noisy '~' is a genuinely hard call
        base_noise = max(base_noise, nz)
        if rng.random() < 0.6:                   # some '~' are motion artifact only, no mains hum
            hum = max(hum, 0.3 * nz)
        wander = max(wander, 0.6 * nz)

    return (x + base_noise * rng.standard_normal(n)
            + hum * np.sin(2 * np.pi * 50 * t)
            + wander * np.sin(2 * np.pi * 0.4 * t))


def _rpeaks(ecg, fs):
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], "band")
    f = filtfilt(b, a, ecg)
    d = np.diff(f, prepend=f[0]) ** 2
    w = max(1, int(0.15 * fs))
    integ = np.convolve(d, np.ones(w) / w, "same")
    pk, _ = find_peaks(integ, height=0.3 * np.max(integ) + 1e-9, distance=int(0.3 * fs))
    return pk


class ECGCinC2017Track(TrackAdapter):
    meta = TrackMeta(
        name="ECG rhythm (AF)",
        dataset="PhysioNet/CinC Challenge 2017 (single-lead ECG)",
        dataset_version="1.0.0",
        license="Open (PhysioNet Challenge)",
        citation="Clifford GD, et al. PhysioNet/CinC Challenge 2017.",
        url="https://physionet.org/content/challenge-2017/1.0.0/",
        signals=["single-lead ECG @ 300 Hz"],
        task_type="multiclass",
        classes=["N", "A", "O", "~"],           # Normal, AF, Other, Noisy
        split_unit="record",
        default_metrics=["macro_f1", "cohens_kappa"],
        smoke_test_records=["A00001", "A00004", "A00006"],
        expected_runtime="~2–5 min on a subset (Colab, CPU)",
        dsp_focus="band-pass + notch, Pan–Tompkins QRS, R–R irregularity/HRV, signal quality",
        difficulty=3,
        eval_modes=("new-record",),
        submission_granularity="record",         # one label per record for hold-out evaluation
    )

    #: Capability declaration (see `TrackAdapter.SUPPORTED_CFG_KEYS`). Like
    #: Sleep-EDF, this track's `preprocess()` forwards the whole Chapter 8
    #: denoise menu, and its HRV/quality features include band powers integrated
    #: from a Chapter 7 estimator — so both stage-2 and stage-3 menus are real.
    SUPPORTED_CFG_KEYS = DENOISE_CFG_KEYS | SPECTRAL_CFG_KEYS | {"ecg_band"}

    #: Class counts for the synthetic smoke set, in the same lopsided shape as the
    #: real challenge (Normal dominates; AF and Noisy are rare). The rubric grades
    #: how you address that imbalance — it cannot be graded on a balanced fake.
    SMOKE_COUNTS = {"N": 34, "A": 16, "O": 24, "~": 12}

    def smoke(self, per_class=None, fs=300, dur=20.0):
        """Synthetic four-class smoke set with a per-record **typicality** draw, so
        the classes overlap the way they do on the real challenge instead of being
        four separable caricatures — and with the real challenge's **class
        imbalance** (see `SMOKE_COUNTS`). GroupKFold-by-record lands around
        macro-F1 ≈ 0.7 (below the 0.83 challenge-winner yardstick, as it should
        be), with N and A the pair that trades places — exactly the confusion the
        notebook asks you to predict before running.

        Pass `per_class=<int>` for the old balanced cohort if you specifically
        want to isolate a change from the imbalance effect."""
        recs, k = [], 0
        for cls in self.meta.classes:
            n_cls = per_class if per_class is not None else self.SMOKE_COUNTS.get(cls, 20)
            for _ in range(n_cls):
                r = np.random.default_rng(1000 + k)
                x = _synth_record(cls, fs, dur, r, typicality=r.uniform(0.0, 1.0))
                recs.append(Recording(group=f"R{k:04d}", fs=fs,
                                      epochs={"ecg": x[None, :]}, labels=np.array([cls])))
                k += 1
        return recs

    # ---- module 2: PREPROCESSING (opt-in; the recipe is your decision) ----
    def preprocess(self, rec: Recording, cfg=None) -> Recording:
        """Identity by default — the shipped baseline lets `_rpeaks()` own its own
        band-pass and lets every other feature see the raw strip. That is a
        starting point, not a recommendation, and hoisting the cleaning up here is
        one of the decisions this track exists to make you argue for.

        **Two supplied recipes, answering different noise signatures.** Pick with
        `cfg["preprocess"]`:

            ECGCinC2017Track(cfg={"preprocess": "none"})                             # default
            ECGCinC2017Track(cfg={"preprocess": "bandpass", "ecg_band": (0.5, 40.0), "notch": 50.0})
            ECGCinC2017Track(cfg={"preprocess": "wavelet",  "wavelet": "db4"})
            ECGCinC2017Track(cfg={"preprocess": "denoise",  "impulsive": "median",
                                  "baseline": "highpass", "hp_fc": 0.5,
                                  "powerline": "notch"})

        | `preprocess` | attacks | on THIS track that means | costs you |
        |---|---|---|---|
        | `"bandpass"` | **stationary, narrow-band** interference | 50 Hz mains hum and slow baseline wander are exactly that: fixed in frequency, present throughout. A 0.5-40 Hz band plus a notch removes both for almost nothing | the **QRS complex is a broadband, ~80 ms transient** — band-limiting rounds its edges, widens its apparent duration and shifts the peak the R-R interval is measured from. And your powerline-power feature was *detecting* the hum: notch it away and you have deleted the evidence the `~` class rests on |
        | `"wavelet"` | **non-stationary transients** | motion artifact, electrode contact loss, muscle bursts — the corruption that *defines* the Noisy (`~`) class. These are broadband and brief, so no band contains them; DWT thresholding removes them where they are, in time and scale | over-thresholding erodes low-amplitude morphology (the P wave whose *absence* is the AF signature) and, again, quietly cleans away the very corruption the `~` class is labelled for. Cleaning the noisy class into looking normal is a way to lose the problem, not solve it |
        | `"none"` *(default)* | nothing | detection owns its filter; features see the raw signal | the filter is invisible in the report and cannot be swapped as a stage |

        **`"denoise"` is the per-noise-type menu** (Chapter 8's own structure: one
        key per corruption, not one key per technique), and ECG is the track it
        fits best, because a real single-lead strip carries *all four* of Ch. 8's
        categories at once and they want different remedies:

        | what you see (§8.3's three lenses) | Ch. 8 category | key |
        |---|---|---|
        | brief huge outliers, broadband transient | electrode pop / impulsive (§8.7) | `impulsive="median"` — and it runs FIRST, because §9.7 says a linear filter cannot delete a spike, it only smears it |
        | slow drift under the trace | baseline wander (§8.8) | `baseline="highpass"` at 0.5 Hz for rhythm work — but §8.8 warns a diagnostic ECG needs content down to 0.05 Hz, and too aggressive a corner "can manufacture artificial ST shifts that mimic ischaemia" |
        | fine steady buzz, line at 50/60 Hz | powerline (§8.6) | `powerline="notch"` — "the friendliest artifact" |
        | structureless fuzz across the band | white/broadband (§8.4) | `broadband="wavelet"` to keep the QRS sharp, or `"savgol"` to keep peak height with a linear filter |

        The sharpest lesson on this track is that **denoising can destroy the
        label**. Two of the eleven shipped features (the QRS/high-frequency ratio
        and the 49-51 Hz powerline power) exist to *measure* corruption. Any
        preprocessing that removes corruption before those features are computed
        makes the `~` class harder, not easier. Measure it both ways and report
        which you chose.

        Note: `bandpass_notch` is zero-phase (`filtfilt`) and therefore
        **non-causal** — legitimate here, because CinC-2017 is retrospective
        analysis of complete 30 s strips. A wearable rhythm monitor deciding in
        real time could not use it; see `adapter.bandpass_notch.__doc__` and pass
        `cfg={"causal": True}` to measure what a deployable filter costs.
        """
        cfg = self._cfg(cfg)
        how = str(cfg.get("preprocess", "none") or "none").lower()
        if how in ("denoise", "by_noise", "noise"):
            return denoise(rec, impulsive=cfg.get("impulsive"), baseline=cfg.get("baseline"),
                           powerline=cfg.get("powerline"), broadband=cfg.get("broadband"),
                           note=self.note,
                           **{k: cfg[k] for k in ("hp_fc", "f0", "q", "win", "wavelet", "causal",
                                                  "poly_order", "savgol_order", "harmonics", "mu",
                                                  "order", "threshold_mode", "level")
                              if cfg.get(k) is not None})
        if how in ("wavelet", "dwt", "wavelet_denoise"):
            return wavelet_denoise(rec, wavelet=cfg.get("wavelet", "db4"),
                                   level=cfg.get("wavelet_level"),
                                   threshold_mode=cfg.get("wavelet_mode", "soft"))
        band, notch = cfg.get("ecg_band"), cfg.get("notch")
        if how in ("bandpass", "bandpass_notch", "filter") and band is None and notch is None:
            band = (0.5, 40.0)
            self.note("preprocess='bandpass' with no ecg_band given — using 0.5-40 Hz. Set "
                      "cfg['ecg_band'] yourself and justify the edges; note that this band also "
                      "removes the 50 Hz content one of your features measures.")
        if band is None and notch is None:
            return rec
        return bandpass_notch(rec, band=band, notch=notch,
                              causal=bool(cfg.get("causal", False)))

    def extract_features(self, rec: Recording, cfg=None):
        """Module 3 — one feature row per RECORD (R-R/HRV descriptors + a
        signal-quality index). Where the band-pass that feeds R-peak detection
        lives is a design choice: inside `_rpeaks` (as here, so detection owns its
        own filter) or hoisted into `preprocess()` so every feature sees the same
        cleaned signal — `cfg={"preprocess": "bandpass"}` now does exactly that.
        Both are defensible; the second makes the filter a reportable, swappable
        stage.

        **Three of the eleven features are band powers, so the Ch. 7 spectral
        estimator is a knob here too** — `cfg["spectral_method"]` ∈
        `{"periodogram", "bartlett", "welch"(default), "multitaper", "ar"}`; see
        `adapter.make_spectral_estimator`. The trade-off has a specific shape on
        this track. The 49-51 Hz powerline feature is the narrowest band in the
        whole scaffold — 2 Hz wide — so it is the one most punished by an
        estimator whose resolution has coarsened, and the one most helped by
        `"multitaper"` on a short strip. Conversely `"ar"` will happily place a
        pole near 50 Hz and report a confident, smooth, possibly fictitious hum
        peak, which is §7.9's over-fitting warning arriving as a feature value.
        A 20-30 s strip at 300 Hz is long enough for Welch to be comfortable, so
        expect small movements — say so honestly if that is what you measure."""
        from scipy.stats import kurtosis
        cfg = self._cfg(cfg)
        method = str(cfg.get("spectral_method", "welch") or "welch").lower()
        mkw = {"ar_order": int(cfg.get("ar_order", 16)),
               "bandwidth": float(cfg.get("mt_bandwidth", 4.0))}

        def _bp(sig, f_s, band):
            if method in ("welch", "default"):
                return bio.bandpower(sig, f_s, band)
            return spectral_bandpower(sig, f_s, band, method=method, **mkw)
        ecg = rec.epochs["ecg"][0]; fs = rec.fs
        pk = _rpeaks(ecg, fs); rr = np.diff(pk) / fs
        if len(rr) >= 3:
            hr = 60.0 / np.mean(rr); sdnn = np.std(rr) * 1000
            rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000
            pnn50 = np.mean(np.abs(np.diff(rr)) > 0.05) * 100
            cv = np.std(rr) / (np.mean(rr) + 1e-9)
            irr = np.mean((rr < 0.6) | (rr > 1.2))
        else:
            hr = sdnn = rmssd = pnn50 = cv = irr = 0.0
        qrs_bp = _bp(ecg, fs, (5, 15))
        hf = _bp(ecg, fs, (40, min(fs / 2 - 1, 90)))
        feats = [hr, sdnn, rmssd, pnn50, cv, irr, float(len(rr)),
                 qrs_bp / (hf + 1e-9),                      # signal-quality index
                 _bp(ecg, fs, (49, 51)),                    # powerline power
                 float(np.std(ecg)), float(kurtosis(ecg))]
        return np.array([feats]), np.array([rec.labels[0]]), rec.group

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

    _PN_BASE = "https://physionet.org/files/challenge-2017/1.0.0/training/"

    def download(self, cache_dir, subset=400, seed=0, workers=8):
        """Fetch a class-stratified subset of the CinC-2017 training set into
        cache_dir. Needs network; `wfdb` not required for the download itself.

        The full training set is ~600 MB (8,528 records); each record is only a
        ~18 KB single-lead ECG, so we pull just `subset` records — stratified
        across N/A/O/~ with a floor per class so the rare classes (AF, Noisy)
        stay learnable. Files land flat in cache_dir as `A0xxxx.hea`/`.mat`
        alongside `REFERENCE-v3.csv`.
        """
        import os, urllib.request, urllib.error, concurrent.futures as cf
        os.makedirs(cache_dir, exist_ok=True)
        ref_dst = os.path.join(cache_dir, "REFERENCE-v3.csv")
        if not os.path.exists(ref_dst):
            urllib.request.urlretrieve(self._PN_BASE + "REFERENCE-v3.csv", ref_dst)
        # group record ids (with their A0x/ subfolder prefix) by class label
        by_cls = {c: [] for c in self.meta.classes}
        for line in open(ref_dst):
            line = line.strip()
            if "," not in line:
                continue
            rid, lab = line.split(",", 1)
            by_cls.setdefault(lab, []).append(rid)
        # stratified pick: proportional share of `subset`, but >= floor per class
        rng = np.random.default_rng(seed)
        floor = 40
        total_ref = sum(len(v) for v in by_cls.values())
        picks = []
        for c in self.meta.classes:
            ids = by_cls.get(c, [])
            want = max(floor, round(subset * len(ids) / max(total_ref, 1)))
            want = min(want, len(ids))
            idx = rng.choice(len(ids), size=want, replace=False)
            picks.extend(ids[i] for i in idx)

        def _grab(rid_full):
            base = os.path.basename(rid_full)               # 'A00/A00001' -> 'A00001'
            for ext in (".hea", ".mat"):
                dst = os.path.join(cache_dir, base + ext)
                if os.path.exists(dst) and os.path.getsize(dst) > 0:
                    continue
                for attempt in range(3):
                    try:
                        urllib.request.urlretrieve(self._PN_BASE + rid_full + ext, dst)
                        break
                    except (urllib.error.URLError, OSError):
                        if attempt == 2:
                            raise
            return base

        got = 0
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(_grab, picks):
                got += 1
        print(f"[ecg_cinc2017] downloaded {got} records into {cache_dir}")
        return cache_dir

    def load(self, cache_dir):
        """Read CinC-2017 WFDB records + REFERENCE labels; one Recording per record.
        Robust to both the flat layout (records copied into cache_dir) and the
        nested `A0x/` layout from unzipping training2017.zip; REFERENCE ids may
        carry an `A0x/` prefix, so we match on the basename."""
        import os, csv, glob
        import wfdb
        ref = {}
        for name in ("REFERENCE-v3.csv", "REFERENCE.csv"):
            p = os.path.join(cache_dir, name)
            if os.path.exists(p):
                for row in csv.reader(open(p)):
                    if len(row) >= 2:
                        ref[os.path.basename(row[0])] = row[1]
                break
        recs = []
        for hea in sorted(glob.glob(os.path.join(cache_dir, "**", "*.hea"), recursive=True)):
            rid = os.path.splitext(os.path.basename(hea))[0]
            if rid not in ref:
                continue
            rec = wfdb.rdrecord(os.path.splitext(hea)[0])
            x = rec.p_signal[:, 0].astype(float)
            recs.append(Recording(group=rid, fs=float(rec.fs),
                                  epochs={"ecg": x[None, :]}, labels=np.array([ref[rid]])))
        return recs


if __name__ == "__main__":
    t = ECGCinC2017Track()
    print(t.dataset_card())
    rep = t.run_smoke()
    from sklearn.metrics import f1_score
    print("SMOKE (synthetic) CV: macroF1=%.3f kappa=%.3f n_records=%d"
          % (f1_score(rep["y_true"], rep["y_pred"], average="macro"),
             rep["cohens_kappa"], rep["n_groups"]))
    print("spread:", rep["summary"])          # never the pooled number alone (§16.3)
    print("classes:", rep["labels"], "\nconfusion (rows=true):\n", rep["confusion"])
    # hold-out demo: fit on train records, write a predictions.csv on held-out records
    # smoke() emits records grouped by class, so shuffle before an index-based split
    recs = list(np.random.default_rng(0).permutation(t.smoke()))
    split = int(0.8 * len(recs))
    model = t.train_baseline(recs[:split])
    import tempfile, os
    out = os.path.join(tempfile.gettempdir(), "ecg_predictions.csv")
    t.write_submission(recs[split:], out, model)
    print("wrote submission:", out, "->", open(out).readline().strip(), "...")
