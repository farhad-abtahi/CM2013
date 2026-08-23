"""
tracks.ctg_ctu_uhb — CTG (fetal) track: CTU-UHB Intrapartum Cardiotocography.

Real data: PhysioNet CTU-UHB (ODC-BY, direct download — no agreement). 552 records,
each 90 min of **fetal heart rate (FHR)** + **uterine contraction (UC/toco)** at 4 Hz,
with delivery outcome measures (umbilical-artery **pH**, base deficit, Apgar) in the
WFDB header. The task is a **record-level binary** call: normal vs pathological, from pH.

DSP the student does: dropout/spike removal + interpolation (the raw FHR is full of
signal-loss zeros), FHR **baseline** and **variability** (short- and long-term), accel/
decel detection, and uterine-contraction features. `smoke()` synthesises both classes
offline so CI/Colab-sanity runs green.
"""
from __future__ import annotations
import numpy as np

from adapter import TrackAdapter, TrackMeta, Recording, default_baseline
import biosignals as bio

# Umbilical-artery pH threshold for the binary label. pH < 7.05 is severe acidemia
# (few records, very imbalanced); pH < 7.15 is the common borderline split used in the
# CTU-UHB literature and gives a more learnable balance. Documented, not hidden.
PH_PATHOLOGICAL = 7.15


def _clean_fhr(fhr, fs):
    """Replace signal-loss zeros / out-of-range spikes with interpolated values and
    median-filter residual spikes. Returns (clean_fhr, dropout_fraction)."""
    from scipy.signal import medfilt
    x = np.asarray(fhr, float).copy()
    bad = (x < 50) | (x > 210) | ~np.isfinite(x)
    good = ~bad
    if good.sum() < 2:
        return np.full(len(x), 140.0), 1.0
    idx = np.arange(len(x))
    x = np.interp(idx, idx[good], x[good])
    k = int(fs) | 1                      # odd kernel ~1 s
    x = medfilt(x, kernel_size=max(3, k))
    return x, float(1.0 - good.mean())


def _count_excursions(mask, fs, min_s):
    """Count runs of True lasting at least `min_s` seconds (accel/decel episodes)."""
    need = int(min_s * fs)
    n = run = 0
    for v in mask:
        if v:
            run += 1
        else:
            if run >= need:
                n += 1
            run = 0
    if run >= need:
        n += 1
    return n


#: How tightly the synthetic TRACE follows the latent compromise that sets the pH
#: label. 1.0 would mean the CTG reads the cord gas perfectly (and every model
#: scores 1.000); 0.0 would mean the trace is pure noise. The real relationship is
#: weak — which is why intrapartum CTG has such poor positive predictive value —
#: so the smoke set uses a deliberately loose coupling. This one number is what
#: keeps the track honestly hard; it is documented, not hidden.
TRACE_PH_COUPLING = 0.45


def _synth_ctg(severity, fs, dur, seed):
    """One synthetic CTG trace at a given **severity** of fetal compromise
    (0 = untroubled, 1 = severe), from the book's `bio.ctg` generator.

    Severity is CONTINUOUS on purpose. The old version generated two caricatures
    ("normal" vs. "pathological"), which made the classes perfectly separable and
    let the baseline score κ = 1.00 — on a task whose real-world inter-observer
    κ is 0.3–0.5. Here the abnormality grows smoothly with severity (variability
    falls, the baseline sags, decelerations become more frequent and deeper) and
    the *label* is a threshold on a noisily-coupled pH, so the two classes overlap
    exactly where clinicians disagree."""
    t, fhr_raw, fhr_clean, uc = bio.ctg(duration=dur, fs=fs, seed=seed)
    r = np.random.default_rng(seed + 777)
    s = float(np.clip(severity, 0.0, 1.0))
    if s > 0:
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(fhr_clean, size=max(1, int(30 * fs)))
        # variability suppression + mild bradycardia, both proportional to severity
        fhr_clean = smooth - 12.0 * s + (1.0 - 0.7 * s) * (fhr_clean - smooth)
        for i in range(int(round(4 * s))):                        # recurrent late decels
            c = 120 + i * 140 + r.uniform(-30, 30)
            fhr_clean = fhr_clean - (12 + 18 * s) * np.exp(-0.5 * ((t - c - 14) / 16) ** 2)
    # signal loss and spikes afflict EVERY trace, not just the sick ones — otherwise
    # "dropout fraction" leaks the label instead of measuring recording quality.
    fhr_raw = fhr_clean.copy()
    n_drop = int(r.integers(20, 60))
    fhr_raw[r.integers(0, len(fhr_raw), size=n_drop)] = 0
    n_spike = int(r.integers(10, 35))
    si = r.integers(0, len(fhr_raw), size=n_spike)
    fhr_raw[si] = fhr_raw[si] + r.uniform(-60, 60, size=n_spike)
    return fhr_raw.astype(float), uc.astype(float)


class CTGTrack(TrackAdapter):
    meta = TrackMeta(
        name="CTG (fetal)",
        dataset="CTU-UHB Intrapartum Cardiotocography",
        dataset_version="1.0.0",
        license="Open Data Commons Attribution (ODC-BY)",
        citation="Chudáček V, et al. (2014); Goldberger AL, et al. PhysioNet (2000).",
        url="https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/",
        signals=["FHR @ 4 Hz", "uterine contraction (toco) @ 4 Hz"],
        task_type="binary",
        classes=["normal", "pathological"],
        split_unit="recording",
        default_metrics=["macro_f1", "cohens_kappa"],
        smoke_test_records=["1001", "1002", "1004"],
        expected_runtime="~2–4 min on a subset (Colab, CPU)",
        dsp_focus="dropout removal + interpolation, FHR baseline/variability (STV/LTV), "
                  "accel/decel detection, uterine-contraction features",
        difficulty=3,
        eval_modes=("new-recording",),
        submission_granularity="record",
    )

    #: Capability declaration (see `TrackAdapter.SUPPORTED_CFG_KEYS`). CTG ships
    #: **no** stage-2 stage at all (there is no `preprocess()` override: dropout
    #: handling lives inside `extract_features` via `_clean_fhr`, which is exactly
    #: the misplacement the track asks you to argue about) and **no** PSD-based
    #: feature, so neither the denoise menu nor the spectral menu is real here.
    #: What IS real on this track — and what it exists to teach — is the stage-5
    #: imbalance menu, which comes from `BASE_CFG_KEYS`.
    #:
    #: Move `_clean_fhr` into a proper `preprocess()` and you have earned the
    #: right to declare its knobs: `track.declare_cfg_keys("dropout", "window_s")`.
    SUPPORTED_CFG_KEYS = frozenset()

    CFG_KEY_HINTS = {
        "preprocess": "CTG has no preprocess() stage yet — dropout handling (`_clean_fhr`) is "
                      "called from extract_features(). Hoisting it into preprocess() and choosing "
                      "between interpolating, excluding and flagging the gaps is one of this "
                      "track's real decisions; declare the keys once you have.",
        "spectral_method": "CTG's features are baseline / STV / LTV / accel-decel counts computed "
                           "in the TIME domain — no band power is integrated from a PSD, so there "
                           "is no spectral estimator to choose here.",
    }

    # ---- synthetic smoke (offline / CI) ----
    def smoke(self, n_records=100, fs=4, dur=600.0, seed=2000,
              coupling=TRACE_PH_COUPLING, ph_threshold=PH_PATHOLOGICAL):
        """Synthetic cohort built the way the real one is *labelled*: draw a latent
        degree of fetal compromise, turn it into an umbilical pH, and threshold
        that pH. The trace only follows the latent state with strength
        `coupling` (< 1) — so pathological and normal traces overlap, the class
        is rare (~18 %), and the baseline detects the pathological class about a
        third of the time.

        That is the point of this track. A CTG smoke set on which everything
        scores 1.000 would quietly teach that intrapartum monitoring is a solved
        problem; it is the opposite. Expect κ ≈ 0.35 here, against a clinician
        inter-observer κ of 0.3–0.5."""
        rng = np.random.default_rng(seed)
        recs = []
        for k in range(n_records):
            z = rng.normal()                                   # latent compromise
            ph = 7.25 - 0.10 * z                               # -> umbilical-artery pH
            cls = "pathological" if ph < ph_threshold else "normal"
            # what the TRACE shows follows the latent state only loosely
            sev = coupling * z + (1.0 - coupling) * rng.normal()
            sev = float(np.clip(sev, 0.0, 1.6)) / 1.6
            fhr, uc = _synth_ctg(sev, fs, dur, seed=seed + k)
            recs.append(Recording(group=f"C{k:04d}", fs=float(fs),
                                  epochs={"fhr": fhr[None, :], "uc": uc[None, :]},
                                  labels=np.array([cls]),
                                  meta={"pH": round(float(ph), 3), "severity": round(sev, 3)}))
        return recs

    # ---- module 3: feature extraction (the DSP the student does) ----
    def extract_features(self, rec: Recording, cfg=None):
        """Module 3 — one feature row per recording (baseline, STV/LTV, accel/decel
        counts, contraction features). Note that dropout handling (`_clean_fhr`)
        is called here for convenience but is really *preprocessing*: moving it
        into `preprocess()` — and choosing between interpolating gaps, excluding
        them, or flagging the dropout fraction as its own feature — is one of the
        decisions this track exists to make you argue for. The 30-minute analysis
        window below is likewise a choice, not a law."""
        from scipy.signal import find_peaks
        fhr = rec.epochs["fhr"][0]; uc = rec.epochs["uc"][0]; fs = rec.fs
        # Focus on the last 30 min before delivery: the intrapartum acidemia signature
        # (falling variability, recurrent decelerations) is concentrated there — using the
        # whole ~90 min trace dilutes it. A standard, defensible CTG analysis window.
        tail = int(30 * 60 * fs)
        if len(fhr) > tail:
            fhr, uc = fhr[-tail:], uc[-tail:]
        fhr_c, drop = _clean_fhr(fhr, fs)
        baseline = float(np.median(fhr_c))
        stv = float(np.mean(np.abs(np.diff(fhr_c))))                    # short-term variability
        w = int(60 * fs)
        ltv = (float(np.mean([np.std(fhr_c[i:i + w]) for i in range(0, len(fhr_c) - w + 1, w)]))
               if len(fhr_c) >= w else float(np.std(fhr_c)))            # long-term variability
        dev = fhr_c - baseline
        accel = _count_excursions(dev > 15, fs, min_s=15)
        decel = _count_excursions(dev < -15, fs, min_s=15)
        uc_c = np.asarray(uc, float)
        pk, _ = find_peaks(uc_c, height=np.median(uc_c) + 15, distance=int(60 * fs))
        uc_amp = float(np.mean(uc_c[pk])) if len(pk) else 0.0
        feats = [baseline, stv, ltv, float(accel), float(decel), drop * 100.0,
                 float(baseline > 160), float(baseline < 110),
                 float(len(pk)), uc_amp, float(np.std(fhr_c))]
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

    # ---- REAL data: CTU-UHB via wfdb ----
    _PN_DIR = "ctu-uhb-ctgdb/1.0.0"
    _PN_BASE = "https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/"

    def download(self, cache_dir, subset=150, workers=8):
        """Fetch a subset of CTU-UHB records (.hea + .dat) into cache_dir. Needs network.
        Each record is ~150 KB; the full set is only ~40 MB, so `subset` just bounds runtime."""
        import os, glob, urllib.request, urllib.error, concurrent.futures as cf
        os.makedirs(cache_dir, exist_ok=True)
        # Cache-first: if enough records are already present, don't touch the network
        # (so a cached-but-offline student still works, like the HAR/EMG loaders).
        have = len(glob.glob(os.path.join(cache_dir, "*.hea")))
        if subset and have >= subset:
            print(f"[ctg_ctu_uhb] {have} records already cached in {cache_dir}; skipping download")
            return cache_dir
        rec_ids = urllib.request.urlopen(self._PN_BASE + "RECORDS", timeout=30).read().decode().split()
        if subset:
            rec_ids = rec_ids[:subset]

        def _grab(rid):
            for ext in (".hea", ".dat"):
                dst = os.path.join(cache_dir, rid + ext)
                if os.path.exists(dst) and os.path.getsize(dst) > 0:
                    continue
                for attempt in range(3):
                    try:
                        urllib.request.urlretrieve(self._PN_BASE + rid + ext, dst); break
                    except (urllib.error.URLError, OSError):
                        if attempt == 2:
                            raise
            return rid

        got = 0
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(_grab, rec_ids):
                got += 1
        print(f"[ctg_ctu_uhb] downloaded {got} records into {cache_dir}")
        return cache_dir

    @staticmethod
    def _ph_from_comments(comments):
        for c in comments:
            toks = str(c).replace("#", "").split()
            if toks and toks[0].lower() == "ph":
                try:
                    return float(toks[1])
                except (IndexError, ValueError):
                    return None
        return None

    def load(self, cache_dir, ph_threshold=PH_PATHOLOGICAL):
        """Read CTU-UHB records; label normal/pathological by umbilical-artery pH.
        FHR = channel 0, UC = channel 1. Records without a parseable pH are skipped."""
        import os, glob
        import wfdb
        recs = []
        for hea in sorted(glob.glob(os.path.join(cache_dir, "**", "*.hea"), recursive=True)):
            rid = os.path.splitext(os.path.basename(hea))[0]
            base = os.path.splitext(hea)[0]
            rec = wfdb.rdrecord(base)
            ph = self._ph_from_comments(rec.comments)
            if ph is None:
                continue
            names = [n.upper() for n in (rec.sig_name or [])]
            fi = names.index("FHR") if "FHR" in names else 0
            ui = names.index("UC") if "UC" in names else 1
            fhr = rec.p_signal[:, fi].astype(float)
            uc = rec.p_signal[:, ui].astype(float)
            label = "pathological" if ph < ph_threshold else "normal"
            recs.append(Recording(group=rid, fs=float(rec.fs),
                                  epochs={"fhr": fhr[None, :], "uc": uc[None, :]},
                                  labels=np.array([label]), meta={"pH": ph}))
        return recs


if __name__ == "__main__":
    t = CTGTrack()
    print(t.dataset_card())
    rep = t.run_smoke()
    from sklearn.metrics import f1_score
    print("SMOKE (synthetic) CV: macroF1=%.3f kappa=%.3f n_records=%d"
          % (f1_score(rep["y_true"], rep["y_pred"], average="macro"),
             rep["cohens_kappa"], rep["n_groups"]))
    print("spread:", rep["summary"])          # never the pooled number alone (§16.3)
    print("classes:", rep["labels"], "\nconfusion (rows=true):\n", rep["confusion"])
