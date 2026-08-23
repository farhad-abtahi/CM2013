"""
bsp.sleep_pipeline — the ONE continuous sleep-staging mini-pipeline that threads
through the book, now as a DIFFICULTY CURRICULUM rather than a single realism
setting. Same API, same labels, increasing biological variability and nuisance
structure:

    night = synthetic_night(n_epochs=120, fs=100, subject="S01", seed=0,
                            difficulty="easy")    # easy | medium | hard

  * easy   — clean figures, concept demos, CI smoke tests (strongly separable).
  * medium — realistic teaching; honest models work but LOSO is not perfect.
  * hard   — capstone practice before real data: subject shift, artifacts,
             label noise, missingness, imbalance.

Everything is SYNTHETIC and reproducible (STABLE seeding — no salted hash());
numbers are illustrative, never clinical benchmarks.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy import signal as _sig

from . import biosignals as bio

STAGES = ["W", "N1", "N2", "N3", "REM"]
_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 11),
          "sigma": (11, 16), "beta": (16, 30)}          # non-overlapping (feature-safe)

# relative EEG band amplitudes per stage (the "fingerprints")
BAND_AMP = {
    "W":   {"delta": 0.3, "theta": 0.4, "alpha": 1.7, "sigma": 0.2, "beta": 1.0},
    "N1":  {"delta": 0.5, "theta": 1.1, "alpha": 0.6, "sigma": 0.2, "beta": 0.3},
    "N2":  {"delta": 1.1, "theta": 0.8, "alpha": 0.3, "sigma": 0.6, "beta": 0.2},
    "N3":  {"delta": 2.8, "theta": 0.5, "alpha": 0.15, "sigma": 0.1, "beta": 0.1},
    "REM": {"delta": 0.4, "theta": 1.2, "alpha": 0.5, "sigma": 0.15, "beta": 0.4},
}
_COMMON = {b: np.mean([BAND_AMP[s][b] for s in STAGES]) for b in _BANDS}
_TONE = {"W": 1.0, "N1": 0.5, "N2": 0.30, "N3": 0.25, "REM": 0.10}
_BLINK_RATE = {"W": 6.0, "N1": 2.0, "N2": 1.0, "N3": 0.5, "REM": 5.0}

# difficulty presets (the curriculum knobs)
DIFFICULTY = {
    "easy":   dict(sep=1.00, subject_gain_sd=0.05, alpha_peak_sd=0.3, noise_sd=0.22,
                   pink_mix=0.10, brown_mix=0.0, baseline_wander=0.0, powerline=0.0,
                   artifact_prob=0.0, dropout_prob=0.0, clipping_prob=0.0,
                   missing_channel_prob=0.0, label_noise=0.0, transition_label_noise=0.0,
                   spindle_prob=(1.0, 1.0), kcomplex_prob=(0.6, 0.6), fs_jitter=False),
    "medium": dict(sep=0.62, subject_gain_sd=0.20, alpha_peak_sd=0.8, noise_sd=0.45,
                   pink_mix=0.30, brown_mix=0.05, baseline_wander=0.12, powerline=0.02,
                   artifact_prob=0.08, dropout_prob=0.02, clipping_prob=0.0,
                   missing_channel_prob=0.0, label_noise=0.03, transition_label_noise=0.08,
                   spindle_prob=(0.4, 0.8), kcomplex_prob=(0.2, 0.5), fs_jitter=False),
    "hard":   dict(sep=0.47, subject_gain_sd=0.45, alpha_peak_sd=1.5, noise_sd=0.55,
                   pink_mix=0.40, brown_mix=0.18, baseline_wander=0.22, powerline=0.07,
                   artifact_prob=0.18, dropout_prob=0.07, clipping_prob=0.04,
                   missing_channel_prob=0.05, label_noise=0.07, transition_label_noise=0.15,
                   spindle_prob=(0.2, 0.7), kcomplex_prob=(0.1, 0.4), fs_jitter=True),
}


# ----------------------------------------------------------- stable seeding
def _stable_seed(*keys) -> int:
    """Deterministic 32-bit seed from arbitrary keys — reproducible ACROSS
    processes (unlike Python's salted hash())."""
    h = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()
    return int(h[:8], 16)


def rng_for(*keys):
    return np.random.default_rng(_stable_seed(*keys))


# ----------------------------------------------------------- subject profile
def subject_profile(subject, seed, difficulty="easy"):
    """Stable per-subject parameters so LOSO held-out subjects are NOT clones."""
    d = DIFFICULTY[difficulty]
    r = rng_for("profile", subject, seed)
    return {
        "eeg_gain": float(np.exp(r.normal(0, d["subject_gain_sd"]))),
        "eog_gain": float(np.exp(r.normal(0, d["subject_gain_sd"]))),
        "emg_gain": float(np.exp(r.normal(0, d["subject_gain_sd"]))),
        "alpha_peak": float(np.clip(10.0 + r.normal(0, d["alpha_peak_sd"]), 8.0, 12.0)),
        "spindle_freq": float(np.clip(13.5 + r.normal(0, 0.6), 11.5, 15.5)),
        "noise_floor": float(np.exp(r.normal(0, 0.25))),
        "artifact_tendency": float(np.clip(r.uniform(0.5, 1.5), 0.2, 2.0)),
        "gain_shift": float(np.exp(r.normal(0, d["subject_gain_sd"]))),
    }


# ----------------------------------------------------------- hypnogram
_TRANS = {
    "W":   {"W": .60, "N1": .35, "REM": .05},
    "N1":  {"W": .10, "N1": .30, "N2": .55, "REM": .05},
    "N2":  {"N1": .08, "N2": .65, "N3": .15, "REM": .12},
    "N3":  {"N2": .25, "N3": .70, "W": .05},
    "REM": {"REM": .55, "N1": .10, "N2": .25, "W": .10},
}


def generate_hypnogram(n_epochs, rng, difficulty):
    if difficulty == "easy":
        template = (["W"] * 4 + ["N1"] * 3 + ["N2"] * 8 + ["N3"] * 6 + ["N2"] * 4 +
                    ["REM"] * 4 + ["N2"] * 5 + ["N3"] * 3 + ["N2"] * 4 + ["REM"] * 5)
        return (template * (n_epochs // len(template) + 1))[:n_epochs]
    # medium/hard: Markov chain modulated by time-of-night (N3 early, REM late)
    seq = ["W"]
    for i in range(1, n_epochs):
        frac = i / n_epochs
        probs = dict(_TRANS[seq[-1]])
        for s in probs:                      # time-of-night modulation
            if s == "N3":
                probs[s] *= (1.6 - frac)     # more deep sleep early
            if s == "REM":
                probs[s] *= (0.4 + 1.6 * frac)   # more REM late
        keys = list(probs); w = np.array([probs[k] for k in keys], float); w /= w.sum()
        seq.append(keys[rng.choice(len(keys), p=w)])
    return seq


# ----------------------------------------------------------- epoch synthesis
def _stage_eeg(stage, t, fs, prof, d, rng):
    centers = {"delta": 2.0, "theta": 6.0, "alpha": prof["alpha_peak"],
               "sigma": prof["spindle_freq"], "beta": 20.0}
    x = np.zeros_like(t)
    for band, f0 in centers.items():
        amp = (_COMMON[band] * (1 - d["sep"]) + BAND_AMP[stage][band] * d["sep"]) * prof["eeg_gain"]
        ph = rng.uniform(0, 2 * np.pi)
        fm = f0 + 0.3 * np.sin(2 * np.pi * 0.05 * t + ph)
        x += amp * np.sin(2 * np.pi * fm * t + ph)
    nf = prof["noise_floor"]
    x += d["noise_sd"] * nf * rng.standard_normal(len(t))
    if d["pink_mix"]:
        x += d["pink_mix"] * nf * bio.pink(len(t), seed=int(rng.integers(1e9)))
    if d["brown_mix"]:
        x += d["brown_mix"] * nf * bio.brown(len(t), seed=int(rng.integers(1e9)))
    events = {"spindle": False, "kcomplex": False}
    if stage == "N2":
        if rng.random() < rng.uniform(*d["spindle_prob"]):
            c = rng.uniform(3, t[-1] - 3)
            x += 1.4 * prof["eeg_gain"] * np.exp(-0.5 * ((t - c) / 0.35) ** 2) * \
                np.sin(2 * np.pi * prof["spindle_freq"] * t)
            events["spindle"] = True
        if rng.random() < rng.uniform(*d["kcomplex_prob"]):
            c = rng.uniform(3, t[-1] - 3)
            x += prof["eeg_gain"] * (-2.4 * np.exp(-0.5 * ((t - c) / 0.10) ** 2) +
                                     1.2 * np.exp(-0.5 * ((t - c - 0.25) / 0.18) ** 2))
            events["kcomplex"] = True
    return x, events


def _stage_eog(stage, t, fs, prof, d, rng):
    x = 0.05 * np.sin(2 * np.pi * 0.1 * t)
    nb = rng.poisson(_BLINK_RATE[stage] * (t[-1] / 10.0))
    for _ in range(int(nb)):
        c = rng.uniform(0, t[-1]); x += rng.uniform(1.0, 2.0) * np.exp(-0.5 * ((t - c) / 0.08) ** 2)
    if stage == "REM":
        for _ in range(int(rng.integers(2, 6))):
            c = rng.uniform(0, t[-1])
            x += 1.2 * np.sin(2 * np.pi * 2 * t) * np.exp(-0.5 * ((t - c) / 0.4) ** 2)
    return x * prof["eog_gain"]


def _stage_emg(stage, t, prof, d, rng):
    return _TONE[stage] * prof["emg_gain"] * rng.standard_normal(len(t))


# ----------------------------------------------------------- artifacts (+masks)
def add_electrode_pop(x, fs, rng, amp=6.0):
    m = np.zeros(len(x), bool)
    c = int(rng.uniform(0, len(x)))
    x = x.copy(); x[c] += amp * rng.choice([-1, 1]); m[max(0, c - 2):c + 3] = True
    return x, m


def flatline_dropout(x, fs, rng, dur=1.0):
    m = np.zeros(len(x), bool)
    n = int(dur * fs); s = int(rng.uniform(0, max(1, len(x) - n)))
    x = x.copy(); x[s:s + n] = 0.0; m[s:s + n] = True
    return x, m


def clipping(x, frac=0.6):
    lim = frac * np.max(np.abs(x) + 1e-9)
    return np.clip(x, -lim, lim), (np.abs(x) > lim)


def baseline_wander(x, fs, rng, amp):
    t = np.arange(len(x)) / fs
    return x + amp * np.max(np.abs(x) + 1e-9) * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t), None


def powerline_hum(x, fs, rng, amp, f0=50.0):
    t = np.arange(len(x)) / fs
    return x + amp * np.max(np.abs(x) + 1e-9) * np.sin(2 * np.pi * f0 * t), None


def add_artifacts(eeg, eog, emg, fs, d, prof, rng):
    """Apply per-epoch artifacts by difficulty; return arrays + per-epoch masks."""
    n = len(eeg)
    flags = {k: np.zeros(n, bool) for k in
             ["baseline_wander", "powerline", "electrode_pop", "dropout",
              "clipping", "emg_in_eeg", "eog_in_eeg", "missing_channel"]}
    p = d["artifact_prob"] * prof["artifact_tendency"]
    for i in range(n):
        if d["baseline_wander"] and rng.random() < p:
            eeg[i], _ = baseline_wander(eeg[i], fs, rng, d["baseline_wander"]); flags["baseline_wander"][i] = True
        if d["powerline"] and rng.random() < p:
            eeg[i], _ = powerline_hum(eeg[i], fs, rng, d["powerline"]); flags["powerline"][i] = True
        if rng.random() < p:
            eeg[i], _ = add_electrode_pop(eeg[i], fs, rng); flags["electrode_pop"][i] = True
        if d["dropout_prob"] and rng.random() < d["dropout_prob"]:
            eeg[i], _ = flatline_dropout(eeg[i], fs, rng); flags["dropout"][i] = True
        if d["clipping_prob"] and rng.random() < d["clipping_prob"]:
            eeg[i], _ = clipping(eeg[i]); flags["clipping"][i] = True
        if rng.random() < p * 0.5:                       # EMG contamination in EEG
            eeg[i] = eeg[i] + 0.4 * emg[i]; flags["emg_in_eeg"][i] = True
        if rng.random() < p * 0.5:                       # EOG blink contamination in EEG
            eeg[i] = eeg[i] + 0.5 * eog[i]; flags["eog_in_eeg"][i] = True
        if d["missing_channel_prob"] and rng.random() < d["missing_channel_prob"]:
            eeg[i] = np.zeros_like(eeg[i]); flags["missing_channel"][i] = True
    return eeg, eog, emg, flags


# ----------------------------------------------------------- night + cohort
def synthetic_night(n_epochs=120, fs=100, subject="S01", seed=0, difficulty="easy"):
    """One subject's night: dict of per-epoch EEG/EOG/EMG arrays + observed labels
    (`stages`), ground-truth labels (`true_stages`), artifact masks, and profile."""
    d = DIFFICULTY[difficulty]
    prof = subject_profile(subject, seed, difficulty)
    r = rng_for("night", subject, seed)
    if d["fs_jitter"]:
        fs = int(r.choice([100, 128]))
    t = np.arange(int(30.0 * fs)) / fs
    true_stages = generate_hypnogram(n_epochs, r, difficulty)
    EEG, EOG, EMG = [], [], []
    for s in true_stages:
        e, _ev = _stage_eeg(s, t, fs, prof, d, r)
        EEG.append(e); EOG.append(_stage_eog(s, t, fs, prof, d, r)); EMG.append(_stage_emg(s, t, prof, d, r))
    EEG, EOG, EMG = np.array(EEG), np.array(EOG), np.array(EMG)
    EEG = EEG * prof["gain_shift"]
    EEG, EOG, EMG, flags = add_artifacts(EEG, EOG, EMG, fs, d, prof, r)
    # label noise: confuse stages the way scorers do
    confuse = {"W": ["N1", "REM"], "N1": ["W", "N2", "REM"], "N2": ["N1", "N3", "REM"],
               "N3": ["N2"], "REM": ["W", "N1", "N2"]}
    obs = list(true_stages)
    for i, s in enumerate(true_stages):
        p = d["label_noise"]
        if i > 0 and true_stages[i] != true_stages[i - 1]:
            p += d["transition_label_noise"]              # boundaries are ambiguous
        if r.random() < p:
            obs[i] = confuse[s][int(r.integers(len(confuse[s])))]
    return {"subject": subject, "fs": fs, "difficulty": difficulty,
            "stages": np.array(obs), "true_stages": np.array(true_stages),
            "eeg": EEG, "eog": EOG, "emg": EMG, "artifacts": flags, "profile": prof}


def cohort(n_subjects=5, n_epochs=120, fs=100, seed=0, difficulty="easy"):
    return [synthetic_night(n_epochs=n_epochs, fs=fs, subject=f"S{i+1:02d}",
                            seed=seed + i, difficulty=difficulty)
            for i in range(n_subjects)]


# ------------------------------------------------------------- 2. features
def _hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    return v0, np.sqrt(v1 / v0), np.sqrt(v2 / v1) / (np.sqrt(v1 / v0) + 1e-12)


def _spectral_entropy(x, fs):
    f, p = _sig.welch(x, fs=fs, nperseg=min(len(x), 256))
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def epoch_features(eeg, eog, emg, fs):
    feats = {}
    for name, band in _BANDS.items():
        feats[f"eeg_{name}"] = bio.bandpower(eeg, fs, band)
    a, m, c = _hjorth(eeg)
    feats["eeg_hjorth_activity"] = a
    feats["eeg_hjorth_mobility"] = m
    feats["eeg_hjorth_complexity"] = c
    feats["eeg_spec_entropy"] = _spectral_entropy(eeg, fs)
    feats["eog_movement"] = float(np.mean(np.abs(np.diff(eog))))
    feats["emg_rms"] = float(np.sqrt(np.mean(emg ** 2)))
    return feats


def feature_table(nights):
    rows, y, groups = [], [], []
    names = None
    for night in nights:
        fs = night["fs"]
        for e, o, m, s in zip(night["eeg"], night["eog"], night["emg"], night["stages"]):
            f = epoch_features(e, o, m, fs)
            if names is None:
                names = list(f.keys())
            rows.append([f[k] for k in names])
            y.append(s); groups.append(night["subject"])
    return np.array(rows), np.array(y), np.array(groups), names


# ------------------------------------------------------- 3. fold-safe model
def default_classifier(n_estimators=200, seed=0):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    return Pipeline([("scale", StandardScaler()),
                     ("clf", RandomForestClassifier(
                         n_estimators=n_estimators, class_weight="balanced",
                         random_state=seed))])


def loso_evaluate(X, y, groups, clf=None):
    from sklearn.model_selection import LeaveOneGroupOut
    from . import metrics as M
    from . import notebook_checks as C
    clf = clf or default_classifier()
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []
    for tr, te in logo.split(X, y, groups):
        C.assert_no_subject_leak(groups[tr], groups[te])
        clf.fit(X[tr], y[tr])
        y_pred.extend(clf.predict(X[te])); y_true.extend(y[te])
    rep = M.report(y_true, y_pred, labels=STAGES)
    rep["y_true"], rep["y_pred"] = np.array(y_true), np.array(y_pred)
    return rep


# ------------------------------------------------------------ 4. hypnogram
def hypnogram_figure(y_true, y_pred=None, max_epochs=120):
    import matplotlib.pyplot as plt
    from . import bookstyle as bs
    order = {"W": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
    yt = [order[s] for s in y_true[:max_epochs]]
    fig, ax = bs.newfig(w=7.2, h=2.6)
    ax.step(range(len(yt)), yt, where="post", color=bs.C["blue"], label="reference")
    if y_pred is not None:
        yp = [order[s] for s in y_pred[:max_epochs]]
        ax.step(range(len(yp)), yp, where="post", color=bs.C["orange"], lw=1.0, alpha=0.8, label="predicted")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_yticks(list(order.values())); ax.set_yticklabels(list(order.keys()))
    ax.set_xlabel("epoch (30 s)"); ax.set_title("Hypnogram")
    return fig


if __name__ == "__main__":
    from sklearn.metrics import f1_score
    for diff in ("easy", "medium", "hard"):
        nights = cohort(n_subjects=8, n_epochs=120, seed=0, difficulty=diff)
        X, y, g, _ = feature_table(nights)
        rep = loso_evaluate(X, y, g)
        mf1 = f1_score(rep["y_true"], rep["y_pred"], average="macro")
        print(f"{diff:6s}  macro-F1={mf1:.3f}  kappa={rep['cohens_kappa']:.3f}  "
              f"acc={rep['accuracy']:.3f}  (n_epochs={len(y)})")
