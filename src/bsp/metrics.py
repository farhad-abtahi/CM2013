"""
bsp.metrics — the small, honest evaluation toolkit the book insists on.
Thin, well-named wrappers so notebooks read like the book: never accuracy alone.
"""
from __future__ import annotations
import numpy as np


def snr_db(signal, noise) -> float:
    """Signal-to-noise ratio in dB from a clean template and a noise residual."""
    ps = np.mean(np.asarray(signal) ** 2)
    pn = np.mean(np.asarray(noise) ** 2) + 1e-15
    return 10.0 * np.log10(ps / pn)


def sqrtN_gain_db(N: int) -> float:
    """Ensemble-averaging law: amplitude SNR improves by sqrt(N) -> +10log10(N) dB."""
    return 10.0 * np.log10(N)


def alias_frequency(f0: float, fs: float) -> float:
    """Apparent (aliased) frequency of a tone f0 sampled at fs."""
    return abs(f0 - fs * round(f0 / fs))


def quantization_snr_db(nbits: int) -> float:
    """Ideal full-scale-sinusoid quantization SNR."""
    return 6.02 * nbits + 1.76


# ---- classification metrics (sklearn if present, else NumPy fallbacks) ----
def _cm(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def confusion(y_true, y_pred, labels=None):
    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    return labels, _cm(y_true, y_pred, labels)


def cohens_kappa(y_true, y_pred) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        labels, m = confusion(y_true, y_pred)
        n = m.sum()
        po = np.trace(m) / n
        pe = np.sum(m.sum(0) * m.sum(1)) / (n * n)
        return float((po - pe) / (1 - pe + 1e-15))


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro"))


def balanced_accuracy(y_true, y_pred) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def report(y_true, y_pred, labels=None) -> dict:
    """The honest panel: never a single headline number."""
    labs, cm = confusion(y_true, y_pred, labels)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return {
        "accuracy": round(acc, 3),
        "cohens_kappa": round(cohens_kappa(y_true, y_pred), 3),
        "macro_f1": round(macro_f1(y_true, y_pred), 3),
        "balanced_accuracy": round(balanced_accuracy(y_true, y_pred), 3),
        "labels": labs,
        "confusion": cm,
    }
