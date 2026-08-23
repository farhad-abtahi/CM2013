"""
bsp.loaders — one setup + data entry point shared by every companion notebook.

Design goals (from the notebook platform contract):
  * synthetic-by-default, reproducible (single seed), offline-safe;
  * an OPTIONAL real-data hook (PhysioNet/Sleep-EDF) that only runs in Colab;
  * environment detection so a notebook can adapt to Colab / JupyterLite / local.
"""
from __future__ import annotations
import os
import numpy as np

from . import bookstyle as bs      # noqa: F401  (applies the shared plotting style on import)
from . import biosignals as bio


# --------------------------------------------------------------- environment
def in_colab() -> bool:
    return "google.colab" in os.sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def in_lite() -> bool:
    # JupyterLite / Pyodide runs on the 'emscripten' platform
    return os.sys.platform == "emscripten"


def environment() -> str:
    if in_colab():
        return "colab"
    if in_lite():
        return "jupyterlite"
    return "local"


# --------------------------------------------------------------- setup
def setup(seed: int = 0):
    """Call once at the top of a notebook. Fixes the seed, returns a NumPy
    Generator, and reports the runtime environment. All randomness in the
    companion notebooks flows from this generator so results are reproducible."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    print(f"environment : {environment()}")
    print(f"seed        : {seed}  (all synthetic data is reproducible)")
    return rng


# --------------------------------------------------------------- data access
_SYNTH = {
    "ecg": bio.ecg,
    "ecg_hrv": bio.ecg_hrv,
    "eeg": bio.eeg,
    "emg": bio.emg_burst,
    "eog": bio.eog,
    "respiration": bio.respiration,
    "ppg": bio.ppg,
    "ctg": bio.ctg,
    "imu": bio.imu,
}


def load_or_synthesize(kind: str, real: bool = False, **kwargs):
    """Return a biosignal. Default is SYNTHETIC (offline, reproducible).

    `real=True` is an opt-in hook for open datasets and is only honoured in
    Colab (where downloads and EDF readers are available); everywhere else it
    silently falls back to the synthetic generator so the notebook still runs.
    """
    if real and in_colab():
        try:
            return _load_real(kind, **kwargs)          # pragma: no cover
        except Exception as exc:                        # pragma: no cover
            print(f"[load_or_synthesize] real data unavailable ({exc}); "
                  f"using synthetic {kind}.")
    if kind not in _SYNTH:
        raise KeyError(f"unknown signal '{kind}'; choose from {sorted(_SYNTH)}")
    return _SYNTH[kind](**kwargs)


def _load_real(kind: str, record: str = "100", channel: int = 0,
               seconds: float = 10.0, **kwargs):        # pragma: no cover
    """Colab-only real-data pilot: stream a real ECG from PhysioNet's MIT-BIH
    Arrhythmia Database via `wfdb`. Returns `(t, signal)` to match `bio.ecg`.

    Implemented for ECG only; other modalities keep the synthetic default. This
    is the concrete opt-in that makes the synthetic→real transition real while
    the synthetic generator remains the reproducible default everywhere else.
    """
    if kind not in ("ecg", "ecg_hrv"):
        raise NotImplementedError(
            f"real-data hook is implemented for ECG only, not {kind!r}; "
            "see docs/GOVERNANCE.md for the other datasets.")
    try:
        import wfdb
    except ImportError:
        import subprocess
        subprocess.run([os.sys.executable, "-m", "pip", "install", "-q", "wfdb"],
                       check=True)
        import wfdb
    rec = wfdb.rdrecord(record, pn_dir="mitdb")          # streams from physionet.org
    fs = float(rec.fs)
    n = int(seconds * fs)
    sig = rec.p_signal[:n, channel].astype(float)
    t = np.arange(len(sig)) / fs
    print(f"[load_or_synthesize] real MIT-BIH record {record}, "
          f"channel {channel}, fs={fs:g} Hz — cite PhysioNet (Moody & Mark, 2001).")
    return t, sig
