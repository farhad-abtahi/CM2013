"""
biosignals.py — reproducible SYNTHETIC biomedical signals and images for the
CM2013 textbook figure factory. No downloads: everything is simulated from
NumPy/SciPy so figures are reproducible offline. Seeds are fixed by default.

Signals: ecg, eeg, emg_burst, eog, respiration, ppg, ctg, imu, ssvep_epochs,
erp_epochs. Noise: white, pink, brown, powerline. Images: head_phantom,
salt_pepper, add_gaussian_image. Helper: bandpower.
"""
import numpy as np
from scipy import signal as _sig
from scipy.integrate import trapezoid as _trapezoid   # np.trapz was removed in NumPy 2.x


def rng(seed=0):
    return np.random.default_rng(seed)


# ============================================================= ECG
def _gauss(t, a, mu, sigma):
    return a * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def ecg(duration=10.0, fs=256, hr=70, noise=0.02, seed=1):
    """Synthetic ECG as a sum of Gaussian P-Q-R-S-T waves per beat."""
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    rr = 60.0 / hr
    beat_times = np.arange(0.35, duration, rr) + r.normal(0, 0.01, size=int(duration / rr) + 4)[: int(duration / rr) + 4] if False else np.arange(0.35, duration, rr)
    for bt in beat_times:
        jitter = r.normal(0, 0.006)
        c = bt + jitter
        x += _gauss(t, 0.10, c - 0.20, 0.025)   # P
        x += _gauss(t, -0.12, c - 0.035, 0.012)  # Q
        x += _gauss(t, 1.00, c, 0.010)           # R
        x += _gauss(t, -0.22, c + 0.035, 0.012)  # S
        x += _gauss(t, 0.28, c + 0.16, 0.040)    # T
    x += noise * r.standard_normal(n)
    # gentle baseline wander
    x += 0.05 * np.sin(2 * np.pi * 0.25 * t)
    return t, x


def ecg_hrv(duration=120.0, fs=250, hr=70, lf=0.10, hf=0.25,
            lf_amp=0.035, hf_amp=0.025, noise=0.02, baseline=0.06, seed=21):
    """ECG whose R-R intervals carry realistic heart-rate variability.

    Beat-to-beat RR interval is modulated by a low-frequency (~0.1 Hz, e.g.
    baroreflex/Mayer wave) and a high-frequency (~0.25 Hz, respiratory sinus
    arrhythmia) oscillation, so the recovered RR tachogram has genuine LF/HF
    structure for HRV analysis, and the ECG has clean R-peaks for a detector
    (e.g. Pan-Tompkins) to find.

    Returns (t, ecg, r_times) where r_times are the TRUE R-peak times (s).
    """
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    rr0 = 60.0 / hr
    # build beat times by accumulating RR intervals that vary over time
    r_times = []
    tc = 0.5
    while tc < duration - 0.5:
        r_times.append(tc)
        rr = (rr0
              + lf_amp * np.sin(2 * np.pi * lf * tc)
              + hf_amp * np.sin(2 * np.pi * hf * tc)
              + 0.006 * r.standard_normal())
        tc += max(0.3, rr)
    r_times = np.array(r_times)
    x = np.zeros(n)
    for c in r_times:
        x += _gauss(t, 0.10, c - 0.20, 0.025)    # P
        x += _gauss(t, -0.12, c - 0.035, 0.012)  # Q
        x += _gauss(t, 1.00, c, 0.010)           # R
        x += _gauss(t, -0.22, c + 0.035, 0.012)  # S
        x += _gauss(t, 0.28, c + 0.16, 0.040)    # T
    x += noise * r.standard_normal(n)
    x += baseline * np.sin(2 * np.pi * 0.25 * t)  # baseline wander
    return t, x, r_times


# ============================================================= EEG
def eeg(duration=30.0, fs=100, seed=2, spindle=True, kcomplex=True, stage="N2"):
    """Synthetic multi-band EEG with optional sleep spindle and K-complex."""
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    bands = {"delta": (2.0, 1.0), "theta": (6.0, 0.5),
             "alpha": (10.0, 0.6), "beta": (20.0, 0.25)}
    if stage == "W":
        bands = {"alpha": (10.0, 1.2), "beta": (22.0, 0.6), "theta": (6.0, 0.3)}
    elif stage == "N3":
        bands = {"delta": (1.5, 2.2), "theta": (5.0, 0.5)}
    elif stage == "REM":
        bands = {"theta": (6.5, 0.9), "alpha": (9.0, 0.5), "beta": (18.0, 0.4)}
    x = np.zeros(n)
    A_MOD, F_MOD = 0.4, 0.05
    for f0, a in bands.values():
        ph = r.uniform(0, 2 * np.pi)
        # Instantaneous frequency fm(t) = f0 + A_MOD*sin(2*pi*F_MOD*t + ph); the
        # phase must be the time-integral of 2*pi*fm(t), not fm(t)*t directly
        # (which adds a spurious t*fm'(t) term that makes the band chirp/drift
        # across the epoch instead of sitting flat, per the book's description).
        phase = (2 * np.pi * f0 * t
                 + (A_MOD / F_MOD) * (np.cos(ph) - np.cos(2 * np.pi * F_MOD * t + ph))
                 + ph)
        x += a * np.sin(phase)
    x += 0.35 * r.standard_normal(n)
    if spindle and stage in ("N2",):
        c = duration * 0.55
        env = np.exp(-0.5 * ((t - c) / 0.35) ** 2)
        x += 1.6 * env * np.sin(2 * np.pi * 13.5 * t)   # 11-16 Hz spindle
    if kcomplex and stage in ("N2",):
        c = duration * 0.30
        k = -2.4 * np.exp(-0.5 * ((t - c) / 0.10) ** 2) + 1.2 * np.exp(-0.5 * ((t - c - 0.25) / 0.18) ** 2)
        x += k
    return t, x


# ============================================================= EMG
def emg_burst(duration=5.0, fs=1000, bursts=((1.0, 2.2), (3.0, 4.2)),
              fatigue=False, seed=3):
    """Broadband, zero-mean sEMG with activation bursts (band-limited noise)."""
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    white = r.standard_normal(n)
    b, a = _sig.butter(4, [20 / (fs / 2), 400 / (fs / 2)], btype="band")
    broadband = _sig.filtfilt(b, a, white)
    env = np.zeros(n)
    for (t0, t1) in bursts:
        mask = (t >= t0) & (t <= t1)
        ramp = np.ones(np.sum(mask))
        env[mask] = ramp
    # smooth the envelope
    env = _sig.filtfilt(*_sig.butter(2, 5 / (fs / 2)), env)
    x = broadband * (0.05 + env)
    if fatigue:
        # progressive low-pass to mimic median-frequency downshift
        drift = np.linspace(1.0, 0.55, n)
        x = _sig.filtfilt(*_sig.butter(2, 150 / (fs / 2)), x) * drift + x * (1 - 0.4)
    return t, x


# ============================================================= EOG
def eog(duration=10.0, fs=100, seed=4):
    """Slow corneo-retinal deflections with saccade-like steps + blinks."""
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for c in [1.5, 4.0, 7.0]:
        x += 1.0 * np.tanh((t - c) * 6.0)
    for c in [2.5, 6.0, 8.5]:
        x += 1.8 * np.exp(-0.5 * ((t - c) / 0.08) ** 2)  # blinks
    x += 0.05 * r.standard_normal(n)
    return t, x


# ============================================================= Respiration
def respiration(duration=60.0, fs=10, rate=15, seed=5):
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    f0 = rate / 60.0
    fm = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.02 * t))
    x = np.sin(2 * np.pi * fm * t) + 0.15 * np.sin(2 * np.pi * 2 * fm * t)
    x += 0.05 * r.standard_normal(n)
    x += 0.1 * np.sin(2 * np.pi * 0.01 * t)  # slow drift
    return t, x


# ============================================================= PPG
def ppg(duration=10.0, fs=100, hr=72, motion=False, seed=6):
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    rr = 60.0 / hr
    x = np.zeros(n)
    for bt in np.arange(0.3, duration, rr):
        x += _gauss(t, 1.0, bt, 0.06)        # systolic peak
        x += _gauss(t, 0.35, bt + 0.18, 0.05)  # dicrotic notch/wave
    x += 0.02 * r.standard_normal(n)
    ref = None
    if motion:
        ref = np.zeros(n)
        for c in [3.0, 6.5]:
            ref += np.exp(-0.5 * ((t - c) / 0.4) ** 2) * np.sin(2 * np.pi * 2.5 * t)
        x = x + 0.9 * ref
    return t, x, ref


# ============================================================= CTG
def ctg(duration=600.0, fs=4, seed=7):
    """Fetal heart rate (bpm) + uterine contraction (toco) with dropouts."""
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    baseline = 140 + 3 * np.sin(2 * np.pi * t / 300)
    variability = _sig.filtfilt(*_sig.butter(2, 0.05 / (fs / 2)),
                                6 * r.standard_normal(n))
    fhr = baseline + variability
    # uterine contractions
    uc = np.zeros(n)
    for c in [120, 260, 400, 520]:
        uc += 50 * np.exp(-0.5 * ((t - c) / 25) ** 2)
    uc += 5 + 2 * r.standard_normal(n)
    # late decelerations tied to contractions
    for c in [260, 400]:
        fhr -= 20 * np.exp(-0.5 * ((t - c - 12) / 18) ** 2)
    # accelerations
    for c in [60, 180, 480]:
        fhr += 15 * np.exp(-0.5 * ((t - c) / 10) ** 2)
    # signal-loss dropouts (spikes to 0 / wild values)
    fhr_raw = fhr.copy()
    drop_idx = r.integers(0, n, size=40)
    fhr_raw[drop_idx] = 0
    spike_idx = r.integers(0, n, size=25)
    fhr_raw[spike_idx] = fhr_raw[spike_idx] + r.uniform(-60, 60, size=25)
    return t, fhr_raw, fhr, uc


# ============================================================= IMU
def imu(duration=20.0, fs=50, seed=8):
    """Tri-axial accelerometer (g) with gravity + rest/walk/run segments."""
    r = rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    seg = np.zeros(n)  # activity label: 0 rest,1 walk,2 run
    seg[(t >= 6) & (t < 13)] = 1
    seg[t >= 13] = 2
    ax = np.zeros(n); ay = np.zeros(n); az = np.full(n, 1.0)  # gravity on z
    cad = np.where(seg == 2, 3.0, np.where(seg == 1, 1.8, 0.0))  # steps/s
    amp = np.where(seg == 2, 1.1, np.where(seg == 1, 0.5, 0.0))
    ax += amp * np.sin(2 * np.pi * cad * t)
    ay += amp * 0.6 * np.sin(2 * np.pi * cad * t + 1.0)
    az += amp * 0.8 * np.sin(2 * np.pi * 2 * cad * t)
    for a in (ax, ay, az):
        a += 0.03 * r.standard_normal(n)
    return t, np.vstack([ax, ay, az]).T, seg


# ============================================================= SSVEP / ERP
def ssvep_epochs(n_epochs=256, dur=1.0, fs=250, f_stim=12.0, snr=0.15, seed=9):
    """Stimulus-locked epochs: a weak sinusoid buried in ongoing EEG noise."""
    r = rng(seed)
    m = int(dur * fs)
    t = np.arange(m) / fs
    sig = snr * np.sin(2 * np.pi * f_stim * t)
    epochs = sig[None, :] + r.standard_normal((n_epochs, m))
    return t, epochs, sig


def erp_epochs(n_epochs=200, dur=0.8, fs=250, snr=0.4, seed=10):
    r = rng(seed)
    m = int(dur * fs)
    t = np.arange(m) / fs
    # a P300-like positive deflection ~0.3 s
    comp = snr * (np.exp(-0.5 * ((t - 0.30) / 0.05) ** 2)
                  - 0.4 * np.exp(-0.5 * ((t - 0.17) / 0.03) ** 2))
    epochs = comp[None, :] + r.standard_normal((n_epochs, m))
    return t, epochs, comp


# ============================================================= NOISE
def white(n, seed=11):
    return rng(seed).standard_normal(n)


def pink(n, seed=12):
    """1/f noise via FFT filtering."""
    r = rng(seed)
    w = r.standard_normal(n)
    f = np.fft.rfftfreq(n, 1.0)
    f[0] = f[1]
    spec = np.fft.rfft(w) / np.sqrt(f)
    x = np.fft.irfft(spec, n=n)
    return x / np.std(x)


def brown(n, seed=13):
    x = np.cumsum(rng(seed).standard_normal(n))
    return (x - x.mean()) / np.std(x)


def powerline(n, fs, f0=50.0, amp=1.0, harmonics=(1, 2, 3), seed=14):
    t = np.arange(n) / fs
    x = np.zeros(n)
    for h in harmonics:
        x += (amp / h) * np.sin(2 * np.pi * f0 * h * t)
    return x


# ============================================================= IMAGES
def head_phantom(size=256):
    """A simple Shepp-Logan-style head phantom (ellipses of differing intensity)."""
    img = np.zeros((size, size))
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    # (intensity, cx, cy, a, b, angle)
    ell = [
        (1.0, 0, 0, 0.69, 0.92, 0),
        (-0.8, 0, -0.0184, 0.6624, 0.874, 0),
        (-0.2, 0.22, 0, 0.11, 0.31, -18),
        (-0.2, -0.22, 0, 0.16, 0.41, 18),
        (0.1, 0, 0.35, 0.21, 0.25, 0),
        (0.1, 0, 0.1, 0.046, 0.046, 0),
        (0.1, 0, -0.1, 0.046, 0.046, 0),
        (0.1, -0.08, -0.605, 0.046, 0.023, 0),
        (0.1, 0.06, -0.605, 0.023, 0.046, 0),
    ]
    for (val, cx, cy, a, b, ang) in ell:
        th = np.deg2rad(ang)
        xr = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        yr = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1
        img[mask] += val
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img = img / img.max()
    return img


def ct_backproject(image, n_views):
    """Reconstruct `image` from `n_views` equally-spaced CT projections.

    Rotate-and-sum forward projection (a Radon transform) followed by *unfiltered*
    back-projection over angles in [0, pi). Unfiltered (ramp-free) back-projection
    is used deliberately: it makes the streak artifacts of angular undersampling
    plainly visible. Reconstructions are comparable across `n_views` (each is
    normalized by the number of views), so an RMSE against a many-view reference
    is a fair, quantitative measure of adequacy — the Ch2 exploration exercise.
    """
    from scipy.ndimage import rotate

    thetas = np.linspace(0, np.pi, int(n_views), endpoint=False)
    # forward project: each column is the line-integral profile at one angle
    sino = np.array([rotate(image, -np.rad2deg(th), reshape=False, order=1).sum(axis=0)
                     for th in thetas]).T  # (detector, angle)
    n = sino.shape[0]
    recon = np.zeros((n, n))
    for i, th in enumerate(thetas):
        strip = np.tile(sino[:, i], (n, 1))
        recon += rotate(strip, np.rad2deg(th), reshape=False, order=1)
    return recon / len(thetas)


def salt_pepper(img, amount=0.08, seed=15):
    r = rng(seed)
    out = img.copy()
    n = img.size
    k = int(amount * n / 2)
    coords = (r.integers(0, img.shape[0], k), r.integers(0, img.shape[1], k))
    out[coords] = 1.0
    coords = (r.integers(0, img.shape[0], k), r.integers(0, img.shape[1], k))
    out[coords] = 0.0
    return out


def add_gaussian_image(img, sigma=0.08, seed=16):
    return np.clip(img + rng(seed).normal(0, sigma, img.shape), 0, 1)


def test_photo(size=512):
    """A standard grayscale PHOTOGRAPHIC test image in [0,1], square.

    Uses the Grace Hopper photograph bundled with matplotlib (offline, openly
    usable). We deliberately do NOT use the historic "Lena" image: it is
    copyrighted and has been formally retired by the imaging community
    (IEEE, Nature) for licensing and ethical reasons. A photograph — with a
    face, smooth regions, fine texture and sharp edges — makes noise, filtering,
    wavelet and Fourier effects far more visible than a smooth phantom.
    Falls back to a synthetic textured natural-like image if unavailable.
    """
    try:
        import matplotlib.cbook as cbook
        import matplotlib.pyplot as plt
        with cbook.get_sample_data("grace_hopper.jpg") as f:
            rgb = plt.imread(f).astype(float) / 255.0
        # luminosity grayscale
        g = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        h, w = g.shape
        s = min(h, w)
        top = (h - s) // 2
        left = (w - s) // 2
        g = g[top:top + s, left:left + s]
        from PIL import Image
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize(
            (size, size), Image.LANCZOS)).astype(float) / 255.0
        g = (g - g.min()) / (g.max() - g.min() + 1e-9)
        return g
    except Exception:
        # synthetic fallback: gradients + shapes + texture
        r = rng(17)
        yy, xx = np.mgrid[0:size, 0:size] / size
        img = 0.4 + 0.3 * xx + 0.2 * np.sin(2 * np.pi * 3 * yy)
        for (cx, cy, rad, val) in [(0.3, 0.35, 0.18, 0.9), (0.68, 0.6, 0.12, 0.15),
                                   (0.5, 0.8, 0.09, 0.7)]:
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad ** 2
            img[m] = val
        img += 0.05 * r.standard_normal((size, size))
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        return img


# ============================================================= UTIL
def bandpower(x, fs, band, nperseg=None):
    """Integrated power in a frequency band via a Welch PSD.

    `band` is (lo, hi) in Hz. The edges are linearly interpolated onto the PSD
    and included as the first/last integration points, so adjacent bands that
    tile the spectrum (e.g. alpha (8, 11) and sigma (11, 16)) share the exact
    same boundary point rather than each losing the half-bin trapezoid slice
    that straddles it -- restricting the integration to bins strictly inside
    a half-open [lo, hi) range drops that boundary area from *both* neighbours,
    so a set of adjacent bands systematically undercounts the total power.
    """
    f, pxx = _sig.welch(x, fs=fs, nperseg=nperseg or min(len(x), 256))
    lo, hi = band
    lo, hi = max(lo, f[0]), min(hi, f[-1])
    if hi <= lo:
        return 0.0
    inside = (f > lo) & (f < hi)
    f_band = np.concatenate(([lo], f[inside], [hi]))
    p_band = np.concatenate(([np.interp(lo, f, pxx)], pxx[inside], [np.interp(hi, f, pxx)]))
    return _trapezoid(p_band, f_band)


def ar_psd(a, sigma2, fs, nfft=1024, onesided=True):
    """Parametric all-pole AR power spectral density: sigma2 / |A(f)|^2 / fs.

    `a` are the AR coefficients in x[n] = sum_k a_k x[n-k] + e[n]; `sigma2` is the
    innovation variance. `scipy.signal.freqz(1, A)` returns H(f) = 1/A(f), so the
    AR PSD is sigma2 * |H(f)|^2 / fs. Multiplying (not dividing) by |H|^2 is
    essential: dividing yields sigma2 * |A(f)|^2 — the reciprocal — which inverts
    the spectrum so a spectral peak renders as a dip.

    onesided=True (default) doubles the interior positive-frequency bins (not DC or
    Nyquist) so the estimate matches scipy's one-sided periodogram/welch convention:
    integrating it over 0..fs/2 then recovers the full signal variance and its band
    powers are directly comparable to Welch's. Returns (freqs_Hz, psd).
    """
    a_full = np.concatenate(([1.0], -np.asarray(a, float)))
    w, h = _sig.freqz(1.0, a_full, worN=nfft, fs=fs)     # w spans 0..fs/2
    psd = sigma2 * np.abs(h) ** 2 / fs                    # two-sided density
    if onesided and psd.size > 2:
        psd = psd.copy()
        psd[1:-1] *= 2.0                                  # one-sided: double interior bins
    return w, psd
