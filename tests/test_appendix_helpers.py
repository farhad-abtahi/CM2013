"""Build-time checks for the Appendix H reader-facing API and tested helpers.
Guards the bug classes fixed in the 2026-07-17 review (an inverted AR spectrum,
a double-counted band edge) plus the Appendix H API/normalization claims."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_bsp_public_api_is_importable():
    # Appendix H documents `bsp.ar_psd`, `bsp.bandpower`, ... — they must exist
    # after a bare `import bsp` (re-exported from bsp/__init__.py).
    import bsp
    for name in ("ar_psd", "bandpower", "snr_db", "epoch_features",
                 "default_classifier", "loso_evaluate",
                 "assert_no_subject_leak", "assert_fold_safe_scaling", "expect_close"):
        assert hasattr(bsp, name), f"bsp.{name} is not importable"


def test_ar_psd_is_a_peak_not_a_dip():
    import bsp
    fs = 100.0
    r, th = 0.95, 2 * np.pi * 10 / fs
    a = [2 * r * np.cos(th), -r ** 2]
    w, psd = bsp.ar_psd(a, 1.0, fs, nfft=2048)
    assert abs(w[np.argmax(psd)] - 10.0) < 0.5
    assert psd.max() / psd.min() > 100


def test_ar_psd_onesided_integrates_to_variance():
    import bsp
    from scipy.integrate import trapezoid
    fs, r = 100.0, 0.9
    th = 2 * np.pi * 12 / fs
    a = [2 * r * np.cos(th), -r ** 2]
    rng = np.random.default_rng(0)
    n = 200000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for k in range(2, n):
        x[k] = a[0] * x[k - 1] + a[1] * x[k - 2] + e[k]
    var = x.var()
    w, psd = bsp.ar_psd(a, 1.0, fs, nfft=8192, onesided=True)
    integ = trapezoid(psd, w)
    assert abs(integ - var) / var < 0.05, f"one-sided AR integral {integ:.2f} vs var {var:.2f}"


def test_bandpower_edge_is_half_open():
    import bsp
    fs = 200.0
    t = np.arange(int(fs * 8)) / fs
    x = np.sin(2 * np.pi * 11.0 * t)
    alpha = bsp.bandpower(x, fs, (8, 11))
    sigma = bsp.bandpower(x, fs, (11, 16))
    whole = bsp.bandpower(x, fs, (8, 16))
    assert alpha + sigma <= whole * 1.05


if __name__ == "__main__":
    test_bsp_public_api_is_importable()
    test_ar_psd_is_a_peak_not_a_dip()
    test_ar_psd_onesided_integrates_to_variance()
    test_bandpower_edge_is_half_open()
    print("Appendix H API + helper tests PASS")
