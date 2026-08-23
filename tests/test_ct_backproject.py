"""Guards the CT reconstruction helper behind the Ch2 exploration exercise.

The exercise asks students to reconstruct the head phantom at increasing view
counts and judge adequacy with a *quantitative* RMSE against a many-view
reference (not a hand-wave about "clinically negligible" streaks). For that to
be a real experiment, more views must actually reduce the RMSE.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_more_views_reduce_rmse_vs_reference():
    import bsp.biosignals as bio

    img = bio.head_phantom(size=96)
    ref = bio.ct_backproject(img, 256)
    e8 = _rmse(bio.ct_backproject(img, 8), ref)
    e32 = _rmse(bio.ct_backproject(img, 32), ref)
    e128 = _rmse(bio.ct_backproject(img, 128), ref)
    # angular undersampling: fewer views -> larger error vs the reference
    assert e8 > e32 > e128, f"RMSE should fall with more views, got {e8}, {e32}, {e128}"
    # 128 views should already be close to the 256-view reference
    assert e128 < 0.05


def test_reconstruction_shape_matches_image():
    import bsp.biosignals as bio

    img = bio.head_phantom(size=64)
    recon = bio.ct_backproject(img, 32)
    assert recon.shape == img.shape
