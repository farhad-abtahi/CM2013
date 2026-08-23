"""bsp — the book's small, tested helper package.

Re-exports the reader-facing API documented in Appendix H (Python Implementation
Map), so that `import bsp; bsp.ar_psd(...)` works directly rather than requiring
the fully-qualified submodule path.
"""
from .biosignals import ar_psd, bandpower
from .metrics import snr_db, sqrtN_gain_db, alias_frequency, quantization_snr_db
from .sleep_pipeline import (
    epoch_features, feature_table, default_classifier, loso_evaluate,
)
from .notebook_checks import (
    assert_no_subject_leak, assert_fold_safe_scaling, expect_close, expected_output_table,
)

__all__ = [
    "ar_psd", "bandpower",
    "snr_db", "sqrtN_gain_db", "alias_frequency", "quantization_snr_db",
    "epoch_features", "feature_table", "default_classifier", "loso_evaluate",
    "assert_no_subject_leak", "assert_fold_safe_scaling", "expect_close", "expected_output_table",
]
