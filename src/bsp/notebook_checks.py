"""
bsp.notebook_checks — tiny guards that make the book's discipline executable.
Used across the ML/pipeline notebooks so leakage and reproducibility are
*asserted*, not merely described.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------- leakage guards
def assert_no_subject_leak(train_groups, test_groups):
    """Fail loudly if any subject/recording appears in both train and test."""
    overlap = set(np.asarray(train_groups).tolist()) & set(np.asarray(test_groups).tolist())
    assert not overlap, f"Subject leakage detected: {sorted(overlap)} in both splits."
    return True


def assert_fold_safe_scaling(scaler_fitted_on_train_only: bool):
    assert scaler_fitted_on_train_only, (
        "Scaling leakage: the scaler must be fit on the training fold only "
        "(use an sklearn Pipeline so it is refit inside every fold).")
    return True


# --------------------------------------------------------- reproducibility
def expect_close(name, value, expected, tol=1e-6, kind="abs"):
    """Print a pass/fail row for an expected numeric output and return a dict.

    kind: 'abs' | 'rel' | 'sign' | 'range'. For 'range', pass expected=(lo, hi)
    and the value must fall within [lo, hi] — a genuine assertion (not just >chance)."""
    value = float(value)
    if kind == "abs":
        ok = abs(value - expected) <= tol; exp_str = f"{expected:.4g}"
    elif kind == "rel":
        ok = abs(value - expected) <= tol * abs(expected); exp_str = f"{expected:.4g}"
    elif kind == "sign":            # only the direction/sign matters
        ok = np.sign(value) == np.sign(expected); exp_str = f"sign {int(np.sign(expected))}"
    elif kind == "range":           # value must fall inside [lo, hi]
        lo, hi = expected
        ok = lo <= value <= hi; exp_str = f"[{lo:.4g}, {hi:.4g}]"
    else:
        raise ValueError(kind)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: got {value:.4g}, expected {exp_str} ({kind})")
    return {"name": name, "value": value, "expected": exp_str, "ok": bool(ok)}


def expected_output_table(rows):
    """rows: list of dicts from expect_close(). Returns a small markdown table string."""
    head = "| Quantity | Got | Expected | Status |\n|---|---:|---:|:--:|\n"
    body = "\n".join(
        f"| {r['name']} | {r['value']:.4g} | {r['expected']} | "
        f"{'✅' if r['ok'] else '❌'} |" for r in rows)
    return head + body
