"""
tracks.report — module 7 of the seven-stage pipeline: **reporting**.

Chapter 16 §16.8 is blunt about the order: *lead with the confusion matrix and the
primary metric, never bare accuracy*, then read the errors, then (for staging-style
tracks) show the whole-night picture. This module supplies that panel so nobody
spends their last week re-inventing a step plot — the *reading* of it is still
yours, and is what the rubric grades.

    from report import summarize_results, plot_hypnogram

    rep = track.evaluate(X, y, groups)      # or: track.run_smoke()
    track.report(rep)                       # -> summarize_report(rep) below
    plot_hypnogram(y_true_night, y_pred_night)

Everything here is a **light template, not a house style**. A confusion matrix
normalised per true class, a spread quoted as mean/range, a hypnogram drawn as a
step plot: these are common, defensible conventions, not the only ones. Swap the
plot, add per-class recall, quote an IQR instead of a range — as long as the
report leads with the diagnostic and states the split unit with every number.
"""
from __future__ import annotations

import numpy as np

from adapter import metrics_for, metric_spread, spread_line, per_group_metrics  # noqa: F401
from bsp import metrics as M

#: A conventional AASM ordering for sleep hypnograms (deepest at the bottom).
#: Any ordered sequence of class labels works — pass your own `stage_order`.
AASM_ORDER = ["W", "REM", "N1", "N2", "N3"]

_PANEL = ("cohens_kappa", "macro_f1", "balanced_accuracy", "accuracy")


# ------------------------------------------------------------------ the panel
def confusion_table(y_true, y_pred, labels=None, normalize=True) -> str:
    """The confusion matrix as a markdown table — rows = true, columns = predicted.
    Text, so it survives a terminal, a log file and a report draft alike."""
    labs, cm = M.confusion(list(y_true), list(y_pred), labels)
    head = "| true \\ pred | " + " | ".join(map(str, labs)) + " | n |"
    rule = "|---" * (len(labs) + 2) + "|"
    rows = [head, rule]
    for i, l in enumerate(labs):
        n = cm[i].sum()
        if normalize and n:
            cells = [f"{v / n:.2f}" for v in cm[i]]
        else:
            cells = [str(int(v)) for v in cm[i]]
        rows.append(f"| **{l}** | " + " | ".join(cells) + f" | {int(n)} |")
    return "\n".join(rows)


def summarize_results(per_group_results, y_true_all, y_pred_all, labels=None,
                      primary_metric="cohens_kappa", group_unit="group",
                      title=None, show=True) -> dict:
    """The §16.8 panel for ANY track: confusion matrix first, then the primary
    metric **with its spread across groups**, then macro-F1 and balanced accuracy.

    Parameters
    ----------
    per_group_results : list of dict
        One dict per held-out group/fold, as produced by `evaluate()`'s
        `per_group` / `per_fold` (keys: the metric names + `n`). Pass `[]` if you
        genuinely have only a pooled result — but then say so in the report,
        because a number without a spread is half a result (§16.3).
    y_true_all, y_pred_all : sequences
        The pooled predictions, for the confusion matrix and the pooled metrics.
    labels : list, optional
        Class order for the confusion matrix (defaults to sorted observed labels).
    primary_metric : str
        The track's headline metric — `TrackMeta.default_metrics[0]`.
    group_unit : str
        What one row of `per_group_results` is ("subject", "record", "fold") —
        printed with the spread so the reader knows what varied.

    Returns
    -------
    dict with `pooled`, `spread` (per metric), `spread_unit`, `confusion_md`,
    `text` (the whole panel as a string, ready to paste into a draft).
    """
    pooled = metrics_for(y_true_all, y_pred_all, labels)
    spreads = {m: metric_spread(per_group_results or [], m) for m in _PANEL}
    conf_md = confusion_table(y_true_all, y_pred_all, labels)

    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += ["**Confusion matrix** (rows = true, columns = predicted, row-normalised)",
              "", conf_md, ""]
    lines.append(f"**{primary_metric}** — "
                 + spread_line(primary_metric, spreads.get(primary_metric, {}), group_unit)
                 + f"; pooled {pooled.get(primary_metric, float('nan')):.3f}")
    for m in _PANEL:
        if m == primary_metric:
            continue
        sp = spreads.get(m, {})
        mean = sp.get("mean", float("nan"))
        extra = f", mean over {group_unit}s {mean:.3f}" if np.isfinite(mean) else ""
        lines.append(f"- {m}: pooled {pooled.get(m, float('nan')):.3f}{extra}")
    if per_group_results:
        worst = min((r for r in per_group_results
                     if np.isfinite(r.get(primary_metric, float("nan")))),
                    key=lambda r: r[primary_metric], default=None)
        if worst is not None:
            key = "group" if "group" in worst else "fold"
            lines.append(f"- worst {group_unit}: {worst[key]} "
                         f"({primary_metric} {worst[primary_metric]:.3f}, n={worst['n']}) "
                         "— read this one's errors before the mean's")
    lines += ["", "*Now read it:* which classes trade places in the matrix, is the spread "
                  "driven by one hard group, and what would you change next?"]
    text = "\n".join(lines)
    if show:
        print(text)
    return {"pooled": pooled, "spread": spreads, "spread_unit": group_unit,
            "primary_metric": primary_metric, "confusion_md": conf_md, "text": text}


def summarize_report(rep: dict, show: bool = True, title=None, **kw) -> dict:
    """`summarize_results` fed straight from a `TrackAdapter.evaluate()` dict."""
    return summarize_results(
        rep.get("spread_rows") or rep.get("per_group") or [],
        rep["y_true"], rep["y_pred"],
        labels=rep.get("labels"),
        primary_metric=kw.pop("primary_metric", rep.get("primary_metric", "cohens_kappa")),
        group_unit=kw.pop("group_unit", rep.get("spread_unit", rep.get("split_unit", "group"))),
        title=title or f"Results — split unit: {rep.get('split_unit', '?')} "
                       f"(n={rep.get('n_groups', '?')})",
        show=show)


def plot_confusion(y_true, y_pred, labels=None, normalize="true", ax=None, title=None):
    """The confusion matrix as a figure (the thing §16.8 says to put first)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    labs = labels if labels is not None else sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labs, normalize=normalize,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title or "Confusion matrix")
    return ax.figure


# ------------------------------------------------------- staging-track views
def plot_hypnogram(y_true, y_pred=None, epoch_seconds=30, stage_order=None,
                   ax=None, title=None, mark_disagreements=True, labels=("reference", "predicted")):
    """A whole-recording **stage sequence** as a step plot — the hypnogram of
    §16.5/Figure 16.2(a) for the sleep track, and equally the "what did the model
    say, epoch by epoch" view for any track whose labels form a time sequence
    (staging, activity over a session, rhythm over a long strip).

    Tracks whose unit of prediction is a whole record (one label per ECG strip,
    one label per CTG trace) have no sequence to draw — for those, the confusion
    matrix and per-record spread *are* the picture.

    Parameters
    ----------
    y_true : sequence of labels, in time order (one per epoch).
    y_pred : sequence of labels, optional — overlaid for comparison.
    epoch_seconds : float — epoch length, used for the time axis (30 s for AASM).
    stage_order : list, optional — the vertical order, top to bottom. Defaults to
        the AASM convention `["W", "REM", "N1", "N2", "N3"]` when the labels look
        like sleep stages, otherwise the sorted unique labels.
    mark_disagreements : bool — tick the epochs where the two disagree; the errors
        cluster at transitions, which is exactly the thing worth seeing.

    Returns the matplotlib Figure. Restyle it freely — this is a starting point,
    not a required figure format.
    """
    import matplotlib.pyplot as plt
    yt = [str(v) for v in np.asarray(y_true).tolist()]
    yp = None if y_pred is None else [str(v) for v in np.asarray(y_pred).tolist()]
    if stage_order is None:
        seen = set(yt) | (set(yp) if yp else set())
        stage_order = AASM_ORDER if seen <= set(AASM_ORDER) else sorted(seen)
    pos = {s: len(stage_order) - 1 - i for i, s in enumerate(stage_order)}   # first = top
    hrs = np.arange(len(yt)) * epoch_seconds / 3600.0

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.0))
    ax.step(hrs, [pos.get(s, np.nan) for s in yt], where="post", color="k", lw=1.2,
            label=labels[0])
    if yp is not None:
        n = min(len(yt), len(yp))
        ax.step(hrs[:n], [pos.get(s, np.nan) for s in yp[:n]], where="post",
                color="C1", lw=1.0, alpha=0.85, label=labels[1])
        if mark_disagreements:
            bad = [i for i in range(n) if yt[i] != yp[i]]
            if bad:
                ax.plot(hrs[bad], np.full(len(bad), len(stage_order) - 0.4), "|",
                        color="C3", ms=5, label=f"disagreement ({len(bad)}/{n})")
        ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_yticks(range(len(stage_order)))
    ax.set_yticklabels(list(reversed(stage_order)))
    ax.set_ylim(-0.6, len(stage_order) - 0.1)
    ax.set_xlabel(f"time (hours, {epoch_seconds:g}-s epochs)")
    ax.set_title(title or "Hypnogram — predicted vs. reference")
    ax.figure.tight_layout()
    return ax.figure


def stage_summary(y, epoch_seconds=30, wake_label="W", rem_label="REM",
                  sleep_labels=None) -> dict:
    """Clinical-style summaries of one staged recording (§16.5): total sleep time,
    sleep efficiency, wake after sleep onset, sleep-onset latency, REM latency.

    **One reasonable convention, stated explicitly** — time in bed = the whole
    scored sequence, sleep onset = the first non-wake epoch, WASO = wake epochs
    after that. Clinical definitions vary (some require N epochs of sustained
    sleep for onset); if you use a different one, say which in the report. The
    point of the exercise is §16.5's warning: a decent per-epoch kappa can still
    mis-estimate the numbers a clinician actually reads.

    Returns minutes (and a fraction for efficiency); NaN where undefined.
    """
    y = [str(v) for v in np.asarray(y).tolist()]
    n = len(y)
    per_min = epoch_seconds / 60.0
    sleep = set(sleep_labels) if sleep_labels else {s for s in set(y) if s != wake_label}
    idx_sleep = [i for i, s in enumerate(y) if s in sleep]
    nan = float("nan")
    if n == 0:
        return {k: nan for k in ("time_in_bed_min", "total_sleep_min", "sleep_efficiency",
                                 "waso_min", "sleep_onset_latency_min", "rem_latency_min")}
    tib = n * per_min
    tst = len(idx_sleep) * per_min
    onset = idx_sleep[0] if idx_sleep else None
    waso = (sum(1 for s in y[onset:] if s == wake_label) * per_min) if onset is not None else nan
    rem_i = next((i for i, s in enumerate(y) if s == rem_label), None)
    return {
        "time_in_bed_min": round(tib, 1),
        "total_sleep_min": round(tst, 1),
        "sleep_efficiency": round(tst / tib, 3) if tib else nan,
        "waso_min": round(waso, 1) if waso == waso else nan,
        "sleep_onset_latency_min": round(onset * per_min, 1) if onset is not None else nan,
        "rem_latency_min": (round((rem_i - onset) * per_min, 1)
                            if (rem_i is not None and onset is not None) else nan),
    }


def compare_stage_summaries(y_true, y_pred, epoch_seconds=30, show=True, **kw) -> dict:
    """Reference vs. predicted clinical summaries for one recording, side by side
    with the difference — the night-level check of §16.5 ("a scorer can post a
    decent per-epoch kappa yet still mis-estimate what a physician reads").

    With several recordings, collect the `difference` column across nights and a
    Bland-Altman plot (difference against mean) is one further step; whether that
    is worth your remaining time is a judgement call, not a requirement.
    """
    ref = stage_summary(y_true, epoch_seconds, **kw)
    pred = stage_summary(y_pred, epoch_seconds, **kw)
    rows = [(k, ref[k], pred[k], (pred[k] - ref[k]) if
             (isinstance(ref[k], float) and ref[k] == ref[k] and pred[k] == pred[k]) else float("nan"))
            for k in ref]
    md = ["| summary | reference | predicted | difference |", "|---|---|---|---|"]
    md += [f"| {k} | {a} | {b} | {d:+.2f} |" if d == d else f"| {k} | {a} | {b} | — |"
           for k, a, b, d in rows]
    text = "\n".join(md)
    if show:
        print(text)
    return {"reference": ref, "predicted": pred, "rows": rows, "markdown": text}


__all__ = ["summarize_results", "summarize_report", "confusion_table", "plot_confusion",
           "plot_hypnogram", "stage_summary", "compare_stage_summaries", "AASM_ORDER"]
