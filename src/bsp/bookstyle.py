"""
bookstyle.py — shared plotting style for the Biomedical Signal Processing &
Data Analytics companion notebooks. Everything imports this so the whole book
and its notebooks share one uniform, colour-blind-safe visual grammar.

Usage
-----
    import bookstyle as bs
    fig, ax = bs.newfig()
    ax.plot(t, x, color=bs.C['blue'])
"""
import os
import matplotlib
# Use headless Agg only when NOT inside IPython/Jupyter, so notebooks render
# figures inline while the figure-generation scripts still work on a display-less
# server (e.g. CI).
try:
    get_ipython  # type: ignore  # noqa: F821  (defined only inside IPython)
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- palette
# Okabe-Ito colour-blind-safe palette + a few neutrals.
C = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#222222",
    "grey":   "#7F7F7F",
    "lightgrey": "#CCCCCC",
}
CYCLE = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"]]

# ---------------------------------------------------------------- rcParams
def apply():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "lines.linewidth": 1.4,
        "figure.autolayout": False,
    })
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=CYCLE)

apply()

# ---------------------------------------------------------------- helpers
def newfig(w=7.0, h=4.0, nrows=1, ncols=1, **kw):
    """Return (fig, ax/axes) at a standard book width (~7in fits the text block)."""
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return fig, ax

def _figdir(chapter):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "figures", f"ch{int(chapter):02d}")
    base = os.path.normpath(base)
    os.makedirs(base, exist_ok=True)
    return base

def save(fig, chapter, name, tight=True):
    """Save fig to figures/chNN/figNN_<name>.png and close it. Returns the path."""
    d = _figdir(chapter)
    if not name.startswith("fig"):
        name = f"fig{int(chapter):02d}_{name}"
    path = os.path.join(d, name + ".png")
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def panel_label(ax, s, dx=-0.06, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight="bold",
            fontsize=11, va="top", ha="right")

def takeaway(fig, text, y=-0.02):
    """A one-line caption-style takeaway across the bottom of a figure."""
    fig.text(0.5, y, text, ha="center", va="top", fontsize=8.5,
             style="italic", color="#555555", wrap=True)
