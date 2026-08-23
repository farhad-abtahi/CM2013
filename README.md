# CM2013 — Signal Processing and Data Analytics in Biomedical Engineering

Course code repository for **CM2013** (KTH, HT26), the companion course to
*Biomedical Signal Processing & Data Analytics: From Physiology to Machine
Learning*. Contains the demo notebooks, the graded labs, and the capstone
project scaffold.

**→ [Notebook gallery](https://farhad-abtahi.github.io/CM2013/)** — every demo, lab, and
capstone-track notebook in one searchable, filterable page, each with its Colab/nbviewer/
JupyterLite/download links already wired up.

## Structure

```
notebooks/    Book demo notebooks — reproduce the book's figures & worked examples
              (ch01–ch16, one per chapter) plus one track_*.ipynb per capstone track
labs/         Type-2 lab notebooks — one per chapter (lab01–lab14), the design
              clinic (after Ch11), and the lab16 capstone-planning worksheet.
              Formative — nothing to submit; see the course's Lab_Instructions.
tracks/       Capstone project scaffold: adapter.py (the shared pipeline
              interface) + 6 track adapters, each with a dataset card, and the
              grading rubric / individual-assessment documents
src/bsp/      Shared Python library the notebooks import (loaders, metrics,
              plotting style, sanity checks)
src/figures/  Pre-rendered book figures used by the demo notebooks
tests/        Regression test suite for the track adapters and shared library
```

## Three ways to run every notebook

Every notebook (demo, track, and lab) carries three badges:

| Badge | What it gives you | Best for |
|---|---|---|
| **Open in Colab** | A real, runnable Jupyter environment in the cloud (free Google account) | Actually running/editing the code |
| **nbviewer** | A fast, static rendered view with all outputs already shown | Reading without running anything |
| **JupyterLite** | An in-browser Jupyter environment — no install, no account, runs entirely client-side (Pyodide) | Quick interactive runs without leaving the browser |

You can also clone the repo and run notebooks locally — see **Local setup** below.

## Local setup

```bash
git clone https://github.com/farhad-abtahi/CM2013.git
cd CM2013
pip install -r requirements.txt
# Only needed for the capstone tracks' real-data loaders (mne, wfdb):
pip install -r requirements-real.txt
```

For an exact, reproducible environment matching what these notebooks were
validated against, use `requirements-lock.txt` instead of `requirements.txt`.

## Capstone datasets — you download them, not us

The capstone tracks (`tracks/`) never bundle real data in this repo — `git clone`
gets you code only. Each track's adapter downloads its dataset directly from the
original source **the first time you run it**, from the links below. You're
downloading directly from the source and are bound by its own license, not a
copy we redistribute.

| Track | Dataset | License | Source |
|---|---|---|---|
| `sleep_edf` | Sleep-EDF Database (PhysioNet) | ODC-BY (open, attribution) | https://physionet.org/content/sleep-edfx/1.0.0/ |
| `ecg_cinc2017` | PhysioNet/CinC Challenge 2017 (AF) | Open (PhysioNet Challenge) | https://physionet.org/content/challenge-2017/1.0.0/ |
| `ctg_ctu_uhb` | CTU-UHB Intrapartum Cardiotocography (PhysioNet) | ODC-BY (open, attribution) | https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/ |
| `emg_ninapro` | Ninapro DB1 (surface EMG) | **CC BY-NC-ND** — non-commercial, no derivatives; cite Atzori et al. (2014) | https://ninapro.hevs.ch/instructions/DB1.html |
| `har` | UCI Human Activity Recognition | Open (UCI ML Repository) | https://archive.ics.uci.edu/dataset/341/ |
| `bci_eegmmidb` | EEG Motor Movement/Imagery DB (PhysioNet) | ODC-BY (open, attribution) | https://physionet.org/content/eegmmidb/1.0.0/ |

All six are **open-direct downloads — no signed data-use agreement required**.
Ninapro's CC BY-NC-ND terms are the one that actually restrict what you can do
with the *data* (non-commercial, no redistribution of modified data) — this
doesn't affect the adapter code, which is MIT-licensed.

## License

- **Code** (`tracks/*.py`, `src/bsp/*.py`, `tests/*.py`, CI scripts) — MIT, see `LICENSE`.
- **Notebooks and written content** (`notebooks/`, `labs/`, the Markdown docs
  under `tracks/`) — CC BY-NC-SA 4.0, see `LICENSE-CONTENT.md`.

## The book

This repo is the code companion to *Biomedical Signal Processing & Data
Analytics: From Physiology to Machine Learning*.
