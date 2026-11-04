# Python for biosignal processing — a choose-your-own-path tutorial

Nine short modules, from "never used Python" to "reproducible machine-learning pipeline". You don't need all
of them. Modules 1–8 open with a **self-check**: if you can already do everything on its list, skip it. Module 9 is
practice: broken code for you to debug.

Everything uses synthetic signals, so no data downloads are needed. Run it in **Google Colab** or on your own
machine (see *Setup first* below).

## Setup first

**In Google Colab** (easiest): click the *Open in Colab* badge at the top of a module. Colab already has
NumPy, SciPy, scikit-learn, matplotlib and pandas; Modules 3 and 4 install the extra libraries they need
(`wfdb`, `mne`, PyWavelets) in their first cell if Colab lacks them. You need an internet connection for Colab itself.

**On your own computer**, once, from the root of the course repository (the folder with
`requirements-lock.txt`), with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt -r requirements-real.txt
```

You need **both** files: Module 3 reads and writes ECG and EEG formats with `wfdb`, `mne` and `pandas`, which
are in `requirements-real.txt`. Then open a module in VS Code or Jupyter and **select the `.venv` environment
as the kernel**. After installation everything runs offline. Module 6 explains environments in detail, and
what to do when a library breaks.

The optional part of Module 7 also uses `mlflow` and `wandb`; the notebook works without them.

## Where should I start?

| If you… | Start with | Then |
|---|---|---|
| have never programmed in Python | Module 1 | 2 → 3 → 4 → 5 |
| know Python, but not NumPy or matplotlib | Module 2 | 3 → 4 → 5 |
| know NumPy, but are new to signal processing in Python | Module 4 | 3, 5 |
| know signal processing, but are new to machine learning on biosignals | Module 5 | 6, 7 |
| are starting the capstone (everyone) | Module 6 | 7, 8 |
| have a problem and want the fix | Module 8 | — |
| want to practise finding bugs yourself | Module 9 | — |

**Quick self-test.** Can you, without looking anything up…

1. write a function that returns the mean of a list, with a loop? *If not → Module 1.*
2. build a time axis for 10 s at 250 Hz and select the samples between 2 s and 3 s without a loop? *If not → Module 2.*
3. save a signal together with its sampling rate and channel names, and load it back? *If not → Module 3.*
4. design a 0.5–40 Hz band-pass filter and apply it with zero phase? *If not → Module 4.*
5. explain why a random train/test split overestimates performance on a new patient? *If not → Module 5.*
6. make sure a teammate gets exactly your numbers on their laptop? *If not → Module 6.*
7. re-run a model without recomputing all features, and save a figure that is readable in print? *If not → Module 7.*

## The modules

| # | Module | Covers | Course link |
|---|---|---|---|
| 1 | [Python essentials](01_python_essentials.ipynb) | notebooks, variables, lists, dicts, loops, functions, imports, reading errors | needed for everything |
| 2 | [NumPy arrays and plotting signals](02_numpy_and_plotting.ipynb) | arrays, the time axis, slicing, masks, multi-channel data, epochs, vectorisation, matplotlib | Ch 1–2, Labs 1–2 |
| 3 | [Storing signals and time series](03_storing_signals.ipynb) | metadata, `.npz`, pandas, CSV, WFDB, EDF/FIF with MNE, `X` / `y` / `groups` | Lab 1, track `load()` |
| 4 | [Signal processing libraries](04_signal_processing_libraries.ipynb) | which library for what, FFT, Welch, spectrogram, SOS filters, zero phase, resampling, peaks, Hilbert, wavelets | Ch 3–10, Labs 3–10 |
| 5 | [Machine learning for biosignals](05_machine_learning.ipynb) | features, Pipelines, `GroupKFold`, leakage, metrics for imbalance, spread across folds | Ch 11–14, Labs 11–14 |
| 6 | [Best practices for a reproducible project](06_best_practices.ipynb) | virtual environments and fixing library versions, config and seeds, functions, sanity checks, units, git, debugging | capstone |
| 7 | [Saving results, figures and caching](07_results_figures_caching.ipynb) | one folder per run, figures for reports (size, fonts, resolution, colour), caching features; optional: MLflow and W&B | capstone, `RESULTS.md` |
| 8 | [Troubleshooting and common mistakes](08_troubleshooting.ipynb) | environment problems, notebook pitfalls, signal-processing and ML mistakes with demonstrations | everything |
| 9 | [Debugging challenge](09_debugging_challenge.ipynb) | ten broken cells, one bug each, with hints and checks for each part of the goal; [worked troubleshooting steps](09_debugging_challenge_solutions.ipynb) in a separate notebook | everything |

**Solutions** to the *Try it yourself* exercises are in a separate notebook, [solutions.ipynb](solutions.ipynb), so you
are not tempted before you have tried. Each solution continues its module.

Module 7 ends with an **optional, advanced** part on experiment tracking (MLflow, Weights & Biases). It is not
needed for the course, but worth knowing for a thesis or a job.

## After the tutorial

- **Demo notebooks** (`../notebooks/chNN_*.ipynb`) reproduce the book's figures — read and run them.
- **Lab notebooks** (`../labs/`) are where you make the decisions yourself.
- **Capstone** starts at [`../tracks/START_HERE.md`](../tracks/START_HERE.md).
