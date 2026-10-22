# Start here — capstone tracks (students)

You will take **one real biomedical signal** through the **whole pipeline** you learned in the course,
and present it beside other teams' signals. Everyone uses the same methods; the payoff is seeing how
they behave differently on EEG vs ECG vs IMU vs EMG vs CTG.

## The learning path (5 steps)

```
  1. PICK a track ─▶ 2. REFRESH + LIT REVIEW ─▶ 3. RUN the baseline ─▶ 4. IMPROVE it ─▶ 5. REPORT + SUBMIT
     (card +           (BACKGROUND_MAP.md:          (notebook:            (honestly,        (predictions.csv
      instructions)     book §§ + guiding Qs)        synthetic → real)     no leakage)       + showcase)
```

| Step | What you do | Where |
|------|-------------|-------|
| **1. Pick a track** | Read its **dataset card** (task, signals, split unit, known pitfalls, measured baseline + yardstick) and **instructions**. | `<track>_card.md`, `<track>_instructions.md` |
| **2. Refresh + literature review** | Skim the **book sections** that cover each method (a refresher — you already learned them), then do a short (~5–8 source) **literature review** of the *application* to **motivate** your design. This is course outcome **L5** and is graded. | **`BACKGROUND_MAP.md`** |
| **3. Run the baseline** | Open the track's notebook. It runs on **synthetic data offline** by default; set `USE_REAL = True` in **Colab or on your own machine** (see *Running locally* below) for the real dataset. Read the **honest metric panel** (κ / macro-F1 / confusion), not accuracy. | `notebooks/track_<name>.ipynb` |
| **4. Improve the baseline** | Do **real DSP** — better preprocessing and **feature construction** (the point of the course), then a better model. Validate **inside the folds**, keep the **declared split unit**. Log each iteration as you go. | **`DESIGN_MENUS.md`** (the options at each stage and what each costs); the adapter's `preprocess()` and `extract_features()` are where you work; `results_log_TEMPLATE.md` is where you log it |
| **5. Report + submit** | Write up your design (justified, read against the **yardstick**), submit `predictions.csv` for hold-out evaluation, and take a slot in the **cross-track showcase**. | `HOLDOUT_EVALUATION.md`, `CAPSTONE_REPORT_RUBRIC.md`, `report.py` |

## Running locally, and getting the same numbers every time

Colab and your own machine both work, so use whichever suits your team. For a local run you need
Python 3.10 or newer (the course CI uses 3.11):

```bash
git clone https://github.com/farhad-abtahi/CM2013.git
cd CM2013
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt -r requirements-real.txt
```

Any virtual-environment tool is fine. The three common ones:

- **`venv`** (comes with Python): `python -m venv .venv`, then activate it (`source .venv/bin/activate`, or
  `.venv\Scripts\Activate.ps1` in Windows PowerShell).
- **conda:** `conda create -n cm2013 python=3.11`, then `conda activate cm2013`.
- **`uv`** (optional, faster): `uv venv --python 3.11`, then `uv pip install -r ...`.

In VS Code, pick that environment with *Python: Select Interpreter* and as the notebook kernel. Check it
with `python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"`;
you should see `2.2.6 1.15.3 1.7.2`. If it breaks, delete the environment and recreate it from the lock
file. Never commit `.venv`.

If your team keeps its own repository, copy in `tracks/`, `src/` and both requirements files.
Real data downloads into `data_cache/` the first time you run with `USE_REAL = True` and is
ignored by git, so **do not commit it**. Rough sizes of what the scaffold fetches: ECG ~15 MB
(about a minute), EMG ~120 MB, Sleep-EDF ~400 MB.

To get the same numbers on every run and every teammate's machine:

1. **Set `cfg["seed"]`** and make sure it reaches every random step: the classifier, any splitting,
   resampling and feature selection.
2. **Use the same environment.** Install from `requirements-lock.txt` on every machine and write
   the Python version in your README. Different library versions are the usual cause of
   differences between machines.
3. **Expect tiny differences.** Floating-point results can differ in the last digits across
   machines. A difference smaller than the fold-to-fold spread in `rep["summary"]` is noise, not
   an improvement or a regression.
4. **Record what produced each number** (seed, code revision, environment) in `RESULTS.md`.

Your team repository has no prescribed layout. A grader should be able to clone it, install from
your lock file, and reproduce your reported numbers from the commands in your README. It needs
`RESULTS.md`, your pinned environment, and that README. You submit the report PDF, a
`predictions.csv` written with `write_submission()` (never by hand), and the repository link.

## Download once, parse once (Colab and local)

Real data is slow to download and slow to parse, so do each **once** and keep the result.

1. **One persistent data folder, outside your repository.** Locally, e.g. `~/cm2013_data`. In Colab
   the disk is wiped when the session ends, so put the data on Google Drive:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   DATA = "/content/drive/MyDrive/CM2013_data"      # survives between sessions
   ```
2. **Download into it.** Every track's `download()` skips files that are already there, so re-running
   the cell costs nothing. Sleep-EDF now downloads from PhysioNet's official AWS mirror, which is much
   faster than physionet.org; mne checks every file against PhysioNet's checksums either way.
   ```python
   trk.download(f"{DATA}/train", subset=range(0, 30))
   ```
3. **Parse once with `load_cached()`.** It runs `load()` the first time and saves the parsed recordings
   as a binary cache next to the data; later runs load them in seconds. The cache is rebuilt
   automatically when the files or the `load()` settings change.
4. **Cache every slow step, with a version number.** Preprocessing and feature extraction are worth
   caching too, so that changing the classifier doesn't recompute everything. `cached()` stores a
   step's result under a key; put in the key everything the result depends on: a **version number
   for that step's code** (bump it whenever you change the code), its settings, and the key of the
   step before it. Changing preprocessing then rebuilds the features as well; changing only the
   features reuses the preprocessing.
   ```python
   import numpy as np
   from adapter import cached, data_fingerprint

   PREPROCESS_VERSION = 1      # bump when you change your preprocessing code
   FEATURE_VERSION = 1         # bump when you change your feature code
   CACHE = f"{DATA}/cache"

   recs = trk.load_cached(f"{DATA}/train")
   pre_key = {"step": "preprocess", "version": PREPROCESS_VERSION, "cfg": cfg,
              "data": data_fingerprint(f"{DATA}/train")}
   clean = cached("preprocess", pre_key, lambda: [trk.preprocess(r, cfg) for r in recs], CACHE)
   feat_key = {"step": "features", "version": FEATURE_VERSION, "cfg": cfg, "after": pre_key}
   per_rec = cached("features", feat_key, lambda: [trk.extract_features(r, cfg) for r in clean], CACHE)
   X = np.vstack([x for x, _, _ in per_rec])
   y = np.concatenate([lab for _, lab, _ in per_rec])
   groups = np.concatenate([[g] * len(lab) for _, lab, g in per_rec])
   ```
   Cache only steps that work on one recording at a time. Anything **fitted** across recordings or on
   labels (scaling, feature selection, CSP, the model) belongs inside the cross-validation folds and
   is not cached this way. If in doubt, delete the cache folder: correctness first, speed second.

Keep data and cache files **out of git**. In Colab, Drive is slow with many small files (e.g. ECG
records): if loading feels slow, copy the folder to the local disk once per session
(`!cp -r "$DATA/train" /content/train`) and load from there.

## The seven modules (this is what "pipeline integrity" means)

The adapter is not one `run_everything()` — it is the book's §16.2 pipeline with one method per
stage, so any stage can be swapped, tested, or rewritten without touching the others:

```
 download/load/smoke ─▶ preprocess ─▶ extract_features ─▶ select_features ─▶ baseline (classify)
                                                                                     │
                                      report  ◀── evaluate (folds)                   │
                                        ▲                                            ▼
                                        └──────────────── infer (frozen, no refit) ──┘
```

| # | Module | Method on the adapter | Note |
|---|--------|----------------------|------|
| 1 | data loading | `download()` / `load()` / `smoke()` | supplied per track |
| 2 | preprocessing | `preprocess(rec, cfg)` | identity by default — **your** filtering/artifact decisions go here |
| 3 | feature extraction | `extract_features(rec, cfg)` | the DSP that earns most of the marks |
| 4 | feature selection | `select_features(X, y, cfg)` | fit **inside** every fold; pass-through, ANOVA, mutual-info and tree-importance options ship with it |
| 5 | classification | `baseline()` or your own estimator | |
| 6 | inference | `infer(model, rec, cfg)` | **frozen** — applies a fitted pipeline, fits nothing |
| 7 | reporting | `report(rep)` → `report.py` | confusion matrix first, metric **with its spread**, hypnogram for staging tracks |

Defaults are *starting points, not recommendations*: pass-through selection and identity
preprocessing exist so the baseline runs on day one. The alternatives and their trade-offs are
documented on each stage (`adapter.py`), and choosing among them — and writing down why — is the
assessed part. The book's k-NN → SVM → random-forest → tuned-RF ladder is likewise an *illustrative*
history (its numbers are explicitly simulated), not a route you must walk: a team that keeps one
learner and spends every iteration on features has an equally defensible story, provided each rung is
measured under the same honest harness.

## The code is a starting point, not a lab instruction

The adapter, the track files and the menus give you a **working baseline and a set of common
options**, not the boundary of the project. You are expected to develop the code further and to
change or replace any part of it when you have a better idea: write your own preprocessing and
feature functions, try other classifiers than the random forest, change the number of folds, cut
recordings to the relevant period (e.g. the sleep period of a night), and so on. The menus in
`DESIGN_MENUS.md` are shortcuts, not a list you must choose from. Feature extraction and the
classifier have no menu at all — they are meant to be your own work.

What is **not** open is the honesty of the evaluation: keep the declared split unit, fit every
scaler/selector/model inside the training fold, report the full metric panel with its spread, and
never tune on the hold-out set (`HOLDOUT_EVALUATION.md`).

**Compare, don't just pick.** At each stage of the pipeline, try **two or three different
approaches** — from the menu, from the literature, or your own — and keep the comparison in
`RESULTS.md`. Change one thing at a time against your current best and work stage by stage; you do
not need a full grid of every combination (roughly 10–15 runs, not hundreds).

*One example of how a team could do this* — an illustration, not a checklist; your track and your
literature review should decide what is worth comparing:

| Stage | Example comparison |
|---|---|
| Preprocessing | baseline wander: high-pass vs detrend vs wavelet; powerline: notch vs adaptive; broadband noise: low-pass vs Savitzky–Golay vs wavelet denoising |
| Spectral estimation | non-parametric (periodogram, Welch, multitaper) vs parametric (AR) |
| Features | features from at least two domains, e.g. frequency band powers plus time-domain, wavelet (e.g. sub-band energies), time–frequency or nonlinear features |
| Classifier | two models, e.g. random forest vs logistic regression, SVM or gradient boosting |

Not every noise type is present in every track: Sleep-EDF EEG is sampled at 100 Hz, so 50 Hz sits at
the Nyquist frequency, and Ninapro DB1 ships a rectified, low-pass EMG envelope rather than raw EMG. Look at
the spectrum first; if a noise is not there, show that and say so rather than filtering it.

What earns the marks is not the winning option but the evidence: the alternatives you tried, their
numbers with the spread across subjects/records, at least one idea that did not help (reported as
such), and at least one earlier decision you went back and changed because of a later result
(`CAPSTONE_REPORT_RUBRIC.md`, criteria 7 and 9). Also report how many configurations you compared.

These seven modules are units of **work**, not a headcount: one person may own several, two people
may share one, and ownership can rotate between iterations. Divide them however your team's size and
strengths suggest, and record who actually did what (`results_log_TEMPLATE.md` has a place for it).

## Every iteration gets logged (definition of "done")

Chapter 16 §16.3 calls an iteration finished only when it (1) runs end to end to a result, (2) reports
the primary metric **with its spread** across subjects/records/folds, (3) is committed with a note of
what changed and why, and (4) beats the previous iteration — or explains in writing why the change was
kept anyway. Copy **[`results_log_TEMPLATE.md`](results_log_TEMPLATE.md)** into your team repo as
`RESULTS.md` on day one and add a row each time; `rep["summary"]` from `evaluate()` prints the metric
in exactly the required shape.

## How you'll be assessed (published in full — nothing held back)

Two separate grades, both published in full, no hidden criteria:

| | Points | Scope | Rubric |
|---|:-:|---|---|
| **Team project** | 30 | Shared by every team member | **[`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md)** |
| **Individual** | 10 (5 + 5) | Yours alone — not shared with your team | **[`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md)** |

Read both now, not the night before you submit — they're written as checklists as much as grading
sheets, so you can hold your work against them before handing anything in.

### Team project — 30 points, three instruments

| Instrument | Points | Format | When |
|---|:-:|---|---|
| **Code** | 11 | The submitted repository/notebook itself — pipeline structure, DSP, leakage-safe validation, reproducibility | Report deadline |
| **Report** | 12 | Written, submitted with `predictions.csv` | Report deadline |
| **Presentation** | 6 | 10–15 min talk + 5–10 min Q&A | Showcase day |
| **Deadlines** | 1 | On-time submission of the code, report, `predictions.csv`, and your showcase slot | Throughout |

**Shadow pairing.** Ahead of the project (at track assignment), your team is paired with one other team
working a **different track**. Look in on each other's progress informally around the mid-point — no
deliverable required then. After the showcase, **you individually** (not your team) write a short
critical comparison of your project against theirs — see the Individual section below. This is what
used to be a team-graded "Opposition" line; it's personal now, so it can't be delegated to whoever on
the team feels like writing it.

**Presentation Q&A — the rules, stated up front, not sprung on you:**
- Every question maps to one of the rubric's published criteria. Nothing off-rubric.
- **Every team member will be asked at least one question about a part of the work that isn't
  "theirs."** Know your teammates' sections, not just your own — this is how we check the whole team
  understands the whole pipeline, not just their own slice of it.

### Individual — 10 points, yours alone

The team's 30 points are shared equally, but two things are graded **per student**, so your grade isn't
just whoever on your team wrote the best report:

| | Points | What it checks |
|---|:-:|---|
| **Critical-comparison essay** | 5 | You're shadow-paired with one other team on a different track — after the showcase, write a short, specific comparison of your project against theirs: what they did differently, what you'd borrow, what you'd change |
| **Teamwork & contribution** | 5 | Fair workload split, communication, and how your team handled disagreements — via a private peer evaluation and a short individual reflection (ClickUp is expected but not graded) |

Full format and criteria for both: `INDIVIDUAL_ASSESSMENT.md`. The teamwork component exists because a
good report can hide an unequal team — it's not there to catch you out, it's there so doing your fair
share is worth exactly as much as it should be.

**If you think a score is wrong:** point to the specific rubric criterion and the specific evidence you
believe was misread, and ask for a regrade on that criterion — email the instructor or raise it during
a project session, within one week of the grade being posted.

## The three golden rules (you will be graded on these)

1. **No leakage.** Never let the same subject/record appear in train and test; fit scalers and any
   feature/spatial selection **inside** each CV fold. See `../docs/LEAKAGE.md`.
2. **Honest metrics.** Never accuracy alone — report Cohen's κ, macro-F1, and the confusion matrix.
   State the **split unit** (and evaluation mode) with **every** number, and quote the metric **with
   its spread** ("mean κ 0.61, range 0.34–0.73 across 8 subjects"), never the pooled number alone.
3. **Compare against the *supplied* baseline, honestly.** Beating it is not required — a
   defended decision to keep a lower-scoring pipeline earns full marks (Criterion 7); an
   unexplained gain does not. A small honest gain beats a large leaky one. Smoke/CI numbers
   are plumbing checks — never report them as results.

## The tracks

Three default tracks, matched on difficulty (all ★★★, roughly comparable engineering lift) —
**not** on the raw baseline number below, which differs hugely by design (a sleep-staging κ and
an ECG-rhythm κ are not the same claim). Read each track's own card for what "good" looks like
on that domain:

| Track | Signal | What you're predicting | Split | Modes | Baseline (honest) |
|-------|--------|------|-------|-------|---|
| `sleep_edf` | EEG + EOG + EMG | **5-stage sleep staging** — Wake / N1 / N2 / N3 / REM, one label per 30-s epoch | subject | new-subject | κ 0.84 (reference) — **inflated**: above the κ≈0.76 human-rater ceiling; see `sleep_edf_card.md` before treating this as a target |
| `ecg_cinc2017` | single-lead ECG | **4-class rhythm** — Normal / AF / Other / Noisy, one label per recording | record | new-record | κ 0.52 |
| `emg_ninapro` | 10-ch surface EMG | **12-class hand-gesture recognition**, one label per 200-ms window | subject / repetition | **within + new-subject** | κ 0.77 within / **0.12 new-subject** |

**Two evaluation modes** (EMG): report **within-subject** *and* **new-subject** — they answer
different deployment claims, and cross-subject is much harder. Do not hide a weak cross-subject number.

**Opt-in stretch track — `bci_eegmmidb`** (64-ch EEG, **binary motor imagery** — did the subject imagine moving their left or right hand, one label per ~4-s trial, subject/trial split,
within + new-subject modes, near-chance naive baseline). Available only by explicit request with
instructor sign-off — it's meaningfully harder than the three defaults, and its honest baseline is
close to chance until you add a spatial filter (CSP). That's the point of the track, not a flaw.

*(`har` and `ctg_ctu_uhb` are not offered this year — CTG's honest baseline sits far below the other
tracks, closer to a floor than a moderate challenge.)*

## Reality check (this is a feature, not a bug)

Real data is messy and some tracks are genuinely hard: **EMG**'s cross-subject accuracy collapses far
below its within-subject number, and the naive **BCI** baseline (if you take the opt-in) is near chance
until you add a spatial filter (CSP). Your dataset card's **yardstick** line tells you what "good"
actually is for that domain — read your result against it, not against 100 %. Honest, well-motivated
work on a hard signal beats a leaky 0.99 on an easy one.
