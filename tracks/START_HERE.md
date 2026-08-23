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
| **3. Run the baseline** | Open the track's notebook. It runs on **synthetic data offline** by default; set `USE_REAL = True` in **Colab** for the real dataset. Read the **honest metric panel** (κ / macro-F1 / confusion), not accuracy. | `notebooks/track_<name>.ipynb` |
| **4. Improve the baseline** | Do **real DSP** — better preprocessing and **feature construction** (the point of the course), then a better model. Validate **inside the folds**, keep the **declared split unit**. Log each iteration as you go. | the adapter's `preprocess()` and `extract_features()` are where you work; `results_log_TEMPLATE.md` is where you log it |
| **5. Report + submit** | Write up your design (justified, read against the **yardstick**), submit `predictions.csv` for hold-out evaluation, and take a slot in the **cross-track showcase**. | `HOLDOUT_EVALUATION.md`, `CAPSTONE_REPORT_RUBRIC.md`, `report.py` |

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
| **Teamwork & contribution** | 5 | Fair workload split, communication, and how your team handled disagreements — via a task log, peer evaluation, and a short individual reflection |

Full format and criteria for both: `INDIVIDUAL_ASSESSMENT.md`. The teamwork component exists because a
good report can hide an unequal team — it's not there to catch you out, it's there so doing your fair
share is worth exactly as much as it should be.

**If you think a score is wrong:** point to the specific rubric criterion and the specific evidence you
believe was misread, and ask for a regrade on that criterion. *(Instructor: fill in the actual
contact/process here.)*

## The three golden rules (you will be graded on these)

1. **No leakage.** Never let the same subject/record appear in train and test; fit scalers and any
   feature/spatial selection **inside** each CV fold. See `../docs/LEAKAGE.md`.
2. **Honest metrics.** Never accuracy alone — report Cohen's κ, macro-F1, and the confusion matrix.
   State the **split unit** (and evaluation mode) with **every** number, and quote the metric **with
   its spread** ("mean κ 0.61, range 0.34–0.73 across 8 subjects"), never the pooled number alone.
3. **Beat the *supplied* baseline, honestly.** A small honest gain beats a large leaky one. Smoke/CI
   numbers are plumbing checks — never report them as results.

## The tracks

| Track | Signal | Task | Split | Modes | ★ |
|-------|--------|------|-------|-------|---|
| `har` | tri-axial accelerometer | activity (3-class) | subject | new-subject | ★★ (gentle on-ramp) |
| `sleep_edf` | EEG + EOG + EMG | sleep stage (5-class) | subject | new-subject | ★★★ (reference) |
| `ecg_cinc2017` | single-lead ECG | rhythm N/A/O/~ | record | new-record | ★★★ |
| `ctg_ctu_uhb` | FHR + uterine contraction | normal / pathological | recording | new-recording | ★★★ (hard, honest) |
| `emg_ninapro` | 10-ch surface EMG | hand gesture (12-class) | subject / repetition | **within + new-subject** | ★★★ |
| `bci_eegmmidb` | 64-ch EEG | L/R motor imagery | subject / trial | **within + new-subject** | ★★★★ (advanced) |

**Two evaluation modes** (EMG, BCI): report **within-subject** *and* **new-subject** — they answer
different deployment claims, and cross-subject is much harder. Do not hide a weak cross-subject number.

## Reality check (this is a feature, not a bug)

Real data is messy and some tracks are genuinely hard: **CTG** cord-pH prediction has low sensitivity,
and the naive **BCI** baseline is near chance until you add a spatial filter (CSP). Your dataset card's
**yardstick** line tells you what "good" actually is for that domain — read your result against it, not
against 100 %. Honest, well-motivated work on a hard signal beats a leaky 0.99 on an easy one.
