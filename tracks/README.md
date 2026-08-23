# Capstone tracks — the multi-domain project system

Different student groups take **different signals** through the **same pipeline** and present
their results side by side. That is the peer-learning payoff: everyone sees the same methods
behave differently on EEG vs ECG vs IMU vs EMG.

> **Students start at [`START_HERE.md`](START_HERE.md)** — the 5-step learning path (pick a track →
> refresh + literature review via [`BACKGROUND_MAP.md`](BACKGROUND_MAP.md) → run the baseline → improve
> it honestly → report + submit). This README is the *architecture* view for instructors/builders.

## The two-layer rule
- **Synthetic** data (`src/bsp/biosignals.py`, `sleep_pipeline.py`) is the **teaching / debug /
  CI** layer used by the 16 chapter notebooks. It never goes away.
- **Real public data** (this folder) is the **capstone / application** layer. Every dataset here
  is a **direct, license-only download — no data-use agreement** (no SHHS/NSRR, DEAP, WESAD, VitalDB).

## What every track ships (the handover)
1. A **supplied baseline model** — students *improve* it, never start from a blank page
   (`adapter.default_baseline`).
2. A **dataset card** — license, citation, task, labels, **split unit**, runtime, default metric,
   known pitfalls (auto-rendered from metadata by `TrackAdapter.dataset_card()`; see `*_card.md`).
3. A **smoke-test subset** — a synthetic `smoke()` path so CI / offline / Colab-sanity runs green.
4. A **declared leakage unit** (`SPLIT_UNIT`) enforced on every fold.
5. A **common rubric** (`CAPSTONE_REPORT_RUBRIC.md`, 30 pts) — same 8 criteria on every track, only the two
   domain-specific ones differ. (`RUBRIC.md` is the superseded first draft, kept for history.)
6. **Two evaluation modes** where relevant (BCI/EMG): `new-subject` and `within-subject`.
7. An **iteration results log** (`results_log_TEMPLATE.md`) students copy into their own team repo —
   the per-iteration record (what changed and why · metric **with spread** · better than last time,
   or why it was kept anyway · commit) that Chapter 16 §16.3 makes part of "done".

## Files
- `adapter.py` — the `TrackAdapter` contract + `TrackMeta` (a dataset card in code) + the book's
  **seven separable stages** (`load`/`smoke` → `preprocess` → `extract_features` → `select_features`
  → `baseline` → `infer` → `report`, §16.2 / rubric Criterion 1) + the shared, never-overridden
  `evaluate()` (leave-one-group-out for few groups, 5-fold GroupKFold for many, leakage guard,
  **selection fit inside every fold**) which returns per-group / per-fold results and the metric's
  **spread**, not just a pooled number + `default_baseline()` + the hold-out evaluation harness
  (`train_baseline`, `write_submission`, `holdout_score`). `preprocess()` and `select_features()`
  default to pass-through and document their alternatives with trade-offs — the scaffold offers
  options, it does not pick the pipeline for the student. Each track declares the cfg keys it
  **actually consumes** in `SUPPORTED_CFG_KEYS`, on top of the shared `BASE_CFG_KEYS` (stages 4-5
  and the CV geometry, which work everywhere): the notebook generator renders only the menus a
  track really has, and a key the track never reads raises `UnsupportedCfgKey` instead of being
  silently ignored — an advertised option that provably does nothing is worse than no option.
  Implemented a stub yourself? `track.declare_cfg_keys("csp_components", ...)` makes its knobs
  first-class. `train_baseline()` records the resolved cfg on the returned `FittedModel`, and
  `infer()` / `write_submission()` reuse it, so a model can never be scored on features built by a
  different pipeline than the one it was validated under.
- `report.py` — **module 7**: `summarize_results()` / `summarize_report()` (confusion matrix first,
  primary metric with spread, macro-F1, balanced accuracy), `plot_confusion()`, `plot_hypnogram()`
  (predicted vs. reference stage sequence, disagreements marked — sleep and any sequence-labelled
  track), and `stage_summary()` / `compare_stage_summaries()` for the clinical night-level numbers
  (TST, sleep efficiency, WASO, SOL, REM latency).
- `sleep_edf.py` — **reference track** (Sleep-EDF Expanded). Real `mne` loader (Colab) +
  synthetic smoke (offline). R&K→AASM (S3+S4→N3), drop MOVEMENT/UNKNOWN, split by **subject**.
- `ecg_cinc2017.py` — **second built track** (PhysioNet/CinC-2017 single-lead ECG). Real `wfdb`
  loader (Colab) + synthetic smoke. 4-class rhythm N/A/O/~; split by **record**; QRS/HRV/SQI
  features; record-level hold-out evaluation.
- `har.py` — **HAR** (UCI-HAR raw inertial), IMU activity; split by subject. Real loader validated.
- `ctg_ctu_uhb.py` — **CTG (fetal)** (CTU-UHB, `wfdb`), FHR+UC, normal/pathological by umbilical pH;
  split by recording. Real loader validated (a deliberately hard problem).
- `emg_ninapro.py` — **EMG gesture** (Ninapro DB1), 10-ch sEMG; **two eval modes** (within- vs
  new-subject) via `evaluate_modes()`. Real loader validated.
- `bci_eegmmidb.py` — **BCI motor imagery** (EEGMMIDB, `mne`), 64-ch EEG L/R imagery; **two eval
  modes**. Real loader validated (near-chance naive baseline — CSP is the student's job).
- `HOLDOUT_EVALUATION.md`, `TRACK_INSTRUCTIONS_TEMPLATE.md`, `dataset_card_TEMPLATE.md`, `results_log_TEMPLATE.md`,
  `readiness_matrix_TEMPLATE.md` — the hold-out evaluation mechanics + templates for each new track.
- `CAPSTONE_REPORT_RUBRIC.md` — `RUBRIC.md`'s common/domain-specific split made concrete: a
  30-point (code + report + presentation), 8-criterion **team** grading sheet **per track**, each
  with 4-tier point-band descriptors, so grading is transparent to students and consistent across
  tracks and graders.
- `INDIVIDUAL_ASSESSMENT.md` — the critical-comparison essay and teamwork/contribution component
  (10 points, graded **per student**, kept separate from the team's 30 so an unequal team can't hide
  behind a good report, and so peer feedback can't be delegated to one team member).
- `*_card.md` (rendered dataset cards) and `*_instructions.md` (per-track student handouts) for each
  built track. Tracks with two eval modes (EMG, BCI) add an `evaluate_modes()` method to the adapter.
- `dataset_manifest.json` — a **machine-readable provenance record** for all six datasets
  (version · licence · URL · citation · split-unit · smoke-test record IDs) plus, for the local
  `data_cache/`, a total-size + **aggregate sha256 fingerprint** of every cached file and per-file
  checksums for the smoke-test records present. Built by `../tools/build_dataset_manifest.py`.

## Reproducibility & provenance (the real-data layer is *conditionally verified*)

The synthetic layer is fully reproducible and CI-gated. The **real-data** layer is honest about
its status: the six loaders' synthetic smoke paths pass in CI, and the six real loaders were
validated locally (see each card's "Measured on real data"), but a fresh download is not
re-verified on every CI run. Two artifacts make that layer auditable:

- **Toolchain pins** — `../requirements-real.txt` pins `mne==1.10.1` and `wfdb==4.3.0` (the
  versions the tracks were validated with), because a silent EDF/WFDB upgrade can change scaling
  or record parsing and quietly shift a baseline.
- **Provenance manifest** — `dataset_manifest.json` records exactly which dataset version, licence,
  and records were used, with checksums. Run `python ../tools/build_dataset_manifest.py --verify`
  to check a local cache **byte-for-byte** against it (it catches a changed/missing cached file).
  Scope: this verifies raw cached bytes, **not** parsed output — the exact pins above constrain
  parser behaviour; parsed-output stability would need canonical record/segment fingerprints.
- `BACKGROUND_MAP.md` — per track, **which book sections refresh each method** (background students
  already have from the course) plus the **required literature-review** anchors, search terms, and
  guiding questions. This is what motivates the design (course outcome L5).

## Add a new track (the whole job)
1. Copy `sleep_edf.py` → `mytrack.py`; fill in `TrackMeta`.
2. Implement `smoke()` (synthetic, from `bsp.biosignals`), `extract_features()` (the DSP), optionally
   `preprocess()`, and `download()/load()` (Colab real data). Inherit `baseline()`, `select_features()`,
   `infer()`, `report()` and the shared `evaluate()`. (The pre-refactor single `features(rec)` method
   still works — the base class detects the override and warns — but new tracks should split it.)
3. `python3 mytrack.py` prints the card and runs the offline smoke LOSO — that is the CI gate.
4. Score it on `readiness_matrix_TEMPLATE.md`; write the human pitfalls into its `*_card.md`;
   hand students `TRACK_INSTRUCTIONS_TEMPLATE.md`.

## First-pass scope (do not start ten tracks)
**Built:** Sleep-EDF (reference) and **ECG / CinC-2017** (second track) — two different signals,
one adapter interface, one rubric, one hold-out evaluation harness. That proves the architecture; the
remaining tracks (HAR, CTG, EMG, BCI, …) are now a config/data-card addition, not new architecture.

## Hold-out evaluation
Every track runs a per-track hold-out evaluation (submit `predictions.csv`, instructor scores it
against withheld labels on the track's default metric — reported to your team only, not ranked
against other teams) alongside the cross-track showcase.
Format and mechanics: `HOLDOUT_EVALUATION.md`.

## Fully-open core (agreement-free)
Sleep-EDF · UCI-HAR/PAMAP2 · MIT-BIH/CinC-2017 (ECG) · CTU-UHB (CTG) · Ninapro (EMG) ·
EEGMMIDB (BCI). Agreement-free extras later: Apnea-ECG, CinC-2016 (PCG), CHB-MIT (seizure),
BIDMC (resp rate), Abdominal-Fetal-ECG (ICA showcase), Daphnet/gaitpdb, drivedb (stress).
