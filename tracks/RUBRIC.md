> ⚠️ **SUPERSEDED — do not grade from this file.** The live rubric is
> **[`CAPSTONE_REPORT_RUBRIC.md`](CAPSTONE_REPORT_RUBRIC.md)** (30 points, 9 criteria, one sheet
> per track) plus [`INDIVIDUAL_ASSESSMENT.md`](INDIVIDUAL_ASSESSMENT.md) (10 points, individual).
> This first draft is kept only as design history — its percentage weights and its team-level
> Opposition line are no longer in force.

# Capstone rubric — common discipline, domain-weighted *(superseded draft)*

The **process** is graded identically across tracks; the **numbers** are read against each
domain's own difficulty, so a group on a hard signal (BCI, cross-subject) is not punished
relative to an easy one (HAR).

## Common (same for every track) — 70%

| Criterion | Weight | What we look for |
|-----------|:------:|------------------|
| **Pipeline integrity** | 15% | data_loader → preprocessing → features → selection → classification → inference → report, as reusable modules |
| **Signal processing** | 15% | real DSP done on raw signals (filtering, transforms, spectral, denoising, feature *construction*) — not just ML on given features |
| **Validation honesty** | 20% | correct **split unit** (no subject/patient leakage); scaling & selection inside folds; the honest metric panel (confusion, κ / macro-F1, balanced acc) |
| **Reproducibility & engineering** | 10% | fixed seed, config-driven, smoke test passes, environment pinned |
| **Report & defence** | 10% | design decisions justified; results read against the inter-rater/chance ceiling; honest about limits |

## Domain-specific — 30%

| Criterion | Weight | Notes |
|-----------|:------:|-------|
| **Domain metric vs baseline** | 20% | improvement over the *supplied* baseline on the track's **default metric** (e.g. sleep κ; HAR macro-F1; ECG AF-F1; BCI decoding acc) |
| **Difficulty weight** | 10% | scaled by the track's difficulty (1–5 in its `TrackMeta`) and evaluation mode (cross-subject > within-subject) |

## Rules
- **Beat the supplied baseline, honestly.** A smaller honest gain > a large leaky one.
- **State the split unit and evaluation mode** with every number.
- **Two modes where required** (BCI/EMG): report `new-subject` *and* `within-subject`; do not
  hide a weak cross-subject result behind a strong within-subject one.
- **Synthetic vs real:** smoke/CI numbers are plumbing checks, never reported as results.
- **Deliverables:** report (≤ N pages), a `predictions.csv` on the held-out split (per-track
  hold-out evaluation, optional), and a slot in the **cross-track showcase**.
