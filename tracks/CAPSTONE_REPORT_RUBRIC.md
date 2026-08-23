# Capstone assessment rubric — 30 points, one sheet per track

Each team is graded out of **30 points**, across **three instruments** — the **code**,
the written **report**, and the **presentation** (talk + Q&A). Every track uses the
**same 9 criteria and the same point values**, so grading is consistent across tracks;
only the two *domain-specific* criteria (7–8) are concretized per track, using each
track's own `TrackMeta` (default metric, split unit, eval modes) from `adapter.py` —
never a criterion invented ad hoc per report. This is a **raw point score** — there is
no letter-grade conversion; the 30-point total and its per-criterion breakdown *is* the
grade record.

**The individual critical-comparison essay and teamwork component are graded
separately, outside these 30 points.** They test each *student's* own understanding and
engagement (catching free-riding, and making sure peer-learning isn't delegated to one
team member) — see `INDIVIDUAL_ASSESSMENT.md`.

Structurally, this rubric borrows one idea from a KI-style master-thesis grading sheet:
a small **Deadlines** line (separating "good work" from "good work, on time"). The other
borrowed idea — **Opposition**, giving structured feedback on another team's project —
lives entirely in `INDIVIDUAL_ASSESSMENT.md` now, as an individual critical-comparison
essay (replacing what used to be a closed-book quiz) rather than a team-graded line
here: every student personally reads and critiques a shadow-paired (different-track)
team's project, rather than the team delegating one piece of feedback to whoever
volunteers to write it. Teams are still shadow-paired (cross-track, assigned once at
track-assignment time) so every student knows which project to compare against — see
`Course_PM_HT26.md`'s track-assignment section.

**How to grade:** for each criterion, match the evidence to the row that best fits and
record that many points. Half-points are fine. Do not award points for a claim made but
not supported (a metric stated with no confusion matrix, a split described but not
shown in code/config) — evidence, not assertion, earns points throughout. This mirrors
the book's own §1.8 principle: no result without a diagnostic.

## The 7 common criteria (identical across all 4 tracks) — 23 points

Criteria 2 and 5 absorbed the 2 points that used to be the team-level Opposition
criterion (now an individual essay, see above) — signal-processing rigor and the
report's defence of its own design decisions are the two places that extra weight
does the most good.

> **⚠️ Point rebalance — a deliberate judgement call, flagged for the instructor.**
> Criterion **9 (Iteration & revision history, 3 pts)** was added because the course's
> stated philosophy — *"mostly needing to revisit [design choices] based on results"* —
> was previously required by `results_log_TEMPLATE.md` but worth **zero** rubric points,
> and that template even said "nothing here is graded on its own". Rather than inflate
> the sheet to 33, the 3 points were taken from the two criteria the new one partly
> subsumes: **Criterion 7 (Improvement on the baseline) 6 → 4** and **Criterion 5 (Report
> quality & defence) 4 → 3**. The rationale is the rubric's own stated intent — *"a team
> with a modest metric but honest, well-diagnosed work should outscore a team with a
> better number and sloppy execution"* — so weight moves from the *outcome* (the number
> improved) toward the *process that makes an improvement trustworthy*. **The total stays
> 30**, and Criterion 3's "~15/30" soft gate is unaffected. If you disagree with where the
> 3 points came from, change *that* — not the existence of the criterion.

| # | Criterion | Points | Instrument | Full marks | Partial | Minimal | Missing |
|---|---|:---:|---|---|---|---|---|
| 1 | **Pipeline integrity** | 3 | Code | The submitted code shows all seven modules — load → preprocess → features → selection → classification → inference → report — as separable, reusable stages | Pipeline runs end to end but 1–2 stages are entangled or undocumented (e.g. feature extraction hard-coded inside training) | Pipeline is a single script; stages exist implicitly but are not separable | No coherent pipeline; ad hoc / notebook-only exploration |
| 2 | **Signal processing rigor** | 5 | Code (3) + Report (2) | Real DSP performed on the raw signal in the code (filtering, transforms, spectral estimation, denoising, or fiducial detection as appropriate); the report justifies each choice against the track's `dsp_focus`; features are *constructed*, not just handed to a classifier | DSP is present and correct but under-justified in the report (right method, no rationale for parameters) | Minimal DSP; mostly relies on the signal's raw amplitude/statistics or the supplied baseline's features unchanged | No signal processing — classifier applied directly to raw or pre-given features |
| 3 | **Validation honesty** | 6 | Code (3) + Report (2) + Presentation (1) | The code enforces the correct **split unit** everywhere (see Criterion 8 for what that means on this track) and fits any scaler/selector inside the training fold only; the report states the split unit and shows the full metric panel — confusion matrix + the track's default metric(s) + at least one more (e.g. balanced accuracy); the team can explain the leakage risk live | Split unit is correct in code but one leakage risk is present elsewhere (e.g. feature selection fit on the whole set), or the report doesn't state it clearly | Split unit is inconsistently applied in code, or only a single metric (often bare accuracy) is reported | No group-aware split in the code (random/epoch-level split on grouped data) — a hard fail on this criterion regardless of the reported number |
| 4 | **Reproducibility & engineering** | 2 | Code | Fixed seed, config-driven run, environment pinned (`requirements.txt`/lock file), and the track's `smoke()` path passes — a grader can re-run the submitted code and get the same numbers | Reproducible in spirit but missing one element (e.g. seed fixed, but no pinned environment) | Runs, but re-running plausibly gives different numbers (unseeded randomness) | Not reproducible from what's submitted |
| 5 | **Report quality & defence** | 3 | Presentation | Design decisions are justified against the spec; results are read against a meaningful ceiling (chance, the supplied baseline, or an inter-rater/label-noise ceiling where relevant); limitations are stated honestly, including a concrete "what would go wrong on new data" | Clear report, but justification or limitations section is thin | Report describes *what* was done but rarely *why*; no honest limitations | Report does not explain the pipeline well enough to follow |
| 6 | **Deadlines** | 1 | Process | Report, code, `predictions.csv`, and the presentation slot are all delivered on time, no extension needed | Late by less than 48h with prior notice, no disruption to grading | — | Late without notice, or a deliverable is missing entirely |
| 9 | **Iteration & revision history** | 3 | `RESULTS.md` (2) + Presentation (1) | `RESULTS.md` (from `results_log_TEMPLATE.md`) shows a dated iteration log with the primary metric **and its spread** per row, a decision log naming the alternatives rejected, **and at least one decision that was explicitly revisited and changed because of a later result** — with the downstream symptom that triggered it named (e.g. "worst-subject κ 0.14 vs mean 0.61 → went back to stage 2 and added per-subject normalisation"). The team can walk through that reversal live | An iteration log with spreads and a decision log, but every change moves *forward* — no earlier decision is ever revisited, or a revision happened but the triggering result isn't named | `RESULTS.md` exists but is a bare list of numbers: no spreads, no reasons, no alternatives, no revisions | No iteration record; the pipeline is presented as if it arrived finished — which no honest pipeline does |

**On Criterion 9.** This is the criterion that grades the *loop*, not the destination.
Going forward — try something, keep it, try the next thing — is a to-do list, not an
investigation. What earns the marks is one place where a downstream number sent the team
back **up** the pipeline: a per-class recall that exposed a missing feature, a
worst-subject κ that indicted the preprocessing, a "feature selection changed nothing"
harness note that revealed `select_k` exceeded the feature count. A revision that *lowered*
the headline metric but was kept for a stated reason ("κ fell 0.02 but the worst subject
rose from 0.18 to 0.31") is **full marks** evidence, not a weakness. `git log` is the
cross-check that the log was kept as the work happened rather than reconstructed the night
before; a `RESULTS.md` committed once, complete, on the deadline scores the "minimal" row
however well written it is.

## The 2 domain-specific criteria — 7 points

Filled in per track below. Both are graded the same way as the common criteria — match
the evidence to a row.

**On Criterion 7 — it grades the comparison, not the raw metric.** This criterion used
to require beating `default_baseline()` for the top bands, which put "finished below the
baseline" in the *Missing* row no matter how rigorous the work behind it was. That
contradicted this rubric's own stated intent (*"a team with a modest metric but honest,
well-diagnosed work should outscore a team with a better number and sloppy execution"*)
and, worse, it paid teams to stop investigating the moment the number went up. The bands
above now grade the **quality of the evidence**: controlled comparisons that change one
thing at a time, results read **with their spread**, alternatives that were genuinely
tried and **falsified**, and a final trade-off that is *chosen and priced* rather than
merely arrived at.

Three consequences a grader should apply consistently:

- **A well-justified decision to keep a lower-scoring pipeline earns full marks.** "κ fell
  0.02 but the worst subject rose from 0.18 to 0.31, and the filter is now causal, so the
  real-time claim is honest" is a *better* answer than an unexplained +0.03. What must be
  present is the alternative, the number, and the reason — not the win.
- **A difference smaller than the per-group spread is not a result.** A 0.01 macro-F1 gain
  on a five-subject cohort whose spread is 0.4 is noise; a report that presents it as an
  improvement should not score above *Partial*, however large the table around it.
- **How many configurations were compared is itself evidence, and belongs in the report.**
  Comparing twenty configurations against the same folds and reporting the best one is a
  selection effect the folds cannot protect against (see the "garden of forking paths"
  section in the notebooks and in `results_log_TEMPLATE.md`). A team that says "we
  compared six configurations, chose on a held-out development set, and report the winner
  once on the evaluation folds" is doing better science than one that reports a longer
  table — grade it that way.

The point value is unchanged at **4**, and the sheet still totals **30**.

---

## Track 1 — Sleep staging (Sleep-EDF Expanded)

`default_metrics`: Cohen's κ, macro-F1 · `split_unit`: **subject** · `eval_modes`: new-subject · difficulty 3/4

| # | Criterion | Points | Instrument | Full marks | Partial | Minimal | Missing |
|---|---|:---:|---|---|---|---|---|
| 7 | **Improvement on the baseline — evidence-based** | 4 | Report (+ `RESULTS.md`) | A **controlled comparison** against `default_baseline()` under the correct subject-level split: ≥2 alternatives actually tried, each changing **one thing at a time**, reported with κ **and its per-subject spread**, and at least one alternative **falsified** (tried, didn't help, said so). The final pipeline is then justified — either because κ rose and the report says *which* choice drove it, **or** because a lower/equal-κ pipeline was deliberately kept for a stated reason (worst-subject κ, interpretability, a causal/deployable filter, robustness to a nuisance) with the trade-off named and priced | The comparison is real but thin — one alternative, or several changed together so no claim about cause survives — or the choice of final pipeline is asserted rather than shown against the spread | Final and baseline κ are reported side by side but nothing was controlled: no alternative was falsified, and the reason for the final configuration is not given | No comparison to the supplied baseline at all, κ not reported, or computed under an invalid split |
| 8 | **Track-specific execution: subject vs. night** | 3 | Report (2) + Presentation (1) | Report explicitly confirms the split is by **subject**, not by night — and names this as a leakage risk it checked for, since Sleep-Cassette has 2 nights per subject | Split is correct by subject but the report doesn't explain why night-level splitting would leak | Split unit is ambiguous in the report (can't tell if it's subject- or night-based) | Split is by night or by epoch — a subject's two nights appear in both train and test |

**Total: ___ / 30**

---

## Track 2 — ECG rhythm classification (PhysioNet/CinC 2017)

`default_metrics`: macro-F1, Cohen's κ · `split_unit`: **record** · `eval_modes`: new-record · classes: N/A/O/~ (Normal/AFib/Other/Noisy) · difficulty 3/4

| # | Criterion | Points | Instrument | Full marks | Partial | Minimal | Missing |
|---|---|:---:|---|---|---|---|---|
| 7 | **Improvement on the baseline — evidence-based** | 4 | Report (+ `RESULTS.md`) | A **controlled comparison** against `default_baseline()` under the correct record-level split: ≥2 alternatives tried one change at a time, reported with macro-F1, **per-class** performance and the fold spread, at least one alternative **falsified**, **and** the noisy class (`~`) handled deliberately — never silently collapsed or dropped. The final pipeline is justified either by a gain whose source is named, **or** by a defended decision to keep a lower/equal-macro-F1 pipeline (e.g. one that keeps `~` recall usable, or whose filter does not destroy the quality evidence the `~` class depends on) with the trade-off priced | Comparison is real but thin, or several things changed at once, or `~` is treated like any other class with no discussion | Final and baseline macro-F1 appear side by side with no controlled comparison behind them, or the result is given as bare accuracy (which hides collapse onto Normal) | No comparison to the supplied baseline, not reported, or computed under an invalid split |
| 8 | **Track-specific execution: 4-class imbalance & signal quality** | 3 | Report (2) + Presentation (1) | Report shows per-class performance (not just the macro average) and discusses the class imbalance (Normal dominates; AFib and Noisy are rare) and how it was addressed (class weights, resampling, or an explicit rationale for not doing so) | Per-class breakdown shown but imbalance not discussed | Only the aggregate macro-F1 is shown, no per-class view | No acknowledgement that this is an imbalanced 4-class problem |

**Total: ___ / 30**

---

## Track 3 — EMG gesture recognition (Ninapro DB1, exercise E1)

`default_metrics`: macro-F1, Cohen's κ · `split_unit`: **subject** · `eval_modes`: **within-subject AND new-subject (both required)** · classes: 12 finger gestures · difficulty 3/4

| # | Criterion | Points | Instrument | Full marks | Partial | Minimal | Missing |
|---|---|:---:|---|---|---|---|---|
| 7 | **Improvement on the baseline — evidence-based** | 4 | Report (+ `RESULTS.md`) | A **controlled comparison** against `default_baseline()` on **both** required eval modes, one change at a time, with spreads and at least one alternative **falsified** — **and the within-subject vs. new-subject gap discussed**, because on this track that gap is the finding. The final pipeline is justified either by a gain (named source), **or** by a defended decision to keep a lower/equal-scoring pipeline that *closes the gap* — e.g. per-subject normalisation that costs within-subject macro-F1 but makes the cross-subject number usable, with the required calibration procedure named. **A "gain" that appears only within-subject is not an improvement on this track and must not be presented as one** | Both modes compared but thinly, or the within/new gap isn't discussed | Comparison exists on only one of the two required modes, or the two modes are reported without any controlled comparison behind them | Only one mode attempted and not labelled as partial, or no comparison to the supplied baseline |
| 8 | **Track-specific execution: reporting both eval modes** | 3 | Report (2) + Presentation (1) | Both `within-subject` and `new-subject` results are reported **side by side**, correctly labelled, with the (expected) cross-subject drop named and not hidden | Both are reported but not clearly labelled or compared | Both numbers appear somewhere in the report but are not distinguished as the two required modes | Only within-subject is reported, letting a weak cross-subject result go unmentioned — this is the specific shortcut this criterion exists to catch |

**Total: ___ / 30**

---

## Track 4 — BCI motor imagery (EEGMMIDB) — opt-in stretch track

`default_metrics`: macro-F1, Cohen's κ · `split_unit`: **subject** · `eval_modes`: **within-subject AND new-subject (both required)** · classes: L/R (binary) · difficulty 4/4 (hardest track; naive baseline is near chance). **Opt-in only** — a team must request this track explicitly with instructor sign-off; it is not in the default assignment pool.

| # | Criterion | Points | Instrument | Full marks | Partial | Minimal | Missing |
|---|---|:---:|---|---|---|---|---|
| 7 | **Improvement on the baseline — evidence-based** | 4 | Report (+ `RESULTS.md`) | A **controlled comparison** against `default_baseline()` on both modes, one change at a time, reported with the **per-subject spread** — which on this track is the actual result, since a pooled number near chance can hide one clearly decodable subject. Full marks for **either** reliable above-chance `new-subject` performance (rare here, and a real achievement) **or** a well-designed set of experiments that *falsifies* alternatives and explains why — e.g. "CSP fit inside the fold raised subject S3 from 0.51 to 0.72 and left the other five at chance; the fixed C3/C4 montage cannot represent their sensorimotor pattern" — with the conclusion stated as a limitation rather than dressed up as a win. **On this track a rigorous negative result, correctly diagnosed, is worth full marks; an unexplained near-chance number is not** | Above-chance on `within-subject` only, or a small real `new-subject` gain, or a genuine comparison whose diagnosis of the near-chance result is thin | Near chance on both modes with a genuine, well-described attempt (band-power or CSP at C3/C4/Cz) but no controlled comparison and no per-subject reading | No meaningful attempt beyond the naive baseline, or results indistinguishable from chance with no discussion of that fact |
| 8 | **Track-specific execution: reporting both eval modes** | 3 | Report (2) + Presentation (1) | Both `within-subject` and `new-subject` results reported side by side, correctly labelled; report explicitly discusses *why* this track is hard (subject-to-subject variability in sensorimotor rhythms) | Both reported but the "why it's hard" discussion is missing | Only one mode reported, or modes conflated | Cross-subject result missing or silently substituted with the easier within-subject number |

**Total: ___ / 30**

---

## Using this rubric

- **Grade on the work, not the raw metric.** The hold-out evaluation
  (`HOLDOUT_EVALUATION.md`) reports each team's score on the withheld set back to that
  team only — it is not a cross-team ranking; this rubric grades the *code, report, and
  presentation*. A
  team with a modest metric but honest, well-diagnosed work should outscore a team with
  a better number and sloppy execution — that is the point of Criteria 5 and 9.
- **Criterion 3 (validation honesty) is a soft gate.** Code that fails the "no
  group-aware split" bottom row should not score above ~15/30 regardless of how good
  the other criteria look — a leaked number is not evidence of anything.
- **"Code" means the actual submitted repository/notebook, not the report's description
  of it.** Where code and report disagree (report claims fold-safe scaling, code shows
  otherwise), grade Criterion 3 from the code — it's the ground truth.
- **Criterion 9 needs the log to exist *during* the project.** Say so at track assignment,
  not at the deadline: `results_log_TEMPLATE.md` is copied into the team repo as `RESULTS.md`
  in week 1 and committed as the work happens. A log reconstructed afterwards reads exactly
  like one — every row an improvement, no dead ends, no reversals — and that pattern is the
  "partial" row, because a project with no dead ends is a project with no investigation.
- **The critical-comparison essay and teamwork are separate.** Both are graded
  individually, outside this 30-point sheet — see `INDIVIDUAL_ASSESSMENT.md`. The essay
  is what used to be Opposition; it's now personal, not delegated to one team member.
- **Shadow pairing needs assigning once, early.** Pair each team with one other team on
  a different track at track-assignment time (`Course_PM_HT26.md`), not near the
  showcase — students need to have actually followed the other project to write a real
  comparison, not a cold read on the day.
- **Same rubric, six sheets.** Because criteria 1–6 and 9 and their point values are identical
  across tracks, a grader handling multiple tracks does not need to re-learn the scale
  per team — only criteria 7–8 change, and only in what counts as evidence, not in how
  many points they're worth.
- **No letter-grade conversion.** The 30-point total and its criterion-by-criterion
  breakdown is the grade record itself — report it to students exactly as scored.
