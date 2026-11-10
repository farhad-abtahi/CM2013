# Results log — EMG Ninapro

**Track:** `emg_ninapro` · **Split unit:** subject / repetition · **Primary metric:** macro-F1 · **Evaluation modes:** within-subject + new-subject

**Current development pool:** S1–S22, E1. S23–S27 are reserved for final hold-out. Iterations 1–7 used the earlier S1–S5 subset; iterations 8–13 use the 22-subject population; iterations 14–15 test normalisation on S1–S10 with separate calibration repetitions.

## Iteration log

| # | Date | What changed & why (one line) | Primary metric **with spread** | Better than previous? | If not — why it was kept | Commit |
|---|---|---|---|---|---|---|
| 1 | 2026-10-04 | Supplied baseline on real S1–S5, E1; establish the baseline. Synthetic smoke test passed. | Within-subject: mean 0.790 (SD 0.054, range 0.743–0.880). New-subject: mean 0.161 (SD 0.043, range 0.097–0.210). Across 5 subjects. | — (baseline) | — | Base `d910d38`; changes uncommitted |
| 2 | 2026-10-15 | Repeat no-extra-smoothing baseline on real S1–S5, E1, in the same environment as the smoothing trials. | Within-subject (repetition split): mean 0.788 (SD 0.054, range 0.741–0.881); New-subject (subject split): mean 0.159 (SD 0.045, range 0.092–0.216). Across 5 subjects. | Reference for iterations 3–4; earlier baseline retained above. | Unchanged default; matched reference for this experiment. | Base `d5159c4`; changes uncommitted |
| 3 | 2026-10-15 | 3-sample causal moving average (30 ms), per channel and per 200 ms window; test whether steadier envelopes help. | Within-subject (repetition split): mean 0.785 (SD 0.050, range 0.746–0.873); New-subject (subject split): mean 0.160 (SD 0.042, range 0.097–0.210). Across 5 subjects. | Vs matched baseline #2: within -0.003; new +0.001. No clear gain relative to subject variation. | Logged as a comparison; not adopted as the default. | Base `d5159c4`; changes uncommitted |
| 4 | 2026-10-15 | 5-sample causal moving average (50 ms), per channel and per 200 ms window; test whether steadier envelopes help. | Within-subject (repetition split): mean 0.782 (SD 0.050, range 0.743–0.869); New-subject (subject split): mean 0.163 (SD 0.047, range 0.092–0.217). Across 5 subjects. | Vs matched baseline #2: within -0.006; new +0.004. No clear gain relative to subject variation. | Logged as a comparison; not adopted as the default. | Base `d5159c4`; changes uncommitted |
| 5 | 2026-10-15 | Rerun the unchanged baseline through the revised notebook track with preprocess="none"; matched reference for helper trials. | Within-subject (repetition split): mean 0.788 (SD 0.054, range 0.741–0.881); New-subject (subject split): mean 0.159 (SD 0.045, range 0.092–0.216). Across 5 subjects. | Matches baseline #2; reference for iterations 6–7. | No extra preprocessing remains the default. | Base `0dd3e88`; changes uncommitted |
| 6 | 2026-10-15 | Use adapter.denoise from preprocess(): broadband="movavg", win=3 (30 ms); centred, zero-padded, per channel/window; offline. | Within-subject (repetition split): mean 0.788 (SD 0.052, range 0.743–0.877); New-subject (subject split): mean 0.157 (SD 0.044, range 0.095–0.210). Across 5 subjects. | Vs matched baseline #5: within +0.000; new -0.002. No clear gain relative to subject variation. | Retained in the experiment history; not adopted as the default. | Base `0dd3e88`; changes uncommitted |
| 7 | 2026-10-15 | Use adapter.denoise from preprocess(): broadband="movavg", win=5 (50 ms); centred, zero-padded, per channel/window; offline. | Within-subject (repetition split): mean 0.785 (SD 0.051, range 0.742–0.872); New-subject (subject split): mean 0.156 (SD 0.047, range 0.090–0.210). Across 5 subjects. | Vs matched baseline #5: within -0.002; new -0.003. No clear gain relative to subject variation. | Retained in the experiment history; not adopted as the default. | Base `0dd3e88`; changes uncommitted |
| 8 | 2026-11-04 | Expand development to real S1–S22, E1; rerun the unchanged baseline as the reference for this population. | Within-subject (repetition split): mean 0.795 (SD 0.051, range 0.697–0.881); New-subject (5-fold subject GroupKFold): mean 0.176 (SD 0.055, range 0.075–0.266). Across 22 subjects. | Reference for iterations 9–10; earlier S1–S5 scores are not matched comparisons. | Current default: no extra preprocessing. | Base `3456d56`; changes uncommitted |
| 9 | 2026-11-04 | Extra envelope smoothing using adapter.denoise: broadband="movavg", win=5, on real S1–S22, E1; offline. | Within-subject (repetition split): mean 0.790 (SD 0.049, range 0.695–0.872); New-subject (5-fold subject GroupKFold): mean 0.173 (SD 0.051, range 0.079–0.259). Across 22 subjects. | Vs matched baseline #8: within -0.004; new -0.003. No clear overall gain relative to subject variation. | Logged and kept as an available option; baseline remains the default. | Base `3456d56`; changes uncommitted |
| 10 | 2026-11-04 | Extra envelope smoothing using adapter.denoise: broadband="savgol", win=5, savgol_order=2, on real S1–S22, E1; offline. | Within-subject (repetition split): mean 0.792 (SD 0.051, range 0.694–0.880); New-subject (5-fold subject GroupKFold): mean 0.179 (SD 0.053, range 0.076–0.266). Across 22 subjects. | Vs matched baseline #8: within -0.002; new +0.003. No clear overall gain relative to subject variation. | Logged and kept as an available option; baseline remains the default. | Base `3456d56`; changes uncommitted |
| 11 | 2026-11-04 | 100 ms windows; no extra smoothing; S1–S22, E1; shared 300 ms slices cropped to the same endings; 150 ms step. | Within-subject (repetition split): mean 0.775 (SD 0.052, range 0.680–0.866); New-subject (5-fold subject GroupKFold): mean 0.181 (SD 0.054, range 0.057–0.261). Across 22 subjects, 58,446 shared windows. | Vs matched 200 ms reference #12: within -0.022; new -0.004. | Not adopted: both scores fell. | Base `3456d56`; changes uncommitted |
| 12 | 2026-11-04 | 200 ms windows; no extra smoothing; S1–S22, E1; shared 300 ms slices cropped to the same endings; 150 ms step. | Within-subject (repetition split): mean 0.798 (SD 0.049, range 0.706–0.881); New-subject (5-fold subject GroupKFold): mean 0.184 (SD 0.053, range 0.069–0.262). Across 22 subjects, 58,446 shared windows. | Matched reference for this window-length test; do not compare directly with iteration 8, which used a different set of windows. | Keep 200 ms as the default. | Base `3456d56`; changes uncommitted |
| 13 | 2026-11-04 | 300 ms windows; no extra smoothing; S1–S22, E1; shared 300 ms slices cropped to the same endings; 150 ms step. | Within-subject (repetition split): mean 0.810 (SD 0.047, range 0.710–0.895); New-subject (5-fold subject GroupKFold): mean 0.188 (SD 0.060, range 0.057–0.273). Across 22 subjects, 58,446 shared windows. | Vs matched 200 ms reference #12: within +0.013; new +0.003. | Highest observed means, but the new-subject gain is small and response to gesture changes was not tested; keep as an option. | Base `3456d56`; changes uncommitted |
| 14 | 2026-11-10 | Matched baseline: S1–S10, E1; repetitions 3–10 only; no normalisation; 200 ms windows, 150 ms step. | Within-subject (repetition split): mean 0.812 (SD 0.038, range 0.762–0.869); New-subject (10-fold LOSO): mean 0.156 (SD 0.070, range 0.055–0.270). Across 10 subjects, 21,802 scoring windows. | Matched reference for iteration 15; earlier full-repetition/other-population results are not direct comparisons. | Keep as the reference for this calibration protocol. | Base `3456d56`; changes uncommitted |
| 15 | 2026-11-10 | Per-subject, per-channel RMS normalisation; scale from separate repetitions 1–2; classify repetitions 3–10; S1–S10, E1. | Within-subject (repetition split): mean 0.812 (SD 0.038, range 0.762–0.869); New-subject (10-fold LOSO; RMS uses separate target calibration): mean 0.188 (SD 0.068, range 0.074–0.293). Across 10 subjects, 21,802 scoring windows. | Vs matched baseline #14: within +0.000; new +0.032. | Promising for calibrated use: improved 8 of 10 new subjects; check on S1–S22 before changing the default. | Base `3456d56`; changes uncommitted |

## Decision log — the choices behind the numbers

| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| Data loading | DB1, E1, S1–S5 | All 27 subjects | Keep the supplied subset for the baseline. | 1 | — |
| Classification, incl. imbalance | Supplied random forest; seed 0; `imbalance="balanced"` | `"none"` | Keep the default weighting for less frequent gestures. | 1 | — |
| Feature selection | none (all 50 features) | anova k=25, lasso C=0.1 | Neither improved either mode: anova lowered new-subject 0.161 → 0.132 (larger spread), lasso kept 46–50 features and matched none exactly. | 1 | — |
| Preprocessing (Stage 2) | No extra smoothing remains the default | 3-sample and 5-sample current/past moving averages | The small changes in new-subject mean do not establish a useful gain relative to subject variation; both smoothed settings slightly reduce within-subject mean. | 2–4 | Smoothing hypothesis tested; neither option adopted. |
| Preprocessing implementation (Stage 2 revision) | Use the supplied adapter.denoise helper for the comparison; keep no extra preprocessing as the default | Previous custom causal average; helper movavg with win=3 and win=5 | The user requested the scaffold helper; both widths were declared before rerunning, and neither shows a clear benefit. Report the centred/zero-padding behaviour as an offline experiment. | 5–7 | Implementation revised at user request; earlier results preserved in iterations 2–4. |
| Development data | S1–S22, E1; S23–S27 reserved | Earlier S1–S5 smoke subset | Use all non-hold-out subjects for development and rerun the baseline on the same population as the alternatives. | 8–10 | Expanded at user request. |
| Extra envelope smoothing | No extra smoothing remains the default | Built-in moving average (5 samples); Savitzky–Golay (5 samples, order 2) | Moving average slightly lowers both means; the small Savitzky–Golay cross-subject increase trades off a within-subject decrease. Keep both comparisons and their padding/overshoot limitations. | 8–10 | Two different built-in smoothing approaches tested; neither adopted. |
| Window length | Keep 200 ms as the default | 100 ms, 200 ms, 300 ms; fixed 150 ms step, no extra smoothing | 300 ms had the highest means, but its cross-subject gain was small and gesture-change response was not tested. 100 ms lowered both scores. All lengths used the same signal endings. | 11–13 | Window length tested; 300 ms remains an option. |
| Per-channel normalisation | Keep RMS as a candidate; default unchanged pending the S1–S22 check | No normalisation vs channel RMS from separate repetitions 1–2 | New-subject mean rose by 0.032 and 8 of 10 subjects improved; within-subject predictions were identical. RMS requires calibration from each new subject. | 14–15 | New ten-subject calibration experiment. |

**Reproduce:** restart the kernel after the loader change. Run setup and S1–S22 data loading, then the **Stage 2: window length** cells. The saved window test uses `window_subject_count = 22`; set it to 10 for a future quick test and rerun every length on the same ten subjects. The smoothing section is a separate earlier comparison.

## Stage 2 smoothing comparison — declared before running, 2026-10-15

- **Comparison budget:** three settings, `smooth_samples=1` (unchanged baseline), `3` (30 ms), and `5` (50 ms). Two new smoothing configurations; the earlier feature-selection comparisons above remain part of the project history.
- **Fixed:** real S1–S5, E1; 200 ms windows / 150 ms step; all 50 features; supplied random forest; seed 0; balanced class weights; both existing evaluation modes and the same folds.
- **Implementation:** current/past-sample moving average, separately per channel and per window. Use available samples at window start; reset each window. No fitted preprocessing statistics. Evaluate all settings on the same unchanged recordings.
- **Validation approach:** the assignment's limited comparison budget. No separate DEV or new final test set is created. These reused-fold scores are exploratory; a budget does not remove selection bias. Leave any final hold-out data unused until the design is fixed.
- **Decision rule:** record all three results with subject spread. Do not expand the smoothing search or automatically choose the largest score. Differences small relative to subject variation are inconclusive; retain the baseline unless the evidence justifies a change.

### Observed result and decision

The three declared runs are recorded as iterations 2–4 above. Both smoothing settings reduce within-subject mean slightly. The largest new-subject increase is +0.004; this is small relative to subject variation. Keep the original envelope as the default and stop this smoothing comparison at the declared budget. These are exploratory reused-fold results, with no untouched final test evaluated.

**Historical run:** iterations 2–4 used the custom `smooth_samples` implementation: at each sample, average the current and up to `size - 1` preceding samples available within that window, without zero padding. Local environment: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1. All three settings used identical labels and evaluation folds, with iteration 2 as the matched baseline. The notebook now reproduces the helper-based revision below; these older scores are retained as history, not presented as results of `adapter.denoise()`.

## Stage 2 revision — use `adapter.denoise()`, declared before rerun, 2026-10-15

- **Reason for revision:** user requested the supplied Chapter 8 helper instead of the custom past-only implementation. The existing scores were not used to choose new window sizes.
- **Comparison budget:** no preprocessing (`preprocess="none"`), then `preprocess="denoise", broadband="movavg"` with `win=3` and `win=5`. Two additional helper-based configurations. The earlier custom-filter trials (iterations 2–4) and feature-selection trials remain part of the history; this is not a new three-trial project total.
- **Fixed:** real S1–S5, E1; 13,875 windows; 200 ms windows / 150 ms step; all 50 features; supplied random forest, seed 0, balanced class weights; both existing evaluation modes and identical folds.
- **Implementation:** notebook `DenoiseEMGTrack.preprocess()` calls `adapter.denoise()`. Register only `preprocess`, `broadband`, and `win`. Move time to the last axis before the helper call and restore the EMG layout afterwards. No raw-EMG filtering or other unmotivated denoising stages are enabled.
- **Regime and edge behaviour:** centred moving average, zero padding outside each window; offline and non-causal. These outputs differ from the previous past-only filter and may attenuate the edges. No real-time deployment claim is made.
- **Evaluation:** rerun all three settings, report per-subject mean/SD/range in both modes, preserve old results, and stop at the declared settings. No learned preprocessing statistics, no extra development set, and no final hold-out data used. Reused-fold results remain exploratory. Keep the baseline unless the observed benefit justifies a change relative to subject variation.

### Helper-based result and decision

Iterations 5–7 record the rerun using `adapter.denoise()`. The 3-sample setting leaves within-subject mean essentially unchanged, while the 5-sample setting slightly lowers it. Both slightly lower new-subject mean. No extra preprocessing remains the default. The changes are small relative to subject variation; no additional window widths were searched.

The helper smooths the interior but attenuates the window edges because of zero padding. Changes in the extracted features include this boundary effect. These are offline, centred-filter results; they do not validate a real-time controller or its latency. Earlier causal runs remain part of the project history, and all reused-fold comparisons remain exploratory.

**Historical helper comparison (iterations 5–7):** set `USE_REAL = True`, run setup and data loading, then the three code cells under **Stage 2: a small smoothing experiment using adapter.denoise()**. Baseline config: `{"preprocess": "none"}`. Helper configs: `{"preprocess": "denoise", "broadband": "movavg", "win": 3}` and the same with `win=5`. All use seed 0, no feature selection, balanced class weights, and the existing two-mode evaluation. The original `track` and the supplied shared helper are unchanged; `denoise_track` is the notebook's implementation. Environment: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1.

## S1–S22 extra envelope smoothing — declared before running, 2026-11-04

- **Development pool:** S1–S22, E1. S23–S27 remain separate final hold-out subjects. `USE_REAL=True` and `download(..., subset=range(1, 23))` are now explicit in the notebook.
- **Exactly three configurations:** baseline `preprocess="none"`; moving average `preprocess="denoise", broadband="movavg", win=5`; Savitzky–Golay `preprocess="denoise", broadband="savgol", win=5, savgol_order=2`. Both smoothers are built into `adapter.denoise()`. No parameter sweep, clipping, or other noise-removal stages are included.
- **Rationale:** compare simple averaging with a quadratic local smoother that aims to preserve peak shape. DB1 is already an envelope, and fluctuations are not assumed to be noise.
- **Fixed:** 200 ms windows / 150 ms step; all 50 features; supplied random forest; seed 0; balanced class weights. With 22 subjects the existing new-subject harness uses 5-fold GroupKFold by subject. Within-subject evaluation holds out repetitions for each subject. Rebuild features per configuration, and report both modes with per-subject mean/SD/range.
- **Implementation and regime:** notebook `DenoiseEMGTrack.preprocess()` calls the supplied helper after moving time to the last axis, then restores the EMG layout. Only consumed config keys are declared. Both methods are centred/offline; moving average zero-pads while Savitzky–Golay uses nearest padding and may overshoot below zero. Report their actual outputs without clipping. No learned preprocessing statistics or controller latency claims.
- **History and comparison rule:** earlier S1–S5 trials (iterations 1–7) remain history; scores on different populations/splits are not evidence of a smoothing gain. Compare all three methods against the new S1–S22 baseline, preserve negative results, and keep the baseline unless a useful benefit is justified relative to subject variation. Stop at these three configurations; this does not reset the cumulative trial history.
- **Status:** completed on 22 subjects and 60,182 windows. All 345 model fits completed (115 per configuration), with identical labels/folds across settings. Old S1–S5 numerical outputs were cleared from the notebook, with all history retained above. No final hold-out data were downloaded or used.

### S1–S22 result and decision

Iterations 8–10 record the new population and its matched comparison. Baseline: within-subject mean 0.795 (SD 0.051), new-subject mean 0.176 (SD 0.055). Moving average: 0.790 (SD 0.049) / 0.173 (SD 0.051). Savitzky–Golay: 0.792 (SD 0.051) / 0.179 (SD 0.053). Each spread is across 22 subjects; full ranges and split definitions are in the iteration rows.

No extra smoothing remains the default. Savitzky–Golay's small cross-subject increase trades off a within-subject decrease, so the observed results do not establish a clear overall benefit. Moving average slightly lowers both means. Both methods remain configured in the notebook; no additional parameters were searched.

The moving average's zero-padding attenuates window edges. Savitzky–Golay uses nearest padding and produced 0.0079% negative channel samples across the overlapping windows; outputs were not clipped. The plotted example stays on S1 explicitly, since alphabetical filename order puts S10 before S1 once more subjects are loaded. Dataset/model row ordering was not changed for the plots.

Built-in use is configured in `smoothing_configs` under **Stage 2: try built-in extra envelope smoothing**. `DenoiseEMGTrack.preprocess()` forwards `broadband`, `win`, and `savgol_order` to `adapter.denoise()` after the time-axis conversion. All four consumed keys, including `preprocess`, are registered. These remain exploratory offline results; final hold-out evaluation is reserved for after the design is frozen.

## Window length test — 2026-11-04

- Compare 100, 200, and 300 ms on S1–S22, E1. Keep the 150 ms step, no extra smoothing, all 50 features, random forest, seed 0, and balanced class weights fixed.
- Load valid 300 ms slices and take the last 10, 20, or 30 samples. This gives all lengths the same labels, repetition groups, and prediction times. The shared set excludes windows near boundaries that would be valid only at shorter lengths; rerun 200 ms as the matched reference instead of reusing iteration 8.
- A 100 ms window with a 150 ms step leaves 50 ms gaps. The decision interval stays 150 ms. This test measures classification scores, not controller latency.
- Use the existing two evaluation modes: 5-fold GroupKFold by subject and repetition-held-out folds within each subject. Report mean/SD/range across 22 subjects. S23–S27 stay out.
- These three lengths were chosen before running. Log every result and stop at these lengths. Earlier comparisons remain in the history.
- Status: completed all three lengths on 22 subjects and 58,446 shared windows. All 345 fits completed. Labels and folds matched across lengths. Tests confirmed unchanged default 200 ms loading and no windows crossing gesture/repetition boundaries.

### Window length results

Iterations 11–13 contain this test. 300 ms scored highest: within-subject mean 0.810 (SD 0.047), new-subject mean 0.188 (SD 0.060). The matched 200 ms reference gave 0.798 (SD 0.049) and 0.184 (SD 0.053). At 100 ms the scores fell to 0.775 (SD 0.052) and 0.181 (SD 0.054). Ranges are in the iteration rows.

Keep 200 ms as the default for now. The 300 ms result is promising for the steady-gesture windows tested here, but the cross-subject gain is small and response to a gesture change was not measured. The comparison uses only slices valid for 300 ms, so it leaves out some onset/boundary windows that shorter lengths could accept. It is not a test of onset response.

The notebook now has `window_subject_count`: leave it at 22 to reproduce these results, or set it to 10 for future quick comparisons on S1–S10. No ten-subject window test was run here. Recheck a chosen setup on all S1–S22 before final hold-out prediction. S23–S27 remain unused.

## Per-channel normalisation test — 2026-11-10

- Compare no normalisation with per-subject, per-channel RMS division on S1–S10, E1. Keep 200 ms windows, 150 ms step, no extra smoothing, all 50 features, the supplied random forest, seed 0, and balanced class weights.
- Reserve repetitions 1–2 of each subject as a separate calibration set. Calculate one RMS per channel across its calibration windows and samples, without using gesture labels. A zero RMS uses scale 1. Apply these fixed scales in `NormalisedEMGTrack.preprocess()` before feature extraction.
- Use only repetitions 3–10 for classifier training and scoring, for both baseline and RMS. Reuse the supplied evaluator: leave-one-subject-out over ten subjects, plus five repetition-grouped folds within each subject. Calibration windows never enter classifier training, scoring, or feature selection.
- The RMS new-subject result assumes unlabelled calibration recordings from the new subject before use. Repetitions 1–2 cover the exercise rather than one chronological opening segment. This is a calibrated protocol, not zero-shot prediction. Do not compare these scores directly with older runs on different subjects/repetitions.
- Exactly two settings; keep both results. This is a quick development comparison; S23–S27 remain unused. Check any promising change on S1–S22 before choosing the final design.
- Status: completed all 120 model fits (60 per setting: 10 new-subject LOSO folds plus 50 within-subject repetition folds). Both settings used identical labels, scoring windows, subject groups, and fold assignments. Three tests passed, including a check that changing scored signals/labels cannot change calibration values. Raw recordings were preserved and all features were finite.
- Calibration: 5,773 windows; scoring/training pool: 21,802 windows. RMS was calculated across the overlapping calibration windows as loaded, so overlapping samples contribute more than once. No channels had zero calibration RMS in this run.
- Environment: Python 3.12.1, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1. Settings and per-subject results: `outputs/emg_inspection/channel_normalisation_results.json`.

### Normalisation result and decision

The matched baseline gave within-subject macro-F1 0.812 ± 0.038 and new-subject 0.156 ± 0.070. Channel RMS normalisation gave 0.812 ± 0.038 and 0.188 ± 0.068, respectively (mean ± SD across ten subjects). New-subject scores improved for eight subjects and fell for S4 and S8. Within-subject predictions were identical.

Keep RMS as a promising option for a calibrated setup, and check it on S1–S22 before changing the default. The +0.032 gain is from this matched ten-subject test, not a comparison with earlier 22-subject results. The test does not assess performance without new-user calibration.

To reproduce: run the notebook setup and real-data loading cells, then **Stage 2: per-channel normalisation**. The section selects S1–S10 explicitly and leaves the other experiments separate. `normalisation_settings` contains exactly the baseline and RMS choices. Calibration scales are fixed from the separate repetitions before any folds; no scored window contributes to them.

## Who did what

| Iteration | Who | Modules / tasks owned | Reviewed by |
|---|---|---|---|
| 1 | Marissa | Real-data setup, signal inspection, baseline validation | |
| 1 | Barnus | Recording-selection fix, evaluation and smoke-test checks | |
| 1 | Haripriya | Baseline feature check (shape, finite values, stats, flat windows, redundancy, subject scale), feature documentation, fold-safe selection A/B | |
