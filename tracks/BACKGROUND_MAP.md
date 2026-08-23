# Background & literature map — capstone tracks

This map tells each project team **where to refresh the method** and **what to read** before
they build.

Two important framings:

1. **The textbook is background you already have.** Every section referenced below was covered
   in the course. The book references are a *refresher / cross-reference* — "the method you need
   is in §X" — **not** new material to learn from scratch. If a team is fluent, they can skip
   straight to the code; the pointers are there when a step feels unfamiliar.
2. **Each project also requires a short literature review of its application domain.** The book
   teaches the *methods* generically; it does not teach the *clinical/application problem* (why AF
   matters, how FIGO defines a deceleration, what accuracy the field considers good). That gap is
   the point of the literature review — and it is what lets you **motivate** your design, which the
   course learning outcome L5 ("design, **motivate**, implement, and evaluate a method") explicitly
   requires. The syllabus's ethical-approach clause also requires you to disclose sources.

---

## How to do the literature review (all tracks)

Aim for **~5–8 sources**, skimmed for relevance, not read cover to cover. Start from the three
anchors listed per track (the dataset's own paper, the governing clinical/technical standard, and
one methods/benchmark paper), then branch out.

**Where to search:** PubMed, IEEE Xplore, Google Scholar; the dataset's PhysioNet/UCI page and its
"cited by" list; recent survey/review papers (fastest way into a field).

**What to extract (put this in your report's Background section):**

- The **clinical/application problem** and why it matters (who is affected, what decision the signal informs).
- The **operational definition of the target** — how clinicians/standards define the thing you are classifying (e.g. what *is* a deceleration; what *is* AF on an ECG).
- **State of the art**: what methods are used and roughly what performance (accuracy / F1 / κ / sensitivity) is considered good — so your result has a yardstick.
- **Known pitfalls / failure modes**: label noise, inter-rater disagreement, class imbalance, domain shift.
- **How your approach compares** — one or two sentences positioning your pipeline against the above.

**Cite properly** (any consistent style). Full citations for the dataset are in each track's
`*_card.md`.

---

## Sleep staging — `sleep_edf.py`

**Signal / task:** multi-channel PSG (EEG, EOG, EMG) → 5-class sleep stage per 30-s epoch.

**Course background (refresher):** EEG/EOG/EMG signatures §1.4 · sampling & epoching Ch2 ·
spectrogram & wavelets for spindles/K-complexes §3.9, §5.3–5.4 · band power via Welch/multitaper
§7.5–7.6 · EOG/EMG cross-contamination §8.9 (ICA §10.7, optional) · the sleep-stage feature set
**§11.4** · epochs→hypnogram + Cohen's κ §12.6 · the whole worked system Ch16.

**Literature review anchors:**

- *Dataset:* Kemp et al. (2000), Sleep-EDF; Goldberger et al. (2000), PhysioNet.
- *Standard:* AASM Manual for the Scoring of Sleep and Associated Events (Berry et al.) — the scoring rules; Rechtschaffen & Kales (1968) for the R&K legacy the Sleep-EDF labels use.
- *Methods/benchmark:* e.g. Supratak et al. (2017), DeepSleepNet; Perslev et al. (2021), U-Sleep.
- *Search terms:* "automatic sleep stage classification EEG", "Sleep-EDF benchmark", "sleep spindle detection".

**Guiding questions:** What AASM rules separate N2 from N3? Which channel/feature carries REM vs
Wake? Why is N1 the hardest stage, and what inter-rater agreement (κ ≈ 0.76) do human scorers
themselves reach? How do published methods split train/test (subject-wise — and why)?

---

## ECG rhythm / AF — `ecg_cinc2017.py`

**Signal / task:** single-lead ECG (300 Hz) → 4-class rhythm (Normal / AF / Other / Noisy).

**Course background (refresher):** ECG/QRS morphology §1.4 · band-pass + 50 Hz notch **§9.6** ·
**Pan–Tompkins QRS detection §9.9** · R–R → HRV (SDNN/RMSSD/LF–HF) **§7.11** · rhythm regularity via
autocorrelation §6.3 · signal-quality index for the Noisy class §8.2 (QC) + §8.6 (powerline) ·
validation & metrics Ch12.

**Literature review anchors:**

- *Dataset:* Clifford et al. (2017), AF classification from a short single-lead ECG — PhysioNet/CinC Challenge 2017.
- *Standards:* Pan & Tompkins (1985), *IEEE TBME* 32:230–236 (QRS detection); HRV Task Force (1996), *Circulation* 93:1043–1065 (HRV measures).
- *Methods:* the top-ranked CinC-2017 entries and later single-lead AF reviews.
- *Search terms:* "atrial fibrillation detection single lead ECG", "RR irregularity AF", "ECG signal quality index".

**Guiding questions:** What ECG signature defines AF (absent P-waves, irregularly-irregular R–R)?
Why is the "Noisy" class as important as the rhythm classes here? What macro-F1 did the challenge
winners reach (~0.83) — how far is your baseline from that, and why? How is a recording labelled
"too noisy to interpret"?

---

## CTG / fetal — `ctg_ctu_uhb.py`

**Signal / task:** fetal heart rate + uterine contraction (toco), 4 Hz → binary normal /
pathological (from cord-blood pH).

**Course background (refresher):** dropout removal + gap interpolation + missing-at-random
**§12.9–12.11** · FHR baseline by moving-average/median smoothing §9.7 + §10.2–10.3 (baseline-wander
concept §8.8) · variability (STV/LTV) as windowed variance features §6.2 + §11.3 ·
**deceleration/acceleration detection** — compose from the Pan–Tompkins event-detection template
§9.9 (filter → transform → threshold → duration rule) plus the "subtract an averaged signal ⇒
high-pass" duality §3.12 / §9.8 to get deviations from the FHR baseline · FHR–contraction coupling
via cross-correlation §6.4 · imbalance-aware metrics §12.5.

**Literature review anchors:**

- *Dataset:* Chudáček et al. (2014), CTU-UHB Intrapartum CTG database.
- *Standard:* **Ayres-de-Campos et al. (2015), FIGO consensus guidelines on intrapartum fetal monitoring: Cardiotocography, *Int J Gynecol Obstet* 131:13–24** — the definitions of baseline, variability, accelerations, and deceleration *types* (early / late / variable / prolonged).
- *Methods:* reviews of automated CTG analysis / FHR classification.
- *Search terms:* "cardiotocography deceleration classification", "fetal heart rate variability pH outcome", "CTG interobserver agreement".

**Guiding questions:** How does FIGO define a deceleration, and how do early / late / variable
decelerations differ? What umbilical-artery pH threshold defines "pathological", and how noisy is
that label? Why is interobserver agreement on CTG notoriously poor, and what does that imply for
your achievable accuracy? Why is high *sensitivity* hard here (and why does that matter clinically)?

---

## EMG gesture — `emg_ninapro.py`

**Signal / task:** 10-channel surface-EMG envelope (100 Hz) → 12-class hand gesture; two eval
modes (within-subject vs new-subject).

**Course background (refresher):** EMG signature §1.4 · rectified/analytic envelope §5.5 (+ the
EMG-envelope example §5.4) · time-domain features (MAV, RMS, waveform length, variance) §11.3 ·
mean/median frequency Ch7 · within- vs cross-subject = domain shift §12.3, §12.13 + external
validation §12.8 · classifiers Ch13.

**Literature review anchors:**

- *Dataset:* Atzori et al. (2014), *Sci Data* — Ninapro DB1.
- *Methods:* Hudgins et al. (1993), the classic time-domain feature set for myoelectric control; Phinyomark et al. (2012), EMG feature selection.
- *Search terms:* "sEMG gesture recognition features", "myoelectric control", "cross-subject EMG electrode shift".

**Guiding questions:** Which time-domain features are standard (the Hudgins set) and why do they
work on EMG? Why does within-subject accuracy (~0.78) collapse across subjects (~0.19)? What is
"electrode shift" and how does the field mitigate it (per-user calibration, domain adaptation)?
What accuracy is typical on Ninapro DB1?

---

## BCI motor imagery — `bci_eegmmidb.py`

**Signal / task:** 64-channel EEG (160 Hz) → binary left/right-hand motor imagery; two eval modes.

**Course background (refresher):** EEG §1.4 · the SSVEP-BCI averaging worked example **§10.4** ·
band-pass to mu/beta §9.6 · ERD as band-power decrease §7.5 · STFT view §3.9 · spatial unmixing
(PCA/ICA) as the on-ramp to CSP §10.7 · two eval modes §12.3, §12.13.

**Literature review anchors:**

- *Dataset:* Schalk et al. (2004), BCI2000 — the EEG Motor Movement/Imagery database.
- *Methods:* Pfurtscheller & Lopes da Silva (1999), *Clin Neurophysiol* 110:1842–1857 (ERD/ERS); Ramoser et al. (2000) / Blankertz et al. (2008) on Common Spatial Patterns (CSP); Lotte et al. (2018) BCI classification review.
- *Search terms:* "motor imagery EEG classification CSP", "mu beta ERD C3 C4", "subject-independent BCI".

**Guiding questions:** What is event-related desynchronization in the mu/beta bands, and over which
electrodes (C3/C4) does hand imagery show it? Why does a naive band-power baseline barely beat
chance (~0.51)? What does CSP do that fixed C3/C4 features cannot? Why is cross-subject BCI so hard
(EEG non-stationarity / individual anatomy)?

---

## Human activity / IMU — `har.py`

**Signal / task:** tri-axial accelerometer → 3-class activity (static / walk / stairs). The gentle
on-ramp track.

**Course background (refresher):** sampling Ch2 · gravity-vs-motion separation (low-pass) §9.2–9.4,
§9.7 · cadence = dominant frequency §3.5 (FFT) + §7.5 (Welch) · windowing Ch4 · features Ch11 ·
classifiers Ch13.

**Literature review anchors:**

- *Dataset:* Anguita et al. (2013), UCI Human Activity Recognition Using Smartphones.
- *Reviews:* Lara & Labrador (2013), survey on HAR with wearable sensors; Bulling et al. (2014), tutorial on HAR with body-worn inertial sensors.
- *Search terms:* "human activity recognition accelerometer features", "gravity separation IMU low-pass", "sliding window HAR overlap".

**Guiding questions:** How is gravity separated from body acceleration (and why is that a filter
choice)? What window length / overlap is standard (UCI-HAR uses 2.56 s, 50 %)? Which activities are
most confusable (walking vs stairs) and what feature helps? Why must the split be subject-wise?

---

*The unifying thread worth noticing across tracks: §9.9 (Pan–Tompkins) is the generic template for
"detect a transient event in a physiological time series" — it reappears as QRS beats (ECG), FHR
decelerations (CTG), and EMG onsets. Learn it once, reuse it three ways.*
