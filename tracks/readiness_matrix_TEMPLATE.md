# Track-readiness matrix

Score each candidate track 1–3 (3 = easiest/best) before committing. This makes the scope
decision evidence-based, not vibes-based. A track is "ready" if it has raw signals (DSP=3),
low license friction (=3, agreement-free), and a clear split unit.

| Track | Raw signal | Download size | License friction | Runtime | Label quality | Split-unit clarity | DSP coverage | Total | Decision |
|-------|:----------:|:-------------:|:----------------:|:-------:|:-------------:|:------------------:|:------------:|:-----:|----------|
| Sleep-EDF | 3 | 2 | 3 | 2 | 2 | 3 | 3 | 18 | build 1st |
| UCI-HAR (raw) | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 20 | build 2nd |
| ECG MIT-BIH / CinC-2017 | 3 | 3 | 3 | 3 | 2 | 2 | 3 | 19 | strong |
| CTU-UHB (CTG) | 3 | 3 | 3 | 3 | 2 | 2 | 3 | 19 | strong |
| Ninapro (EMG) | 3 | 2 | 3 | 2 | 3 | 2 | 3 | 18 | good |
| EEGMMIDB (BCI) | 3 | 2 | 3 | 2 | 2 | 2 | 3 | 17 | advanced |
| `<candidate>` |  |  |  |  |  |  |  |  |  |

**Scoring guide.** *Raw signal*: 3 = raw waveforms/IMU shipped, 1 = only pre-computed features.
*License friction*: 3 = agreement-free direct download, 1 = DUA/EULA (excluded here).
*Split-unit clarity*: 3 = subject/patient IDs present, 1 = ambiguous. *DSP coverage*: 3 = exercises
Chapters 3–10 heavily, 1 = analytics-only (tabular).

**Reject if:** any agreement/credential is required, or DSP coverage = 1 (tabular-only).
