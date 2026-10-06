---
title: First Live Sensor Run - Runner Crash, Sample Drops and False AF on Device
date: 2026-10-06
reviewer: Claude (Opus 5.5)
component: firmware (ArrhythmiaNode.ino), edge runner, ibi_af_v1 on real MAX30102 data
status: open — two defects fixed; false-AF finding unresolved
---

# First Live Sensor Run — Runner Crash, Sample Drops and False AF on Device

First end-to-end run of `ibi_af_v1` on the project's own MAX30102 (fingertip, bare
sensor, no housing), on the Pi, 2026-10-06.

## Wiring found wrong (fixed by the team, hardware)

Both the MAX30102 and the SSD1306 OLED were wired with SDA/SCL swapped. A pin-pair I²C
scan found the sensor at `0x57` only on SDA=9/SCL=8. After the swap both answer on
SDA=8/SCL=9 (`0x57`, `0x3C`). This is the first time the OLED has ever been live on the bus.

## Defect 1 — firmware dropped ~10% of samples once the OLED worked (fixed)

`loop()` read one FIFO sample per 10 ms `micros()` tick. A full SSD1306 redraw at
400 kHz takes ~23 ms, five times a second, so ticks were skipped, the 32-deep MAX30102
FIFO overflowed, and the Pi would receive ~90 samples/s while assuming 100 — shortening
every inter-beat interval `ibi_af_v1` consumes. It never showed before because with no
OLED on the bus each redraw failed fast.

**Fix:** drain the FIFO every pass; the sensor's own clock is the sample clock.
**Verified on hardware:** 100.1 Hz by MCU timestamps over 1,881 frames, 0 CRC rejects.

Same change replaced the OLED heuristic (fixed threshold 150, 4-beat mean) with a
band-pass + adaptive-threshold + parabolic-interpolation + 8-beat-median pulse estimate,
and **removed the OLED's `STATUS: NSR (NORMAL)` / `IRREGULAR RHYTHM` verdict** — the node
has no validated rhythm model; the only rhythm decision is `ibi_af_v1` on the Pi.
Wire protocol unchanged (19 bytes). `firmware/src/main.cpp` (PlatformIO) was not updated.

## Defect 2 — runner crashed on the first real window; would have classified empty features (fixed)

`IBIAFClassifier.explain_intervals()` returned no `top_intervals` key when a window had
fewer than 3 usable intervals, and `runner.py` always read it → `KeyError`, service
crash-loop on every contacted window with few beats. Underneath that, the runner
classified the **all-zero** feature vector that `ibi_features_from_peaks` returns for such
windows, although its docstring says zeros mean "insufficient quality, not non-AF".

**Fix:** the runner now publishes these as `measurement_valid: false`,
`invalid_reason: "insufficient_beats"` (no probability, no ledger event); the
classifier's short-window branch returns the full key set; the dashboard has a message
for the new reason. Regression tests added to `test_runner_weights.py` and
`test_ibi_classifier.py` — the runner test fails on the old code.

## Finding — `ibi_af_v1` flags a resting non-AF fingertip as AF (OPEN)

With the chain running, most valid windows scored **P(AF) 62–98%, AF=1**, on a team
member at rest. Reported BPM swung 36.7–87.5 within seconds, while the node's own
estimator read a steady 80–86 bpm on the same finger. Sample rate was confirmed at
100 Hz (one window per 100 frames per second).

A 25 s raw capture (runner paused with SIGSTOP, not restarted) showed:

| Finger-on run | Observation | Pi peak detector |
|---|---|---|
| 8.0–16.7 s | slope ~100x the pulse: contact settling/motion | 3 beats, IBIs 5.6 s / 1.9 s |
| 17.1–25.1 s | pulse slope ~30 counts on ~100k DC — very low perfusion signal | IBIs 300–1140 ms |

**Signal polarity was tested and ruled out** as the cause: inverted and as-is give
equally chaotic intervals.

Mechanism: weak pulse + unsteady contact → missed and spurious beats → irregular IBI
series → an irregularity model reads it as AF. The current SQI gate passes most of these
windows (`SQI=1.0`). This is the reflectance-vs-transmissive gap the project already
lists as its top blocker, now observed rather than hypothesised. Clean data is
achievable on this hardware: a steady 18 s stretch earlier the same day gave IBIs
665–785 ms.

**Recommended, in order:**
1. Steady the contact (strap/clip, ambient-light shield) — the unbuilt fingertip housing.
2. Morphology-based SQI (beat-template correlation) to reject artefact windows **without**
   rejecting genuinely irregular rhythm. A rule like "reject irregular IBIs" would also
   reject true AF and must not be used.
3. Do **not** raise τ to suppress this — it trades away real-AF sensitivity silently.
4. Collect labelled data on this device; that is the only measurement of real performance.

## Data integrity consequence

During this run the live service wrote false-positive AF events to the hash-chained
ledger under `PAT-CAL-001` (a fabricated patient). They are artefacts of an unvalidated
sensor path and must not be read as findings. The chain is append-only; cleaning it
means re-initialising from genesis (see the deployment runbook §6).

---
Back to [[Index|Code Review Index]]
