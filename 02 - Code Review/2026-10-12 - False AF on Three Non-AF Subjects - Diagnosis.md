---
title: False AF on Three Non-AF Subjects - Diagnosis
date: 2026-10-12
reviewer: Claude (Opus 5.5)
component: signal_processing/sqi.py, ibi_af_v1, firmware sensor config
status: open — one defect fixed (not yet deployed to the Pi); device data still needed
---

# False AF on Three Non-AF Subjects — Diagnosis

The team tested `ibi_af_v1` on the Pi + MAX30102 with three people who do not have AF.
All three were flagged as AF. Follows on from
[[2026-10-06 - First Live Sensor Run - Runner Crash, Sample Drops and False AF on Device]].
The Pi was unreachable during this session (no Tailscale/LAN route), so the three
subjects' raw data was **not** inspected. Everything below comes from offline tests of the
deployed code and weights.

## 0. Ledger evidence from the three sessions (pulled from the Pi later the same day)

The ledger has 200 AF events in three sessions a few minutes apart, one per tester. The
Pi clock reads 2026-10-05 because it has no RTC or NTP. Each event stores the top three
intervals that drove it:

| Session | AF events | BPM range | Driving IBIs | Driving IBIs >200 ms off window median |
|---|---|---|---|---|
| 1 | 118 | 52–107 | 0.34–2.10 s | 37% |
| 2 | 39 | 75–98 | 0.36–1.32 s | 29% |
| 3 | 43 | 50–94 | 0.31–2.06 s | 51% |

Intervals of 2.1 s (28 bpm) and BPM swinging 50 → 107 within minutes are not resting sinus
rhythm. Respiratory sinus arrhythmia moves intervals by ~50–70 ms, not >200 ms. **These
false positives are beat-detection failures (§2–3), not the model limitation in §4.** No
raw waveform is stored, so pulse amplitude per session cannot be recovered.

The SQI fix was copied to the Pi and checked there: weak drifting pulse → PI 0.03, invalid;
good pulse → valid. The service restart is pending because it needs the sudo password.

## 1. With clean beats, the model and DSP chain do not produce false AF

The deployed weights and the full chain (detrend → Butterworth → Elgendi → features → LR)
were tested on synthetic input:

| Input | Windows flagged AF |
|---|---|
| Ideal beats, 70–85 bpm, RMSSD 0–25 ms | 0% |
| Realistic PPG morphology through full DSP, SNR down to 5 dB | 0% |
| One missed or one extra beat in a clean window | not flagged (P < 0.001) |
| Clean beats + ±30 ms peak-timing error | 11% |
| Clean beats + ±50 ms peak-timing error | 52% |

So the classifier is not broken and there is no polarity or sample-rate bug. False AF
needs **beat timings that are wrong by tens of ms on most beats**, which is what an
unstable, weak signal gives.

## 2. Defect (FIXED locally): the quality gate passed weak-pulse windows whenever the finger drifted

`SignalQualityAssessor` computed perfusion index as **raw max − min / mean**. Raw max − min
counts baseline drift and contact settling as pulse. The 2026-10-06 capture had a ~30-count
pulse on ~100k DC (true PI 0.03%, far below the 0.15% floor):

| Drift over the 10 s window | Old PI | Old verdict |
|---|---|---|
| 0 counts | 0.06% | rejected |
| 300 counts | 0.31% | **valid** |
| 1,000 counts | 1.0% | **valid** |

The gate therefore rejected a still, weak finger but **accepted a moving one**, the exact
windows that produce false AF. **Fix:** AC is now the 5th–95th percentile range of the
band-passed window. Regression test `test_baseline_drift_does_not_pass_weak_pulse`.
Training is unaffected because `build_dataset_v2.py` does not use PI.
Tests: 69 passed. Pre-existing and unrelated: 4 modules need torch/matplotlib (not
installed on this laptop) and `test_deepbeat_loader_real_data` fails without this change too.

**Expected effect:** more windows reported as `measurement_valid: false` rather than AF.
That is correct behaviour. If nearly every window becomes invalid, the signal itself is
too weak (see §3).

## 3. Likely root cause upstream: the pulse signal is too weak

Firmware runs the LEDs at `0x1F` (~6.4 mA), with no hardware averaging and a 4096 nA
range. A 0.03% pulse is roughly 10–50× weaker than a good reflectance fingertip signal.
The usual causes are, in order:

1. **Pressing too hard** on the bare sensor squeezes blood out of the fingertip and
   flattens the pulse.
2. **Contact moving** (no clip or strap).
3. **Ambient light** reaching the photodiode.
4. **LED drive too low** for this person or finger.

Proposed firmware trial (protocol unchanged, still 100 Hz out): `sampleRate 400` +
`sampleAverage 4` (about 2× lower noise), LED amplitude `0x3F`–`0x7F`, `adcRange 16384`
to keep headroom. Check that DC stays well below 262,143 (saturation). **Not yet applied:**
it has to be flashed and checked on the Pi.

## 4. Model limitation: young people with strong sinus arrhythmia

Even with perfect beat timing, healthy young rhythms with strong respiratory sinus
arrhythmia score as AF in some windows: **32% at 65 bpm / RMSSD 50 ms, 44% at 55 bpm /
RMSSD 70 ms** (pNN50 ≈ 0.6). MIMIC PERform's non-AF subjects are critically ill ICU adults
with low HRV, so the model never saw healthy high-HRV negatives. The three test subjects
were students, the group most affected. 3-of-5 consensus reduces this but does not remove
it. It is a **specificity limit on young/healthy users**, which also matters for the
primary-care target population. Fix: add healthy-HRV negatives, ideally the team's own
labelled MAX30102 recordings. **Do not raise τ.**

## Next steps

1. Get a route to the Pi and pull the three subjects' raw windows. Check PI, IBI series
   and which features drove each call (`explain`) to tell §3 apart from §4.
2. Deploy the SQI fix, then re-test with the finger resting **lightly** and kept still,
   shaded from light.
3. Trial the firmware settings in §3.
4. Record the three subjects as labelled non-AF device data. They are the first real
   negatives for the transfer measurement.

---
Back to [[Index|Code Review Index]]

## Update — raw capture, three more defects, fixes deployed (same day)

A 2-minute raw **wrist** capture of a non-AF tester (`03 - ML/data/device_captures/2026-10-12_tester1_wrist_1.npz`,
100.2 Hz, taken with the new `scripts/capture_raw.py`, which pauses the runner instead of
restarting it) showed:

- **No saturation and moderate PI (0.2–1.0%), yet false AF.** The pulse is ~50–100 counts
  on a baseline wandering by hundreds. The detector splits beats (e.g. 340 + 430 ms where
  one ~800 ms gap belongs), misses beats (1,580 ms), and times peaks ±50 ms. Median RMSSD
  was **409 ms**; a resting non-AF person is around 20–60 ms. **Inverting polarity doesn't
  help** (RMSSD 557 ms): the problem is waveform quality, not orientation.
- **Saturation (found in the live log):** windows at `median_IR=262143` (ADC full scale)
  were classified, at 25–50 bpm and confidently non-AF. That output is meaningless.

**Defect 3 — no beat-validity gate (FIXED).** The IBI model sees only beat times, so detector
errors are indistinguishable from AF. Added `beat_morphology_sqi` (template correlation of
each beat's upstroke + peak, r ≥ 0.90, ≥ 70% good beats, ≤ 1 undetected prominent pulse;
after Orphanidou 2015 / Li & Clifford 2012). It judges shape, not timing, so it does not
reject irregular rhythm. On MIMIC (in-sample, so a gate-behaviour check, not an accuracy
figure) it keeps **96.8% of classifiable AF windows** and 92.2% of non-AF. On the device
capture at live cadence it marks 96 of 111 windows `unreliable_beats`. Longer segments
covering the downstroke failed (MIMIC AF retention < 93%, because AF beat lengths vary).
**Tuned on one device capture, so it must be re-checked on more.**

**Defect 4 — saturated windows classified (FIXED).** New `invalid_reason: sensor_saturated`.

**Defect 5 — consensus did not match the validated operating point and was not used (FIXED).**
The validated sens 100% / spec 93.8% is 3-of-5 over **non-overlapping** 10 s windows (50 s).
The runner took 5 consecutive 1 s-stride windows (90% overlap, ~14 s), skipped invalid
windows, and then **neither the ledger nor the dashboard used it**:
- the ledger wrote an event for every single AF window;
- the banner fired on `af_detected === 1 || afProb >= 0.50`.

The live alarm therefore ran at the *window* operating point (spec 83.6%), not the quoted
one. Now:
- consensus uses windows one window-length apart, with invalid windows counted as not-AF;
- the ledger records only consensus AF, at most once per 10 s;
- the banner follows `af_consensus`.

The banner texts no longer mention Grad-CAM or claim "normal sinus rhythm", and the
pre-data placeholder says "Waiting for a valid measurement".

**Effect on the capture:** old logic raised the alarm in 8 of 15 classified windows; new
logic raises it in 0 of 111 windows. Tests: 79 passed. Pre-existing and unrelated:
`test_deepbeat_loader_real_data` fails, and 4 modules need torch/matplotlib.

**Deployed to the Pi** (files copied; replay on the Pi matches the laptop exactly). Pending:
edge service restart (sudo), the firmware range change (`adcRange 16384`), and data from
more testers to check the thresholds. Expect many "no measurement" windows on this sensor
until contact is steadied (wrist strap / enclosure): that is now the honest output.

## Correction — measurement site was the WRIST

The tester confirmed after the fact that **the post-restart testing and the raw capture
were taken at the wrist**, and that the team intends to use the wrist. This note's §3
reasoning about fingertip pressure doesn't apply to the capture. The weak pulse (~50–100
counts) and the wandering baseline are typical of reflectance PPG at the wrist with IR at
~6.4 mA. The beat-morphology thresholds were therefore tuned on a **wrist** capture.

## Update — radial-site firmware change and signal polarity (same day)

Site: wrist, **over the radial pulse** (team decision 13). Same tester, same placement,
60 s before and 60 s after the firmware change.

**Firmware** (`ArrhythmiaNode.ino`, flashed from the Pi, upload hash verified):
- IR/red drive 0x1F → **0x7F** (~6.2 → 25.4 mA);
- ADC range 4096 → **16384 nA**;
- 400 sps with **4× on-chip averaging** → still 100 Hz out (measured 99.9 Hz).

Wire protocol unchanged; DC stayed ~100k as predicted.

| | Before | After |
|---|---|---|
| Sample-to-sample noise | 33.1 counts | **13.3** |
| Pulse amplitude (p5–p95, filtered) | 133 counts (0.14%) | **256 (0.23%)** |
| Windows failing PI/SQI | 61% | **0%** |

**Polarity was the next defect (FIXED).** With a clean pulse, the raw trace shows each beat
as a sharp *fall* (reflected light dips as blood volume rises). The Elgendi detector and the
morphology gate expect systole-up, as in MIMIC, so they locked onto the rounded diastolic
tops and recovery bumps (300–440 ms "beats"). The 2026-10-06 conclusion that polarity was
"ruled out" was drawn on a signal too noisy to tell. Inverting the DSP path:

| Capture | Not inverted | Inverted |
|---|---|---|
| Before flash | 3 AF windows, beat-shape 0.84 | **0 AF**, beat-shape 1.00 |
| After flash | 14 AF / 27 unreliable / 9 no-AF | **13 AF / 1 unreliable / 36 no-AF** |

The runner now inverts by default (`PPG_INVERT=1`); raw counts still drive the contact,
saturation and PI checks. The simulator now emits dipping pulses like the sensor.
Regression test added. 80 tests pass.

**Remaining:** 26% of valid windows on the after capture are still classified AF, with no
3-of-5 alarm in the minute. They are a mix of:
- occasional split or missed beats (1,030 + 420 ms; 1,470 ms);
- near-regular windows such as 700–770 ms with one 930 ms beat (P 0.59).

The second kind is the §4 limitation (ICU-trained negatives vs young, high-HRV wrists),
plus some peak-timing error at the wrist. Over longer sessions occasional consensus alarms
are still expected. **Own-device wrist data, and eventually a model adapted to it, is the
fix; thresholds are not.** Captures: `2026-10-12_tester1_radial_{before,after}.npz`.
