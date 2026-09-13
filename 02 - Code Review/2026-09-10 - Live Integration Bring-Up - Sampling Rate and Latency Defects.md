---
date: 2026-09-10
component: ESP32-C3 firmware (both toolchains), 07 - Website/backend bridge, systemd deployment
reviewer: Claude (Research & Theoretical Architect)
tags: [code-review, firmware, dsp, latency, iso25010, integration]
status: open
---

# 2026-09-10 — Live Integration Bring-Up: Sampling Rate and Latency Defects

## Scope

Bring-up of the full acquisition chain after flashing the new Arduino sketch onto the
ESP32-C3 remotely via `arduino-cli` over Tailscale. Covers the firmware sampling loop
(`03 - ML/firmware/arduino/ArrhythmiaNode/ArrhythmiaNode.ino` and
`03 - ML/firmware/src/main.cpp`), the WebSocket bridge
(`07 - Website/backend/src/bridge/pubsub_relay.js`), and the deployed systemd units.

## Summary

The pipeline now runs end to end on real sensor data for the first time — sensor to
serial to DSP to CNN to Grad-CAM to SQLite to dashboard, with `Source=SERIAL` confirmed
in the edge log. Bringing it up exposed two defects that were invisible while the link
was dead, both of which would have silently corrupted the thesis's quantitative results
rather than causing a visible failure.

## Findings

| # | Severity | Issue | File / Location | Fix | Status |
|---|----------|-------|------------------|-----|--------|
| 1 | **Critical** | Firmware sampled at **49.8 Hz, not 100 Hz**. `getIR()` and `getRed()` each call the SparkFun library's `safeCheck()`, which blocks until a *new* FIFO sample arrives, so calling both consumed two sensor samples per loop pass and halved the effective rate. The MCU's own timestamps confirmed 20 ms spacing. | `ArrhythmiaNode.ino` and `main.cpp`, sampling loop | Replaced with a single `check()` + `available()` + `getFIFOIR()`/`getFIFORed()` + `nextSample()` sequence, reading both channels from one FIFO entry. Re-measured at **99.6 Hz**. | fixed |
| 2 | **Critical** | Bridge reported `avgBridgeLatencyMs: 1789037633217.9` (~56 years). `payload.ts` is the ESP32-C3's monotonic `millis()` uptime, not a Unix epoch, but the bridge subtracted it from `Date.now()`. This value is stamped into **every** outbound WebSocket message as `transitLatencyMs` and rendered by the frontend as the sensor-to-browser latency. | `pubsub_relay.js:196` | Use the edge runner's `iso_time` (wall-clock UTC from the same host) and reject implausible values rather than letting them poison the moving average. Re-measured at **1.2 ms**. | fixed |
| 3 | High | Inference latency measured **27.7–42.1 ms** on real windows, breaching the 25 ms budget in `config.js` on every window (`<25ms: False`). | `03 - ML/model/inference_model.py` on the Pi 5 | Not fixed. Requires the TFLite/ONNX quantisation already deferred in the plan. | open |
| 4 | High | AGY's replacement systemd unit committed `JWT_SECRET` to the repository and set `ADMIN_PASSWORD=admin123` — reinstating the exact credential the same sprint had just removed. | `03 - ML/deploy/systemd/arrhythmia-website.service` | Rejected. Both units now use `EnvironmentFile=` pointing at `0600` files under `~/.config/arrhythmia/` on the Pi, with randomly generated values. No secret is under version control. | fixed |
| 5 | Medium | `runner.py` correctly refuses to start without `TARGET_PATIENT_ID`, but the deployed unit did not set it, so the edge service crash-looped after the update. | deployed systemd unit | `edge.env` created with the target patient. | fixed |
| 6 | Medium | SQI oscillates between 0.4 and 1.0 with no finger on the sensor, and windows at SQI 0.4 still reach inference. The quality gate does not appear to be rejecting unusable windows. | `03 - ML/signal_processing/sqi.py`, `runner.py` | Not investigated. | open |
| 7 | Medium | The two seeded accounts (`admin`, `clinician`) whose passwords are in git history remain **live rows** in the Pi database. Removing the seeding code does not remove rows already written, and AGY's purge utility does not touch the `users` table. | live `arrhythmia_edge.db` | Deferred at the user's explicit instruction not to touch login credentials yet. | deferred |

## Notes / architecture observations

- Finding 1 is the most consequential result of this session. The entire DSP chain is
  configured for `fs = 100.0` — the 4th-order Butterworth cutoffs, the Elgendi detector,
  and the CNN's window semantics all assume it. At 49.8 Hz every reported BPM and IBI
  would have been **almost exactly double the true value**, with no error raised
  anywhere. A heart rate of 60 bpm would have been reported as ~120 bpm and read as
  tachycardia. Because `main.cpp` carried the identical defect, this was inherited from
  the reference firmware rather than introduced during the Arduino port — meaning no
  build of this firmware has ever sampled at the documented rate.
- The remote flashing path is worth recording as a capability: `arduino-cli` on the Pi,
  `--fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc`, uploading over `/dev/ttyACM0` while the
  edge service is stopped. The `CDCOnBoot=cdc` build flag replaces the Arduino IDE's
  **Tools → USB CDC On Boot → Enabled** menu setting, so no physical access is needed
  for firmware iteration. Note that `arduino-cli upload` does **not** recompile — a
  `compile --clean` must precede it or the previous binary is silently reflashed.
- `arrhythmia_events` grew from 4 to 110 rows during bring-up with `chain_valid: true`
  throughout, so the SHA-256 backward chain holds under real write load, not only in
  tests.
- The AF probabilities being written are noise: they swing between 4% and 85% across
  consecutive one-second windows with no finger on the sensor. This is the untrained
  model behaving exactly as the 2026-09-03 finding predicted. The waveform is now real;
  the classification is not, and the two must not be presented together as a result.

## Follow-ups

- [ ] Correct `ANTIGRAVITY.md` §5: the Performance Efficiency row still claims "inference
      ~13.08 ms", measured on synthetic data. The real figure is 27.7–42.1 ms and the
      budget is breached.
- [ ] Investigate finding 6 before any clinical demonstration — an SQI gate that passes
      finger-off windows undermines the Reliability claim.
- [ ] Repoint `TARGET_PATIENT_ID` at a real clinician-created patient, then purge the
      synthetic events, the three seeded patients, and the two compromised accounts.
- [ ] Re-verify the 100 Hz rate with a finger actually on the sensor — all measurements
      so far were taken with IR ≈ 1670 (no contact).

## Related

- [[Index|Code Review Index]]
- [[2026-09-10 - Seeded Demo Data, Dead Sensor Link & Offline Blocking Assets|Seeded demo data & dead sensor link]]
- [[2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights|1D-CNN untrained weights]]
- [[../05 - Claude Notes/2026-09-10 - Website Startup Latency & Offline Readiness Diagnosis|Latency & offline readiness diagnosis]]
