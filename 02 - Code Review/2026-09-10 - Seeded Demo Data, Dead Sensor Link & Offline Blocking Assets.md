---
date: 2026-09-10
component: 07 - Website (backend + frontend), 03 - ML/edge_inference, ESP32-C3 firmware
reviewer: Claude (Research & Theoretical Architect)
tags: [code-review, demo-data, offline, firmware, security]
status: open
---

# 2026-09-10 — Seeded Demo Data, Dead Sensor Link & Offline Blocking Assets

## Scope

Pre-demonstration audit requested by the user, covering three questions: why the
dashboard is slow to load, whether any placeholder/dummy content remains in the
system, and why no live PPG waveform appears now that the MAX30102 is powered.
Reviewed `07 - Website/backend/src/`, `07 - Website/frontend/`,
`03 - ML/edge_inference/`, `03 - ML/firmware/src/main.cpp`, and the live state of
the Raspberry Pi (`raspberrypi` @ `100.77.17.38`) over Tailscale SSH.

## Summary

The pipeline is architecturally complete end to end, but it is currently running on
fabricated data and is not receiving anything from the sensor. Three fake patients,
two fake clinician accounts and four synthetic AF events are auto-seeded into the
shared SQLite database, the login screen publicly displays working credentials, and
the ESP32-C3 has been emitting zero bytes for three days because the sketch actually
flashed on it speaks a different protocol than the Pi expects. Separately, the page's
first paint is blocked on a Google Fonts CDN request, which stalls for 21 s whenever
the deployment is genuinely offline.

## Findings

| # | Severity | Issue | File / Location | Suggested Fix | Status |
|---|----------|-------|------------------|----------------|--------|
| 1 | High | ESP32-C3 is running the legacy Arduino sketch, which prints a human-readable BPM string every 500 ms instead of the `0xAA`/`0x55` + CRC16 binary frames the Pi parses. The edge runner opened `/dev/ttyACM0` on Sep 7, logged "Ingesting raw 100 Hz frames" and has logged nothing since; it has consumed 827 ms of CPU in 3 days and sits in `poll`. A 5 s raw capture of the port returned zero bytes. Net effect: no waveform, no inference, no events. | `~/MyEspProject/MyEspProject.ino` on the Pi vs. `03 - ML/firmware/src/main.cpp` | Port the framed protocol into an Arduino-IDE-compatible `.ino` and reflash; see PLAN.md §3 | open |
| 2 | High | The legacy sketch does not compile as written: it includes `<wire.h>` and calls `wire.begin(8, 9)` and `particleSensor.begin(wire, ...)` in lowercase, but the Arduino library object is `Wire`. Whatever is currently on the board is therefore an older build than this source, so the source on the Pi is not a reliable record of device behaviour. | `~/MyEspProject/MyEspProject.ino` lines 1, 23, 27 | Superseded by the rewrite in finding 1 | open |
| 3 | High | Three fabricated patient records are auto-seeded on first run — full names, ages, Caloocan barangay addresses, phone numbers and medical histories. Confirmed present in the live Pi database. These are indistinguishable from real records in the UI and in any exported PDF/CSV. | `07 - Website/backend/src/db.js:121-160` | Remove the patient seeding block; require clinicians to create records through the CRUD UI | open |
| 4 | High | The login screen renders a "Demo Credentials (ADR-001)" panel with one-click fill buttons for the `admin` and `clinician` accounts, exposing working credentials to anyone who loads the page. | `07 - Website/frontend/index.html:44-56`, `07 - Website/frontend/js/auth.js:31` | Delete the panel and the fill handlers; provision the first admin via a one-time setup step instead | open |
| 5 | High | Two seeded user accounts ship with hard-coded bcrypt hashes committed to the repository, including a fictitious clinician "Dr. Maria Santos, MD (Brgy. 171 Health Center)". Because the hashes are in version control, the passwords are effectively public. | `07 - Website/backend/src/db.js:97-119` | Replace with a first-run admin bootstrap that forces a password change; never commit hashes | open |
| 6 | High | First paint is blocked by a render-blocking Google Fonts stylesheet. Measured 21.1 s stall when the host is unreachable, versus 134 ms when online. Contradicts the "100% offline-capable" claim in `ANTIGRAVITY.md` section 1 and breaches the 150 ms budget in `config.js`. | `07 - Website/frontend/index.html:7-9` | Self-host both font families locally and delete the external links | open |
| 7 | Medium | The four `arrhythmia_events` rows in the live database are synthetic output from the simulation path, all dated 2026-09-03, all `af_detected = 1` with confidences of 0.768/0.941 produced by the untrained model. They will be read by a reviewer as detection results. | live `arrhythmia_edge.db`; generated via `07 - Website/backend/src/bridge/pubsub_relay.js:242` | Purge before demonstration; regenerate only from real acquisition | open |
| 8 | Medium | The synthetic waveform generator toggles rhythm state every 15 windows "to demonstrate Grad-CAM explainability", and is reachable from the UI through a "Start Live Demo" button and the `/api/edge` simulation toggle. With real hardware attached this is now a liability: a demo stream is visually identical to a real one. | `07 - Website/backend/src/bridge/pubsub_relay.js:242`, `07 - Website/backend/src/routes/edge.js:46`, `07 - Website/frontend/js/live-stream.js:141` | Keep the code path for bench testing but gate it behind an env flag, default off, and label it unmistakably in the UI when active | open |
| 9 | Medium | The JWT signing secret has a hard-coded fallback committed to the repository. If `JWT_SECRET` is unset — as it is in the current systemd unit — anyone with the repo can mint valid clinician tokens. | `07 - Website/backend/src/config.js:17` | Fail fast on startup when `JWT_SECRET` is unset; supply it via the systemd unit | open |
| 10 | Low | The edge runner's target patient is hard-coded to `PAT-CAL-001`, one of the seeded fake records, so removing the seeds without changing this leaves events attributed to a non-existent patient. | `03 - ML/edge_inference/runner.py` | Make the target patient configurable and validate it exists at startup | open |

## Notes / architecture observations

- The correct firmware already exists and is well written: `03 - ML/firmware/src/main.cpp`
  (311 lines) implements deterministic 100 Hz sampling, the `PpgPacket` struct with
  CRC-16-CCITT over a 15-byte payload, `0xAA`/`0x55` delimiters, and the SSD1306 OLED
  fallback path. The gap is purely one of delivery: it is a PlatformIO project, and the
  team flashes the board from the Arduino IDE, so this source has never reached the
  device. This is a toolchain mismatch, not a missing implementation.
- `platformio.ini` sets `-D ARDUINO_USB_MODE=1` and `-D ARDUINO_USB_CDC_ON_BOOT=1`.
  These have no equivalent in a bare `.ino` file — in the Arduino IDE they are set from
  **Tools → USB CDC On Boot → Enabled**. The board enumerates as `303a:1001 Espressif
  USB JTAG/serial debug unit`, so without that setting `Serial` is routed to the UART0
  pins and nothing reaches `/dev/ttyACM0` over USB. Any port of this firmware must carry
  that instruction with it or it will silently produce no data.
- A discrepancy worth resolving separately: the Pi reports `up 22 min` while both
  systemd units report continuous operation since Sep 7. This was not investigated and
  may indicate a clock or suspend/resume anomaly on the Pi.
- Findings 3, 4, 5 and 7 are grouped deliberately. Individually each looks like ordinary
  development scaffolding; together they mean the dashboard currently presents an
  entirely fictional clinical picture — fake patients, a fake attending clinician, fake
  AF detections from an untrained model — with no visual marker distinguishing any of it
  from real output. That is the specific risk to a thesis demonstration.

## Follow-ups

- [ ] Antigravity to execute `PLAN.md` sections 1-4 (offline assets, demo-data removal,
      Arduino firmware port, live-stream activation).
- [ ] Purge the four synthetic events and re-verify the SHA-256 hash chain afterwards,
      since deleting ledger rows breaks the backward chain by design.
- [ ] Re-run `07 - Website/backend/tests/backend.test.js` after seed removal — several of
      the 32 passing tests may depend on the seeded fixtures.
- [ ] Decide whether the untrained-model caveat from the 2026-09-03 finding should be
      surfaced in the dashboard UI itself, given real waveforms will now flow through it.

## Related

- [[Index|Code Review Index]]
- [[2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights|1D-CNN untrained weights]]
- [[2026-09-03 - Clinician Web Dashboard & Edge PubSub Gateway|Clinician dashboard & edge pub/sub gateway]]
- [[../05 - Claude Notes/2026-09-10 - Website Startup Latency & Offline Readiness Diagnosis|Latency & offline readiness diagnosis]]
