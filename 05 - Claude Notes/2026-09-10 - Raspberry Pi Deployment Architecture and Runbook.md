---
title: Raspberry Pi Deployment Architecture and Runbook
tags: [deployment, architecture, runbook, raspberry-pi, firmware, tailscale]
date: 2026-09-10
status: current — describes the live system as of this date
reviewer: Claude (Research & Theoretical Architect)
---

# Raspberry Pi Deployment Architecture and Runbook

This note is the single place to look when picking the project up cold. It records
what is deployed, where it lives, how to reach it, how to change it, and what is
known to be broken. It was written after the session that first brought the full
acquisition chain up on real sensor data.

## 1. Where everything runs

The system runs on a **Raspberry Pi 5 (8 GB)** that is not physically accessible to
the project owner — it lives at a teammate's location and is reached only over
Tailscale. The Windows workstation holds the git repository and the Obsidian vault;
it is an authoring environment, not a runtime. Never start a local copy of the
website on Windows to reproduce behaviour.

| | |
|---|---|
| Host | `raspberrypi` / `raspberrypi.tail03bdf4.ts.net` |
| Tailscale IP | `100.77.17.38` |
| Login user | `ranilo` |
| Project path | `/home/ranilo/Arrhythmia Thesis/` |
| Dashboard | `http://localhost:8080` on the Pi, or `http://100.77.17.38:8080` over the tailnet |
| OS | Debian 13 (trixie), Pi 5, 8 GB RAM, ~12 GB free disk |

### Reaching it

SSH is served by **Tailscale SSH**, not the Pi's own `sshd` — reading the TCP banner
on port 22 returns `SSH-2.0-Tailscale`. Authentication is by tailnet identity, so no
SSH key or password is involved. The tailnet policy uses `"action": "check"`, which
means the first connection of a session prints a one-time URL to **stderr** and then
blocks until a human clicks it:

```
# Tailscale SSH requires an additional check.
# To authenticate, visit: https://login.tailscale.com/a/<token>
```

Any tooling that pipes stderr will not surface that line until the process exits, and
it never exits on its own — the symptom is a completely silent hang that looks like a
network failure but is not. Once cleared, the check is cached for the tailnet's
`checkPeriod` (12 h by default), so subsequent commands connect straight through.

## 2. Services

Both run under systemd, enabled at boot.

| Unit | What it does | Working directory |
|---|---|---|
| `arrhythmia-edge.service` | `python3 edge_inference/runner.py` — serial ingestion, DSP, CNN, Grad-CAM, SQLite writes, TCP publish to the dashboard | `03 - ML` |
| `arrhythmia-website.service` | `node src/server.js` — REST API, static frontend, WebSocket relay, TCP bridge listener on 5051 | `07 - Website/backend` |

### Secrets are not in the repository

Both units load configuration through `EnvironmentFile=` from `0600` files owned by
`ranilo`. Nothing secret is committed. An earlier iteration inlined `JWT_SECRET` and
`ADMIN_PASSWORD=admin123` directly into the unit files; that was rejected and replaced.

- `/home/ranilo/.config/arrhythmia/website.env` — `JWT_SECRET`, `ADMIN_USERNAME`,
  `ADMIN_PASSWORD`, `NODE_ENV`, `PORT`. The backend **throws on startup** if
  `JWT_SECRET` is unset; there is deliberately no fallback.
- `/home/ranilo/.config/arrhythmia/edge.env` — `TARGET_PATIENT_ID`, `DEVICE_ID`.
  `runner.py` **refuses to start** if `TARGET_PATIENT_ID` is unset or names a patient
  that does not exist in the registry.

### Passwordless sudo

A scoped rule at `/etc/sudoers.d/arrhythmia-flash` allows stopping, starting and
restarting the services without a password. This exists so firmware can be reflashed
unattended. Installing new unit files, editing `/etc/systemd/`, or `daemon-reload`
still requires an interactive password from a real terminal — Claude Code's `!` prefix
runs with a non-interactive stdin and cannot answer a `sudo` prompt.

## 3. The acquisition chain

```
MAX30102 ──I2C(GPIO 8/9)──> ESP32-C3 ──USB CDC 115200──> Pi /dev/ttyACM0
   │                            │                            │
   └ 100 Hz, Red+IR             └ 19-byte CRC framing        └ runner.py
                                  + SSD1306 OLED fallback       DSP → SQI gate →
                                                                CNN → Grad-CAM →
                                                                SQLite hash chain →
                                                                TCP :5051 → Node →
                                                                WebSocket → browser
```

### The wire protocol

Fixed **19 bytes**, `#pragma pack(1)`, little-endian, guarded by a compile-time
`static_assert(sizeof(PpgPacket) == 19)` on the firmware side and `PACKET_SIZE = 19`
on the Python side.

```
[0xAA][0x01][timestamp_ms:4][ir_raw:4][red_raw:4][bpm_x10:2][crc16:2][0x55]
CRC-16-CCITT, polynomial 0x1021, init 0xFFFF, computed over bytes 1..15
```

`timestamp_ms` is the **MCU's monotonic `millis()` uptime**, not a Unix epoch. Treating
it as an epoch is what produced a reported latency of ~56 years; use the edge runner's
`iso_time` field for any wall-clock arithmetic.

### Two firmware builds, one protocol

`03 - ML/firmware/src/main.cpp` (PlatformIO) and
`03 - ML/firmware/arduino/ArrhythmiaNode/ArrhythmiaNode.ino` (Arduino IDE) are two
builds of the same contract. Any change to the frame layout must be applied to both.
The `.ino` is the one actually deployed.

## 4. Flashing the firmware remotely

Because nobody can touch the hardware, firmware is compiled and flashed **on the Pi**
over Tailscale. `arduino-cli` 1.5.1 is installed with the `esp32:esp32` 3.3.11 core and
the three required libraries.

```bash
# 1. free the serial port (the runner holds it open)
ssh ranilo@100.77.17.38 "sudo -n systemctl stop arrhythmia-edge.service"

# 2. compile — --clean matters, see the caveat below
ssh ranilo@100.77.17.38 "cd '/home/ranilo/Arrhythmia Thesis/03 - ML/firmware/arduino' && \
  arduino-cli compile --clean --fqbn 'esp32:esp32:esp32c3:CDCOnBoot=cdc' ./ArrhythmiaNode"

# 3. upload
ssh ranilo@100.77.17.38 "cd '/home/ranilo/Arrhythmia Thesis/03 - ML/firmware/arduino' && \
  arduino-cli upload -p /dev/ttyACM0 --fqbn 'esp32:esp32:esp32c3:CDCOnBoot=cdc' ./ArrhythmiaNode"

# 4. bring the runner back
ssh ranilo@100.77.17.38 "sudo -n systemctl start arrhythmia-edge.service"
```

Two things that will silently waste time if forgotten:

- **`CDCOnBoot=cdc` is mandatory.** It is the command-line equivalent of the Arduino
  IDE's *Tools → USB CDC On Boot → Enabled*. The ESP32-C3 enumerates as
  `303a:1001 USB JTAG/serial debug unit`; without this flag `Serial` is routed to the
  UART0 pins and nothing reaches `/dev/ttyACM0`. The failure mode is zero bytes with
  no error anywhere.
- **`arduino-cli upload` does not recompile.** Without a preceding `compile --clean`
  it reflashes the previous binary and reports success, including "Hash of data
  verified". Edits appear to have no effect.

## 5. Validity gating — read this before interpreting any reading

The MAX30102 always returns samples. With no finger present it returns low-amplitude
ambient noise (IR ≈ 1,670 counts) rather than silence, and the bandpass filter plus
peak detector will happily manufacture a plausible heart rate from that noise. Three
defects in this area were fixed on 2026-09-10:

1. `runner.py` substituted a hardcoded **`72.0` BPM** whenever peak detection failed.
2. The firmware's finger check only early-returned from the OLED draw routine, so
   frames still carried a heuristic BPM locked onto noise.
3. The frontend ignored validity entirely and, because `null !== undefined` is `true`
   in JavaScript, rendered **"NORMAL SINUS RHYTHM — 0.0% AF Index"** for a device
   nobody was wearing.

The current contract:

- Firmware sends `heuristic_bpm_x10 = 0` ("unknown") below
  `FINGER_PRESENT_IR_THRESHOLD` (50,000 IR counts).
- `runner.py` rejects a window when median IR is below `CONTACT_IR_THRESHOLD`
  (default 50,000) or SQI is below `MIN_SQI_SCORE` (default 0.7), and when a
  contacted window yields no resolvable pulse. Both are environment-tunable.
- A rejected window publishes `measurement_valid: false` with an `invalid_reason` of
  `no_skin_contact`, `low_signal_quality` or `no_pulse_detected`, and carries
  `bpm: null`, `af_probability: null` and an **empty** `raw_window`. No ledger event
  is written.
- The frontend renders a neutral dashed **NO MEASUREMENT** banner, blanks BPM and AF
  to `--`, and clears the canvas. It must never show a rhythm classification for an
  invalid window.

Any future consumer of the telemetry must check `measurement_valid` first.

## 6. Data and integrity

SQLite at `03 - ML/storage/arrhythmia_edge.db`, WAL mode, shared between the Python
edge writer (`arrhythmia_events`) and the Node backend (`patients`, `users`) per
ADR-001. Events are SHA-256 backward hash-chained; `chain_valid` stayed `true` across
198 events written under real load, so the chain holds outside of tests.

**Known contamination, deliberately not yet cleaned** (the owner asked that login
credentials be left alone for now):

- 3 fabricated patients (`PAT-CAL-001/002/003`) with invented names, ages, Caloocan
  addresses and medical histories.
- 2 seeded accounts, `admin` and `clinician`, whose passwords were committed to git
  and were printed on the login page. The seeding *code* is gone, but the *rows*
  remain — deleting seed code does not delete rows already written, and the purge
  utility at `03 - ML/scripts/purge_synthetic_data.py` does not touch the `users`
  table.
- ~198 events, the first ~4 synthetic and the remainder produced during bring-up from
  noise by an untrained model.

Purging the patients requires repointing `TARGET_PATIENT_ID` at a real
clinician-created patient first, or the edge service will refuse to start. Deleting
ledger rows breaks the backward chain by design, so the chain must be reinitialised
from genesis rather than patched.

## 7. Verified working as of 2026-09-10

- Sampling at **99.6 Hz** (was 49.8 Hz — see the sampling-rate defect below).
- Frames parsing with valid CRC, `Source=SERIAL` confirmed in the edge log.
- Full chain live: sensor → serial → DSP → CNN → Grad-CAM → SQLite → WebSocket.
- Backend **32/32 tests passing**, including authenticated WebSocket delivery,
  waveform payload and Grad-CAM weights, and rejection of unauthenticated sockets.
- Bridge latency **1.2 ms** (was 1.79e12 ms).
- Dashboard fully offline-capable: self-hosted WOFF2 fonts in
  `07 - Website/frontend/assets/fonts/`, no external URLs, first paint well under a
  second with the network disconnected.
- Synthetic simulation gated behind `ENABLE_SIMULATION`, off by default.

## 8. Known broken / open

| Issue | Detail |
|---|---|
| **Model is untrained** | Weights are random He-init, never trained on MIMIC PERform AF. AF probabilities swing 4%–85% between consecutive seconds on noise. The waveform is real; the classification is not. Never present them together as a result. |
| **Inference exceeds budget** | 27.7–42.1 ms measured on real windows against the 25 ms budget in `config.js`; `<25ms: False` on every window. `ANTIGRAVITY.md` §5 still claims "~13.08 ms", measured on synthetic data — that figure is stale and must be corrected. |
| **Chromium on the Pi cannot load any URL** | Not a website problem. It fails on `localhost:8080`, `example.com` and `duckduckgo.com` alike, with zero established TCP connections, while `curl` succeeds in 1.5 ms and **Firefox renders the dashboard correctly**. Reproduces with a fresh profile. Use Firefox on the Pi; repairing Chromium is a separate sysadmin task. |
| **Clock inconsistency** | `uptime` and `ps` disagree with systemd's "since" timestamps on this Pi. Not investigated; may affect any wall-clock reasoning. |
| **Never tested with a real pulse** | Every measurement in this note was taken with IR ≈ 1,670, i.e. no skin contact. The 100 Hz fix and the whole DSP path still need verification against an actual finger on the sensor. |

## 9. Session-level gotchas worth remembering

- `agy` (Antigravity) rejects a positional prompt. Use `agy --print "<prompt>"`; `-p`
  takes the prompt as its value and does not read stdin. `orchestrate.bat` used the
  positional form and therefore never worked as documented.
- Windows PowerShell 5.1 mangles multi-line arguments whose lines begin with `-`,
  parsing them as flags. Pass long briefs via a file instead.
- Windows filenames cannot contain `:`, which matters when naming vault notes.

## Related

- [[2026-09-10 - Connecting to the Raspberry Pi over Tailscale SSH|Tailscale SSH connection notes]]
- [[2026-09-10 - Website Startup Latency & Offline Readiness Diagnosis|Latency & offline readiness diagnosis]]
- [[../02 - Code Review/2026-09-10 - Live Integration Bring-Up - Sampling Rate and Latency Defects|Live integration bring-up defects]]
- [[../02 - Code Review/2026-09-10 - Seeded Demo Data, Dead Sensor Link & Offline Blocking Assets|Seeded demo data & dead sensor link]]

---
Back to [[Index|Claude Notes Index]]
