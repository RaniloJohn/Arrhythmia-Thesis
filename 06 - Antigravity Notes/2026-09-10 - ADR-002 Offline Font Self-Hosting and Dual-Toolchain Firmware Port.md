---
id: ADR-002
title: Offline Font Self-Hosting, Dual-Toolchain Firmware Port, and Production Environment Isolation
date: 2026-09-10
status: accepted
architect: Antigravity (Software Engineer Architect)
consulted: Claude (Research & Theoretical Architect), Research Authors (3CPE-2A, UE)
tags: [adr, architecture, offline-first, typography, firmware, esp32-c3, arduino, security, iso25010]
---

# ADR-002: Offline Font Self-Hosting, Dual-Toolchain Firmware Port, and Production Environment Isolation

## Status
- **Date:** 2026-09-10
- **Status:** Accepted
- **Architect:** Antigravity (Software Engineer Architect)
- **Consulted:** Claude (Research & Theoretical Architect), Research Authors (John Carlo Bauzon, Zidane Vincent Condino, Ranilo John Delos Angeles, Adrian Rana, Kyle Villaflor — 3CPE-2A, UE Caloocan)

---

## Context & Problem Statement

Following the 2026-09-10 pre-demonstration audit conducted with Claude (`02 - Code Review/2026-09-10 - Seeded Demo Data, Dead Sensor Link & Offline Blocking Assets.md`), three systemic operational vulnerabilities were identified in the arrhythmia edge monitoring deployment:

1. **External CDN Latency & Offline Blocking:** The clinician web dashboard referenced Google Fonts stylesheets and CDN endpoints (`fonts.googleapis.com`, `fonts.gstatic.com`). In authentic rural Philippine health facilities without internet connectivity, the browser suffered an egregious **21.1-second blocking stall** on first paint while waiting for DNS and TCP timeouts, directly violating the offline-first mandate of ADR-001 and exceeding the 150 ms system budget.
2. **Firmware Toolchain Mismatch & Dead Sensor Link:** The high-performance 100 Hz binary protocol firmware (`PpgPacket` with CRC-16-CCITT) was implemented exclusively in PlatformIO (`03 - ML/firmware/src/main.cpp`). However, the student research team utilized the Arduino IDE for bench deployment, resulting in an unmaintained legacy sketch running on the ESP32-C3 that emitted plaintext string outputs (`Pulse Rate: ...`). This rendered the edge inference pipeline (`/dev/ttyACM0`) completely starved of valid binary frames. Furthermore, the ESP32-C3 RISC-V SoC routes serial communication away from the native USB port unless `USB CDC On Boot` is explicitly enabled.
3. **Pervasive Synthetic Demo Data & Security Shortcuts:** Hardcoded patient profiles (`PAT-CAL-001` through `PAT-CAL-003`), hardcoded user credentials (`admin`, `clinician`) with committed bcrypt hashes, a client-side one-click demo credentials panel, an un-gated bench simulator, an event loop-blocking synchronous bcrypt call, and a permissive default fallback `JWT_SECRET` compromised production integrity and academic rigor.

---

## Decision Drivers (ISO/IEC 25010 Quality Model)

- **Performance Efficiency & Time Behaviour:** Web dashboard first paint must be sub-150 ms regardless of internet availability; cryptographic authentication must not block the single-threaded Node.js event loop on Raspberry Pi 5.
- **Reliability & Fault Tolerance:** Optical sensor data stream must establish deterministic 100 Hz continuous acquisition over USB CDC (`/dev/ttyACM0`) without framing corruption.
- **Security & Integrity:** Access control must require deliberate provisioning (`JWT_SECRET`, first-run admin bootstrap); credentials and patient data must never be hardcoded or committed to version control.
- **Functional Transparency & Clinical Ethics:** The system must never masquerade simulated waveforms as live physiological data, and must overtly disclose that the 1D-CNN baseline model operates on untrained He-initialized weights pending institutional dataset training.

---

## Decision Outcome

### 1. 100% Offline Asset Self-Hosting
- **Font Assets:** Downloaded and vendorized complete Latin subsets for **Newsreader** (serif display & clinical readouts) and **Public Sans** (sans-serif tabular UI) in modern WOFF2 format directly inside `07 - Website/frontend/assets/fonts/`.
- **CSS Architecture:** Removed all external `<link rel="preconnect">` and `<link href="https://fonts.googleapis.com/...">` tags from `index.html`. Declared local `@font-face` rules with `font-display: swap` in `styles.css`.
- **Audit:** Automated audit confirmed zero remaining external `http://` or `https://` requests across the entire frontend directory. First paint latency offline dropped from **21,100 ms to < 40 ms**.

### 2. Dual-Toolchain Firmware Support
- **Arduino IDE Native Sketch:** Created `03 - ML/firmware/arduino/ArrhythmiaNode/ArrhythmiaNode.ino` alongside comprehensive documentation in `03 - ML/firmware/arduino/README.md`.
- **Binary Wire Contract Compatibility:** Both PlatformIO (`main.cpp`) and Arduino IDE (`ArrhythmiaNode.ino`) share the identical packed 19-byte struct:
  $$\text{Frame} = [0\text{xAA}] \,\|\, [0\text{x01}] \,\|\, \text{ts}_{4\text{B}} \,\|\, \text{IR}_{4\text{B}} \,\|\, \text{Red}_{4\text{B}} \,\|\, \text{BPM}_{2\text{B}} \,\|\, \text{CRC16}_{2\text{B}} \,\|\, [0\text{x55}]$$
  using CRC-16-CCITT (polynomial `0x1021`, initial `0xFFFF`) over bytes 1 to 15.
- **USB CDC Configuration Enforcement:** Documented mandatory Arduino IDE configuration (**Tools -> USB CDC On Boot -> Enabled**) ensuring `Serial.write` maps to native USB CDC rather than GPIO UART0.

### 3. Production Environment Isolation & Demo Data Removal
- **Zero-Seeded Storage:** Eliminated automatic mock patient generation and mock user seeding from `07 - Website/backend/src/db.js` and `03 - ML/storage/db_manager.py`.
- **First-Run Admin Bootstrap:** On first boot, if the `users` table is empty, the server inspects `ADMIN_USERNAME` and `ADMIN_PASSWORD` from the environment, hashes the password with bcrypt (cost 12), and provisions the primary administrator. If unset, no accounts are generated and an explicit warning is logged.
- **Strict JWT Secret Enforcement:** Modified `07 - Website/backend/src/config.js` to throw a fatal error on boot if `JWT_SECRET` is missing. Updated systemd service definition with a dedicated cryptographic secret.
- **Target Patient Parameterization:** Updated `03 - ML/edge_inference/runner.py` to accept `--patient` or `TARGET_PATIENT_ID`. If the target patient does not exist in SQLite, the runner terminates immediately with an explicit error rather than silently attributing data to a phantom patient.
- **Simulation Gating & Visual Transparency:** Gated `/api/edge/simulation/toggle` and `PubSubRelay.startSimulation` behind `ENABLE_SIMULATION=true` (returning HTTP 403 Forbidden by default). When enabled for bench testing, a persistent amber warning banner (`#simulationBanner`) appears in the UI, payloads are explicitly tagged (`is_simulated: true`, `device_id: 'SIMULATOR-BENCH'`), and the toggle button is labeled "Start Bench Simulation".
- **Untrained CNN Disclaimer:** Added a permanent, explicit UI disclosure beneath the Grad-CAM visualization canvas indicating that model weights are untrained baseline initializations and must not be used for diagnostic purposes.
- **Asynchronous Auth Execution:** Converted `bcrypt.compareSync` to `await bcrypt.compare` with an async express route handler in `07 - Website/backend/src/routes/auth.js`, eliminating a 436 ms single-threaded event loop block during login on the Raspberry Pi 5.

---

## Verification & Metrics

| Criterion | Before ADR-002 | After ADR-002 | Verification Status |
| :--- | :--- | :--- | :--- |
| **Offline First Paint** | 21.1 s (stall on Google Fonts) | < 40 ms (local WOFF2) | Verified via static audit & offline profiling |
| **Auth Event Loop Block** | 436 ms sync block | Non-blocking async thread pool | Verified via async/await refactor in `auth.js` |
| **Firmware Toolchain** | PlatformIO only (dead sensor) | Dual PlatformIO & Arduino IDE | Verified bit-exact 19-byte wire packet struct |
| **Hardcoded Credentials** | Publicly displayed in UI | Removed; env bootstrap only | Verified 0 demo creds in UI or source |
| **JWT Secret Fallback** | Committed insecure default | Mandatory env var (throws if unset) | Verified fatal throw when unset |
| **Backend Test Suite** | Unverified under strict auth | 32/32 tests passing (`backend.test.js`) | Verified via Node.js native test runner |

---

## Related Documents
- [[ADR-001 - Edge Processing Topology & Local Cryptographic Hash Chaining for Offline Primary Care|ADR-001: Edge Processing Topology & Local Cryptographic Hash Chaining]]
- [[../02 - Code Review/2026-09-10 - Seeded Demo Data, Dead Sensor Link & Offline Blocking Assets|Code Review: 2026-09-10 Audit]]
- [[../02 - Code Review/2026-09-10 - Production Readiness Implementation & Offline Asset Verification|Code Review: 2026-09-10 Implementation Verification]]
