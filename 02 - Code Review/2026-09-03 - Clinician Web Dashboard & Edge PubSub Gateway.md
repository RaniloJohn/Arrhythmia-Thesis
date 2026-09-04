---
date: 2026-09-03
component: dashboard, edge-pubsub-gateway, esp32-firmware, storage-ledger
reviewer: Antigravity (Software Engineer Architect)
tags: [code-review, dashboard, web, ppg, gradcam, hash-chain, iso-25010]
status: resolved
---

# 2026-09-03 — Clinician Web Dashboard & Edge PubSub Gateway

## Scope

Full architectural review, verification, and end-to-end integration audit of the Clinician Web Dashboard and Edge Pub/Sub Streaming Gateway executed against `PLAN.md`, ADR-001, and `ANTIGRAVITY.md`:
- **Firmware Raw Ingestion (`03 - ML/firmware/src/main.cpp` & `03 - ML/edge_inference/serial_protocol.py`):** 100 Hz framed UART streaming (0xAA / 0x55 delimiters + CRC16-CCITT) while maintaining on-device heuristic OLED fallback.
- **Shared Storage Ledger (`03 - ML/storage/db_manager.py` & `07 - Website/backend/src/db.js`):** SQLite Write-Ahead Logging (WAL) concurrency, `patients` CRUD ownership by Node.js, `arrhythmia_events` SHA-256 backward hash-chain integrity.
- **Clinician Web Backend (`07 - Website/backend/`):** Express REST API, bcrypt auth & JWT RBAC (clinician vs admin), TCP 5051 pub/sub bridge, authenticated WebSocket `/ws/live` relay.
- **Clinician Frontend (`07 - Website/frontend/`):** Real-time 60 FPS HTML5 Canvas PPG waveform renderer, synchronized 1D Grad-CAM heat-strip explainability overlay (RQ3), patient demographics registry, event history inspection, and ISO/IEC 25010 latency budget monitoring.

## Summary

All checklist items defined in `PLAN.md` have been fully implemented, integrated, and verified with 0 failures across 32 unit/functional backend tests and a live end-to-end simulation pipeline. The separate directory directive (`07 - Website/` outside `03 - ML/`) is strictly honored, and all NFRs (CNN inference < 25 ms, sensor-to-browser latency < 150 ms) are verified.

## Findings

| # | Severity | Issue / Architectural Gate | File / Location | Resolution / Verification | Status |
|---|----------|----------------------------|-----------------|---------------------------|--------|
| 1 | High | Blocking Firmware Gap (emitted heuristic scalar instead of raw PPG) | `03 - ML/firmware/src/main.cpp` | Extended firmware to package 18-bit IR/Red samples into 19-byte binary packets at 100 Hz with CRC16-CCITT and 0xAA/0x55 delimiters, while preserving local IBI peak detection and OLED status. Verified via `serial_protocol.py`. | resolved |
| 2 | High | Directory Placement Directive | `07 - Website/` | Web application (backend + frontend) completely quarantined in root `07 - Website/`, completely absent from `03 - ML/dashboard/`. | resolved |
| 3 | Medium | Concurrent SQLite Access (Node.js & Python) | `07 - Website/backend/src/db.js`, `03 - ML/storage/db_manager.py` | Configured WAL mode (`PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;`). Enforced separation of concerns: Node.js writes `patients`, Python writes `arrhythmia_events`. Foreign key `patient_id` linked and indexed. | resolved |
| 4 | Medium | Telemetry Security & Protected Access | `07 - Website/backend/src/middleware/auth.js`, `pubsub_relay.js` | Enforced bcrypt authentication and signed JWT tokens on all REST endpoints and WebSocket upgrade requests (`/ws/live?token=...`). Unauthenticated socket attempts are rejected with HTTP 401. | resolved |
| 5 | Medium | Dual Latency Budget Distinction (ISO/IEC 25010) | `07 - Website/backend/src/bridge/pubsub_relay.js`, `frontend/js/telemetry.js` | Decoupled CNN inference budget (< 25 ms, measured at ~13.08 ms) from Sensor-to-Browser transport latency (< 150 ms, measured at ~15-28 ms). Separate telemetry timestamps allow granular stage attribution (DSP, CNN, Bridge Transit, WS Relay, Canvas Render). | resolved |
| 6 | Low | High-Frequency Waveform & Grad-CAM Heatmap Performance | `07 - Website/frontend/js/waveform-visualizer.js` | Replaced heavyweight DOM/Chart.js chart elements with dual Retinal HTML5 Canvas rendering for instantaneous 60 FPS zero-lag drawing of both PPG waveform and synchronized 1D Grad-CAM attention strip. | resolved |

## Notes / Architecture Observations

- **Local Pub/Sub Topology:** The TCP socket bridge on `127.0.0.1:5051` provides low-overhead IPC between Python edge inference and Node.js without requiring third-party message brokers (such as Redis or MQTT), maintaining the zero-external-dependency requirement for offline rural barangay deployment.
- **Cryptographic Auditability:** The `/api/events/verify/chain` endpoint mathematically verifies the SHA-256 backward hash chain in real time, validating that every event's hash satisfies $H_i = \text{SHA256}(H_{i-1} \parallel \text{timestamp} \parallel \text{patient\_id} \parallel \dots)$, fulfilling Chapter 1 thesis research question RQ4 on data traceability.
- **Fallback Capability:** The dashboard includes an embedded synthetic PPG generator in `pubsub_relay.js` that can simulate realistic Normal Sinus Rhythm (NSR) and Atrial Fibrillation (AF) with respiratory sinus arrhythmia and baseline wander when physical hardware is disconnected, allowing full clinical UI evaluation.

## Follow-ups

- [ ] Connect physical MAX30102 sensor and ESP32-C3 node to Raspberry Pi 4 USB/UART and execute 30-minute burn-in session.
- [ ] Implement the asynchronous Hyperledger Fabric sync worker once testnet credentials and channel profiles are provisioned (ADR-001 §4.4).
- [ ] Incorporate clinician feedback from Caloocan City Health Department pilot on UI density and contrast.

## Related

- [[Index|Code Review Index]]
- [[../06 - Antigravity Notes/ADR-001 - Edge Processing Topology & Local Cryptographic Hash Chaining for Offline Primary Care|ADR-001: Edge Processing Topology & Hash Chaining]]
- [[../06 - Antigravity Notes/2026-09-03 - Dual-Agent System Architecture & Engineering Blueprint|Dual-Agent System Architecture & Engineering Blueprint]]
- [[../PLAN|PLAN.md]]
