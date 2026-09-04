---
id: ADR-001
title: Edge Processing Topology & Local Cryptographic Hash Chaining for Offline Primary Care
date: 2026-09-03
status: accepted
architect: Antigravity
tags: [adr, architecture, edge-computing, cryptography, offline-first, iso25010]
---

# ADR-001: Edge Processing Topology & Local Cryptographic Hash Chaining for Offline Primary Care

## Status
- **Date:** 2026-09-03
- **Status:** Accepted
- **Architect:** Antigravity (Software Engineer Architect)
- **Consulted:** Claude (Research & Theoretical Architect), Research Authors (3CPE-2A, UE)

## Context & Problem Statement
Commercial arrhythmia monitoring wearables (Apple Watch, Fitbit, BioTelemetry) rely heavily on continuous cloud connectivity, expensive proprietary hardware, and centralized server-side deep learning models. In Philippine primary-care facilities (specifically barangay health centers in Caloocan City), continuous high-speed internet is unavailable, power stability fluctuates, and medical record systems are predominantly paper-based.

The thesis requires a unified system capable of:
1. Acquiring high-fidelity optical PPG signals from affordable hardware.
2. Executing real-time ML inference and 1D Grad-CAM interpretability locally with low latency.
3. Guaranteeing data integrity and auditability offline, while seamlessly syncing with a permissioned Hyperledger Fabric blockchain when intermittent WAN connectivity is established.

## Decision Drivers (ISO/IEC 25010 Quality Model)
- **Performance Efficiency:** Total inference cycle must execute in $< 25\text{ ms}$ on edge hardware to support real-time alarms.
- **Reliability:** The system must never crash or drop events due to network loss; database must survive sudden power cuts without corruption.
- **Security & Non-Repudiation:** Offline local logs must be tamper-evident to prevent falsification of diagnostic records before blockchain sync.
- **Functional Suitability:** Must produce continuous BPM, binary AF classification, and 1D Grad-CAM visual heatmaps.
- **Cost & Portability:** Hardware bill of materials (BOM) must remain accessible for community health centers.

## Considered Options

### Option 1: Direct Cloud Ingestion (ESP32 -> MQTT -> Cloud ML -> Cloud DB)
- *Pros:* Offloads ML compute from edge; simplifies edge hardware.
- *Cons:* **Violates core thesis requirement.** Total failure in offline/rural Philippine environments; unacceptable transmission latency; recurring cloud hosting costs.

### Option 2: Monolithic Microcontroller ML (ESP32-C3 On-Chip Inference + Flat File SD Card)
- *Pros:* Lowest cost and form factor.
- *Cons:* ESP32-C3 (160MHz RISC-V, 400KB SRAM) lacks sufficient compute and memory for multi-channel continuous DSP, 1D-CNN float operations, 1D Grad-CAM gradient backpropagation, and web dashboard hosting. Flat files on SD cards are highly prone to sector corruption on brownouts and lack ACID properties.

### Option 3: Dual-Tier Edge Architecture (ESP32-C3 Acquisition Node + Raspberry Pi 4/5 Edge Gateway) with SQLite WAL Hash Chaining
- *Architecture:*
  - **Tier 1 (Wearable Acquisition Node):** ESP32-C3 + MAX30102 PPG sensor + SSD1306 OLED display. Dedicated to deterministic 100 Hz optical sampling and instantaneous heuristic pulse display.
  - **Tier 2 (Edge Gateway):** Raspberry Pi 4/5 (Quad-core ARM Cortex-A72/A76). Hosts the DSP pipeline (0.5–5Hz Butterworth), 1D-CNN inference engine, 1D Grad-CAM explainer, and SQLite database — all in the `03 - ML/` Python codebase.
  - **Tier 2b (Clinician Web Dashboard):** a **separate** Node.js + HTML5/CSS/JS codebase in `07 - Website/` (decided 2026-09-03, after this ADR was first written) — same Pi, different runtime/deploy lifecycle from Tier 2's Python pipeline, communicating via a local TCP pub/sub bridge (port 5051) rather than sharing a process. See §"Implementation Guidance" below.
  - **Persistence & Security Layer:** SQLite in Write-Ahead Logging (`WAL`) mode with cryptographic SHA-256 backward hash chaining. **Ownership split:** the Python process (`03 - ML/storage/db_manager.py`) is the sole writer of `arrhythmia_events` (the hash-chained table); the Node.js website (`07 - Website/backend/src/db.js`) is the sole writer of `patients`. Both share the same `arrhythmia_edge.db` file safely because of WAL mode.
  - **Distributed Ledger Integration (not yet implemented — see below):** planned asynchronous synchronization worker that will negotiate a TLS handshake with a Hyperledger Fabric orderer/peer when network is detected.

## Decision Outcome
Chosen Option: **Option 3 (Dual-Tier Edge Architecture with SQLite WAL Hash Chaining)**.

### Positive Consequences
1. **100% Offline Autonomy:** The system performs acquisition, filtering, deep learning inference, Grad-CAM generation, and tamper-resistant persistence without an internet connection.
2. **Deterministic Sampling:** Offloading sensor I2C reads to the ESP32-C3 eliminates Linux OS scheduling jitter, guaranteeing uniform $\Delta t = 10\text{ ms}$ sampling.
3. **Cryptographic Tamper-Evidence:** Linking consecutive event hashes ($\text{Hash}_n = \text{SHA-256}(\text{Data}_n \,\|\, \text{Hash}_{n-1})$) ensures that any manual modification of the local SQLite database invalidates all subsequent hashes, providing cryptographic proof of tampering.
4. **Resilience to Brownouts:** SQLite `PRAGMA journal_mode = WAL;` and `PRAGMA synchronous = NORMAL;` ensure atomic transactions resilient to sudden battery disconnection.
5. **Decoupled Blockchain Sync:** Asynchronous queueing ensures blockchain latency or network timeouts never block critical patient monitoring loops.

### Negative Consequences / Tradeoffs & Mitigations
- *Tradeoff:* Requires inter-device communication between ESP32-C3 and Raspberry Pi 4.
  - *Mitigation:* Use robust framed UART packets with CRC16 checksums and start/end delimiter bytes (`0xAA ... 0x55`).
- *Tradeoff:* Raspberry Pi 4 draws more power than an MCU alone.
  - *Mitigation:* Power optimization via headless OS configuration, disabling unneeded peripherals (HDMI/BT if unused), and CPU governor tuning to stay within Li-Po power envelope.

## Implementation Guidance & Boundary Contracts

**Status of each contract, verified against the codebase on 2026-09-03 (self-audit —
see [[2026-09-03 - Software Engineer Architect Self-Audit & Codebase Verification|Self-Audit]]):**

- Firmware: `03 - ML/firmware/src/main.cpp` — ✅ implemented (100 Hz framed serial protocol).
- DSP & Engine: `03 - ML/signal_processing/` & `03 - ML/edge_inference/` — ✅ implemented.
- SQLite Cryptographic Schema: `03 - ML/storage/db_manager.py` — ✅ implemented (WAL mode, SHA-256 backward chain, independently verified).
- Clinician Web Dashboard: `07 - Website/backend/` (Node.js) + `07 - Website/frontend/` — ✅ implemented, deployed live on the Pi via systemd. **Not** part of `03 - ML/` — see Option 3's Tier 2b above.
- Boot-time automation: `03 - ML/deploy/systemd/arrhythmia-edge.service` & `arrhythmia-website.service` — ✅ implemented, reboot-tested.
- Fabric Sync Worker: `03 - ML/storage/sync_worker.py` — ⏸️ **Planned / Deferred.** Does not exist yet; local hash-chaining works standalone, `sync_status` column exists in the schema but nothing currently flips it. Out of scope until Fabric channel/chaincode details are specified.
- Python Verification Suite: `03 - ML/tests/test_crypto_chain.py` — ⏸️ **Not yet authored.** Hash-chain verification is currently covered by `07 - Website/backend/tests/backend.test.js` (JS, Phase 4) and an inline `validate_chain()`-equivalent in `db_manager.py`, but no standalone Python pytest suite exists yet.
- 1D-CNN weights: `03 - ML/model/inference_model.py` — ⚠️ **Architecture matches spec; weights are untrained** (random He-init, never fit to the MIMIC PERform AF Dataset). See [[../02 - Code Review/2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights|code review finding]]. Real training explicitly deferred per user decision (2026-09-03) — not in scope for now.

## References
- [[../../04 - Thesis Reference/Thesis Overview|Thesis Overview: Section 2 & 3]]
- Charlton et al. (2022) — *Detecting Beats in the Photoplethysmogram: Benchmarking Open-Source Algorithms* (the actual source of the MIMIC PERform AF Dataset — see [[../../01 - Literature/Pending Zotero Additions/Charlton-2022-Detecting Beats in the Photoplethysmogram|literature note]]; don't confuse with the other Charlton 2022 *Proc. IEEE* review paper)
- Egala et al. (2021) — *Fortified-Chain: A Blockchain-Based Framework for Smart Healthcare*
- ISO/IEC 25010 Software Quality Model
