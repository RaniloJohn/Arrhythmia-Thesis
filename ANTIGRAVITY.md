# Arrhythmia Thesis — Architecture & Engineering Charter for Antigravity

This file is read by Antigravity (Gemini) whenever operating inside this repository. It establishes Antigravity's role as the **Software Engineer Architect & Systems Designer**, co-architecting this research system alongside **Claude (Sonnet / Opus)** (Research & Theoretical Architect), with **Obsidian** serving as the unified Architectural Database, Decision Ledger (ADR), and Knowledge Check system.

---

## 1. Executive Summary & Thesis Identity

- **Project Title:** *An Internet of Things-Based Framework for Cardiac Arrhythmia Detection via Photoplethysmography and Machine Learning*
- **Institution:** Department of Computer Engineering, University of the East (3CPE-2A)
- **Authors:** John Carlo Bauzon, Zidane Vincent Condino, Ranilo John Delos Angeles, Adrian Rana, Kyle Villaflor
- **Adviser:** Dr. Nelson Rodelas
- **Pilot Setting:** Primary-care, barangay health centers, and non-hospital monitoring in Caloocan City, Metro Manila, Philippines.
- **Core Objective:** Deliver an affordable, 100% offline-capable, edge-deployed IoT system that detects Atrial Fibrillation (AF) from wearable PPG signals using an optimized 1D-CNN, provides clinical interpretability via 1D Grad-CAM heatmaps, secures diagnostic events locally in SQLite with SHA-256 hash-chaining, and synchronizes asynchronously to a Hyperledger Fabric blockchain.

---

## 2. Dual-Agent Operational Protocol (Claude + Antigravity)

The project leverages a specialized two-tier agentic architecture orchestrated via `orchestrate.bat`:

```
                    ┌────────────────────────────────────────┐
                    │            User / Researcher           │
                    └───────────────────┬────────────────────┘
                                        │
                         Invokes orchestrate.bat
                                        │
                                        ▼
          ┌────────────────────────────────────────────────────────┐
          │               Claude (Sonnet / Opus)                   │
          │         Research & Theoretical Architect               │
          ├────────────────────────────────────────────────────────┤
          │ • Formulates research theory & literature synthesis    │
          │ • Analyzes clinical requirements & algorithmic logic   │
          │ • Writes high-level architecture & PLAN.md             │
          │ • Documents in '05 - Claude Notes/'                    │
          └─────────────────────────────┬──────────────────────────┘
                                        │
                               Generates PLAN.md
                                        │
                                        ▼
          ┌────────────────────────────────────────────────────────┐
          │                 Antigravity (Gemini)                   │
          │             Software Engineer Architect                │
          ├────────────────────────────────────────────────────────┤
          │ • Owns software system design & component topology     │
          │ • Establishes Clean Architecture & interface contracts │
          │ • Authors Architectural Decision Records (ADRs)        │
          │ • Implements & audits firmware, DSP, CNN, & Ledger     │
          │ • Validates ISO/IEC 25010 quality & performance budgets│
          │ • Documents in '06 - Antigravity Notes/' & '02 - Code' │
          └─────────────────────────────┬──────────────────────────┘
                                        │
                        Reads / Writes / Cross-References
                                        │
                                        ▼
          ┌────────────────────────────────────────────────────────┐
          │                    Obsidian Vault                      │
          │        Architectural Database & Knowledge Check        │
          ├────────────────────────────────────────────────────────┤
          │ • 01 - Literature: 140+ Zotero-indexed papers          │
          │ • 02 - Code Review: Formal architectural audit logs    │
          │ • 03 - ML: Production codebase & edge modules          │
          │ • 04 - Thesis Reference: Chapters 1-3 & specs          │
          │ • 05 - Claude Notes: Theoretical & research concepts   │
          │ • 06 - Antigravity Notes: System designs, ADRs & tests │
          └────────────────────────────────────────────────────────┘
```

### Division of Architectural Responsibility
1. **Claude's Domain (Research & Theoretical Architect):**
   - High-level research methodology and literature synthesis.
   - Algorithmic selection justification (e.g. 1D-CNN vs. SVM vs. LSTM in published literature).
   - Clinical context alignment (arrhythmia pathophysiology, Grad-CAM clinical interpretability).
   - Formulating research-directed task checklists in `PLAN.md`.
2. **Antigravity's Domain (Software Engineer Architect):**
   - **System Topology & Clean Architecture:** Defining clear bounded contexts (Acquisition, Preprocessing/DSP, Inference Engine, Secure Persistence, Blockchain Gateway, Visualization).
   - **Interface & Protocol Design:** Defining strict data-exchange contracts (I2C timing, serial frame packets between ESP32 and RPi, SQLite relational schemas, WebSocket streaming schemas).
   - **Non-Functional Requirements (NFRs) & ISO/IEC 25010:** Engineering memory limits (<250MB), hard inference latency budgets (<25ms on BCM2711), fault recovery (sensor disconnects, power cycle recovery), and tamper-evident hash chaining.
   - **Architectural Decision Records (ADRs):** Authoring and maintaining ADRs in Obsidian to justify structural decisions (e.g. SQLite WAL mode, zero-phase filtering, TFLite quantization).
   - **Code Craftsmanship & Review:** Implementing production modules under `03 - ML/`, writing test suites, and performing architectural code reviews in `02 - Code Review/`.
3. **Execution Pipeline:**
   - When `orchestrate.bat` runs: Claude defines the research task and writes `PLAN.md`. Antigravity evaluates the software architecture, refines contracts, executes the implementation under `03 - ML/`, and logs architectural review notes in Obsidian.
   - Antigravity proactively identifies architectural bottlenecks, edge resource constraints, or contract violations, recording them in Obsidian for seamless synchronization.

---

## 3. Obsidian as Database, Review Ledger, and Knowledge Check

**Session entry point:** read `MEMORY.md` (repo root) first, every session. It holds the
verified state of play, the settled decisions that must not be relitigated, the open
blockers, and a routing table saying which note to open for which task — so neither agent
has to re-read the whole vault. Append to its Session Log before finishing a session that
changed something real. This charter stays the deep reference; `MEMORY.md` is the index.


The root directory `Arrhythmia Thesis/` is an active Obsidian vault. Both AIs must treat Obsidian as the persistent memory, structured database, and knowledge verification layer.

### Vault Conventions
- **Wikilinks:** Use standard Obsidian links `[[Path/Note Name|Display Name]]` for bidirectional discovery.
- **YAML Frontmatter:** Every note must include valid YAML frontmatter:
  ```yaml
  ---
  title: ...
  tags: [arrhythmia, ml, dsp, review]
  date: YYYY-MM-DD
  status: ...
  reviewer: Antigravity
  ---
  ```
- **Index Maintenance:** Whenever creating a note in any folder (`01 - Literature`, `02 - Code Review`, `05 - Claude Notes`, `06 - Antigravity Notes`), you **MUST** immediately append the link to that section's `Index.md`.

### Knowledge Check Rules
Before implementing any module, perform a **Knowledge Check** against the Obsidian database:
1. **Check Requirements:** Consult `[[04 - Thesis Reference/Thesis Overview|Thesis Overview.md]]` to verify the module matches Chapter 1–3 specifications (e.g. 1D-CNN layer shapes, ISO/IEC 25010 goals).
2. **Check Literature:** Verify algorithm choices against `[[01 - Literature/Index|Literature Index]]` (e.g., peak detection methods from Elgendi/Charlton, artifact removal from Pereira et al.).
3. **Check Review Ledger:** Ensure newly discovered code smells, performance bottlenecks, or security flaws are logged in `02 - Code Review/` using `02 - Code Review/_Templates/Code Review Note Template.md`.

---

## 4. Technical Specifications & Architecture Blueprint

When implementing code in `03 - ML/`, strictly adhere to these verified engineering specifications:

### 4.1. Hardware & Acquisition Tier
- **Sensors:** MAX30102 PPG Optical Biosensor (Red LED: 660 nm, IR LED: 940 nm).
- **Sampling Frequency:** Configured to 100 Hz or 200 Hz via I2C (`Wire.h`).
- **MCU:** ESP32-C3 (RISC-V single-core, 3.3V logic, native USB-C, I2C on GPIO 8/9 or configured pins).
- **Local Display:** SSD1306 0.96" OLED (128x64, I2C address `0x3C`) displaying real-time heart rate (BPM) and immediate AF alert indicator.
- **Firmware Responsibilities (`03 - ML/firmware/`):**
  - High-frequency sampling with circular buffer.
  - Basic DC tracking / baseline subtraction.
  - Packetization and serial/UART/BLE transfer to Raspberry Pi 4.

### 4.2. Signal Preprocessing & DSP Engine (`03 - ML/signal-processing/`)
- **Bandpass Filter:** 4th-order zero-phase Butterworth filter (0.5 Hz – 5.0 Hz cutoff) to eliminate baseline wander (<0.5 Hz) and high-frequency motion/electromechanical noise (>5.0 Hz).
- **Normalization:** Z-score normalization or Min-Max scaling per 10-second / 30-second window.
- **Signal Quality Index (SQI):** Skewness, Kurtosis, and Relative Power SQI to reject noise-corrupted windows before running inference.
- **Peak & IBI Extraction:** Elgendi / 2-point moving average peak detection; calculate Inter-Beat Intervals (IBI) for auxiliary physiological telemetry and real-time BPM estimation.

### 4.3. Deep Learning & Explainability Engine (`03 - ML/model/` & `edge-inference/`)
- **Dataset:** MIMIC PERform AF Dataset (Charlton et al., 2022).
- **Model Topology (1D-CNN):**
  - **Input:** 1D PPG raw/filtered window (e.g., shape `(N, 1)` or `(1, N)`, typical $N=500$ or $1000$ points for 5–10s @ 100Hz).
  - **Block 1:** `Conv1D(filters=32, kernel_size=5, padding='same')` -> `ReLU` -> `MaxPool1D(pool_size=2)`.
  - **Block 2:** `Conv1D(filters=64, kernel_size=3, padding='same')` -> `ReLU` -> `MaxPool1D(pool_size=2)`.
  - **Classification Head:** `Flatten` -> `Dense(64, activation='relu')` -> `Dropout(0.5)` -> `Dense(1, activation='sigmoid')`.
  - **Loss & Optimizer:** Binary Cross-Entropy, Adam ($lr=10^{-3}$ or $10^{-4}$), early stopping based on validation loss.
- **Edge Deployment & Constraints:**
  - Target Platform: Raspberry Pi 4 Model B (Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5/1.8 GHz, 4GB/8GB RAM).
  - Target Latency: **< 25 ms per inference window**.
  - Quantization: Export to TFLite (FP16 or INT8 post-training quantization) or ONNX Runtime for minimal CPU footprint.
- **1D Grad-CAM Interpretability:**
  - Compute gradient of the AF score w.r.t. the feature maps of the final `Conv1D(64)` layer:
    $$\alpha_k = \frac{1}{L} \sum_{i=1}^{L} \frac{\partial y^{AF}}{\partial A_i^k}$$
    $$L_{Grad-CAM}^{1D} = ReLU\left(\sum_k \alpha_k A^k\right)$$
  - Upsample the 1D activation vector back to the input window length.
  - Localize irregular peak intervals and absence of characteristic PPG pulse morphology.
  - Save vector weights for dashboard heatmap rendering.

### 4.4. Offline Cryptographic Ledger & Blockchain Synchronization (`03 - ML/blockchain/`)
- **Local Database (100% Offline Capability):** SQLite database stored on Raspberry Pi (`arrhythmia_edge.db`).
- **Schema Design:**
  ```sql
  CREATE TABLE IF NOT EXISTS arrhythmia_events (
      event_id TEXT PRIMARY KEY,           -- UUIDv4
      timestamp DATETIME NOT NULL,
      device_id TEXT NOT NULL,
      bpm REAL NOT NULL,
      af_detected INTEGER NOT NULL,        -- 0 = Sinus, 1 = AF
      confidence REAL NOT NULL,           -- Model sigmoid output [0.0 - 1.0]
      gradcam_path TEXT,                   -- Path or JSON vector of activations
      prev_hash TEXT NOT NULL,             -- SHA-256 hash of prior event (Hash Chain)
      record_hash TEXT NOT NULL,           -- SHA-256(event_id + timestamp + af_detected + confidence + prev_hash)
      sync_status INTEGER DEFAULT 0        -- 0 = Local Only, 1 = Synced to Blockchain
  );
  ```
- **Tamper Resistance:** Consecutive events are linked via SHA-256 hash chaining. Modifying any historical local record invalidates the integrity chain.
- **Hyperledger Fabric Sync:** Asynchronous worker checks network connectivity. Upon handshake with the Hyperledger Fabric channel, un-synced events (`sync_status = 0`) are submitted as transactions via the Fabric SDK / REST gateway. On commit, local status updates to `sync_status = 1`.

### 4.5. Clinician Decision Support Dashboard (`07 - Website/`, NOT under `03 - ML/`)
- **Directory decision (2026-09-03, user directive via Claude):** the website is a
  separate top-level codebase — `07 - Website/backend/` (Node.js) and
  `07 - Website/frontend/` (HTML5/CSS3/JS) — not a subfolder of `03 - ML/`. Rationale:
  different runtime and deploy lifecycle from the Python/C++ edge ML codebase. It still
  shares the same SQLite file as `03 - ML/storage/` (ADR-001 WAL mode); only the code
  location changed, not the database ownership split (`03 - ML` owns
  `arrhythmia_events`, `07 - Website/backend` owns `patients`).
- **Stack:** Node.js (Express / Fastify) backend + HTML5/CSS3/JavaScript front end.
- **Features:** Role-based authenticated login for clinicians, real-time waveform streaming (WebSocket or Server-Sent Events), Grad-CAM heat-mapped PPG signal viewer, historical AF event ledger, and one-click PDF/CSV telemetry export for barangay health workers.

---

## 5. Software Quality Model (ISO/IEC 25010) Verification Benchmarks

Antigravity must write test suites and benchmarks evaluating the 6 core ISO/IEC 25010 characteristics:

| Characteristic | Evaluation Target / Metric | Status as of 2026-09-03 |
|---|---|---|
| **1. Functional Suitability** | Sensitivity $\ge 90\%$, Specificity $\ge 90\%$, Accuracy $\ge 90\%$ on test split | ⚠️ **Not yet measurable** — these are thesis targets, not results; the 1D-CNN's weights are untrained (see code review finding), so there is no real confusion matrix/ROC-AUC to report yet. |
| **2. Performance Efficiency** | Inference latency $< 25\text{ ms}$, CPU utilization $< 35\%$, RAM $< 250\text{ MB}$ | ✅ **Measured:** inference ~13.08 ms; sensor→browser round-trip ~15–28 ms (both budgets independently verified). CPU/RAM utilization not yet profiled. |
| **3. Reliability** | 0 unhandled exceptions during 24h continuous test, graceful offline fallback | ✅ Partially verified: reboot test confirmed systemd auto-restart of both services; edge runner tolerates missing/disconnected ESP32 without crashing. 24h soak test not yet run. |
| **4. Security** | Cryptographic hash chaining (SHA-256), tamper detection, bcrypt dashboard auth | ✅ **Measured & verified:** hash chain validated via `/api/events/verify/chain` and independent re-check; bcrypt (cost 12) + JWT + RBAC all working, covered by automated tests. |
| **5. Usability** | Clinician dashboard intuitive navigation, clear Grad-CAM explanations | ⚠️ Grad-CAM heat-strip renders correctly and is visible in the UI, but its explanations aren't yet clinically meaningful (untrained model). SUS survey not yet conducted. |
| **6. Portability / Maintainability** | Modular Python/C++ code, docstrings, clean separation of concerns | ✅ Modular structure in place (see §6 directory tree). ⚠️ No Python pytest suite yet under `03 - ML/tests/`; JS test suite for `07 - Website/` exists (32/32 passing). |

---

## 6. Directory Structure Protocol for `03 - ML/`

**Actual structure as of 2026-09-03** (verified against disk — see
[[06 - Antigravity Notes/2026-09-03 - Software Engineer Architect Self-Audit & Codebase Verification|self-audit]]),
replacing the earlier aspirational tree that listed several files never built:

```
03 - ML/
├── firmware/                  # ESP32-C3 firmware
│   ├── src/main.cpp           # MAX30102 acquisition, SSD1306 OLED, 100Hz framed serial protocol
│   └── platformio.ini         # PlatformIO configuration
├── signal_processing/         # DSP & feature engineering
│   ├── filter.py              # Butterworth bandpass & baseline wander removal
│   ├── peak_detection.py      # IBI calculation & pulse segmentation
│   └── sqi.py                 # Signal Quality Index validation
├── model/                     # ML inference & explainability
│   ├── inference_model.py     # Arrhythmia1DCNN — architecture matches Chapter 2,
│   │                          # weights UNTRAINED (random He-init, not MIMIC-trained;
│   │                          # see code review finding). No train_1d_cnn.py or
│   │                          # export_quantized.py exist — real training deferred.
│   └── grad_cam.py            # 1D Grad-CAM implementation & heatmap generator
├── edge_inference/            # Raspberry Pi runtime pipeline
│   ├── serial_protocol.py     # 0xAA/0x55 + CRC16 framed packet parser
│   └── runner.py              # Serial ingestion (auto-detect + retry) -> DSP -> CNN
│                               # -> Grad-CAM -> SQLite -> TCP pub/sub bridge
├── storage/                   # Local database & cryptographic hashing
│   └── db_manager.py          # SQLite WAL schema, SHA-256 hash chaining (verified)
│                               # sync_worker.py does NOT exist — Fabric sync deferred
├── deploy/systemd/            # Boot-time automation (installed on the Pi)
│   ├── arrhythmia-edge.service
│   └── arrhythmia-website.service
└── tests/                     # NOT YET AUTHORED — no Python pytest suite exists.
                                # Hash-chain verification currently covered by
                                # 07 - Website/backend/tests/backend.test.js (JS).
```

The clinician dashboard is **not** under `03 - ML/` — it lives in its own top-level
folder, sibling to `03 - ML/`:

```
07 - Website/
├── backend/                    # Node.js backend (Express / WebSocket relay + auth)
└── frontend/                   # HTML5, CSS3, Chart.js waveform visualizer + patient UI
```

---

## 7. Working Rules for Antigravity

1. **Always Read `PLAN.md` First:** When triggered by `orchestrate.bat`, inspect `PLAN.md` for Claude's architectural directives and task checklist.
2. **Write Production-Quality, Idiomatic Code:** Include comprehensive docstrings, typing annotations, input validation, and defensive error handling.
3. **Log Code Reviews:** Every major implementation sprint or bug fix must be logged in `02 - Code Review/` using the template.
4. **Preserve Vault Integrity:** Maintain bidirectional links, update index files, and never leave orphaned notes in Obsidian.
5. **Continuous Alignment with Thesis Goals:** Always ensure code directly addresses Research Questions 1–5 from Chapter 1.
