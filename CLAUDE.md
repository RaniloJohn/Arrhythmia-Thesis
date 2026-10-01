# Arrhythmia Thesis — Project Context for Claude Code

Auto-loaded whenever Claude Code runs in this folder. Kept deliberately short so it
costs little every session — **the durable project state lives in `MEMORY.md`**.

## Session start protocol

1. Read **`MEMORY.md`** (repo root) — state of play, settled decisions, open blockers,
   the routing table, and the session log. That plus this file is enough to start work.
2. Open other notes **only** as the routing table in `MEMORY.md` directs. Do not bulk-read
   `01 - Literature/`, `06 - Antigravity Notes/`, or the code trees speculatively.
3. Before finishing a session that changed something real, append to `MEMORY.md`'s
   Session Log and update its State of Play if a status flipped.

## ⚠️ Read this first — where the system actually runs

**The system does not run on this Windows machine.** It runs on a Raspberry Pi 5 that
nobody on the team can physically touch — it is at a teammate's location and is reached
only over Tailscale. Windows holds the repository and the Obsidian vault; it is an
authoring environment, not a runtime. Never start a local copy of the website here to
reproduce behaviour, and never report Windows-side timings as if they described the
deployment.

- Host `raspberrypi` / `100.77.17.38`, user `ranilo`, path `/home/ranilo/Arrhythmia Thesis/`
- **Credentials are in `CREDENTIALS.md` in this folder, which is gitignored.** It is
  not committed and does not appear in the repository — open it locally when you need
  a login. Never copy a credential from it into this file or any other tracked file.
- Dashboard: `http://100.77.17.38:8080` (use **Firefox** on the Pi — Chromium there
  cannot load any URL at all, including `example.com`; that is a browser fault, not a
  website fault)
- Both services run under systemd and are **enabled at boot**:
  `arrhythmia-edge.service` and `arrhythmia-website.service`
- Firmware is compiled and flashed **remotely** with `arduino-cli` on the Pi — no
  physical access is required for firmware iteration
- You are able to get to sudo, so no worries. just use the credentials provided.

**Full deployment architecture, wire protocol, flashing procedure, validity-gating
contract, and the current list of what is broken:**
[[05 - Claude Notes/2026-09-10 - Raspberry Pi Deployment Architecture and Runbook|Raspberry Pi Deployment Architecture and Runbook]]
— read it before changing anything that touches the Pi, the firmware, or the telemetry
contract.

**Three things that must not be misrepresented:**
1. The deployed classifier is **`ibi_af_v1`** — interval-irregularity features into a
   logistic regression, not the 1D-CNN. Patient-disjoint 5-fold CV: subject AUROC
   **0.970 (95% CI [0.908, 1.000], n=35)**; 100% sens / 93.8% spec at the subject level
   with 3-of-5 consensus. **Always quote the interval and n=35.** It was trained on MIMIC
   PERform AF (bedside pulse-oximetry PPG, transmissive) and has **never been validated on
   this project's own MAX30102**, which is a **reflectance** sensor. Per the team's 2026-09-11
   decision the primary site is now the fingertip, so the residual gap is reflectance vs
   transmissive at a matching site. State that limitation whenever accuracy is described.
   **The 1D-CNN is NOT validated and `cnn_af_v1.npz` must never be cited.** Its Chapter 2
   topology has a 0.12 s receptive field, so no convolutional feature can observe even one
   inter-beat interval. The 2026-09-13 figures (85.4% sensitivity, AUROC 0.494, CV
   0.330) are void — that model scored 0.349 on its own training data. So are the DeepBeat
   dataset figures: it holds **20.0 h** of unique signal (2.4 h AF), not 3,725 h.
2. Latency: the edge pipeline runs in **3.6–7.3 ms** on the dev machine, well inside the
   25 ms budget. **This has not yet been re-measured on the Pi** with `ibi_af_v1`, so do
   not quote it as a deployment figure. `ANTIGRAVITY.md` §5's "~13.08 ms" is stale.
3. The live database still contains 3 fabricated patients and 2 seeded accounts whose
   passwords are in git history. Cleanup is deferred at the owner's request.

## Project

**Title:** An Internet of Things-Based Framework for Cardiac Arrhythmia Detection via
Photoplethysmography and Machine Learning

**Team (3CPE-2A, University of the East, Computer Engineering Dept.):**
Bauzon, John Carlo · Condino, Zidane Vincent · Delos Angeles, Ranilo John · Rana,
Adrian · Villaflor, Kyle — **Adviser:** Dr. Nelson Rodelas

A low-cost, offline-capable IoT system that detects atrial fibrillation from wrist/finger
PPG at the edge, explains each classification at beat level, and secures
event records in a SHA-256 hash-chained SQLite cache that will sync to Hyperledger Fabric
— for primary-care settings in the Philippines (pilot: Caloocan City).

Full background, research questions, scope and methodology:
`04 - Thesis Reference/Thesis Overview.md`. Current build status: `MEMORY.md`.

⚠️ The deployed classifier is `ibi_af_v1` (interval irregularity + logistic regression),
**not** the 1D-CNN, and it has never been validated on the project's own MAX30102 hardware.
See `MEMORY.md` for the full caveat before describing accuracy anywhere.

## This folder is an Obsidian vault

The whole folder doubles as the project's Obsidian vault (Open folder as vault). Notes
are Markdown with `[[wikilinks]]`; each section has an `Index.md` — link new notes into
it so the vault stays navigable.

- **`01 - Literature/`** — one note per source (142 notes). Template:
  `_Templates/Literature Note Template.md`. Name `Author-Year-ShortTitle.md`.
- **`02 - Code Review/`** — review findings, TODOs, decisions. Template:
  `_Templates/Code Review Note Template.md`. Name `YYYY-MM-DD - <component>.md`.
- **`03 - ML/`** — edge ML/DSP/firmware codebase (Python + C++, ESP32-C3 / Raspberry Pi).
  See `03 - ML/README.md`. Does **not** contain the website.
- **`04 - Thesis Reference/`** — the thesis PDF and `Thesis Overview.md`.
- **`05 - Claude Notes/`** — outputs worth keeping from Claude sessions.
- **`06 - Antigravity Notes/`** — Antigravity's blueprints, ADRs, benchmarking logs.
- **`07 - Website/`** — clinician dashboard (Node.js backend + HTML/CSS/JS frontend),
  its own codebase. See `07 - Website/README.md`.
- **`08 - Hardware/`** — parametric CAD for the physical build (CadQuery → STEP/STL).
  `enclosure/params.py` is the single source of truth for every dimension; `build.py`
  runs geometric pre-flight checks before it will export. `print-ready/` holds what
  actually goes to the print shop. See `08 - Hardware/enclosure/README.md`.

## Working conventions

- Findings from reviewing or debugging code → a note in `02 - Code Review/`, not just chat.
- A paper read or summarized → a note in `01 - Literature/`.
- Any diagram, architecture note, or non-trivial summary → a file in `05 - Claude Notes/`.
- Link every new note into its section's `Index.md`.
- **Dual-agent collaboration:** Antigravity (Gemini) serves as the Software Engineer
  Architect working with Claude via `orchestrate.bat`. While Claude leads research
  and theoretical architecture, Antigravity owns software topology, clean architecture,
  interface contracts, ADRs, and edge implementation. When writing implementation steps
  or plans, output a concise, atomic checklist into `PLAN.md` for Antigravity to execute.
  See [[ANTIGRAVITY.md|ANTIGRAVITY.md]] for Antigravity's operational charter.
- Keep `MEMORY.md` current; keep this file short.

## Tech stack (confirmed against actual code)

- **Sensing hardware:** MAX30102 PPG sensor (red 660 nm / IR 940 nm), ESP32-C3
  microcontroller (I²C on GPIO 8/9), SSD1306 0.96" OLED display, Raspberry Pi 4/5,
  3.7V Li-Po battery (500–1000 mAh). Dual-toolchain firmware: PlatformIO project
  (`03 - ML/firmware/src/main.cpp`) and native Arduino IDE sketch
  (`03 - ML/firmware/arduino/ArrhythmiaNode/ArrhythmiaNode.ino`), both adhering to the
  identical packed 19-byte binary protocol (`0xAA`/`0x55` + CRC-16-CCITT, 115200 baud,
  100 Hz deterministic cadence). In Arduino IDE, **Tools → USB CDC On Boot → Enabled**
  is mandatory for USB streaming to `/dev/ttyACM0`.
- **Signal processing:** Python on the Raspberry Pi (`03 - ML/signal_processing/`) —
  Butterworth bandpass filtering, detrending, Signal Quality Index (SQI), Elgendi peak
  detection → inter-beat intervals (IBI)/BPM. Real and working.
- **ML classification:** 1D-CNN (`03 - ML/model/inference_model.py`) implementing Chapter 2
  architecture (2× Conv1D+ReLU+MaxPool → Dense(64, ReLU) → Sigmoid) and 1D Grad-CAM
  (`grad_cam.py`). Trained on DeepBeat wrist reflectance PPG (`03 - ML/model/weights/cnn_af_v1.npz`,
  calibrated threshold $\tau = 0.3700$). Validated on held-out external MIMIC PERform AF cohort.
  Runs in pure NumPy on the Pi (<25 ms budget verified).
- **Data / security:** SQLite (`03 - ML/storage/db_manager.py`) with WAL mode and
  SHA-256 backward hash-chaining per diagnostic event (verified: tamper-evident,
  independently re-checked). Configurable target patient (`TARGET_PATIENT_ID` / `--patient`),
  first-run admin bootstrap via environment, mandatory `JWT_SECRET` (no fallback).
  **Hyperledger Fabric sync is not yet implemented** — explicitly deferred; `sync_status`
  exists in the schema but nothing currently flips it to synced.
- **Dashboard:** `07 - Website/` — Node.js (Express) backend + HTML5/CSS3/JS frontend,
  bcrypt + JWT + RBAC clinician auth (async non-blocking bcrypt), WebSocket live streaming
  with a Grad-CAM heat-strip overlay, patient CRUD, AF event ledger. 100% offline-capable
  with self-hosted WOFF2 fonts (`assets/fonts/`), zero external CDNs, and synthetic
  simulation gated behind `ENABLE_SIMULATION=true` with amber warning banner. Deployed and
  running live on the Pi via systemd (`03 - ML/deploy/systemd/`), auto-starting on boot.
  32/32 backend tests passing (independently re-run).
- **Methodology:** Agile Scrum SDLC; evaluated against the ISO/IEC 25010 software
  quality model (Functional Suitability, Performance Efficiency, Reliability, Security,
  Usability, Maintainability/Portability).

## Research questions the system must ultimately answer for (Chapter 1)

1. What PPG signal inputs / physiological parameters are needed for reliable AF
   detection via a PPG-beat detection framework?
2. How should an IoT-enabled device be designed to acquire PPG signals and run ML
   classification for AF?
3. How can an interpretable ML approach classify AF from PPG-beat features at the edge
   in real time — specifically, how does Grad-CAM highlight the PPG segments driving a
   classification?
4. What outputs/functionality support early AF detection, user notification, and
   clinical decision support outside hospitals?
5. How does the system perform against the six ISO/IEC 25010 sub-characteristics
   (Functional Suitability, Performance Efficiency, Reliability, Security, Usability,
   Maintainability/Portability)?

See `04 - Thesis Reference/Thesis Overview.md` for the fuller write-up (problem
statement, scope/limitations, conceptual framework, methodology).
