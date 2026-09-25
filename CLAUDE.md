# Arrhythmia Thesis — Project Context for Claude Code

This file is read by Claude Code (the CLI) whenever it runs inside this folder. Keep it
up to date as the project evolves — it's the fastest way to get a fresh session oriented.

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
1. The 1D-CNN is **untrained** (random init). AF probabilities are noise. The waveform
   is real; the classification is not.
2. Inference measures **27.7–42.1 ms** against the 25 ms budget — the budget is
   currently breached. The "~13.08 ms" figure in `ANTIGRAVITY.md` §5 is stale
   (synthetic data) and needs correcting.
3. The live database still contains 3 fabricated patients and 2 seeded accounts whose
   passwords are in git history. Cleanup is deferred at the owner's request.

## Project

**Title:** An Internet of Things-Based Framework for Cardiac Arrhythmia Detection via
Photoplethysmography and Machine Learning

**Team (3CPE-2A, University of the East, Computer Engineering Dept.):**
Bauzon, John Carlo · Condino, Zidane Vincent · Delos Angeles, Ranilo John · Rana,
Adrian · Villaflor, Kyle
**Adviser:** Dr. Nelson Rodelas

**One-line summary:** A low-cost, offline-capable IoT system that detects atrial
fibrillation (AF) from wrist/finger PPG signals using an edge-deployed 1D-CNN, explains
its classifications with Grad-CAM, and secures event records with a local SQLite cache
that syncs to a Hyperledger Fabric blockchain — aimed at primary-care / non-hospital
settings in the Philippines (pilot locale: Caloocan City).

## This folder is an Obsidian vault

The whole `Arrhythmia Thesis` folder doubles as the project's Obsidian vault — it's the
database for reading (literature) and for tracking code review notes. Open it in Obsidian
directly (Open folder as vault). Notes are Markdown with `[[wikilinks]]` for
cross-referencing; each section has its own `Index.md` — link new notes into it so the
vault stays navigable.

## Folder layout

- **`01 - Literature/`** — one note per source read for the thesis (citation, summary,
  key findings, relevance). New notes start from `_Templates/Literature Note
  Template.md`. Name files `Author-Year-ShortTitle.md` and link them from
  `01 - Literature/Index.md`.
- **`02 - Code Review/`** — findings from reviewing the codebase (with Claude Code or
  otherwise): issues, TODOs, architecture notes, decisions. New notes start from
  `_Templates/Code Review Note Template.md`. Name files `YYYY-MM-DD - <component>.md`
  and link them from `02 - Code Review/Index.md`.
- **`03 - ML/`** — the edge ML/DSP/firmware codebase: ESP32-C3 firmware, signal
  processing, 1D-CNN + Grad-CAM, and the SQLite/hash-chain storage layer (Python + C++,
  runs on the ESP32/Raspberry Pi). See `03 - ML/README.md`. Does **not** contain the
  website — see `07 - Website/` below.
- **`04 - Thesis Reference/`** — reference material: the thesis document itself and
  `Thesis Overview.md`, a working summary of Chapters 1–3 for quick context.
- **`05 - Claude Notes/`** — outputs produced during Claude sessions (diagrams,
  analyses, summaries) that are worth keeping. Save them here as Markdown instead of
  letting them disappear at the end of the terminal scrollback.
- **`06 - Antigravity Notes/`** — implementation blueprints, engineering analyses, and
  benchmarking logs produced by Antigravity (Gemini).
- **`07 - Website/`** — the clinician-facing web dashboard, kept as its own codebase
  separate from `03 - ML/` (different runtime/deploy lifecycle — Node.js web server vs.
  Python/C++ edge pipeline). `backend/` is the Node.js API + WebSocket relay + auth;
  `frontend/` is the HTML5/CSS3/JS client (patient CRUD, login, live PPG view with
  Grad-CAM overlay). Shares the same SQLite file as `03 - ML/storage/` (WAL mode, per
  ADR-001) but owns a separate `patients` table — see `PLAN.md` and
  `05 - Claude Notes/2026-09-03 - Website Tier Requirements & Firmware Gap Analysis.md`.
- **`08 - Hardware/`** — parametric CAD for the physical build. `enclosure/` is the
  wrist-worn housing for the acquisition node (ESP32-C3 + MAX30102 over the radial
  artery + SSD1306), authored in **CadQuery** (Python 3.12 + OCCT) and exported to
  STEP for the print shop and Fusion 360, plus STL for the slicer. `params.py` is the
  single source of truth for every dimension — change it there and re-run `build.py`,
  which runs geometric pre-flight checks before it will export. See
  `05 - Claude Notes/2026-09-11 - Wrist Enclosure Design Specification.md`.

## Working conventions for Claude Code

- When reviewing code changes or debugging, write findings to `02 - Code Review/`
  using the template — don't leave them only in chat output.
- When reading or summarizing a new paper for the lit review, create a note in
  `01 - Literature/` using the template.
- Any diagram, architecture note, or non-trivial summary produced during a session
  should be saved as a file in `05 - Claude Notes/`, not just printed to the terminal.
- Link every new note into its section's `Index.md`.
- **Dual-agent collaboration:** Antigravity (Gemini) serves as the Software Engineer
  Architect working with Claude via `orchestrate.bat`. While Claude leads research
  and theoretical architecture, Antigravity owns software topology, clean architecture,
  interface contracts, ADRs, and edge implementation. When writing implementation steps
  or plans, output a concise, atomic checklist into `PLAN.md` for Antigravity to execute.
  See [[ANTIGRAVITY.md|ANTIGRAVITY.md]] for Antigravity's operational charter.
- Keep this file (`CLAUDE.md`) current: as source code is added under `03 - ML/`,
  fill in the real directory layout, build/run commands, and dependencies below.

## Tech stack (confirmed against actual code as of 2026-09-03)

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
- **ML classification — ⚠️ untrained placeholder, not yet a result:** a 1D-CNN
  (`03 - ML/model/inference_model.py`) implementing the Chapter 2 architecture
  (2× Conv1D+ReLU+MaxPool → Dense(64, ReLU) → Sigmoid) and a matching 1D Grad-CAM
  (`grad_cam.py`) — but the weights are **randomly initialized, not trained on the
  MIMIC PERform AF Dataset** (see
  [[02 - Code Review/2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights|code review note]]).
  The full pipeline (DSP → inference → Grad-CAM → storage → live dashboard) runs
  correctly end to end, but its AF classification/explainability output is not yet
  evidence-based — don't present current demo output as a validated result.
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
  Usability, Maintainability/Portability) — note the model-training gap above means
  Functional Suitability (sensitivity/specificity) is not yet measurable against real
  results.

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
