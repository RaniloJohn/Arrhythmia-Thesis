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

## Project

**Title:** An Internet of Things-Based Framework for Cardiac Arrhythmia Detection via
Photoplethysmography and Machine Learning

**Team (3CPE-2A, University of the East, Computer Engineering Dept.):**
Bauzon, John Carlo · Condino, Zidane Vincent · Delos Angeles, Ranilo John · Rana,
Adrian · Villaflor, Kyle — **Adviser:** Dr. Nelson Rodelas

A low-cost, offline-capable IoT system that detects atrial fibrillation from wrist/finger
PPG using an edge-deployed 1D-CNN, explains classifications with Grad-CAM, and secures
event records in a SHA-256 hash-chained SQLite cache that will sync to Hyperledger Fabric
— for primary-care settings in the Philippines (pilot: Caloocan City).

Full background, research questions, scope and methodology:
`04 - Thesis Reference/Thesis Overview.md`. Current build status: `MEMORY.md`.

⚠️ The 1D-CNN's weights are **untrained** — never present its classification or Grad-CAM
output as a validated result. See `MEMORY.md` for the full caveat.

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

## Working conventions

- Findings from reviewing or debugging code → a note in `02 - Code Review/`, not just chat.
- A paper read or summarized → a note in `01 - Literature/`.
- Any diagram, architecture note, or non-trivial summary → a file in `05 - Claude Notes/`.
- Link every new note into its section's `Index.md`.
- **Dual-agent collaboration:** Antigravity (Gemini) is the Software Engineer Architect,
  working with Claude via `orchestrate.bat`. Claude leads research and theoretical
  architecture; Antigravity owns software topology, clean architecture, interface
  contracts, ADRs, and edge implementation. Implementation steps go into `PLAN.md` as a
  concise atomic checklist for Antigravity to execute. Charter: `ANTIGRAVITY.md`.
- Keep `MEMORY.md` current; keep this file short.
