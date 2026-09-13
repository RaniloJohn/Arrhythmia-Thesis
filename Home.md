---
name: Home
description: Dashboard for the Arrhythmia Thesis vault — start here.
---

# Arrhythmia Thesis — Home

**IoT-Based Framework for Cardiac Arrhythmia Detection via Photoplethysmography and
Machine Learning** · Bauzon, Condino, Delos Angeles, Rana, Villaflor (3CPE-2A, UE) ·
Adviser: Dr. Nelson Rodelas

This folder is both the working project folder and the Obsidian vault for the thesis —
literature notes, formal code reviews, dual-agent architecture notes (Claude & Antigravity),
and thesis references all live here so everything stays searchable, cross-referenced, and linked.

## Dual-Agent Pipeline

The project runs an automated pair-programming pipeline via `orchestrate.bat`:
- **Claude (Sonnet / Opus):** Chief Research Architect, high-level system modeling, literature synthesis, and plan generation (`PLAN.md`).
- **Antigravity (Gemini):** Software Engineer Architect, software topology, interface contracts, ADRs, edge implementation, and ISO/IEC 25010 benchmarking.
- **Obsidian Vault:** Shared database, epistemic knowledge check, and persistent memory for both agents.

## Start here (agents)

- [[MEMORY|MEMORY.md]] — the AI session memory: state of play, settled decisions, open
  blockers, routing table, and session log. Both agents read this first each session so
  they don't have to re-crawl the vault.

## Sections

- [[01 - Literature/Index|Literature]] — notes on papers and sources read for the study (140+ Zotero sources)
- [[02 - Code Review/Index|Code Review]] — findings and audit logs from reviewing the codebase
- `03 - ML/` — source code (signal processing, 1D-CNN, edge inference, firmware, blockchain)
- [[04 - Thesis Reference/Thesis Overview|Thesis Overview]] — working summary of Chapters 1–3
- [[05 - Claude Notes/Index|Claude Notes]] — architecture diagrams, research syntheses from Claude
- [[06 - Antigravity Notes/Index|Antigravity Notes]] — implementation blueprints, benchmarks, DSP designs from Antigravity

## Quick links

- Thesis document: `04 - Thesis Reference/BAUZON_CONDINO_DELOSANGELES_RAÑA_VILLAFLOR-CHAPTER1-3 REVISED.pdf`
- Agent Instructions for Claude: [[CLAUDE.md|CLAUDE.md]]
- Agent Instructions for Antigravity: [[ANTIGRAVITY.md|ANTIGRAVITY.md]]
- Orchestration script: `orchestrate.bat`

## Status

- [x] Establish Obsidian vault structure, literature index, and dual-agent charter (`CLAUDE.md` & `ANTIGRAVITY.md`)
- [ ] Populate literature notes as papers are read (see the bibliography in the thesis
      document for the current source list — ~150 references already in Chapter 2)
- [x] Implement core modules under `03 - ML/` (DSP, 1D-CNN, Grad-CAM, SQLite ledger, ESP32 firmware)
- [x] Log formal code reviews under `02 - Code Review/` using the review template
- [x] Execute ISO/IEC 25010 benchmarking suites (<25ms inference latency, SHA-256 integrity)
