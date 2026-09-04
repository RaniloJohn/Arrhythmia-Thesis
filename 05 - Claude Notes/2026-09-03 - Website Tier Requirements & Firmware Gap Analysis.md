---
title: Website Tier Requirements & Firmware Gap Analysis
tags: [architecture, dashboard, ppg, streaming, requirements, claude]
date: 2026-09-03
author: Claude
status: active
---

# Website Tier Requirements & Firmware Gap Analysis (Patient Data + Live PPG Streaming)

This note is the research/theory-lead grounding behind [[../PLAN.md|PLAN.md]]'s checklist
for the clinician web dashboard. It doesn't re-specify system topology or code layout —
that's Antigravity's domain and is already covered in
[[../06 - Antigravity Notes/2026-09-03 - Dual-Agent System Architecture & Engineering Blueprint|the dual-agent blueprint]],
[[../06 - Antigravity Notes/ADR-001 - Edge Processing Topology & Local Cryptographic Hash Chaining for Offline Primary Care|ADR-001]],
and `ANTIGRAVITY.md` §4.5/§6. This note instead (1) checks the dashboard requirement
against the thesis text, and (2) records a gap found while inspecting the actual
hardware/firmware currently running, since that gap changes what "live PPG streaming"
can mean until it's closed.

## Requirement, per thesis

[[../04 - Thesis Reference/Thesis Overview|Thesis Overview]]'s conceptual framework
(Chapter 2, Fig. 2.1) is explicit that classification step 3 feeds "the segmented PPG
**waveform** (not hand-engineered HRV features) ... directly into a pre-trained 1D-CNN."
RQ3 further requires that Grad-CAM "highlight the PPG segments most influential to the
classification" — i.e. the interpretability output is a mask *over the raw waveform*,
meant to be shown to a clinician (RQ4: clinical decision support). Visualization step 5
calls for "real-time heart rate, PPG waveform, and AF status shown via a UI (web
dashboard, clinician login)."

So the dashboard's live-stream view isn't a nice-to-have chart of BPM over time — the
thesis's own architecture and its interpretability claim (RQ3) both depend on the raw
waveform being available end-to-end, from sensor to browser.

## Gap found: current firmware doesn't transmit raw PPG

Inspected directly on the Pi (`~/MyEspProject/MyEspProject.ino`, the ESP32-C3 +
MAX30102 sketch already flashed and running). It:

- Samples the MAX30102 at 200 Hz internally.
- Runs a simple DC-removal + threshold peak detector **on the ESP32 itself**.
- Serial-prints only a derived line every 500 ms: `Pulse Rate: X bpm | Avg Pulse Rate: Y bpm`.
- Never transmits the raw IR/red samples to the Pi at all.

This satisfies a standalone BPM readout (useful for the OLED alert path in
ANTIGRAVITY.md §4.1) but provides nothing for the Pi-side DSP → 1D-CNN → Grad-CAM
pipeline in ADR-001 §3.3, and nothing to render as a live waveform in a browser. Right
now, "live stream session of PPG" would have no signal to stream — only a scalar
already computed with a much cruder heuristic than the thesis's bandpass+SQI+CNN
pipeline.

This is flagged as a blocking item in [[../PLAN.md|PLAN.md]] §0, for Antigravity to close
using the framed serial protocol (`0xAA`/`0x55` delimiters + CRC16) already decided as
the ESP32↔Pi transport mitigation in ADR-001, rather than inventing a new one.

## Architecture fit: where the website tier attaches

The blueprint's 5-tier diagram already reserves this exact slot (Sync/Presentation
tiers). Two things worth stating explicitly for whoever implements it, since they
affect ADR-001's guarantees if done carelessly:

1. **SQLite ownership stays split.** The Python edge process is the only writer of the
   hash-chained `arrhythmia_events` table (ADR-001's tamper-evidence chain depends on
   that). A new `patients` table is owned and written only by the Node.js backend. Both
   share the same WAL-mode SQLite file (already the plan per ADR-001) — that mode is
   exactly what makes concurrent access from two processes safe.
2. **Node.js should not reimplement DSP/ML.** The Python edge process already owns
   filtering, CNN inference, and Grad-CAM (ADR-001 §3.2-3.3). The dashboard backend's
   job is to relay that process's output to authenticated browser clients (a local
   pub/sub bridge) and to serve patient CRUD — not to duplicate signal processing in
   JavaScript.

## Mapping to research questions

- **RQ2** (IoT device design for acquisition + ML): closing the firmware gap is what
  actually completes the acquisition→ML path end to end.
- **RQ3** (interpretable, real-time, edge-level classification): the live view's
  Grad-CAM overlay is the concrete answer to "how does Grad-CAM highlight the PPG
  segments driving a classification" — it needs to be visible, not just computed and
  logged.
- **RQ4** (outputs supporting early detection, notification, decision support): patient
  records + per-patient event history with `sync_status` is the decision-support
  surface a clinician actually uses.
- **RQ5** (ISO/IEC 25010): the dashboard is where Security (bcrypt + RBAC, already
  scoped in ANTIGRAVITY.md §5) and Usability (Grad-CAM must read as clinically
  meaningful, not just a heatmap) get evaluated.

## Handoff

Atomic checklist generated from this analysis: [[../PLAN.md|PLAN.md]].

---
Back to [[Index|Claude Notes Index]] · Back to [[../Home|Home]]
