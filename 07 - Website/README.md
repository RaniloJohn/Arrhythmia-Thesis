# Website — Clinician Decision Support Dashboard

The clinician-facing web dashboard — implemented by Antigravity, deployed and running
live on the Raspberry Pi at `http://raspberrypi:8080` (or `http://100.77.17.38:8080`
over Tailscale). Kept as its own codebase, separate from [[../03 - ML/README|03 - ML/]]
(different runtime/deploy lifecycle — Node.js web server vs. the Python/C++ edge
pipeline) per the 2026-09-03 folder-placement decision recorded in [[../PLAN.md|PLAN.md]].

## Layout

- `backend/` — Node.js (Express) API + WebSocket relay + auth:
  - `src/server.js` — entry point; serves the frontend, the REST API, and the
    WebSocket live-stream relay.
  - `src/db.js` — SQLite access (WAL mode, per ADR-001). Owns the `patients` table;
    reads (never writes) `arrhythmia_events`, which `03 - ML/storage/db_manager.py`
    owns and hash-chains.
  - `src/routes/` — `auth.js`, `patients.js`, `events.js`, `edge.js`.
  - `src/bridge/pubsub_relay.js` — local TCP bridge (port 5051) that ingests real-time
    telemetry frames from `03 - ML/edge_inference/runner.py` and re-broadcasts them to
    authenticated browser WebSocket clients.
  - `tests/` — `backend.test.js` (32 tests: auth/RBAC, patient CRUD, hash-chain
    verification, edge ingestion, WebSocket security) and `e2e_integration_test.js`.
- `frontend/` — HTML5/CSS3/JS client: clinician login, patient registry (CRUD), live
  PPG streaming view (real-time waveform + Grad-CAM heat-strip overlay), AF event
  ledger, and the ISO/IEC 25010 telemetry view. Light Clinical design system (Newsreader
  + Public Sans, flat teal/red/green/amber palette, hand-authored SVG icons — no
  gradients, no emoji; see `PLAN.md` §6 for the full spec).

## Running it

```bash
cd backend
npm install
npm test    # 32/32 should pass
npm start   # serves the frontend too, on :8080
```

On the Pi, this runs automatically at boot via the `arrhythmia-website.service`
systemd unit in `03 - ML/deploy/systemd/` (installed to `/etc/systemd/system/`).

## Reference

- Architecture rationale: [[../06 - Antigravity Notes/ADR-001 - Edge Processing Topology & Local Cryptographic Hash Chaining for Offline Primary Care|ADR-001]]
- Requirements & firmware gap analysis: [[../05 - Claude Notes/2026-09-03 - Website Tier Requirements & Firmware Gap Analysis|Website Tier Requirements & Firmware Gap Analysis]]
- Implementation checklist & status: [[../PLAN.md|PLAN.md]]
- Formal review log: [[../02 - Code Review/2026-09-03 - Clinician Web Dashboard & Edge PubSub Gateway|Clinician Web Dashboard & Edge PubSub Gateway]]
