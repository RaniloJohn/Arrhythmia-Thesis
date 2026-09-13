# AGY Handoff — 2026-09-10

**From:** Claude (Research & Theoretical Architect)
**To:** Antigravity (Software Engineer Architect)

Read `PLAN.md` in the workspace root and implement it. It supersedes the 2026-09-03
plan.

## Read this evidence first

- `02 - Code Review/2026-09-10 - Seeded Demo Data, Dead Sensor Link & Offline Blocking Assets.md`
- `05 - Claude Notes/2026-09-10 - Website Startup Latency & Offline Readiness Diagnosis.md`

## Critical context

The system runs on the **Raspberry Pi**, not on this Windows machine:

- host `raspberrypi`, Tailscale IP `100.77.17.38`, user `ranilo`
- project path `/home/ranilo/Arrhythmia Thesis/`
- both `arrhythmia-edge.service` and `arrhythmia-website.service` run there under systemd

Deploy and verify over SSH. Do **not** start a local Windows server to test — the user
corrected us on this explicitly. Windows is for editing source and vault notes only.

## Order of work

1. `PLAN.md` sections 1 and 2 are independent — start there.
2. Section 3 (Arduino IDE firmware port) is the critical path for section 4. It ends in
   a manual flash performed by the user, so prepare the sketch and its README, then stop
   and report that you are ready for them to flash.

## Additional defects found on 2026-09-10, fix these too

1. **Login blocks the event loop.** `07 - Website/backend/src/routes/auth.js` line 38
   uses `bcrypt.compareSync` (bcryptjs, pure JS, cost 12). Measured at 436 ms per login
   on the Pi, and because it is synchronous it stalls every other request — `index.html`
   goes from 1.4 ms to 380 ms while a login is in flight. Switch to the async
   `bcrypt.compare`.

2. **`orchestrate.bat` has never worked.** Line 14 invokes `agy "%USER_PROMPT%..."` with
   a positional prompt, which the CLI rejects with `Error: unexpected argument`. It must
   be `agy --print "<prompt>"`. Note also that on Windows PowerShell a multi-line prompt
   containing lines that begin with `-` gets mis-parsed as flags, so keep any programmatic
   prompt on a single line or pass it via a file like this one.

## Constraints

- Do not delete anything from the live Pi database unattended. Print the exact command
  for review first.
- Follow `ANTIGRAVITY.md` for vault conventions, ADRs, and `Index.md` maintenance.
- The 1D-CNN weights remain untrained. Do not present Grad-CAM output or AF
  classifications as clinically meaningful, and do not relabel anything to imply they are.

## Out of scope

Training the model on MIMIC PERform, Hyperledger Fabric sync, and any redesign of the
dashboard's visual language.
