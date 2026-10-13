---
title: Wrist Re-test of the Three Non-AF Subjects
date: 2026-10-13
reviewer: Claude (Opus 5.5)
component: edge runner (quality gates, 3-of-5 consensus), ibi_af_v1, beat detection
status: open — false alarms gone for S1–S3; S4 alarm unexplained; residual beat-error fix proposed
---

# Wrist Re-test of the Three Non-AF Subjects

Follows on from [[2026-10-12 - False AF on Three Non-AF Subjects - Diagnosis]]. The same
three non-AF testers who were all flagged AF on 2026-10-12 were re-tested on the fixed
pipeline, plus a fourth, new subject whose rhythm status is unknown.

## Setup

- **Code on the Pi:** commit `ac7abd7`, clean tree. The Pi had rebooted shortly before the
  session (`uptime -s` 22:45 Pi-clock), after the 10-12 fixes were written to disk, so the
  morphology/saturation gates, corrected 3-of-5 non-overlapping consensus, PI fix,
  `PPG_INVERT=1` (code default) and AF-window auto-capture were all live. `ibi_af_v1`
  loaded `VALIDATED=True`. **No restart was needed** — the "restart pending" note from 10-12
  had been overtaken by the reboot.
- **Site/posture:** wrist over the radial pulse, seated, forearm resting. 2 min per run,
  one window decision per second (10 s windows, 1 s stride).
- **Recording method:** `journalctl -u arrhythmia-edge -f` for 120 s per run — the live
  service was never paused, so the dashboard stayed live.
- **Pi clock is still ~8 days behind** (reads 2026-10-05). All ledger/capture timestamps
  from this session carry that offset. Ledger rows are under the placeholder `PAT-CAL-001`.
- **Network:** the Pi has only an IPv6 link-local address on the laptop cable (no IPv4, no
  Tailscale on the laptop), which browsers cannot open. The dashboard was reached through
  an SSH tunnel: `ssh -N -L 8080:localhost:8080 ranilo@raspberrypi.local` → `http://localhost:8080`.

## Results

"Usable" = passed every quality gate and was classified. "Alarm" = 3-of-5 consensus fired
and a hash-chained event was written to the ledger.

| Run | Windows | Usable | Called AF (single window) | **Alarms** | BPM median (range) | Max end-to-end |
|---|---|---|---|---|---|---|
| S1 (1st, poor contact) | 119 | 15 | 5 | **0** | 57 (27–82) | 19.7 ms |
| S1 (2nd) | 119 | 67 | 0 | **0** | 102 (47–108) | — |
| S2 (1st, poor contact) | 85 | 5 | 3 | **0** | 63 (54–67) | 12.9 ms |
| S2 (redo) | 119 | 115 | 18 (16%) | **0** | 86 (75–91) | 20.1 ms |
| S3 (stopped at user request) | 87 | — | — | — | — | — |
| S3 (full) | 120 | 71 | 2 (3%) | **0** | 70 (62–84) | 18.7 ms |
| **S4 (new subject)** | 120 | 33 | 5 | **1** | 57 (40–90) | 21.2 ms |

Ledger: 518 → 519 events. The only new row is S4's alarm.

**S1–S3 (the three subjects falsely alarmed on 10-12): 0 alarms.** On 10-12 these same
three produced ~200 false AF ledger events. All windows under the 25 ms budget.

## Findings

### 1. Poor contact is now rejected instead of alarmed (fix confirmed)
The two poor-contact runs (S1 1st, S2 1st) had 87–94% of windows rejected
(`unreliable_beats`, `no_skin_contact`). The few that slipped through and were called AF
did not reach consensus. This is exactly the failure mode of 10-12, now contained.

### 2. Residual single-window false AF = one beat-detection error in a regular rhythm
Every auto-captured AF window from S2 and S3 shows a regular rhythm with one or two
outlier intervals:

- **≈2× median** — a missed beat (S2: 1300, 1220 ms against ~680 ms; S3: 1520 ms against ~850 ms).
- **short–long pair summing to ≈2× median** — a misplaced peak *or* a genuine premature
  beat with compensatory pause (S2: 920+520, 980+480 ms against ~720 ms). PPG alone
  cannot tell these apart; neither is AF.

The beat-morphology gate does not catch these (beat shapes are fine, median r 0.98–0.99),
and the `undetected_pulses` check did not flag the 2× intervals.

**Proposed fix (not implemented):** ectopic/artefact interval filtering before feature
extraction — drop intervals ≈2× the window median and short–long pairs summing to ≈2×
median, as standard PPG-AF pipelines do. **This changes model inputs, so it must be
re-validated on MIMIC PERform AF (AF retention, subject-level CV) before deployment.**
Risk: in true AF, filtering can remove genuine irregularity — measure sensitivity loss.

### 3. S1 heart-rate disagreement between runs
S1's median BPM was 57 on the 1st run and 102 on the 2nd. 102 at seated rest is high;
it may be anxiety, or the detector counting two peaks per pulse. **Unresolved — no manual
reference was taken.** Future runs should include a 30 s manual radial pulse count.

### 4. S4 alarm — unreliable signal, rhythm status unknown
- 86 of 120 windows rejected as `unreliable_beats`; beat-shape similarity fell as low as
  0.18; median IR drifted 124k → 117k (strap pressure / arm movement).
- The 5 AF windows were the borderline ones (BeatShape 0.73–0.80, mostly `missed≥1`).
- Intervals of 300 ms beside 1600 ms at a median 57 BPM fit beat-counting errors better
  than a physiological rhythm.
- The alarmed event (`confidence 0.58`, BPM 62.2) was driven by 1.13 s and 1.59 s intervals.

**Cannot be called false or true yet.** Owed: S4's known rhythm status, and a redo with
re-seated sensor + simultaneous manual pulse palpation (irregularly irregular or not).

### 5. Fewer windows than expected in one run
S2 1st run logged 85 windows in 120 s (expected ~120) with no frame-counter jumps. Not
investigated.

## Scope of the evidence

This is **specificity only**, on 3 non-AF people (plus one unknown), ~2 min each, at the
wrist. It shows the false-alarm problem of 10-12 is contained for these subjects. It says
**nothing about wrist sensitivity** — that still needs consenting AF subjects, and there
is still no wrist accuracy figure.

## Data

Copied to `03 - ML/data/device_captures/2026-10-13_retest/` (gitignored):
- `journal/S1, S1b, S2, S2b, S3, S3b, S4_journal.log` — every window decision per run
  (S3 = the stopped run, S3b = the full one).
- `auto_af_windows/` — the 18 raw 10 s windows the runner auto-saved for each AF call
  (IR, red, peaks, probability, SQI meta). Filenames carry the Pi's wrong clock.

## Next

1. S4: rhythm status + redo with manual pulse check.
2. Implement and MIMIC-validate ectopic/artefact interval filtering (finding 2).
3. Add manual pulse counts to the test protocol.
4. Fix the Pi clock (needs sudo; no NTP offline) and give test sessions real patient IDs
   instead of `PAT-CAL-001`.
