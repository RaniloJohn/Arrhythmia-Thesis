---
name: MEMORY
description: Persistent AI memory and cheap session-bootstrap file — read this first, load everything else on demand.
updated: 2026-10-01
---

# MEMORY.md — AI Session Memory & Vault Bootstrap

**This file is the entry point for every Claude / Antigravity session in this vault.**
It exists so a session can get fully oriented from *one small file* instead of reading
the whole vault (which costs ~50k tokens). Read this, then open only the specific notes
the current task needs — see the routing table.

**Read protocol:** `CLAUDE.md` (auto-loaded) + this file = enough context to start work.
Do **not** bulk-read `01 - Literature/`, the Antigravity notes, or the code trees unless
the task requires them.

**Write protocol:** at the end of any session that changes something real, append 3–6
lines to the **Session Log** below and update **State of Play** if a status flipped.
Keep this file under ~200 lines — it is read every session, so it must stay cheap.
Long-form output still goes to `05 - Claude Notes/` (Claude) or `06 - Antigravity Notes/`
(Antigravity), and is linked from here only by one line.

---

## Routing table — read this file, not the others

| If the task is… | Open only |
|---|---|
| Anything (start here) | `MEMORY.md` (this file) |
| Thesis claims, RQs, scope, Ch. 1–3 wording | `04 - Thesis Reference/Thesis Overview.md` |
| Current implementation checklist / what's next | `PLAN.md` |
| Why a structural decision was made | `06 - Antigravity Notes/ADR-001 - …Hash Chaining….md` |
| Edge code (DSP, CNN, Grad-CAM, storage, firmware) | `03 - ML/README.md` → then the specific `.py`/`.cpp` |
| Dashboard code (API, auth, WebSocket, UI) | `07 - Website/README.md` → then the specific file |
| Known defects / audit history | `02 - Code Review/Index.md` |
| Antigravity's engineering specs & self-audit | `06 - Antigravity Notes/Index.md` |
| A specific paper | `01 - Literature/Index.md` (142 notes, grouped by RQ) — grep, don't read whole |
| Antigravity's operating rules | `ANTIGRAVITY.md` (only when writing for Antigravity) |

---

## State of play (verified 2026-10-01)

| Component | Status |
|---|---|
| ESP32-C3 firmware — MAX30102 @ 100 Hz, framed serial `0xAA`/`0x55` + CRC16, OLED fallback | ✅ built (`03 - ML/firmware/src/main.cpp`) |
| DSP — Butterworth 0.5–5 Hz, detrend, SQI, Elgendi peaks → IBI/BPM | ✅ built (`03 - ML/signal_processing/`) |
| **AF classifier — `ibi_af_v1` (IBI irregularity + logistic regression, pure NumPy)** | ✅ **DEPLOYED & VALIDATED.** Subject AUROC **0.970** (95% CI [0.908, 1.000], n=35, patient-disjoint 5-fold CV, out-of-fold). τ=0.32 → sens 95.9% / spec 83.6%. With 3-of-5 consensus: **sens 100% / spec 93.8%**. Latency **3.6–7.3 ms**. Default in `runner.py`. |
| 1D-CNN (Ch. 2 topology) | ❌ **structurally unable to detect AF as written — amendment recommended** (settled decision 11). Receptive field is 12 samples (0.12 s); an inter-beat interval is 60–120. Subject AUROC **0.533** as written → **0.842–0.849 (fold 0.883 ± 0.145)** at a 6.86 s receptive field. Recommended amendment: kernels 127/63, pooling 8/8, `Flatten` → **global average pooling** (plain GAP matches mean+std pooling once the receptive field is adequate, and is the canonical Grad-CAM setting). Still ~0.12 below the 12-parameter feature model. **`cnn_af_v1.npz` was never validly trained — never cite it.** |
| Interpretability (RQ3) | ✅ **answered**: exact per-feature contributions (`explain`) + faithful beat-level counterfactual attribution (`explain_intervals`, emits a 1000-sample heat strip the dashboard renders unchanged). CNN Grad-CAM implemented for all heads with closed-form gradients verified vs autograd — but the v1 *clinical plausibility verdict* stays void. |
| SQLite + SHA-256 backward hash chain (WAL) | ✅ built & independently verified |
| Edge runner — real serial auto-detect + retry, TCP pub/sub :5051 | ✅ **`ibi_af_v1` live on the Pi since 2026-10-06** (journal: `VALIDATED=True`, AUROC 0.9704). Defaults to `ibi_af_v1` (`--classifier ibi|cnn`). `model_trained` is gated on a validated subject-AUROC floor, so the CNN path always reports `model_trained: false`. 3-of-5 consensus; latency budget met with ~4x headroom. |
| Website — Express API, bcrypt+JWT+RBAC, WS `/ws/live`, patient CRUD, event ledger, Grad-CAM heat-strip | ✅ built, 32/32 tests pass |
| Deployment — systemd units, auto-start on Pi boot | ✅ built, reboot-tested |
| Hyperledger Fabric sync worker | ⏸️ **deferred** — `sync_status` column exists, nothing flips it |
| Python pytest suite (`03 - ML/tests/`) | ✅ **97/97 passing** across 16 modules. New: `test_ibi_classifier.py` (14) and `test_parity_v2.py` (11, all three heads + negative control + autograd gradient checks). |
| Model training / dataset pipeline | ✅ **rebuilt** (`build_dataset_v2.py`, `build_mimic_dense.py`, `baseline_ibi.py`, `train_ibi_model.py`, `sweep_*.py`). File-scoped group keys, no temporal redundancy, scale-aware SQI gating, subject-level metrics with subject-resampled CIs, plus label/subject-integrity and rate-confound test gates. |

**Live deployment:** Raspberry Pi, Tailscale `raspberrypi` / `100.77.17.38`, dashboard on
`:8080`, project at `/home/ranilo/Arrhythmia Thesis/`.

### The one thing that must never be misstated
**There is now a real, validated classifier — and a hard limit on what it has been validated
*on*.**

`ibi_af_v1` achieves subject-level AUROC **0.970 (95% CI [0.908, 1.000], n=35)** under
patient-disjoint cross-validation, with sens 100% / spec 93.8% at the subject level using
3-of-5 consensus. Always quote the **interval and n=35**, never the point estimate alone. The
rate-confound was tested: rate-free features give 0.914, rate-only 0.674, so the signal is
interval irregularity (the AF mechanism), not heart rate.

**The limit, which must be stated in the thesis:** it was trained on MIMIC PERform AF —
bedside pulse-oximetry PPG at 125 Hz, almost certainly **transmissive fingertip** — and has
**never been validated on this project's own MAX30102**, which is a **reflectance** sensor.
Since the team's 2026-09-11 decision moves the primary measurement site to the fingertip, the
remaining gap is **reflectance vs transmissive at a matching site**, which is real but far
smaller than the wrist-vs-finger gap. The measured 0.583 transfer figure is to *wrist*
reflectance (DeepBeat) and now describes the **secondary** wrist comparison, not the primary
deployment. No further modelling closes this; it needs data from the real device.
Also note MIMIC's own external-validity limit: 35 critically-ill ICU adults, not a
primary-care screening cohort.

**The 1D-CNN remains unvalidated and `cnn_af_v1.npz` must never be cited.** The Chapter 2
topology has a 0.12 s receptive field, so no convolutional feature can observe even one
inter-beat interval; its 1.03M-parameter flatten head scored worst of the nine configurations measured.
The v1 Grad-CAM "clinical plausibility" verdict stays void — it was computed on a chance-level
model. Also never repeat these v1 figures: DeepBeat is **20.0 h** of unique signal (2.4 h AF),
not 3,725 h; windows overlap 96% at a 1 s stride across 8 channels.

Audit: `02 - Code Review/2026-10-01 - 1D-CNN Training Audit - Label-Subject Collinearity and Invalid Weights.md`
Rebuild + evidence packs: `02 - Code Review/2026-10-01 - ML Rebuild - Working AF Classifier and Chapter 2 Topology Amendment.md`

---

## Settled decisions — do not relitigate

1. **Website lives in `07 - Website/`**, not `03 - ML/dashboard/` (user directive,
   2026-09-03). Separate runtime and deploy lifecycle; same SQLite file.
2. **DB ownership split:** Python writes `arrhythmia_events` (hash-chained); Node.js
   writes `patients`. Shared `arrhythmia_edge.db` in WAL mode. ADR-001.
3. **Dual-tier edge topology** (ESP32-C3 acquisition + Pi gateway), not cloud, not
   on-MCU inference. ADR-001 Option 3.
4. **Pure-NumPy inference runtime**, no TFLite/ONNX — meets the <25 ms budget with zero
   native deps. TFLite stays an option, not a requirement.
5. **IPC is a local TCP bridge on 127.0.0.1:5051** — no Redis/MQTT broker (offline
   deployment constraint).
6. **Fabric sync and Grad-CAM re-validation are explicitly deferred**, not forgotten.
7. **UI is "Light Clinical"**: flat teal `#0E6B76` on warm off-white `#FAF9F6`,
   Newsreader + Public Sans, hand-authored SVG icons. No gradients, no emoji, no glow.
   Full spec: `PLAN.md` §6.
8. **Dual-agent split:** Claude = research/theory/`PLAN.md`/`05 - Claude Notes/`;
   Antigravity = implementation/ADRs/`06 - Antigravity Notes/`/`02 - Code Review/`.
9. **~~DeepBeat trains; MIMIC PERform AF externally validates.~~ REVERSED 2026-10-01 —
   MIMIC PERform AF trains; DeepBeat is a cross-sensor robustness cohort.** The 2026-09-13
   decision rested on three premises, all since measured false: DeepBeat is **20.0 h** of
   unique signal with **2.4 h of AF** across 11 recordings (not ~500k windows from ~175
   subjects — its windows overlap 96% at a 1 s stride over 8 channels, and its real `train`
   partition was never downloaded); every DeepBeat recording is single-rhythm, so its labels
   are per-recording just like MIMIC's; and although its wrist reflectance geometry does
   match the MAX30102, **32 Hz cannot resolve AF** — one sample is 31 ms against the 50 ms
   pNN50 criterion. Decisive test: a 3-parameter irregularity model scores subject AUROC
   **0.914 on MIMIC and 0.465 (chance) on DeepBeat**. No feature set or architecture found
   usable signal in DeepBeat. **Independently corroborated:** the team's own 2026-09-11
   sensing decision (`05 - Claude Notes/2026-09-11 - Wrist vs Fingertip Sensing - SOP and RRL
   Alignment.md`) had already moved the primary measurement site to the **fingertip** and
   named MIMIC PERform AF as the matching corpus — on entirely separate grounds (DeepBeat
   publishes no reference ECG, so it is unbenchmarkable; the device is a tethered screening
   instrument, not a continuous wearable). Two independent lines of reasoning converge, which
   makes this a much easier case to the adviser than decision 11. It also **erases the Ch. 3
   documentation debt** rather than adding to it: Chapter 2 and §4.3 already name MIMIC
   PERform. Evidence pack is §1 of the 2026-10-01 rebuild note.

10. **Train in PyTorch, infer in NumPy.** Torch is a dev-machine-only dependency; the Pi
   runtime stays pure NumPy, so Decision 4 is intact. Weights cross the boundary as an
   `.npz` + `.meta.json`, guarded by a parity test. The Chapter 2 topology is frozen —
   dropout and weight decay are training-time only. Splits are always patient-isolated
   by subject ID. Decided 2026-09-13.

11. **Chapter 2's CNN topology is amended, and the deployed classifier is not a CNN.**
   Decided 2026-10-01 on measurement. The frozen topology's receptive field is 12 samples
   (0.12 s) while an inter-beat interval is 60–120 samples, so no convolutional feature can
   observe one interval, let alone the variability across several that defines AF; and
   `Flatten → Dense(16000, 64)` is 1.02M of its 1.03M parameters against 35 subjects. Seven
   variants were compared under one patient-disjoint protocol: **receptive field drives
   performance, capacity does not** (0.533 as written → 0.842–0.849 at a 6.86 s receptive field, with the per-fold spread narrowing from ±0.232 to ±0.145 as it widens — generalising better, not just fitting better;
   a 10,625-parameter pooling head beats the 1.03M-parameter one). The amendment keeps both
   conv blocks and `Dense(64) → Sigmoid`, but widens kernels/pooling (127/63, pools 8/8) and
   replaces `Flatten` with **global average pooling** — GAP matches the more elaborate
   mean+std head once the receptive field is adequate, is the smaller edit to Ch. 2, and is
   the setting Grad-CAM was formulated for. **The deployed classifier is `ibi_af_v1`** (12
   interval features → logistic regression, subject AUROC 0.970), because at n=35 subjects a
   raw-waveform CNN cannot match statistics the peak detector already computes exactly.
   **Owed to the adviser.**

12. **Subject-level metrics with subject-resampled bootstrap CIs are the only figures that
   may be quoted.** Labels are constant within a recording, so window counts overstate the
   sample size by ~120x — this is what let v1 present a 14-subject result as 17,106 samples.
   Training may use overlapping windows (splits are patient-disjoint); **every reported
   number must come from non-overlapping windows.** Any operating point must pass a
   non-degeneracy check on both sensitivity and specificity before it ships. Decided
   2026-10-01.

---

## Open questions / blockers

Resolved items are deleted, not archived, so stale figures cannot be re-quoted from here.

**Blocking the thesis claim**
- [ ] **HIGHEST VALUE — collect labelled data from the project's own MAX30102.** `ibi_af_v1`
      is trained on transmissive bedside PPG; the MAX30102 is a reflectance sensor, so
      reflectance-vs-transmissive transfer is unmeasured. Per the 2026-09-11 decision the
      primary site is now the **fingertip**, so this needs the **fingertip housing, which is
      not built yet** (the committed enclosure is the wrist one, retained as the secondary
      comparison). Even a few hours from consenting AF and non-AF subjects enables the
      transfer measurement. **No further modelling closes this gap.**
- [ ] **Fingertip housing not yet designed/printed.** The 2026-09-11 decision makes fingertip
      primary and the wrist enclosure a secondary comparison, but only the wrist part exists.
      The CAD is parametric and the sensor boss / light-seal / aperture logic reuses directly.
- [ ] **Adviser sign-off on two reversals** — settled decisions **9** (dataset roles) and
      **11** (Ch. 2 topology amendment + the deployed classifier not being a CNN). Evidence
      packs: §1 and §3 of `02 - Code Review/2026-10-01 - ML Rebuild - ...`.
- [ ] **Ch. 3 amendment (small).** The chapter's naming of MIMIC PERform as the training
      dataset is *correct again* after the reversal, so there is no dataset debt. What needs
      writing is the justification (≥100 Hz sampling requirement, per-recording labels,
      ECG-verifiable labels) and DeepBeat's demotion to a secondary wrist comparison. Add the
      MIMIC external-validity caveat (35 critically-ill ICU adults) to Ch. 1 Limitations.

**Engineering / deployment**
- [ ] **Hardware verification** (`PLAN.md` §7): ESP32-C3 into the Pi, restart both services,
      confirm a real waveform and a live `ibi_af_v1` decision in the dashboard.
- [ ] **Fabric channel/chaincode contract** undefined — blocks `sync_worker.py`.
- [ ] **No notification channel** exists yet (RQ4 is only partly answered without one).
- [ ] **Unmeasured for RQ5:** CPU/RAM utilisation, 24 h soak test, SUS usability survey.
- [ ] **Live DB still holds 3 fabricated patients and 2 seeded accounts** whose passwords are
      in git history. Cleanup deferred at the owner's request.
- [ ] Stray default-Obsidian files in `Vault/` — safe to delete once confirmed with the user.

---

## Idea inbox

Short lines only; promote anything substantial to a real note and link it here.

- A per-patient calibration pass (a short baseline recording per patient, used to centre the
  interval features) would likely recover much of the wrist-vs-fingertip transfer loss —
  cheap to test once real device data exists.
- Grad-CAM on an amended CNN could be shown *beside* the beat-level attribution as a
  secondary view, since the two explain at different granularities.
- A "validated / unvalidated model" badge in the dashboard, driven by the telemetry's
  `model_validated_subject_auroc`, so a committee member can never read an unvalidated
  score as a result.

---

## Session log

### 2026-10-06 — Claude (Opus 5.5): deployed `ibi_af_v1` to the Pi
- Pi checkout was **not a git repo** (copied over, Sept code: untrained CNN + old SQI bug). Made it
  one in place; the 3 modified files were hash-matched to old commits (no Pi-only edits) before overwrite.
- **Pi had no internet** — eth0 is cabled to the Windows laptop (ICS `192.168.137.1` gives no DHCP);
  reached only via IPv6 link-local `raspberrypi.local`, which drops repeatedly. Code moved by
  `git bundle` + `scp`. Tailscale is not installed on the laptop; `CREDENTIALS.md` is absent.
- `arrhythmia-edge.service` restarted → `IBI classifier loaded: ibi_af_v1 | VALIDATED=True`.
  Classifier-only on the Pi: **0.37 ms/call** (full-pipeline latency still not measured on the Pi).
- Found: passwordless `sudo -n` restart no longer works; pytest not installed on the Pi; Pi clock
  ~1 day behind (no NTP offline); target is still fabricated patient `PAT-CAL-001`.
- Added `03 - ML/deploy/deploy_ibi_on_pi.sh`. **Next:** confirm serial frames + a live decision
  in the dashboard (`PLAN.md` §7), measure Pi latency, fix the Pi's eth0 profile/network.

### 2026-10-01 (later) — Claude (Opus 5), as ML lead: rebuilt the ML pipeline; there is now a working classifier
- **Delivered `ibi_af_v1`**: 12 interval-irregularity features → logistic regression, pure
  NumPy. Patient-disjoint 5-fold CV, all out-of-fold: **subject AUROC 0.970 (95% CI
  [0.908, 1.000], n=35)**, window AUROC 0.936; τ=0.32 → sens 95.9% / spec 83.6%; with 3-of-5
  consensus **sens 100% / spec 93.8%** (19 TP, 1 FP, 15 TN, 0 FN). Latency 3.6–7.3 ms.
  Now the default classifier in `runner.py`.
- **Tested the obvious confound rather than assuming it:** rate-free irregularity features
  give 0.914, rate-only 0.674. The signal is interval irregularity, not heart rate.
- **Measured why the Ch. 2 CNN cannot work:** receptive field is 0.12 s against a 0.6–1.2 s
  inter-beat interval. Nine configurations under one protocol showed receptive field — not
  capacity — drives performance: 0.533 as written → **0.849 at 6.86 s**, with the fold spread
  narrowing ±0.232 → ±0.145 (generalising better, not just fitting better). A
  10.6k-parameter pooling head beats the 1.03M-parameter flatten head. Plain GAP matches
  mean+std pooling once the receptive field is wide enough, so the amendment is small.
  Settled decisions 11 and 12 added.
- **Found three further data facts that invalidate v1 reporting:** DeepBeat is 20.0 h of
  unique signal (2.4 h AF), not 3,725 h — its windows overlap 96% at a 1 s stride across 8
  channels; its real `train` partition was never downloaded; and at 32 Hz one sample is 31 ms
  against the 50 ms pNN50 criterion, so it physically cannot resolve AF. A 3-parameter
  irregularity model gets 0.465 (chance) on DeepBeat vs 0.914 on MIMIC. **Reversed settled
  decision 9** on that evidence.
- **Fixed** the SQI perfusion-index scale bug (no window in a 1.07M-window corpus could score
  above 0.6), and `model_trained` telemetry, which is now gated on a validated subject AUROC
  floor instead of "a file loaded"; the CNN path always reports `model_trained: false`.
- **RQ3 answered with a stronger mechanism than planned:** exact per-feature contributions
  plus a *faithful* beat-level counterfactual attribution (no gradient approximation), which
  emits a heat strip the existing dashboard overlay renders unchanged.
- Tests: **97/97 passing**, including a new parity gate for all three heads with a negative
  control, autograd-verified Grad-CAM gradients, and rate-confound/label-integrity gates.
- Wrote `02 - Code Review/2026-10-01 - ML Rebuild - Working AF Classifier and Chapter 2
  Topology Amendment.md` (evidence packs for both adviser decisions) and trimmed this file's
  superseded 2026-09-13 prompt log.
- **Next:** adviser sign-off on decisions 9 and 11, then collect data from the real MAX30102
  — that is the only remaining way to close the wrist-vs-fingertip gap.

### 2026-10-01 — Claude (Opus 5): audit found the 2026-09-13 training run invalid
- Reproduced Antigravity's evaluation numbers exactly (test AUROC 0.5345, MIMIC 0.4942) — the
  eval code was honest; the defect was upstream in dataset construction.
- Measured the figure never taken: **AUROC 0.3487 on the model's own training distribution.**
  Sub-chance on fitted data means contradictory labels, not a domain gap — so the v1
  "domain generalization collapse" story was unsupported.
- Causes: DeepBeat's `parameters[:, 2]` is **file-local**, so IDs 146–153 are different people
  with opposite labels across the two archives and regrouping merged them; and AF is
  perfectly collinear with subject (0 of 16 carry both classes), making the test set's
  effective n **14, not 17,106**. Shipped weights came from a capped smoke run (4 subjects,
  4.9% of windows, `best_epoch: 1`); τ=0.37 was the always-say-AF corner (spec 10.5%, PPV 5.2%).
- Full detail: `02 - Code Review/2026-10-01 - 1D-CNN Training Audit - ...`.

### 2026-09-13 — Antigravity (Gemini): Prompts 0–8 (dataset layer → training → tests)
*Condensed 2026-10-01; superseded by the audit and rebuild. Detail in git history and
`02 - Code Review/2026-09-13 - 1D-CNN Training Results & Honest Performance Bounds.md`.*
- Built the full training stack: dataset loaders + `to_100hz`, `build_dataset.py`,
  `torch_model.py` + `export_weights.py`, `train.py`, `calibrate_threshold.py`, `evaluate.py`,
  `cross_validate.py`, `gradcam_validation.py`, and a 71-test pytest suite.
- **Kept and reused:** the PyTorch↔NumPy parity gate with its negative control, autograd
  verification of the Grad-CAM gradient, the anti-contamination guards, and the honest CI /
  calibration reporting that made the audit possible.
- **Superseded:** every performance number, `cnn_af_v1.npz`, τ=0.37, and the domain-gap story.

### 2026-09-13 — Claude (Opus 5): v1 ML training architecture and dataset choice
*Condensed 2026-10-01; both decisions here were later reversed on measurement.*
- Pinned the training contract from the real code (1000 samples @ 100 Hz, `detrend_ppg` →
  Butterworth → `zscore_normalize`) and the train-in-PyTorch / infer-in-NumPy boundary.
- Caught the flatten-order bug (time-major vs channel-major) before any training time was
  spent, which is why the parity gate exists and still earns its keep.
- Reversed MIMIC→DeepBeat as the training cohort. That reversal was itself reversed on
  2026-10-01 once DeepBeat's real size and 32 Hz limit were measured — see decision 9.

### 2026-09-09 — Claude (Opus 5): vault documentation sweep + memory bootstrap
- Read the full documentation set, verified it against the file trees, and created this file
  as the single cheap session entry point, trimming `CLAUDE.md` to a lean bootstrap.

---
Back to [[Home|Home]] · Agent charters: [[CLAUDE.md|CLAUDE.md]] · [[ANTIGRAVITY.md|ANTIGRAVITY.md]]
