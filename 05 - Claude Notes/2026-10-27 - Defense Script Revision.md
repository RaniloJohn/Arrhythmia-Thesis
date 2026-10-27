# 2026-10-27 — Defense Script Revision

The full revised script lives in a shared Claude Doc so the team can edit and comment on it:
**https://claude.ai/artifact/DV1cXsD3uEw2hNKRwFBBut**

## What was wrong with the previous (proposal-defense) script

| Old page | Claim | Correction |
|---|---|---|
| 10, 16 | 1D-CNN is the primary algorithm; "approaches medical-grade ECG" | Deployed model is `ibi_af_v1`; Ch. 2 CNN has a 0.12 s receptive field (settled decision 11) |
| 6, 12, 17, 18 | "Blockchain-based protection", "encrypted and tamper-proof" | Hash-chained SQLite is tamper-evident, not encrypted; Fabric sync not built |
| 4, 7, 9 | "24/7 / non-stop monitoring", "life-saving" | Seated spot-check screening; no 24 h soak test |
| 4 | "Most studies do not focus on wearables" | False (Steinhubl 2018, Apple Heart Study); gap is closed/costly/cloud-dependent |
| 3 | "Nearly double the rate from 1990" | Death *count* roughly doubled; age-standardised rate fell |
| 12 | ESP32 does "initial preprocessing" | ESP32 only samples + frames packets; all DSP is on the Pi |
| 18 | "Expected output" | Real results now exist and are presented |

## Structure of the revision

11 parts, about 25 slides: opening → background → gaps + RQ table → literature (each slide ends in
a design decision) → framework/method → system as built (+ architecture diagram, live demo plan)
→ classifier and why the CNN was dropped → results (AUROC with CI and n, Pi 5 latency, wrist
re-test story) → limitations → conclusion revisiting the RQ table → 13 prepared Q&A answers.

## Open items the team must fill before the defense

- Speaker assignment; total time limit.
- Original source for the 19.2 M CVD-deaths figure; citations for AF stroke risk (~5×) and the
  ≥30 s AF-episode convention.
- Whether MATLAB was actually used (no MATLAB code in the vault).
- Whether the concept validation survey has been run.
- The team's actual reason for the 2026-10-12 wrist decision.
- Bill-of-materials total in pesos.
- Whether the fabricated patients can be hidden before any live dashboard demo.

## 2026-10-28 — citation check (Parts 12–13 added to the doc)

Checked against the abstracts in `01 - Literature/`. Lines the old script/manuscript get wrong:
- [[Steinhubl-2018-Effect of a Home-Based Wearable|Steinhubl 2018]] (mSToPS) is an **ECG patch**, not PPG.
- [[Turakhia-2019-Rationale and design of a large-scale,|Turakhia 2019]] is the Apple Heart Study **design** paper; results = Perez et al. 2019 (not in vault).
- [[Roth-2020-Global Burden of Cardiovascular|Roth 2020]]: "nearly doubled" = prevalent cases (271 M → 523 M); deaths 12.1 M → 18.6 M (1990–2019). No vault source for "19.2 M in 2023".
- Kumar is **2022** (Deepaware), not 2021. Zhu 2022 = 87.8% sens / 97.4% spec; Tran 2023 gives no figures.
- Fivefold stroke risk: only secondary ([[Lei-2021-Hybrid Decision Support to Monitor|Lei 2021]]). ≥30 s AF episode: ESC 2020 guideline, not in vault.
- Useful support not previously cited in the script: [[Bashar-2019-Atrial Fibrillation Detection from|Bashar 2019]] (RMSSD + entropy from wrist PPG intervals), [[Cinotti-2025-An Edge AI Approach for Low-Power,|Cinotti 2025]] (edge AF from heartbeat intervals).

---
Back to [[Index|Claude Notes Index]] · [[../MEMORY|MEMORY]]
