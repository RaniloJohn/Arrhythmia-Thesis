---
title: Ectopic Interval Filtering - MIMIC Re-validation
date: 2026-10-19
reviewer: Claude (Opus 5.5)
component: ibi_af_v1 feature extraction (proposed pre-filter), edge runner
status: closed — not adopted; kept as a documented negative result on branch exp/ectopic-filter
---

# Ectopic Interval Filtering — MIMIC Re-validation

Follows [[2026-10-13 - Wrist Re-test of the Three Non-AF Subjects]], finding 2, which proposed
correcting beat-detection artefacts (missed beat, misplaced peak) before feature extraction, and
required re-validation on MIMIC PERform AF before any deployment.

**Verdict: do not adopt.** It does not improve the validated model on MIMIC, it costs a little
sensitivity, and on the device it fixes only a minority of the residual false-AF windows — the
kind that consensus already absorbs. Nothing was deployed, `ibi_af_v1` is unchanged, and `main`
is untouched.

## What was tested

Script: `03 - ML/training/eval_ectopic_filter.py` (read-only; results in
`training/runs/ectopic_filter_eval.json`). Relative to the window median interval M, it corrects
a missed beat (one interval ≈2M → two halves), a misplaced peak (short+long ≈2M → two of sum/2),
and an extra peak (short+short ≈M → merged). An **AF guard** applies a correction only when the
other intervals are regular (robust CV below a limit), so AF windows should pass untouched.
Six variants: no filter, naive (no guard), guard at CV 0.08/0.10/0.15, and tolerance 0.15/0.20.

Two modes, both patient-disjoint 5-fold GroupKFold, out-of-fold, same folds as training:
- **infer-only** — keep the `ibi_af_v1` weights, filter at runtime only.
- **retrain** — fit and score on filtered features (would need an `ibi_af_v2`).

**Harness check:** the unfiltered run reproduces the deployed figures exactly — features are
bit-identical to the training cache, subject AUROC 0.9704 [0.908, 1.000], window sens 95.9% /
spec 83.6%, 3-of-5 consensus sens 100% / spec 93.8%. So differences below come from the filter,
not from the harness.

## Results — MIMIC PERform AF (n=35 subjects)

Representative variant (guard CV 0.10, tol 0.15); the others are similar or worse.

| | Subject AUROC (95% CI) | Window sens / spec @ τ=0.32 | 3-of-5 subject sens / spec (deployed rule) |
|---|---|---|---|
| No filter (deployed) | 0.970 [0.908, 1.000] | 95.9% / 83.6% | 100% / 93.8% |
| Filter, infer-only | 0.974 [0.915, 1.000] | 94.1% / 85.5% | 94.7% / 87.5% (1 FN, 2 FP) |
| Filter, retrain | 0.974 [0.913, 1.000] | 95.7% / 84.2% | 100% / 87.5% (0 FN, 2 FP) |

- **AUROC change is noise** (+0.003, intervals overlap almost completely).
- **The guard is leaky.** It still modified 10.8% of AF windows (naive: 55.9%). AF windows' mean
  probability dropped 0.01–0.06 per subject — a real sensitivity cost, which is the risk the
  10-13 note predicted.
- **The subject-level loss is a knife-edge, not a mechanism.** Non-AF subject `non_af_012` moved
  from consensus flag rate 0.91 → 0.95 (mean probability 0.71 → 0.70, i.e. unchanged) and so
  crossed the 0.92 cutoff; AF subject `af_005` moved 0.93 → 0.90. **Side finding:** the deployed
  subject rule's 0.92 flag-rate cutoff has two subjects within ±0.01 of it, so the quoted
  100% / 93.8% is fragile to any small perturbation. Quote subject AUROC with its CI as the
  primary figure (as decision 12 already says); treat the consensus operating point as indicative.

## Results — own device, wrist (non-AF testers, deployed weights)

| Data | AF windows without filter | With filter (guard 0.10) |
|---|---|---|
| 10-12 tester 1, radial after LED change (49 classified) | 13 | 10 |
| 10-12 tester 1, wrist (45 classified) | 32 | 32 |
| 10-13 auto-captured AF windows (18) | 18 | 14 |

The 18 windows from 10-13 fall into two kinds:

1. **Regular rhythm + one glitch (~8 windows, mostly S2b/S3).** Exactly the target. The filter
   fixes half of them cleanly (P 0.74 → 0.16, 0.47 → 0.03, 0.55 → 0.10, 0.33 → 0.13). The rest
   are just outside the rule (an interval at 1.70× M, a lone short beat, two long intervals at a
   window edge). Widening the rule to catch them would be tuning on the evaluation set.
2. **Grossly irregular throughout (~10 windows: S1's poor-contact run, S4, unattributed).**
   Robust CV 0.34–0.61, intervals 300–1900 ms. **No interval filter can separate these from AF**,
   because to the classifier they *are* AF-shaped. They passed the beat-morphology gate (e.g.
   good_fraction 0.75, median r 0.93). This is a signal-quality problem.

Type 1 never produced an alarm on 10-13 — 3-of-5 consensus already absorbs isolated glitches.
So the filter would mostly lower the single-window AF rate shown in the logs, not the alarm rate.

## Recommendation

1. **Do not merge the filter into the runtime.** No MIMIC gain, a measurable AF-window cost, and
   the device benefit lands where consensus already protects us.
2. **The next lever is signal quality, not intervals.** The windows that matter (type 2) are
   irregular because the beats are wrong, and they got past the morphology gate. Candidate: a
   stricter or additional gate (e.g. beat-amplitude consistency, IR baseline drift within the
   window), validated the same way — MIMIC AF retention first, then device replay.
3. **The real fix remains own-device wrist data.** With labelled wrist recordings, the
   classifier's negatives can include healthy wrist HRV and real detector jitter, which no
   hand-written rule replicates.
4. If revisited: it is only worth doing as part of a retrain (`ibi_af_v2`) on wrist data, never
   as a runtime-only patch over `ibi_af_v1`.

## Branch / reproducibility

Work lives on branch `exp/ectopic-filter` (script + results JSON + this note). `main`, the
deployed weights and the Pi were not modified. Rerun with
`python training/eval_ectopic_filter.py` from `03 - ML/` (device part needs the gitignored
captures under `data/device_captures/`).
