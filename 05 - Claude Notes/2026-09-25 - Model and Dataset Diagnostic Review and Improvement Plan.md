---
title: Model and Dataset Diagnostic Review and Improvement Plan
tags: [ml, diagnosis, dataset, training, validation, improvement-plan, honest-bounds]
date: 2026-09-25
status: proposed — diagnosis and plan only, nothing implemented
reviewer: Claude (Research & Theoretical Architect)
---

# Model and Dataset Diagnostic Review and Improvement Plan

A diagnostic review of the trained 1D-CNN (`cnn_af_v1.npz`) and the DeepBeat / MIMIC
PERform AF corpora. **No dataset was changed, no model retrained, no implementation code
written.** Everything below is derived from files already in the repository:
`cnn_af_v1.meta.json`, `training/runs/20260913_205107/history.csv`, the split manifests,
`training/train.py`, `training/cross_validate.py`, `training/build_dataset.py`, and the
raw `.npz` headers on disk.

Findings are marked **[CONFIRMED]** where a file in the repository demonstrates them
directly, and **[HYPOTHESIS]** where the evidence is suggestive but insufficient.

---

## Step 1 — The current setup, as verified

| Item | Verified value | Source |
|---|---|---|
| Input | 1 × 1000 samples, 10 s @ 100 Hz, detrend → Butterworth 0.5–5 Hz → z-score | `meta.json` `input_contract` |
| Architecture | Conv1D(32,5) → ReLU → MaxPool2 → Conv1D(64,3) → ReLU → MaxPool2 → Flatten → Dense(64) → Dropout(0.5) → Dense(1) | `torch_model.py` |
| Parameters | **1,030,529** — of which `Dense(16000→64)` is 1,024,064 (99.4%) | computed |
| Training corpus | DeepBeat, wrist reflectance, green 525 nm | `data/README.md` |
| External corpus | MIMIC PERform AF, ICU fingertip transmissive, red/IR 660/940 nm | `data/README.md` |
| Subjects available | **30** total — 12 train / 4 val / 14 test | `splits/deepbeat_split_manifest.json` |
| Windows prepared | train 451,791 · val 255,874 · test 17,106 | `data/README.md` |
| **Windows actually trained on** | **21,938, from 4 subjects** (147, 150, 152, 153) | `meta.json` |
| Best epoch | **1** of 6 | `history.csv` |
| Loss | `BCEWithLogitsLoss(pos_weight)`, clamped to [0.2, 10.0] | `train.py:420` |
| Windowing | stride = window = 1000 → **non-overlapping**, 2 children per 25 s parent | `build_dataset.py:333` |
| Quality gate | drop if `dataset_quality == 2` or `SQI < 0.50`; dropped 43.5% of train | `build_dataset.py:14` |
| Decision threshold | 0.37, Youden's J, on the validation split | `meta.json` `calibration` |

### Information I do not have, and am not assuming

1. **Why the window cap was set.** `--max-train-windows` defaults to `None`; something
   passed it. Whether this was a deliberate smoke test later mistaken for the real run, or
   an accident, is not recorded anywhere. This matters for whether to treat it as a process
   failure or a one-off.
2. **Subject demographics** — age, sex, skin tone, comorbidity. Skin tone in particular is
   a documented confounder for reflectance PPG, and with 12 training subjects an unbalanced
   distribution is plausible. Not available in what has been processed.
3. **Which subjects are in `train.npz`.** `parameters.npy` (2.8 M × 3, object dtype) has
   never been parsed. Until it is, the true total subject count of DeepBeat is unknown —
   it may be materially more than 30.
4. **Per-subject device/wavelength in DeepBeat.** The corpus spans Simband, Cardiaband and
   Empatica. Which subject used which is unrecorded here.
5. **MIMIC raw CSVs are not on this machine**, so the external evaluation cannot be
   independently recomputed locally.
6. **Zero labelled windows from our own MAX30102.** Nothing in the pipeline has ever been
   evaluated on a single window from the actual device. See §2.10.

---

## Step 2 — Diagnosed weaknesses

### 2.1 The model was trained on 4 subjects and 4.9% of the prepared data — [CONFIRMED]

**Problem.** `meta.json` records `subjects_used: ["147","150","152","153"]` and
`window_count: 21938`, against 451,791 prepared windows from 12 subjects.

**Why it hurts.** At 1,030,529 parameters this is **47 parameters per training window**.
For context, `MEMORY.md` decision 9 rejected MIMIC PERform as a training corpus explicitly
because ~245 params/window is "a memorization regime", and justified DeepBeat on the basis
that it would give ~2 params/window. The run as executed sits at 47 — the same regime that
was rejected, not the one that was argued for.

**Severity: critical.** Every downstream number — threshold, internal metrics, external
metrics, CV, the Grad-CAM evidence — derives from these weights.

### 2.2 Early stopping and threshold calibration used different class priors — [CONFIRMED]

**This is the most specific defect found, and it is not documented anywhere.**

Reconstructing the confusion matrices in `meta.json`:

| Stage | Windows | AF windows | **Prevalence** |
|---|---|---|---|
| Early-stopping validation (`val_metrics`) | 17,562 | 12,640 | **71.97% AF** |
| Threshold calibration (`calibration`) | 255,874 | 12,640 | **4.94% AF** |

The AF count is **identical (12,640)** in both — the subject-wise subsampler retained every
positive window and discarded only negatives. So the model was selected and early-stopped
against a distribution that is **14.6× richer in AF** than the one the operating threshold
was then calibrated on, and 14.6× richer than the corpus itself.

**Why it hurts.** Model selection under a 72% prior optimises a different decision surface
than deployment under a 5% prior. It is a direct explanation for the collapse in precision:
at τ = 0.37 the calibration split yields PPV = 5.24% against a base rate of 4.94% — the
model contributes **0.3 percentage points over guessing**.

**Severity: critical.** This is a methodological error an examiner could isolate, and it
invalidates the early-stopping decision rather than merely weakening it.

### 2.3 The headline validation AUPRC is barely above chance — [CONFIRMED]

`best_val_auroc: 0.6346`, `val_auprc: 0.8107` are recorded as the training result and
`MEMORY.md` cites the AUROC. But AUPRC's no-skill baseline **equals the positive
prevalence**, and that validation subsample was 71.97% AF.

> A coin flip scores AUPRC = 0.720 on that set. The model scored 0.811.

So the most impressive-looking number in the record represents roughly nine points over
random, on a set whose composition was an artefact of subsampling.

**Severity: high.** This is precisely the "evaluation metric that hides poor performance"
category. Any AUPRC reported in the thesis must be accompanied by its prevalence baseline.

### 2.4 The cross-validation was capped identically, so it is not evidence — [CONFIRMED]

`cross_validate.py` defaults to `max_train_windows_per_fold = 20000` and `epochs = 3`; the
recorded fold sizes (21,291 / 16,242 / 22,124 / 24,091 / 22,124) confirm the cap was active.

**Why it hurts.** The AUROC 0.330 ± 0.118 result is currently the strongest apparent
evidence that the *corpus or architecture* is inadequate. It cannot carry that weight,
because it measures the same crippled configuration. **The question "is the dataset
sufficient?" is currently unanswered, not answered negatively.**

**Severity: high** — it is corrupting the team's model of what is wrong.

### 2.5 Immediate overfitting — [CONFIRMED]

| Epoch | Train loss | Val AUROC |
|---|---|---|
| 1 | 0.553 | **0.635** |
| 2 | 0.531 | 0.610 |
| 3 | 0.515 | 0.579 |
| 4 | 0.492 | 0.555 |
| 6 | 0.454 | 0.567 |

Training loss falls monotonically while validation AUROC degrades from the first epoch on.
The saved model is epoch 1 — effectively an undertrained model kept because everything
after it was worse. Consistent with 2.1: four subjects is enough to memorise.

**Severity: critical**, but expected to resolve with 2.1.

### 2.6 Architectural bottleneck: `Flatten → Dense` — [HYPOTHESIS, well-supported]

99.4% of parameters sit in the single `Dense(16000→64)` layer, which is a fully-connected
map over **absolute time position**. A pulse at t = 2.0 s and the identical pulse at
t = 2.1 s reach entirely different weights. For a task defined by *interval irregularity* —
a translation-invariant property — this is close to the worst possible inductive bias, and
it is where the memorisation capacity lives.

Global average pooling over the 250 time steps would reduce the layer to 64×64 and cut the
model to roughly **8,000 parameters**, a 128× reduction, while making it translation-
invariant.

**Caveat, and it is decisive:** the topology is **frozen** — it is specified in Chapter 2
and `MEMORY.md` decision 10 states regularization is training-time only. Changing it is a
thesis-document decision, not an engineering one. Recorded as an option, not a
recommendation.

**Severity: high if confirmed, but blocked on a documentation decision.**

### 2.7 Class imbalance handled, but only partially — [CONFIRMED]

DeepBeat is 9.7% AF overall; prepared train is 12.34%, prepared val 4.94%. `pos_weight` is
applied and clamped to ≤ 10 — adequate for a 7.1 ratio. But the val/train prevalence
mismatch (4.94% vs 12.34%) is itself a distribution shift between fitting and selection,
independent of the far larger one in 2.2.

**Severity: medium.**

### 2.8 The quality gate is uncalibrated and drops 43.5% of training data — [CONFIRMED]

`SQI < 0.50` and `min_pi = 0.15` were chosen by eye during bring-up. DeepBeat ships
`qa_label`, a 3-class human-labelled quality ground truth over the same signals, which
makes this gate measurable — and it has never been measured.

Counter-evidence that the gate may be actively wrong: in the results note's stratification,
**low-SQI windows scored *higher* AUROC (0.626) than high-SQI windows (0.512)**. If the
gate were selecting good signal, that should not happen. Either the SQI is miscalibrated,
or it is correlated with a confound.

**Severity: medium-high.** Discarding 43.5% of data on an unvalidated criterion is
expensive, and it compounds 2.1.

### 2.9 Label provenance in DeepBeat — [HYPOTHESIS, externally documented]

A published assessment (arXiv:2307.08766) states most DeepBeat labels were produced by an
unpublished quality-assessment model rather than human adjudication, and the reference ECG
was never released. The team correctly routed the internal *test* split to the
cardiologist-adjudicated partition — but **train and validation are drawn from the
algorithmically-labelled portion**.

**Why it hurts.** If training labels are noisy, there is a ceiling on achievable
performance that no amount of retraining will lift, and it would look exactly like what is
being observed.

**Severity: high, and it is the main competing explanation to 2.1.** The two are separable
by experiment (see Step 5).

### 2.10 The model has never seen a single window from our own device — [CONFIRMED by absence]

Training is green-LED wrist reflectance (Simband/Cardiaband). Deployment is red/IR
MAX30102. The external MIMIC failure (AUROC 0.494) already demonstrates that this model
does not survive a change of optical pathway — and **our own sensor is a different optical
pathway from the training data too**, just less different than MIMIC's.

There is currently **no evidence whatsoever** about how the model behaves on MAX30102 data,
because no labelled window from the device exists.

**Severity: critical for the thesis claim**, and it is the gap that is never closed by any
amount of work on DeepBeat.

### 2.11 Subject count overstated in the project's own record — [CONFIRMED]

`MEMORY.md` decision 9 justifies DeepBeat on "~500k windows from **~175 subjects**". The
manifest shows **30**. Subject count is the quantity that governs generalisation, so the
stated rationale rests on a figure 6× too high.

Mitigating: `train.npz` (2.8 M windows) has never been parsed, so the true count may be
higher — but it is not currently known, and the number in the record is not supported.

**Severity: medium** (thesis integrity, not model performance).

### 2.12 Sibling-window correlation within subject — [HYPOTHESIS, low severity]

Each 25 s parent yields exactly 2 non-overlapping 10 s children. Siblings are highly
correlated and both land in the same split. Since splits are subject-isolated this is not
leakage, but it does mean the **effective** sample size is nearer the parent count than the
window count, inflating apparent data volume by ~2×.

**Severity: low.** Worth one sentence in the thesis, no action.

### 2.13 Probability calibration is broken — [CONFIRMED]

`meta.json` records `calibration_assessment: "POORLY_CALIBRATED"`, ECE 0.424, and
**Brier skill score −3.95** — i.e. worse than always predicting the base rate. The
dashboard renders these as percentages to clinicians.

**Severity: high for the dashboard**, medium for the thesis. Partly downstream of 2.2.

### 2.14 Ruled out — things that are correctly done

Stated for completeness, because they are the parts usually got wrong:

- **No train/test leakage.** Three independent guards in `train.py`; manifest-level and
  array-level subject disjointness both asserted.
- **No MIMIC contamination.** Path and subject-ID guards in both `train.py` and
  `calibrate_threshold.py`, which also refuses any path containing `test`.
- **Threshold not tuned on test.** Enforced in code.
- **Subject-wise splitting**, not window-wise.
- **Train/inference skew avoided** — `build_dataset.py` imports the production DSP modules
  rather than reimplementing them.
- **PyTorch ↔ NumPy parity** verified to 5.96e-08, with a negative control that fails on a
  wrong flatten permutation.

These are the expensive things to build and they are sound. The defects above are
configuration and methodology, not architecture of the pipeline.

---

## Step 3 — Prioritisation

Ranked by expected impact × confidence, against cost and risk.

| # | Issue | Confidence | Impact | Cost | Risk of fix | Priority |
|---|---|---|---|---|---|---|
| 2.1 | Trained on 4 subjects / 4.9% of data | Confirmed | Very high | Low | Very low | **P0** |
| 2.2 | Prior mismatch between selection and calibration | Confirmed | Very high | Low | Low | **P0** |
| 2.4 | CV capped, so "dataset is bad" is unproven | Confirmed | High (decision-blocking) | Low | Very low | **P0** |
| 2.10 | Never evaluated on our own sensor | Confirmed | Very high (thesis validity) | Medium | Low | **P1** |
| 2.3 | AUPRC reported without prevalence baseline | Confirmed | High (integrity) | Trivial | None | **P1** |
| 2.9 | Training labels likely algorithm-derived | Hypothesis | High | Medium | Low | **P1** |
| 2.8 | Quality gate uncalibrated, drops 43.5% | Confirmed | Medium-high | Medium | Medium | **P2** |
| 2.13 | Probabilities uncalibrated | Confirmed | Medium | Low | Low | **P2** |
| 2.5 | Immediate overfitting | Confirmed | — | — | — | resolves with 2.1 |
| 2.11 | Subject count overstated | Confirmed | Medium (integrity) | Trivial | None | **P2** |
| 2.6 | `Flatten→Dense` inductive bias | Hypothesis | High | Low to test | **High** — frozen topology | **P3, gated** |
| 2.7 | Train/val prevalence differ | Confirmed | Medium | Low | Low | folds into 2.2 |
| 2.12 | Sibling-window correlation | Hypothesis | Low | Trivial | None | **P4, document** |

**The P0 group shares one property: it is all configuration, and all three are answered by
re-running existing scripts with different flags.** No new code, no new data, no thesis
changes. That is where to start, and until it is done the project cannot distinguish a
data problem from an operator error.

---

## Step 4 — Proposed fixes for the highest-priority issues

### Fix A — Re-run training uncapped (addresses 2.1, 2.5)

**Change.** Run `train.py` with `--max-train-windows` and `--max-val-windows` unset, on all
12 training subjects (451,791 windows), patience 5 on validation AUROC.

**Why it addresses it.** Moves from 47 to 2.28 params/window — out of the memorization
regime and into the one decision 9 actually argued for.

**Needed.** The processed `.npz` files, which are absent from this machine and must be
rebuilt via `build_dataset.py --dataset deepbeat`, plus PyTorch. The raw DeepBeat files are
present (`validate.npz` 3.3 GB, `test.npz` 114 MB).

**Risks.** Compute time; 43.5% quality-gate drop still applies; may still overfit, which
would itself be informative.

**Measure.** Validation AUROC at best epoch, and the epoch number. If best epoch is again 1,
capacity or labels dominate and attention moves to 2.6/2.9.

### Fix B — Make the selection and calibration sets the same distribution (addresses 2.2, 2.7, 2.3)

**Change.** Early-stop on the **full** validation split, at its natural 4.94% prevalence —
the same set the threshold is calibrated on. If a subsample is ever needed for speed,
subsample **stratified**, preserving prevalence, and record the prevalence in `meta.json`.
Add a hard guard that refuses to proceed if selection and calibration prevalence differ by
more than a set tolerance.

**Why.** Removes the 14.6× prior shift between selection and operating point.

**Risks.** Validation becomes slower (255,874 windows/epoch). Mitigate with stratified
subsampling, not the current subject-greedy one.

**Measure.** PPV and F1 at the chosen threshold, versus base rate — the model must beat
4.94% PPV by a margin that matters, not by 0.3 points.

### Fix C — Re-run cross-validation uncapped (addresses 2.4)

**Change.** `cross_validate.py` with `--max-train-windows` unset and enough epochs to
early-stop honestly.

**Why.** Converts the central open question — is the corpus sufficient? — from unanswerable
to answered. This is the experiment that decides whether the project needs more data or
better training.

**Measure.** Mean ± sd AUROC across folds, compared against the capped 0.330 ± 0.118
baseline. Report the spread, not just the mean.

### Fix D — Acquire a small labelled set from the real device (addresses 2.10)

**Change.** Record PPG from the actual MAX30102 with a simultaneous reference — a
single-lead ECG or a validated pulse oximeter — on whatever subjects are available,
including at minimum sinus rhythm at rest, after exertion, and with deliberate motion
artefact. Even 30–60 minutes across a handful of people is transformative relative to zero.

**Why.** It is the only way to say anything defensible about the deployed system, as
opposed to about DeepBeat. It also converts RQ5's Functional Suitability from an inference
into a measurement.

**Trade-off, stated plainly.** Without an AF-positive subject this cannot measure
sensitivity. But it *can* measure specificity and false-alarm rate on known-sinus subjects,
and given the measured specificity of 10.9% that is the more urgent number. A device that
alarms on healthy volunteers is demonstrable in a defence room.

**Needs a decision from the team**, since it requires hardware access and an ethics posture
on recording classmates. Not a Claude task.

### Fix E — Calibrate the quality gate against `qa_label` (addresses 2.8)

**Change.** Compute the sensitivity/specificity of `SQI ≥ 0.50` against DeepBeat's 3-class
`qa_label`, sweep the threshold, and choose it on evidence. Record as an ADR.

**Why.** It is currently discarding 43.5% of training data on an unvalidated rule, and the
inverted stratification result suggests the rule may be wrong.

**Measure.** Agreement with `qa_label`, and downstream validation AUROC with the gate at the
chosen value versus at 0.50 versus disabled.

### Fix F — Corrections to the written record (addresses 2.3, 2.11)

Zero-risk, do immediately: correct "~175 subjects" to 30 in `MEMORY.md`; add the prevalence
baseline beside every AUPRC; correct the `data/README.md` directory listing which omits
`train.npz`.

---

## Step 5 — Validation plan

**Governing rule: every comparison below is against the frozen 14-subject
cardiologist-adjudicated test split, which is touched exactly once per candidate model, and
never used to choose anything.** Threshold selection, early stopping and gate calibration
all happen on validation only. The existing code already enforces this and the guards must
not be relaxed.

### Baseline to beat

The current `cnn_af_v1.npz`: test window sens 85.4% / spec 10.9%, subject-level 100% / 75%,
external MIMIC AUROC 0.494, capped CV 0.330 ± 0.118.

### Experiment 1 — Does uncapped training fix it?

- Fix A + Fix B together.
- Report on validation: AUROC, AUPRC **with prevalence baseline**, best epoch, and the
  train/val loss curves.
- **Success:** validation AUPRC exceeds prevalence by a clear margin and best epoch > 1.
- **Failure:** best epoch is 1 again → capacity or labels dominate; go to Experiment 4.

### Experiment 2 — Honest generalisation bounds

- Fix C: uncapped 5-fold subject-grouped CV over the development subjects.
- Report mean ± sd of AUROC, AUPRC, sensitivity, specificity, per fold and pooled.
- **Success:** mean AUROC ≥ 0.70 with sd < 0.10.
- **Interpretation:** a high mean with a wide spread means subject diversity is the binding
  constraint, which points at `train.npz`, not at the architecture.

### Experiment 3 — The operating point actually matters

- Sweep the threshold on validation, choose by Youden's J **and** by a
  clinically-motivated criterion (e.g. the highest sensitivity at specificity ≥ 70%).
  Report both.
- Report the full confusion matrix, PPV, NPV, F1, and the subject-level consensus result
  with the 3-of-5 filter the device actually runs.
- **Success:** a threshold exists at which specificity ≥ 70% and sensitivity ≥ 70% on
  validation, holding on test.
- **Note:** subject-level consensus is the deployed decision unit, so it is the primary
  endpoint; the per-window number is diagnostic, not the headline.

### Experiment 4 — Separate "bad labels" from "bad training" (tests 2.9 vs 2.1)

The two competing explanations are separable, and this is the experiment that does it.

- Train on the **cardiologist-adjudicated** subjects only, with subject-grouped CV *within*
  that partition, and compare against training on the algorithmically-labelled portion at
  matched subject count and window count.
- **If adjudicated-only performs markedly better at equal scale**, label noise is the
  ceiling (2.9 confirmed) and more algorithmic data will not help.
- **If they perform the same**, labels are not the binding constraint and the priority
  returns to scale and diversity.
- **Caveat:** the adjudicated partition is the 14-subject test split. Using it for training
  consumes the clean test set. **Do this only as a diagnostic, with the resulting model
  discarded and never reported as a result.** Record it as an experiment, not a deliverable.

### Experiment 5 — Capacity ablation (tests 2.6, only if the topology is unfrozen)

- Global average pooling in place of flatten, ~8k parameters, same data and protocol.
- **Success:** matches or beats the frozen topology at a fraction of the size.
- **Blocked on:** a team/adviser decision, since Chapter 2 specifies the architecture.
  Running it as a documented ablation and *reporting* the comparison is defensible even if
  the deployed model stays frozen — and it is a genuine contribution if GAP wins.

### Experiment 6 — Real-device evaluation

- Fix D's recordings through the unmodified production DSP path.
- Report false-alarm rate per hour on known-sinus subjects at the deployed threshold, with
  and without the 3-of-5 consensus filter.
- **Success:** substantially fewer than one false AF assertion per hour of sinus recording.
- This is the number that decides whether the system is demonstrable.

### What would falsify the whole approach

Stated so the team is not tempted to keep retrying indefinitely: if uncapped training on
all 12 subjects with matched priors still yields validation AUPRC near prevalence and CV
AUROC below ~0.6, then **the corpus, the labels, or the frozen architecture cannot support
this task**, and the honest thesis outcome is a negative result with a clear account of
why. A well-documented negative result with sound methodology is defensible; an unexplained
positive one is not.

---

## Step 6 — Prioritised action plan

| # | Problem | Evidence needed | Fix | Experiment | Success criterion |
|---|---|---|---|---|---|
| 1 | Trained on 4/12 subjects, 4.9% of data | Already confirmed | Fix A — retrain uncapped | Exp 1 | Best epoch > 1; val AUPRC clears prevalence baseline |
| 2 | Selection prior 72% AF vs calibration 4.94% | Already confirmed | Fix B — same distribution for both, stratified if subsampled + a guard | Exp 1, Exp 3 | PPV materially above the 4.94% base rate |
| 3 | "Is the dataset sufficient?" unanswered | Uncapped CV | Fix C | Exp 2 | Mean AUROC ≥ 0.70, sd < 0.10 |
| 4 | Written record overstates results | Already confirmed | Fix F | — | `MEMORY.md`, `data/README.md` corrected; AUPRC always paired with its baseline |
| 5 | Never tested on the real sensor | New recordings | Fix D (**team decision**) | Exp 6 | < 1 false AF assertion per hour of sinus |
| 6 | Label noise may cap performance | Adjudicated-vs-algorithmic comparison | — | Exp 4 (diagnostic only, model discarded) | Determines whether more data helps |
| 7 | Quality gate unvalidated, drops 43.5% | `qa_label` agreement sweep | Fix E + ADR | Exp 1 re-run at chosen gate | Defensible threshold, no AUROC loss |
| 8 | Probabilities uncalibrated (BSS −3.95) | Post-fix reliability curve | Platt or isotonic on validation | Exp 3 | ECE well below 0.424 |
| 9 | Frozen `Flatten→Dense` may be wrong bias | Ablation | **Gated on adviser decision** | Exp 5 | GAP matches or beats at ~128× fewer params |

### Sequencing

Items **1–3 are one session** and unblock everything else — they need only existing scripts
run with different flags, plus rebuilding the processed `.npz` (the raw DeepBeat files are
on disk). Item **4 costs minutes** and should not wait. Item **5 needs a team decision**
about hardware access and should be raised now, because it has the longest lead time and
the highest thesis value. Items 6–9 are contingent on what 1–3 reveal.

**Nothing in 1–4 changes the thesis text, the architecture, or the deployed system.**

### One process recommendation

The root cause of defects 2.1, 2.2 and 2.4 is the same: a debug flag set for a quick run,
then not cleared, with nothing in the record distinguishing a smoke test from a real one.
`train.py` should refuse to write to `model/weights/` when any cap is active unless an
explicit `--smoke-test` flag is passed, and `meta.json` should carry a
`run_type: smoke | full` field. That is a ten-line change that makes this class of error
impossible to repeat, and it is worth more than any single retrain.

---

## Related

- [[2026-09-25 - The Training Dataset Explained, and What Is Wrong With It|The training dataset explained, and what is wrong with it]] — plain-language companion
- [[../02 - Code Review/2026-09-13 - 1D-CNN Training Results & Honest Performance Bounds|Training results & honest performance bounds]] — Antigravity's results record
- [[2026-09-10 - Model Training Plan and Project Roadmap|Model training plan & roadmap]]
- [[2026-09-11 - Wrist vs Fingertip Sensing - SOP and RRL Alignment|Wrist vs fingertip sensing]]

---
Back to [[Index|Claude Notes Index]]
