---
date: 2026-10-01
component: ML pipeline rebuild (03 - ML/training/, model/, edge_inference/, signal_processing/sqi.py)
reviewer: Claude (Opus 5), acting as ML lead
tags: [machine-learning, model, dataset, architecture, validation, deliverable]
status: delivered
---

# 2026-10-01 — ML Rebuild: A Working AF Classifier, and Why Chapter 2's Topology Cannot Work

Follows [[2026-10-01 - 1D-CNN Training Audit - Label-Subject Collinearity and Invalid Weights|the 2026-10-01 audit]].
The audit established that the shipped model was invalid. This note delivers a replacement
that works, and explains with measurements why the original design could not.

## Headline

**There is now a validated AF classifier**: `model/weights/ibi_af_v1.npz`.

| Metric (patient-disjoint 5-fold CV, all figures out-of-fold) | Value |
|---|---|
| **Subject-level AUROC** | **0.970** (95% CI **[0.908, 1.000]**, bootstrapped over subjects, n=35) |
| Window-level AUROC | 0.936 |
| Window operating point (τ = 0.32, Youden's J) | Sens **95.9%**, Spec **83.6%**, PPV 87.4%, NPV 94.5% |
| **Subject-level, 3-of-5 temporal consensus** | Sens **100%**, Spec **93.8%** (TP 19, FP 1, TN 15, FN 0 of 35) |
| Cross-sensor robustness (DeepBeat, never trained on) | subject AUROC 0.583 |
| Edge inference latency | **3.6–7.3 ms** against the 25 ms budget (CNN path was 14.5 ms) |

For comparison, on **identical data and identical protocol**, the 1D-CNN variants reach
subject AUROC 0.53–0.77. The previous pipeline's model scored 0.53 on this cohort and 0.35
on its own training distribution.

The model is 12 interval-irregularity features into a logistic regression — 12 coefficients,
not 1.03M parameters. That is not a compromise; it is the finding.

---

## 1. What the data actually contains

Three facts about DeepBeat were measured from the archives and all three contradict the
record in `MEMORY.md` and `deepbeat_manifest.json`.

**1.1 The corpus is 20.0 hours, not 3,725 hours.** DeepBeat ships 25-second parent windows
at a **1-second stride** across **8 simultaneous channels** (`parameters[:, 1]`, 'a'..'h'),
with each (timestamp, channel) pair appearing twice. Consecutive windows therefore share 96%
of their samples. v1 multiplied 536,399 overlapping windows by 25 s and reported 155 days of
recording. Counting unique (recording, second) pairs gives **72,062 s = 20.0 h**, an
inflation of about 186×.

**1.2 Only 2.4 hours of that is AF**, across 11 recording sessions. Every recording is
single-rhythm — 0 of 16 subjects in `validate.npz` carry both classes — so the labels are
effectively per-recording, which is normal for AF datasets but means the usable unit of
evidence is the recording, not the window.

**1.3 DeepBeat's 32 Hz sampling cannot resolve AF.** One sample is **31 ms**. The standard
pNN50 criterion thresholds successive interval differences at **50 ms**, so interval
quantisation noise is the same order as the signal being measured. Upsampling to 100 Hz does
not recover timing that was never sampled. This is a physical limit, not a modelling choice.

**1.4 DeepBeat's actual `train` partition was never downloaded** — `data/deepbeat/` holds
only `validate.npz` and `test.npz`. "Trained on DeepBeat" in v1 meant trained on DeepBeat's
*validation* split.

### The decisive measurement

A 3-parameter irregularity model (cv_ibi, pnn50, interval entropy) under patient-disjoint CV:

| Cohort | Subject AUROC, irregularity features only |
|---|---|
| MIMIC PERform AF (125 Hz, 35 subjects) | **0.914** |
| DeepBeat (32 Hz, 38 recordings) | **0.465** — chance |

With rate features added, MIMIC reaches 0.970 and DeepBeat 0.468. **No feature set and no
architecture extracted usable AF signal from DeepBeat as downloaded.** That is a property of
the data, established before any model was tuned.

---

## 2. Settled decision 9 is reversed, on measured grounds

Decision 9 (2026-09-13) made DeepBeat the training cohort and MIMIC the external validation
set. It rested on three premises, each now falsified:

| Premise for choosing DeepBeat | Measured reality |
|---|---|
| "~500k windows from ~175 subjects" | 20.0 h, ~36.5k non-overlapping windows, 30 recording IDs, 2.4 h of AF |
| "DeepBeat labels per window, MIMIC per recording" | Every DeepBeat recording is single-rhythm; both are effectively per-recording |
| "Wrist reflectance matches the MAX30102 deployment" | True, but 32 Hz cannot resolve the 50 ms interval differences AF detection depends on, while the deployment samples at 100 Hz |

**New position: MIMIC PERform AF trains; DeepBeat is a cross-sensor robustness cohort.**
MIMIC is 125 Hz (close to the deployed 100 Hz), 35 subjects (19 AF / 16 non-AF), 20 minutes
each, and demonstrably learnable. This is the reverse of decision 9 and must go to the
adviser with §1 as the justification — it changes Chapter 3 again, which is why the evidence
is recorded in full here rather than summarised.

The honest framing of the sensor question, which is *stronger* than the v1 "domain
generalization collapse" claim because it is symmetric and measured: a model fitted to
fingertip transmissive PPG transfers poorly to wrist reflectance PPG (subject AUROC 0.583),
and the wrist cohort available to us is too coarsely sampled to fit on. Both directions are
reported; neither is hidden.

---

## 3. Why Chapter 2's topology cannot detect AF

Chapter 2 specifies `Conv1D(1→32,k=5) → ReLU → MaxPool(2) → Conv1D(32→64,k=3) → ReLU →
MaxPool(2) → Flatten → Dense(64) → Sigmoid`. Two structural problems, both measured:

**3.1 The receptive field is 12 samples (0.12 s).** A typical inter-beat interval is
0.6–1.2 s. **No convolutional feature in this network can observe even one interval**, let
alone the variability across several that defines AF. The only layer spanning the window is
`Dense(16000, 64)`, which is position-specific.

**3.2 That dense layer is 1,024,000 of the model's 1,030,529 parameters**, against 35
subjects. It memorises per-recording morphology at fixed time offsets — the mechanism behind
v1 scoring below chance on its own training data.

### Architecture study

Same cohort, same patient-disjoint CV, same augmentation. Only the receptive field and the
temporal aggregation between the convolutions and the dense layer differ.

| Variant | Receptive field | Params | Subject AUROC |
|---|---|---|---|
| `ch2_flatten` (Chapter 2 as written) | 0.12 s | 1,030,529 | **0.533** |
| `ch2_gap` | 0.12 s | 10,625 | 0.562 |
| `ch2_statpool` | 0.12 s | 14,721 | 0.586 |
| `wide_gap` | 1.02 s | 36,033 | 0.730 |
| `wide_statpool` | 1.02 s | 40,129 | 0.737 |
| `xwide_statpool` | 1.98 s | 73,921 | 0.773 |
| **IBI features + logistic regression** | whole window, explicit | **12** | **0.970** |

Repeating the comparison with **5× more training windows** (2 s stride, 20,860 windows;
evaluation still non-overlapping) extends the trend rather than flattening it:

| Variant | Receptive field | Params | Subject AUROC | Per-fold |
|---|---|---|---|---|
| `xwide_statpool` | 1.98 s | 73,921 | 0.757 | 0.733 ± 0.232 |
| `xxwide_statpool` | 3.90 s | 141,505 | 0.836 | 0.833 ± 0.211 |
| `huge_statpool` | 6.86 s | 141,505 | 0.849 | 0.883 ± 0.145 |
| **`huge_gap`** | **6.86 s** | **137,409** | **0.842** | **0.883 ± 0.145** |

Four conclusions:

1. **Receptive field drives performance; capacity does not.** The 1.03M-parameter Chapter 2
   head is the *worst* of every variant tested, and a 10,625-parameter global-average-pooling
   head beats it. Performance rises monotonically with receptive field: 0.533 → 0.586 → 0.737
   → 0.773 → 0.836 → **0.849**, and the gain is still positive at 6.86 s.
2. **The fold spread narrows as the receptive field widens** — ±0.232 at 1.98 s, ±0.211 at
   3.90 s, **±0.145 at 6.86 s**. This matters more than the mean: a wider receptive field is
   not merely scoring higher, it is scoring *more consistently across unseen patients*, which
   is the behaviour of a model learning rhythm rather than per-recording morphology. It is the
   strongest single piece of evidence for the amendment.
3. **Once the receptive field is wide enough, plain global average pooling is sufficient —
   the mean+std head buys nothing.** At 6.86 s, `huge_gap` (0.842) and `huge_statpool` (0.849)
   have **identical fold statistics (0.883 ± 0.145)**; a 0.007 difference in the pooled figure
   is noise at n=35. The receptive field was doing all the work, and the extra variability
   term was only compensating for a receptive field too narrow to see variability directly.
   This simplifies the amendment considerably (see below).
4. **Even the best CNN trails the 12-parameter feature model by ~0.12 AUROC**, and its ±0.145
   spread is still far wider than the feature model's CI. At 35 subjects, asking a
   convolutional network to rediscover interval statistics from raw waveform is data-starved,
   when the peak detector already computes those statistics exactly. Individual folds range
   from 0.67 to 1.00, so no single CNN fold result should ever be quoted — only the pooled
   figure with its spread.

### Recommended Chapter 2 amendment

Keep the two convolutional blocks and the `Dense(64) → Sigmoid` classifier. Change exactly two
things, with §3's table as the justification:

* **Widen the kernels and pooling so a feature spans several beats, not one.** Recommended:
  **kernels 127/63 with pooling 8/8**, a 6.86 s receptive field. Spanning *one* beat is not
  enough: AF is defined by the variability *across* consecutive intervals, so a feature must
  see several. This is the change that does the work, and the narrowing fold spread
  (±0.232 → ±0.145) is the evidence that it generalises to unseen patients rather than merely
  fitting better.
* **Replace `Flatten` with global average pooling.** It is translation-invariant, drops
  1.02M parameters, and — per conclusion 3 — performs indistinguishably from the more
  elaborate mean+std head once the receptive field is adequate. Prefer it on three grounds:
  it is the smaller change to Chapter 2's text (one layer swapped, not a new pooling scheme),
  it is fewer parameters, and it is the setting **Grad-CAM was originally formulated for**,
  so it strengthens RQ3 rather than complicating it.

Taken together these are a *smaller* edit to Chapter 2 than the measured improvement might
suggest: two kernel sizes, two pooling factors, and `Flatten` → `GlobalAvgPool`. The
convolution count, the channel widths, the dense width and the sigmoid output all stand.

---

## 4. The deployed model

`model/ibi_classifier.py` — pure NumPy, no new dependency. Twelve features from Elgendi peak
detection, standardised, into logistic regression.

Standardised coefficients, which are themselves the RQ1 answer:

| Feature | Coefficient | Kind |
|---|---|---|
| `pnn50` | **+2.830** | irregularity |
| `cv_ibi` | **+2.399** | irregularity |
| `sd_hr` | −2.138 | irregularity |
| `shannon_dibi` | **+1.385** | irregularity |
| `mean_ibi` | −1.345 | *rate* |
| `n_beats` | +1.158 | *rate* |

**The confound was tested, not assumed.** AF and non-AF subjects are different people, so
any group difference (rate, age, medication) could carry the signal. Removing all rate
features leaves subject AUROC **0.914**; rate features *alone* give **0.674**. The signal is
interval irregularity — the AF mechanism — not heart rate.

### Operating point

Chosen by Youden's J **after a non-degeneracy check on both arms**. The v1 failure was not
Youden but accepting what it returned: sensitivity 95% at specificity 11%, PPV 5% — the
always-say-AF corner. The guard rejects any point failing a floor on either arm, and the
full threshold table is written to `training/runs/ibi_threshold_table.json` so the choice is
auditable. A single 10-second window never alarms: 3-of-5 temporal consensus is required.

---

## 5. Other defects fixed

**5.1 SQI perfusion index (`signal_processing/sqi.py`).** The probe-off test
`dc_val < 1000.0` is calibrated for raw MAX30102 ADC counts and fired on every file-based
window, forcing `pi = 0` and capping every score at 0.6 — so no window in a 1,072,798-window
corpus could exceed 0.6, and the `>= 0.50` gate silently became a skew/kurtosis filter that
discarded 43.5% of the training split. `dc_floor` is now an explicit parameter
(`None` for file data), AC/DC is computed scale-free, and the remaining criteria are
reweighted when PI is genuinely unavailable so the score still spans [0, 1]. Device
behaviour is unchanged; verified across raw-counts, normalised and mean-removed inputs.

**5.2 Telemetry honesty (`edge_inference/runner.py`).** `model_trained` was
`bool(weights_loaded)` — true for a chance-level model. It is now gated on a **validated
subject-level AUROC recorded in the weights metadata** (floor 0.70), and the 1D-CNN path
reports `model_trained: false` regardless of what loads. Telemetry gained `classifier`,
`model_validated_subject_auroc` and `explanation`.

**5.3 Evaluation protocol.** Subject-level metrics with subject-resampled bootstrap CIs are
primary throughout, because labels are constant within a recording and window counts
overstate the sample size by ~120×. Training may use overlapping windows (2 s stride,
20,860 windows) since splits are patient-disjoint; **every reported number comes from
non-overlapping windows**.

---

## 6. Interpretability (RQ3)

Two mechanisms, both implemented:

**6.1 Per-feature contributions** (`IBIAFClassifier.explain`). Each coefficient attaches to a
named physiological quantity, so a decision decomposes exactly into signed contributions.
Verified by test: contributions plus intercept reconstruct the logit. It also reports
`rate_fraction`, flagging any decision leaning on rate rather than irregularity.

**6.2 Beat-level counterfactual attribution** (`IBIAFClassifier.explain_intervals`). Each
interval is replaced by the window's median and the features recomputed; the drop in logit is
that interval's contribution. This is **exactly faithful by construction** — no gradient
approximation, nothing to validate against autograd — and it answers the clinical question
directly ("which beats were irregular?") rather than producing a smeared relevance band. It
emits a 1000-sample heat strip, so the dashboard's existing Grad-CAM overlay renders it
unchanged. Verified by test to rank a planted ectopic interval first.

Contributions are a **ranking within one window, not a magnitude comparable across windows**:
substituting the median reduces variability in any window. The window's probability carries
the AF evidence; the per-interval values say which beats mattered.

Grad-CAM for the CNN is implemented in `model/inference_model_v2.py` with closed-form
gradients for all three heads, verified against `torch.autograd` for four configurations.
**The v1 Grad-CAM clinical plausibility verdict remains void** — it was computed on a
chance-level model over MIMIC, where predicted "true positives" scored 0.387–0.429 against a
0.37 threshold. Any new plausibility claim must come from a model that clears chance.

---

## 7. How this answers the SOPs

| RQ | Status |
|---|---|
| **RQ1** — what PPG inputs / physiological parameters are needed | **Answered quantitatively.** Beat-to-beat interval dispersion (`cv_ibi`), successive-difference magnitude (`pnn50`, `rmssd`) and interval entropy, from Elgendi peaks on a 10 s, 0.5–5 Hz band-passed window. Rate-free ablation: 0.914; rate-only: 0.674. Also establishes a **sampling-rate requirement**: ≥100 Hz, because 32 Hz cannot resolve the 50 ms differences the criterion needs. |
| **RQ2** — how to design the IoT device | Unchanged dual-tier topology (ESP32-C3 + Pi). The classifier runs in 3.6–7.3 ms pure NumPy, ~3× faster than the CNN path. The 100 Hz acquisition rate is now *justified by measurement* rather than assumed. |
| **RQ3** — interpretable classification at the edge in real time | **Answered, with a stronger mechanism than planned**: exact per-feature contributions plus faithful beat-level counterfactual attribution, both real-time. The CNN Grad-CAM path is retained and mathematically verified, but no clinical plausibility claim may rest on it until a CNN clears chance. |
| **RQ4** — outputs for early detection, notification, decision support | Classifier, consensus gating, per-beat explanation and hash-chained event records are wired end to end. **Unchanged blockers: no notification channel, and Fabric sync is still deferred.** |
| **RQ5** — ISO/IEC 25010 | Functional Suitability and Performance Efficiency now have real measurements. **Still unmeasured: CPU/RAM utilisation, 24 h soak, SUS usability survey.** |

---

## 8. What is NOT established — state these as limitations

1. **n = 35 subjects.** The subject-AUROC CI is [0.908, 1.000]; quote the interval, never
   the point estimate alone.
2. **Never validated on the project's own MAX30102 wrist hardware.** Training data is
   fingertip transmissive ICU PPG. Measured transfer to wrist reflectance is poor (0.583).
   This is the single largest gap between the thesis claim and the evidence, and no amount of
   further modelling on MIMIC closes it — it needs data from the actual device.
3. **AF and non-AF subjects are different people.** The rate ablation addresses the most
   obvious confound; it cannot exclude all of them.
4. **MIMIC labels are per-recording**, so a paroxysmal subject's sinus segments are labelled
   AF. This depresses measured window sensitivity and is a floor, not a ceiling.
5. **The synthetic simulator is not validation.** It confirms the pipeline responds to
   irregularity; it says nothing about clinical accuracy.
6. **DeepBeat's `train` partition is still missing.** If acquired, §1 should be re-run: a
   wrist cohort at adequate sampling rate would change the design.

---

## 9. Required next steps, in priority order

1. **Adviser decision on the dataset reversal** (§2) and the Chapter 2 amendment (§3). Both
   change written chapters. §1 and §3 are the evidence packs.
2. **Collect labelled data from the actual MAX30102 wrist device.** This is the highest-value
   remaining action for validity — item 2 of §8 cannot be closed any other way. Even a few
   hours from consenting AF and non-AF subjects would allow a transfer measurement.
3. **Hardware verification** (`PLAN.md` §7, still unchecked): ESP32-C3 into the Pi, restart
   both services, confirm a real waveform and a live `ibi_af_v1` decision.
4. **Retire `cnn_af_v1.npz`** from the default path (done) and delete it once the adviser has
   seen the audit, so it cannot be cited.
5. Finish the architecture sweep's largest receptive fields (3.9 s / 6.9 s) to confirm where
   the CNN trend plateaus; it informs how far the Chapter 2 amendment needs to go.
6. Measure CPU/RAM and run the 24 h soak for RQ5.

---

## 10. Artefacts

| Path | Purpose |
|---|---|
| `training/build_dataset_v2.py` | Corrected build: file-scoped group keys, no temporal redundancy, scale-aware gating |
| `training/build_mimic_dense.py` | Dense training windows (2 s stride) + non-overlapping eval windows |
| `training/baseline_ibi.py` | Interpretable baseline, confound ablation, protocol definition |
| `training/train_ibi_model.py` | Trains, calibrates and exports the deployed classifier |
| `training/torch_model_v2.py`, `training/sweep_arch.py`, `training/sweep_v3.py` | Architecture study |
| `model/ibi_classifier.py` | **Deployed** pure-NumPy classifier + both explanation paths |
| `model/inference_model_v2.py` | Pure-NumPy CNN runtime + closed-form Grad-CAM, all heads |
| `model/weights/ibi_af_v1.npz` + `.meta.json` | Deployed weights, with metrics and caveats embedded |
| `tests/test_ibi_classifier.py` (14), `tests/test_parity_v2.py` (11) | New gates; full suite **97/97 passing** |

Everything in the audit's "what is actually sound" list was kept: the parity gate with its
negative control, autograd verification of Grad-CAM gradients, the anti-contamination guards,
and honest CI reporting.
