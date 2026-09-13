---
title: Model Training Plan and Project Roadmap
tags: [ml, training, deepbeat, roadmap, iso25010, planning]
date: 2026-09-10
status: proposed — decisions 1 and 2 need the team's sign-off before implementation
reviewer: Claude (Research & Theoretical Architect)
---

# Model Training Plan and Project Roadmap

Written after the session that brought the acquisition chain live. The pipeline now
works end to end on real sensor data; what it lacks is a trained model, which is the
single thing standing between "a working system" and "a defensible thesis result".

## Where we actually are

| Layer | State |
|---|---|
| Acquisition (ESP32-C3 + MAX30102) | Working. 99.6 Hz, CRC-framed, contact-gated. |
| DSP (filter, SQI, peak detection) | Working. Real windows flowing. |
| 1D-CNN | **Architecture only.** Hand-written NumPy forward pass, random He-init weights. Never trained. |
| Grad-CAM | Working mechanically against the NumPy forward pass. Explains noise, because the model is noise. |
| Storage / hash chain | Working. Verified across 224 real writes. |
| Dashboard | Working, offline-capable, integration-ready. |

**Functional Suitability under ISO/IEC 25010 is currently unmeasurable.** There is no
sensitivity, specificity, accuracy or ROC-AUC to report, because the weights are
random. Every AF probability the system emits is noise. This is the critical path.

## What we found on disk

The team has already downloaded **DeepBeat** (Synapse `syn22006404`, Torres-Soto &
Ashley 2020) — not the MIMIC PERform AF dataset named in Chapter 2 and
`ANTIGRAVITY.md` §4.3.

```
03 - ML/data/deepbeat/
  train.npz      18.1 GB
  validate.npz    3.3 GB
  test.npz      113.7 MB
  VGG_singletask.h5  889 MB   (the authors' own pretrained model)
```

Verified structure of `test.npz`, read directly from the archive headers:

```
signal.npy     (17617, 800, 1)  float64   800 samples @ 32 Hz = 25 s per segment
rhythm.npy     (17617, 2)       float32   one-hot: non-AF / AF
qa_label.npy   (17617, 3)       float32   3-class signal quality
parameters.npy (17617, 3)       object
```

Two useful properties: the members are **stored uncompressed** (compressed size equals
raw size), so `signal.npy` can be extracted once and then `np.load(..., mmap_mode='r')`
memory-mapped rather than read into RAM — essential, since 18 GB will not fit in the
RTX 2060's 6 GB VRAM or comfortably in system memory. And `qa_label` gives a
human-labelled signal-quality ground truth, which is directly relevant to the SQI gate
we hand-tuned this session.

## Decision 1 — DeepBeat as the training corpus, and Chapter 2 must be updated

**Recommendation: train on DeepBeat; keep MIMIC PERform as optional external
validation.**

Rationale. DeepBeat is wearable-wrist PPG, which is the actual deployment domain of
this thesis; MIMIC PERform is hospital bedside monitoring, a different signal
distribution. DeepBeat is already downloaded (an 18 GB acquisition that took hours),
is AF-labelled at scale, and has a published benchmark to compare against
([[../01 - Literature/Torres-Soto-2020-Multi-task deep learning for cardiac|Torres-Soto & Ashley 2020]]).
Chapter 2 and `ANTIGRAVITY.md` §4.3 both currently name MIMIC PERform, so **the thesis
text has to be corrected either way** — this cannot be left as a silent divergence
between the document and the code.

Using both is the stronger scientific position if time allows: train and tune on
DeepBeat, then report unseen-domain performance on MIMIC PERform. Cross-dataset
generalisation is a genuine contribution and pre-empts the obvious examiner question.

## Decision 2 — Reconcile 800 @ 32 Hz with 1000 @ 100 Hz

The datasets and the deployed pipeline disagree on window geometry:

| | Samples | Rate | Duration |
|---|---|---|---|
| DeepBeat segment | 800 | 32 Hz | 25 s |
| Deployed pipeline / Chapter 2 | 1000 | 100 Hz | 10 s |

**Recommendation: resample DeepBeat up to 100 Hz and slice into 10 s windows.** Keep
the deployed contract exactly as it is.

Why this direction rather than retraining at 32 Hz. The 100 Hz / N=1000 geometry is
already specified in Chapter 2, implemented in the firmware, assumed by the Butterworth
cutoffs and the Elgendi detector, and baked into the Grad-CAM upsampling. Changing it
means touching every layer and re-verifying the whole chain we just brought up.
Resampling the training data instead confines the change to an offline preprocessing
script.

The honest caveat, which belongs in the thesis: upsampling 32 → 100 Hz adds no
information. The effective bandwidth stays at the 16 Hz Nyquist limit of the source.
This is defensible because the analysis band is 0.5–5 Hz — the bandpass the pipeline
already applies — so nothing the model relies on is lost. State this explicitly rather
than letting an examiner find it.

Each 25 s segment yields two non-overlapping 10 s windows, or more with a stride, which
doubles as augmentation. Windows must be assigned to train/validate/test **by subject**,
never by window, or overlapping windows from one person leak across the split and the
reported accuracy becomes meaningless.

## Decision 3 — Train in Keras, deploy the weights into the existing NumPy model

`03 - ML/model/inference_model.py` is a hand-written NumPy forward pass with **no
autodiff**. It cannot be trained in place without writing backpropagation by hand.

**Recommendation: rebuild the identical architecture in Keras, train it on the RTX
2060, then export the learned weights to `.npz` and load them into the existing NumPy
class at startup.**

This keeps everything that already works: the Pi's runtime stays NumPy-only (no
TensorFlow on ARM, fast start, small footprint), `grad_cam.py` continues to operate on
the same forward pass, and the deployed inference path is unchanged apart from reading
weights from a file instead of a seeded RNG.

Rejected alternative: exporting to TFLite and running that on the Pi. It would help the
latency budget, but **Grad-CAM needs gradients with respect to the final convolutional
feature maps**, which TFLite does not expose. Explainability is Research Question 3 —
it is not negotiable for a latency win. If quantisation later proves necessary, the
right shape is a quantised model for the classification path plus the NumPy model
retained for explanation, and that trade-off should be recorded as an ADR.

The architecture to replicate exactly, per `ANTIGRAVITY.md` §4.3:

```
Input(1000, 1)
Conv1D(32, 5, padding='same') -> ReLU -> MaxPool1D(2)
Conv1D(64, 3, padding='same') -> ReLU -> MaxPool1D(2)
Flatten -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid)
Loss: binary cross-entropy   Optimiser: Adam(1e-3)   Early stopping on val loss
```

The NumPy implementation must be checked against the Keras one on the same input
before deployment: forward-pass outputs should agree to within floating-point
tolerance. If they diverge, the exported weights are being interpreted incorrectly
(padding, channel order and kernel orientation are the usual culprits) and every
downstream metric would be silently wrong.

## Environment — training happens in Colab, on a teammate's machine

**Training is not done locally.** A teammate runs it in **Google Colab**. The Windows
workstation does have an RTX 2060, but it is not the training host and should not be
planned around. Nothing in this repository needs a local CUDA install.

That makes our side of the work an **interface contract**, not a training run. What has
to be right is that whatever comes back from Colab drops into the deployed NumPy
runtime and behaves identically. Three artefacts define the boundary:

1. **Preprocessing spec** — how DeepBeat becomes model input (Decision 2 below).
   The teammate needs this to build the same tensors we expect at inference time.
2. **Architecture spec** — the exact layer definition (Decision 3 below), so the Colab
   model and the NumPy forward pass are the same function.
3. **Weight-export format** — a `.npz` of named arrays that `inference_model.py` loads.
   The deliverable coming back is *weights*, not a `.h5` or a `.tflite`.

Practical Colab notes worth passing on: the 18 GB `train.npz` will not fit in a Colab
session's disk or RAM, so it needs to be either sharded and uploaded to Drive after the
resampling step (which shrinks it — `float64` → `float32` alone halves it), or
regenerated in-session from a smaller preprocessed subset. Colab sessions also expire,
so checkpoints must be written to Drive rather than local session storage.

**Do not train on the Pi** under any circumstance: 18 GB of data against ~12 GB free
disk, on a CPU that already needs 27–42 ms for a single forward pass.

## Evaluation protocol

The thesis targets sensitivity ≥ 90%, specificity ≥ 90%, accuracy ≥ 90%. To report
these credibly:

- Subject-level splits, as above. Report the subject counts per split.
- Report sensitivity, specificity, accuracy, F1, ROC-AUC and PR-AUC with the confusion
  matrix. **PR-AUC matters more than ROC-AUC here** because AF is the minority class.
- Report the class balance. If AF prevalence is low, address it with class weighting
  or resampling and say which was used.
- Choose the decision threshold on the validation split, never the test split, and
  report the threshold. The deployed code currently assumes 0.5; if the chosen
  threshold differs, `inference_model.py` must be updated to match.
- Re-measure inference latency on the Pi with the trained weights. Latency is
  architecture-dependent, not weight-dependent, so the current 27.7–42.1 ms should
  hold — but it must be re-measured, not assumed.

## Roadmap

The model is on a separate track owned by a teammate, so nothing below waits on it.
Our work proceeds in parallel and is sequenced by what unblocks whom.

**Track A — hand off to the teammate (our side, do first, small)**

1. Write the **preprocessing script** and hand it over: extract, memory-map, resample
   32 → 100 Hz, window to 1000 samples, subject-level split, save `float32` shards.
   Producing this ourselves rather than describing it removes the largest source of
   train/inference skew, and shrinks the data enough to be workable in Colab.
2. Write the **Keras model definition** matching `ANTIGRAVITY.md` §4.3 exactly, and the
   **weight-export snippet** that writes the `.npz` our NumPy class will read. The
   teammate then owns training, tuning and the metrics table.
3. Write the **equivalence check** — a script that loads an exported `.npz` into the
   NumPy model, runs the same input through both, and asserts agreement. This is the
   acceptance test for whatever comes back from Colab.

**Track B — ours, independent of the model entirely**

4. **Fail loudly on missing weights.** `inference_model.py` currently falls back to
   seeded random initialisation, which is exactly why an untrained model looked
   operational for a week. It should refuse to start without a weights file, or at
   minimum stamp every payload as `model_trained: false` so the dashboard can label the
   output as non-diagnostic. Do this now, before the trained weights arrive.
5. **Quantitative hardware validation** under sustained real pulse: confirm 100 Hz holds
   under load, BPM against a reference device, and the SQI distribution on genuine
   signal. None of this needs the model.
6. **Calibrate the SQI gate** against DeepBeat's `qa_label` ground truth instead of the
   hand-picked 0.7 threshold currently deployed — a defensible number for the thesis
   rather than one chosen by eye.
7. **Purge synthetic data:** repoint `TARGET_PATIENT_ID` at a real clinician-created
   patient, delete the seeded patients, the two compromised accounts and the 224 noise
   events, reinitialise the hash chain from genesis.
8. **Correct `ANTIGRAVITY.md` §5:** the "~13.08 ms" Performance Efficiency figure is a
   stale synthetic-data measurement. The real figure is 27.7–42.1 ms and the 25 ms
   budget is breached.

**Track C — after the trained weights land**

9. Run the equivalence check from step 3; reject the weights if it fails.
10. Deploy, re-measure latency on the Pi, and verify Grad-CAM now highlights
    physiologically plausible regions rather than noise.
11. Address the latency budget — quantisation, or a justified revision of the budget,
    recorded as an ADR.

## Risks

- **Silent weight-loading failure.** If the export/import is wrong, the system will run
  and produce confident numbers that are wrong. The Keras-vs-NumPy equivalence check in
  step 5 is the guard and must not be skipped.
- **Subject leakage across splits.** The most common way PPG papers report inflated
  accuracy. Split by subject.
- **Optimistic thresholds.** Tuning the decision threshold on test data invalidates the
  result.
- **Dataset divergence from the thesis text.** Chapter 2 says MIMIC PERform. Whichever
  corpus is used, the document must be corrected.
- **The model may not reach 90%.** Those are stated targets, not findings. If the
  honest result is lower, report it — a truthful 84% with sound methodology is a
  stronger thesis than an unexplained 95%.

## Related

- [[2026-09-10 - Raspberry Pi Deployment Architecture and Runbook|Deployment architecture & runbook]]
- [[../02 - Code Review/2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights|Untrained weights finding]]
- [[../01 - Literature/Torres-Soto-2020-Multi-task deep learning for cardiac|Torres-Soto & Ashley 2020 — DeepBeat]]
- [[../01 - Literature/Charlton-2022-Wearable Photoplethysmography for|Charlton 2022 — wearable PPG]]

---
Back to [[Index|Claude Notes Index]]
