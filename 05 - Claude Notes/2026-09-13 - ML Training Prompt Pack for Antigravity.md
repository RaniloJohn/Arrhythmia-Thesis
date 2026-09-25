---
tags: [claude-notes, machine-learning, training, antigravity, prompt-pack]
created: 2026-09-13
revised: 2026-09-13 (v2 — dataset reversed to DeepBeat primary after accuracy review)
author: Claude (Opus 5) — architect role
status: ready to execute
---

# ML Training Prompt Pack for Antigravity

Nine sequential prompts that take the 1D-CNN from **untrained random weights** to
**trained, parity-verified, threshold-calibrated, externally-validated weights running on
the Pi**. This closes the top blocker in [[../MEMORY|MEMORY.md]].

**How to use:** paste one prompt per Antigravity Pro turn, **in order**. Prompts 2, 6 and 7
are *gates* — if a gate's acceptance check fails, fix it before moving on. Do not merge
prompts; each is scoped to one reviewable commit.

> **v2 revision note.** v1 of this pack made MIMIC PERform AF the training set. That was
> wrong for accuracy — it was chosen because the thesis chapters already named it. Reversed
> on 2026-09-13 after review: **DeepBeat trains, MIMIC PERform AF externally validates.**
> Reasoning in the table below. Chapter 3's methodology needs an amendment to match; that
> is a Claude task, tracked in `MEMORY.md`, not one of these prompts.

---

## Architectural decisions locked before coding (do not relitigate)

| # | Decision | Rationale |
|---|---|---|
| A1 | **DeepBeat (Torres-Soto & Ashley 2020) is the training set. MIMIC PERform AF (Charlton 2022) is the held-out external validation set.** | Three independent reasons, below. |
| A2 | **Train in PyTorch, infer in NumPy.** PyTorch is a dev-machine-only dependency; the Pi runtime stays pure NumPy. | Preserves settled Decision 4 (no TFLite/ONNX on the edge) while giving real autograd. PyTorch `Conv1d.weight` is `(out, in, k)` — byte-identical layout to `Conv1DBlock.weights`. |
| A3 | **The Chapter 2 topology is frozen.** No new layers, no changed filter counts or kernel sizes. Regularization (dropout, weight decay) is **training-time only** and must vanish at inference. | Ch. 2 Fig 2.1 is written into the thesis. Dropout-at-train-only changes nothing about the deployed forward pass. |
| A4 | **Splits are subject-isolated.** Grouping key = subject ID, always. A window-level random split is forbidden. | 10 s windows from one recording are near-duplicates; window-level splitting leaks the subject across train/test. The most common fatal flaw in PPG-AF papers. |
| A5 | **Training preprocessing imports `signal_processing/`; it never reimplements it.** Exact chain, exact order: `detrend_ppg` → `ButterBandpassFilter(0.5, 5.0, fs=100, order=4).apply` → `zscore_normalize`. | `runner.py:248-250` defines the contract. Note detrend comes **before** the bandpass. Any divergence means the model is trained on a different distribution than it is served. |
| A6 | **Decision threshold is a calibrated artifact, not the literal `0.50`.** Chosen on validation data, stored in the weights metadata. | `inference_model.py` hardcodes `prob >= 0.50`. With class imbalance the operating point must be chosen against the thesis's sensitivity target. |
| A7 | **All datasets are resampled to 100 Hz and re-windowed to 1000 samples** before they touch the model. | Firmware, `runner.py` window_size and the Ch. 2 input length all fix this. One canonical input contract; datasets adapt to it, never the reverse. |

### Why DeepBeat trains and MIMIC validates (A1)

1. **Domain match.** Deployment is a MAX30102: wrist/finger, **reflectance** mode, consumer
   optics, ambulatory patients. DeepBeat is wrist-worn Simband/Cardiaband — reflectance,
   consumer optics, real motion artifact. MIMIC PERform is ICU **transmissive** fingertip
   PPG from a clinical monitor on supine patients. Training on MIMIC trains on a cleaner,
   differently-shaped signal than inference will ever see.
2. **Capacity vs. data.** The frozen topology is ~1.03 M parameters. MIMIC PERform AF gives
   ~4 200 non-overlapping 10 s windows from ~35 subjects — about **245 parameters per
   training window**, squarely a memorization regime. DeepBeat gives on the order of
   500 000 windows from ~175 subjects: ~2 parameters per window, and five times the
   subjects, which is what actually governs generalization.
3. **Label granularity.** MIMIC PERform AF labels are **per recording** — the patient's
   diagnosis applied to all 20 minutes. AF is frequently paroxysmal, so an AF subject's
   sinus segments would be trained as AF. That is irreducible label noise on the positive
   class and a hard ceiling on precision. DeepBeat has **per-window** rhythm labels, and
   its test partition is cardiologist-adjudicated.

Keeping MIMIC as the external set is not a consolation prize — reporting performance on an
independent dataset, different device, different population is a substantially stronger
claim than single-dataset cross-validation, and it keeps MIMIC PERform in the methodology.

**Known caveat to handle, not hide:** DeepBeat's headline metrics have been contested in
follow-up work, and much of its training-partition labelling is algorithm-derived rather
than adjudicated. Mitigation is built into Prompt 5: benchmark internally on its
adjudicated test partition, and treat MIMIC as the independent check.

### Numbers to verify at download, not trust from this note

I am not fully confident of DeepBeat's native sample rate (32 vs 64 Hz), its exact window
count, or MIMIC PERform AF's non-AF subject count (19/19 vs 19/16). **Every loader must
read these from the data and print them**; no prompt below hardcodes a subject or window
count as an acceptance assertion. Verify against each dataset's own documentation.

### The hazard that will silently destroy accuracy if missed

`inference_model.py` flattens with `p2.flatten()` where `p2` has shape `(250, 64)` —
**C-order, time-major**: element index is `t * 64 + c`.
PyTorch flattens `(batch, 64, 250)` — **channel-major**: index is `c * 250 + t`.

The two orderings are different permutations of the same 16 000 values. If the exporter
copies `dense1` weights without correcting for it, the NumPy model still runs, still emits
a plausible probability, and is **completely wrong** — no crash, no warning. Prompt 2
exists solely to catch this class of bug before any training time is spent.

---

## Prompt 0 — Dataset layer: a common loader interface for both datasets

```
Context: Arrhythmia Thesis repo, `03 - ML/`. We are starting model training. Read
`MEMORY.md` and `ANTIGRAVITY.md` §4.3 first. The 1D-CNN in `03 - ML/model/inference_model.py`
currently runs on He-init random weights and has never been trained — that is what this
series of tasks fixes.

Two datasets are involved, with different roles:
- DeepBeat (Torres-Soto & Ashley 2020, npj Digital Medicine) — wrist-worn Simband/
  Cardiaband PPG, ~500k windows, ~175 subjects, PER-WINDOW rhythm labels (AF / non-AF)
  plus signal-quality labels. This is the TRAINING set, because it matches our MAX30102
  reflectance wrist deployment and is large enough for a 1.03M-parameter model.
- MIMIC PERform AF (Charlton et al. 2022) — ICU fingertip PPG at 125 Hz, ~20 min per
  subject, ~19 AF + ~19 non-AF subjects, PER-RECORDING labels. This is the HELD-OUT
  EXTERNAL VALIDATION set only. It must never appear in training or in threshold
  calibration.

Task: build the dataset layer with ONE common interface so the two are swappable.

Create `03 - ML/training/datasets/base.py` defining an abstract `PPGDatasetLoader` with:
    def subjects(self) -> Iterator[SubjectRecord]
where `SubjectRecord` is a dataclass: `subject_id: str`, `signal: np.ndarray` (float32,
1-D), `fs: float`, `label: int` (1 = AF, 0 = non-AF), `quality: Optional[np.ndarray]`,
`source_dataset: str`, and `native_window_s: Optional[float]` (set for pre-windowed
datasets like DeepBeat, None for continuous recordings like MIMIC).

Create `03 - ML/training/datasets/deepbeat.py` — `DeepBeatLoader`:
- reads from `03 - ML/data/deepbeat/`
- DO NOT hardcode the sample rate or window length. Read them from the dataset's own
  metadata/documentation, print what you found, and assert they are self-consistent with
  the array shapes. If the sample rate cannot be determined from the data, stop and ask
  rather than guessing.
- surfaces DeepBeat's official train/validation/test partition assignments and its
  per-window quality labels through the interface, plus subject IDs.
- surfaces which partition is the cardiologist-adjudicated one — we need that separately in
  Prompt 5 as the clean internal benchmark.

Create `03 - ML/training/datasets/mimic_perform.py` — `MimicPerformAFLoader`:
- reads from `03 - ML/data/mimic_perform/`, yields per-subject continuous recordings.
- reads the true AF / non-AF subject counts from the data and prints them; do not assume.

Create `03 - ML/training/resample.py` — a single `to_100hz(signal, fs_in)` helper used by
BOTH loaders, implementing the A7 contract:
- compute the rational ratio with `fractions.Fraction(100, int(fs_in)).limit_denominator()`
  and apply `scipy.signal.resample_poly(signal, frac.numerator, frac.denominator)`
- assert the output length matches `len(signal) * 100 / fs_in` to within one sample
- for DeepBeat this is UPSAMPLING, which is information-preserving for our purposes and
  must be documented as such in a comment: our bandpass is 0.5–5 Hz, far below even a
  32 Hz Nyquist, so no frequency the model uses is fabricated or lost. Do not let this
  become a silent assumption — write the reasoning down.

Also create:
- `03 - ML/data/README.md` — for each dataset: source URL, access/licence procedure
  (DeepBeat requires Stanford registration), citation, expected directory layout, and its
  ROLE (train vs external validation) with a one-line reason.
- `03 - ML/training/requirements-train.txt` pinning torch, scipy, numpy, scikit-learn,
  matplotlib. DEV-MACHINE ONLY — never imported by anything under `edge_inference/`,
  `model/`, `signal_processing/` or `storage/`.

Fix `.gitignore`: keep `03 - ML/data/deepbeat/` ignored, ADD `03 - ML/data/mimic_perform/`,
`03 - ML/data/processed/` and `03 - ML/model/weights/*.npz`, and keep a `.gitkeep` in each
directory so they survive a clone.

Constraints: additive only. Do not modify anything under `model/`, `signal_processing/`,
`edge_inference/` or `storage/` in this task. Each loader must raise a clear, actionable
error if its data directory is missing — never silently fall back to synthetic data.

Acceptance: `python -m training.datasets.deepbeat --summary` and
`python -m training.datasets.mimic_perform --summary` each print the detected sample rate,
window length where applicable, subject count, AF/non-AF balance and total duration.
Report the actual numbers found so we can correct this note's estimates.
```

---

## Prompt 1 — Windowing, quality gating, and subject-isolated splits

```
Context: continues the training work. Both loaders expose the common `PPGDatasetLoader`
interface and resample to 100 Hz.

Task: `03 - ML/training/build_dataset.py` — turn loader output into windowed, split,
on-disk arrays. It must work for BOTH datasets via a `--dataset {deepbeat,mimic}` flag,
because the external-validation set has to go through the identical pipeline or the
comparison is meaningless.

Requirements:
1. Windowing to the canonical contract: 1000 samples = 10 s @ 100 Hz, matching
   `runner.py` window_size.
   - DeepBeat is already windowed at a longer native length (~25 s). After resampling to
     100 Hz, slice each native window into consecutive 1000-sample windows, inheriting the
     parent window's rhythm label and quality label. Discard any remainder shorter than
     1000 samples. Record how many child windows each native window produced.
   - MIMIC is continuous: slide non-overlapping 1000-sample windows across each recording,
     inheriting the subject's recording-level label. Note in the output metadata that this
     is a coarse label and a documented limitation of the MIMIC PERform AF annotations.
   - `--train-overlap` (default 0.0 for DeepBeat, since it is already large) applies to the
     TRAIN split only. Never overlap val/test windows.
2. Preprocessing — import the production functions, do NOT reimplement:
       from signal_processing.filter import detrend_ppg, zscore_normalize, ButterBandpassFilter
       from signal_processing.sqi import SignalQualityAssessor
   Apply in exactly the order `runner.py:248-250` uses:
       detrended  = detrend_ppg(raw_window)
       filtered   = bp_filter.apply(detrended)
       normalized = zscore_normalize(filtered)
   with `bp_filter = ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=100.0, order=4)`.
   Store `normalized` as the model input.
3. Quality gating, two sources:
   - Compute our own `sqi_score` per window with `SignalQualityAssessor` and store it.
   - Where the dataset provides quality labels (DeepBeat does), store those too as
     `dataset_quality`.
   - Drop windows from the TRAIN split only when the dataset labels them poor quality
     OR our `sqi_score < 0.5`. Keep every window in val/test so metrics reflect real input.
   - Print a cross-tabulation of `dataset_quality` against our `sqi_score` bins. This is a
     free validation of the hand-tuned thresholds in `sqi.py` — report whether they agree.
4. Splitting — this is the part that must be right.
   - For DeepBeat: prefer the dataset's OWN official train/validation/test partitions, but
     first VERIFY they are subject-disjoint. If any subject appears in two official
     partitions, discard the official split and regroup from scratch by subject.
   - For MIMIC: the entire dataset is one external-validation split. No train/val split.
   - Never split by window. Grouping key is always subject_id.
   - Write the subject→split assignment to
     `03 - ML/training/splits/<dataset>_split_manifest.json`, seeded and reproducible.
   - ASSERT the subject-ID intersection between any two splits is empty and fail loudly.
   - ASSERT no subject_id appears in both the DeepBeat and MIMIC manifests (they are
     different cohorts, so this should be trivially true — but assert it, because a
     subject-ID collision across datasets would silently corrupt the external-validation
     claim, which is the strongest claim in the thesis).
5. Output `03 - ML/data/processed/<dataset>_{train,val,test}.npz` with arrays `X` (float32,
   (N, 1000)), `y` (int8), `subject_id` (str), `sqi` (float32), `dataset_quality` (where
   available). Write a sibling `<dataset>_manifest.json` recording window counts, AF
   prevalence, drop counts by reason, and the preprocessing chain used.

Acceptance: both datasets build; per-split window counts and AF prevalence printed; all
subject-overlap assertions pass; the quality cross-tabulation is reported. Warn if any
split has fewer than 4 subjects.
```

---

## Prompt 2 — GATE: PyTorch mirror + bit-parity exporter

```
Context: this is the most important task in the training series, and it involves NO
training. We must prove that a PyTorch model and the production pure-NumPy model in
`03 - ML/model/inference_model.py` compute the SAME function, and that weights can move
from one to the other losslessly. If this is wrong, every later result is worthless.

Task A — `03 - ML/training/torch_model.py`: a `torch.nn.Module` mirroring
`Arrhythmia1DCNN` exactly:
  Conv1d(1, 32, kernel_size=5, padding=2) -> ReLU -> MaxPool1d(2)
  Conv1d(32, 64, kernel_size=3, padding=1) -> ReLU -> MaxPool1d(2)
  Flatten -> Linear(16000, 64) -> ReLU -> Linear(64, 1) -> sigmoid
Add `nn.Dropout(p=dropout)` before `Linear(16000, 64)`, default p=0.5, exposed as a
constructor argument. Dropout is training-only and is identity under `model.eval()`, so the
deployed topology is unchanged (locked decision A3).

CRITICAL — flatten ordering. `inference_model.py` computes `p2.flatten()` where `p2` has
shape `(time=250, channels=64)`, giving C-order **time-major** indexing `t * 64 + c`.
PyTorch's tensor is `(batch, channels=64, time=250)`, so a plain flatten gives
**channel-major** `c * 250 + t`. These are different permutations of the same 16000 values.
Handle it in ONE place and document it: in the torch module, permute to
`(batch, time, channels)` before flattening, i.e. `x = x.permute(0, 2, 1).reshape(batch, -1)`,
so the torch flatten order matches the NumPy runtime's. Do not also transpose in the
exporter to compensate — pick the module-side permute, and leave a comment naming
`inference_model.py`'s `p2.flatten()` as the reason.

Task B — `03 - ML/training/export_weights.py`, converting a trained torch state_dict to the
NumPy runtime's format, saved as `.npz`:
  conv1.weights     <- conv1.weight.numpy()         # (32, 1, 5)  — same layout, no transpose
  conv1.bias        <- conv1.bias.numpy()           # (32,)
  conv2.weights     <- conv2.weight.numpy()         # (64, 32, 3) — same layout
  conv2.bias        <- conv2.bias.numpy()           # (64,)
  dense1.weights    <- linear1.weight.numpy().T     # torch (64, 16000) -> numpy (16000, 64)
  dense1.bias       <- linear1.bias.numpy()         # (64,)
  dense_out.weights <- linear2.weight.numpy().T     # torch (1, 64) -> numpy (64, 1)
  dense_out.bias    <- linear2.bias.numpy()         # (1,)
Everything float32. Also write a sibling `<name>.meta.json` holding: git commit, training
date, TRAINING dataset name and version, EXTERNAL VALIDATION dataset name, split manifest
hashes, the calibrated decision threshold (default 0.5 until Prompt 5 sets it), internal
and external metrics, and an `input_contract` block recording fs=100, window=1000 and the
preprocessing chain order.

Task C — minimal edits to `03 - ML/model/inference_model.py` so weights can actually be
loaded. Add `Arrhythmia1DCNN.load_weights(path: str)` which loads the .npz, ASSERTS every
array's shape against the layer it targets (fail loudly on mismatch, never broadcast),
assigns them, reads the threshold from the sibling meta json into `self.threshold`, and sets
`self.weights_loaded = True` and `self.weights_source = path`. Replace the hardcoded
`prob >= 0.50` with `prob >= self.threshold` (`self.threshold` defaults to 0.5). Delete the
`_calibrate_weights()` fake-physiological-prior hack and its call — it is a placeholder that
becomes actively misleading once real weights exist; keep `self.weights_loaded = False` as
the default so callers can detect an untrained model. Do not otherwise change the forward
pass, and keep the `forward_with_cache` return-dict keys identical, because `grad_cam.py`
depends on `a2`, `p2`, `argmax2`, `flat`, `a_dense1` and on `model.dense1.weights` /
`model.dense_out.weights`.

Task D — `03 - ML/tests/test_parity.py` (pytest). The gate:
  1. Build a randomly-initialized torch model, export it, load it into
     `Arrhythmia1DCNN(input_length=1000)`.
  2. Push 32 random float32 windows of shape (1000,) through both, torch under
     `model.eval()` and `torch.no_grad()`.
  3. Assert `np.allclose(torch_prob, numpy_prob, atol=1e-5)` for every window.
  4. Add a NEGATIVE control: deliberately export `dense1` with the wrong flatten
     permutation and assert the parity check FAILS. This proves the test can actually
     detect the ordering bug rather than passing vacuously.
  5. Assert `load_weights` raises on a shape-mismatched npz.

Acceptance: `pytest "03 - ML/tests/test_parity.py" -v` — all pass, including the negative
control. Report the maximum absolute probability difference observed. STOP and report if it
exceeds 1e-5; do not proceed to training.
```

---

## Prompt 3 — Training loop

```
Context: parity between the torch model and the NumPy runtime is proven, and the windowed
subject-isolated DeepBeat dataset exists. Now train. MIMIC must not be touched by this task.

Task: `03 - ML/training/train.py`, training on the DeepBeat train split, early-stopping on
the DeepBeat validation split.

Compute budget matters here, unlike the small-dataset plan it replaces. Roughly 4.3 M MACs
per window forward (conv2 and dense1 dominate) times ~500 k windows is a few TMAC per
epoch. On a GPU that is minutes per epoch; on CPU it can be tens of minutes. So:
- auto-detect CUDA and use it; print the device chosen
- `--max-train-windows` (default: no cap) to subsample for a fast smoke run, with the
  subsample taken BY SUBJECT so a capped run is still subject-isolated
- `--epochs` default 30 (not 100 — with this much data, convergence is far earlier)
- checkpoint every epoch, so a long run can be resumed rather than restarted

Hyperparameters:
- AdamW, weight decay default 1e-4, lr default 1e-3, ReduceLROnPlateau on val loss
- `BCEWithLogitsLoss` — the model emits logits during training and sigmoid is applied at
  eval; confirm this stays consistent with the parity harness — with `pos_weight` computed
  from the train-split class ratio
- `nn.Dropout(p=0.5)` before Linear(16000, 64), already in `torch_model.py`. Keep it even
  though ~500k windows against 1.03M parameters is a far healthier ratio than the small-
  dataset alternative — subject count (~175), not window count, is the real generalization
  constraint, and dropout is nearly free.
- batch size 64, early stopping on **validation AUROC** with patience 5, best checkpoint
  restored
- `--seed` default 42, seeding torch, numpy and python RNGs, value recorded in metadata

Log per epoch to `03 - ML/training/runs/<timestamp>/history.csv`: train loss, val loss, val
AUROC, val AUPRC, val sensitivity, val specificity, lr, epoch wall-clock. Save `best.pt`,
and at the end automatically call the Prompt 2 exporter to produce
`03 - ML/model/weights/cnn_af_v1.npz` and `cnn_af_v1.meta.json`.

Hard guards:
- REFUSE to train if the split manifest's subject-overlap assertion does not pass.
- REFUSE to train if any MIMIC-derived file appears in the training data paths. The
  external-validation claim depends on MIMIC never having been seen; enforce it in code,
  not by convention.
- Print train/val subject counts and AF prevalence before the first epoch.

Acceptance: a full run completes, `history.csv` shows val AUROC rising well above 0.5, and
the exported .npz loads cleanly into `Arrhythmia1DCNN.load_weights`. Report device, epochs
run, best val AUROC, and wall-clock time.
```

---

## Prompt 4 — Threshold calibration

```
Context: trained weights exist. The decision threshold is still the placeholder 0.5.

Task: `03 - ML/training/calibrate_threshold.py`.

Choose the threshold on the DeepBeat VALIDATION split — never on any test split, and never
on MIMIC. Select the smallest threshold meeting the thesis's target sensitivity; read that
target from `04 - Thesis Reference/Thesis Overview.md`, and if no explicit figure is stated
there, use the Youden J statistic and say so plainly in the output.

Write the chosen value into `cnn_af_v1.meta.json` as `decision_threshold`, so
`Arrhythmia1DCNN.load_weights` picks it up automatically. Print the full
threshold-vs-sensitivity/specificity/PPV/NPV table at 0.01 resolution so the choice is
auditable, and a reliability/calibration diagram.

Also report whether the model is well-calibrated in the probabilistic sense (Brier score,
and the calibration curve's deviation from the diagonal). The dashboard shows a probability
to a clinician, so systematic over-confidence is a clinical-safety issue, not just a
metric. If it is badly calibrated, note that Platt scaling or isotonic regression on the
validation split is the fix — but do NOT apply it yet, because it would add a step the
NumPy runtime does not implement. Flag it for a decision instead.

Acceptance: `decision_threshold` in the meta json is no longer the placeholder 0.5 unless
calibration genuinely selected 0.5; the table, Brier score and calibration plot exist.
```

---

## Prompt 5 — Evaluation: internal benchmark + external validation

```
Context: trained and calibrated weights exist. Now measure them honestly — these numbers
go into the thesis, so the evaluation code must be conservative and reproducible.

There are TWO evaluations and they must be reported separately and never averaged:
  (a) INTERNAL — DeepBeat's held-out test partition. Use the cardiologist-adjudicated
      partition where DeepBeat distinguishes one; say which partition was used, because
      DeepBeat's algorithm-derived labels are weaker evidence and follow-up work has
      contested its headline metrics.
  (b) EXTERNAL — the full MIMIC PERform AF set, run through the identical Prompt 1
      pipeline. Different device, different population, unseen during training and
      calibration. This is the honest generalization number and should be the headline
      result in the thesis even if it is lower than (a). It almost certainly will be lower;
      that is the finding, not a failure.

Task A — `03 - ML/training/evaluate.py`, taking `--split {deepbeat_test,mimic_external}`:
- Window-level metrics: AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, accuracy and
  the confusion matrix, each with 95% CIs by bootstrap resampling **over subjects**, not
  over windows. Windows within a subject are correlated, so bootstrapping windows would
  report falsely tight intervals.
- Subject-level metrics: aggregate each subject's windows by majority vote and by mean
  probability, then report sensitivity/specificity per subject. This is the number a
  clinician actually cares about.
- Stratify the internal results by the stored quality labels — performance on poor-quality
  windows tells us whether the SQI gate in `runner.py` needs to become a hard gate rather
  than a reported metric.
- Plots per split into `03 - ML/training/runs/<timestamp>/`: ROC, precision-recall,
  confusion matrix, calibration diagram.

Task B — `03 - ML/training/cross_validate.py`: 5-fold GroupKFold over DeepBeat subjects,
retraining from scratch per fold, reporting mean ± std of every metric, so the internal
number carries a variance estimate rather than being a single lucky split.

Task C — write the results up at
`02 - Code Review/2026-09-13 - 1D-CNN Training Results & Honest Performance Bounds.md`
using `_Templates/Code Review Note Template.md`, linked from `02 - Code Review/Index.md`.
State plainly:
- training dataset and version, subject counts per split, and that splits were
  subject-isolated with the assertion that enforced it
- internal (DeepBeat test) and external (MIMIC PERform AF) metrics side by side, per-window
  AND per-subject, with subject-bootstrapped CIs, plus the cross-validated spread
- the internal-vs-external gap, explicitly, as the measure of domain generalization
- DeepBeat's label-provenance caveat and MIMIC's coarse per-recording label caveat
- that NEITHER dataset is MAX30102 data. DeepBeat is wrist reflectance like ours, so the
  shift is small, but it is a different sensor and remains unvalidated on our hardware.
- do not round metrics upward, and never report a single-split number without its
  cross-validated counterpart

Acceptance: both evaluations, both plot sets, the CV spread and the note are produced, and
the note contains an explicit statement of the internal-to-external performance gap.
```

---

## Prompt 6 — GATE: wire trained weights into the edge runtime

```
Context: trained, calibrated, externally-validated weights exist at
`03 - ML/model/weights/cnn_af_v1.npz`. Get them into the live pipeline without breaking the
25 ms latency budget or the offline constraint.

Task:
1. `03 - ML/edge_inference/runner.py`: after constructing `Arrhythmia1DCNN`, load the
   weights. Path resolution order: the `--weights` CLI argument, then the
   `ARRHYTHMIA_WEIGHTS` environment variable, then the default
   `03 - ML/model/weights/cnn_af_v1.npz`. If the file is absent, log a loud, unmissable
   WARNING that the model is running UNTRAINED and set `model_trained: false` in every
   emitted payload — never fail silently into random-weight inference.
2. Add to the JSON payload emitted over the TCP bridge and the HTTP ingest:
   `model_trained` (bool), `model_version`, `decision_threshold`, and `training_dataset`
   (from the meta json). The website needs these to render the scaffolding badge from the
   MEMORY.md idea inbox, and a clinician-facing UI should be able to say what the model
   was trained on.
3. Re-benchmark on the Raspberry Pi 4 (Tailscale host `raspberrypi`, project at
   `/home/ranilo/Arrhythmia Thesis/`): 100 consecutive windows, reporting mean, p50, p95
   and max for DSP, inference and Grad-CAM separately. The <25 ms inference budget from
   ANTIGRAVITY.md §4.3 must still hold — trained weights do not change the arithmetic cost,
   so a regression here means something else broke.
4. Confirm torch is NOT importable anywhere in the edge path: grep `edge_inference/`,
   `model/`, `signal_processing/` and `storage/` for `torch` and assert zero hits. The Pi
   must stay pure-NumPy per settled Decision 4.
5. Restart both systemd units on the Pi and confirm the dashboard shows a real probability
   produced by trained weights.

Acceptance: report the p95 inference latency, the grep result, and a sample emitted payload
showing `model_trained: true` with its version and training-dataset strings.
```

---

## Prompt 7 — GATE: re-validate Grad-CAM against trained weights

```
Context: `03 - ML/model/grad_cam.py`'s math was reviewed as correct, but it has only ever
explained a random-weight model, so its output has never meant anything. MEMORY.md lists
Grad-CAM re-validation as deferred; trained weights make it actionable.

Task:
1. `03 - ML/tests/test_gradcam.py`: verify the analytic gradient path in `GradCAM1D` against
   `torch.autograd` on identical weights — compute d(logit)/d(conv2 activations) both ways
   and assert they agree to 1e-4. This proves the hand-derived backprop through
   `dense_out -> relu -> dense1 -> unpool -> a2` is right, now that the weights mean
   something.
2. Sanity-check the upsampling from the 500-sample `a2` axis back to the 1000-sample input
   axis, and that the returned weights are normalized to [0, 1] with length 1000.
3. Clinical plausibility check: for 10 correctly-classified AF windows and 10 correctly-
   classified non-AF windows — draw them from the MIMIC external set, not DeepBeat, since
   MIMIC has simultaneous ECG to corroborate where the irregular beats actually are —
   overlay the Grad-CAM heat strip on the PPG waveform and save the figures. Then judge and
   write down whether AF attributions actually concentrate on the irregular inter-beat
   intervals and absent dicrotic notches. If they look like noise, say so — a correct
   implementation explaining a weak model is a finding, not a failure to hide.
4. Append the verdict to
   `02 - Code Review/2026-09-13 - 1D-CNN Training Results & Honest Performance Bounds.md`.

Acceptance: the gradient-agreement test passes, the 20 overlay figures exist, and the note
carries an explicit plausible / not-plausible verdict with reasoning.
```

---

## Prompt 8 — Close the Python test-suite blocker

```
Context: MEMORY.md lists "no Python test suite under `03 - ML/tests/`" as an open blocker
that hurts the ISO/IEC 25010 Maintainability row. Prompts 2 and 7 already created
`test_parity.py` and `test_gradcam.py`; finish the suite.

Task: build out `03 - ML/tests/` with pytest coverage for the modules that already exist:
- `test_filter.py` — Butterworth passband/stopband attenuation on synthetic tones (a 0.2 Hz
  drift and a 12 Hz tone must be attenuated, a 1.2 Hz pulse must survive), zero-phase
  behaviour, detrend, and z-score output statistics.
- `test_sqi.py` — clean synthetic PPG scores high, a flat/DC-only trace and a
  motion-corrupted trace are rejected, and the `dc_val < 1000` probe-off branch fires.
- `test_peak_detection.py` — Elgendi peak count and BPM on a synthetic 72 BPM signal, plus
  a known-irregular IBI series.
- `test_serial_protocol.py` — round-trip a framed `0xAA`/`0x55` packet, and confirm a
  corrupted CRC16 is rejected.
- `test_db_manager.py` — hash-chain append, chain verification, and detection of a tampered
  row.
- `test_inference_model.py` — output shape and range, threshold behaviour, `load_weights`
  shape assertions, and that `weights_loaded` is False before loading.
- `test_resample.py` — `to_100hz` round-trips a synthetic 1.2 Hz pulse from 32, 64 and
  125 Hz to 100 Hz with the peak frequency preserved, and its output length assertion holds.
Add `03 - ML/pytest.ini` (or a `pyproject.toml` section) configuring rootdir and test paths
so `pytest "03 - ML/tests"` works from the repo root, plus a short `03 - ML/tests/README.md`.

Constraints: tests must not require the MAX30102 hardware, the Pi, either dataset, or
network access — use synthetic signals and temp-file SQLite databases. They must run on
Windows and Linux.

Acceptance: `pytest "03 - ML/tests" -v` fully green. Report test count and runtime, and
update the Python-test-suite row in MEMORY.md's State of Play from ❌ to ✅.
```

---

## Execution order and what each prompt unblocks

| Prompt | Produces | Unblocks |
|---|---|---|
| 0 | common loader interface, both datasets, 100 Hz resampler, fixed `.gitignore` | swappable datasets; verified real subject/rate numbers |
| 1 | windowed npz + subject-isolated split manifests for both datasets | prevents the leakage that would invalidate every metric |
| 2 **gate** | torch mirror, exporter, `load_weights`, parity test | makes trained weights *transferable* at all |
| 3 | `train.py`, trained checkpoint, exported npz | the core blocker: real weights |
| 4 | calibrated threshold + calibration diagnostics | a defensible operating point, not a magic 0.5 |
| 5 | internal + external metrics with subject-bootstrapped CIs, 5-fold CV, results note | the thesis's headline numbers |
| 6 **gate** | weights wired into `runner.py`, Pi re-benchmark | live system runs a trained model |
| 7 **gate** | Grad-CAM validated against autograd + plausibility figures | closes the deferred Grad-CAM item |
| 8 | full pytest suite | closes the Maintainability blocker |

Prompts 0–2 can be *written* without data, but Prompt 0's acceptance checks and everything
from Prompt 1 onward need it. **Start the DeepBeat Stanford registration/licence request
now** — it is the longest-lead item in this plan. MIMIC PERform AF is openly available and
is not needed until Prompt 5.

### Not an Antigravity task — Claude's follow-up

Chapter 3's methodology names MIMIC PERform as the training dataset and needs an amendment
to the new roles (DeepBeat trains, MIMIC PERform externally validates), with the
domain-match, data-volume and label-granularity justification written out for the adviser.
Per the dual-agent split that is Claude's work, not Antigravity's. Tracked in `MEMORY.md`.

Until Prompt 5's results note exists, the MEMORY.md caveat stands unchanged: **the model is
untrained and no accuracy figure anywhere is a measured result.**

---
Back to [[Index|05 - Claude Notes Index]] · [[../MEMORY|MEMORY.md]] · [[../PLAN|PLAN.md]]
