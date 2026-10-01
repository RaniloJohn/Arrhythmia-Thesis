---
date: 2026-10-01
component: 1D-CNN training pipeline (03 - ML/training/), shipped weights cnn_af_v1.npz
reviewer: Claude (Opus 5)
tags: [code-review, model, machine-learning, validation, data-integrity, blocker]
status: open
---

# 2026-10-01 — 1D-CNN Training Audit: Label/Subject Collinearity and Invalid Shipped Weights

Independent audit of the training work recorded in
[[2026-09-13 - 1D-CNN Training Results & Honest Performance Bounds|2026-09-13 - 1D-CNN Training Results & Honest Performance Bounds]].

## Verdict

**The reported numbers are real — I reproduced them exactly — but the conclusion drawn from
them is wrong, and the shipped model must not be presented as trained.**

Re-running `model/weights/cnn_af_v1.npz` through the pure-NumPy runtime reproduced
Antigravity's figures to four decimals (DeepBeat test AUROC `0.5345`, MIMIC `0.4942`).
The evaluation code is honest and there is no fabrication. The problem is upstream, in the
dataset construction, and it invalidates the training run rather than merely weakening it.

The decisive measurement is one that was never taken: **the model scores AUROC `0.3487` on
its own training distribution** (20,000-window sample of `deepbeat_train.npz`). A model
cannot land below chance on the data it was fit to unless the labels it was fit against are
internally contradictory. This is not a generalization gap; it is a data-integrity defect.

---

## Finding 1 — CRITICAL: `parameters[:, 2]` is file-local, so the regrouping merged contradictory labels into single pseudo-subjects

`training/datasets/deepbeat.py:343` takes the subject ID from `parameters[:, 2]` of the
DeepBeat `.npz` containers. That column is **not globally unique across files**. Measured
directly from the raw archives:

| Subject ID | `validate.npz` | `test.npz` |
|---|---|---|
| 149 | n=6,320, **100.00% AF** | n=2,368, **0.00% AF** |
| 150 | n=8,623, **100.00% AF** | n=2,544, **0.00% AF** |
| 151 | n=31,970, **100.00% AF** | n=1,328, **0.00% AF** |
| 152 | n=218, **100.00% AF** | n=10, **0.00% AF** |
| 153 | n=476, **100.00% AF** | n=8, **0.00% AF** |

The same identifier carries the **opposite label** in the two files. `build_dataset.py`
detected the ID collision (logged as "official partition subject overlap `146..153`") and
responded by *regrouping by subject*, which merged the two populations. That is why
`deepbeat_train.npz` reports subject `150` at 78.9% AF when the subject is 100% pure in
every source file: `8623 / (8623 + 2544) = 77.2%`, matching the observed value after quality
gating.

Consequence: the training set presents **near-identical waveforms under opposite labels**.
This fully accounts for the sub-chance training-set AUROC, and it means
`subject_isolation_verified: true` in `deepbeat_manifest.json` is asserting something the
data cannot support.

## Finding 2 — CRITICAL: AF label is perfectly collinear with subject identity, so the task as constructed is subject recognition, not rhythm classification

Within each source file, **every subject is either 0.00% or 100.00% AF — 0 of 16 subjects
contain both classes**, and 0 of 120 `(subject, session-letter)` pairs do either.

```
RAW validate.npz:  138-148 -> 0.00% AF        149-153 -> 100.00% AF
RAW test.npz:      146-161 -> 0.00% AF        162-167 -> 100.00% AF
```

There is **zero within-subject rhythm contrast anywhere in the data**. The only signal
available to a classifier is inter-subject morphology — perfusion, skin tone, device fit,
baseline wander — which is exactly the thing that does not transfer. Every window-level
metric in the previous review is therefore 14 (or 16) independent subject decisions
replicated thousands of times: the effective sample size on the test set is **14, not
17,106**. Antigravity's own subject-level bootstrap already revealed this — specificity CI
`[4.38%, 97.83%]` — but the summary still led with the window numbers.

This also means **Finding A of the previous review is not supported**. The MIMIC collapse
was attributed to a wrist-reflectance vs. fingertip-transmissive sensor-physics gap. But
in-domain performance is also at chance (test AUROC `0.5345`, 95% CI `[0.4211, 0.9670]`,
Youden's J **negative** at `-0.0369`). A domain gap cannot explain failure on the source
domain. The sensor-physics discussion is plausible physiology, but it is not what these
numbers measure.

## Finding 3 — CRITICAL: the shipped weights come from a smoke-test-sized run

`cnn_af_v1.meta.json` records what actually produced the production weights:

- **4 of 12 available training subjects** (`147, 150, 152, 153`), **21,938 of 451,791
  windows = 4.9%** of the built split.
- `subsample_by_subject()` (`train.py:190`) sorts subjects **ascending by window count** and
  packs smallest-first, so it selected the four smallest and discarded subject `151`
  (43,908 windows), the largest AF subject.
- Of the four, exactly **one** (`147`) contributes non-AF data. The run had a single
  subject's worth of normal-rhythm morphology, and 75% of its windows were AF.
- `"best_epoch": 1` of 6 — validation AUROC peaked at epoch 1 (`0.6346`) and fell
  monotonically thereafter (`0.610 -> 0.579 -> 0.555 -> 0.578 -> 0.567`). Epoch 1 is near
  initialization; early stopping selected the least-trained checkpoint.
- An earlier run in `runs/20260913_203440/` had validation loss of **131 -> 389** (diverged)
  and never exceeded AUROC `0.494`.

The `--max-train-windows` / `--max-val-windows` caps default to `None`, so this was a
deliberately capped invocation. A capped smoke run was then exported as the production model
and wired into the edge runner.

`MEMORY.md` and the previous review both state the model was trained on DeepBeat; the
review's status table says "724k windows". The real figure is **21,938**, a 33×
overstatement.

## Finding 4 — MAJOR: the SQI gate is inoperative on file-based data and cannot transfer to the device

`signal_processing/sqi.py:39-44` computes the perfusion index as
`dc_val = np.mean(raw_data)` and treats `dc_val < 1000.0` as "probe off", forcing `pi = 0.0`.
That threshold is calibrated for **raw MAX30102 ADC counts**. Dataset windows are scaled
floats, so the branch fires on every window and the `-0.4` PI penalty is applied
unconditionally.

Verified consequence: **no window in the entire corpus can score above 0.6**. Observed SQI
values across all 1,072,798 windows are exactly `{0.0, 0.3, 0.6}`, and
`deepbeat_manifest.json`'s own cross-tabulation has **0 counts in the `SQI >= 0.70` column**.
`deepbeat_train.npz` has `sqi` constant at `0.6`.

So the gate at `sq < 0.50` is in practice a skewness/kurtosis filter that **discarded 348,027
windows (43.5% of train)** on a metric whose dominant term is broken. It is also applied to
the train split only (`build_dataset.py:516`), leaving val and test ungated — an engineered
train/eval distribution mismatch. And because PI *does* work on the Pi, where counts exceed
1000, the offline gate and the deployed gate are not the same function.

The previous review describes this as a "Shannon-entropy / skewness Signal Quality Index".
There is no Shannon entropy in the implementation; it is PI + skewness + kurtosis.

## Finding 5 — MAJOR: the Grad-CAM clinical plausibility verdict rests on a chance-level classifier

The Grad-CAM validation (`training/runs/gradcam_validation/`) draws its "10 AF true
positives" from **MIMIC**, the cohort where AUROC is `0.4942`. Their predicted probabilities
are `0.387-0.429` against a threshold of `0.37` — they are "true positives" only because the
threshold flags nearly everything. The conclusion that a 2.38× inter-beat saliency ratio
"**proves** the 1D-CNN filters have learned to attend to irregular cycle lengths" cannot be
drawn from a model with no measurable discrimination, and it rests on N=10 vs. N=10 with no
significance test.

The gradient mathematics is sound and worth keeping: analytical `d logit / d A2` matches
`torch.autograd` to `6.22e-06`. That verifies the *implementation*, not the *interpretation*.
**This matters because RQ3 depends on it** — the Grad-CAM plausibility argument is currently
the thesis's answer to the interpretability research question, and it is not yet earned.

## Finding 6 — MAJOR: threshold `0.37` is a degenerate operating point

Calibration selected `tau = 0.37` by Youden's J on the full validation split, yielding
**sensitivity 95.28% at specificity 10.52%, PPV 5.24%, Youden's J = 0.058**. The model flags
87.3% of all test windows as AF. The headline "85.4% sensitivity" is the trivial
always-say-AF corner, not a detection result.

`cnn_af_v1.meta.json` reports this itself and is right to: `brier_skill_score = -3.9523`
(worse than predicting the base rate), `expected_calibration_error = 0.424`,
`calibration_assessment: "POORLY_CALIBRATED"`.

Two further inconsistencies:
- Early stopping used a **17,562-window val subsample at 72% AF prevalence**; threshold
  calibration used the **full 255,874-window split at 4.94%**. Two different distributions
  called "validation".
- The model's output range is compressed to `[0.003, 0.527]` — it never emits a probability
  above `0.53` on any dataset. All decisions are made inside a narrow band around `0.37`.

## Finding 7 — MODERATE: DeepBeat's actual training partition was never acquired

`data/deepbeat/` contains only `validate.npz` and `test.npz` (plus four Keras `.h5`
checkpoints). DeepBeat's `train` partition is absent. "Trained on DeepBeat" therefore means
trained on DeepBeat's *validation* split, regrouped. This is the reason only 30 subject IDs
exist where `MEMORY.md` plans around "~175 subjects" as the generalization constraint.

## Finding 8 — MODERATE: 5-fold CV inherits every defect above

Mean AUROC `0.3302 ± 0.1179`, with **4 of 5 folds below chance** (`0.183`-`0.518`). The
previous review reads this as "high inter-subject morphological variance". Consistently
sub-chance ranking is not variance — it is inverted ranking, the expected result of Findings
1 and 2. Fold validation sizes also range from 833 to 80,776 windows, so the unweighted mean
is not a meaningful summary. Each fold trained only 3 epochs on ~16-24k windows.

## Finding 9 — MODERATE: the dashboard now reports a chance-level model as trained

`edge_inference/runner.py:450` emits `"model_trained": bool(self.model.weights_loaded)` —
true whenever a weights file loads, with no skill criterion. The prior UNTRAINED banner
(`runner.py:222`) now stays silent, so the clinician dashboard presents AF probabilities
from a chance-level classifier with `model_trained: true`. On the honesty axis this is a
**regression** from the random-weights state, which at least announced itself.

## Minor

- The previous review says "ESP32-S3"; the hardware is **ESP32-C3**. Worth fixing before it
  propagates into the thesis.
- 6 of 14 test "subjects" contribute <=24 windows each. The subject-level specificity of 75%
  is computed over 8 non-AF subjects, 6 of which are essentially empty.
- Subject-level sensitivity "100%" is 6/6 — exact binomial 95% CI is about `[54%, 100%]`.

---

## What is actually sound

Worth stating plainly, because most of the engineering is good and should be kept:

- **PyTorch<->NumPy parity** is real and valuable. The time-major vs. channel-major flatten
  trap was caught before training, with a negative control (`max_diff = 3.42e-02` on the
  wrong permutation vs. `5.96e-08` on the right one).
- **Grad-CAM gradient implementation** is verified against autograd.
- **Anti-contamination guards** in `train.py` and `calibrate_threshold.py` work: MIMIC
  genuinely never entered training or calibration.
- **Evaluation and reporting code is honest** — it computed and published subject-level
  bootstrap CIs, negative Youden's J, and a `POORLY_CALIBRATED` verdict rather than hiding
  them. I reproduced its outputs exactly.
- **Latency**: the <25 ms budget claim holds (mean 16.39 ms, p95 18.43 ms), and zero `torch`
  imports in edge modules is confirmed.

The defect is confined to dataset construction and to the interpretation layered on top of
correct measurements.

---

## Required actions, in order

1. **Stop describing the 1D-CNN as trained.** Revert `MEMORY.md`'s State of Play for the
   model and Grad-CAM rows, and restore the honest-caveat wording. Set
   `model_trained: false` in telemetry, or gate it on a measured-AUROC floor, until a model
   clears one.
2. **Fix the subject identity.** Key subjects as `(source_file, parameters[:,2])` — or
   better, `(file, parameters[:,2], parameters[:,1])` — so no two populations merge.
   Re-assert disjointness afterwards; the current `subject_isolation_verified` flag is not
   meaningful.
3. **Confront Finding 2 head-on — this is the real blocker.** With AF perfectly collinear
   with subject identity, DeepBeat *as downloaded* cannot support an AF-vs-non-AF model, no
   matter how the training loop is tuned. Options, best first:
   - Acquire DeepBeat's real **`train` partition** and check whether subjects there carry
     both classes. If they do, the problem dissolves.
   - If no partition has within-subject contrast, the thesis must either (a) move to a
     dataset that does, or (b) reframe the contribution away from a trained classifier.
     MIMIC PERform AF has 19 AF / 16 non-AF subjects but is also per-recording, so it does
     not by itself solve this.
   - Either way, report **subject-level** metrics with exact CIs as primary. Window-level
     numbers on this data structure are not interpretable.
4. **Fix the SQI PI branch.** Normalize to the signal's own scale (e.g. relative AC/DC on
   detrended amplitude) so the same function behaves identically on file data and MAX30102
   counts. Then re-run gating and apply it to all splits or none.
5. **Retrain without caps** once 2-4 are settled: all available subjects, no
   `--max-train-windows`, and an early-stopping criterion whose validation set matches the
   one used for threshold calibration.
6. **Re-do the Grad-CAM plausibility study only after a model clears chance**, on in-domain
   data, with a significance test. Keep the gradient-parity test as-is.
7. **Correct the previous review's summary** rather than deleting it — its tables are
   accurate and its CIs were what made this audit possible. The framing is what needs
   amending, plus the "724k windows", "Shannon-entropy", and "ESP32-S3" errors.

## Reproduction

Per-subject purity in the raw archives, for `validate.npz` (repeat for `test.npz`):

```python
import numpy as np
d = np.load('data/deepbeat/validate.npz', allow_pickle=True)
s = np.array([str(x).strip() for x in d['parameters'][:, 2]])
r = np.argmax(d['rhythm'], axis=1)
for u in sorted(set(s), key=int):
    m = s == u
    print(u, m.sum(), round(100 * r[m].mean(), 2))
```

Training-distribution AUROC (`0.3487`) was measured by loading `cnn_af_v1.npz` into
`model.inference_model.Arrhythmia1DCNN` and scoring a seed-0 20,000-window sample of
`data/processed/deepbeat_train.npz`.

Caveat on this audit's own limits: `scipy` is absent from the current Windows interpreter, so
`torch`-based parity tests and `sqi.py` could not be re-executed here. The parity and
gradient-parity results are taken from the prior run's logs and were **not** independently
re-verified; the SQI conclusions are read from the code plus the `{0.0, 0.3, 0.6}` value
distribution and the manifest's empty `SQI >= 0.70` column, which are decisive on their own.
