---
title: The Training Dataset Explained, and What Is Wrong With It
tags: [dataset, deepbeat, mimic, training, ml, onboarding, honest-bounds]
date: 2026-09-25
status: current — written for a newcomer to the project, and as a defect list
reviewer: Claude (Research & Theoretical Architect)
---

# The Training Dataset Explained, and What Is Wrong With It

Written for someone picking the project up with no background. The first half explains
what the data is and how the model was trained, in plain language. The second half is a
defect list — several of these were found by reading
`03 - ML/model/weights/cnn_af_v1.meta.json` and the training history, and are **not**
stated in the existing results note.

---

## 0. Two different things get called "the database"

They are unrelated, and confusing them wastes time.

| | **The dataset** | **The live database** |
|---|---|---|
| What | Recordings of other people's pulses, used to teach the model | `arrhythmia_edge.db`, a SQLite file |
| Where | `03 - ML/data/` (not in git — too large) | `03 - ML/storage/` on the Pi |
| Purpose | Teaching material, used once, offline | Patient records and AF events, written live |
| Analogy | The textbook a student learns from | The logbook the working clinic fills in |

This note is about **the dataset**.

---

## 1. What the data actually is

The sensor on the device (a MAX30102) shines light into the skin and measures how much
bounces back. Blood absorbs light, so with each heartbeat the reading dips and recovers.
Plot that over time and you get a wave — one bump per beat. That is a **PPG signal**.
It is the same thing a fingertip pulse clip at a clinic measures.

**Atrial fibrillation (AF)** is a heart rhythm problem where beats arrive at irregular
intervals. On a PPG trace the bumps become unevenly spaced. The model's entire job is to
look at ten seconds of that wave and answer one question: *were these beats irregular in
the way AF is irregular?*

To learn that, it needs many examples already labelled by a doctor as AF or not-AF. The
project uses two collections, for two **different jobs**.

### DeepBeat — the teaching set

- Recorded from **smartwatches on people's wrists** — same place on the body and same
  method (light bounced back off the skin) as our device, but **a different colour of
  light**: those watches use green (525 nm), our MAX30102 uses red and infrared
  (660 / 940 nm). Close, not identical. See §4.9.
- Stanford, published 2020 (Torres-Soto & Ashley). Free, but requires registration.
- **30 people**, about 536,000 ten-second clips, roughly 3,725 hours.
- Each individual clip carries its own label, which is what makes it teachable.

### MIMIC PERform AF — the exam set

- Recorded from **fingertip clips on hospital ICU patients**.
- 35 people (19 with AF, 16 without), 20 minutes each.
- Deliberately **never** used for teaching — only to test afterward.

### Why two sets, and why it matters

If you teach and test on the same material, you are measuring memorisation, not learning.
Testing on a completely separate group — different hospital, different equipment, different
patients — is the honest test. The project got this structure right. The result of that
honest test, however, was bad; see §4.

---

## 2. How the training worked, step by step

### Step 1 — Make everything the same shape

The sources disagree with our device, so everything is converted to one format:

| | Recorded at | Clip length |
|---|---|---|
| DeepBeat | 32 readings/second | 25 seconds |
| MIMIC PERform | 125 readings/second | 20 minutes |
| **Our device** | **100 readings/second** | **10 seconds** |

Everything is converted to **100 readings/second, 10-second clips (1,000 numbers each)**.
The device's format was chosen as the target because changing the device would mean
re-verifying the firmware, the filters and the beat detector — far riskier than converting
files on a desktop.

Honest caveat: stretching 32/second up to 100/second **adds no new detail**. It is like
enlarging a small photo — bigger, not sharper. This is acceptable here because the useful
part of a pulse signal sits well within what 32/second already captures, but it belongs in
the thesis rather than being left for an examiner to notice.

### Step 2 — Clean each clip

The same three steps the live device performs, deliberately importing the *same code* so
training and real use cannot drift apart:

1. **Detrend** — remove slow drift from breathing or the sensor shifting.
2. **Bandpass filter (0.5–5 Hz)** — discard anything too slow or too fast to be a heartbeat.
3. **Normalise** — rescale so a strong and a weak signal look comparable.

### Step 3 — Throw away unusable clips

A quality score rejects clips where the sensor lost contact or the person moved.
**43.5% of training clips were discarded.** That is normal for wearable data and better
than training on noise.

### Step 4 — Split by person, never by clip

The 30 people were divided: **12 to teach, 4 to check progress, 14 to test**.

That is the *plan*. The run that produced the shipped weights only actually used **4 of
those 12** — see §4.1. The split itself is correct; the training run did not use all of it.

The critical rule is splitting by *person*. If clips from one person landed on both sides,
the model could recognise that individual rather than the condition — like memorising that
one patient's heartbeat instead of learning what AF looks like. This would inflate the
score and mean nothing. The project enforced this correctly, and the code refuses to run
if the groups overlap.

### Step 5 — Train

The model sees a clip, guesses AF or not, is told the right answer, and nudges its internal
settings. Repeat across thousands of clips. One pass over everything is an **epoch**.

Training used PyTorch on a desktop; the Pi runs the finished model in plain NumPy so no
heavy software ships to the device. A **parity test** confirms both produce identical
answers (difference < 0.00000006), which guards against the classic failure where weights
are loaded slightly wrong and the system confidently produces nonsense.

### Step 6 — Pick the cut-off

The model outputs a number from 0 to 1. Something must decide where "AF" begins. Chosen on
the check group only — never the test group — which is correct: tuning on the exam is
cheating. The chosen value was **0.37**.

---

## 3. The result, stated plainly

| Test | Sensitivity (catches real AF) | Specificity (leaves healthy alone) |
|---|---|---|
| DeepBeat, per 10-s clip | 85.4% | **10.9%** |
| MIMIC PERform (external) | 21.6% | 71.6% |
| DeepBeat, per person (votes pooled) | **100%** | **75%** |

**10.9% specificity means the model calls almost everything AF.** A smoke alarm that
sounds constantly does catch every fire, and is useless.

The one genuinely good number is the last row: pooling many clips per person and taking a
majority vote gives 100% sensitivity and 75% specificity across 14 patients (12 of 14
correct). The device already does this — a 3-of-5 rolling vote. That is the defensible
result, and it must always be reported as *subject-level*, not per-clip.

---

## 4. Problems that need fixing

Ordered by how much they threaten the thesis.

### 4.1 The model was trained on 4 people, not 12 — and stopped learning after one pass

**This is the most serious finding and it is not mentioned in the existing results note.**
From `cnn_af_v1.meta.json`:

```json
"subjects_used": ["147", "150", "152", "153"],
"window_count": 21938,
"best_epoch": 1
```

The prepared training set holds **451,791 clips from 12 people**. The run that produced the
shipped weights used **21,938 clips from 4 people** — under 5% of the data, a third of the
people. A `--max-train-windows` cap intended for quick test runs appears to have been left
on for the real one.

The training history shows what followed:

| Epoch | Training loss ↓ better | Val AUROC ↑ better |
|---|---|---|
| 1 | 0.553 | **0.635** |
| 2 | 0.531 | 0.610 |
| 3 | 0.515 | 0.579 |
| 6 | 0.454 | 0.567 |

The model improved on material it had seen while getting **worse** on material it had not —
classic overfitting, from the very first pass. It memorised four people. The saved model is
the epoch-1 one, i.e. barely trained at all.

**Fix:** re-run without the cap, on all 12 subjects. This is a single command and is by far
the highest-value action available. Everything downstream — the threshold, all metrics, the
Grad-CAM evidence — is derived from these weights and should be regenerated afterward.

### 4.2 The probabilities are not trustworthy as probabilities

The metadata says so outright:

```json
"calibration_assessment": "POORLY_CALIBRATED",
"expected_calibration_error": 0.424,
"brier_skill_score": -3.9523
```

A negative skill score means the confidence numbers are **worse than always guessing the
average rate**. So when the dashboard shows "73% AF", that 73% does not mean 73% — it is a
ranking, not a probability. The dashboard should not present it as a percentage likelihood
until this is fixed (Platt scaling or isotonic regression, already flagged in the metadata).

### 4.3 Cross-validation scores below chance

Repeating the experiment five times with different people gave **AUROC 0.330 ± 0.118**.
0.5 is coin-flipping; below 0.5 is worse than chance.

**This number cannot be used as evidence about the dataset**, because the
cross-validation was crippled in the same way as the main run. `cross_validate.py`
defaults to `max_train_windows_per_fold = 20000` and `epochs = 3`, and the recorded fold
sizes (21,291 / 16,242 / 22,124 / 24,091 / 22,124) confirm the cap was active. So every
fold trained on roughly 5% of the available data for 3 passes.

The consequence is important for how the thesis is written: **we do not currently know
how much of the poor performance belongs to the model, and how much to the crippled
training run.** No claim in either direction is supportable until the experiment is
repeated uncapped. Re-run both `train.py` and `cross_validate.py` with the caps removed
before drawing any conclusion about whether the architecture or the corpus is at fault.

### 4.4 The subject count in the project's own notes is wrong

`MEMORY.md` (settled decision 9) justifies choosing DeepBeat partly on "~500k windows from
**~175 subjects**". The verified figure from the split manifest is **30 subjects** (12 + 4
+ 14). The window count is right; the person count is off by roughly 6×. Since subject
count is what actually governs generalisation, this weakens the stated rationale and an
examiner checking the manifest would find it. Correct the note.

### 4.5 The largest part of DeepBeat is on disk but was never processed

**Corrected 2026-09-25.** An earlier version of this section said `train.npz` had not been
downloaded, on the strength of the directory listing in `03 - ML/data/README.md`, which
omits it. That was wrong — the file is on the Windows workstation, it is simply gitignored:

```
03 - ML/data/deepbeat/train.npz      18,109,812,108 bytes
  signal.npy      (2803934, 800, 1)  float64
  rhythm.npy      (2803934, 2)       float64
  qa_label.npy    (2803934, 3)       float32
  parameters.npy  (2803934, 3)       object
```

**2,803,934 parent windows** — roughly **5× the 536,399** that were actually processed.
`build_dataset.py` was run over `validate.npz` + `test.npz` only, so the official training
partition, and whatever additional subjects it contains, has never entered the pipeline.

This is a bigger finding than the original one: the corpus is not missing, it is unused.
The `README.md` directory listing should be corrected so nobody else concludes it is absent.

### 4.6 Severe class imbalance

DeepBeat is **9.7% AF / 90.3% non-AF**; the validation split is worse at 4.94% AF. A model
can score 95% accuracy by answering "no AF" every time. This is why accuracy is the wrong
headline number here and why PR-AUC and the confusion matrix must be reported instead. The
cut-off of 0.37 was chosen to maximise catching AF, which is why specificity collapsed —
on the validation split that meant 217,650 false alarms against 12,044 correct catches.

### 4.7 The exam set is labelled per recording, not per clip

MIMIC PERform labels a whole 20-minute recording "AF". But AF often comes in bursts, so
many 10-second clips inside an "AF" recording are actually normal rhythm — and get counted
as mistakes when the model correctly calls them normal. Some of the poor external score is
this measurement artefact rather than model failure. Worth stating; it does not rescue the
result, since AUROC 0.494 is near chance regardless.

### 4.8 Wrist-trained, and the device may end up fingertip

The external collapse is real physics, not a bug. Wrist sensors use green light bouncing off
shallow vessels; fingertip clips shine red/infrared straight through tissue. The waveforms
genuinely differ. The model learned the wrist version.

This is unresolved at the hardware level: the enclosure is wrist-worn, but the firmware's
LED drive (`0x1F`, ~6.4 mA) is a fingertip setting, and the system to date was validated on
a finger. **If the device ends up fingertip, the training corpus is wrong for it.** The
signal test in
[[2026-09-11 - Wrist vs Fingertip Sensing - SOP and RRL Alignment|the wrist-vs-fingertip note]]
settles this with a measurement and should be run before any more training effort.

---

### 4.9 "Same sensor as ours" is an overstatement

The reason for choosing DeepBeat is that it matches our device. It does match on the two
things that matter most — **same place on the body** (wrist) and **same method** (light
bounced back off the skin, rather than shone through a finger).

It does not match on **colour of light**:

| | DeepBeat watches | Our MAX30102 |
|---|---|---|
| Light | Green, 525 nm | Red 660 nm + infrared 940 nm |
| Reaches | Shallow surface vessels | Deeper tissue |

Different colours reach different depths, so the pulse wave they draw is not quite the
same shape. The match is real but weaker than "same as our device" suggests. Write it as
*matched on site and modality, with a wavelength difference* — an examiner who knows PPG
will otherwise find the gap first.

## 5. What to do, in order

1. **Re-run training without the window cap**, on all 12 subjects (§4.1).
2. **Download `train.npz`** to recover the rest of the corpus (§4.5).
3. **Recalibrate the threshold and re-measure everything** from the new weights (§4.2).
4. **Run the wrist-vs-fingertip signal test** before committing further (§4.8).
5. **Correct the 175-subject figure** in `MEMORY.md` (§4.4).
6. In the thesis, **lead with the subject-level numbers**, and always print specificity
   next to sensitivity.

Nothing here is a crisis. The pipeline is sound, the honest-evaluation structure is right,
and the parity and subject-splitting guards are the hard parts to get correct — those are
done. The weakness is that the model was accidentally trained on a fraction of the data.

---

## Related

- [[../02 - Code Review/2026-09-13 - 1D-CNN Training Results & Honest Performance Bounds|Training results & honest performance bounds]]
- [[2026-09-10 - Model Training Plan and Project Roadmap|Model training plan & roadmap]]
- [[2026-09-11 - Wrist vs Fingertip Sensing - SOP and RRL Alignment|Wrist vs fingertip sensing]]
- `03 - ML/data/README.md` — dataset provenance and download procedure
- `03 - ML/model/weights/cnn_af_v1.meta.json` — the training run's own record

---
Back to [[Index|Claude Notes Index]]
