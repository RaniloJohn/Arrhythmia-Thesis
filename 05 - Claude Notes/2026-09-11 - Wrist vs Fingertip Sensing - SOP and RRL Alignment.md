# Wrist vs Fingertip Sensing — Alignment with the SOP and the RRL

**Date:** 2026-09-11
**Question:** Would switching from wrist (radial artery) to fingertip PPG make the system
better, and would it still answer the Statement of the Problem and the Review of Related
Literature?

**Short answer — revised 2026-09-11 after the team pushed back:** **Switch to fingertip.**

The first version of this note recommended staying with the wrist, weighted heavily on
"continuous monitoring". The team correctly pointed out that **this device is not a continuous
wearable at all** — it is USB-C tethered to a Raspberry Pi. Nobody wears it overnight. It is a
**screening instrument** that happens to strap on, used for a seated reading of a minute or
two in a primary-care setting.

Once that is accepted, the argument inverts. Continuity was never on the table, so it cannot
be a cost of switching. And the finger-PPG corpus that matches a seated screening reading is
**MIMIC PERform AF — the dataset Chapter 2 already names.**

---

## Finger PPG datasets with AF labels

| Dataset | Site | Size | ECG reference | Verdict |
|---|---|---|---|---|
| **MIMIC PERform AF** (Charlton 2022) | **Finger** (bedside pulse ox) | 35 subjects — 19 AF, 16 non-AF, 20 min each | **Yes** | **The one to use.** Already named in Chapter 2. |
| MIMIC III Waveform | Finger | ~125,629 PPG segments, 35 human-annotated individuals | Yes | The larger pool PERform is drawn from; needs own labelling work. |
| VitalDB | Finger | ~1,209 h, **unlabelled** | No | Useful for self-supervised pretraining only. |
| UMMC Simband | **Wrist** (smartwatch) | 37 subjects, 292 clean segments | Yes | Wrong site, and very small. |
| DeepBeat (Torres-Soto & Ashley 2020) | **Wrist** | Large | **No — reference ECG not released** | Wrong site for this decision, and unbenchmarkable. |

### Verified details — MIMIC PERform AF

Checked against the [Zenodo record](https://zenodo.org/records/6967256) and the
[ppg-beats documentation](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/):

| Property | Value |
|---|---|
| Subjects | 35 critically-ill adults — **19 AF, 16 non-AF** |
| Duration | 20 minutes each |
| Sampling rate | **125 Hz** |
| Signals | PPG, ECG, respiration |
| Source | MIMIC-III Waveform Database Matched Subset, bedside monitor |
| AF labels | Manual, from a figshare repository, ECG-adjudicated |
| Formats | MATLAB `.mat`, WFDB, CSV |
| Licence | ODC Open Database License v1.0 |
| Size | **1.6 GB** (versus DeepBeat's 18 GB) |

### Two honest caveats about the "finger" claim

**The documentation never states the measurement site.** It says only "measured using a bedside
monitor". Bedside pulse oximetry in an ICU is overwhelmingly a fingertip probe — occasionally
ear or forehead where perfusion is poor — so finger is a very strong inference, but it is an
**inference, not a documented fact**. Do not write "finger PPG" into Chapter 3 as though the
dataset asserts it. What can be stated safely is that it is **bedside pulse-oximetry PPG from a
peripheral site**, which is categorically not wrist reflectance.

**Modality differs even if the site matches.** A clinical pulse oximeter is **transmissive**
(emitter and detector on opposite sides of the finger). The MAX30102 is an integrated
**reflectance** sensor and cannot work transmissively. So the pairing is transmissive-finger
training data against reflectance-finger deployment. That is a real domain gap — but a far
smaller one than finger-versus-wrist, since both sit on high-perfusion fingertip tissue with
similar pulsatile morphology.

Also note the **125 Hz → 100 Hz resampling** requirement, the same class of problem Decision 2
in the roadmap already solved for DeepBeat's 32 Hz.

Switching the sensor to the fingertip therefore does not create a corpus problem — it
*resolves* one. Decision 1 in [[2026-09-10 - Model Training Plan and Project Roadmap]] chose
DeepBeat and noted that "Chapter 2 and `ANTIGRAVITY.md` §4.3 both currently name MIMIC PERform,
so the thesis must be updated." Going to the finger **erases that documentation debt** instead
of adding to it.

**DeepBeat does not publish its reference ECG**, which precludes benchmarking against the
published metrics. MIMIC PERform AF is ECG-paired, so its labels are verifiable and results
are comparable to other published work. For a thesis that must defend Functional Suitability
numbers, that is a significant advantage.

### The honest caveat

MIMIC PERform AF is **small (35 subjects) and drawn from critically-ill ICU adults**, not a
primary-care screening population. That is a genuine external-validity limitation and belongs
in Chapter 1's Limitations, stated plainly. It is, however, *at-rest supine bedside* data,
which is a far better match to a seated screening reading than DeepBeat's free-living
wrist-worn data is to a USB-tethered device on a desk.

## The Statement of the Problem does not constrain the acquisition site at all

This is the central finding, and it was obscured in the first draft of this note by mixing
three different chapters together.

Searching the SOP (`Thesis Overview.md`, "Specific problems (research questions)") for
*wrist*, *wearable*, *continuous* and *finger* returns **nothing**. All five research
questions are written in terms of PPG generally, an "IoT-enabled device", edge processing and
ISO/IEC 25010 — never an anatomical site.

Those words live elsewhere:

| Word | Where it actually appears |
|---|---|
| "wearable" | **Scope**, lines 75 and 78 |
| "continuous" | **Significance**, line 66, and the **RRL** synthesis, line 103 |

So the objections raised in the first version of this note were against **Scope** and
**Significance**, not against the Statement of the Problem. **Switching to fingertip requires
no change to the SOP whatsoever.** RQ1 and RQ5 are actively strengthened by it.

## Effect on each research question

| RQ | Effect of switching to fingertip |
|---|---|
| **RQ1** — what PPG inputs are required for reliable AF detection | **Improves.** Fingertip PPG has far higher perfusion and SNR. Beat-to-beat intervals and morphology are cleaner, so IBI extraction and SQI gating both get easier. |
| **RQ2** — how to design an IoT device to acquire PPG and run ML | **Neutral — RQ2 names no site.** The device is unchanged: same MAX30102, same ESP32-C3, same 19-byte wire protocol, same edge pipeline. The "wearable" wording is in Scope, not here. |
| **RQ3** — interpretable edge classification with Grad-CAM | **Neutral.** Grad-CAM over a 1D CNN is indifferent to acquisition site. |
| **RQ4** — outputs for early detection and decision support outside hospitals | **Neutral, once framed honestly.** AF is paroxysmal and a single reading can miss an episode — but that limitation applies to the tethered wrist build identically. The device is a screening instrument either way; the fix is to state that, not to change the sensor. |
| **RQ5** — ISO/IEC 25010 | **Improves.** Functional Suitability becomes measurable because corpus and site finally match, and it can be benchmarked against published work since MIMIC PERform AF is ECG-paired. Usability improves on a cleaner signal. |

## Against the RRL

The Chapter 2 synthesis leans wearable at exactly the points that matter:

- **RQ2's RRL** states that "IoT devices enable **continuous** real-time PPG collection"
  (Pedrosa-Rodriguez et al. 2024) and that "**commercial wrist-worn PPG devices** are already
  a viable low-cost, non-invasive channel" (Aliamiri & Shen; Cinotti et al. 2024). A fingertip
  device is not the thing those citations establish the viability of.
- **RQ3's RRL** is about classification "within **wearable**/edge constraints".
- **RQ4's RRL** is more mixed and is worth being honest about: the Apple Heart Study
  (Turakhia et al. 2019) and mSToPS (Steinhubl et al. 2018) are continuous/wearable, but
  **FibriCheck** (Proesmans et al. 2019) is fingertip PPG via a smartphone camera and is
  cited approvingly. So fingertip screening is not unsupported by the literature — it is
  simply not what the rest of this thesis's framing promises.

The Significance chapter claims value for "patients/at-risk individuals (early, **continuous**,
low-effort monitoring)". A fingertip probe does not deliver "continuous" — **but neither does
the tethered wrist build.** That word is already unsupported by the system as it exists, and
needs correcting whichever site is chosen.

**FibriCheck is the precedent to lean on.** It is fingertip PPG, it is cited approvingly in
RQ4's synthesis, and it is exactly this thesis's use case: a low-cost, non-continuous,
point-of-care AF screen outside a hospital. Citing it as the model for the acquisition
approach costs nothing and is already in the reference list.

## Where fingertip genuinely wins, and it is not a small thing

Wrist reflectance PPG over the radial artery is hard. The signal is an order of magnitude
weaker than at the fingertip, it is far more motion-sensitive, and it depends critically on
contact pressure. Two concrete signs the current build is already tuned for a finger:

1. `ArrhythmiaNode.ino:278-279` drives both LEDs at `0x1F` (~6.4 mA) — a **fingertip
   transmission** figure. Wrist reflectance through thicker tissue typically needs
   substantially more drive current.
2. `PLAN.md` records that the acquisition chain was brought up and "validated with a real
   **finger**".

So the working system that exists today is a finger system. The wrist enclosure is a bet that
the wrist signal can be made good enough.

## Revision 3 — "we need accuracy, so DeepBeat, so wrist"

The team raised the strongest counter-argument: MIMIC PERform AF's 35 subjects are too few to
train a 1D-CNN, so DeepBeat's volume wins, which forces the wrist. That is reasonable, but the
size advantage is smaller than the raw numbers suggest.

### What DeepBeat actually contains

From the [npj Digital Medicine paper](https://www.nature.com/articles/s41746-020-00320-4) and
[arXiv:2001.00155](https://arxiv.org/abs/2001.00155):

- Pretrained on **over 1 million *simulated*** physiological signals
- Fine-tuned on **~500,000 labelled signals from just over 100 individuals**, across 3 wearable
  devices, wrist (Simband), 128 Hz

### Two things that blunt the accuracy argument

**Segment count is not statistical power — subject count is.** 500K segments drawn from ~100
people is ~100 independent observations for generalisation purposes, not 500,000. Against MIMIC
PERform AF's 35, that is roughly **3× the subjects, not four orders of magnitude**. Segments
from the same wrist on the same day are heavily correlated.

**DeepBeat's labels are partly machine-generated and unverifiable.** A published quality
assessment ([arXiv:2307.08766](https://arxiv.org/pdf/2307.08766)) states that most DeepBeat
data "was labeled by a quality assessment model, which means that the labels are not entirely
reliable, and the metrics of this model's performance were not presented." Combined with the
unreleased reference ECG, this means a high score on DeepBeat's own test split partly measures
agreement with an unpublished labelling model rather than clinical ground truth.

MIMIC PERform AF's labels are manual and ECG-adjudicated, with the ECG shipped in the file.

**Conclusion: this is not "large and accurate" versus "small and accurate". It is "moderately
larger with weaker label provenance" versus "small with verifiable labels."** That does not
settle the question — DeepBeat is still the better training corpus on volume — but it removes
accuracy as a decisive argument, and it means the decision should turn on something else.

## The decision should be made by measurement, not argument

The binding risk in this whole thesis is not corpus choice. It is that **the MAX30102 may not
produce a usable signal at the wrist**, in which case the demo fails on defense day no matter
how well the model trained. That is an empirical question, it is answerable remotely this week,
and it should be settled before either path is committed to.

### Test protocol (runs entirely over Tailscale)

1. Raise the LED drive in firmware from `0x1F` (~6.4 mA) to roughly `0x3F`–`0x7F`
   (~12.6–25.4 mA) — `ArrhythmiaNode.ino:278-279`. Reflash remotely per the runbook.
2. Record **60 s at rest on a fingertip** — this is the known-good reference, since the chain
   is already validated there.
3. Record **60 s at rest on the volar wrist over the radial artery**, same session, same
   subject, sensor held with firm even pressure.
4. For each, compute **perfusion index (AC/DC)**, the existing **SQI**, and **IBI coefficient
   of variation** during sinus rhythm.

**Decision rule.** If wrist perfusion index and SQI come within roughly a third of the finger
baseline, and IBI CV is comparable, the wrist is viable → **wrist + DeepBeat**. If the wrist is
dramatically worse, no enclosure design will rescue it → **finger + MIMIC PERform AF**.

This costs one firmware change and ten minutes of a teammate's time, and it replaces an
argument nobody can win from documents with a number.

## Recommendation

**Default to wrist + DeepBeat, gated on the signal test above, with MIMIC PERform AF as
external validation.**

Reasons, in order of weight:

1. **Site-matched training and deployment is the single most defensible property** the thesis
   can have on RQ5. Wrist training data deployed on a wrist device needs no domain-gap excuse.
2. **MIMIC PERform AF then does its best work as a held-out external set**, which is exactly
   what Decision 1 in the roadmap originally proposed. Reporting unseen-domain performance on
   an ECG-adjudicated corpus directly answers the label-provenance criticism of DeepBeat — a
   stronger result than training on either alone.
3. **The wrist enclosure already exists**, is validated, and fits the real boards.
4. The 18 GB is already downloaded and Decision 1 is already documented. Reversing twice costs
   time the team does not have.

**Switch to finger only if the signal test fails.** In that case everything in the sections
below applies and the switch is clean, because the SOP does not constrain acquisition site.

### Non-negotiable regardless of which path wins

- **Raise the LED drive current.** `0x1F` is a fingertip figure and is almost certainly too low
  for wrist reflectance. This is required before any claim about wrist signal quality is made.
- **State DeepBeat's label provenance in Limitations** if training on it: labels largely
  produced by an unpublished quality-assessment model, reference ECG not released.
- **Fix the "continuous" claim in Significance.** The device is USB-tethered; it was never a
  continuous monitor on either path.

### If the fingertip path wins, the pieces line up as follows:

| | Wrist + DeepBeat | Finger + MIMIC PERform AF |
|---|---|---|
| Chapter 2 text | must be rewritten | **already correct** |
| Sensor site vs corpus | matched | **matched** |
| Reference ECG for benchmarking | **absent** | present |
| Signal quality | weak reflectance, pressure-critical | **strong, high perfusion** |
| Matches a seated screening reading | poorly (free-living data) | **well (at-rest bedside)** |
| Current firmware LED drive (6.4 mA) | too low, needs raising | **already correct** |
| System as validated to date | unproven at the wrist | **already validated on a finger** |

The hardware does not change: **the same MAX30102 module**, just held against a fingertip
instead of the radial artery. RQ2 is about designing an IoT device to acquire PPG and run ML
on it, and that is untouched — the sensor, the ESP32-C3, the 19-byte wire protocol, the Pi
pipeline, the dashboard and the Grad-CAM path are all identical.

### What has to change in the document

1. **Scope — "wearable".** Amend to describe a fingertip screening probe. This is a small
   edit, and a fingertip probe is still defensibly *worn*: clinical pulse oximeters are
   fingertip-worn devices. "Affordable wearable sensors" becomes something like "an affordable
   fingertip PPG probe for point-of-care screening".
2. **Significance — "continuous".** The claim of "early, **continuous**, low-effort
   monitoring" must be softened to **episodic / point-of-care screening**. Note this edit is
   required *regardless of sensor site*: the device is USB-tethered to a Raspberry Pi and was
   never capable of continuous wear. Fixing it is overdue, not a cost of this decision.
3. **Chapter 2 corpus.** No change needed — MIMIC PERform AF is what it already says. Revert
   Decision 1 in the roadmap note and record why.
4. **Limitations.** Add the MIMIC PERform AF external-validity caveat: 35 critically-ill ICU
   adults, not a primary-care screening cohort.

### What this costs

The 18 GB DeepBeat download is discarded. That is a sunk cost and not a reason to keep an
unbenchmarkable wrist corpus for a finger device. Nothing else is lost — the enclosure CAD is
parametric, so the fingertip housing reuses the sensor boss, light-seal ring and aperture logic
directly; only the outer form and the finger registration change.

### A cheap addition worth keeping

Retain the wrist enclosure as a **secondary comparison**, not the primary deliverable. Running
the same hardware at both sites gives an SQI / IBI-stability comparison that answers **RQ1**
with evidence rather than assertion, and directly addresses the stated Limitation that "PPG
accuracy can be affected by ... **sensor placement**". It costs one extra printed part.

---

**Related:** [[2026-09-11 - Wrist Enclosure Design Specification]] ·
[[2026-09-10 - Model Training Plan and Project Roadmap]] ·
[[../04 - Thesis Reference/Thesis Overview]]
