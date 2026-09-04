---
name: Thesis Overview
description: Working summary of Chapters 1-3 of the thesis, for quick context without reopening the PDF.
tags: [thesis, reference]
source: "BAUZON_CONDINO_DELOSANGELES_RAÑA_VILLAFLOR-CHAPTER1-3 REVISED.pdf"
---

# Thesis Overview

**Title:** An Internet of Things-Based Framework for Cardiac Arrhythmia Detection via
Photoplethysmography and Machine Learning
**Prepared for:** Dr. Nelson Rodelas, CpE Department, University of the East
**Prepared by:** Bauzon, John Carlo · Condino, Zidane Vincent · Delos Angeles, Ranilo
John · Rana, Adrian · Villaflor, Kyle — 3CPE-2A

Full text: [[BAUZON_CONDINO_DELOSANGELES_RAÑA_VILLAFLOR-CHAPTER1-3 REVISED.pdf]]

## Background (Chapter 1)

Cardiovascular disease remains the leading cause of death worldwide (~19.2 million
deaths in 2023), with atrial fibrillation (AF) a major and growing contributor to that
burden. In the Philippines specifically, AF prevalence in adults is reported at ~0.2%
(higher in older age groups) but is expected to rise with an aging population and
increasing hypertension/diabetes comorbidity. Diagnosis is hampered by low awareness,
absence of routine ECGs, limited cardiology access (especially in rural areas), reliance
on paper health records, and the high cost/internet-dependence of most existing
commercial wearables — leaving many arrhythmias silent and underdiagnosed until a
stroke or other serious complication occurs.

The study's premise: PPG-based wearables combined with edge-deployed machine learning
can close this gap affordably, without requiring continuous internet connectivity —
but no existing study had combined PPG-based AF detection, edge-level deep learning
classification, *and* blockchain-secured data management into one low-cost,
offline-capable system designed for the Philippine primary-care setting.

## Statement of the problem

No existing unified framework simultaneously integrates PPG-based AF detection,
edge-level deep learning classification, and blockchain-secured data management in a
low-cost, offline-capable IoT system for Philippine primary care. Existing IoT+PPG
systems (Kumar et al. 2021; Pedrosa-Rodriguez et al. 2024) depend on continuous cloud
connectivity; blockchain-secured health IoT (Egala et al. 2021; Sumathi et al. 2022)
remains largely theoretical at the edge; and local AF care is hampered by absent routine
ECG screening and paper-based records (Fadreguilan, 2023).

### Specific problems (research questions)

1. What PPG signal inputs and physiological parameters are required for reliable AF
   detection via a PPG-beat detection framework?
2. How can an IoT-enabled device be designed and implemented to acquire PPG signals and
   process them through ML for AF detection?
3. How can an interpretable ML approach classify AF from PPG-beat features while
   ensuring real-time, edge-level processing? Specifically, how can Grad-CAM highlight
   the PPG segments most influential to the classification?
4. What system outputs/functionality support early AF detection, user notification, and
   clinical decision support in non-hospital settings?
5. What is the system's performance and quality under the ISO/IEC 25010 model, across
   Functional Suitability, Performance Efficiency, Reliability, Security, Usability, and
   Maintainability/Portability?

## Significance

Framed as relevant to: the DOH (national CVD surveillance/early detection), local
government units and primary health centers (affordable outreach screening), healthcare
professionals (ML-assisted rhythm interpretation), patients/at-risk individuals (early,
continuous, low-effort monitoring), researchers/academic institutions, and future
developers/system designers (a modular, ISO/IEC-25010-evaluated reference
architecture). Positioned as supporting SDG 3 (target 3.4, reducing NCD mortality) and
Philippine Universal Health Coverage.

## Scope

- Design, development, and evaluation of an IoT-enabled prototype for AF detection from
  PPG signals via ML.
- PPG acquisition via affordable wearable sensors; preprocessing and beat detection to
  produce beat-to-beat intervals and heart-rate features.
- An IoT architecture for real-time acquisition, local processing, and transmission from
  the wearable to a monitoring interface.
- An interpretable supervised ML model for AF classification from PPG-beat features.
- System outputs: real-time alerts, heart-rhythm trend visualizations, decision-support
  indicators.

## Limitations

- Detects AF only — not atrial flutter, ventricular arrhythmias, or bradyarrhythmia.
- PPG accuracy can be affected by motion artifacts, skin tone variability, sensor
  placement.
- Not a replacement for clinical-grade diagnostics (e.g. ECG) — screening/decision
  support only, not a definitive diagnostic instrument.
- Pilot deployment is within Caloocan City (urban, Metro Manila); findings may not
  generalize to rural provincial settings with different infrastructure, demographics,
  and health literacy.

## Related literature synthesis (Chapter 2, condensed)

The RRL (~150 sources) supports each research question:

- **PPG signal inputs for AF (RQ1):** beat-to-beat interval, inter-beat interval,
  interval irregularity, waveform morphology, and signal-quality indices are the
  established physiological markers (Millán et al. 2020; Pereira et al. 2020; Bashar et
  al. 2019); preprocessing (filtering, motion-artifact removal, signal-quality
  assessment) is a prerequisite for reliable extraction.
- **IoT + ML for AF from PPG (RQ2):** IoT devices enable continuous real-time PPG
  collection (Pedrosa-Rodriguez et al. 2024); commercial wrist-worn PPG devices are
  already a viable low-cost, non-invasive channel (Aliamiri & Shen; Cinotti et al.
  2024).
- **Interpretable, edge-optimized classification (RQ3):** 1D-CNNs and lightweight
  attention-based models classify AF from beat-level PPG within wearable/edge
  constraints (Santala et al. 2022; Zhou et al. 2022; Zheng et al. 2023; Sideshwar et
  al. 2021). Grad-CAM (originally for 2D images) generalizes well to 1D physiological
  time series (Jahmunah et al. 2022) and localizes clinically meaningful regions (absent
  P-wave, T-wave abnormality, irregular R-R intervals — Luo et al. 2025), building
  clinical trust in an otherwise "black-box" model.
- **System outputs / decision support (RQ4):** the field has moved from simple binary
  notifications (Apple Heart Study; Turakhia et al. 2019) toward structured,
  clinician-facing outputs — rhythm summaries, episode data, telehealth integration
  (TeleCheck-AF, Pluymaekers et al. 2021; FibriCheck, Proesmans et al. 2019), and
  evidence that structured outputs change clinical management (mSToPS, Steinhubl et al.
  2018).
- **ISO/IEC 25010 evaluation (RQ5):** an established framework for embedded/IoT health
  software quality (Argotti et al. 2024; Souza-Pereira et al. 2021); benchmark figures
  cited in the literature include ~87.8–91%+ AF detection sensitivity/specificity (Zhu
  et al. 2022; Tran et al. 2023) and edge-classification latency in the low tens of
  milliseconds (Cinotti et al. 2024/2025).

**Identified gap driving this study:** most prior work optimizes these pieces (signal
preprocessing, feature extraction, interpretability, edge deployment) independently
rather than as one integrated pipeline — motivating the four-stage unified framework
below.

## Conceptual framework / system architecture (Chapter 2, Fig. 2.1)

1. **Acquisition:** MAX30102 PPG sensor (red 660 nm / IR 940 nm) captures raw PPG via
   I2C to an ESP32-C3 microcontroller, which does preliminary signal conditioning.
2. **Preprocessing (Raspberry Pi):** bandpass filtering, motion-artifact removal, peak
   detection → inter-beat intervals (IBI) → BPM.
3. **Classification (Raspberry Pi):** the segmented PPG waveform (not hand-engineered
   HRV features) is fed directly into a pre-trained 1D-CNN (trained offline on the
   MIMIC PERform AF Dataset, Charlton et al. 2022) → binary output, AF vs. non-AF.
4. **Decision node:** no AF → continue monitoring/visualization only. AF detected →
   trigger the secure data layer: classification + metadata (timestamp, device ID,
   features) → SHA-256 hash → stored locally in SQLite (offline-capable) → synced to a
   Hyperledger Fabric blockchain network once internet is available (immutability,
   traceability, tamper resistance).
5. **Visualization:** real-time heart rate, PPG waveform, and AF status shown via a UI
   (web dashboard, clinician login) simultaneously with logging.

### Unified ML framework (four pipelines, Chapter 2 §3.3)

1. Artifact-aware preprocessing at the edge.
2. Hybrid feature extraction (HFE): statistical IBI descriptors + morphological
   waveform features.
3. Attention-guided, interpretable classification via a lightweight 1D-CNN optimized
   for edge deployment.
4. Multiview output module: confidence-weighted alerts, rhythm-irregularity summaries,
   clinician-facing decision-support indicators.

## Hardware

- MAX30102 PPG sensor (I2C)
- ESP32-C3 microcontroller (C++, Type-C USB, I2C/SPI/UART)
- SSD1306 0.96" OLED display (128×64, I2C) — real-time BPM/AF alert
- Raspberry Pi 4 Model B (4/8GB RAM) — signal processing, 1D-CNN inference, storage
- 3.7V Li-Po battery, 500–1000 mAh

## Software / methodology

- **SDLC:** Agile Scrum (iterative sprints: plan → develop → test → review →
  retrospective), chosen for the system's modular structure (acquisition, processing,
  edge integration, UI).
- **Languages/stack:** Python (ML + signal processing engine, on-device backend), C++
  (ESP32 firmware), HTML5/CSS3 + Node.js (dashboard), SQLite (local DB), Hyperledger
  Fabric (permissioned blockchain).
- **Model:** 1D-CNN — Conv1D(32, k=5) + ReLU + MaxPool → Conv1D(64, k=3) + ReLU +
  MaxPool → Flatten → Dense(64, ReLU, dropout 0.5) → Dense(1, Sigmoid); Adam optimizer,
  binary cross-entropy loss; 80/20 train-validation split with early stopping; exported
  as a frozen inference file for on-device use (reported <25ms per classification).
- **Dataset:** MIMIC PERform AF Dataset (Charlton et al., 2022) — synchronized,
  annotated physiological signals (incl. PPG) for AF detection benchmarking.

## Research methodology (Chapter 3)

- **Locale:** Caloocan City, Metro Manila — chosen for population density/diversity and
  documented gaps in barangay-level healthcare access.
- **Design:** Descriptive (system profile vs. ISO/IEC 25010) + Experimental (test-case
  validation of signal classification, hardware integration, ML performance, database
  logging, blockchain logging, usability/interface — see Tables 1.8–1.14 in the source
  PDF for the full test-case matrices).
- **Sampling:** Purposive — respondents with cardiac arrhythmia history and/or
  experience using PPG-capable wearable devices.
- **Instrument:** Concept Validation Survey Questionnaire, five-point Likert scale
  (Strongly Disagree–Strongly Agree), covering Expected Ease of Use, Innovation,
  Perceived Usefulness, and System Feasibility.
- **Statistical treatment:** frequency/percentage distribution, weighted mean
  (interpreted via a 1.00–5.00 five-band scale), standard deviation, and Cronbach's
  Alpha for instrument reliability (≥0.70 considered acceptable).

## Open items to track

- [x] Source code for the pipeline is implemented and deployed live on the Pi — see
      [[../03 - ML/README|03 - ML/README]] and [[../07 - Website/README|07 - Website/README]].
- [ ] **The 1D-CNN is architecturally correct but untrained (random weights, never
      fit to the MIMIC PERform AF Dataset)** — AF classification/Grad-CAM output is
      not yet an evidence-based result. See
      [[../02 - Code Review/2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights|code review note]]
      before citing any sensitivity/specificity figures as achieved.
- [ ] Hyperledger Fabric blockchain sync is not yet implemented (local SQLite
      hash-chaining is; the sync worker is explicitly deferred — see `PLAN.md`).
- [ ] Populate [[../01 - Literature/Index|Literature Index]] with individual notes on
      the ~150 cited sources as they're (re)read in depth.
- [x] Review sessions are being logged in [[../02 - Code Review/Index|Code Review Index]].

---
Back to [[../Home|Home]]
