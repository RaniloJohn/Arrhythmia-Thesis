---
title: Detecting Beats in the Photoplethysmogram — Benchmarking Open-Source Algorithms
authors: "Charlton, P. H., Kotzen, K., Mejía-Mejía, E., Aston, P. J., Budidha, K., Mant, J., Pettit, C., Behar, J. A., Kyriacou, P. A."
year: 2022
source: "Physiological Measurement, 43(8), 085007"
tags: [literature, dataset, critical, needs-zotero]
status: to-read
---

# Detecting Beats in the Photoplethysmogram — Benchmarking Open-Source Algorithms

**Citation:** Charlton, P. H., Kotzen, K., Mejía-Mejía, E., Aston, P. J., Budidha, K.,
Mant, J., Pettit, C., Behar, J. A., & Kyriacou, P. A. (2022). Detecting beats in the
photoplethysmogram: benchmarking open-source algorithms. *Physiological Measurement*,
43(8), 085007. https://doi.org/10.1088/1361-6579/ac826d

## ⚠️ Action needed

**This is a different paper from the existing
[[Charlton-2022-Wearable Photoplethysmography for|Charlton-2022-Wearable Photoplethysmography for]]
note** ("Wearable Photoplethysmography for Cardiovascular Monitoring," *Proc. IEEE*,
DOI 10.1109/JPROC.2022.3149785) — same first author, same year, different venue,
different content. **This is the paper that actually introduces the MIMIC PERform AF
Dataset** used throughout Chapter 2/3 and `03 - ML/`. Add it to Zotero and check
whichever Chapter 2 citation currently points to the *Proc. IEEE* review paper for the
dataset claim — it should point here instead.

## Summary

*(fill in after reading the full paper)* — found and verified via the accompanying
`ppg-beats` open-source toolkit/documentation (Charlton's own GitHub project) rather
than the full PDF. The paper benchmarks open-source PPG beat-detection algorithms
across several datasets, and its accompanying data/code release is what packages the
**MIMIC PERform AF Dataset**: ECG + PPG recordings from 35 critically-ill adults in the
MIMIC-III Waveform Database (19 subjects during atrial fibrillation, 16 during normal
sinus rhythm), 20-minute recordings each, hosted on PhysioNet/Zenodo via the
`peterhcharlton/ppg-beats` project.

## Key findings

- Most open-source PPG beat detectors perform well on resting hospital data but
  degrade substantially during movement, stress, **and atrial fibrillation** — directly
  relevant to why this thesis's SQI (Signal Quality Index) stage exists before
  classification.

## Method / dataset (if relevant)

- **MIMIC PERform AF Dataset**: 35 subjects total (19 AF / 16 non-AF), 20-min
  synchronized ECG+PPG per subject, sourced from the MIMIC-III Waveform Database
  Matched Subset. This is a **small dataset for training a deep CNN from scratch** —
  worth flagging in Chapter 3/4 methodology and cross-referencing against
  [[Talukdar-n.d.-Evaluation of Atrial Fibrillation|Talukdar (n.d.)]], which reports
  training set sizes (2,134 MIMIC-III samples) and accuracy benchmarks (97% with Random
  Forest) for a comparable MIMIC-derived PPG/AF task — useful precedent for what
  training-set size and expected accuracy range to plan for.

## Relevance to this thesis

This is the **actual source citation** for the dataset our `03 - ML/model/` pipeline is
built to use. Directly relevant to
[[../02 - Code Review/2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights|the open code review finding]]
that the 1D-CNN has never actually been trained on this dataset — this note should be
the citation anchor once real training is implemented, and its small-N caveat should
inform the training methodology write-up (e.g. whether augmentation, cross-validation,
or a supplementary dataset is needed given only 35 subjects).

## Notable quotes / figures

>

## Related notes

- [[../Index|Literature Index]]
- [[Charlton-2022-Wearable Photoplethysmography for|Charlton-2022-Wearable Photoplethysmography for]] (the other Charlton 2022 paper — don't conflate the two)
- [[Talukdar-n.d.-Evaluation of Atrial Fibrillation|Talukdar-n.d.-Evaluation of Atrial Fibrillation]]
- [[../02 - Code Review/2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights|2026-09-03 - 1D-CNN Inference Model Uses Untrained Random Weights]]
