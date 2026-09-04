---
title: SQUWA - Signal Quality Aware DNN Architecture for Enhanced Accuracy in Atrial Fibrillation Detection from Noisy PPG Signals
authors: "Yan, R. et al."
year: 2024
source: "arXiv"
tags: [literature]
status: to-read
zotero_id: http://zotero.org/users/19622725/items/TKR7NUAH
---

# SQUWA - Signal Quality Aware DNN Architecture for Enhanced Accuracy in Atrial Fibrillation Detection from Noisy PPG Signals

**Citation:** Yan, R. et al. (2024). SQUWA: Signal Quality Aware DNN Architecture for Enhanced Accuracy in Atrial Fibrillation Detection from Noisy PPG Signals. *arXiv*. https://doi.org/10.48550/arXiv.2404.15353

## Summary

*(fill in after reading)*

## Key findings

-

## Method / dataset (if relevant)

-

## Relevance to this thesis

*(how does this connect to PPG signal inputs, IoT architecture, ML classification, Grad-CAM interpretability, system outputs, or ISO/IEC 25010 evaluation?)*

## Abstract (from Zotero)

> Atrial fibrillation (AF), a common cardiac arrhythmia, significantly increases the risk of stroke, heart disease, and mortality. Photoplethysmography (PPG) offers a promising solution for continuous AF monitoring, due to its cost efficiency and integration into wearable devices. Nonetheless, PPG signals are susceptible to corruption from motion artifacts and other factors often encountered in ambulatory settings. Conventional approaches typically discard corrupted segments or attempt to reconstruct original signals, allowing for the use of standard machine learning techniques. However, this reduces dataset size and introduces biases, compromising prediction accuracy and the effectiveness of continuous monitoring. We propose a novel deep learning model, Signal Quality Weighted Fusion of Attentional Convolution and Recurrent Neural Network (SQUWA), designed to learn how to retain accurate predictions from partially corrupted PPG. Specifically, SQUWA innovatively integrates an attention mechanism that directly considers signal quality during the learning process, dynamically adjusting the weights of time series segments based on their quality. This approach enhances the influence of higher-quality segments while reducing that of lower-quality ones, effectively utilizing partially corrupted segments. This approach represents a departure from the conventional methods that exclude such segments, enabling the utilization of a broader range of data, which has great implications for less disruption when monitoring of AF risks and more accurate estimation of AF burdens. Our extensive experiments show that SQUWA outperform existing PPG-based models, achieving the highest AUCPR of 0.89 with label noise mitigation. This also exceeds the 0.86 AUCPR of models trained with using both electrocardiogram (ECG) and PPG data.


## Notable quotes / figures

>

## Related notes

- [[../Index|Literature Index]]
-
