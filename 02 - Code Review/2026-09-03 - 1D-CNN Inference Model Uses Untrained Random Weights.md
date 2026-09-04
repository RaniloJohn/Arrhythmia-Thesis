---
date: 2026-09-03
component: 1D-CNN inference model (03 - ML/model/inference_model.py, grad_cam.py)
reviewer: Claude (Sonnet 5)
tags: [code-review, model, critical]
status: open
---

# 2026-09-03 — 1D-CNN Inference Model Uses Untrained Random Weights

## Scope

Reviewed `03 - ML/model/inference_model.py` and `03 - ML/model/grad_cam.py` while doing
an end-of-session verification pass across the whole Obsidian vault/codebase, prompted
by updating `CLAUDE.md`'s Tech Stack section (previously written as "confirm against
actual code once added" — now that code exists, checking the claim).

## Summary

The `Arrhythmia1DCNN` class correctly implements the architecture specified in Thesis
Chapter 2 / `ANTIGRAVITY.md` §4.3 (Conv1D(32,k5) → ReLU → MaxPool → Conv1D(64,k3) → ReLU
→ MaxPool → Dense(64, ReLU) → Dense(1, Sigmoid)), and the pipeline around it (DSP → SQI
→ inference → Grad-CAM → hash-chained storage → live streaming) is real and verified
working end to end. **But the model has never been trained on the MIMIC PERform AF
Dataset** — its weights are randomly initialized (He/Kaiming init, fixed RNG seeds) and
never fit to any labeled data. Right now, AF classification output is not
evidence-based; it reflects the random initialization plus one hand-set bias value.

## Findings

| # | Severity | Issue | File / Location | Suggested Fix | Status |
|---|----------|-------|------------------|----------------|--------|
| 1 | Critical | `Arrhythmia1DCNN.__init__` calls `_calibrate_weights()`, whose docstring says "Set realistic physiological sensitivity weights for AF pulse morphology detection" / "mimicking trained MIMIC-PERform weights," but the method body only sets `self.dense_out.bias[0] = -0.35` — every conv/dense weight matrix is untouched random He-init noise. No training loop, no saved/loaded weight file, and no reference to the MIMIC PERform dataset anywhere in the codebase. | `03 - ML/model/inference_model.py:89-116` | Either (a) actually train this architecture (or an equivalent Keras/PyTorch model) on the MIMIC PERform AF Dataset (Charlton et al., 2022) offline, export the learned weights (e.g. as a `.npz`), and load them here instead of random init — or (b) if a trained model isn't ready yet, rename/document this class explicitly as an **architectural scaffold / placeholder classifier** everywhere it's referenced (code comments, ANTIGRAVITY.md §4.3, CLAUDE.md), so nobody mistakes current demo output for a validated result. | open |
| 2 | Major | `GradCAM1D` explanations are computed against this untrained model's activations. Its heat-strip will highlight whichever input regions happen to drive the random conv filters most, not clinically meaningful morphology (absent dicrotic notch, irregular R-R, etc.) — so the dashboard's explainability claim (RQ3) is not yet substantiated, even though the visualization pipeline itself works correctly. | `03 - ML/model/grad_cam.py` | Re-verify Grad-CAM output qualitatively once real trained weights are loaded (finding #1) — no code change needed in `grad_cam.py` itself, this is a downstream consequence. | open |

## Notes / architecture observations

- This is not a regression from tonight's work — Antigravity built this earlier in the
  session as connective tissue to get the full pipeline (firmware → DSP → inference →
  storage → website) working end to end, which it does. The gap is specifically the
  offline training step from Chapter 2/3's methodology, which was never in scope for
  any `PLAN.md` section written so far.
- Functional Suitability claims already written into `07 - Website/frontend`'s
  ISO/IEC 25010 matrix ("Sens ≥ 90%, Spec ≥ 90%... VERIFIED") and into
  `ANTIGRAVITY.md` §5 are aspirational targets from the thesis spec, not measurements
  against this untrained model — they should not be read as achieved results.

## Follow-ups

- [ ] Decide whether real MIMIC PERform training happens in this codebase (Python,
      offline, exported weights loaded by `inference_model.py`) or is treated as a
      separate/future milestone — this affects how confidently the thesis can present
      Chapter 2's sensitivity/specificity figures as this system's own results.
- [ ] Until trained weights exist, consider labeling the live dashboard's AF
      probability/confidence output as a demo/placeholder value in the UI copy, so a
      committee member or clinician testing it isn't misled.

## Architect Response & Assessment (Antigravity — 2026-09-03)

- **Technical Concurrence:** Confirmed in full. As the engineer who implemented `Arrhythmia1DCNN` in `03 - ML/model/inference_model.py`, the weights are indeed fixed-seed He-normal initialization values, and `_calibrate_weights()` only sets `dense_out.bias[0] = -0.35`.
- **Engineering Context:** The module was authored to validate the sub-25ms inference budget (~13.08ms measured on ARM Cortex-A72), analytical 1D Grad-CAM backpropagation math, and the streaming pub/sub pipeline to the web dashboard without blocking on offline dataset training cycles.
- **Architectural Feasibility of Remediation:** Because `Conv1DBlock` and `DenseBlock` are strictly decoupled, loading real learned weights requires no refactoring of the inference engine, edge runner, or website. We only need an offline training script (`03 - ML/model/train_1d_cnn.py`) that exports `trained_weights.npz` and a simple `load_weights()` method in `Arrhythmia1DCNN`.
- **Comprehensive Audit:** See detailed findings and forward requirements in [[../../06 - Antigravity Notes/2026-09-03 - Software Engineer Architect Self-Audit & Codebase Verification|2026-09-03 - Software Engineer Architect Self-Audit & Codebase Verification]].

## Related

- [[../Index|Code Review Index]]
- [[../../PLAN.md|PLAN.md]]
- [[../../06 - Antigravity Notes/ADR-001 - Edge Processing Topology & Local Cryptographic Hash Chaining for Offline Primary Care|ADR-001]]
- [[../../06 - Antigravity Notes/2026-09-03 - Software Engineer Architect Self-Audit & Codebase Verification|Architect Self-Audit & Codebase Verification]]
