# ML Dataset Directory Reference

This directory contains the datasets utilized in the development, training, and external validation of the 1D-CNN photoplethysmography (PPG) cardiac arrhythmia detection model for the undergraduate thesis:
*An Internet of Things-Based Framework for Cardiac Arrhythmia Detection via Photoplethysmography and Machine Learning* (Bauzon, Condino, Delos Angeles, Raña, Villaflor — University of the East).

Per settled architectural decisions (see `MEMORY.md` Decision 9), the two datasets play strictly separated, non-overlapping roles.

---

## 1. DeepBeat Dataset

- **Repository URL:** [https://www.synapse.org/Synapse:syn21985690/files/](https://www.synapse.org/Synapse:syn21985690/files/) (Project ID: `syn21985690`)
- **Direct File Links:**
  - `train.npz`: [https://www.synapse.org/Synapse:syn22006404](https://www.synapse.org/Synapse:syn22006404)
  - `validate.npz`: [https://www.synapse.org/Synapse:syn22006006](https://www.synapse.org/Synapse:syn22006006)
  - `test.npz`: [https://www.synapse.org/Synapse:syn22006407](https://www.synapse.org/Synapse:syn22006407)
- **Code & Model:** [https://github.com/AshleyLab/deepbeat](https://github.com/AshleyLab/deepbeat)
- **Paper Publication:** [npj Digital Medicine (2020) 3:116](https://doi.org/10.1038/s41746-020-00320-4)
- **Access / Download Procedure:**
  1. Log in to [Synapse.org](https://www.synapse.org/login) (registration is free).
  2. Open the canonical repository files view: [https://www.synapse.org/Synapse:syn21985690/files/](https://www.synapse.org/Synapse:syn21985690/files/).
  3. Download `train.npz`, `validate.npz`, and `test.npz`.
  5. (Alternative CLI) Install the Synapse client and fetch programmatically:
     ```bash
     pip install synapseclient
     synapse get -r syn21985690
     ```
  6. Place the files into `03 - ML/data/deepbeat/`.
- **Citation:**
  > Torres-Soto, J., & Ashley, E. A. (2020). Multi-task deep learning for cardiac rhythm detection in wearable devices. *npj Digital Medicine*, 3(1), 116. https://doi.org/10.1038/s41746-020-00320-4
- **Expected Directory Layout:**
  ```text
  03 - ML/data/deepbeat/
  ├── .gitkeep
  ├── metadata.json                  # Records native fs (32.0 Hz), window duration (25.0 s), labels
  ├── validate.npz                   # 518,782 windows, 16 subjects, 800 samples/win (3.35 GB)
  ├── test.npz                       # 17,617 windows, 22 subjects (cardiologist-adjudicated benchmark)
  └── deepbeat*.h5                   # Pre-trained Keras model weights checkpoints (Torres-Soto & Ashley 2020)
  ```
- **Verified Dataset Statistics:**
  - **Native Sampling Rate:** 32.0 Hz
  - **Native Window Length:** 25.00 s (800 samples per window)
  - **Total Windows (val + test):** 536,399 windows
  - **Total Unique Subjects:** 30 unique subjects
  - **Class Balance:** 51,837 AF windows (9.7%) / 484,562 non-AF windows (90.3%)
  - **Total Recording Duration:** 13,409,975.0 seconds (3,725.0 hours / 155.2 days)
  - **Cardiologist-Adjudicated Partition:** `test.npz` (17,617 windows)
- **Role:** **TRAINING SET** (Train, Validation, and Internal Test Benchmark).
- **One-Line Rationale:** Matches our MAX30102 wrist-worn reflectance optical geometry and provides necessary scale (~536k windows across 30 subjects, 3,725 hours) with fine-grained per-window rhythm labels to train our ~1.03M-parameter model.

---

## 2. MIMIC PERform AF Dataset

- **Source URL:** https://doi.org/10.5281/zenodo.15906524
- **Access / Licence Procedure:** Open access under Creative Commons Attribution 4.0 International (CC-BY 4.0), hosted as an open citable research artifact on Zenodo without requiring PhysioNet clinical database credentialing.
- **Citation:**
  > Charlton, P. H., Celka, P., Farukh, B., Chowienczyk, P., & Alastruey, J. (2022). An assessment of algorithms for estimating respiratory rate from the electrocardiogram and photoplethysmogram in atrial fibrillation. *Physiological Measurement*, 43(11), 115003. DOI: 10.5281/zenodo.15906524.
- **Expected Directory Layout:**
  ```text
  03 - ML/data/mimic_perform/
  ├── .gitkeep
  ├── mimic_perform_af_csv/
  │   ├── mimic_perform_af_001_data.csv ... mimic_perform_af_019_data.csv (19 subjects)
  │   └── mimic_perform_af_001_fix.txt ... mimic_perform_af_019_fix.txt
  └── mimic_perform_non_af_csv/
      ├── mimic_perform_non_af_001_data.csv ... mimic_perform_non_af_016_data.csv (16 subjects)
      └── mimic_perform_non_af_001_fix.txt ... mimic_perform_non_af_016_fix.txt
  ```
- **Role:** **HELD-OUT EXTERNAL VALIDATION SET ONLY.**
- **One-Line Rationale:** Serves as a completely independent, out-of-distribution cohort (ICU transmissive fingertip PPG at 125 Hz) that must never appear in training or threshold calibration.

---

## 3. Processed Datasets

- **Directory:** `03 - ML/data/processed/`
- **Builder:** Produced deterministically by `training/build_dataset.py --dataset {deepbeat,mimic}` conforming to canonical contract (1,000 samples = 10.0 s @ 100 Hz, production DSP pipeline: `detrend` -> `bandpass` [0.5-5 Hz] -> `zscore_normalize`).
- **Files & Verified Yields:**
  - `deepbeat_train.npz` (1.68 GB): 451,791 windows across 12 subjects (12.34% AF / 87.66% non-AF). 348,027 windows dropped via quality gating (SQI < 0.50 or dataset quality == 2).
  - `deepbeat_val.npz` (947 MB): 255,874 windows across 4 subjects (4.94% AF / 95.06% non-AF).
  - `deepbeat_test.npz` (63.6 MB): 17,106 windows across 14 cardiologist-adjudicated subjects (49.46% AF / 50.54% non-AF).
  - `deepbeat_manifest.json`: Full manifest with drop reasons, split hashes, and SQI cross-tabulation.
  - `mimic_external_val.npz` (15.5 MB): 4,200 windows across 35 subjects (54.29% AF / 45.71% non-AF).
  - `mimic_manifest.json`: Full manifest with recording-level limitation notes.
- **Enforcement:** Both raw datasets and processed arrays are excluded from version control via `.gitignore`. Split assignment manifests reside in `03 - ML/training/splits/`.
