# 03 - ML Test Suite

Automated test suite for the machine learning and edge inference modules of the Arrhythmia Thesis project (3CPE-2A, University of the East).

## Test Coverage Overview

All tests are fully self-contained: they require no physical MAX30102 sensor hardware, no Raspberry Pi, no live network access, and execute deterministically across Windows and Linux.

| Module | Test File | Test Focus |
| :--- | :--- | :--- |
| **DSP Filter** | `test_filter.py` | Butterworth passband (0.5-5.0 Hz), stopband attenuation (0.2 Hz & 12 Hz), zero-phase alignment, detrending, Z-score normalization. |
| **Signal Quality** | `test_sqi.py` | Skewness, kurtosis, perfusion index (PI), clean pulse acceptance, flatline rejection, motion spike rejection, probe-off branch (`dc < 1000`). |
| **Peak Detection** | `test_peak_detection.py` | Elgendi two-moving-average detector, BPM estimation (72 BPM), RMSSD/CV-IBI on irregular intervals, 300 ms refractory period. |
| **UART Protocol** | `test_serial_protocol.py` | 19-byte binary packet framing (`0xAA`/`0x55`), CRC-16-CCITT integrity, byte-by-byte streaming parser, noise resynchronization. |
| **Storage & Hash Chain** | `test_db_manager.py` | SQLite WAL mode initialization, SHA-256 backward hash chain append, ledger validation, tamper detection on corrupted data/hashes. |
| **Inference Model** | `test_inference_model.py` | Pure-NumPy `Arrhythmia1DCNN` forward pass, activation caching, calibrated threshold decision logic, weight shape assertions. |
| **Resampler** | `test_resample.py` | Polyphase resampling to 100 Hz (`to_100hz`), peak frequency preservation across 32, 64, and 125 Hz. |
| **Dataset Loaders** | `test_datasets.py` | Common `PPGDatasetLoader` contract, DeepBeat and MIMIC loaders, metadata extraction. |
| **Dataset Pipeline** | `test_build_dataset.py` | Windowing (1000 samples @ 100 Hz), quality gating, subject-isolated splitting. |
| **Model Parity Gate** | `test_parity.py` | PyTorch mirror vs. pure-NumPy bit-level parity ($\Delta < 5.96 \times 10^{-8}$), negative control on flatten permutation. |
| **Training Pipeline** | `test_train.py` | Anti-MIMIC hard guards, subject-isolated subsampling, validation early stopping. |
| **Threshold Calibration** | `test_calibrate_threshold.py` | Anti-MIMIC and anti-test hard guards, Youden's J calibration, metadata persistence. |
| **Edge Runner Wiring** | `test_runner_weights.py` | Default weight loading, CLI `--weights`, `ARRHYTHMIA_WEIGHTS` env var, fallback warning banner, consensus tracking. |
| **Grad-CAM Gate** | `test_gradcam.py` | Analytical backprop $\partial(\text{logit})/\partial A_2$ vs. `torch.autograd` agreement ($\Delta < 10^{-4}$), upsampling fidelity, [0, 1] normalization. |

## Running Tests

From the repository root:

```powershell
# Using the virtual environment
.venv\Scripts\pytest "03 - ML/tests" -v
```

From within the `03 - ML/` directory:

```bash
pytest -v
```
