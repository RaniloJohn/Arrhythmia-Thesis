"""
PyTest Suite: PyTorch <-> Pure-NumPy Bit-Parity Gate (ANTIGRAVITY.md §4.3)
========================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Tests:
1. Bit-level parity: random float32 windows produce equivalent output within atol=1e-5.
2. Negative control: wrong flatten permutation (channel-major vs time-major) FAILS parity.
3. Shape assertion: load_weights raises ValueError on shape mismatches across all layers.
4. Key assertion: load_weights raises ValueError if any required parameter key is missing.
5. Missing file assertion: load_weights raises FileNotFoundError for missing path.
6. Calibrated threshold: load_weights correctly loads decision threshold from sibling meta json.
7. Grad-CAM compatibility: verified forward_with_cache remains fully compatible with GradCAM1D.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure 03 - ML is on sys.path
ml_dir = Path(__file__).resolve().parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from model.inference_model import Arrhythmia1DCNN as NumPyArrhythmia1DCNN
from model.grad_cam import GradCAM1D
from training.torch_model import Arrhythmia1DCNN as TorchArrhythmia1DCNN
from training.export_weights import export_weights, extract_numpy_weights


def test_torch_numpy_bit_parity(tmp_path: Path):
    """
    Primary Gate: Randomly initialized PyTorch model exported to .npz and loaded into
    pure-NumPy Arrhythmia1DCNN must compute the exact same function within atol=1e-5.
    """
    torch.manual_seed(42)
    torch_model = TorchArrhythmia1DCNN(input_length=1000, dropout=0.5)
    torch_model.eval()  # Ensure dropout is identity

    weights_npz = tmp_path / "test_weights.npz"
    export_weights(torch_model, weights_npz, threshold=0.50)

    np_model = NumPyArrhythmia1DCNN(input_length=1000)
    np_model.load_weights(weights_npz)

    assert np_model.weights_loaded is True
    assert np_model.weights_source == str(weights_npz.resolve())

    # Generate 32 distinct random float32 windows (1000 samples)
    rng = np.random.default_rng(2026)
    windows = rng.standard_normal(size=(32, 1000)).astype(np.float32)

    max_abs_diff = 0.0
    diffs = []

    for i in range(32):
        win = windows[i]

        # NumPy runtime inference
        np_cache = np_model.forward_with_cache(win)
        np_prob = np_cache["prob"]
        np_pred, np_af, np_lat = np_model.predict(win)
        assert np.isclose(np_prob, np_pred, atol=1e-6)

        # PyTorch model inference
        with torch.no_grad():
            t_win = torch.from_numpy(win).unsqueeze(0).unsqueeze(1)
            torch_prob = float(torch_model(t_win, return_logits=False).item())

        diff = abs(torch_prob - np_prob)
        diffs.append(diff)
        if diff > max_abs_diff:
            max_abs_diff = diff

        assert np.allclose(torch_prob, np_prob, atol=1e-5), (
            f"Window {i} failed parity check: torch={torch_prob:.8f}, numpy={np_prob:.8f}, diff={diff:.8e}"
        )

    print(f"\n[Parity Gate Result] Evaluated 32 windows. Max absolute diff: {max_abs_diff:.4e} (tolerance: 1e-5)")
    assert max_abs_diff < 1e-5, f"Maximum difference {max_abs_diff} exceeds tolerance 1e-5!"


def test_negative_control_wrong_flatten_permutation_fails(tmp_path: Path):
    """
    Negative Control: Deliberately export dense1.weights with wrong channel-major
    permutation and assert that the parity check FAILS. This guarantees that the parity
    test is sensitive to the flatten ordering and cannot pass vacuously.
    """
    torch.manual_seed(101)
    torch_model = TorchArrhythmia1DCNN(input_length=1000, dropout=0.5)
    torch_model.eval()

    weights_npz = tmp_path / "clean_weights.npz"
    export_weights(torch_model, weights_npz, threshold=0.50)

    # Corrupt dense1.weights with wrong channel-major permutation
    # In time-major, rows are indexed: t * 64 + c (time=250, channels=64)
    # In channel-major, rows are indexed: c * 250 + t (channels=64, time=250)
    with np.load(weights_npz) as raw:
        corrupted_dict = {k: raw[k] for k in raw.files}

    w_clean = corrupted_dict["dense1.weights"]  # (16000, 64)
    # Permute time and channel dimensions:
    w_corrupted = (
        w_clean.reshape(250, 64, 64)  # (time, channel, out_features)
        .transpose(1, 0, 2)           # (channel, time, out_features)
        .reshape(16000, 64)           # flattened channel-major
    )
    corrupted_dict["dense1.weights"] = w_corrupted

    corrupted_npz = tmp_path / "corrupted_weights.npz"
    np.savez(corrupted_npz, **corrupted_dict)

    # Load corrupted weights into NumPy model
    np_corrupted_model = NumPyArrhythmia1DCNN(input_length=1000)
    np_corrupted_model.load_weights(corrupted_npz)

    # Evaluate on test windows
    rng = np.random.default_rng(999)
    windows = rng.standard_normal(size=(32, 1000)).astype(np.float32)

    diffs = []
    for i in range(32):
        win = windows[i]
        with torch.no_grad():
            t_win = torch.from_numpy(win).unsqueeze(0).unsqueeze(1)
            torch_prob = float(torch_model(t_win, return_logits=False).item())

        np_prob = np_corrupted_model.forward_with_cache(win)["prob"]
        diffs.append(abs(torch_prob - np_prob))

    max_diff = max(diffs)
    print(f"\n[Negative Control Result] Max difference with corrupted flatten: {max_diff:.4e}")

    # Parity MUST fail under wrong permutation
    assert max_diff > 1e-3, (
        f"Negative control failed: expected significant difference (> 1e-3), but got max diff {max_diff}"
    )
    # Confirm np.allclose fails
    assert not np.allclose([torch_model(torch.from_numpy(w).unsqueeze(0).unsqueeze(1)).item() for w in windows],
                           [np_corrupted_model.forward_with_cache(w)["prob"] for w in windows],
                           atol=1e-5)


def test_load_weights_shape_mismatch_raises(tmp_path: Path):
    """
    Verifies that load_weights asserts exact array shapes for all layers and raises ValueError.
    """
    torch.manual_seed(77)
    torch_model = TorchArrhythmia1DCNN(input_length=1000)
    clean_dict = extract_numpy_weights(torch_model)

    test_shapes = {
        "conv1.weights": (16, 1, 5),     # Wrong out_channels (16 vs 32)
        "conv1.bias": (16,),             # Wrong bias size
        "conv2.weights": (64, 16, 3),    # Wrong in_channels (16 vs 32)
        "conv2.bias": (32,),             # Wrong out_channels (32 vs 64)
        "dense1.weights": (8000, 64),    # Wrong in_features (8000 vs 16000)
        "dense1.bias": (32,),            # Wrong out_features (32 vs 64)
        "dense_out.weights": (32, 1),    # Wrong in_features (32 vs 64)
        "dense_out.bias": (2,),          # Wrong out_features (2 vs 1)
    }

    for key_to_corrupt, bad_shape in test_shapes.items():
        corrupted = dict(clean_dict)
        corrupted[key_to_corrupt] = np.zeros(bad_shape, dtype=np.float32)

        bad_path = tmp_path / f"bad_{key_to_corrupt}.npz"
        np.savez(bad_path, **corrupted)

        model = NumPyArrhythmia1DCNN(input_length=1000)
        with pytest.raises(ValueError, match=f"Shape mismatch for '{key_to_corrupt}'"):
            model.load_weights(bad_path)


def test_load_weights_missing_key_raises(tmp_path: Path):
    """
    Verifies that load_weights fails loudly if any required parameter key is missing.
    """
    torch.manual_seed(77)
    torch_model = TorchArrhythmia1DCNN(input_length=1000)
    clean_dict = extract_numpy_weights(torch_model)

    for key_to_remove in clean_dict.keys():
        corrupted = {k: v for k, v in clean_dict.items() if k != key_to_remove}
        bad_path = tmp_path / f"missing_{key_to_remove}.npz"
        np.savez(bad_path, **corrupted)

        model = NumPyArrhythmia1DCNN(input_length=1000)
        with pytest.raises(ValueError, match=f"Missing required weight key '{key_to_remove}'"):
            model.load_weights(bad_path)


def test_load_weights_nonexistent_file_raises():
    """
    Verifies that load_weights raises FileNotFoundError on missing file.
    """
    model = NumPyArrhythmia1DCNN(input_length=1000)
    with pytest.raises(FileNotFoundError, match="Weights file not found"):
        model.load_weights("nonexistent_path_to_weights.npz")


def test_load_weights_calibrated_threshold(tmp_path: Path):
    """
    Verifies that load_weights reads and applies the calibrated threshold from sibling .meta.json.
    """
    torch.manual_seed(55)
    torch_model = TorchArrhythmia1DCNN(input_length=1000)
    weights_npz = tmp_path / "threshold_test.npz"
    export_weights(torch_model, weights_npz, threshold=0.72)

    model = NumPyArrhythmia1DCNN(input_length=1000)
    model.load_weights(weights_npz)

    assert model.threshold == 0.72

    # Manually test decision gating based on self.threshold
    # Mock prob calculation
    model.dense_out.bias[0] = 0.0  # Reset
    # Simulate window with known output
    win = np.zeros(1000, dtype=np.float32)
    cache = model.forward_with_cache(win)
    prob = cache["prob"]
    expected_detected = int(prob >= 0.72)
    assert cache["af_detected"] == expected_detected


def test_grad_cam_compatibility_with_loaded_weights(tmp_path: Path):
    """
    Verifies that GradCAM1D works seamlessly with weights loaded into Arrhythmia1DCNN.
    """
    torch.manual_seed(12)
    torch_model = TorchArrhythmia1DCNN(input_length=1000)
    weights_npz = tmp_path / "gradcam_test.npz"
    export_weights(torch_model, weights_npz, threshold=0.50)

    model = NumPyArrhythmia1DCNN(input_length=1000)
    model.load_weights(weights_npz)

    explainer = GradCAM1D(model)
    win = np.sin(np.linspace(0, 20 * np.pi, 1000)).astype(np.float32)

    explanation = explainer.explain(win)
    assert "af_probability" in explanation
    assert "af_detected" in explanation
    assert "weights" in explanation
    assert len(explanation["weights"]) == 1000
    assert 0.0 <= min(explanation["weights"]) <= max(explanation["weights"]) <= 1.0
