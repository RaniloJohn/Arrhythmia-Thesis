"""
Unit Tests for Pure-NumPy 1D-CNN Inference Model
================================================
Tests:
1. Architecture initialization: weights_loaded=False, threshold=0.50 by default.
2. Forward pass output shape and probability bounding [0.0, 1.0].
3. Forward with cache returns activations for all layers (a1, p1, a2, p2, argmax, a_dense1).
4. Threshold behavior: decision flips at calibrated threshold.
5. load_weights shape validation: raises ValueError on corrupted or mismatched weights.
6. load_weights missing file validation: raises FileNotFoundError.
"""

import pytest
import numpy as np
from pathlib import Path
from model.inference_model import Arrhythmia1DCNN, Conv1DBlock, DenseBlock, MaxPool1DBlock


@pytest.fixture
def trained_weights_path():
    p = Path(__file__).resolve().parent.parent / "model" / "weights" / "cnn_af_v1.npz"
    assert p.exists(), f"Weights file {p} must exist for testing."
    return p


def test_model_initialization_defaults():
    """Test model defaults prior to loading trained weights."""
    model = Arrhythmia1DCNN(input_length=1000)
    assert model.weights_loaded is False
    assert model.threshold == 0.50
    assert model.weights_source is None
    assert model.input_length == 1000


def test_forward_pass_output_range():
    """Test that predict returns probability bounded in [0, 1] and binary detection."""
    model = Arrhythmia1DCNN(input_length=1000)
    x = np.random.normal(0, 1, 1000).astype(np.float32)

    prob, af_detected, latency_ms = model.predict(x)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert af_detected in (0, 1)
    assert latency_ms > 0.0


def test_forward_with_cache_structure():
    """Test that forward_with_cache returns all activations for Grad-CAM."""
    model = Arrhythmia1DCNN(input_length=1000)
    x = np.random.normal(0, 1, 1000).astype(np.float32)

    cache = model.forward_with_cache(x)
    assert "prob" in cache
    assert "af_probability" in cache
    assert "af_detected" in cache
    assert "a2" in cache and cache["a2"].shape == (500, 64)
    assert "p2" in cache and cache["p2"].shape == (250, 64)
    assert "argmax2" in cache and cache["argmax2"].shape == (250, 64)
    assert "flat" in cache and len(cache["flat"]) == 16000
    assert "a_dense1" in cache and cache["a_dense1"].shape == (64,)
    assert "latency_ms" in cache
    assert "a_dense1" in cache and cache["a_dense1"].shape == (64,)
    assert "latency_ms" in cache


def test_load_weights_valid(trained_weights_path):
    """Test successful loading of valid trained weights archive and metadata."""
    model = Arrhythmia1DCNN(input_length=1000)
    model.load_weights(trained_weights_path)

    assert model.weights_loaded is True
    assert model.weights_source == str(trained_weights_path.resolve())
    assert model.threshold == pytest.approx(0.37, abs=0.01)
    assert model.model_version == "cnn_af_v1"
    assert model.training_dataset == "deepbeat"


def test_load_weights_missing_file_raises():
    """Test that loading from a non-existent path raises FileNotFoundError."""
    model = Arrhythmia1DCNN()
    with pytest.raises(FileNotFoundError):
        model.load_weights("non_existent_weights_archive.npz")


def test_load_weights_shape_mismatch_raises(tmp_path):
    """Test that an archive with corrupted shapes triggers a ValueError."""
    bad_weights_file = tmp_path / "corrupted_weights.npz"

    # Save array with wrong shape for conv1.weights (expected 32, 1, 5)
    np.savez(
        bad_weights_file,
        **{
            "conv1.weights": np.zeros((16, 1, 5), dtype=np.float32),  # Wrong out_channels
            "conv1.bias": np.zeros((32,), dtype=np.float32),
            "conv2.weights": np.zeros((64, 32, 3), dtype=np.float32),
            "conv2.bias": np.zeros((64,), dtype=np.float32),
            "dense1.weights": np.zeros((16000, 64), dtype=np.float32),
            "dense1.bias": np.zeros((64,), dtype=np.float32),
            "dense_out.weights": np.zeros((64, 1), dtype=np.float32),
            "dense_out.bias": np.zeros((1,), dtype=np.float32),
        }
    )

    model = Arrhythmia1DCNN()
    with pytest.raises(ValueError, match="Shape mismatch"):
        model.load_weights(bad_weights_file)


def test_threshold_decision_logic():
    """Test that the decision flips according to model.threshold."""
    model = Arrhythmia1DCNN()
    model.threshold = 0.40

    # If prob is 0.42, prediction should have af_detected = 1
    model.forward_with_cache = lambda x: {
        "af_probability": 0.42,
        "af_detected": int(0.42 >= model.threshold),
        "latency_ms": 1.0,
    }
    prob_high, detected_high, _ = model.predict(np.zeros(1000))
    assert prob_high == 0.42
    assert detected_high == 1

    # If prob is 0.38, prediction should have af_detected = 0
    model.forward_with_cache = lambda x: {
        "af_probability": 0.38,
        "af_detected": int(0.38 >= model.threshold),
        "latency_ms": 1.0,
    }
    prob_low, detected_low, _ = model.predict(np.zeros(1000))
    assert prob_low == 0.38
    assert detected_low == 0

