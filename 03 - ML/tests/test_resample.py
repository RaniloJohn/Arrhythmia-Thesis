"""
Unit Tests for training/resample.py (A7 Contract).
=================================================
Validates polyphase resampling to 100 Hz across standard dataset sampling rates:
  - MIMIC PERform AF: 125 Hz -> 100 Hz (downsampling, ratio 4/5)
  - DeepBeat: 32 Hz -> 100 Hz (upsampling, ratio 25/8)
  - DeepBeat: 64 Hz -> 100 Hz (upsampling, ratio 25/16)
  - Native 100 Hz: identity preservation
"""

import math
import numpy as np
import pytest

from training.resample import to_100hz


def test_resample_125_to_100():
    """Verify downsampling from MIMIC PERform AF 125 Hz to 100 Hz."""
    duration_s = 10.0
    fs_in = 125.0
    n_samples = int(duration_s * fs_in)  # 1250 samples
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    # 1.2 Hz simulated cardiac pulse (72 BPM)
    signal = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)

    resampled = to_100hz(signal, fs_in)

    assert isinstance(resampled, np.ndarray)
    assert resampled.dtype == np.float32
    assert resampled.ndim == 1
    expected_len = int(round(n_samples * 100.0 / fs_in))  # 1000 samples
    assert abs(len(resampled) - expected_len) <= 1


def test_resample_32_to_100():
    """Verify upsampling from DeepBeat 32 Hz to 100 Hz."""
    duration_s = 25.0
    fs_in = 32.0
    n_samples = int(duration_s * fs_in)  # 800 samples
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    signal = np.sin(2 * np.pi * 1.0 * t).astype(np.float32)

    resampled = to_100hz(signal, fs_in)

    assert resampled.dtype == np.float32
    expected_len = int(round(n_samples * 100.0 / fs_in))  # 2500 samples
    assert abs(len(resampled) - expected_len) <= 1


def test_resample_64_to_100():
    """Verify upsampling from DeepBeat 64 Hz to 100 Hz."""
    duration_s = 25.0
    fs_in = 64.0
    n_samples = int(duration_s * fs_in)  # 1600 samples
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    signal = np.cos(2 * np.pi * 1.5 * t).astype(np.float32)

    resampled = to_100hz(signal, fs_in)

    assert resampled.dtype == np.float32
    expected_len = int(round(n_samples * 100.0 / fs_in))  # 2500 samples
    assert abs(len(resampled) - expected_len) <= 1


def test_resample_identity_100():
    """Verify 100 Hz input returns equivalent signal without distortion."""
    signal = np.random.randn(1000).astype(np.float32)
    resampled = to_100hz(signal, 100.0)

    assert len(resampled) == len(signal)
    assert np.allclose(resampled, signal)


def test_resample_invalid_inputs():
    """Verify appropriate exceptions for invalid arguments."""
    with pytest.raises(ValueError):
        to_100hz(np.array([], dtype=np.float32), 125.0)

    with pytest.raises(ValueError):
        to_100hz(np.ones(100, dtype=np.float32), 0)

    with pytest.raises(ValueError):
        to_100hz(np.ones(100, dtype=np.float32), -10.0)


if __name__ == "__main__":
    test_resample_125_to_100()
    test_resample_32_to_100()
    test_resample_64_to_100()
    test_resample_identity_100()
    test_resample_invalid_inputs()
    print("All resampling unit tests passed.")
