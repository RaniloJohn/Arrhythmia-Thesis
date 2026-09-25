"""
Signal Resampling Utilities (A7 Contract).
==========================================
Provides standardized polyphase rational resampling to convert varying input sampling
rates (e.g. 125 Hz from MIMIC PERform AF, 32/64 Hz from DeepBeat) to the 100 Hz rate
mandated by the edge MAX30102 sensor contract and the frozen 1D-CNN input layer.
"""

import fractions
import math
from typing import Union
import numpy as np


def to_100hz(signal: np.ndarray, fs_in: Union[float, int]) -> np.ndarray:
    """
    Resample a 1-D PPG signal to 100 Hz via polyphase rational resampling.

    Information-Preserving Rationale for DeepBeat Upsampling:
    ---------------------------------------------------------
    For DeepBeat (which is sampled at native ~32 Hz or ~64 Hz), converting to 100 Hz is
    an UPSAMPLING operation. This operation is strictly INFORMATION-PRESERVING for our
    arrhythmia detection task and clinical signal processing pipeline:
      1. Clinical Bandpass Limitation: Our preprocessing filter (ANTIGRAVITY.md §4.2) is a
         4th-order zero-phase Butterworth bandpass filter operating between 0.5 Hz and 5.0 Hz.
         This isolates the cardiac fundamental and harmonic pulse components while removing
         respiratory baseline wander (<0.5 Hz) and electromechanical/sensor noise (>5.0 Hz).
      2. Nyquist Headroom: Even at the lowest candidate native DeepBeat sample rate of 32 Hz,
         the Nyquist limit is 16.0 Hz. At 64 Hz, the Nyquist limit is 32.0 Hz. Both are far
         above the 5.0 Hz cutoff required by the 1D-CNN and peak detection algorithms.
      3. No Spectral Fabrication: Because the physiological information band (0.5–5 Hz) is
         entirely below the original Nyquist frequency, polyphase interpolation up to 100 Hz
         reconstructs the band-limited continuous waveform without introducing spurious
         frequencies or losing clinical features.
      4. Contract Uniformity: Upsampling ensures that all model weights, convolution kernels,
         and peak detection windows evaluate an identical temporal scale (10 s = 1000 samples).

    Parameters:
        signal: 1-D array containing PPG samples.
        fs_in: Input sampling frequency in Hz (e.g., 125.0, 64.0, 32.0).

    Returns:
        Resampled 1-D float32 array at 100 Hz.

    Raises:
        ImportError: If scipy is not available.
        ValueError: If fs_in <= 0 or signal is empty.
        AssertionError: If output length deviates by more than 1 sample from expected.
    """
    try:
        from scipy import signal as scipy_signal
    except ImportError as exc:
        raise ImportError(
            "scipy is required for polyphase resampling. Please install training "
            "dependencies using `pip install -r 03 - ML/training/requirements-train.txt`."
        ) from exc

    fs_float = float(fs_in)
    if fs_float <= 0:
        raise ValueError(f"fs_in must be positive, got {fs_in}")

    sig_arr = np.asarray(signal, dtype=np.float32)
    if sig_arr.ndim != 1:
        sig_arr = sig_arr.flatten()

    if len(sig_arr) == 0:
        raise ValueError("Cannot resample an empty signal.")

    # If already 100 Hz, return array directly
    fs_int = int(round(fs_float))
    if fs_int == 100 and math.isclose(fs_float, 100.0, rel_tol=1e-3):
        return sig_arr.astype(np.float32, copy=False)

    # Compute rational ratio: target / source = 100 / int(fs_in)
    # E.g., for fs_in = 125 Hz -> 100/125 = 4/5 (downsampling)
    #       for fs_in = 32 Hz  -> 100/32  = 25/8 (upsampling)
    #       for fs_in = 64 Hz  -> 100/64  = 25/16 (upsampling)
    frac = fractions.Fraction(100, int(fs_in)).limit_denominator()
    up = frac.numerator
    down = frac.denominator

    # Apply polyphase filtering with rational ratio
    resampled = scipy_signal.resample_poly(sig_arr, up, down)

    # Assert output length matches len(signal) * 100 / fs_in to within one sample
    expected_len = int(round(len(sig_arr) * 100.0 / fs_float))
    actual_len = len(resampled)
    assert abs(actual_len - expected_len) <= 1, (
        f"Resampled length {actual_len} deviates from expected length {expected_len} "
        f"(within 1 sample tolerance) for len(signal)={len(sig_arr)}, fs_in={fs_in}"
    )

    return resampled.astype(np.float32)
