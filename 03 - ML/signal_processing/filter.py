"""
DSP Preprocessing & Filtering Module (ANTIGRAVITY.md §4.2)
==========================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Features:
- 4th-order zero-phase Butterworth bandpass filter (0.5 Hz - 5.0 Hz).
- Suppresses baseline respiratory wander (<0.5 Hz) and high-frequency EMG noise (>5.0 Hz).
- Window detrending and Z-score normalization.
"""

import numpy as np
from scipy import signal


class ButterBandpassFilter:
    def __init__(self, lowcut: float = 0.5, highcut: float = 5.0, fs: float = 100.0, order: int = 4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        # 4th order forward-backward filter gives an effective 8th order steep roll-off
        self.b, self.a = signal.butter(order, [low, high], btype='band')

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply zero-phase forward-backward Butterworth bandpass filter.
        Uses filtfilt to avoid phase distortion of pulse morphology.
        """
        if len(data) < 3 * max(len(self.b), len(self.a)):
            # Too short for filter transient padding
            return np.array(data, dtype=np.float32)

        # Pad with reflect mode to minimize edge transients
        filtered = signal.filtfilt(self.b, self.a, data, method="pad", padlen=min(150, len(data) - 1))
        return filtered.astype(np.float32)


def detrend_ppg(data: np.ndarray) -> np.ndarray:
    """Remove linear and baseline trends from the window."""
    return signal.detrend(data, type='linear').astype(np.float32)


def zscore_normalize(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score normalization: zero mean, unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    if std < eps:
        std = eps
    return ((data - mean) / std).astype(np.float32)


def minmax_normalize(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale signal amplitude to [0.0, 1.0]."""
    min_val = np.min(data)
    max_val = np.max(data)
    diff = max_val - min_val
    if diff < eps:
        diff = eps
    return ((data - min_val) / diff).astype(np.float32)
