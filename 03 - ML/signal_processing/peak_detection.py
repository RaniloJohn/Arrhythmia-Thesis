"""
Pulse Segmentation & Peak Detection Module (ANTIGRAVITY.md §4.2)
================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Features:
- Elgendi et al. two-moving-average systolic peak detector optimized for 100 Hz PPG.
- Inter-Beat Interval (IBI) extraction.
- Instantaneous and mean Heart Rate (BPM) estimation.
- Root Mean Square of Successive Differences (RMSSD) and coefficient of variation for rhythm analysis.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class ElgendiPeakDetector:
    def __init__(self, fs: float = 100.0, w1_sec: float = 0.111, w2_sec: float = 0.667, beta: float = 0.02):
        self.fs = fs
        self.w1 = max(1, int(np.round(w1_sec * fs)))  # Peak window ~11 samples @ 100Hz
        self.w2 = max(2, int(np.round(w2_sec * fs)))  # Beat window ~67 samples @ 100Hz
        self.beta = beta

    def detect_peaks(self, filtered_ppg: np.ndarray) -> np.ndarray:
        """
        Executes Elgendi two-moving-average systolic peak detection.
        Returns array of peak sample indices.
        """
        n = len(filtered_ppg)
        if n < self.w2:
            return np.array([], dtype=np.int32)

        # 1. Square the signal to emphasize peaks
        squared = np.square(np.maximum(0, filtered_ppg))

        # 2. Moving average 1 (MA_peak)
        kernel1 = np.ones(self.w1) / self.w1
        ma_peak = np.convolve(squared, kernel1, mode='same')

        # 3. Moving average 2 (MA_beat)
        kernel2 = np.ones(self.w2) / self.w2
        ma_beat = np.convolve(squared, kernel2, mode='same')

        # 4. Thresholding: MA_peak > MA_beat + beta * mean(squared)
        threshold = ma_beat + (self.beta * np.mean(squared))
        blocks = ma_peak > threshold

        # 5. Extract maximum within each block
        peaks = []
        in_block = False
        block_start = 0

        for i in range(n):
            if blocks[i] and not in_block:
                in_block = True
                block_start = i
            elif not blocks[i] and in_block:
                in_block = False
                block_end = i
                # Find argmax in original filtered signal within the block
                peak_idx = block_start + np.argmax(filtered_ppg[block_start:block_end])
                peaks.append(peak_idx)

        if in_block:
            peak_idx = block_start + np.argmax(filtered_ppg[block_start:n])
            peaks.append(peak_idx)

        # Refractory period: discard any peak closer than 300 ms (200 BPM)
        min_distance = int(0.30 * self.fs)
        filtered_peaks = []
        for p in peaks:
            if not filtered_peaks or (p - filtered_peaks[-1]) >= min_distance:
                filtered_peaks.append(p)

        return np.array(filtered_peaks, dtype=np.int32)

    def analyze_intervals(self, peaks: np.ndarray) -> Dict[str, Any]:
        """
        Computes IBI series, BPM, and heart rate variability metrics.
        """
        if len(peaks) < 2:
            return {
                "peak_indices": peaks.tolist(),
                "ibi_ms": [],
                "bpm": 0.0,
                "rmssd_ms": 0.0,
                "cv_ibi": 0.0,
                "irregular_intervals": False
            }

        ibi_samples = np.diff(peaks)
        ibi_ms = (ibi_samples / self.fs * 1000.0).astype(np.float32)

        # Average BPM
        mean_ibi = np.mean(ibi_ms)
        bpm = float(60000.0 / mean_ibi) if mean_ibi > 0 else 0.0

        # RMSSD (Root Mean Square of Successive Differences)
        if len(ibi_ms) >= 2:
            successive_diffs = np.diff(ibi_ms)
            rmssd = float(np.sqrt(np.mean(np.square(successive_diffs))))
            cv_ibi = float(np.std(ibi_ms) / mean_ibi)
        else:
            rmssd = 0.0
            cv_ibi = 0.0

        # Irregular interval indicator (AF characteristic: CV > 0.15 or RMSSD > 80ms)
        irregular = (cv_ibi > 0.15) or (rmssd > 80.0)

        return {
            "peak_indices": peaks.tolist(),
            "ibi_ms": [round(float(x), 1) for x in ibi_ms],
            "bpm": round(bpm, 1),
            "rmssd_ms": round(rmssd, 1),
            "cv_ibi": round(cv_ibi, 3),
            "irregular_intervals": bool(irregular)
        }
