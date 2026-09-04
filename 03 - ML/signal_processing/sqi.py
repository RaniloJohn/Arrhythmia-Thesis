"""
Signal Quality Index (SQI) Module (ANTIGRAVITY.md §4.2)
=======================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Features:
- Skewness SQI (S_sqi): Positive skewness corresponds to clean pulsatile morphology.
- Kurtosis SQI (K_sqi): Leptokurtic distribution corresponds to clean peak profiles.
- Perfusion Index (PI): AC/DC ratio reflecting pulsatile vascular amplitude.
- Rejection of motion-corrupted or loose-probe windows before running 1D-CNN.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple


class SignalQualityAssessor:
    def __init__(self, min_pi: float = 0.15, min_sqi: float = 0.50):
        self.min_pi = min_pi
        self.min_sqi = min_sqi

    def compute_metrics(self, raw_data: np.ndarray, filtered_data: np.ndarray) -> Dict[str, Any]:
        """
        Calculates SQI metrics for a single PPG window.
        Returns: {
            'skewness': float,
            'kurtosis': float,
            'perfusion_index': float,
            'sqi_score': float, # 0.0 to 1.0
            'is_valid': bool
        }
        """
        if len(raw_data) == 0 or len(filtered_data) == 0:
            return {"skewness": 0.0, "kurtosis": 0.0, "perfusion_index": 0.0, "sqi_score": 0.0, "is_valid": False}

        # 1. Perfusion Index (PI): AC amplitude / DC baseline * 100%
        dc_val = np.mean(raw_data)
        if dc_val < 1000.0:  # Probe off or disconnected
            pi = 0.0
        else:
            ac_val = np.max(raw_data) - np.min(raw_data)
            pi = float((ac_val / dc_val) * 100.0)

        # 2. Skewness and Kurtosis of zero-phase filtered window
        skew = float(stats.skew(filtered_data))
        kurt = float(stats.kurtosis(filtered_data))

        # 3. Composite SQI calculation
        # Clean physiological PPG typically has skewness between -0.5 and 2.5,
        # kurtosis between -1.0 and 8.0, and PI > 0.2%.
        score = 1.0
        if pi < self.min_pi:
            score -= 0.4
        if skew < -0.8 or skew > 3.0:
            score -= 0.3
        if kurt < -1.2 or kurt > 10.0:
            score -= 0.3

        score = max(0.0, min(1.0, score))
        is_valid = (score >= self.min_sqi) and (pi >= self.min_pi)

        return {
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "perfusion_index": round(pi, 2),
            "sqi_score": round(score, 2),
            "is_valid": bool(is_valid)
        }
