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
from typing import Any, Dict, Optional, Tuple


class SignalQualityAssessor:
    """
    Window quality scoring.

    `dc_floor` guards against a loose or absent probe by rejecting windows whose raw DC
    mean is implausibly low. It is expressed in **raw MAX30102 ADC counts**, which is only
    meaningful for live sensor data. Pass `dc_floor=None` for signals that are not raw
    counts — normalised or pre-filtered file-based datasets — otherwise the branch fires on
    every window, forces `pi = 0`, and silently caps every score at 0.6. That is exactly
    what happened in the v1 training build: no window in a 1,072,798-window corpus could
    score above 0.6, so the `>= 0.50` gate degenerated into a skew/kurtosis filter and
    dropped 43.5% of the training split on a metric whose dominant term was inoperative.
    See `02 - Code Review/2026-10-01 - 1D-CNN Training Audit…`.
    """

    def __init__(self, min_pi: float = 0.15, min_sqi: float = 0.50,
                 dc_floor: Optional[float] = 1000.0):
        self.min_pi = min_pi
        self.min_sqi = min_sqi
        self.dc_floor = dc_floor

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
        #    AC/DC is inherently scale-free; only the absolute probe-off floor is not, so
        #    that check is applied solely when a floor is configured (live sensor data).
        dc_val = float(np.mean(raw_data))
        pi_available = True
        if self.dc_floor is not None and dc_val < self.dc_floor:
            pi = 0.0  # probe off or disconnected
        elif abs(dc_val) < 1e-12:
            # DC is zero (e.g. an already mean-removed signal): AC/DC is undefined, so the
            # PI criterion is marked unavailable rather than scored as a failure.
            pi = 0.0
            pi_available = False
        else:
            ac_val = float(np.max(raw_data) - np.min(raw_data))
            pi = float((ac_val / abs(dc_val)) * 100.0)

        # 2. Skewness and Kurtosis of zero-phase filtered window
        skew = float(stats.skew(filtered_data))
        kurt = float(stats.kurtosis(filtered_data))

        # 3. Composite SQI calculation
        # Clean physiological PPG typically has skewness between -0.5 and 2.5,
        # kurtosis between -1.0 and 8.0, and PI > 0.2%.
        #    When PI cannot be computed the remaining criteria are reweighted so the score
        #    still spans [0, 1]; otherwise an unavailable term would cap the score at 0.6.
        if pi_available:
            score = 1.0
            if pi < self.min_pi:
                score -= 0.4
            if skew < -0.8 or skew > 3.0:
                score -= 0.3
            if kurt < -1.2 or kurt > 10.0:
                score -= 0.3
        else:
            score = 1.0
            if skew < -0.8 or skew > 3.0:
                score -= 0.5
            if kurt < -1.2 or kurt > 10.0:
                score -= 0.5

        score = max(0.0, min(1.0, score))
        is_valid = (score >= self.min_sqi) and (pi >= self.min_pi or not pi_available)

        return {
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "perfusion_index": round(pi, 2),
            "sqi_score": round(score, 2),
            "pi_available": bool(pi_available),
            "is_valid": bool(is_valid)
        }
