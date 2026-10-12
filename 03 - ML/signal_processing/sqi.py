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
            # AC is taken from the band-passed window, not raw max-min: raw max-min counts
            # baseline drift and contact settling as "pulse", so a sensor sliding on the skin
            # inflated PI ~10x and passed the gate with a 0.03% true pulse — the
            # windows that produced false AF on the device (2026-10-06/12 field tests).
            # 5th-95th percentile keeps one spike from standing in for the pulse amplitude.
            lo, hi = np.percentile(filtered_data, [5, 95])
            ac_val = float(hi - lo)
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


# MAX30102 ADC full scale at 18-bit resolution. A window that touches it has clipped pulse
# tops (or no pulse at all) — typically a finger pressed hard enough to saturate the IR
# photodiode — so nothing measured from it is a heartbeat.
MAX30102_FULL_SCALE = 262143


def is_saturated(raw_data: np.ndarray, full_scale: int = MAX30102_FULL_SCALE,
                 margin: int = 64) -> bool:
    """True when any raw sample is at (or within `margin` counts of) ADC full scale."""
    raw = np.asarray(raw_data)
    return bool(raw.size and np.max(raw) >= full_scale - margin)


def beat_morphology_sqi(filtered: np.ndarray, peaks: np.ndarray, fs: float = 100.0,
                        pre_s: float = 0.25, post_s: float = 0.15, r_min: float = 0.90,
                        min_good_fraction: float = 0.70,
                        max_undetected: int = 1) -> Dict[str, Any]:
    """
    Beat-template quality check: are the detected peaks actually pulses?

    The IBI classifier sees only beat times, so a split beat (one pulse counted twice) or a
    missed beat is indistinguishable to it from AF. This gate catches those detection errors
    from the waveform rather than from the intervals, so it does **not** reject a window for
    being irregular — AF beats keep a normal pulse shape even when their timing is chaotic.
    (Template matching after Orphanidou et al. 2015 and Li & Clifford 2012.)

    1. Each detected peak is cut out as a short segment (upstroke + peak), mean-removed, and
       correlated with the window's median beat. A beat is good when r >= `r_min`.
    2. Prominent maxima that no detected peak accounts for are counted as undetected pulses.

    The window is valid when at least `min_good_fraction` of beats are good and no more than
    `max_undetected` pulses were missed.

    Thresholds were chosen on 2026-10-12 against MIMIC PERform AF (non-overlapping eval
    windows, all 35 subjects) and one 2-min raw MAX30102 WRIST capture of a non-AF subject:
    keeps 95.2% of AF and 91.4% of non-AF MIMIC windows; rejects 21 of 23 device windows,
    and neither retained device window was classified AF. Longer segments (covering the
    downstroke) dropped MIMIC AF retention below 93%, because AF beat lengths vary.
    The thresholds rest on one device capture and must be re-checked on more device data.
    """
    x = np.asarray(filtered, dtype=np.float64)
    p = np.asarray(peaks, dtype=np.int64).ravel()
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    usable = p[(p >= pre) & (p < len(x) - post)]
    out = {"n_beats": int(usable.size), "good_fraction": 0.0, "median_r": 0.0,
           "undetected_pulses": 0, "is_valid": False}
    if usable.size < 4:
        return out

    segs = np.stack([x[i - pre:i + post] for i in usable])
    segs = segs - segs.mean(axis=1, keepdims=True)
    tmpl = np.median(segs, axis=0)
    tmpl = tmpl - tmpl.mean()  # Pearson r: both vectors centred
    denom = np.linalg.norm(segs, axis=1) * np.linalg.norm(tmpl)
    r = np.where(denom > 0, segs @ tmpl / np.where(denom > 0, denom, 1.0), 0.0)

    # Prominence of each detected peak against the lowest point within 0.4 s either side.
    half = int(round(0.4 * fs))
    prom = np.array([x[i] - min(x[max(0, i - half):i + 1].min(), x[i:i + half].min())
                     for i in p])
    from scipy.signal import find_peaks
    cand, _ = find_peaks(x, prominence=0.5 * float(np.median(prom)),
                         distance=max(1, int(round(0.30 * fs))))
    tol = int(round(0.15 * fs))
    undetected = int(sum(np.min(np.abs(p - c)) > tol for c in cand)) if p.size else 0

    good = float(np.mean(r >= r_min))
    out.update(good_fraction=round(good, 3), median_r=round(float(np.median(r)), 3),
               undetected_pulses=undetected,
               is_valid=bool(good >= min_good_fraction and undetected <= max_undetected))
    return out
