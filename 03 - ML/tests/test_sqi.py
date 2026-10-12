"""
Unit Tests for Signal Quality Index (SQI) Module
================================================
Tests:
1. Clean synthetic physiological PPG passes SQI assessment (score >= 0.5, is_valid = True).
2. Flat / DC-only signal is rejected (zero AC perfusion).
3. Motion-corrupted signal is rejected (severe skewness/kurtosis anomaly).
4. Probe-off condition (DC baseline < 1000) fires and is marked invalid.
5. Empty input handling.
"""

import pytest
import numpy as np
from signal_processing.sqi import SignalQualityAssessor
from signal_processing.filter import ButterBandpassFilter


@pytest.fixture
def sqi_assessor():
    return SignalQualityAssessor(min_pi=0.15, min_sqi=0.50)


@pytest.fixture
def bp_filter():
    return ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=100.0, order=4)


def test_clean_synthetic_ppg_scores_high(sqi_assessor, bp_filter):
    """Clean synthetic PPG with realistic pulse morphology should be marked valid."""
    fs = 100.0
    t = np.linspace(0, 10.0, 1000)
    # Realistic PPG: DC ~ 50,000, AC amplitude ~ 1,500 with dicrotic wave
    hr_freq = 1.2  # 72 BPM
    phase = 2 * np.pi * hr_freq * t
    systolic = np.exp(-(((phase % (2 * np.pi)) - 1.2) ** 2) / 0.15)
    dicrotic = 0.35 * np.exp(-(((phase % (2 * np.pi)) - 2.3) ** 2) / 0.25)
    ac = (systolic + dicrotic) * 1500.0
    raw = 50000.0 + ac + np.random.normal(0, 10, len(t))
    filtered = bp_filter.apply(ac)

    res = sqi_assessor.compute_metrics(raw, filtered)

    assert res["is_valid"] is True
    assert res["sqi_score"] >= 0.50
    assert res["perfusion_index"] > 0.15


def test_flat_dc_only_rejected(sqi_assessor, bp_filter):
    """A flat line with no pulsatile activity should have zero PI and be rejected."""
    raw = np.ones(1000) * 50000.0
    filtered = bp_filter.apply(raw)

    res = sqi_assessor.compute_metrics(raw, filtered)

    assert res["is_valid"] is False
    assert res["perfusion_index"] == 0.0


def test_motion_corrupted_trace_rejected(sqi_assessor, bp_filter):
    """A trace with massive erratic noise / motion artifacts should fail."""
    # A sharp asymmetric baseline jump / motion artifact
    raw = np.ones(1000) * 50000.0
    raw[500:520] += 50000.0
    filtered = bp_filter.apply(raw - 50000.0)

    res = sqi_assessor.compute_metrics(raw, filtered)

    # Severe skewness and kurtosis penalize composite score below acceptance threshold
    assert res["is_valid"] is False
    assert res["sqi_score"] < 0.50


def test_probe_off_detection(sqi_assessor, bp_filter):
    """Probe-off / disconnected sensor (DC < 1000) must fire probe-off branch."""
    raw = np.ones(1000) * 350.0  # DC < 1000 threshold
    filtered = bp_filter.apply(raw)

    res = sqi_assessor.compute_metrics(raw, filtered)

    assert res["is_valid"] is False
    assert res["perfusion_index"] == 0.0


def test_empty_input_handling(sqi_assessor):
    """Empty inputs should return invalid metrics without crashing."""
    res = sqi_assessor.compute_metrics(np.array([]), np.array([]))
    assert res["is_valid"] is False
    assert res["sqi_score"] == 0.0


def test_baseline_drift_does_not_pass_weak_pulse(sqi_assessor, bp_filter):
    """
    Regression (device false-AF, 2026-10-06/12): a 30-count pulse on ~100k DC is a 0.03%
    perfusion index and must be rejected, even when the finger drifts by 1,000 counts over
    the window. Raw max-min PI counted the drift as pulse and passed this window.
    """
    from signal_processing.filter import detrend_ppg
    t = np.arange(1000) / 100.0
    rng = np.random.default_rng(0)
    raw = 100000.0 + 15.0 * np.sin(2 * np.pi * 1.2 * t) + 100.0 * t + rng.normal(0, 5, 1000)
    filtered = bp_filter.apply(detrend_ppg(raw))

    res = sqi_assessor.compute_metrics(raw, filtered)

    assert res["perfusion_index"] < 0.15
    assert res["is_valid"] is False


# ---------------------------------------------------------------- beat-morphology gate
from signal_processing.sqi import beat_morphology_sqi, is_saturated
from signal_processing.peak_detection import ElgendiPeakDetector


def _ppg_from_beats(beat_times_s, n=1000, fs=100.0):
    """Systolic + dicrotic Gaussian pulse at each beat onset; amplitude varies per beat."""
    t = np.arange(n) / fs
    x = np.zeros(n)
    rng = np.random.default_rng(3)
    for b in beat_times_s:
        a = rng.uniform(0.6, 1.0)  # AF has beat-to-beat amplitude variation too
        x += a * (np.exp(-((t - b - 0.15) / 0.06) ** 2) + 0.4 * np.exp(-((t - b - 0.42) / 0.09) ** 2))
    return x


def _beats(ibis_s, start=0.2):
    return start + np.concatenate([[0.0], np.cumsum(ibis_s)])


def test_morphology_gate_keeps_regular_rhythm(bp_filter):
    x = bp_filter.apply(_ppg_from_beats(_beats([0.83] * 11)))
    peaks = ElgendiPeakDetector().detect_peaks(x)
    assert beat_morphology_sqi(x, peaks)["is_valid"] is True


def test_morphology_gate_keeps_irregular_af_like_rhythm(bp_filter):
    """The gate must not reject a window for irregular timing — that would reject real AF."""
    rng = np.random.default_rng(7)
    ibis = rng.uniform(0.45, 1.10, size=14)
    x = bp_filter.apply(_ppg_from_beats(_beats(ibis)))
    peaks = ElgendiPeakDetector().detect_peaks(x)
    res = beat_morphology_sqi(x, peaks)
    assert res["is_valid"] is True, res


def test_morphology_gate_rejects_split_beats(bp_filter):
    """Regression (device, 2026-10-12): the dicrotic wave counted as a second beat."""
    beats = _beats([0.83] * 11)
    x = bp_filter.apply(_ppg_from_beats(beats))
    true_peaks = ElgendiPeakDetector().detect_peaks(x)
    dicrotic = (np.asarray(beats) * 100 + 42).astype(int)
    peaks = np.sort(np.concatenate([true_peaks, dicrotic[dicrotic < 990]]))
    assert beat_morphology_sqi(x, peaks)["is_valid"] is False


def test_morphology_gate_rejects_missed_beats(bp_filter):
    x = bp_filter.apply(_ppg_from_beats(_beats([0.83] * 11)))
    peaks = ElgendiPeakDetector().detect_peaks(x)
    res = beat_morphology_sqi(x, peaks[::2])  # every other pulse undetected
    assert res["undetected_pulses"] >= 2
    assert res["is_valid"] is False


def test_morphology_gate_needs_four_beats(bp_filter):
    x = bp_filter.apply(_ppg_from_beats(_beats([0.83] * 11)))
    assert beat_morphology_sqi(x, np.array([100, 183, 266]))["is_valid"] is False


def test_saturation_detection():
    assert is_saturated(np.full(1000, 262143.0)) is True
    assert is_saturated(np.r_[np.full(999, 105000.0), 262100.0]) is True
    assert is_saturated(np.full(1000, 105000.0)) is False
    assert is_saturated(np.array([])) is False
