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
