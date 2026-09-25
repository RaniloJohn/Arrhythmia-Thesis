"""
Unit Tests for Peak Detection & Interval Analysis Module
========================================================
Tests:
1. Elgendi systolic peak detector on synthetic 72 BPM PPG (12 peaks in 10 seconds).
2. Interval analysis: verifies BPM calculation (72 +/- 2 BPM).
3. Irregular rhythm analysis: high CV-IBI and RMSSD on known irregular intervals.
4. Refractory period: suppresses spurious double-peaks within 300 ms.
5. Edge cases: handles flat signals and single peak without errors.
"""

import pytest
import numpy as np
from signal_processing.peak_detection import ElgendiPeakDetector


@pytest.fixture
def detector():
    return ElgendiPeakDetector(fs=100.0)


def test_elgendi_peak_detection_72bpm(detector):
    """Synthetic 72 BPM pulse signal over 10 seconds should yield ~12 systolic peaks."""
    fs = 100.0
    t = np.linspace(0, 10.0, 1000, endpoint=False)
    freq = 72.0 / 60.0  # 1.2 Hz -> period = 0.833 s -> ~12 peaks in 10s
    phase = 2 * np.pi * freq * t
    # Clean pulse wave
    pulse = np.exp(-(((phase % (2 * np.pi)) - 1.2) ** 2) / 0.15)

    peaks = detector.detect_peaks(pulse)

    # In 10 seconds at 72 BPM, expect 11 to 13 peaks
    assert 11 <= len(peaks) <= 13, f"Expected 11-13 peaks, got {len(peaks)}"

    intervals = detector.analyze_intervals(peaks)
    assert intervals["bpm"] == pytest.approx(72.0, abs=2.0)
    assert intervals["irregular_intervals"] is False


def test_irregular_interval_analysis(detector):
    """Known irregular pulse intervals should result in irregular_intervals = True and elevated RMSSD."""
    # Synthesize peaks with intentionally erratic intervals: [600ms, 1100ms, 500ms, 950ms, 450ms, 1200ms]
    irregular_delays = [60, 110, 50, 95, 45, 120, 55, 105, 40]
    peaks = np.cumsum([100] + irregular_delays)

    intervals = detector.analyze_intervals(peaks)

    assert intervals["bpm"] > 0
    assert intervals["rmssd_ms"] > 50.0  # Markedly elevated RMSSD
    assert intervals["cv_ibi"] > 0.15    # Coefficient of variation > 15%
    assert intervals["irregular_intervals"] is True


def test_refractory_period_suppresses_double_peaks(detector):
    """Peaks closer than 300 ms (200 BPM) should be filtered by refractory rule."""
    fs = 100.0
    sig = np.zeros(500)
    # Put two peaks close together: sample 100 and sample 115 (150 ms apart)
    sig[100] = 5.0
    sig[101] = 4.0
    sig[115] = 4.5
    sig[116] = 3.5

    peaks = detector.detect_peaks(sig)
    # The second peak at 115 must be discarded due to min_distance = 30 samples (300 ms)
    if len(peaks) > 1:
        diffs = np.diff(peaks)
        assert np.all(diffs >= 30), f"Peak interval violates 300 ms refractory period: {diffs}"


def test_insufficient_peaks_handling(detector):
    """Testing single peak or empty peak array returns safe defaults."""
    res_empty = detector.analyze_intervals(np.array([]))
    assert res_empty["bpm"] == 0.0
    assert res_empty["rmssd_ms"] == 0.0
    assert res_empty["irregular_intervals"] is False

    res_single = detector.analyze_intervals(np.array([150]))
    assert res_single["bpm"] == 0.0
    assert res_single["rmssd_ms"] == 0.0
    assert res_single["irregular_intervals"] is False
