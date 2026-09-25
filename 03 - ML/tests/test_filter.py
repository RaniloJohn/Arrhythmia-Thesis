"""
Unit Tests for DSP Filtering & Normalization Module
===================================================
Tests:
1. Butterworth passband and stopband frequency attenuation:
   - 0.2 Hz baseline drift attenuated (>10 dB)
   - 12.0 Hz high-frequency noise attenuated (>10 dB)
   - 1.2 Hz cardiac pulse signal survives with minimal attenuation (<3 dB)
2. Zero-phase preservation (filtfilt avoids phase shift/delay).
3. Detrending (removes linear trends and DC drift).
4. Z-score normalization output statistics (mean=0, std=1, zero-variance handling).
5. Minmax normalization output bounds [0.0, 1.0].
"""

import pytest
import numpy as np
from signal_processing.filter import (
    ButterBandpassFilter,
    detrend_ppg,
    zscore_normalize,
    minmax_normalize,
)


def test_butterworth_frequency_response():
    """
    Test that the Butterworth 0.5-5.0 Hz filter passes cardiac frequencies (1.2 Hz)
    while substantially attenuating out-of-band signals (0.2 Hz drift, 12 Hz noise).
    """
    fs = 100.0
    duration = 10.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    bp = ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=fs, order=4)

    # 1. Low frequency baseline wander (0.2 Hz)
    sig_low = np.sin(2 * np.pi * 0.2 * t).astype(np.float32)
    filt_low = bp.apply(sig_low)
    amp_in_low = np.max(sig_low[200:-200]) - np.min(sig_low[200:-200])
    amp_out_low = np.max(filt_low[200:-200]) - np.min(filt_low[200:-200])
    atten_low_db = 20 * np.log10(amp_out_low / amp_in_low)
    assert atten_low_db < -10.0, f"0.2 Hz attenuation insufficient: {atten_low_db:.1f} dB"

    # 2. In-band physiological pulse frequency (1.2 Hz = 72 BPM)
    sig_mid = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)
    filt_mid = bp.apply(sig_mid)
    amp_in_mid = np.max(sig_mid[200:-200]) - np.min(sig_mid[200:-200])
    amp_out_mid = np.max(filt_mid[200:-200]) - np.min(filt_mid[200:-200])
    atten_mid_db = 20 * np.log10(amp_out_mid / amp_in_mid)
    assert atten_mid_db > -3.0, f"1.2 Hz passband attenuated too much: {atten_mid_db:.1f} dB"

    # 3. High frequency EMG/motion noise (12.0 Hz)
    sig_high = np.sin(2 * np.pi * 12.0 * t).astype(np.float32)
    filt_high = bp.apply(sig_high)
    amp_in_high = np.max(sig_high[200:-200]) - np.min(sig_high[200:-200])
    amp_out_high = np.max(filt_high[200:-200]) - np.min(filt_high[200:-200])
    atten_high_db = 20 * np.log10(amp_out_high / amp_in_high)
    assert atten_high_db < -10.0, f"12.0 Hz attenuation insufficient: {atten_high_db:.1f} dB"


def test_zero_phase_alignment():
    """
    Test that zero-phase filtering preserves peak location in time without phase lag.
    """
    fs = 100.0
    t = np.linspace(0, 10.0, 1000, endpoint=False)
    bp = ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=fs, order=4)

    # Signal with a distinctive pulse peak at sample 500 (t = 5.0 s)
    sig = np.sin(2 * np.pi * 1.0 * t)
    filt = bp.apply(sig)

    # Locate peak in central period
    peak_in = np.argmax(sig[400:600]) + 400
    peak_out = np.argmax(filt[400:600]) + 400

    assert abs(peak_in - peak_out) <= 1, f"Peak shifted by {abs(peak_in - peak_out)} samples; not zero-phase"


def test_detrend_ppg():
    """Test that detrend_ppg removes linear drift."""
    n = 1000
    t = np.arange(n, dtype=np.float32)
    clean_wave = np.sin(2 * np.pi * 0.01 * t)
    drift = 0.5 * t + 100.0
    corrupted = clean_wave + drift

    detrended = detrend_ppg(corrupted)

    # Linear regression slope of detrended signal should be near 0
    slope, _ = np.polyfit(t, detrended, 1)
    assert abs(slope) < 1e-4
    assert np.mean(detrended) == pytest.approx(0.0, abs=1e-3)


def test_zscore_normalize():
    """Test that zscore_normalize produces mean=0 and std=1."""
    rng = np.random.default_rng(42)
    sig = rng.normal(50.0, 15.0, 1000).astype(np.float32)
    norm = zscore_normalize(sig)

    assert np.mean(norm) == pytest.approx(0.0, abs=1e-5)
    assert np.std(norm) == pytest.approx(1.0, abs=1e-5)

    # Constant signal (zero standard deviation) should not divide by zero or produce NaNs
    constant_sig = np.ones(500, dtype=np.float32) * 42.0
    norm_const = zscore_normalize(constant_sig)
    assert not np.isnan(norm_const).any()
    assert not np.isinf(norm_const).any()


def test_minmax_normalize():
    """Test scaling to [0.0, 1.0]."""
    sig = np.array([-10.0, 0.0, 20.0, 30.0], dtype=np.float32)
    res = minmax_normalize(sig)
    assert np.min(res) == pytest.approx(0.0, abs=1e-5)
    assert np.max(res) == pytest.approx(1.0, abs=1e-5)
