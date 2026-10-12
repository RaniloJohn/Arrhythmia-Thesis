"""
Unit Tests for Edge Runner Weights Loading & Consensus Tracking (Prompt 6 GATE)
==============================================================================
Tests:
1. Default weights loading: automatically loads cnn_af_v1.npz and meta.json.
2. Explicit path loading: loads custom weights path correctly.
3. Non-existent explicit path: raises FileNotFoundError.
4. ARRHYTHMIA_WEIGHTS environment variable resolution.
5. Missing weights fallback: logs warning, stays operational with random weights.
6. Telemetry payload schema: includes weights_loaded, decision_threshold, af_consensus.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from edge_inference.runner import EdgeInferenceRunner, SyntheticPPGGenerator


@pytest.fixture(autouse=True)
def _isolated_raw_capture(tmp_path, monkeypatch):
    """Keep the runner's AF auto-capture out of the real device_captures folder."""
    monkeypatch.setenv("RAW_CAPTURE_DIR", str(tmp_path / "captures"))


@pytest.fixture
def repo_weights_path():
    p = Path(__file__).resolve().parent.parent / "model" / "weights" / "cnn_af_v1.npz"
    assert p.exists(), f"Production weights file {p} must exist for testing."
    return p


def test_default_weights_loading(repo_weights_path):
    """Test that EdgeInferenceRunner loads cnn_af_v1.npz by default."""
    runner = EdgeInferenceRunner()
    assert runner.model.weights_loaded is True
    assert runner.model.weights_source == str(repo_weights_path.resolve())
    assert runner.model.threshold == pytest.approx(0.37, abs=0.01)


def test_explicit_weights_loading(repo_weights_path):
    """Test explicit weights_path parameter loading."""
    runner = EdgeInferenceRunner(weights_path=repo_weights_path)
    assert runner.model.weights_loaded is True
    assert runner.model.weights_source == str(repo_weights_path.resolve())
    assert runner.model.threshold == pytest.approx(0.37, abs=0.01)


def test_explicit_nonexistent_weights_raises():
    """Test that specifying an explicit missing weights path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        EdgeInferenceRunner(weights_path="non_existent_weights_12345.npz")


def test_env_var_weights_loading(repo_weights_path, monkeypatch):
    """Test loading via ARRHYTHMIA_WEIGHTS environment variable."""
    monkeypatch.setenv("ARRHYTHMIA_WEIGHTS", str(repo_weights_path.resolve()))
    runner = EdgeInferenceRunner()
    assert runner.model.weights_loaded is True
    assert runner.model.weights_source == str(repo_weights_path.resolve())


def test_missing_weights_fallback_warning(capsys, monkeypatch):
    """Test that missing weights triggers a loud warning banner but does not crash."""
    # Point default lookup to non-existent file
    fake_default = Path("non_existent_dir/weights.npz")
    monkeypatch.delenv("ARRHYTHMIA_WEIGHTS", raising=False)

    with patch("edge_inference.runner.Path.exists", return_value=False):
        runner = EdgeInferenceRunner(weights_path=None)
        captured = capsys.readouterr()
        assert "NO TRAINED WEIGHTS FILE FOUND" in captured.out
        assert "UNTRAINED" in captured.out
        assert runner.model.weights_loaded is False


def _prefilled_runner(**kwargs):
    runner = EdgeInferenceRunner(**kwargs)
    gen = SyntheticPPGGenerator(fs=100.0)
    for _ in range(1000):
        s = gen.next_sample()
        runner.raw_ir_buffer.append(s["ir_raw"])
        runner.raw_red_buffer.append(s["red_raw"])
        runner.timestamps_buffer.append(s["timestamp_ms"])
    runner.sample_count = 1000
    return runner


def test_telemetry_payload_schema_and_consensus(repo_weights_path):
    """
    Default classifier is the validated IBI model, so telemetry must identify *it*, not the
    1D-CNN. Updated 2026-10-01: this test previously asserted `cnn_af_v1`/`deepbeat`, which
    is the behaviour the audit found unsafe to ship — `model_trained` was true for a
    chance-level model.
    """
    runner = _prefilled_runner(weights_path=repo_weights_path)
    payload = runner.process_window()
    assert payload is not None
    for key in ("af_detected", "af_consensus", "af_probability", "weights_loaded",
                "model_trained", "model_version", "training_dataset",
                "decision_threshold", "classifier", "explanation",
                "model_validated_subject_auroc"):
        assert key in payload, f"missing telemetry key {key!r}"

    assert payload["classifier"] == "ibi"
    assert payload["weights_loaded"] is True
    assert payload["model_version"] == "ibi_af_v1"
    assert payload["training_dataset"] == "mimic_perform_af"
    assert payload["af_detected"] in (0, 1)
    assert payload["af_consensus"] in (0, 1)
    assert payload["latencies"]["inference_ms"] < 25.0

    # model_trained must mean "validated above a measured skill floor"
    assert payload["model_trained"] is True
    assert payload["model_validated_subject_auroc"] >= 0.70


def test_cnn_path_is_reported_as_unvalidated(repo_weights_path):
    """
    The 1D-CNN path stays available for comparison but must never claim to be validated,
    however loadable its weights are. This is the honesty regression the audit flagged.
    """
    runner = _prefilled_runner(weights_path=repo_weights_path, classifier="cnn")
    payload = runner.process_window()
    assert payload["classifier"] == "cnn"
    assert payload["model_trained"] is False
    assert payload["model_validated_subject_auroc"] is None
    assert payload["decision_threshold"] == pytest.approx(0.37, abs=0.01)


def test_too_few_beats_is_an_invalid_measurement_not_a_crash(repo_weights_path, monkeypatch):
    """
    Regression (2026-10-06, first run on real sensor data): a contacted window with only a
    couple of resolvable beats crashed the service with KeyError 'top_intervals', and would
    otherwise have classified an all-zero feature vector. It must be published as an
    invalid measurement, with no AF probability and no ledger event.
    """
    import numpy as np
    runner = _prefilled_runner(weights_path=repo_weights_path)
    monkeypatch.setattr(runner.peak_detector, "detect_peaks",
                        lambda filtered: np.array([100, 180, 260]))
    payload = runner.process_window()
    assert payload["measurement_valid"] is False
    assert payload["invalid_reason"] == "insufficient_beats"
    assert payload["af_probability"] is None
    assert payload["af_detected"] is None
    assert payload["event_id"] is None


def test_saturated_window_is_an_invalid_measurement(repo_weights_path):
    """
    Regression (2026-10-12): with the IR ADC pinned at full scale the detector still found
    'beats' and reported 25-50 bpm with a confident non-AF score. Must be invalid.
    """
    runner = _prefilled_runner(weights_path=repo_weights_path)
    runner.raw_ir_buffer = [262143] * 1000
    payload = runner.process_window()
    assert payload["measurement_valid"] is False
    assert payload["invalid_reason"] == "sensor_saturated"
    assert payload["af_probability"] is None
    assert payload["event_id"] is None


def test_unreliable_beats_are_not_classified(repo_weights_path, monkeypatch):
    """
    Regression (2026-10-12, false AF on three non-AF testers): split beats — the dicrotic
    wave detected as its own beat — make a regular rhythm look irregular to the IBI model.
    Peaks that do not look like the window's pulse must yield an invalid measurement,
    not an AF probability or a ledger event.
    """
    import numpy as np
    runner = _prefilled_runner(weights_path=repo_weights_path)
    real = runner.peak_detector.detect_peaks
    monkeypatch.setattr(runner.peak_detector, "detect_peaks",
                        lambda f: np.sort(np.concatenate([real(f), real(f)[:-1] + 35])))
    payload = runner.process_window()
    assert payload["measurement_valid"] is False
    assert payload["invalid_reason"] == "unreliable_beats"
    assert payload["af_probability"] is None
    assert payload["event_id"] is None
    assert "beat_morphology" in payload["sqi"]


def test_consensus_uses_non_overlapping_windows_and_counts_invalid_as_not_af(repo_weights_path):
    """
    3-of-5 must be taken over windows one window-length apart (5 x 10 s), as validated —
    not over 5 consecutive 1 s-stride windows that share 90% of their signal — and an
    invalid window must count against AF, not be skipped.
    """
    runner = _prefilled_runner(weights_path=repo_weights_path)
    W = runner.window_size

    def at(sample, flag):
        runner.sample_count = sample
        return runner._record_window_flag(flag)

    # Three AF windows one second apart: overlapping, so no consensus.
    base = 10 * W
    assert [at(base + i * 100, 1) for i in range(3)] == [0, 0, 0]
    # AF in three windows one window-length apart: consensus.
    runner.window_flags.clear()
    assert at(base, 1) == 0
    assert at(base + W, 1) == 0
    assert at(base + 2 * W, 1) == 1
    # Invalid windows count as not-AF. Skipping them would see three consecutive valid AF
    # windows here and alarm; counting them leaves 2 of the last 5.
    runner.window_flags.clear()
    at(base, 1); at(base + W, None); at(base + 2 * W, 1); at(base + 3 * W, None)
    at(base + 4 * W, None)
    assert at(base + 5 * W, 1) == 0


def test_single_af_window_writes_no_ledger_event(repo_weights_path, monkeypatch):
    """Regression (2026-10-12): every single AF window wrote a hash-chained ledger event."""
    runner = _prefilled_runner(weights_path=repo_weights_path)
    calls = []
    monkeypatch.setattr(runner.db, "record_event",
                        lambda **kw: calls.append(kw) or {"event_id": f"e{len(calls)}"})
    monkeypatch.setattr(runner.ibi_model, "predict_from_features", lambda f: (0.99, 1, 5.0))
    monkeypatch.setattr("edge_inference.runner.beat_morphology_sqi",
                        lambda *a, **k: {"is_valid": True, "good_fraction": 1.0,
                                         "undetected_pulses": 0})
    W = runner.window_size
    runner.sample_count = 10 * W
    p1 = runner.process_window()
    assert p1["af_detected"] == 1 and p1["af_consensus"] == 0
    assert p1["event_id"] is None and calls == []
    runner.sample_count += W; runner.process_window()
    runner.sample_count += W
    p3 = runner.process_window()
    assert p3["af_consensus"] == 1 and p3["event_id"] is not None and len(calls) == 1


def test_ppg_is_inverted_before_beat_detection(repo_weights_path, monkeypatch):
    """
    The MAX30102 reports reflected light, which dips with each pulse; the detector and the
    beat-shape gate expect systole-up (as in MIMIC). The DSP path must invert by default,
    while contact/saturation checks keep using raw counts.
    """
    import numpy as np
    runner = _prefilled_runner(weights_path=repo_weights_path)
    assert runner.invert_ppg is True
    seen = {}
    real = runner.peak_detector.detect_peaks
    monkeypatch.setattr(runner.peak_detector, "detect_peaks",
                        lambda f: seen.setdefault("f", f) is not None and real(f))
    payload = runner.process_window()
    raw = np.asarray(runner.raw_ir_buffer, dtype=float)
    # filtered signal must be anti-correlated with the raw counts
    assert np.corrcoef(seen["f"], raw - raw.mean())[0, 1] < -0.5
    assert payload.get("measurement_valid", True) is True
    assert payload["af_probability"] is not None


def test_af_call_saves_raw_window_once_per_window_length(repo_weights_path, monkeypatch, tmp_path):
    """Each AF call leaves its raw evidence on disk, rate-limited to non-overlapping windows."""
    import numpy as np
    runner = _prefilled_runner(weights_path=repo_weights_path)
    monkeypatch.setattr(runner.db, "record_event", lambda **kw: {"event_id": "e"})
    monkeypatch.setattr(runner.ibi_model, "predict_from_features", lambda f: (0.9, 1, 2.0))
    monkeypatch.setattr("edge_inference.runner.beat_morphology_sqi",
                        lambda *a, **k: {"is_valid": True, "good_fraction": 1.0,
                                         "undetected_pulses": 0})
    W = runner.window_size
    runner.sample_count = 10 * W
    runner.process_window()
    runner.sample_count += 100           # next 1 s-stride window: overlapping, not saved
    runner.process_window()
    files = sorted((tmp_path / "captures").glob("af_*.npz"))
    assert len(files) == 1
    z = np.load(files[0])
    assert z["ir"].shape == (W,) and z["af_probability"] == pytest.approx(0.9)
    assert bool(z["ppg_inverted"]) is True and z["peaks"].size > 0
    runner.sample_count += W             # a full window later: saved
    runner.process_window()
    assert len(list((tmp_path / "captures").glob("af_*.npz"))) == 2


def test_capture_failure_never_breaks_the_pipeline(repo_weights_path, monkeypatch):
    runner = _prefilled_runner(weights_path=repo_weights_path)
    monkeypatch.setattr(runner.ibi_model, "predict_from_features", lambda f: (0.9, 1, 2.0))
    monkeypatch.setattr("edge_inference.runner.beat_morphology_sqi",
                        lambda *a, **k: {"is_valid": True, "good_fraction": 1.0,
                                         "undetected_pulses": 0})
    monkeypatch.setattr("edge_inference.runner.np.savez_compressed",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    payload = runner.process_window()
    assert payload["af_detected"] == 1
