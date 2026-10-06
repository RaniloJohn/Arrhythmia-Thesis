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
