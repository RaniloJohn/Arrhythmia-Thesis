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


def test_telemetry_payload_schema_and_consensus(repo_weights_path):
    """Test that process_window emits full telemetry including consensus and weights state."""
    runner = EdgeInferenceRunner(weights_path=repo_weights_path)
    gen = SyntheticPPGGenerator(fs=100.0)

    # Prefill buffer to 1000 samples
    for _ in range(1000):
        s = gen.next_sample()
        runner.raw_ir_buffer.append(s["ir_raw"])
        runner.raw_red_buffer.append(s["red_raw"])
        runner.timestamps_buffer.append(s["timestamp_ms"])
    runner.sample_count = 1000

    payload = runner.process_window()
    assert payload is not None
    assert "af_detected" in payload
    assert "af_consensus" in payload
    assert "af_probability" in payload
    assert "weights_loaded" in payload
    assert "model_trained" in payload
    assert "model_version" in payload
    assert "training_dataset" in payload
    assert "decision_threshold" in payload

    assert payload["weights_loaded"] is True
    assert payload["model_trained"] is True
    assert payload["model_version"] == "cnn_af_v1"
    assert payload["training_dataset"] == "deepbeat"
    assert payload["decision_threshold"] == pytest.approx(0.37, abs=0.01)
    assert payload["af_detected"] in (0, 1)
    assert payload["af_consensus"] in (0, 1)
    assert payload["latencies"]["inference_ms"] < 25.0
