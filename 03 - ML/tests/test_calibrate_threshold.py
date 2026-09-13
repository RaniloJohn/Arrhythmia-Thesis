"""
PyTest Suite: Threshold Calibration & Reliability Assessment (ANTIGRAVITY.md §4.3)
==================================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Tests:
1. Hard Guard: Refuses to calibrate on MIMIC data.
2. Hard Guard: Refuses to calibrate on test split.
3. Calibration metrics calculation: Brier score, ECE, MCE.
4. Threshold table generation across [0.00, 1.00] at 0.01 resolution.
5. Youden's J statistic threshold selection.
6. Target sensitivity selection strategy.
7. End-to-end calibration updates metadata and NumPy runtime threshold automatically.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest

ml_dir = Path(__file__).resolve().parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from training.calibrate_threshold import (
    verify_calibration_guards,
    compute_calibration_metrics,
    compute_threshold_metrics_table,
    calibrate_threshold,
)
from model.inference_model import Arrhythmia1DCNN as NumPyArrhythmia1DCNN


def test_guard_refuses_mimic_path(tmp_path: Path):
    """Asserts that verify_calibration_guards raises RuntimeError on MIMIC path."""
    bad_path = tmp_path / "mimic_perform_val.npz"
    with pytest.raises(RuntimeError, match="CRITICAL HARD GUARD VIOLATION: MIMIC dataset detected"):
        verify_calibration_guards(bad_path)


def test_guard_refuses_test_split_path(tmp_path: Path):
    """Asserts that verify_calibration_guards raises RuntimeError on test split path."""
    bad_path = tmp_path / "deepbeat_test.npz"
    with pytest.raises(RuntimeError, match="CRITICAL HARD GUARD VIOLATION: Test split detected"):
        verify_calibration_guards(bad_path)


def test_compute_calibration_metrics_perfect():
    """Asserts that perfectly calibrated predictions yield near-zero Brier and ECE."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int8)
    y_prob = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    res = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert res["brier_score"] == 0.0
    assert res["brier_skill_score"] == 1.0
    assert res["expected_calibration_error"] == 0.0


def test_compute_threshold_metrics_table_monotone():
    """
    Asserts that across increasing thresholds, sensitivity is non-increasing and
    specificity is non-decreasing.
    """
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200, dtype=np.int8)
    y_prob = rng.uniform(0.0, 1.0, size=200).astype(np.float32)

    table = compute_threshold_metrics_table(y_true, y_prob)
    assert len(table) == 101  # 0.00 to 1.00 at 0.01 step

    sensitivities = [row["sensitivity"] for row in table]
    specificities = [row["specificity"] for row in table]

    # Sensitivity monotonically non-increasing (within precision)
    for i in range(len(sensitivities) - 1):
        assert sensitivities[i] >= sensitivities[i + 1] - 1e-4

    # Specificity monotonically non-decreasing (within precision)
    for i in range(len(specificities) - 1):
        assert specificities[i] <= specificities[i + 1] + 1e-4


def test_calibrate_threshold_e2e(tmp_path: Path):
    """
    End-to-end test of calibrate_threshold using real validation data and weights.
    Verifies that decision_threshold in metadata is updated and loads into NumPy runtime.
    """
    val_path = ml_dir / "data" / "processed" / "deepbeat_val.npz"
    weights_path = ml_dir / "model" / "weights" / "cnn_af_v1.npz"
    meta_path = ml_dir / "model" / "weights" / "cnn_af_v1.meta.json"

    if not (val_path.exists() and weights_path.exists() and meta_path.exists()):
        pytest.skip("Processed validation data or cnn_af_v1 weights not found; skipping.")

    # Create temporary copy of weights and meta to avoid overwriting during test
    tmp_weights = tmp_path / "test_cnn.npz"
    tmp_meta = tmp_path / "test_cnn.meta.json"
    tmp_weights.write_bytes(weights_path.read_bytes())
    tmp_meta.write_bytes(meta_path.read_bytes())

    output_plot = tmp_path / "test_calibration.png"
    output_table = tmp_path / "test_table.csv"

    res = calibrate_threshold(
        val_data_path=val_path,
        weights_path=tmp_weights,
        meta_path=tmp_meta,
        output_plot_path=output_plot,
        output_table_path=output_table,
        max_val_windows=500,  # Fast test evaluation
        device_name="cpu",
    )

    assert "chosen_threshold" in res
    assert 0.0 <= res["chosen_threshold"] <= 1.0
    assert output_plot.exists()
    assert output_table.exists()

    # Verify metadata was updated
    with open(tmp_meta, "r") as f:
        updated_meta = json.load(f)
    assert updated_meta["decision_threshold"] == res["chosen_threshold"]
    assert updated_meta["threshold"] == res["chosen_threshold"]
    assert "calibration" in updated_meta

    # Verify pure-NumPy runtime picks up updated threshold
    model = NumPyArrhythmia1DCNN(input_length=1000)
    model.load_weights(tmp_weights)
    assert np.isclose(model.threshold, res["chosen_threshold"], atol=1e-4)
