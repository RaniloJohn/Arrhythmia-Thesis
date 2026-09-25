"""
PyTest Suite: Training Loop & Hard Guards (ANTIGRAVITY.md §4.3)
================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Tests:
1. Hard Guard: Anti-MIMIC detection refuses to train if MIMIC appears in paths.
2. Hard Guard: Split manifest leakage triggers error.
3. Subsampling by subject preserves complete subject isolation.
4. Smoke run: Small end-to-end training execution produces valid history.csv and exported npz.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest
import torch

ml_dir = Path(__file__).resolve().parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from training.train import verify_hard_guards, subsample_by_subject, train
from model.inference_model import Arrhythmia1DCNN as NumPyArrhythmia1DCNN


def test_hard_guard_anti_mimic_path(tmp_path: Path):
    """Asserts that verify_hard_guards raises RuntimeError if any path contains 'mimic'."""
    mimic_path = tmp_path / "mimic_perform_train.npz"
    clean_val = tmp_path / "deepbeat_val.npz"
    clean_manifest = tmp_path / "deepbeat_split_manifest.json"

    mock_data = {"subject_id": ["101", "102"]}

    with pytest.raises(RuntimeError, match="CRITICAL HARD GUARD VIOLATION: MIMIC dataset detected"):
        verify_hard_guards(
            train_path=mimic_path,
            val_path=clean_val,
            manifest_path=clean_manifest,
            train_data=mock_data,
            val_data=mock_data,
        )


def test_hard_guard_manifest_overlap(tmp_path: Path):
    """Asserts that verify_hard_guards raises RuntimeError if split manifest contains overlapping subjects."""
    clean_train = tmp_path / "deepbeat_train.npz"
    clean_val = tmp_path / "deepbeat_val.npz"
    bad_manifest = tmp_path / "bad_split_manifest.json"

    # Write manifest with deliberate train/val subject leakage
    with open(bad_manifest, "w") as f:
        json.dump({
            "splits": {
                "train": ["101", "102", "103"],
                "val": ["103", "104"],
                "test": ["105"]
            }
        }, f)

    mock_train = {"subject_id": ["101", "102"]}
    mock_val = {"subject_id": ["104"]}

    with pytest.raises(RuntimeError, match="Split manifest contains subject leakage"):
        verify_hard_guards(
            train_path=clean_train,
            val_path=clean_val,
            manifest_path=bad_manifest,
            train_data=mock_train,
            val_data=mock_val,
        )


def test_subsample_by_subject_isolation():
    """Asserts that subsample_by_subject includes whole subjects only."""
    subjects = np.array(["101"] * 50 + ["102"] * 50 + ["103"] * 100 + ["104"] * 100)
    y = np.array([0] * 50 + [1] * 50 + [0] * 100 + [1] * 100)

    indices = subsample_by_subject(subjects, y, max_windows=120, seed=42)
    selected_subjs = set(subjects[indices])

    # Check that for any selected subject, ALL their windows are included
    for s in selected_subjs:
        expected_count = int(np.sum(subjects == s))
        actual_count = int(np.sum(subjects[indices] == s))
        assert actual_count == expected_count, f"Subject {s} was partially sliced!"

    # Ensure both AF and Non-AF represented
    selected_y = y[indices]
    assert np.sum(selected_y == 1) > 0, "No AF windows in subsample!"
    assert np.sum(selected_y == 0) > 0, "No non-AF windows in subsample!"


def test_train_smoke_run(tmp_path: Path):
    """
    End-to-end smoke test executing 2 training epochs on a capped subset of real DeepBeat data.
    Verifies history logging, best checkpoint creation, and export to pure-NumPy runtime.
    """
    train_npz = ml_dir / "data" / "processed" / "deepbeat_train.npz"
    val_npz = ml_dir / "data" / "processed" / "deepbeat_val.npz"
    manifest_path = ml_dir / "training" / "splits" / "deepbeat_split_manifest.json"

    if not (train_npz.exists() and val_npz.exists() and manifest_path.exists()):
        pytest.skip("Processed DeepBeat datasets not found; skipping training smoke run.")

    output_weights = tmp_path / "test_weights.npz"
    runs_dir = tmp_path / "runs"

    res = train(
        train_data_path=train_npz,
        val_data_path=val_npz,
        manifest_path=manifest_path,
        output_weights_path=output_weights,
        runs_dir=runs_dir,
        epochs=2,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        patience=2,
        max_train_windows=1000,
        max_val_windows=500,
        seed=42,
    )

    assert "best_val_auroc" in res
    assert Path(res["history_csv"]).exists()
    assert Path(res["output_npz"]).exists()
    assert Path(res["output_meta"]).exists()

    # Verify history.csv content
    with open(res["history_csv"], "r") as f:
        lines = f.readlines()
        assert len(lines) >= 3  # Header + 2 epochs
        assert "epoch,train_loss,val_loss,val_auroc" in lines[0]

    # Verify export loads cleanly into NumPy runtime
    np_model = NumPyArrhythmia1DCNN(input_length=1000)
    np_model.load_weights(res["output_npz"])
    assert np_model.weights_loaded is True
