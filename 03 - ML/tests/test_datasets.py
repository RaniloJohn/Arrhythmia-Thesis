"""
Unit Tests for training/datasets/base.py, deepbeat.py, and mimic_perform.py.
===========================================================================
Tests loader interfaces, subject records, summary calculations, and strict error handling.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from training.datasets.base import PPGDatasetLoader, SubjectRecord
from training.datasets.deepbeat import DeepBeatLoader
from training.datasets.mimic_perform import MimicPerformAFLoader


def test_subject_record_validation():
    """Verify SubjectRecord validates types and attributes properly."""
    rec = SubjectRecord(
        subject_id="subj_001",
        signal=np.arange(100, dtype=np.float64),
        fs=125,
        label=1,
        quality=None,
        source_dataset="mimic_perform",
        native_window_s=None,
    )
    assert rec.signal.dtype == np.float32
    assert rec.signal.ndim == 1
    assert rec.fs == 125.0
    assert rec.label == 1

    # Invalid label
    with pytest.raises(ValueError):
        SubjectRecord(
            subject_id="bad",
            signal=np.ones(10),
            fs=100.0,
            label=2,
            quality=None,
            source_dataset="test",
            native_window_s=None,
        )

    # Invalid fs
    with pytest.raises(ValueError):
        SubjectRecord(
            subject_id="bad",
            signal=np.ones(10),
            fs=-5.0,
            label=0,
            quality=None,
            source_dataset="test",
            native_window_s=None,
        )


def test_mimic_perform_loader_real_data():
    """Verify MimicPerformAFLoader on real extracted MIMIC PERform AF dataset."""
    loader = MimicPerformAFLoader()
    stats = loader.summary()

    assert stats["total_subjects"] == 35
    assert stats["af_subjects"] == 19
    assert stats["non_af_subjects"] == 16
    assert "125.0 Hz" in stats["detected_sample_rate"]

    # Test iterating records
    count = 0
    af_count = 0
    for rec in loader.subjects():
        count += 1
        assert isinstance(rec, SubjectRecord)
        assert rec.fs == 125.0
        assert rec.signal.dtype == np.float32
        assert rec.source_dataset == "mimic_perform"
        assert rec.native_window_s is None
        if rec.label == 1:
            af_count += 1

    assert count == 35
    assert af_count == 19


def test_deepbeat_loader_missing_dir_raises():
    """Verify DeepBeatLoader raises actionable FileNotFoundError when empty/missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = DeepBeatLoader(data_dir=tmpdir)
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.summary()
        assert "No dataset files" in str(exc_info.value) or "not found" in str(exc_info.value)


def test_deepbeat_loader_with_mock_dataset():
    """Verify DeepBeatLoader metadata detection and partition surfacing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fs = 32.0
        win_s = 25.0
        n_samples = int(round(fs * win_s))  # 800 samples

        # Save metadata.json
        meta = {
            "sampling_rate": fs,
            "window_length_s": win_s,
            "description": "Mock DeepBeat dataset for unit testing",
        }
        with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # Create train.npz
        np.savez(
            tmp_path / "train.npz",
            signals=np.ones((4, n_samples), dtype=np.float32),
            labels=np.array([1, 0, 1, 0]),
            subject_ids=np.array(["p1", "p2", "p3", "p4"]),
            qualities=np.array([1.0, 0.9, 0.8, 0.95]),
        )

        # Create test_adjudicated.npz
        np.savez(
            tmp_path / "test_adjudicated.npz",
            signals=np.ones((2, n_samples), dtype=np.float32),
            labels=np.array([1, 0]),
            subject_ids=np.array(["p5", "p6"]),
            qualities=np.array([1.0, 1.0]),
        )

        loader = DeepBeatLoader(data_dir=tmp_path)
        stats = loader.summary()

        assert "32.0 Hz" in stats["detected_sample_rate"]
        assert "25.00 s" in stats["detected_window_length"]
        assert stats["total_windows"] == 6
        assert stats["total_unique_subjects"] == 6
        assert stats["af_windows"] == 3
        assert stats["non_af_windows"] == 3
        assert "test_adjudicated" in stats["cardiologist_adjudicated_partition"]


def test_deepbeat_loader_inconsistent_metadata_raises():
    """Verify DeepBeatLoader raises AssertionError if metadata does not match array shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fs = 32.0
        # Inconsistent: claims 10s but array has 800 samples (which is 25s at 32 Hz)
        meta = {
            "sampling_rate": fs,
            "window_length_s": 10.0,
        }
        with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        np.savez(
            tmp_path / "train.npz",
            signals=np.ones((2, 800), dtype=np.float32),
            labels=np.array([1, 0]),
        )

        loader = DeepBeatLoader(data_dir=tmp_path)
        with pytest.raises(AssertionError):
            loader.get_fs_and_window_length()


def test_deepbeat_loader_missing_fs_raises():
    """Verify DeepBeatLoader raises ValueError if fs cannot be determined (no guessing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        np.savez(
            tmp_path / "train.npz",
            signals=np.ones((2, 800), dtype=np.float32),
            labels=np.array([1, 0]),
        )

        loader = DeepBeatLoader(data_dir=tmp_path)
        with pytest.raises(ValueError) as exc_info:
            loader.get_fs_and_window_length()
        assert "MUST NOT be guessed" in str(exc_info.value)


def test_deepbeat_loader_real_data():
    """Verify DeepBeatLoader on actual data if available in repo."""
    real_data_dir = Path(__file__).resolve().parents[1] / "data" / "deepbeat"
    if not any(real_data_dir.glob("*.npz")):
        pytest.skip("Real DeepBeat data files not yet present.")

    loader = DeepBeatLoader(data_dir=real_data_dir)
    stats = loader.summary()

    assert "32.0 Hz" in stats["detected_sample_rate"]
    assert "25.00 s" in stats["detected_window_length"]
    assert stats["total_windows"] > 0
    assert stats["total_unique_subjects"] > 0
    assert stats["af_windows"] > 0


if __name__ == "__main__":
    test_subject_record_validation()
    test_mimic_perform_loader_real_data()
    test_deepbeat_loader_missing_dir_raises()
    test_deepbeat_loader_with_mock_dataset()
    test_deepbeat_loader_inconsistent_metadata_raises()
    test_deepbeat_loader_missing_fs_raises()
    test_deepbeat_loader_real_data()
    print("All dataset unit tests passed.")
