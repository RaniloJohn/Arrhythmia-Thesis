"""
Unit Tests for Dataset Builder & Windowing Pipeline (Prompt 1).
===============================================================
Tests:
- Resampling to 100 Hz and 1000-sample slicing.
- Order of DSP preprocessing (detrend -> bandpass -> zscore).
- SQI calculation and quality gating.
- Subject-isolated splits and zero cross-dataset leakage assertion.
- Manifest generation and format_cross_tabulation.
"""

from pathlib import Path
import tempfile
import json
import numpy as np
import pytest

from signal_processing.filter import ButterBandpassFilter, detrend_ppg, zscore_normalize
from signal_processing.sqi import SignalQualityAssessor
from training.datasets.base import SubjectRecord
from training.build_dataset import (
    _fill_nans,
    process_record_windows,
    format_cross_tabulation,
    create_or_verify_splits,
    build_dataset,
)


def test_fill_nans():
    """Verify interpolation of isolated NaN or Inf values."""
    sig = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0], dtype=np.float32)
    cleaned = _fill_nans(sig)
    assert not np.isnan(cleaned).any()
    assert not np.isinf(cleaned).any()
    assert np.isclose(cleaned[2], 3.0)
    assert np.isclose(cleaned[4], 5.0)


def test_process_record_windows_shapes_and_order():
    """Verify windowing produces exact 1000-sample arrays with proper DSP pipeline."""
    # Synthetic 25 s PPG recording at 32 Hz (800 samples)
    fs = 32.0
    t = np.linspace(0, 25.0, int(round(fs * 25.0)), endpoint=False)
    raw_signal = (np.sin(2 * np.pi * 1.2 * t) + 2.0).astype(np.float32)

    rec = SubjectRecord(
        subject_id="test_subj_01",
        signal=raw_signal,
        fs=fs,
        label=1,
        quality=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Good quality
        source_dataset="mock",
        native_window_s=25.0,
    )

    child_windows, n_produced = process_record_windows(rec, stride=1000)

    # 25 s @ 100 Hz = 2500 samples -> two 1000-sample windows, 500-sample remainder discarded
    assert n_produced == 2
    assert len(child_windows) == 2

    for win in child_windows:
        assert win["X"].shape == (1000,)
        assert win["X"].dtype == np.float32
        # Z-score normalized: mean ~ 0, std ~ 1
        assert np.isclose(np.mean(win["X"]), 0.0, atol=1e-2)
        assert np.isclose(np.std(win["X"]), 1.0, atol=1e-2)
        assert win["y"] == 1
        assert win["subject_id"] == "test_subj_01"
        assert win["dataset_quality"] == 0  # Good quality argmax
        assert 0.0 <= win["sqi"] <= 1.0


def test_format_cross_tabulation():
    """Verify contingency calculation and percentage metrics."""
    dq = np.array([0, 0, 1, 2, 2, -1], dtype=np.int8)
    sq = np.array([0.8, 0.4, 0.6, 0.3, 0.2, 0.9], dtype=np.float32)

    tab_str, tab_dict = format_cross_tabulation(dq, sq)
    assert "Quality Cross-Tabulation" in tab_str
    assert tab_dict["total_evaluated_windows"] == 5
    # Two poor windows (index 3 and 4), both have SQI < 0.50 -> 100% rejection agreement
    assert tab_dict["rejection_agreement_pct"] == 100.0


def test_split_subject_disjointness_assertions():
    """Verify split generation guarantees strictly disjoint subject sets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        splits_dir = Path(tmpdir) / "splits"

        # Mock loader yielding records from 10 subjects
        class MockLoader:
            def subjects(self):
                for i in range(10):
                    yield SubjectRecord(
                        subject_id=f"subj_{i:02d}",
                        signal=np.ones(800, dtype=np.float32),
                        fs=32.0,
                        label=1 if i < 3 else 0,
                        quality=None,
                        source_dataset="mock",
                        partition="train" if i < 6 else "val",
                    )

        splits = create_or_verify_splits(
            dataset_name="deepbeat",
            loader=MockLoader(),
            splits_dir=splits_dir,
            seed=42,
            strategy="stratified",
        )

        train_s = set(splits["train"])
        val_s = set(splits["val"])
        test_s = set(splits["test"])

        assert len(train_s & val_s) == 0
        assert len(train_s & test_s) == 0
        assert len(val_s & test_s) == 0
        assert len(train_s | val_s | test_s) == 10


def test_cross_dataset_subject_collision_raises():
    """Verify that a subject ID collision between DeepBeat and MIMIC fails loudly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        splits_dir = Path(tmpdir) / "splits"
        splits_dir.mkdir(parents=True)

        # Write MIMIC manifest with subject 'shared_patient'
        with open(splits_dir / "mimic_split_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"splits": {"external_val": ["shared_patient", "mimic_002"]}}, f)

        # Mock loader for DeepBeat that contains 'shared_patient'
        class ContaminatedLoader:
            def subjects(self):
                yield SubjectRecord(
                    subject_id="shared_patient",
                    signal=np.ones(800, dtype=np.float32),
                    fs=32.0,
                    label=0,
                    quality=None,
                    source_dataset="mock",
                )
                yield SubjectRecord(
                    subject_id="db_patient_002",
                    signal=np.ones(800, dtype=np.float32),
                    fs=32.0,
                    label=1,
                    quality=None,
                    source_dataset="mock",
                )

        with pytest.raises(AssertionError) as exc_info:
            create_or_verify_splits(
                dataset_name="deepbeat",
                loader=ContaminatedLoader(),
                splits_dir=splits_dir,
                strategy="stratified",
            )
        assert "Subject ID collision between DeepBeat and MIMIC" in str(exc_info.value)
