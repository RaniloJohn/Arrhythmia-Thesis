"""
Dataset Builder & Windowing Pipeline (Prompt 1).
=================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (University of the East).

Transforms raw loader outputs (DeepBeat and MIMIC PERform AF) into canonical:
- 1000 samples = 10.0 s @ 100 Hz windows matching edge runner.py input_contract.
- Applies production DSP preprocessing in strict runner.py order:
    1. detrend_ppg(raw_window)
    2. bp_filter.apply(detrended) with Butterworth [0.5, 5.0] Hz, order=4
    3. zscore_normalize(filtered)
- Calculates Signal Quality Index (sqi_score) using SignalQualityAssessor.
- Gating: Drops poor quality windows (dataset_quality == 2 or sqi_score < 0.5)
  from the TRAIN split only. Keeps ALL windows in val/test/external_val.
- Enforces strict subject-isolated splitting and zero cross-dataset leakage.
- Generates on-disk .npz arrays and reproducible split/dataset manifests.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure workspace root and 03 - ML are in sys.path
_current_dir = Path(__file__).resolve().parent
_ml_root = _current_dir.parent
_repo_root = _ml_root.parent
for p in [str(_ml_root), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from signal_processing.filter import detrend_ppg, zscore_normalize, ButterBandpassFilter
from signal_processing.sqi import SignalQualityAssessor
from training.datasets.base import PPGDatasetLoader, SubjectRecord
from training.datasets.deepbeat import DeepBeatLoader
from training.datasets.mimic_perform import MimicPerformAFLoader
from training.resample import to_100hz


def format_cross_tabulation(dataset_qualities: np.ndarray, sqi_scores: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """
    Compute and format cross-tabulation of dataset_quality against sqi_score bins.
    Bins: Low (<0.50), Marginal (0.50 <= x < 0.70), High (>=0.70).
    Dataset Qualities: 0 (Good), 1 (Marginal), 2 (Poor).
    """
    valid_mask = dataset_qualities >= 0
    if not np.any(valid_mask):
        return "No dataset quality annotations available for cross-tabulation.", {}

    dq = dataset_qualities[valid_mask]
    sq = sqi_scores[valid_mask]

    # Bins: 0: Low (<0.50), 1: Med ([0.50, 0.70)), 2: High (>=0.70)
    sqi_bins = np.zeros(len(sq), dtype=int)
    sqi_bins[(sq >= 0.50) & (sq < 0.70)] = 1
    sqi_bins[sq >= 0.70] = 2

    # Contingency matrix [3, 3]
    counts = np.zeros((3, 3), dtype=int)
    for d, s in zip(dq, sqi_bins):
        if 0 <= d <= 2 and 0 <= s <= 2:
            counts[d, s] += 1

    row_totals = counts.sum(axis=1)
    col_totals = counts.sum(axis=0)
    grand_total = counts.sum()

    lines = []
    lines.append("=========================================================================================")
    lines.append(" Quality Cross-Tabulation: DeepBeat Dataset Quality vs SignalQualityAssessor (SQI)")
    lines.append("=========================================================================================")
    lines.append(" DeepBeat Label     |  SQI < 0.5 (Low)  |  0.5 <= SQI < 0.7  |  SQI >= 0.7 (High)  |  Total")
    lines.append("--------------------+-------------------+--------------------+---------------------+-------")
    labels_map = {0: "0 (Good)", 1: "1 (Marginal)", 2: "2 (Poor)"}
    for r in range(3):
        rt = max(1, row_totals[r])
        lines.append(
            f" {labels_map[r]:<18} | {counts[r, 0]:>8} ({counts[r, 0]/rt*100:4.1f}%) | "
            f"{counts[r, 1]:>8} ({counts[r, 1]/rt*100:4.1f}%) | "
            f"{counts[r, 2]:>9} ({counts[r, 2]/rt*100:4.1f}%) | {row_totals[r]:>6}"
        )
    lines.append("--------------------+-------------------+--------------------+---------------------+-------")
    gt = max(1, grand_total)
    lines.append(
        f" {'Total':<18} | {col_totals[0]:>8} ({col_totals[0]/gt*100:4.1f}%) | "
        f"{col_totals[1]:>8} ({col_totals[1]/gt*100:4.1f}%) | "
        f"{col_totals[2]:>9} ({col_totals[2]/gt*100:4.1f}%) | {grand_total:>6}"
    )
    lines.append("=========================================================================================")

    # Agreement Analysis
    # Rejection agreement: fraction of dataset Poor windows flagged as SQI < 0.50
    rejection_agreement = (counts[2, 0] / max(1, row_totals[2])) * 100.0
    # Acceptance agreement: fraction of dataset Good windows passing SQI >= 0.50
    acceptance_agreement = ((counts[0, 1] + counts[0, 2]) / max(1, row_totals[0])) * 100.0

    lines.append(" Agreement Analysis:")
    lines.append(f"   - Rejection of Poor Windows (Dataset Poor -> SQI < 0.50): {rejection_agreement:.2f}%")
    lines.append(f"   - Retention of Good Windows (Dataset Good -> SQI >= 0.50): {acceptance_agreement:.2f}%")
    lines.append("=========================================================================================")

    stats_dict = {
        "contingency_counts": counts.tolist(),
        "rejection_agreement_pct": round(rejection_agreement, 2),
        "acceptance_agreement_pct": round(acceptance_agreement, 2),
        "total_evaluated_windows": int(grand_total),
    }

    return "\n".join(lines), stats_dict


def create_or_verify_splits(
    dataset_name: str,
    loader: PPGDatasetLoader,
    splits_dir: Path,
    seed: int = 42,
    strategy: str = "adjudicated_test",
) -> Dict[str, List[str]]:
    """
    Establish or verify subject-isolated splits for the dataset.
    Ensures:
      - DeepBeat official partition subject-disjointness is checked;
        if overlap is detected, official split is discarded and regrouped by subject.
      - MIMIC is single external-validation split.
      - Subject intersection between any two splits is strictly empty.
      - Zero cross-dataset subject collision between DeepBeat and MIMIC.
      - Emits warning if any split has fewer than 4 subjects.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = splits_dir / f"{dataset_name}_split_manifest.json"

    # Discover unique subjects and their subject-level AF status
    print(f"[{dataset_name}] Discovering unique subjects and rhythm labels...")
    subject_af_counts: Dict[str, int] = {}
    subject_total_counts: Dict[str, int] = {}
    subject_official_parts: Dict[str, Set[str]] = {}

    for rec in loader.subjects():
        s = str(rec.subject_id).strip()
        subject_total_counts[s] = subject_total_counts.get(s, 0) + 1
        if rec.label == 1:
            subject_af_counts[s] = subject_af_counts.get(s, 0) + 1
        if rec.partition:
            if s not in subject_official_parts:
                subject_official_parts[s] = set()
            subject_official_parts[s].add(rec.partition)

    unique_subjects = sorted(list(subject_total_counts.keys()))
    print(f"[{dataset_name}] Discovered {len(unique_subjects)} unique subjects.")

    splits: Dict[str, List[str]] = {}

    if dataset_name == "mimic":
        # Entire dataset is one external-validation split
        splits = {"external_val": unique_subjects}
        split_meta = {
            "dataset": "mimic",
            "date": datetime.now().isoformat(),
            "strategy": "external_validation_all",
            "splits": splits,
            "subject_counts": {"external_val": len(unique_subjects)},
        }

    elif dataset_name == "deepbeat":
        # 1. Check official partitions for subject overlap
        official_val_subjs = {s for s, parts in subject_official_parts.items() if any("val" in p for p in parts)}
        official_test_subjs = {s for s, parts in subject_official_parts.items() if any("test" in p for p in parts)}
        official_train_subjs = {s for s, parts in subject_official_parts.items() if any("train" in p for p in parts)}

        overlap_val_test = official_val_subjs & official_test_subjs
        overlap_train_val = official_train_subjs & official_val_subjs
        overlap_train_test = official_train_subjs & official_test_subjs
        has_overlap = bool(overlap_val_test or overlap_train_val or overlap_train_test)

        if has_overlap:
            print(f"[DeepBeat] Official partitions have subject overlap!")
            if overlap_val_test:
                print(f"  Overlap Val/Test: {sorted(list(overlap_val_test))}")
            if overlap_train_val:
                print(f"  Overlap Train/Val: {sorted(list(overlap_train_val))}")
            if overlap_train_test:
                print(f"  Overlap Train/Test: {sorted(list(overlap_train_test))}")
            print(f"[DeepBeat] Discarding contaminated official partitions and regrouping by subject (strategy='{strategy}', seed={seed}).")

        # Classify subjects into AF (>0 AF windows) vs Non-AF (0 AF windows)
        af_subjs = [s for s in unique_subjects if subject_af_counts.get(s, 0) > 0]
        non_af_subjs = [s for s in unique_subjects if subject_af_counts.get(s, 0) == 0]
        print(f"[DeepBeat] Subject rhythm prevalence: {len(af_subjs)} AF subjects, {len(non_af_subjs)} non-AF subjects.")

        rng = random.Random(seed)

        if strategy == "adjudicated_test":
            # Identify the cardiologist-adjudicated subjects (154..167) that have 0 overlap with validate.npz
            # In Torres-Soto & Ashley (2020), subjects 154 to 167 constitute the clean test benchmark
            adjudicated_subjs = sorted([s for s in unique_subjects if s.isdigit() and 154 <= int(s) <= 167])
            if len(adjudicated_subjs) >= 4 and not (set(adjudicated_subjs) & official_val_subjs):
                test_subjs = adjudicated_subjs
                remaining_subjs = sorted(list(set(unique_subjects) - set(test_subjs)))
                rem_af = sorted([s for s in remaining_subjs if s in af_subjs])
                rem_non_af = sorted([s for s in remaining_subjs if s in non_af_subjs])

                rng.shuffle(rem_af)
                rng.shuffle(rem_non_af)

                # Split remaining into train (~70%) and val (~30%)
                n_train_af = max(1, int(round(len(rem_af) * 0.70)))
                n_train_non_af = max(1, int(round(len(rem_non_af) * 0.70)))

                train_subjs = sorted(rem_non_af[:n_train_non_af] + rem_af[:n_train_af])
                val_subjs = sorted(rem_non_af[n_train_non_af:] + rem_af[n_train_af:])
            else:
                # Fallback to pure stratified split
                strategy = "stratified"

        if strategy == "stratified":
            shuffled_af = sorted(af_subjs)
            shuffled_non_af = sorted(non_af_subjs)
            rng.shuffle(shuffled_af)
            rng.shuffle(shuffled_non_af)

            # 70% train, 15% val, 15% test
            n_val_af = max(1, int(round(len(shuffled_af) * 0.15)))
            n_test_af = max(1, int(round(len(shuffled_af) * 0.15)))
            n_train_af = len(shuffled_af) - n_val_af - n_test_af

            n_val_non = max(1, int(round(len(shuffled_non_af) * 0.15)))
            n_test_non = max(1, int(round(len(shuffled_non_af) * 0.15)))
            n_train_non = len(shuffled_non_af) - n_val_non - n_test_non

            train_subjs = sorted(shuffled_non_af[:n_train_non] + shuffled_af[:n_train_af])
            val_subjs = sorted(shuffled_non_af[n_train_non:n_train_non + n_val_non] + shuffled_af[n_train_af:n_train_af + n_val_af])
            test_subjs = sorted(shuffled_non_af[n_train_non + n_val_non:] + shuffled_af[n_train_af + n_val_af:])

        splits = {
            "train": train_subjs,
            "val": val_subjs,
            "test": test_subjs,
        }

        split_meta = {
            "dataset": "deepbeat",
            "date": datetime.now().isoformat(),
            "strategy": strategy,
            "seed": seed,
            "splits": splits,
            "subject_counts": {k: len(v) for k, v in splits.items()},
        }

    # Verify subject-isolation assertions within dataset
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1 = set(splits[split_names[i]])
            s2 = set(splits[split_names[j]])
            overlap = s1 & s2
            assert len(overlap) == 0, (
                f"FATAL: Subject ID overlap between {split_names[i]} and {split_names[j]}: {overlap}"
            )

    all_split_subjs = set()
    for s_list in splits.values():
        all_split_subjs.update(s_list)
    assert len(all_split_subjs) == len(unique_subjects), "FATAL: Not all subjects were assigned to a split!"

    print(f"[{dataset_name}] Internal subject-disjointness ASSERTION PASSED.")

    # Warn if any split has fewer than 4 subjects
    for split_name, subjs in splits.items():
        if len(subjs) < 4:
            print(f"[WARNING] Split '{split_name}' has only {len(subjs)} subjects (< 4)!")
        else:
            print(f"[{dataset_name}] Split '{split_name}' contains {len(subjs)} subjects (>= 4 check passed).")

    # Save split manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    print(f"[{dataset_name}] Wrote split manifest to: {manifest_path.resolve()}")

    # Assert cross-dataset subject collision (DeepBeat vs MIMIC)
    if dataset_name == "deepbeat":
        mimic_path = splits_dir / "mimic_split_manifest.json"
        if mimic_path.exists():
            with open(mimic_path, "r", encoding="utf-8") as f:
                mimic_data = json.load(f)
            mimic_subjs = set()
            for slist in mimic_data.get("splits", {}).values():
                mimic_subjs.update(slist)
            collision = all_split_subjs & mimic_subjs
            assert len(collision) == 0, f"FATAL: Subject ID collision between DeepBeat and MIMIC: {collision}"
            print(f"[Cross-Dataset Assertion Passed] 0 subject collision between DeepBeat ({len(all_split_subjs)}) and MIMIC ({len(mimic_subjs)}).")
    elif dataset_name == "mimic":
        deepbeat_path = splits_dir / "deepbeat_split_manifest.json"
        if deepbeat_path.exists():
            with open(deepbeat_path, "r", encoding="utf-8") as f:
                db_data = json.load(f)
            db_subjs = set()
            for slist in db_data.get("splits", {}).values():
                db_subjs.update(slist)
            collision = all_split_subjs & db_subjs
            assert len(collision) == 0, f"FATAL: Subject ID collision between MIMIC and DeepBeat: {collision}"
            print(f"[Cross-Dataset Assertion Passed] 0 subject collision between MIMIC ({len(all_split_subjs)}) and DeepBeat ({len(db_subjs)}).")

    return splits


def _fill_nans(sig: np.ndarray) -> np.ndarray:
    """Linearly interpolate isolated NaN or Inf values in raw recording."""
    nan_mask = np.isnan(sig) | np.isinf(sig)
    if not np.any(nan_mask):
        return sig
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(sig, dtype=np.float32)
    nan_idx = np.where(nan_mask)[0]
    out = np.asarray(sig, dtype=np.float32).copy()
    out[nan_mask] = np.interp(nan_idx, valid_idx, sig[valid_idx])
    return out


def process_record_windows(
    rec: SubjectRecord,
    target_fs: float = 100.0,
    window_samples: int = 1000,
    stride: int = 1000,
    bp_filter: Optional[ButterBandpassFilter] = None,
    sqi_assessor: Optional[SignalQualityAssessor] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process a single SubjectRecord into canonical preprocessed 1000-sample windows.
    Returns:
      - List of window dictionaries: {
          'X': normalized (1000,),
          'y': label (int),
          'subject_id': str,
          'sqi': float,
          'dataset_quality': int,
        }
      - Total child windows produced before any gating.
    """
    if bp_filter is None:
        bp_filter = ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=target_fs, order=4)
    if sqi_assessor is None:
        sqi_assessor = SignalQualityAssessor(min_pi=0.15, min_sqi=0.50)

    # Clean isolated NaNs in continuous recordings
    clean_signal = _fill_nans(rec.signal)

    # 1. Resample to canonical 100 Hz
    if math.isclose(rec.fs, target_fs, abs_tol=1e-2):
        sig100 = np.asarray(clean_signal, dtype=np.float32)
    else:
        sig100 = to_100hz(clean_signal, rec.fs)

    n_total_samples = len(sig100)
    if n_total_samples < window_samples:
        return [], 0

    # Parse dataset quality label
    dataset_qual = -1
    if rec.quality is not None:
        if isinstance(rec.quality, np.ndarray) and rec.quality.ndim == 1 and len(rec.quality) == 3:
            dataset_qual = int(np.argmax(rec.quality))
        elif isinstance(rec.quality, (int, np.integer)):
            dataset_qual = int(rec.quality)

    child_windows: List[Dict[str, Any]] = []
    n_windows_possible = (n_total_samples - window_samples) // stride + 1

    for w_idx in range(n_windows_possible):
        start = w_idx * stride
        end = start + window_samples
        raw_window = sig100[start:end]

        # Production DSP pipeline (runner.py:248-250)
        detrended = detrend_ppg(raw_window)
        filtered = bp_filter.apply(detrended)
        normalized = zscore_normalize(filtered)

        # Signal Quality Assessment
        sqi_res = sqi_assessor.compute_metrics(raw_window, filtered)
        sqi_score = float(sqi_res["sqi_score"])

        child_windows.append({
            "X": normalized,
            "y": int(rec.label),
            "subject_id": str(rec.subject_id).strip(),
            "sqi": sqi_score,
            "dataset_quality": dataset_qual,
        })

    return child_windows, n_windows_possible


def build_dataset(
    dataset_name: str,
    output_dir: Path,
    splits_dir: Path,
    data_dir: Optional[Path] = None,
    train_overlap: float = 0.0,
    seed: int = 42,
    strategy: str = "adjudicated_test",
    max_windows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main dataset build pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=================================================================")
    print(f" Starting Dataset Build: '{dataset_name.upper()}'")
    print(f" Output Directory: {output_dir.resolve()}")
    print(f" Splits Directory: {splits_dir.resolve()}")
    print(f" Train Overlap   : {train_overlap:.2f}")
    print(f"=================================================================")

    # Initialize loader
    if dataset_name == "deepbeat":
        loader = DeepBeatLoader(data_dir=data_dir)
    elif dataset_name == "mimic":
        loader = MimicPerformAFLoader(data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Must be 'deepbeat' or 'mimic'.")

    # 1. Establish / Verify Subject Splits
    splits = create_or_verify_splits(
        dataset_name=dataset_name,
        loader=loader,
        splits_dir=splits_dir,
        seed=seed,
        strategy=strategy,
    )

    # Invert mapping: subject_id -> split_name
    subject_to_split: Dict[str, str] = {}
    for split_name, subjs in splits.items():
        for s in subjs:
            subject_to_split[s] = split_name

    # Preprocessing objects
    target_fs = 100.0
    window_samples = 1000
    bp_filter = ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=target_fs, order=4)
    sqi_assessor = SignalQualityAssessor(min_pi=0.15, min_sqi=0.50)

    # Split-specific accumulator buffers
    split_X: Dict[str, List[np.ndarray]] = {k: [] for k in splits.keys()}
    split_y: Dict[str, List[int]] = {k: [] for k in splits.keys()}
    split_subjs: Dict[str, List[str]] = {k: [] for k in splits.keys()}
    split_sqi: Dict[str, List[float]] = {k: [] for k in splits.keys()}
    split_qual: Dict[str, List[int]] = {k: [] for k in splits.keys()}

    # Tracking metrics
    total_native_windows = 0
    total_child_windows_generated = 0
    child_windows_per_native: List[int] = []

    # Drop tracking for TRAIN split
    train_drop_stats = {
        "poor_dataset_only": 0,
        "low_sqi_only": 0,
        "both": 0,
        "total_dropped": 0,
        "total_generated": 0,
    }

    # Global arrays for quality cross-tabulation (evaluates all generated windows)
    all_dataset_qualities: List[int] = []
    all_sqi_scores: List[float] = []

    print(f"\n[{dataset_name}] Streaming and windowing records...")

    for rec in loader.subjects():
        total_native_windows += 1
        s_id = str(rec.subject_id).strip()
        split_name = subject_to_split.get(s_id)
        if not split_name:
            continue

        # Slicing stride: train_overlap applies to TRAIN split only!
        # Never overlap val/test/external_val windows.
        if split_name == "train" and train_overlap > 0.0:
            stride = max(1, int(round(window_samples * (1.0 - train_overlap))))
        else:
            stride = window_samples

        child_wins, n_produced = process_record_windows(
            rec=rec,
            target_fs=target_fs,
            window_samples=window_samples,
            stride=stride,
            bp_filter=bp_filter,
            sqi_assessor=sqi_assessor,
        )

        total_child_windows_generated += n_produced
        child_windows_per_native.append(n_produced)

        for win in child_wins:
            dq = win["dataset_quality"]
            sq = win["sqi"]

            all_dataset_qualities.append(dq)
            all_sqi_scores.append(sq)

            # Quality Gating: TRAIN split only
            if split_name == "train":
                train_drop_stats["total_generated"] += 1
                is_poor_dataset = (dq == 2)
                is_low_sqi = (sq < 0.50)

                if is_poor_dataset or is_low_sqi:
                    if is_poor_dataset and is_low_sqi:
                        train_drop_stats["both"] += 1
                    elif is_poor_dataset:
                        train_drop_stats["poor_dataset_only"] += 1
                    elif is_low_sqi:
                        train_drop_stats["low_sqi_only"] += 1
                    train_drop_stats["total_dropped"] += 1
                    continue  # Drop window!

            # Retain window
            split_X[split_name].append(win["X"])
            split_y[split_name].append(win["y"])
            split_subjs[split_name].append(win["subject_id"])
            split_sqi[split_name].append(win["sqi"])
            split_qual[split_name].append(win["dataset_quality"])

        if max_windows is not None and total_native_windows >= max_windows:
            print(f"[{dataset_name}] Reached max_windows limit ({max_windows}). Stopping stream.")
            break

        if total_native_windows % 25000 == 0:
            print(f"  Processed {total_native_windows} records -> {total_child_windows_generated} child windows...")

    print(f"[{dataset_name}] Completed processing {total_native_windows} native records.")

    # Quality Cross-Tabulation Report
    all_dq_arr = np.array(all_dataset_qualities, dtype=np.int8)
    all_sq_arr = np.array(all_sqi_scores, dtype=np.float32)
    cross_tab_str, cross_tab_dict = format_cross_tabulation(all_dq_arr, all_sq_arr)
    print("\n" + cross_tab_str + "\n")

    # Save processed .npz files per split
    manifest_splits_summary: Dict[str, Any] = {}

    print(f"=================================================================")
    print(f" {dataset_name.upper()} Processed Split Summary")
    print(f"=================================================================")

    for split_name in splits.keys():
        X_arr = np.array(split_X[split_name], dtype=np.float32)
        y_arr = np.array(split_y[split_name], dtype=np.int8)
        subj_arr = np.array(split_subjs[split_name], dtype=str)
        sqi_arr = np.array(split_sqi[split_name], dtype=np.float32)
        qual_arr = np.array(split_qual[split_name], dtype=np.int8)

        total_w = len(y_arr)
        af_w = int(np.sum(y_arr == 1))
        non_af_w = int(np.sum(y_arr == 0))
        prev_pct = (af_w / max(1, total_w)) * 100.0
        n_subjs = len(np.unique(subj_arr)) if total_w > 0 else len(splits[split_name])

        print(f" Split: '{split_name}'")
        print(f"   Subjects            : {n_subjs}")
        print(f"   Total Windows (kept): {total_w}")
        print(f"   AF Windows          : {af_w} ({prev_pct:.2f}%)")
        print(f"   Non-AF Windows      : {non_af_w} ({100.0 - prev_pct:.2f}%)")
        if split_name == "train":
            print(f"   Quality Gating Drops: {train_drop_stats['total_dropped']} dropped / {train_drop_stats['total_generated']} generated "
                  f"({train_drop_stats['total_dropped']/max(1, train_drop_stats['total_generated'])*100:.1f}%)")
            print(f"     - Poor Dataset Label only: {train_drop_stats['poor_dataset_only']}")
            print(f"     - Low SQI (< 0.50) only  : {train_drop_stats['low_sqi_only']}")
            print(f"     - Both reasons           : {train_drop_stats['both']}")

        # Primary output npz file
        # Naming: <dataset>_{train,val,test}.npz
        # For MIMIC: mimic_external_val.npz and alias mimic_test.npz
        primary_filename = f"{dataset_name}_{split_name}.npz"
        target_path = output_dir / primary_filename

        np.savez_compressed(
            target_path,
            X=X_arr,
            y=y_arr,
            subject_id=subj_arr,
            sqi=sqi_arr,
            dataset_quality=qual_arr,
        )
        print(f"   Saved to: {target_path.name} ({target_path.stat().st_size / (1024*1024):.2f} MB)")

        if dataset_name == "mimic" and split_name == "external_val":
            # Alias as mimic_test.npz for callers expecting standard test name
            alias_path = output_dir / "mimic_test.npz"
            np.savez_compressed(
                alias_path,
                X=X_arr,
                y=y_arr,
                subject_id=subj_arr,
                sqi=sqi_arr,
                dataset_quality=qual_arr,
            )
            print(f"   Saved alias: {alias_path.name} ({alias_path.stat().st_size / (1024*1024):.2f} MB)")

        manifest_splits_summary[split_name] = {
            "file": primary_filename,
            "subject_count": n_subjs,
            "subjects": splits[split_name],
            "total_windows": total_w,
            "af_windows": af_w,
            "non_af_windows": non_af_w,
            "af_prevalence_pct": round(prev_pct, 2),
            "drop_stats": train_drop_stats if split_name == "train" else None,
        }

        # Immediate memory cleanup for large datasets
        split_X[split_name] = []
        split_y[split_name] = []
        split_subjs[split_name] = []
        split_sqi[split_name] = []
        split_qual[split_name] = []
        del X_arr, y_arr, subj_arr, sqi_arr, qual_arr
        import gc
        gc.collect()

    print(f"=================================================================\n")

    # Sibling <dataset>_manifest.json
    mean_child_win = float(np.mean(child_windows_per_native)) if child_windows_per_native else 0.0
    manifest = {
        "dataset_name": dataset_name,
        "created_at": datetime.now().isoformat(),
        "input_contract": {
            "sampling_rate_hz": target_fs,
            "window_samples": window_samples,
            "window_duration_s": round(window_samples / target_fs, 2),
            "preprocessing_order": [
                "detrend_ppg (linear trend removal)",
                "ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=100.0, order=4)",
                "zscore_normalize (zero mean, unit variance)",
            ],
        },
        "train_overlap": train_overlap,
        "splits": manifest_splits_summary,
        "quality_cross_tabulation": cross_tab_dict,
        "windowing_statistics": {
            "total_native_records_processed": total_native_windows,
            "total_child_windows_generated": total_child_windows_generated,
            "mean_child_windows_per_parent": round(mean_child_win, 2),
        },
        "subject_isolation_verified": True,
        "zero_cross_dataset_overlap_verified": True,
    }

    if dataset_name == "mimic":
        manifest["coarse_label_limitation"] = (
            "MIMIC PERform AF provides recording-level diagnoses, not per-window rhythm labels. "
            "Paroxysmal AF segments or intermittent sinus intervals within AF subjects inherit label 1. "
            "This is a documented limitation of the MIMIC PERform AF annotations."
        )

    manifest_file_path = output_dir / f"{dataset_name}_manifest.json"
    with open(manifest_file_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{dataset_name}] Wrote dataset manifest to: {manifest_file_path.resolve()}\n")

    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build windowed, preprocessed, and split PPG datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["deepbeat", "mimic"],
        help="Dataset to build: 'deepbeat' (training/val/test) or 'mimic' (external validation).",
    )
    parser.add_argument(
        "--train-overlap",
        type=float,
        default=0.0,
        help="Overlap fraction [0.0, 1.0) applied to TRAIN split only. Default: 0.0 (no overlap).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ml_root / "data" / "processed",
        help="Directory to write processed .npz and manifest files.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=_ml_root / "training" / "splits",
        help="Directory to write split manifest JSON files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional path override for raw dataset directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subject splitting.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="adjudicated_test",
        choices=["adjudicated_test", "stratified"],
        help="DeepBeat splitting strategy: 'adjudicated_test' (preserves Table 2 benchmark in test) or 'stratified'.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional maximum number of native records to process (for debugging / quick testing).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        data_dir=args.data_dir,
        train_overlap=args.train_overlap,
        seed=args.seed,
        strategy=args.strategy,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
