"""
Weight Exporter: PyTorch -> Pure-NumPy Runtime Format (ANTIGRAVITY.md §4.3)
===========================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Converts PyTorch state_dict parameters into the NumPy runtime's .npz format:
  conv1.weights     <- conv1.weight.numpy()         # (32, 1, 5) float32
  conv1.bias        <- conv1.bias.numpy()           # (32,) float32
  conv2.weights     <- conv2.weight.numpy()         # (64, 32, 3) float32
  conv2.bias        <- conv2.bias.numpy()           # (64,) float32
  dense1.weights    <- linear1.weight.numpy().T     # torch (64, 16000) -> numpy (16000, 64) float32
  dense1.bias       <- linear1.bias.numpy()         # (64,) float32
  dense_out.weights <- linear2.weight.numpy().T     # torch (1, 64) -> numpy (64, 1) float32
  dense_out.bias    <- linear2.bias.numpy()         # (1,) float32

Also outputs a sibling `<name>.meta.json` recording provenance, git commit hash,
dataset splits SHA-256 hashes, calibrated decision threshold, and input contract.
"""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn


def get_git_commit(repo_dir: Optional[Path] = None) -> str:
    """Retrieves current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        res = subprocess.run(
            cmd,
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    if not path.exists():
        return "missing"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_numpy_weights(model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
    """
    Extracts float32 NumPy arrays matching the Arrhythmia1DCNN pure-NumPy runtime layout.
    """
    if isinstance(model_or_state_dict, nn.Module):
        state_dict = model_or_state_dict.state_dict()
    else:
        state_dict = model_or_state_dict

    # Helper to convert tensor to float32 numpy array
    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    # Note on transpose:
    # PyTorch Linear weight is (out_features, in_features).
    # NumPy DenseBlock computes x @ weights where x is (..., in_features).
    # Therefore, weights must have shape (in_features, out_features) = weight.T.
    # Conv1d layout in PyTorch is (out_channels, in_channels, kernel_size),
    # which matches Conv1DBlock.weights in inference_model.py exactly.
    weights_dict = {
        "conv1.weights": _to_np(state_dict["conv1.weight"]),
        "conv1.bias": _to_np(state_dict["conv1.bias"]),
        "conv2.weights": _to_np(state_dict["conv2.weight"]),
        "conv2.bias": _to_np(state_dict["conv2.bias"]),
        "dense1.weights": _to_np(state_dict["linear1.weight"]).T,
        "dense1.bias": _to_np(state_dict["linear1.bias"]),
        "dense_out.weights": _to_np(state_dict["linear2.weight"]).T,
        "dense_out.bias": _to_np(state_dict["linear2.bias"]),
    }

    # Validate shapes
    expected_shapes = {
        "conv1.weights": (32, 1, 5),
        "conv1.bias": (32,),
        "conv2.weights": (64, 32, 3),
        "conv2.bias": (64,),
        "dense1.weights": (16000, 64),
        "dense1.bias": (64,),
        "dense_out.weights": (64, 1),
        "dense_out.bias": (1,),
    }
    for k, shape in expected_shapes.items():
        actual = weights_dict[k].shape
        if actual != shape:
            raise ValueError(f"Extracted weight shape mismatch for {k}: expected {shape}, got {actual}")

    return weights_dict


def export_weights(
    model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]],
    output_path: Union[str, Path],
    threshold: float = 0.50,
    metrics: Optional[Dict[str, Any]] = None,
    training_dataset_meta: Optional[Dict[str, Any]] = None,
    external_val_dataset_meta: Optional[Dict[str, Any]] = None,
    splits_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Exports PyTorch model weights to .npz and generates sibling .meta.json.

    Args:
        model_or_state_dict: Trained PyTorch Arrhythmia1DCNN model or state_dict.
        output_path: Target path for .npz file (e.g., '03 - ML/model/weights/arrhythmia_1d_cnn.npz').
        threshold: Calibrated decision threshold (default 0.50).
        metrics: Dictionary of internal and external validation metrics.
        training_dataset_meta: Metadata describing training dataset.
        external_val_dataset_meta: Metadata describing external validation dataset.
        splits_dir: Path to directory containing split manifests.

    Returns:
        (npz_path, meta_path)
    """
    out_npz = Path(output_path)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if not out_npz.name.endswith(".npz"):
        out_npz = out_npz.with_suffix(".npz")

    # Extract NumPy weights
    weights_dict = extract_numpy_weights(model_or_state_dict)

    # Save .npz archive
    np.savez(out_npz, **weights_dict)

    # Determine split manifest hashes
    repo_root = Path(__file__).resolve().parent.parent.parent
    if splits_dir is None:
        splits_dir = Path(__file__).resolve().parent / "splits"

    deepbeat_split_path = splits_dir / "deepbeat_split_manifest.json"
    mimic_split_path = splits_dir / "mimic_split_manifest.json"

    split_hashes = {
        "deepbeat_split_manifest": get_file_sha256(deepbeat_split_path),
        "mimic_split_manifest": get_file_sha256(mimic_split_path),
    }

    # Meta json payload
    meta = {
        "git_commit": get_git_commit(repo_root),
        "training_date": datetime.now().isoformat(),
        "training_dataset": training_dataset_meta or {
            "name": "deepbeat",
            "version": "1.0",
            "source": "syn21985690",
            "citation": "Torres-Soto & Ashley (2020), npj Digital Medicine 3:116",
        },
        "external_validation_dataset": external_val_dataset_meta or {
            "name": "mimic_perform_af",
            "version": "1.0",
            "source": "Zenodo doi:10.5281/zenodo.15906524",
            "citation": "Charlton et al. (2022), Physiol. Meas. 43:115003",
        },
        "split_manifest_hashes": split_hashes,
        "threshold": float(threshold),
        "metrics": metrics or {
            "internal_val": {},
            "internal_test_adjudicated": {},
            "external_val_mimic": {},
        },
        "input_contract": {
            "fs": 100.0,
            "window_samples": 1000,
            "window_duration_s": 10.0,
            "preprocessing_chain": [
                "detrend_ppg",
                "ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=100.0, order=4)",
                "zscore_normalize",
            ],
            "runtime_engine": "inference_model.py (pure NumPy)",
            "flatten_order": "time_major (t * 64 + c)",
        },
    }

    out_meta = out_npz.with_name(out_npz.stem + ".meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_npz, out_meta
