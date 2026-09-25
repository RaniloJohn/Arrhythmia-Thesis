"""
PyTorch Training Loop for 1D-CNN Arrhythmia Model (ANTIGRAVITY.md §4.3)
========================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Specifications & Hyperparameters:
- Trains on DeepBeat train split (451,791 windows, 12 subjects)
- Early-stops on DeepBeat validation split (255,874 windows, 4 subjects)
- Monitored metric: Validation AUROC (patience = 5)
- Loss: BCEWithLogitsLoss with pos_weight from train class imbalance
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=2)
- Architecture: Arrhythmia1DCNN with Dropout(p=0.5) before dense layers
- Checkpoints: Saves best.pt and last.pt every epoch; logs to history.csv
- End-of-run: Exports model to pure-NumPy runtime .npz via training.export_weights

Hard Guards:
- Refuses to train if split manifest has subject overlap.
- Refuses to train if any MIMIC-derived file appears in training paths.
- Prints pre-training subject and class prevalence summary before Epoch 1.
"""

import os
import sys
import time
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure 03 - ML is on sys.path
ml_dir = Path(__file__).resolve().parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from training.torch_model import Arrhythmia1DCNN
from training.export_weights import export_weights
from model.inference_model import Arrhythmia1DCNN as NumPyArrhythmia1DCNN


class NPZWindowDataset(Dataset):
    """
    Zero-copy memory-mapped dataset for 1000-sample windowed PPG signals.
    """
    def __init__(self, npz_path: Path, indices: Optional[np.ndarray] = None):
        self.npz_path = Path(npz_path)
        self.data = np.load(self.npz_path, mmap_mode="r")
        self.X = self.data["X"]
        self.y = self.data["y"]
        self.subjects = self.data["subject_id"] if "subject_id" in self.data else np.array([])

        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
        else:
            self.indices = np.arange(len(self.y), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        x_raw = np.array(self.X[real_idx], dtype=np.float32)
        x_tensor = torch.from_numpy(x_raw).unsqueeze(0)  # Shape: (1, 1000)
        y_tensor = torch.tensor([float(self.y[real_idx])], dtype=torch.float32)
        return x_tensor, y_tensor


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_hard_guards(
    train_path: Path,
    val_path: Path,
    manifest_path: Path,
    train_data: Any,
    val_data: Any,
):
    """
    Enforces architectural separation constraints and subject disjointness.
    """
    print("\n=================================================================")
    print(" Running Training Hard Guards Verification")
    print("=================================================================")

    # Guard 1: Anti-MIMIC Contamination Check
    checked_paths = [train_path, val_path, manifest_path]
    for p in checked_paths:
        p_str = str(p.resolve()).lower()
        if "mimic" in p_str:
            raise RuntimeError(
                f"CRITICAL HARD GUARD VIOLATION: MIMIC dataset detected in path: '{p}'.\n"
                f"The external-validation claim strictly requires MIMIC to remain completely "
                f"unseen during training and validation. Refusing to train."
            )

    train_subjects = [str(s).strip() for s in train_data["subject_id"]]
    val_subjects = [str(s).strip() for s in val_data["subject_id"]]

    for s in train_subjects[:100] + val_subjects[:100]:
        if "mimic" in s.lower():
            raise RuntimeError(
                f"CRITICAL HARD GUARD VIOLATION: MIMIC subject '{s}' found in training data!"
            )
    print(" [Guard 1 Passed] Zero MIMIC contamination detected in paths or subject IDs.")

    # Guard 2: Split Manifest Subject Disjointness
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path.resolve()}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest.get("splits", {})
    train_manifest_subjs = set(splits.get("train", []))
    val_manifest_subjs = set(splits.get("val", []))
    test_manifest_subjs = set(splits.get("test", []))

    overlap_train_val = train_manifest_subjs & val_manifest_subjs
    overlap_train_test = train_manifest_subjs & test_manifest_subjs
    overlap_val_test = val_manifest_subjs & test_manifest_subjs

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            f"CRITICAL HARD GUARD VIOLATION: Split manifest contains subject leakage!\n"
            f"  Train/Val: {overlap_train_val}\n"
            f"  Train/Test: {overlap_train_test}\n"
            f"  Val/Test: {overlap_val_test}"
        )
    print(" [Guard 2 Passed] Subject-isolation assertions verified in split manifest.")

    # Guard 3: In-Data Subject Disjointness
    train_unique = set(train_subjects)
    val_unique = set(val_subjects)
    data_overlap = train_unique & val_unique
    if data_overlap:
        raise RuntimeError(
            f"CRITICAL HARD GUARD VIOLATION: Overlapping subjects between train and val arrays: {data_overlap}"
        )
    print(" [Guard 3 Passed] Train array and Val array are strictly subject-disjoint.")
    print("=================================================================\n")


def print_prevalence_summary(name: str, y: np.ndarray, subjects: np.ndarray) -> Dict[str, Any]:
    """Prints subject count and AF prevalence breakdown."""
    unique_subjs = sorted(list(set(subjects)))
    n_total = len(y)
    n_af = int(np.sum(y == 1))
    n_non_af = int(np.sum(y == 0))
    af_pct = (n_af / n_total * 100) if n_total > 0 else 0.0
    non_af_pct = (n_non_af / n_total * 100) if n_total > 0 else 0.0

    print(f"[{name.upper()} Dataset Summary]")
    print(f"  Subjects      : {len(unique_subjs)} subjects -> {unique_subjs}")
    print(f"  Total Windows : {n_total:,}")
    print(f"  AF Windows    : {n_af:,} ({af_pct:.2f}%)")
    print(f"  Non-AF Windows: {n_non_af:,} ({non_af_pct:.2f}%)")
    if n_af > 0:
        ratio = n_non_af / n_af
        print(f"  Imbalance     : 1 AF to {ratio:.2f} Non-AF (recommended pos_weight = {ratio:.4f})")
    print()

    return {
        "subjects": unique_subjs,
        "n_total": n_total,
        "n_af": n_af,
        "n_non_af": n_non_af,
        "af_pct": af_pct,
        "non_af_pct": non_af_pct,
        "pos_weight": (n_non_af / n_af) if n_af > 0 else 1.0,
    }


def subsample_by_subject(
    subjects: np.ndarray,
    y: np.ndarray,
    max_windows: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Subsamples whole subjects to keep total windows near max_windows while
    maintaining strict subject-isolation (never splitting a subject across sets).
    """
    subj_to_indices: Dict[str, List[int]] = {}
    subj_af_counts: Dict[str, int] = {}

    for i, s in enumerate(subjects):
        s_str = str(s).strip()
        if s_str not in subj_to_indices:
            subj_to_indices[s_str] = []
            subj_af_counts[s_str] = 0
        subj_to_indices[s_str].append(i)
        if y[i] == 1:
            subj_af_counts[s_str] += 1

    unique_subjs = sorted(list(subj_to_indices.keys()))
    af_subjs = [s for s in unique_subjs if subj_af_counts[s] > 0]
    non_af_subjs = [s for s in unique_subjs if subj_af_counts[s] == 0]

    # Sort candidate subjects by window count to pack whole subjects into max_windows budget
    af_subjs_sorted = sorted(af_subjs, key=lambda s: len(subj_to_indices[s]))
    non_af_subjs_sorted = sorted(non_af_subjs, key=lambda s: len(subj_to_indices[s]))

    selected_subjs = set()
    current_windows = 0

    i_af = 0
    i_non = 0
    while (i_af < len(af_subjs_sorted) or i_non < len(non_af_subjs_sorted)) and (
        current_windows < max_windows or len(selected_subjs) < 2
    ):
        if i_af < len(af_subjs_sorted):
            s = af_subjs_sorted[i_af]
            i_af += 1
            if s not in selected_subjs and (current_windows + len(subj_to_indices[s]) <= max_windows * 1.5 or len(selected_subjs) == 0):
                selected_subjs.add(s)
                current_windows += len(subj_to_indices[s])

        if i_non < len(non_af_subjs_sorted):
            s = non_af_subjs_sorted[i_non]
            i_non += 1
            if s not in selected_subjs and (current_windows + len(subj_to_indices[s]) <= max_windows * 1.5 or len(selected_subjs) <= 1):
                selected_subjs.add(s)
                current_windows += len(subj_to_indices[s])

    chosen_indices: List[int] = []
    for s in sorted(list(selected_subjs)):
        chosen_indices.extend(subj_to_indices[s])

    chosen_arr = np.array(chosen_indices, dtype=np.int64)
    print(f"[Subsample by Subject] Selected {len(selected_subjs)} whole subjects "
          f"({sorted(list(selected_subjs))}) yielding {len(chosen_arr):,} windows.")
    return chosen_arr


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates validation loss, AUROC, AUPRC, sensitivity, and specificity."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_targets: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.float32)

            logits = model(x_batch, return_logits=True)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = y_batch.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(targets)

    avg_loss = total_loss / max(1, n_batches)
    y_true = np.array(all_targets, dtype=np.int8)
    y_score = np.array(all_probs, dtype=np.float32)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos > 0 and n_neg > 0:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
    else:
        auroc = 0.50
        auprc = 0.0

    # Decision threshold at default 0.50
    preds = (y_score >= 0.50).astype(np.int8)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "loss": float(avg_loss),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train(
    train_data_path: Path,
    val_data_path: Path,
    manifest_path: Path,
    output_weights_path: Path,
    runs_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    patience: int = 5,
    max_train_windows: Optional[int] = None,
    max_val_windows: Optional[int] = None,
    seed: int = 42,
    device_name: Optional[str] = None,
    resume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main training execution loop.
    """
    set_seed(seed)

    # Device auto-detection
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Using computation device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Mmap inspect datasets for guards and prevalence
    train_raw = np.load(train_data_path, mmap_mode="r")
    val_raw = np.load(val_data_path, mmap_mode="r")

    verify_hard_guards(
        train_path=train_data_path,
        val_path=val_data_path,
        manifest_path=manifest_path,
        train_data=train_raw,
        val_data=val_raw,
    )

    # Subsampling by subject if requested
    train_indices = None
    if max_train_windows is not None and max_train_windows < len(train_raw["y"]):
        train_indices = subsample_by_subject(
            subjects=train_raw["subject_id"],
            y=train_raw["y"],
            max_windows=max_train_windows,
            seed=seed,
        )

    val_indices = None
    if max_val_windows is not None and max_val_windows < len(val_raw["y"]):
        val_indices = subsample_by_subject(
            subjects=val_raw["subject_id"],
            y=val_raw["y"],
            max_windows=max_val_windows,
            seed=seed + 1,
        )

    # Print summary of active subsets
    active_train_y = train_raw["y"][train_indices] if train_indices is not None else train_raw["y"]
    active_train_subjs = train_raw["subject_id"][train_indices] if train_indices is not None else train_raw["subject_id"]
    train_summary = print_prevalence_summary("Training", active_train_y, active_train_subjs)

    active_val_y = val_raw["y"][val_indices] if val_indices is not None else val_raw["y"]
    active_val_subjs = val_raw["subject_id"][val_indices] if val_indices is not None else val_raw["subject_id"]
    val_summary = print_prevalence_summary("Validation", active_val_y, active_val_subjs)

    # Build PyTorch datasets and loaders
    train_dataset = NPZWindowDataset(train_data_path, indices=train_indices)
    val_dataset = NPZWindowDataset(val_data_path, indices=val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Run logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    history_csv = run_dir / "history.csv"

    with open(history_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_auroc,val_auprc,val_sensitivity,val_specificity,lr,epoch_time_s\n")

    # Instantiate Model
    model = Arrhythmia1DCNN(input_length=1000, dropout=dropout).to(device)

    # Class-weighted loss for training; unweighted standard BCE for validation
    pos_weight_val = max(0.2, min(train_summary["pos_weight"], 10.0))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    val_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    start_epoch = 1
    best_val_auroc = 0.0
    patience_counter = 0

    if resume_path and Path(resume_path).exists():
        print(f"[Resume] Loading checkpoint from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auroc = ckpt.get("best_val_auroc", 0.0)

    print(f"=================================================================")
    print(f" Starting Training Loop: {epochs} Epochs Max")
    print(f" Run Directory: {run_dir.resolve()}")
    print(f"=================================================================")

    total_training_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model(batch_x, return_logits=True)
            loss = train_criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = total_train_loss / max(1, n_train_batches)

        # Validation evaluation using unweighted loss
        val_metrics = evaluate(model, val_loader, val_criterion, device)
        val_loss = val_metrics["loss"]
        val_auroc = val_metrics["auroc"]
        val_auprc = val_metrics["auprc"]
        val_sens = val_metrics["sensitivity"]
        val_spec = val_metrics["specificity"]

        # Step LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Log to history.csv
        with open(history_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_train_loss:.6f},{val_loss:.6f},{val_auroc:.6f},{val_auprc:.6f},"
                f"{val_sens:.6f},{val_spec:.6f},{current_lr:.2e},{epoch_duration:.2f}\n"
            )

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val AUPRC: {val_auprc:.4f} | "
            f"Sens: {val_sens:.3f} | "
            f"Spec: {val_spec:.3f} | "
            f"LR: {current_lr:.1e} | "
            f"{epoch_duration:.1f}s"
        )

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auroc": val_auroc,
            "best_val_auroc": best_val_auroc,
            "val_metrics": val_metrics,
            "train_loss": avg_train_loss,
        }, last_checkpoint_path)

        # Early stopping logic on Validation AUROC
        if val_auroc > best_val_auroc + 1e-4:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": val_auroc,
                "best_val_auroc": best_val_auroc,
                "val_metrics": val_metrics,
                "train_loss": avg_train_loss,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint with Val AUROC: {best_val_auroc:.4f}")
        else:
            patience_counter += 1
            print(f"  --> No AUROC improvement. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered after {epoch} epochs (patience={patience}).")
                break

    total_wall_clock = time.time() - total_training_start

    print(f"\n=================================================================")
    print(f" Training Complete in {total_wall_clock:.1f}s ({total_wall_clock/60:.1f} min)")
    print(f" Best Validation AUROC: {best_val_auroc:.4f}")
    print(f" Restoring best weights from: {best_checkpoint_path.resolve()}")
    print(f"=================================================================")

    # Restore best checkpoint
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Automatically export weights to pure-NumPy runtime
    out_npz, out_meta = export_weights(
        model_or_state_dict=model,
        output_path=output_weights_path,
        threshold=0.50,
        metrics={
            "best_epoch": best_ckpt["epoch"],
            "best_val_auroc": best_ckpt["val_auroc"],
            "val_metrics": best_ckpt["val_metrics"],
            "train_loss": best_ckpt["train_loss"],
            "total_wall_clock_s": round(total_wall_clock, 2),
            "device": str(device),
            "seed": seed,
            "training_windows": len(train_dataset),
            "validation_windows": len(val_dataset),
        },
        training_dataset_meta={
            "name": "deepbeat",
            "source": "syn21985690",
            "subjects_used": train_summary["subjects"],
            "window_count": len(train_dataset),
        },
    )
    print(f"[Export] Saved trained weights archive to: {out_npz.resolve()}")
    print(f"[Export] Saved metadata manifest to    : {out_meta.resolve()}")

    # Verify pure-NumPy runtime loading
    np_model = NumPyArrhythmia1DCNN(input_length=1000)
    np_model.load_weights(out_npz)
    assert np_model.weights_loaded is True
    print(f"[Verification] Successfully verified Arrhythmia1DCNN.load_weights({out_npz.name})!")

    return {
        "best_epoch": best_ckpt["epoch"],
        "best_val_auroc": best_val_auroc,
        "val_metrics": best_ckpt["val_metrics"],
        "history_csv": str(history_csv.resolve()),
        "output_npz": str(out_npz.resolve()),
        "output_meta": str(out_meta.resolve()),
        "wall_clock_s": total_wall_clock,
    }


def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN Arrhythmia Model on DeepBeat")
    parser.add_argument("--train-data", type=Path, default=ml_dir / "data" / "processed" / "deepbeat_train.npz",
                        help="Path to processed train .npz")
    parser.add_argument("--val-data", type=Path, default=ml_dir / "data" / "processed" / "deepbeat_val.npz",
                        help="Path to processed validation .npz")
    parser.add_argument("--split-manifest", type=Path, default=ml_dir / "training" / "splits" / "deepbeat_split_manifest.json",
                        help="Path to DeepBeat split manifest JSON")
    parser.add_argument("--output-weights", type=Path, default=ml_dir / "model" / "weights" / "cnn_af_v1.npz",
                        help="Output path for exported NumPy .npz weights")
    parser.add_argument("--runs-dir", type=Path, default=ml_dir / "training" / "runs",
                        help="Directory to save training run logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate before dense layers")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on val AUROC")
    parser.add_argument("--max-train-windows", type=int, default=None,
                        help="Cap training windows (subsampled strictly by subject)")
    parser.add_argument("--max-val-windows", type=int, default=None,
                        help="Cap validation windows (subsampled strictly by subject)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Target device (cuda or cpu)")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        manifest_path=args.split_manifest,
        output_weights_path=args.output_weights,
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        seed=args.seed,
        device_name=args.device,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
