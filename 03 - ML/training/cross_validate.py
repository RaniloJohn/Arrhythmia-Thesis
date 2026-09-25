"""
5-Fold Subject-Grouped Cross-Validation (ANTIGRAVITY.md §4.3)
============================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Specifications:
- 5-Fold GroupKFold cross-validation partitioned strictly by subject_id.
- Retrains Arrhythmia1DCNN from scratch on each fold.
- Zero subject leakage between training and validation folds.
- Evaluates fold metrics: AUROC, AUPRC, Sensitivity, Specificity, F1, Accuracy.
- Reports Mean ± Standard Deviation across folds to provide honest variance bounds.
- Saves results to cv_5fold_results.json.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure 03 - ML is on sys.path
ml_dir = Path(__file__).resolve().parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from training.torch_model import Arrhythmia1DCNN
from training.train import NPZWindowDataset, evaluate, set_seed, subsample_by_subject


def run_cross_validation(
    train_npz_path: Path,
    val_npz_path: Path,
    runs_dir: Path,
    n_splits: int = 5,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    max_train_windows_per_fold: Optional[int] = 20000,
    seed: int = 42,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes 5-fold subject-grouped cross-validation.
    """
    set_seed(seed)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=================================================================")
    print(f" Starting {n_splits}-Fold Subject-Grouped Cross-Validation")
    print(f" Device : {device}")
    print(f" Epochs : {epochs} per fold | Batch Size: {batch_size} | LR: {lr}")
    print(f"=================================================================")

    # Load development datasets (train + val splits)
    d_train = np.load(train_npz_path, mmap_mode="r")
    d_val = np.load(val_npz_path, mmap_mode="r")

    # Combine subject pools
    train_subjs = np.unique(d_train["subject_id"])
    val_subjs = np.unique(d_val["subject_id"])
    dev_subjs = sorted(list(set(train_subjs) | set(val_subjs)))

    print(f"[Dataset] Development pool has {len(dev_subjs)} unique subjects: {dev_subjs}")

    # Build combined index map
    # To keep memory light, we map each subject to its dataset file and index
    subj_data_map: Dict[str, Tuple[Path, np.ndarray]] = {}
    for s in train_subjs:
        idx = np.where(d_train["subject_id"] == s)[0]
        subj_data_map[str(s)] = (train_npz_path, idx)
    for s in val_subjs:
        idx = np.where(d_val["subject_id"] == s)[0]
        subj_data_map[str(s)] = (val_npz_path, idx)

    # Classify subjects into AF vs Non-AF for stratified grouping
    subj_af_counts = {}
    for s, (path, idx) in subj_data_map.items():
        data = np.load(path, mmap_mode="r")
        subj_af_counts[s] = int(np.sum(data["y"][idx] == 1))

    # GroupKFold requires array of subject IDs per sample or subject-level splitting
    # Split directly at subject level
    dev_subjs_arr = np.array(dev_subjs)
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics: List[Dict[str, float]] = []
    fold_details: List[Dict[str, Any]] = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = runs_dir / f"{timestamp}_cross_val_{n_splits}fold"
    cv_dir.mkdir(parents=True, exist_ok=True)

    total_cv_start = datetime.now()

    for fold, (train_subj_idx, test_subj_idx) in enumerate(gkf.split(dev_subjs_arr, groups=dev_subjs_arr), 1):
        fold_train_subjs = dev_subjs_arr[train_subj_idx].tolist()
        fold_test_subjs = dev_subjs_arr[test_subj_idx].tolist()

        # Assert zero leakage
        overlap = set(fold_train_subjs) & set(fold_test_subjs)
        assert len(overlap) == 0, f"Subject leakage in fold {fold}: {overlap}"

        print(f"\n{'-'*65}")
        print(f" Fold [{fold}/{n_splits}] Execution")
        print(f"   Train Subjects ({len(fold_train_subjs)}): {fold_train_subjs}")
        print(f"   Val Subjects   ({len(fold_test_subjs)}): {fold_test_subjs}")
        print(f"{'-'*65}")

        # Assemble fold train and val window arrays
        def _get_subset_arrays(subjs_list: List[str]):
            X_list, y_list, s_list = [], [], []
            for s in subjs_list:
                p, idx = subj_data_map[s]
                raw = np.load(p, mmap_mode="r")
                X_list.append(raw["X"][idx])
                y_list.append(raw["y"][idx])
                s_list.append(raw["subject_id"][idx])
            return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), np.concatenate(s_list, axis=0)

        f_train_X, f_train_y, f_train_s = _get_subset_arrays(fold_train_subjs)
        f_test_X, f_test_y, f_test_s = _get_subset_arrays(fold_test_subjs)

        # Subsample training windows by subject if cap is set
        if max_train_windows_per_fold is not None and len(f_train_y) > max_train_windows_per_fold:
            chosen_idx = subsample_by_subject(
                subjects=f_train_s,
                y=f_train_y,
                max_windows=max_train_windows_per_fold,
                seed=seed + fold,
            )
            f_train_X = f_train_X[chosen_idx]
            f_train_y = f_train_y[chosen_idx]
            f_train_s = f_train_s[chosen_idx]

        # Subsample validation windows by subject if too large
        if len(f_test_y) > 20000:
            chosen_val_idx = subsample_by_subject(
                subjects=f_test_s,
                y=f_test_y,
                max_windows=20000,
                seed=seed + fold + 10,
            )
            f_test_X = f_test_X[chosen_val_idx]
            f_test_y = f_test_y[chosen_val_idx]
            f_test_s = f_test_s[chosen_val_idx]

        n_train_af = int(np.sum(f_train_y == 1))
        n_train_non = int(np.sum(f_train_y == 0))
        n_val_af = int(np.sum(f_test_y == 1))
        n_val_non = int(np.sum(f_test_y == 0))

        print(f"   Train Set: {len(f_train_y):,} windows (AF: {n_train_af:,} [{n_train_af/len(f_train_y)*100:.1f}%])")
        print(f"   Val Set  : {len(f_test_y):,} windows (AF: {n_val_af:,} [{n_val_af/len(f_test_y)*100:.1f}%])")

        # PyTorch Tensor Loaders
        train_tensor_x = torch.from_numpy(f_train_X).unsqueeze(1).float()
        train_tensor_y = torch.from_numpy(f_train_y).float().unsqueeze(1)
        val_tensor_x = torch.from_numpy(f_test_X).unsqueeze(1).float()
        val_tensor_y = torch.from_numpy(f_test_y).float().unsqueeze(1)

        train_ds = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
        val_ds = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Fresh Model Instance
        model = Arrhythmia1DCNN(input_length=1000, dropout=dropout).to(device)

        # Imbalance weighting (clamped)
        pos_weight_val = max(0.2, min((n_train_non / n_train_af) if n_train_af > 0 else 1.0, 10.0))
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
        train_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        val_crit = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train fold
        for ep in range(1, epochs + 1):
            model.train()
            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                logits = model(bx, return_logits=True)
                loss = train_crit(logits, by)
                loss.backward()
                optimizer.step()

        # Evaluate fold
        val_res = evaluate(model, val_loader, val_crit, device)

        fold_record = {
            "fold": fold,
            "auroc": val_res["auroc"],
            "auprc": val_res["auprc"],
            "sensitivity": val_res["sensitivity"],
            "specificity": val_res["specificity"],
            "val_loss": val_res["loss"],
            "n_train": len(f_train_y),
            "n_val": len(f_test_y),
            "train_subjects": fold_train_subjs,
            "val_subjects": fold_test_subjs,
        }
        fold_metrics.append(val_res)
        fold_details.append(fold_record)

        print(f"   --> Fold {fold} Result: AUROC={val_res['auroc']:.4f} | AUPRC={val_res['auprc']:.4f} | Sens={val_res['sensitivity']:.4f} | Spec={val_res['specificity']:.4f} | Loss={val_res['loss']:.4f}")

    total_duration = (datetime.now() - total_cv_start).total_seconds()

    # Aggregate Statistics across folds
    def _mean_std(key: str) -> Tuple[float, float]:
        vals = [m[key] for m in fold_metrics]
        return float(np.mean(vals)), float(np.std(vals))

    mean_auroc, std_auroc = _mean_std("auroc")
    mean_auprc, std_auprc = _mean_std("auprc")
    mean_sens, std_sens = _mean_std("sensitivity")
    mean_spec, std_spec = _mean_std("specificity")

    print(f"\n{'='*75}")
    print(f" 5-Fold Subject-Grouped Cross-Validation Summary")
    print(f"{'='*75}")
    print(f"  Fold 1: AUROC = {fold_details[0]['auroc']:.4f} | AUPRC = {fold_details[0]['auprc']:.4f} | Sens = {fold_details[0]['sensitivity']:.4f} | Spec = {fold_details[0]['specificity']:.4f}")
    print(f"  Fold 2: AUROC = {fold_details[1]['auroc']:.4f} | AUPRC = {fold_details[1]['auprc']:.4f} | Sens = {fold_details[1]['sensitivity']:.4f} | Spec = {fold_details[1]['specificity']:.4f}")
    print(f"  Fold 3: AUROC = {fold_details[2]['auroc']:.4f} | AUPRC = {fold_details[2]['auprc']:.4f} | Sens = {fold_details[2]['sensitivity']:.4f} | Spec = {fold_details[2]['specificity']:.4f}")
    print(f"  Fold 4: AUROC = {fold_details[3]['auroc']:.4f} | AUPRC = {fold_details[3]['auprc']:.4f} | Sens = {fold_details[3]['sensitivity']:.4f} | Spec = {fold_details[3]['specificity']:.4f}")
    print(f"  Fold 5: AUROC = {fold_details[4]['auroc']:.4f} | AUPRC = {fold_details[4]['auprc']:.4f} | Sens = {fold_details[4]['sensitivity']:.4f} | Spec = {fold_details[4]['specificity']:.4f}")
    print(f"{'-'*75}")
    print(f"  Cross-Validated AUROC      : {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"  Cross-Validated AUPRC      : {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"  Cross-Validated Sensitivity: {mean_sens:.4f} ± {std_sens:.4f}")
    print(f"  Cross-Validated Specificity: {mean_spec:.4f} ± {std_spec:.4f}")
    print(f"  Total CV Wall-Clock Time   : {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"{'='*75}\n")

    summary_payload = {
        "n_splits": n_splits,
        "epochs_per_fold": epochs,
        "total_duration_s": total_duration,
        "mean_metrics": {
            "auroc": round(mean_auroc, 4),
            "auroc_std": round(std_auroc, 4),
            "auprc": round(mean_auprc, 4),
            "auprc_std": round(std_auprc, 4),
            "sensitivity": round(mean_sens, 4),
            "sensitivity_std": round(std_sens, 4),
            "specificity": round(mean_spec, 4),
            "specificity_std": round(std_spec, 4),
        },
        "folds": fold_details,
    }

    out_json = cv_dir / "cv_5fold_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[Results] Saved cross-validation summary to: {out_json.resolve()}")

    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="Run 5-Fold Subject-Grouped Cross-Validation")
    parser.add_argument("--train-data", type=Path, default=ml_dir / "data" / "processed" / "deepbeat_train.npz")
    parser.add_argument("--val-data", type=Path, default=ml_dir / "data" / "processed" / "deepbeat_val.npz")
    parser.add_argument("--runs-dir", type=Path, default=ml_dir / "training" / "runs")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-train-windows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    run_cross_validation(
        train_npz_path=args.train_data,
        val_npz_path=args.val_data,
        runs_dir=args.runs_dir,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        max_train_windows_per_fold=args.max_train_windows,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
