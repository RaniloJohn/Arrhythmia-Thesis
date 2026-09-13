"""
Threshold Calibration & Probabilistic Reliability Analysis (ANTIGRAVITY.md §4.3)
=================================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Specifications:
- Calibrates decision threshold on DeepBeat VALIDATION split only.
- Strict Hard Guards: Refuses to run if path contains 'mimic' or 'test'.
- Computes sensitivity, specificity, PPV, NPV, F1, and Youden's J statistic across
  thresholds [0.00, 1.00] at 0.01 resolution.
- Selects optimal threshold maximizing Youden's J or satisfying target sensitivity.
- Evaluates probabilistic calibration: Brier score, Expected Calibration Error (ECE).
- Generates 2-panel calibration plot: Reliability diagram & Threshold trade-off curve.
- Writes chosen value into cnn_af_v1.meta.json as `decision_threshold` and `threshold`.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# Ensure 03 - ML is on sys.path
ml_dir = Path(__file__).resolve().parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from training.torch_model import Arrhythmia1DCNN
from model.inference_model import Arrhythmia1DCNN as NumPyArrhythmia1DCNN


def verify_calibration_guards(val_path: Path):
    """
    Enforces strict architectural boundaries for threshold calibration.
    Threshold calibration must happen on the internal VALIDATION split only.
    """
    p_str = str(val_path.resolve()).lower()
    if "mimic" in p_str:
        raise RuntimeError(
            f"CRITICAL HARD GUARD VIOLATION: MIMIC dataset detected in path: '{val_path}'.\n"
            f"MIMIC is strictly held out for external validation. Threshold calibration "
            f"must NEVER touch MIMIC. Refusing to run."
        )
    if "test" in p_str:
        raise RuntimeError(
            f"CRITICAL HARD GUARD VIOLATION: Test split detected in path: '{val_path}'.\n"
            f"Threshold calibration must be performed on the VALIDATION split, never on the test set. "
            f"Refusing to run."
        )


def load_numpy_weights_into_torch(torch_model: nn.Module, npz_path: Path) -> None:
    """
    Loads exported NumPy .npz weights into a PyTorch Arrhythmia1DCNN model for fast batch inference.
    """
    data = np.load(npz_path)
    state_dict = torch_model.state_dict()

    state_dict["conv1.weight"] = torch.from_numpy(data["conv1.weights"])
    state_dict["conv1.bias"] = torch.from_numpy(data["conv1.bias"])
    state_dict["conv2.weight"] = torch.from_numpy(data["conv2.weights"])
    state_dict["conv2.bias"] = torch.from_numpy(data["conv2.bias"])

    # Transpose dense weights: NumPy (in_features, out_features) -> PyTorch (out_features, in_features)
    state_dict["linear1.weight"] = torch.from_numpy(data["dense1.weights"].T)
    state_dict["linear1.bias"] = torch.from_numpy(data["dense1.bias"])
    state_dict["linear2.weight"] = torch.from_numpy(data["dense_out.weights"].T)
    state_dict["linear2.bias"] = torch.from_numpy(data["dense_out.bias"])

    torch_model.load_state_dict(state_dict)
    torch_model.eval()


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Computes Brier score, Brier skill score, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE).
    """
    brier = float(brier_score_loss(y_true, y_prob))
    base_rate = float(np.mean(y_true))
    brier_ref = base_rate * (1.0 - base_rate)
    bss = float(1.0 - (brier / brier_ref)) if brier_ref > 0 else 0.0

    # Calibration curve bins
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Compute ECE and MCE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    n_total = len(y_true)

    for b in range(n_bins):
        mask = (bin_indices == b)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (count / n_total) * diff
            if diff > mce:
                mce = diff

    return {
        "brier_score": round(brier, 6),
        "brier_reference": round(brier_ref, 6),
        "brier_skill_score": round(bss, 4),
        "expected_calibration_error": round(float(ece), 4),
        "maximum_calibration_error": round(float(mce), 4),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "base_rate": round(base_rate, 4),
    }


def compute_threshold_metrics_table(y_true: np.ndarray, y_prob: np.ndarray) -> List[Dict[str, Any]]:
    """
    Evaluates sensitivity, specificity, PPV, NPV, F1, and Youden's J at 0.01 threshold resolution.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    table = []

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    for t in thresholds:
        t_val = round(float(t), 2)
        preds = (y_prob >= t_val).astype(np.int8)

        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        youden_j = sens + spec - 1.0
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        table.append({
            "threshold": t_val,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "youden_j": round(youden_j, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    return table


def plot_calibration_and_tradeoff(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    table: List[Dict[str, Any]],
    chosen_threshold: float,
    output_plot_path: Path,
):
    """
    Generates a 2-panel diagnostic figure:
    1. Reliability diagram (Calibration curve) with confidence histogram.
    2. Threshold trade-off curve (Sensitivity, Specificity, Youden's J).
    """
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Panel 1: Reliability Diagram
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    ax1.plot(prob_pred, prob_true, "s-", color="#0E6B76", label="1D-CNN (Validation Split)", linewidth=2)
    ax1.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Empirical Fraction of Positives (AF)", fontsize=10, fontweight="bold")
    ax1.set_title("Reliability Diagram (Calibration Curve)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Inset or twin axis histogram for prediction distribution
    ax1_hist = ax1.twinx()
    ax1_hist.hist(y_prob, bins=20, range=(0, 1), color="#0E6B76", alpha=0.15, edgecolor="none")
    ax1_hist.set_ylabel("Window Count", fontsize=9, color="#555")
    ax1_hist.tick_params(axis="y", labelcolor="#555")
    ax1_hist.grid(False)

    # Panel 2: Threshold Trade-Off Curve
    ts = [row["threshold"] for row in table]
    sens = [row["sensitivity"] for row in table]
    spec = [row["specificity"] for row in table]
    youden = [row["youden_j"] for row in table]

    ax2.plot(ts, sens, label="Sensitivity (Recall)", color="#D9534F", linewidth=2)
    ax2.plot(ts, spec, label="Specificity", color="#0275D8", linewidth=2)
    ax2.plot(ts, youden, label="Youden's J (Sens + Spec - 1)", color="#5CB85C", linewidth=2, linestyle="--")
    ax2.axvline(chosen_threshold, color="#333", linestyle=":", linewidth=2,
                label=f"Chosen Threshold = {chosen_threshold:.2f}")

    ax2.set_xlabel("Decision Threshold", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Metric Value", fontsize=10, fontweight="bold")
    ax2.set_title("Validation Threshold Optimization", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="center left", fontsize=9)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    print(f"[Plot] Saved calibration and threshold trade-off diagram to: {output_plot_path.resolve()}")


def calibrate_threshold(
    val_data_path: Path,
    weights_path: Path,
    meta_path: Path,
    output_plot_path: Path,
    output_table_path: Path,
    target_sensitivity: Optional[float] = None,
    batch_size: int = 128,
    max_val_windows: Optional[int] = None,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes threshold calibration and reliability assessment.
    """
    verify_calibration_guards(val_data_path)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=================================================================")
    print(f" Starting Threshold Calibration on Validation Split")
    print(f" Validation Path: {val_data_path.resolve()}")
    print(f" Weights Path   : {weights_path.resolve()}")
    print(f" Metadata Path  : {meta_path.resolve()}")
    print(f" Hardware Device: {device}")
    print(f"=================================================================")

    # Load validation data
    val_raw = np.load(val_data_path, mmap_mode="r")
    X_val = val_raw["X"]
    y_val = np.array(val_raw["y"], dtype=np.int8)
    subjs_val = np.array(val_raw["subject_id"])

    n_val = len(y_val)
    if max_val_windows is not None and max_val_windows < n_val:
        indices = np.arange(max_val_windows)
        X_eval = X_val[indices]
        y_eval = y_val[indices]
        subjs_eval = subjs_val[indices]
    else:
        X_eval = X_val
        y_eval = y_val
        subjs_eval = subjs_val

    print(f"[Validation Summary] Evaluating on {len(y_eval):,} windows across {len(set(subjs_eval))} subjects.")
    print(f"  AF Windows    : {int(np.sum(y_eval==1)):,} ({np.mean(y_eval==1)*100:.2f}%)")
    print(f"  Non-AF Windows: {int(np.sum(y_eval==0)):,} ({np.mean(y_eval==0)*100:.2f}%)")

    # Load PyTorch model for batch inference
    torch_model = Arrhythmia1DCNN(input_length=1000, dropout=0.0).to(device)
    load_numpy_weights_into_torch(torch_model, weights_path)

    # Batch forward pass
    all_probs = []
    n_batches = (len(y_eval) + batch_size - 1) // batch_size
    t_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    print("\n[Inference] Computing predicted probabilities across validation windows...")
    with torch.no_grad():
        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, len(y_eval))
            x_chunk = torch.from_numpy(np.array(X_eval[start:end], dtype=np.float32)).unsqueeze(1).to(device)
            probs = torch_model(x_chunk, return_logits=False).cpu().numpy().flatten()
            all_probs.extend(probs)

    y_prob = np.array(all_probs, dtype=np.float32)

    # 1. Calibration Metrics
    cal_metrics = compute_calibration_metrics(y_eval, y_prob, n_bins=10)
    print("\n=================================================================")
    print(" Probabilistic Calibration & Reliability Assessment")
    print("=================================================================")
    print(f"  Base Rate (Prevalence)     : {cal_metrics['base_rate']:.4f}")
    print(f"  Brier Score                : {cal_metrics['brier_score']:.6f} (Ref: {cal_metrics['brier_reference']:.6f})")
    print(f"  Brier Skill Score (BSS)    : {cal_metrics['brier_skill_score']:.4f}")
    print(f"  Expected Calib. Error (ECE): {cal_metrics['expected_calibration_error']:.4f}")
    print(f"  Maximum Calib. Error (MCE) : {cal_metrics['maximum_calibration_error']:.4f}")

    if cal_metrics["expected_calibration_error"] > 0.10:
        calib_assessment = "POORLY_CALIBRATED"
        print("  --> Assessment: Model exhibits noticeable calibration error (ECE > 0.10).")
        print("      Note: Post-hoc Platt scaling or isotonic regression is recommended if")
        print("      probabilities are presented directly to clinicians, but deferred as it")
        print("      requires extra runtime steps outside the pure-NumPy inference engine.")
    else:
        calib_assessment = "ACCEPTABLY_CALIBRATED"
        print("  --> Assessment: Model demonstrates acceptable probabilistic calibration (ECE <= 0.10).")
    print("=================================================================\n")

    # 2. Threshold Metrics Table
    table = compute_threshold_metrics_table(y_eval, y_prob)

    # Save CSV table
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_table_path, "w", encoding="utf-8") as f:
        f.write("threshold,sensitivity,specificity,ppv,npv,youden_j,f1,tp,fp,tn,fn\n")
        for row in table:
            f.write(f"{row['threshold']:.2f},{row['sensitivity']:.4f},{row['specificity']:.4f},"
                    f"{row['ppv']:.4f},{row['npv']:.4f},{row['youden_j']:.4f},{row['f1']:.4f},"
                    f"{row['tp']},{row['fp']},{row['tn']},{row['fn']}\n")
    print(f"[Table] Saved threshold tuning table to: {output_table_path.resolve()}")

    # 3. Threshold Selection Strategy
    chosen_threshold = 0.50
    chosen_row = None
    strategy_used = "Youden_J"

    if target_sensitivity is not None:
        # Select smallest threshold meeting target sensitivity
        candidates = [row for row in table if row["sensitivity"] >= target_sensitivity]
        if candidates:
            # Pick smallest threshold meeting sensitivity with best specificity
            chosen_row = max(candidates, key=lambda r: (r["specificity"], -r["threshold"]))
            chosen_threshold = chosen_row["threshold"]
            strategy_used = f"Target_Sensitivity_{target_sensitivity:.1%}"
            print(f"[Selection Strategy] Criterion: Target Sensitivity >= {target_sensitivity:.1%}.")
        else:
            print(f"[Selection Strategy] No threshold achieved target sensitivity {target_sensitivity:.1%}. Falling back to Youden's J.")

    if chosen_row is None:
        # Fallback to Youden's J statistic
        chosen_row = max(table, key=lambda r: r["youden_j"])
        chosen_threshold = chosen_row["threshold"]
        strategy_used = "Youden_J"
        print(f"[Selection Strategy] Criterion: Maximum Youden's J statistic (Sens + Spec - 1).")

    print(f"\n=================================================================")
    print(f" Calibrated Decision Threshold: {chosen_threshold:.2f} (Criterion: {strategy_used})")
    print(f"=================================================================")
    print(f"  Sensitivity (Recall)     : {chosen_row['sensitivity']:.4f}")
    print(f"  Specificity              : {chosen_row['specificity']:.4f}")
    print(f"  PPV (Precision)          : {chosen_row['ppv']:.4f}")
    print(f"  NPV                      : {chosen_row['npv']:.4f}")
    print(f"  Youden's J               : {chosen_row['youden_j']:.4f}")
    print(f"  F1 Score                 : {chosen_row['f1']:.4f}")
    print(f"  Confusion Matrix (Val)   : TP={chosen_row['tp']:,}, FP={chosen_row['fp']:,}, TN={chosen_row['tn']:,}, FN={chosen_row['fn']:,}")
    print(f"=================================================================\n")

    # Print Table Preview (0.05 steps + chosen threshold)
    print("Threshold Tuning Table Summary (Selected Steps):")
    print(" Thresh | Sens   | Spec   | PPV    | NPV    | Youden J | F1     | Status")
    print("--------+--------+--------+--------+--------+----------+--------+------------------")
    for row in table:
        t = row["threshold"]
        is_chosen = (t == chosen_threshold)
        is_sample = (int(round(t * 100)) % 5 == 0)
        if is_chosen or is_sample:
            status = " <--- CHOSEN" if is_chosen else ""
            print(f"  {t:0.2f}  | {row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['ppv']:.4f} | {row['npv']:.4f} | {row['youden_j']:8.4f} | {row['f1']:.4f} |{status}")
    print("----------------------------------------------------------------------------------\n")

    # 4. Generate Plot
    plot_calibration_and_tradeoff(
        y_true=y_eval,
        y_prob=y_prob,
        table=table,
        chosen_threshold=chosen_threshold,
        output_plot_path=output_plot_path,
    )

    # 5. Update Metadata JSON
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta["decision_threshold"] = chosen_threshold
        meta["threshold"] = chosen_threshold
        meta["calibration"] = {
            "criterion": strategy_used,
            "brier_score": cal_metrics["brier_score"],
            "brier_skill_score": cal_metrics["brier_skill_score"],
            "expected_calibration_error": cal_metrics["expected_calibration_error"],
            "maximum_calibration_error": cal_metrics["maximum_calibration_error"],
            "calibration_assessment": calib_assessment,
            "val_metrics_at_threshold": {
                "threshold": chosen_threshold,
                "sensitivity": chosen_row["sensitivity"],
                "specificity": chosen_row["specificity"],
                "ppv": chosen_row["ppv"],
                "npv": chosen_row["npv"],
                "youden_j": chosen_row["youden_j"],
                "f1": chosen_row["f1"],
                "tp": chosen_row["tp"],
                "fp": chosen_row["fp"],
                "tn": chosen_row["tn"],
                "fn": chosen_row["fn"],
            },
            "post_hoc_scaling_note": (
                "Platt scaling (logistic sigmoid calibration) or isotonic regression flagged "
                "for future edge runtime implementation if probability scores require post-hoc adjustment."
            ),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[Metadata] Updated '{meta_path.resolve()}' with calibrated decision_threshold = {chosen_threshold:.2f}")

    # 6. Verify NumPy Model Loading
    np_model = NumPyArrhythmia1DCNN(input_length=1000)
    np_model.load_weights(weights_path)
    assert np.isclose(np_model.threshold, chosen_threshold, atol=1e-4), (
        f"NumPy model loaded threshold {np_model.threshold} != chosen threshold {chosen_threshold}!"
    )
    print(f"[Verification] Verified Arrhythmia1DCNN.load_weights({weights_path.name}) automatically adopted self.threshold = {np_model.threshold:.4f}!")

    return {
        "chosen_threshold": chosen_threshold,
        "strategy": strategy_used,
        "metrics_at_threshold": chosen_row,
        "calibration_metrics": cal_metrics,
        "plot_path": str(output_plot_path.resolve()),
        "table_path": str(output_table_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate Decision Threshold on DeepBeat Validation Split")
    parser.add_argument("--val-data", type=Path, default=ml_dir / "data" / "processed" / "deepbeat_val.npz",
                        help="Path to validation dataset .npz")
    parser.add_argument("--weights", type=Path, default=ml_dir / "model" / "weights" / "cnn_af_v1.npz",
                        help="Path to trained weights .npz")
    parser.add_argument("--meta", type=Path, default=ml_dir / "model" / "weights" / "cnn_af_v1.meta.json",
                        help="Path to sibling metadata .meta.json")
    parser.add_argument("--output-plot", type=Path, default=ml_dir / "model" / "weights" / "calibration_curve.png",
                        help="Output path for calibration and trade-off plot")
    parser.add_argument("--output-table", type=Path, default=ml_dir / "model" / "weights" / "threshold_tuning_table.csv",
                        help="Output path for full threshold table CSV")
    parser.add_argument("--target-sensitivity", type=float, default=None,
                        help="Target sensitivity constraint (e.g. 0.878 from literature, default: Youden J)")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--max-val-windows", type=int, default=None, help="Optional validation window limit")
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cpu or cuda)")

    args = parser.parse_args()

    calibrate_threshold(
        val_data_path=args.val_data,
        weights_path=args.weights,
        meta_path=args.meta,
        output_plot_path=args.output_plot,
        output_table_path=args.output_table,
        target_sensitivity=args.target_sensitivity,
        batch_size=args.batch_size,
        max_val_windows=args.max_val_windows,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
