"""
Model Evaluation Suite: Internal Benchmark & External Validation (ANTIGRAVITY.md §4.3)
=====================================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Specifications:
1. Internal Benchmark: DeepBeat held-out cardiologist-adjudicated test partition
   (03 - ML/data/processed/deepbeat_test.npz: 17,106 windows, 14 subjects).
2. External Validation: Full MIMIC PERform AF held-out cohort
   (03 - ML/data/processed/mimic_external_val.npz: 4,200 windows, 35 subjects).
   - Independent population, transmissive fingertip sensor, unseen during training & tuning.
3. Window-level metrics: AUROC, AUPRC, Sens, Spec, PPV, NPV, F1, Accuracy, Confusion Matrix.
   - 95% Confidence Intervals via SUBJECT-LEVEL bootstrap resampling (1,000 iterations).
4. Subject-level metrics: Aggregated by majority vote and mean probability.
5. Signal-Quality Stratification: Performance breakdown across quality labels and SQI bins.
6. Publication-grade diagnostic plots: ROC, PR, Confusion Matrix, Calibration curves.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# Ensure 03 - ML is on sys.path
ml_dir = Path(__file__).resolve().parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from training.torch_model import Arrhythmia1DCNN
from training.calibrate_threshold import load_numpy_weights_into_torch


def compute_window_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """Computes all standard binary classification metrics at a given threshold."""
    preds = (y_prob >= threshold).astype(np.int8)

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    total = len(y_true)

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    acc = float((tp + tn) / total) if total > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    youden = sens + spec - 1.0

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos > 0 and n_neg > 0:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = 0.50
        auprc = 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "accuracy": acc,
        "f1": f1,
        "youden_j": youden,
        "brier_score": brier,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def compute_subject_bootstrap_cis(
    subjs: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% Confidence Intervals via SUBJECT-LEVEL bootstrap resampling.
    Resamples whole subjects with replacement to account for intra-subject correlation.
    """
    unique_subjs = np.unique(subjs)
    n_subjs = len(unique_subjs)
    subj_to_idx = {s: np.where(subjs == s)[0] for s in unique_subjs}

    rng = np.random.default_rng(seed)
    boot_metrics: Dict[str, List[float]] = {
        "auroc": [], "auprc": [], "sensitivity": [], "specificity": [],
        "ppv": [], "npv": [], "accuracy": [], "f1": [],
    }

    valid_iterations = 0
    for _ in range(n_bootstraps * 2):
        if valid_iterations >= n_bootstraps:
            break

        sampled_subjs = rng.choice(unique_subjs, size=n_subjs, replace=True)
        sampled_indices = np.concatenate([subj_to_idx[s] for s in sampled_subjs])

        y_sample = y_true[sampled_indices]
        prob_sample = y_prob[sampled_indices]

        # Must have both classes to calculate AUROC
        if len(np.unique(y_sample)) < 2:
            continue

        m = compute_window_metrics(y_sample, prob_sample, threshold=threshold)
        for k in boot_metrics.keys():
            boot_metrics[k].append(m[k])
        valid_iterations += 1

    ci_dict = {}
    for k, values in boot_metrics.items():
        if values:
            low = float(np.percentile(values, 2.5))
            high = float(np.percentile(values, 97.5))
            ci_dict[k] = (round(low, 4), round(high, 4))
        else:
            ci_dict[k] = (0.0, 0.0)

    return ci_dict


def compute_subject_level_metrics(
    subjs: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Aggregates window-level predictions to subject-level predictions via:
    1. Majority vote (> 50% windows predicted AF)
    2. Mean probability (mean window probability >= threshold)
    """
    unique_subjs = sorted(list(set(subjs)))
    subj_results = []

    for s in unique_subjs:
        mask = (subjs == s)
        subj_y = y_true[mask]
        subj_probs = y_prob[mask]

        # True subject label (1 if >= 50% AF or subject diagnosed with AF)
        true_label = int(np.mean(subj_y) >= 0.50)

        # Majority vote
        window_preds = (subj_probs >= threshold).astype(np.int8)
        vote_pred = int(np.mean(window_preds) >= 0.50)

        # Mean probability
        mean_prob = float(np.mean(subj_probs))
        prob_pred = int(mean_prob >= threshold)

        subj_results.append({
            "subject_id": str(s),
            "true_label": true_label,
            "n_windows": int(np.sum(mask)),
            "af_window_pct": float(np.mean(subj_y) * 100),
            "mean_prob": mean_prob,
            "vote_pred": vote_pred,
            "prob_pred": prob_pred,
        })

    # Summary metrics helper
    def _eval_subj_preds(preds: List[int], trues: List[int]) -> Dict[str, Any]:
        p = np.array(preds, dtype=np.int8)
        t = np.array(trues, dtype=np.int8)
        tp = int(np.sum((p == 1) & (t == 1)))
        fp = int(np.sum((p == 1) & (t == 0)))
        tn = int(np.sum((p == 0) & (t == 0)))
        fn = int(np.sum((p == 0) & (t == 1)))
        total = len(t)
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        acc = float((tp + tn) / total) if total > 0 else 0.0
        return {
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "accuracy": round(acc, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total,
        }

    trues = [r["true_label"] for r in subj_results]
    vote_preds = [r["vote_pred"] for r in subj_results]
    prob_preds = [r["prob_pred"] for r in subj_results]

    return {
        "n_subjects": len(unique_subjs),
        "majority_vote": _eval_subj_preds(vote_preds, trues),
        "mean_probability": _eval_subj_preds(prob_preds, trues),
        "subjects": subj_results,
    }


def compute_quality_stratified_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dataset_quality: Optional[np.ndarray],
    sqi_scores: Optional[np.ndarray],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluates performance stratified by stored dataset quality and SQI scores."""
    strat_dict = {}

    # Stratify by Dataset Quality (0: Good, 1: Marginal, 2: Poor)
    if dataset_quality is not None and len(dataset_quality) == len(y_true) and np.any(dataset_quality >= 0):
        qual_map = {0: "Good", 1: "Marginal", 2: "Poor"}
        strat_dict["by_dataset_quality"] = {}
        for q_val, q_name in qual_map.items():
            mask = (dataset_quality == q_val)
            if np.sum(mask) > 0:
                m = compute_window_metrics(y_true[mask], y_prob[mask], threshold=threshold)
                strat_dict["by_dataset_quality"][q_name] = {
                    "count": int(np.sum(mask)),
                    "pct_of_total": round(float(np.mean(mask) * 100), 2),
                    "auroc": round(m["auroc"], 4),
                    "auprc": round(m["auprc"], 4),
                    "sensitivity": round(m["sensitivity"], 4),
                    "specificity": round(m["specificity"], 4),
                }

    # Stratify by SQI Score (< 0.50 vs >= 0.50)
    if sqi_scores is not None and len(sqi_scores) == len(y_true):
        strat_dict["by_sqi"] = {}
        sqi_masks = {
            "Low_SQI (<0.50)": (sqi_scores < 0.50),
            "High_SQI (>=0.50)": (sqi_scores >= 0.50),
        }
        for s_name, mask in sqi_masks.items():
            if np.sum(mask) > 0:
                m = compute_window_metrics(y_true[mask], y_prob[mask], threshold=threshold)
                strat_dict["by_sqi"][s_name] = {
                    "count": int(np.sum(mask)),
                    "pct_of_total": round(float(np.mean(mask) * 100), 2),
                    "auroc": round(m["auroc"], 4),
                    "auprc": round(m["auprc"], 4),
                    "sensitivity": round(m["sensitivity"], 4),
                    "specificity": round(m["specificity"], 4),
                }

    return strat_dict


def generate_diagnostic_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    split_label: str,
    output_dir: Path,
    window_metrics: Dict[str, Any],
    window_cis: Dict[str, Tuple[float, float]],
    subj_metrics: Dict[str, Any],
):
    """Generates ROC, PR, Confusion Matrix, and Reliability plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = window_metrics["auroc"]
    auc_ci = window_cis.get("auroc", (0.0, 0.0))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, color="#0E6B76", lw=2, label=f"1D-CNN (AUC = {auroc:.3f} [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}])")
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", lw=1.5, label="Chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_title(f"ROC Curve — {split_label}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = window_metrics["auprc"]
    pr_ci = window_cis.get("auprc", (0.0, 0.0))
    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(rec, prec, color="#D9534F", lw=2, label=f"1D-CNN (AUPRC = {auprc:.3f} [{pr_ci[0]:.3f}–{pr_ci[1]:.3f}])")
    ax.axhline(prevalence, color="#999", linestyle="--", lw=1.5, label=f"Prevalence ({prevalence:.1%})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Precision (PPV)", fontsize=10, fontweight="bold")
    ax.set_title(f"Precision-Recall Curve — {split_label}", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png")
    plt.close()

    # 3. Confusion Matrix (Window-level & Subject-level side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)

    # Window CM
    cm_w = np.array([[window_metrics["tn"], window_metrics["fp"]],
                     [window_metrics["fn"], window_metrics["tp"]]])
    im1 = ax1.imshow(cm_w, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title(f"Window-Level (N={len(y_true):,})\nThresh = {threshold:.2f}", fontsize=10, fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Non-AF", "AF"])
    ax1.set_yticklabels(["Non-AF", "AF"])
    ax1.set_ylabel("True Label", fontweight="bold")
    ax1.set_xlabel("Predicted Label", fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm_w[i, j]
            ax1.text(j, i, f"{val:,}\n({val/len(y_true)*100:.1f}%)",
                     ha="center", va="center", color="white" if val > cm_w.max()/2 else "black", fontsize=9)

    # Subject CM (Majority Vote)
    sub_mv = subj_metrics["majority_vote"]
    cm_s = np.array([[sub_mv["tn"], sub_mv["fp"]],
                     [sub_mv["fn"], sub_mv["tp"]]])
    im2 = ax2.imshow(cm_s, interpolation="nearest", cmap=plt.cm.Greens)
    ax2.set_title(f"Subject-Level (N={sub_mv['total']})\n(Majority Vote)", fontsize=10, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Non-AF", "AF"])
    ax2.set_yticklabels(["Non-AF", "AF"])
    ax2.set_ylabel("True Label", fontweight="bold")
    ax2.set_xlabel("Predicted Label", fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm_s[i, j]
            ax2.text(j, i, f"{val}\n({val/sub_mv['total']*100:.1f}%)",
                     ha="center", va="center", color="white" if val > cm_s.max()/2 else "black", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    # 4. Reliability Diagram
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    ax.plot(prob_pred, prob_true, "s-", color="#0E6B76", label=f"{split_label}", lw=2)
    ax.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
    ax.set_ylabel("Fraction of Positives (Empirical AF)", fontsize=10, fontweight="bold")
    ax.set_title(f"Reliability Diagram — {split_label}", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_diagram.png")
    plt.close()

    print(f"[Plots] Generated diagnostic plots in: {output_dir.resolve()}")


def evaluate_split(
    split_key: str,
    data_path: Path,
    weights_path: Path,
    meta_path: Path,
    runs_dir: Path,
    batch_size: int = 128,
    n_bootstraps: int = 1000,
    seed: int = 42,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates a single dataset split and writes outputs."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path.resolve()}")

    # Determine split role
    is_internal = ("deepbeat" in split_key)
    split_display = "DeepBeat Test (Internal Cardiologist-Adjudicated Benchmark)" if is_internal else "MIMIC PERform AF (Held-Out External Validation)"

    # Read calibrated threshold from metadata
    threshold = 0.50
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            threshold = float(meta.get("decision_threshold", meta.get("threshold", 0.50)))

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*75}")
    print(f" Starting Evaluation: {split_display}")
    print(f" Data Path : {data_path.resolve()}")
    print(f" Weights   : {weights_path.resolve()}")
    print(f" Threshold : {threshold:.4f}")
    print(f" Device    : {device}")
    print(f"{'='*75}")

    # Load arrays
    raw_data = np.load(data_path, mmap_mode="r")
    X = raw_data["X"]
    y_true = np.array(raw_data["y"], dtype=np.int8)
    subjs = np.array(raw_data["subject_id"])
    dataset_qual = raw_data["dataset_quality"] if "dataset_quality" in raw_data else None
    sqi_scores = raw_data["sqi"] if "sqi" in raw_data else None

    # Load PyTorch model
    torch_model = Arrhythmia1DCNN(input_length=1000, dropout=0.0).to(device)
    load_numpy_weights_into_torch(torch_model, weights_path)

    # Batch inference
    n_batches = (len(y_true) + batch_size - 1) // batch_size
    all_probs = []
    print(f"[Inference] Running forward pass across {len(y_true):,} windows...")

    with torch.no_grad():
        for b in range(n_batches):
            s = b * batch_size
            e = min(s + batch_size, len(y_true))
            chunk = torch.from_numpy(np.array(X[s:e], dtype=np.float32)).unsqueeze(1).to(device)
            p = torch_model(chunk, return_logits=False).cpu().numpy().flatten()
            all_probs.extend(p)

    y_prob = np.array(all_probs, dtype=np.float32)

    # 1. Window-level metrics
    window_m = compute_window_metrics(y_true, y_prob, threshold=threshold)

    # 2. Subject-level 95% CIs
    print(f"[Bootstrap] Computing 95% CIs via subject-level resampling ({n_bootstraps} iterations)...")
    window_cis = compute_subject_bootstrap_cis(
        subjs=subjs,
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # 3. Subject-level aggregated metrics
    subj_m = compute_subject_level_metrics(subjs, y_true, y_prob, threshold=threshold)

    # 4. Quality stratification
    quality_strat = compute_quality_stratified_metrics(
        y_true=y_true,
        y_prob=y_prob,
        dataset_quality=dataset_qual,
        sqi_scores=sqi_scores,
        threshold=threshold,
    )

    # 5. Diagnostic Plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = runs_dir / f"{timestamp}_eval_{split_key}"
    generate_diagnostic_plots(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        split_label=split_key.replace("_", " ").upper(),
        output_dir=out_dir,
        window_metrics=window_m,
        window_cis=window_cis,
        subj_metrics=subj_m,
    )

    # Print Summary Table
    print(f"\n{'-'*75}")
    print(f" Performance Summary: {split_display}")
    print(f"{'-'*75}")
    print(f" Window-Level Metrics (N={len(y_true):,} | Decision Threshold = {threshold:.2f}):")
    print(f"   AUROC          : {window_m['auroc']:.4f}  (95% CI: [{window_cis['auroc'][0]:.4f} – {window_cis['auroc'][1]:.4f}])")
    print(f"   AUPRC          : {window_m['auprc']:.4f}  (95% CI: [{window_cis['auprc'][0]:.4f} – {window_cis['auprc'][1]:.4f}])")
    print(f"   Sensitivity    : {window_m['sensitivity']:.4f}  (95% CI: [{window_cis['sensitivity'][0]:.4f} – {window_cis['sensitivity'][1]:.4f}])")
    print(f"   Specificity    : {window_m['specificity']:.4f}  (95% CI: [{window_cis['specificity'][0]:.4f} – {window_cis['specificity'][1]:.4f}])")
    print(f"   PPV (Precision): {window_m['ppv']:.4f}  (95% CI: [{window_cis['ppv'][0]:.4f} – {window_cis['ppv'][1]:.4f}])")
    print(f"   NPV            : {window_m['npv']:.4f}  (95% CI: [{window_cis['npv'][0]:.4f} – {window_cis['npv'][1]:.4f}])")
    print(f"   F1-Score       : {window_m['f1']:.4f}  (95% CI: [{window_cis['f1'][0]:.4f} – {window_cis['f1'][1]:.4f}])")
    print(f"   Accuracy       : {window_m['accuracy']:.4f}  (95% CI: [{window_cis['accuracy'][0]:.4f} – {window_cis['accuracy'][1]:.4f}])")
    print(f"   Confusion Matrix: TP={window_m['tp']:,}, FP={window_m['fp']:,}, TN={window_m['tn']:,}, FN={window_m['fn']:,}")
    print(f"\n Subject-Level Metrics (N={subj_m['n_subjects']} Subjects):")
    smv = subj_m["majority_vote"]
    print(f"   [Majority Vote]    Sens: {smv['sensitivity']:.4f} | Spec: {smv['specificity']:.4f} | Acc: {smv['accuracy']:.4f} (TP={smv['tp']}, FP={smv['fp']}, TN={smv['tn']}, FN={smv['fn']})")
    smp = subj_m["mean_probability"]
    print(f"   [Mean Probability] Sens: {smp['sensitivity']:.4f} | Spec: {smp['specificity']:.4f} | Acc: {smp['accuracy']:.4f} (TP={smp['tp']}, FP={smp['fp']}, TN={smp['tn']}, FN={smp['fn']})")

    if "by_dataset_quality" in quality_strat:
        print(f"\n Performance Stratified by Dataset Quality:")
        for q_name, q_data in quality_strat["by_dataset_quality"].items():
            print(f"   Quality {q_name:8s}: N={q_data['count']:5d} ({q_data['pct_of_total']:4.1f}%) | AUROC={q_data['auroc']:.4f} | Sens={q_data['sensitivity']:.4f} | Spec={q_data['specificity']:.4f}")

    if "by_sqi" in quality_strat:
        print(f"\n Performance Stratified by Signal Quality Index (SQI):")
        for s_name, s_data in quality_strat["by_sqi"].items():
            print(f"   {s_name:18s}: N={s_data['count']:5d} ({s_data['pct_of_total']:4.1f}%) | AUROC={s_data['auroc']:.4f} | Sens={s_data['sensitivity']:.4f} | Spec={s_data['specificity']:.4f}")

    print(f"{'-'*75}\n")

    # Save JSON results
    result_payload = {
        "split_key": split_key,
        "split_display": split_display,
        "evaluation_timestamp": timestamp,
        "threshold": threshold,
        "data_path": str(data_path.resolve()),
        "weights_path": str(weights_path.resolve()),
        "window_metrics": window_m,
        "window_bootstrap_cis": window_cis,
        "subject_metrics": {
            "n_subjects": subj_m["n_subjects"],
            "majority_vote": subj_m["majority_vote"],
            "mean_probability": subj_m["mean_probability"],
            "subject_breakdown": subj_m["subjects"],
        },
        "quality_stratification": quality_strat,
        "plots_directory": str(out_dir.resolve()),
    }

    res_json_path = out_dir / f"eval_{split_key}_results.json"
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1D-CNN on Internal Benchmark and External Validation")
    parser.add_argument("--split", type=str, default="all",
                        choices=["deepbeat_test", "mimic_external", "all"],
                        help="Which dataset split to evaluate (default: all)")
    parser.add_argument("--weights", type=Path, default=ml_dir / "model" / "weights" / "cnn_af_v1.npz",
                        help="Path to model weights .npz")
    parser.add_argument("--meta", type=Path, default=ml_dir / "model" / "weights" / "cnn_af_v1.meta.json",
                        help="Path to sibling metadata .meta.json")
    parser.add_argument("--runs-dir", type=Path, default=ml_dir / "training" / "runs",
                        help="Runs output directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--n-bootstraps", type=int, default=1000, help="Subject-level bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cpu or cuda)")

    args = parser.parse_args()

    splits_to_run = []
    if args.split in ["deepbeat_test", "all"]:
        splits_to_run.append(("deepbeat_test", ml_dir / "data" / "processed" / "deepbeat_test.npz"))
    if args.split in ["mimic_external", "all"]:
        splits_to_run.append(("mimic_external", ml_dir / "data" / "processed" / "mimic_external_val.npz"))

    for s_key, s_path in splits_to_run:
        evaluate_split(
            split_key=s_key,
            data_path=s_path,
            weights_path=args.weights,
            meta_path=args.meta,
            runs_dir=args.runs_dir,
            batch_size=args.batch_size,
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
            device_name=args.device,
        )


if __name__ == "__main__":
    main()
