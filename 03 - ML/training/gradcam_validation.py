"""
Grad-CAM Clinical Plausibility Validation on MIMIC PERform AF (Prompt 7 GATE)
=============================================================================
Thesis: An Internet of Things-Based Framework for Cardiac Arrhythmia Detection
        via Photoplethysmography and Machine Learning (3CPE-2A, Univ. of the East)

Specifications:
1. Load trained model (cnn_af_v1.npz) and calibrated decision threshold (0.37).
2. Load held-out external validation cohort (mimic_external_val.npz).
3. Identify:
   - 10 correctly classified AF windows (True Positives: y=1, pred=1)
   - 10 correctly classified Non-AF windows (True Negatives: y=0, pred=0)
   Sample across diverse subjects.
4. For each window, compute Grad-CAM heatmap and relevance weights.
5. Generate publication-quality visualization overlays:
   - Waveform plot with color-mapped relevance heat-strip
   - Relevance profile highlighting regions exceeding clinical saliency threshold (0.60)
   - Metadata banner: Subject ID, True Label, AF Probability, Decision, SQI.
6. Save 20 figures into: 03 - ML/training/runs/gradcam_validation/
7. Perform quantitative attribution analysis across morphological features:
   - Systolic peak attribution vs. inter-beat trough attribution vs. baseline.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

# Ensure 03 - ML is in sys.path
ml_dir = Path(__file__).resolve().parent.parent
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from model.inference_model import Arrhythmia1DCNN
from model.grad_cam import GradCAM1D


def plot_gradcam_overlay(
    t: np.ndarray,
    ppg: np.ndarray,
    cam_weights: np.ndarray,
    high_regions: List[Tuple[int, int]],
    window_idx: int,
    subj_id: str,
    true_label: int,
    pred_prob: float,
    threshold: float,
    sqi_val: float,
    output_path: Path,
    title_prefix: str = "AF True Positive",
) -> Dict[str, Any]:
    """Generates a publication-grade 2-panel figure of PPG waveform with Grad-CAM overlay."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    # 1. Top Panel: PPG Waveform with Grad-CAM Color Gradient
    points = np.array([t, ppg]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0.0, 1.0)
    cmap = plt.get_cmap("YlOrRd")
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.0)
    lc.set_array(cam_weights[:-1])
    ax_top.add_collection(lc)

    ax_top.set_xlim(t[0], t[-1])
    y_min, y_max = float(np.min(ppg)), float(np.max(ppg))
    y_pad = max(0.2, (y_max - y_min) * 0.15)
    ax_top.set_ylim(y_min - y_pad, y_max + y_pad)

    # Shaded high relevance regions on waveform
    for s_idx, e_idx in high_regions:
        ax_top.axvspan(t[s_idx], t[e_idx], color="#E65100", alpha=0.18, label="High Relevance (>=0.6)" if s_idx == high_regions[0][0] else "")

    label_str = "Atrial Fibrillation (AF)" if true_label == 1 else "Normal Sinus / Non-AF"
    pred_str = "AF POSITIVE" if pred_prob >= threshold else "NEGATIVE (Sinus)"
    status_color = "#C62828" if true_label == 1 else "#2E7D32"

    ax_top.set_title(
        f"{title_prefix} — Window #{window_idx:04d} | Subject: {subj_id}\n"
        f"Ground Truth: {label_str} | Model Prob: {pred_prob:.1%} (Threshold: {threshold:.2f} -> {pred_str}) | SQI: {sqi_val:.2f}",
        fontsize=12, fontweight="bold", pad=10, color=status_color
    )
    ax_top.set_ylabel("Normalized PPG Amplitude (Z-score)", fontsize=10)
    ax_top.grid(True, linestyle=":", alpha=0.6)

    # Colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_top, orientation="vertical", pad=0.015, aspect=15)
    cbar.set_label("Grad-CAM Activation", fontsize=9)

    # 2. Bottom Panel: Activation Relevance Profile
    ax_bot.plot(t, cam_weights, color="#D84315", linewidth=1.5, label="1D Grad-CAM Score")
    ax_bot.axhline(0.60, color="#B71C1C", linestyle="--", linewidth=1.0, alpha=0.8, label="Relevance Threshold (0.60)")
    ax_bot.fill_between(t, 0, cam_weights, color="#FFCCBC", alpha=0.5)

    for s_idx, e_idx in high_regions:
        ax_bot.axvspan(t[s_idx], t[e_idx], color="#FFAB91", alpha=0.5)

    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.set_xlabel("Time within Window (seconds @ 100 Hz)", fontsize=10)
    ax_bot.set_ylabel("Saliency Weight", fontsize=10)
    ax_bot.grid(True, linestyle=":", alpha=0.6)
    ax_bot.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()

    # Compute morphological attribution breakdown
    # Peak vs baseline analysis:
    p_thresh = np.percentile(ppg, 85)
    systolic_mask = ppg >= p_thresh
    diastolic_mask = ppg <= np.percentile(ppg, 25)
    interbeat_mask = ~systolic_mask & ~diastolic_mask

    mean_systolic_sal = float(np.mean(cam_weights[systolic_mask])) if np.any(systolic_mask) else 0.0
    mean_diastolic_sal = float(np.mean(cam_weights[diastolic_mask])) if np.any(diastolic_mask) else 0.0
    mean_interbeat_sal = float(np.mean(cam_weights[interbeat_mask])) if np.any(interbeat_mask) else 0.0

    return {
        "window_idx": int(window_idx),
        "subject_id": str(subj_id),
        "true_label": int(true_label),
        "pred_prob": float(pred_prob),
        "mean_saliency": float(np.mean(cam_weights)),
        "max_saliency": float(np.max(cam_weights)),
        "high_regions_count": len(high_regions),
        "systolic_saliency": mean_systolic_sal,
        "diastolic_saliency": mean_diastolic_sal,
        "interbeat_saliency": mean_interbeat_sal,
        "figure_path": str(output_path.resolve()),
    }


def run_clinical_validation(
    data_path: Path,
    weights_path: Path,
    out_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs Grad-CAM validation on MIMIC external validation set."""
    print("=" * 75)
    print(" Starting Grad-CAM Clinical Plausibility Validation on MIMIC PERform AF")
    print(f" Data Path   : {data_path.resolve()}")
    print(f" Weights Path: {weights_path.resolve()}")
    print(f" Output Dir  : {out_dir.resolve()}")
    print("=" * 75)

    data = np.load(data_path, mmap_mode="r")
    X = data["X"]
    y = data["y"]
    subjs = data["subject_id"]
    sqi = data["sqi"]

    model = Arrhythmia1DCNN()
    model.load_weights(weights_path)
    threshold = model.threshold
    explainer = GradCAM1D(model)

    print(f"[Model] Loaded weights: {model.weights_source}")
    print(f"[Model] Operating decision threshold: {threshold:.4f}")

    # Forward pass across MIMIC windows to classify
    print(f"[Inference] Classifying {len(y):,} MIMIC windows...")
    probs = np.zeros(len(y), dtype=np.float32)
    for i in range(len(y)):
        cache = model.forward_with_cache(X[i])
        probs[i] = cache["prob"]

    preds = (probs >= threshold).astype(np.int8)

    # Identify candidate pools
    tp_indices = np.where((y == 1) & (preds == 1))[0]
    tn_indices = np.where((y == 0) & (preds == 0))[0]

    print(f"[Cohort] True Positives (AF correctly classified)   : {len(tp_indices):,}")
    print(f"[Cohort] True Negatives (Sinus correctly classified): {len(tn_indices):,}")

    rng = np.random.default_rng(seed)

    # Sample 10 diverse subjects for TP
    tp_subjs = np.unique(subjs[tp_indices])
    chosen_tp = []
    # Try picking one from distinct subjects first
    shuffled_tp_subjs = rng.permutation(tp_subjs)
    for s in shuffled_tp_subjs:
        s_tp_idx = tp_indices[subjs[tp_indices] == s]
        if len(s_tp_idx) > 0:
            chosen_tp.append(int(rng.choice(s_tp_idx)))
        if len(chosen_tp) == 10:
            break
    # Fill remainder if fewer than 10 unique subjects
    if len(chosen_tp) < 10:
        remaining = list(set(tp_indices) - set(chosen_tp))
        chosen_tp.extend(rng.choice(remaining, size=10 - len(chosen_tp), replace=False).tolist())

    # Sample 10 diverse subjects for TN
    tn_subjs = np.unique(subjs[tn_indices])
    chosen_tn = []
    shuffled_tn_subjs = rng.permutation(tn_subjs)
    for s in shuffled_tn_subjs:
        s_tn_idx = tn_indices[subjs[tn_indices] == s]
        if len(s_tn_idx) > 0:
            chosen_tn.append(int(rng.choice(s_tn_idx)))
        if len(chosen_tn) == 10:
            break
    if len(chosen_tn) < 10:
        remaining = list(set(tn_indices) - set(chosen_tn))
        chosen_tn.extend(rng.choice(remaining, size=10 - len(chosen_tn), replace=False).tolist())

    t = np.linspace(0, 10.0, 1000)

    tp_records = []
    print("\n[Grad-CAM] Generating 10 AF True Positive Overlay Figures...")
    for rank, idx in enumerate(chosen_tp, 1):
        x_win = X[idx]
        res = explainer.explain(x_win)
        weights = np.array(res["weights"])
        out_fig = out_dir / f"af_tp_{rank:02d}_win{idx:04d}_{subjs[idx]}.png"

        rec = plot_gradcam_overlay(
            t=t,
            ppg=x_win,
            cam_weights=weights,
            high_regions=res["high_relevance_regions"],
            window_idx=idx,
            subj_id=str(subjs[idx]),
            true_label=int(y[idx]),
            pred_prob=float(probs[idx]),
            threshold=threshold,
            sqi_val=float(sqi[idx]),
            output_path=out_fig,
            title_prefix=f"AF True Positive ({rank}/10)",
        )
        tp_records.append(rec)
        print(f"  [{rank:02d}/10] AF Window #{idx:04d} (Subj: {subjs[idx]}): Prob={probs[idx]:.1%}, Regions={len(res['high_relevance_regions'])}, Output={out_fig.name}")

    tn_records = []
    print("\n[Grad-CAM] Generating 10 Non-AF True Negative Overlay Figures...")
    for rank, idx in enumerate(chosen_tn, 1):
        x_win = X[idx]
        res = explainer.explain(x_win)
        weights = np.array(res["weights"])
        out_fig = out_dir / f"non_af_tn_{rank:02d}_win{idx:04d}_{subjs[idx]}.png"

        rec = plot_gradcam_overlay(
            t=t,
            ppg=x_win,
            cam_weights=weights,
            high_regions=res["high_relevance_regions"],
            window_idx=idx,
            subj_id=str(subjs[idx]),
            true_label=int(y[idx]),
            pred_prob=float(probs[idx]),
            threshold=threshold,
            sqi_val=float(sqi[idx]),
            output_path=out_fig,
            title_prefix=f"Non-AF True Negative ({rank}/10)",
        )
        tn_records.append(rec)
        print(f"  [{rank:02d}/10] Non-AF Window #{idx:04d} (Subj: {subjs[idx]}): Prob={probs[idx]:.1%}, Regions={len(res['high_relevance_regions'])}, Output={out_fig.name}")

    # Aggregate Statistics
    def agg_stats(records: List[Dict[str, Any]]):
        return {
            "mean_saliency": float(np.mean([r["mean_saliency"] for r in records])),
            "systolic_saliency": float(np.mean([r["systolic_saliency"] for r in records])),
            "diastolic_saliency": float(np.mean([r["diastolic_saliency"] for r in records])),
            "interbeat_saliency": float(np.mean([r["interbeat_saliency"] for r in records])),
            "avg_high_regions": float(np.mean([r["high_regions_count"] for r in records])),
        }

    tp_stats = agg_stats(tp_records)
    tn_stats = agg_stats(tn_records)

    print("\n" + "=" * 75)
    print(" Grad-CAM Clinical Attribution Summary & Plausibility Evaluation")
    print("=" * 75)
    print(" Atrial Fibrillation (AF) Windows:")
    print(f"   Mean Saliency Intensity: {tp_stats['mean_saliency']:.4f}")
    print(f"   Systolic Peak Saliency : {tp_stats['systolic_saliency']:.4f}")
    print(f"   Diastolic Saliency     : {tp_stats['diastolic_saliency']:.4f}")
    print(f"   Inter-Beat Saliency    : {tp_stats['interbeat_saliency']:.4f}")
    print(f"   Avg Salient Regions/Win: {tp_stats['avg_high_regions']:.1f}")
    print(" Non-AF (Sinus Rhythm) Windows:")
    print(f"   Mean Saliency Intensity: {tn_stats['mean_saliency']:.4f}")
    print(f"   Systolic Peak Saliency : {tn_stats['systolic_saliency']:.4f}")
    print(f"   Diastolic Saliency     : {tn_stats['diastolic_saliency']:.4f}")
    print(f"   Inter-Beat Saliency    : {tn_stats['interbeat_saliency']:.4f}")
    print(f"   Avg Salient Regions/Win: {tn_stats['avg_high_regions']:.1f}")
    print("=" * 75)

    summary_payload = {
        "timestamp": Path(__file__).stat().st_mtime,
        "model_weights": str(weights_path.resolve()),
        "threshold": threshold,
        "af_true_positives": tp_records,
        "non_af_true_negatives": tn_records,
        "attribution_statistics": {
            "af_tp": tp_stats,
            "non_af_tn": tn_stats,
        },
        "figures_directory": str(out_dir.resolve()),
    }

    out_json = out_dir / "gradcam_validation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[Saved] Summary report written to: {out_json.resolve()}")
    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM Clinical Validation on MIMIC PERform AF")
    parser.add_argument("--data", type=Path, default=ml_dir / "data" / "processed" / "mimic_external_val.npz")
    parser.add_argument("--weights", type=Path, default=ml_dir / "model" / "weights" / "cnn_af_v1.npz")
    parser.add_argument("--out-dir", type=Path, default=ml_dir / "training" / "runs" / "gradcam_validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_clinical_validation(
        data_path=args.data,
        weights_path=args.weights,
        out_dir=args.out_dir,
        seed=args.seed,
    )
