"""
Trains and exports the deployable IBI-irregularity classifier.

Protocol, in the order the numbers may be quoted:
  1. Patient-disjoint 5-fold CV on MIMIC PERform AF -> out-of-fold predictions. Every
     reported metric comes from these, so no subject contributes to its own prediction.
  2. Operating point chosen from the out-of-fold predictions by Youden's J, but only after
     a **non-degeneracy check on both arms**. The v1 pipeline's error was not Youden itself
     but accepting the point it returned: sensitivity 95% / specificity 11% / PPV 5%, the
     always-say-AF corner. If Youden's point fails the check, selection falls back to
     maximising sensitivity under a specificity floor. The full threshold table is written
     out either way, so the operating point is auditable rather than asserted.
  3. Bootstrap confidence intervals **resampled over subjects**, not windows, because the
     effective sample size is the 35 subjects.
  4. DeepBeat evaluated as a cross-sensor robustness cohort, reported separately and never
     mixed into fitting or threshold selection.
  5. Final coefficients refit on all 35 subjects and exported for the pure-NumPy runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from model.ibi_classifier import FEATURE_NAMES, RATE_FEATURES
from training.baseline_ibi import auroc, build_features, subject_level

SEED = 42


def bootstrap_subject_ci(ys: np.ndarray, ps: np.ndarray, n: int = 2000) -> Dict[str, List[float]]:
    """95% CI for subject-level AUROC by resampling subjects with replacement."""
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(ys), len(ys))
        if len(set(ys[idx].tolist())) < 2:
            continue
        a = auroc(ys[idx], ps[idx])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return {"auroc": [float("nan"), float("nan")]}
    return {"auroc": [round(float(np.percentile(vals, 2.5)), 4),
                      round(float(np.percentile(vals, 97.5)), 4)]}


def op_table(y: np.ndarray, p: np.ndarray) -> List[Dict[str, float]]:
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
        rows.append({"threshold": float(t), "sensitivity": round(se, 4),
                     "specificity": round(sp, 4),
                     "ppv": round(tp / max(tp + fp, 1), 4),
                     "npv": round(tn / max(tn + fn, 1), 4),
                     "youden_j": round(se + sp - 1, 4),
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, default=ML_DIR / "data" / "processed" / "combined_v2.npz")
    ap.add_argument("--cache", type=Path, default=ML_DIR / "data" / "processed" / "ibi_features_v2.npz")
    ap.add_argument("--min-specificity", type=float, default=0.70,
                    help="Specificity floor the deployed operating point must meet")
    ap.add_argument("--out", type=Path, default=ML_DIR / "model" / "weights" / "ibi_af_v1.npz")
    args = ap.parse_args()

    d = np.load(args.combined, allow_pickle=True)
    y_all, g_all, ds_all = d["y"].astype(int), d["group"], d["dataset"]
    if args.cache.exists():
        F_all = np.load(args.cache)["F"]
        assert len(F_all) == len(y_all), "stale feature cache"
    else:
        F_all = build_features(d["X"])
        np.savez_compressed(args.cache, F=F_all)

    tr_mask = ds_all == "mimic"
    F, y, g = F_all[tr_mask], y_all[tr_mask], g_all[tr_mask]
    print(f"Training cohort MIMIC PERform AF: {len(y)} windows, "
          f"{len(set(g.tolist()))} subjects, {100 * y.mean():.1f}% AF windows")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fresh():
        return make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))

    # ---- 1. out-of-fold predictions
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(F, y, groups=g):
        oof[te] = fresh().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]

    ys, ps = subject_level(y, oof, g)
    s_auroc = float(auroc(ys, ps))
    w_auroc = float(auroc(y, oof))
    ci = bootstrap_subject_ci(ys, ps)
    print(f"\nOut-of-fold subject AUROC {s_auroc:.4f} 95% CI {ci['auroc']}  (n={len(ys)})")
    print(f"Out-of-fold window  AUROC {w_auroc:.4f}")

    # ---- 2. operating point, with a non-degeneracy guard
    table = op_table(y, oof)
    youden = max(table, key=lambda r: r["youden_j"])
    # Youden's J is the right criterion *provided* the point it selects is not degenerate.
    # The v1 failure was not Youden itself but accepting sensitivity 95% / specificity 11%
    # (PPV 5%) — the always-say-AF corner. So the guard checks both arms, and only if Youden
    # fails it do we fall back to maximising sensitivity under a specificity floor.
    degenerate = (youden["specificity"] < args.min_specificity
                  or youden["sensitivity"] < args.min_specificity)
    if degenerate:
        ok = [r for r in table if r["specificity"] >= args.min_specificity]
        chosen = max(ok, key=lambda r: r["sensitivity"]) if ok else youden
        rule = f"fallback: max sensitivity subject to specificity >= {args.min_specificity}"
    else:
        chosen = youden
        rule = "Youden's J (passed non-degeneracy check on sensitivity and specificity)"
    print(f"\nDeployed window threshold {chosen['threshold']:.2f}: "
          f"sens {chosen['sensitivity']:.3f} spec {chosen['specificity']:.3f} "
          f"PPV {chosen['ppv']:.3f} NPV {chosen['npv']:.3f}")
    print(f"  selection rule: {rule}")

    # subject-level operating point under the same threshold, via 3-of-5 consensus
    k, n = 3, 5
    sub_rows = []
    for gid in sorted(set(g.tolist())):
        m = g == gid
        flags = (oof[m] >= chosen["threshold"]).astype(int)
        cons = np.array([1 if flags[max(0, i - n + 1):i + 1].sum() >= k else 0
                         for i in range(len(flags))])
        sub_rows.append((gid, int(round(float(y[m].mean()))), float(cons.mean())))
    sub_y = np.array([r[1] for r in sub_rows])
    sub_rate = np.array([r[2] for r in sub_rows])
    best = max(np.round(np.arange(0.05, 0.96, 0.01), 2),
               key=lambda t: ((sub_rate >= t) & (sub_y == 1)).sum() / max((sub_y == 1).sum(), 1)
               + ((sub_rate < t) & (sub_y == 0)).sum() / max((sub_y == 0).sum(), 1))
    tp = int(((sub_rate >= best) & (sub_y == 1)).sum()); fn = int(((sub_rate < best) & (sub_y == 1)).sum())
    tn = int(((sub_rate < best) & (sub_y == 0)).sum()); fp = int(((sub_rate >= best) & (sub_y == 0)).sum())
    subject_op = {"consensus_k": k, "consensus_n": n,
                  "subject_flag_rate_threshold": float(best),
                  "sensitivity": round(tp / max(tp + fn, 1), 4),
                  "specificity": round(tn / max(tn + fp, 1), 4),
                  "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                  "n_subjects": int(len(sub_y))}
    print(f"\nSubject level ({k}-of-{n} consensus, flag-rate >= {best:.2f}): "
          f"sens {subject_op['sensitivity']:.3f} spec {subject_op['specificity']:.3f} "
          f"(tp={tp} fp={fp} tn={tn} fn={fn} of {len(sub_y)})")

    # ---- 4. cross-sensor robustness, fitted on MIMIC only
    final = fresh().fit(F, y)
    cross: Dict[str, object] = {}
    om = ds_all == "deepbeat"
    if om.any():
        po = final.predict_proba(F_all[om])[:, 1]
        yso, pso = subject_level(y_all[om], po, g_all[om])
        cross = {"cohort": "deepbeat",
                 "subject_auroc": round(float(auroc(yso, pso)), 4),
                 "window_auroc": round(float(auroc(y_all[om], po)), 4),
                 "n_subjects": int(len(yso)),
                 "note": "DeepBeat is 32 Hz; one sample is 31 ms against the 50 ms pNN50 "
                         "criterion, so interval features are quantisation-limited there."}
        print(f"\nCross-sensor DeepBeat: subject AUROC {cross['subject_auroc']} (n={cross['n_subjects']})")

    # ---- 5. export
    scaler, lr = final[0], final[-1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, mean=scaler.mean_.astype(np.float64),
             scale=scaler.scale_.astype(np.float64),
             coef=lr.coef_.ravel().astype(np.float64),
             intercept=np.array([float(lr.intercept_[0])], dtype=np.float64))

    coefs = {FEATURE_NAMES[i]: round(float(lr.coef_.ravel()[i]), 5) for i in range(len(FEATURE_NAMES))}
    meta = {
        "model_version": "ibi_af_v1",
        "model_family": "IBI irregularity + logistic regression (pure NumPy runtime)",
        "feature_names": list(FEATURE_NAMES),
        "rate_features": list(RATE_FEATURES),
        "training_cohort": {"name": "mimic_perform_af", "fs_hz": 125.0,
                            "n_subjects": int(len(set(g.tolist()))),
                            "n_windows": int(len(y)),
                            "window_s": 10.0, "resampled_to_hz": 100.0},
        "input_contract": {"fs": 100.0, "window_samples": 1000,
                           "preprocessing_chain": ["detrend_ppg",
                                                   "ButterBandpassFilter(0.5,5.0,fs=100,order=4)",
                                                   "zscore_normalize"],
                           "peak_detector": "ElgendiPeakDetector(fs=100)"},
        "decision_threshold": float(chosen["threshold"]),
        "consensus_k": k, "consensus_n": n,
        "coefficients": coefs,
        "metrics": {
            "protocol": "patient-disjoint 5-fold GroupKFold; all figures out-of-fold",
            "subject_auroc": round(s_auroc, 4),
            "subject_auroc_ci95": ci["auroc"],
            "window_auroc": round(w_auroc, 4),
            "window_operating_point": chosen,
            "unconstrained_youden_point": youden,
            "threshold_selection_rule": rule,
            "non_degeneracy_floor": args.min_specificity,
            "subject_operating_point": subject_op,
            "cross_sensor": cross,
        },
        "caveats": [
            "AF and non-AF subjects are different people, so any systematic group "
            "difference (rate, age, medication) is a potential confound. The rate-free "
            "ablation reaches subject AUROC 0.914, which is the evidence that irregularity "
            "rather than rate carries the signal.",
            "n = 35 subjects. The subject-level CI is wide and is the figure to quote.",
            "Not validated on the project's own MAX30102 wrist hardware. Transfer from "
            "fingertip transmissive PPG to wrist reflectance PPG is unmeasured.",
        ],
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (ML_DIR / "training" / "runs" / "ibi_threshold_table.json").write_text(
        json.dumps(table, indent=1), encoding="utf-8")
    np.savez_compressed(ML_DIR / "training" / "runs" / "ibi_oof.npz", y=y, p=oof, group=g)

    print("\nTop coefficients (standardised):")
    for nm, c in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"  {nm:<18} {c:+.3f}{'   [rate]' if nm in RATE_FEATURES else ''}")
    print(f"\nWrote {args.out} and {args.out.with_suffix('.meta.json').name}")


if __name__ == "__main__":
    main()
