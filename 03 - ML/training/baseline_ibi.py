"""
Interpretable IBI-irregularity baseline — establishes the achievable AF signal in the data
before any deep model is trained, and answers RQ1 directly ("what PPG signal inputs /
physiological parameters are needed for reliable AF detection").

Rationale: AF's cardinal PPG signature is beat-to-beat interval irregularity, not waveform
morphology. A handful of interval statistics fed to logistic regression has ~12 free
parameters against the 1D-CNN's 1.03M, so it cannot memorise recording identity. If this
baseline separates AF from non-AF under patient-disjoint cross-validation, the labels and
signals are sound and a CNN is worth training. If it does not, no CNN result on this data
would be trustworthy either.

Evaluation protocol used throughout (and reused by the CNN so comparisons are fair):
  * `GroupKFold` by recording group — a patient never appears in both train and test.
  * **Subject-level** metrics are primary: window labels are constant within a group, so
    window counts overstate the sample size. Subject AUROC is computed over mean predicted
    probability per group.
  * Leave-one-dataset-out quantifies cross-sensor transfer (wrist reflectance <-> fingertip
    transmissive) separately from in-distribution skill.
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

from signal_processing.peak_detection import ElgendiPeakDetector

FS = 100.0

FEATURE_NAMES = [
    "n_beats", "mean_ibi", "sd_ibi", "rmssd", "cv_ibi", "pnn50",
    "ibi_range", "ibi_iqr", "shannon_dibi", "mean_hr", "sd_hr", "median_dibi_abs",
]


# ------------------------------------------------------------------------- features


def ibi_features(window: np.ndarray, det: ElgendiPeakDetector) -> np.ndarray:
    """Interval-irregularity statistics for one 10 s window. NaN-free by construction."""
    peaks = det.detect_peaks(np.asarray(window, dtype=np.float64))
    f = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    if peaks is None or len(peaks) < 4:
        return f  # too few beats to characterise rhythm

    ibi = np.diff(np.asarray(peaks, dtype=np.float64)) / FS  # seconds
    ibi = ibi[(ibi > 0.25) & (ibi < 2.5)]  # physiological 24-240 bpm
    if len(ibi) < 3:
        return f

    dibi = np.diff(ibi)
    mean_ibi = float(np.mean(ibi))
    hr = 60.0 / ibi

    # Shannon entropy of the successive-difference distribution: high in AF.
    hist, _ = np.histogram(dibi, bins=8, range=(-0.5, 0.5))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    shannon = float(-np.sum(p * np.log2(p))) if p.size else 0.0

    f[:] = [
        len(ibi) + 1,
        mean_ibi,
        float(np.std(ibi)),
        float(np.sqrt(np.mean(dibi ** 2))),
        float(np.std(ibi) / mean_ibi) if mean_ibi > 0 else 0.0,
        float(np.mean(np.abs(dibi) > 0.05)),
        float(np.max(ibi) - np.min(ibi)),
        float(np.subtract(*np.percentile(ibi, [75, 25]))),
        shannon,
        float(np.mean(hr)),
        float(np.std(hr)),
        float(np.median(np.abs(dibi))),
    ]
    return f


def build_features(X: np.ndarray) -> np.ndarray:
    det = ElgendiPeakDetector(fs=FS)
    F = np.zeros((len(X), len(FEATURE_NAMES)), dtype=np.float64)
    for i in range(len(X)):
        F[i] = ibi_features(X[i], det)
        if (i + 1) % 5000 == 0:
            print(f"    features {i + 1}/{len(X)}", flush=True)
    return F


# -------------------------------------------------------------------------- metrics


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def subject_level(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gs = np.array(sorted(set(g.tolist())))
    ys = np.array([int(round(float(y[g == k].mean()))) for k in gs])
    ps = np.array([float(p[g == k].mean()) for k in gs])
    return ys, ps


def report(tag: str, y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Dict[str, object]:
    ys, ps = subject_level(y, p, g)
    sa = auroc(ys, ps)
    wa = auroc(y, p)
    # Youden-optimal subject threshold, reported with the operating point it implies
    best_j, best_t = -2.0, 0.5
    for t in np.unique(np.round(ps, 4)):
        tp = int(((ps >= t) & (ys == 1)).sum()); fn = int(((ps < t) & (ys == 1)).sum())
        tn = int(((ps < t) & (ys == 0)).sum()); fp = int(((ps >= t) & (ys == 0)).sum())
        se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
        if se + sp - 1 > best_j:
            best_j, best_t = se + sp - 1, float(t)
    tp = int(((ps >= best_t) & (ys == 1)).sum()); fn = int(((ps < best_t) & (ys == 1)).sum())
    tn = int(((ps < best_t) & (ys == 0)).sum()); fp = int(((ps >= best_t) & (ys == 0)).sum())
    out = {
        "window_auroc": round(wa, 4),
        "subject_auroc": round(sa, 4),
        "n_subjects": int(len(ys)),
        "subject_threshold": round(best_t, 4),
        "subject_sens": round(tp / max(tp + fn, 1), 4),
        "subject_spec": round(tn / max(tn + fp, 1), 4),
        "subject_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    print(f"  {tag:<28} subj AUROC {out['subject_auroc']:.3f} (n={out['n_subjects']})  "
          f"win AUROC {out['window_auroc']:.3f}  "
          f"sens {out['subject_sens']:.2f} spec {out['subject_spec']:.2f}")
    return out


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ML_DIR / "data" / "processed" / "combined_v2.npz")
    ap.add_argument("--cache", type=Path, default=ML_DIR / "data" / "processed" / "ibi_features_v2.npz")
    ap.add_argument("--out", type=Path, default=ML_DIR / "training" / "runs" / "baseline_ibi.json")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, g, ds = d["X"], d["y"].astype(int), d["group"], d["dataset"]

    if args.cache.exists():
        print(f"Loading cached features from {args.cache.name}")
        F = np.load(args.cache)["F"]
        assert len(F) == len(X), "feature cache is stale; delete it and rerun"
    else:
        print(f"Extracting IBI features for {len(X)} windows...")
        F = build_features(X)
        np.savez_compressed(args.cache, F=F)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    def fresh():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        )

    results: Dict[str, object] = {"feature_names": FEATURE_NAMES}

    print(f"\n[1] Patient-disjoint {args.folds}-fold CV on the combined pool")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=args.folds).split(F, y, groups=g):
        m = fresh().fit(F[tr], y[tr])
        oof[te] = m.predict_proba(F[te])[:, 1]
    results["combined_cv"] = report("combined CV (out-of-fold)", y, oof, g)
    for name in ("deepbeat", "mimic"):
        m = ds == name
        results[f"combined_cv_{name}"] = report(f"  ... restricted to {name}", y[m], oof[m], g[m])

    print("\n[2] Within-dataset patient-disjoint CV")
    for name in ("deepbeat", "mimic"):
        m = ds == name
        oof_d = np.zeros(int(m.sum()))
        Fm, ym, gm = F[m], y[m], g[m]
        n_sp = min(args.folds, len(set(gm.tolist())))
        for tr, te in GroupKFold(n_splits=n_sp).split(Fm, ym, groups=gm):
            oof_d[te] = fresh().fit(Fm[tr], ym[tr]).predict_proba(Fm[te])[:, 1]
        results[f"within_{name}"] = report(f"{name} only", ym, oof_d, gm)

    print("\n[3] Leave-one-dataset-out (cross-sensor transfer)")
    for tr_name, te_name in (("deepbeat", "mimic"), ("mimic", "deepbeat")):
        tr, te = ds == tr_name, ds == te_name
        p = fresh().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        results[f"train_{tr_name}_test_{te_name}"] = report(f"train {tr_name} -> test {te_name}", y[te], p, g[te])

    print("\n[4] Feature directions (combined fit, standardised coefficients)")
    m = fresh().fit(F, y)
    coefs = m[-1].coef_[0]
    order = np.argsort(-np.abs(coefs))
    results["coefficients"] = {FEATURE_NAMES[i]: round(float(coefs[i]), 4) for i in order}
    for i in order[:8]:
        print(f"  {FEATURE_NAMES[i]:<18} {coefs[i]:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__" and "--ablation" not in sys.argv:
    main()


# ---------------------------------------------------------------- ablation entrypoint

def ablation() -> None:
    """
    Confound check: is the baseline detecting interval *irregularity* (the AF mechanism)
    or merely heart *rate* (a population confound, since AF and non-AF subjects in these
    cohorts are different people who may differ in rate, age and medication)?
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    d = np.load(ML_DIR / "data" / "processed" / "combined_v2.npz", allow_pickle=True)
    y, g, ds = d["y"].astype(int), d["group"], d["dataset"]
    F = np.load(ML_DIR / "data" / "processed" / "ibi_features_v2.npz")["F"]
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}

    sets = {
        "all 12 features":        FEATURE_NAMES,
        "rate only":             ["n_beats", "mean_ibi", "mean_hr"],
        "irregularity only":     ["cv_ibi", "pnn50", "shannon_dibi"],
        "irregularity + spread": ["cv_ibi", "pnn50", "shannon_dibi", "sd_ibi", "rmssd", "ibi_range"],
    }

    def fresh():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))

    for cohort in ("mimic", "deepbeat"):
        m = ds == cohort
        ym, gm = y[m], g[m]
        print(f"\n=== {cohort} (patient-disjoint CV, n_subjects={len(set(gm.tolist()))}) ===")
        for label, names in sets.items():
            cols = [idx[n] for n in names]
            Fm = F[m][:, cols]
            oof = np.zeros(len(ym))
            n_sp = min(5, len(set(gm.tolist())))
            for tr, te in GroupKFold(n_splits=n_sp).split(Fm, ym, groups=gm):
                oof[te] = fresh().fit(Fm[tr], ym[tr]).predict_proba(Fm[te])[:, 1]
            report(label, ym, oof, gm)


if __name__ == "__main__" and "--ablation" in sys.argv:
    ablation()
    raise SystemExit(0)
