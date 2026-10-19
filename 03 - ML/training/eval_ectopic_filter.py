"""
EXPERIMENT (2026-10-19) — ectopic / beat-detection-artefact interval correction for ibi_af_v1.
Not deployed. Read-only: writes only training/runs/ectopic_filter_eval.json and never touches
the deployed weights (unlike train_ibi_model.py, which overwrites ibi_af_v1.npz by default).

Result and verdict: 02 - Code Review/2026-10-19 - Ectopic Interval Filtering - MIMIC Re-validation.md

The candidate correction targets the three detector-error shapes seen at the wrist on
2026-10-13, relative to the window's median interval M:
  * missed beat      : one interval ~2M            -> split into two equal intervals
  * misplaced peak   : short+long pair summing ~2M -> two intervals of (sum/2)
  * extra/split peak : short+short pair summing ~M -> one merged interval
`context_cv` is the AF guard: a correction is applied only when the *other* intervals are
regular (robust CV = 1.4826*MAD/median below the limit); None disables it (naive filter).

  A. MIMIC PERform AF under the training protocol of train_ibi_model.py (patient-disjoint
     5-fold GroupKFold, all out-of-fold), in two modes:
       infer-only : fold models fit on uncorrected features, scored on corrected ones
                    (= ship the filter, keep the ibi_af_v1 weights)
       retrain    : fit and score on corrected features (= would need an ibi_af_v2)
  B. Own-device wrist captures of non-AF testers (gitignored, so this part is skipped when
     they are absent): raw captures are replayed through the real EdgeInferenceRunner gates
     and every classified window is rescored with the deployed weights.

    python training/eval_ectopic_filter.py
"""

from __future__ import annotations

import gc
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
os.environ["RAW_CAPTURE_DIR"] = "off"
os.environ.pop("TARGET_PATIENT_ID", None)

from model.ibi_classifier import IBIAFClassifier, ibi_features_from_peaks
from signal_processing.peak_detection import ElgendiPeakDetector
from training.baseline_ibi import auroc, subject_level
from training.train_ibi_model import bootstrap_subject_ci

TAU, FLAG_T, K, N = 0.32, 0.92, 3, 5          # deployed ibi_af_v1 operating point
CAPTURES = ML_DIR / "data" / "device_captures"
VARIANTS = {
    "none": None,
    "naive tol.15": dict(tol=0.15, context_cv=None),
    "guard cv.08 tol.15": dict(tol=0.15, context_cv=0.08),
    "guard cv.10 tol.15": dict(tol=0.15, context_cv=0.10),
    "guard cv.15 tol.15": dict(tol=0.15, context_cv=0.15),
    "guard cv.10 tol.20": dict(tol=0.20, context_cv=0.10),
}


# ------------------------------------------------------------------ candidate filter


def robust_cv(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.inf
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) / med if med > 0 else np.inf


def correct_intervals(ibi, tol=0.15, context_cv=0.10, max_fix_frac=0.34):
    """Returns (corrected_ibi, n_corrections). ibi in seconds."""
    ibi = list(np.asarray(ibi, dtype=float))
    n = len(ibi)
    if n < 4:
        return np.asarray(ibi), 0
    M = float(np.median(ibi))
    flagged = np.zeros(n, bool)
    plan = {}  # start index -> (length, replacement intervals)
    i = 0
    while i < n:
        a = ibi[i]
        b = ibi[i + 1] if i + 1 < n else None
        if abs(a - 2 * M) <= tol * 2 * M:
            plan[i] = (1, [a / 2, a / 2]); flagged[i] = True; i += 1; continue
        if b is not None:
            s = a + b
            if (abs(s - 2 * M) <= tol * 2 * M and min(a, b) < (1 - tol) * M
                    and max(a, b) > (1 + tol) * M):
                plan[i] = (2, [s / 2, s / 2]); flagged[i:i + 2] = True; i += 2; continue
            if abs(s - M) <= tol * M and max(a, b) < (1 - tol) * M:
                plan[i] = (2, [s]); flagged[i:i + 2] = True; i += 2; continue
        i += 1
    if not plan or flagged.mean() > max_fix_frac:
        return np.asarray(ibi), 0   # too many "artefacts" = not a regular rhythm with a glitch
    if context_cv is not None and robust_cv(np.asarray(ibi)[~flagged]) > context_cv:
        return np.asarray(ibi), 0
    out, i = [], 0
    while i < n:
        if i in plan:
            ln, rep = plan[i]; out.extend(rep); i += ln
        else:
            out.append(ibi[i]); i += 1
    return np.asarray(out), len(plan)


def corrected_peaks(peaks, fs=100.0, **kw):
    """Peak train rebuilt from corrected intervals, so n_beats etc. stay consistent."""
    peaks = np.asarray(peaks, dtype=float).ravel()
    if peaks.size < 5:
        return peaks, 0
    ibi = np.diff(peaks) / fs
    if np.any((ibi <= 0.25) | (ibi >= 2.5)):   # correct only fully physiological windows
        return peaks, 0
    c, k = correct_intervals(ibi, **kw)
    if k == 0:
        return peaks, 0
    return np.concatenate([[peaks[0]], peaks[0] + np.cumsum(c) * fs]), k


# --------------------------------------------------------------------- evaluation


def fresh():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))


def features(peaks_list, kw):
    F = np.zeros((len(peaks_list), 12)); fixed = np.zeros(len(peaks_list), bool)
    for i, p in enumerate(peaks_list):
        if kw is not None:
            p, k = corrected_peaks(p, 100.0, **kw); fixed[i] = k > 0
        F[i] = ibi_features_from_peaks(p, 100.0)
    return F, fixed


def subject_consensus(y, p, g):
    """Deployed subject rule, fixed (not re-tuned): 3-of-5 consensus, flag rate >= 0.92."""
    sy, rate, ids = [], [], sorted(set(g.tolist()))
    for gid in ids:
        m = g == gid
        f = (p[m] >= TAU).astype(int)
        c = [1 if f[max(0, i - N + 1):i + 1].sum() >= K else 0 for i in range(len(f))]
        sy.append(int(round(y[m].mean()))); rate.append(float(np.mean(c)))
    sy, rate = np.array(sy), np.array(rate)
    pred = rate >= FLAG_T
    return {"sens": round(float((pred & (sy == 1)).sum() / (sy == 1).sum()), 3),
            "spec": round(float((~pred & (sy == 0)).sum() / (sy == 0).sum()), 3),
            "fn": int((~pred & (sy == 1)).sum()), "fp": int((pred & (sy == 0)).sum()),
            "flag_rate": {k: round(r, 3) for k, r in zip(ids, rate)}}


def metrics(y, p, g):
    ys, ps = subject_level(y, p, g)
    pr = p >= TAU
    return {"subj_auroc": round(auroc(ys, ps), 4), "ci95": bootstrap_subject_ci(ys, ps)["auroc"],
            "win_auroc": round(auroc(y, p), 4),
            "win_sens": round(float((pr & (y == 1)).sum() / (y == 1).sum()), 4),
            "win_spec": round(float((~pr & (y == 0)).sum() / (y == 0).sum()), 4),
            "consensus_at_deployed": subject_consensus(y, p, g)}


def part_a():
    from sklearn.model_selection import GroupKFold
    d = np.load(ML_DIR / "data" / "processed" / "combined_v2.npz", allow_pickle=True)
    m = d["dataset"] == "mimic"
    y, g = d["y"][m].astype(int), d["group"][m]
    det = ElgendiPeakDetector(fs=100.0)
    peaks = [np.asarray(det.detect_peaks(np.asarray(x, dtype=np.float64))) for x in d["X"][m]]
    F0, _ = features(peaks, None)
    cache = np.load(ML_DIR / "data" / "processed" / "ibi_features_v2.npz")["F"][m]
    assert np.allclose(F0, cache), "harness does not reproduce the ibi_af_v1 training features"
    print("harness check: features identical to the ibi_af_v1 training cache")

    folds = list(GroupKFold(n_splits=5).split(F0, y, groups=g))
    base = [fresh().fit(F0[tr], y[tr]) for tr, _ in folds]
    res = {}
    for name, kw in VARIANTS.items():
        F, fixed = features(peaks, kw)
        oof_i, oof_r = np.zeros(len(y)), np.zeros(len(y))
        for (tr, te), bm in zip(folds, base):
            oof_i[te] = bm.predict_proba(F[te])[:, 1]
            oof_r[te] = fresh().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        r = {"corrected_af_windows": round(float(fixed[y == 1].mean()), 4),
             "corrected_nonaf_windows": round(float(fixed[y == 0].mean()), 4),
             "infer_only": metrics(y, oof_i, g), "retrain": metrics(y, oof_r, g)}
        res[name] = r
        print(f"\n[{name}] windows corrected: AF {100 * r['corrected_af_windows']:.1f}%  "
              f"non-AF {100 * r['corrected_nonaf_windows']:.1f}%")
        for mode in ("infer_only", "retrain"):
            q = r[mode]; c = q["consensus_at_deployed"]
            print(f"   {mode:<10} subj AUROC {q['subj_auroc']:.4f} {q['ci95']}  "
                  f"win AUROC {q['win_auroc']:.4f}  win@{TAU} sens {q['win_sens']:.4f} "
                  f"spec {q['win_spec']:.4f}  3-of-5: sens {c['sens']} spec {c['spec']} "
                  f"(fn {c['fn']} fp {c['fp']})")
    return res


def replay_peaks(path):
    """Peaks of every window the real runner gates let through to classification."""
    from edge_inference.runner import EdgeInferenceRunner
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False); tmp.close()
    runner = EdgeInferenceRunner(db_path=tmp.name)
    runner.broadcast_telemetry = lambda payload: None
    got, orig = [], runner.ibi_model.explain_intervals

    def spy(pk, fs, n_samples=1000):
        got.append(np.asarray(pk))
        return orig(pk, fs, n_samples=n_samples)

    runner.ibi_model.explain_intervals = spy
    z = np.load(path)
    W, step, total = runner.window_size, runner.step_size, 0
    for s in range(0, len(z["ir"]) - W + 1, step):
        runner.raw_ir_buffer = [int(v) for v in z["ir"][s:s + W]]
        runner.raw_red_buffer = [int(v) for v in z["red"][s:s + W]]
        runner.timestamps_buffer = [int(v) for v in z["timestamp_ms"][s:s + W]]
        runner.sample_count += step
        runner.process_window(); total += 1
    runner = None; gc.collect()   # release SQLite handles before deleting (Windows)
    for sfx in ("", "-wal", "-shm"):
        try:
            os.remove(tmp.name + sfx)
        except OSError:
            pass
    return got, total


def part_b():
    if not CAPTURES.exists():
        print("\n(device captures absent — part B skipped)")
        return {}
    clf = IBIAFClassifier().load(ML_DIR / "model" / "weights" / "ibi_af_v1.npz")
    sets = {}
    for p in sorted(glob.glob(str(CAPTURES / "2026-10-12_*.npz"))):
        pk, tot = replay_peaks(p)
        sets[f"{Path(p).stem} ({len(pk)}/{tot} windows classified)"] = pk
    # 10-13 auto-captured AF windows, attributed to a run by matching AF_Prob in its journal
    retest = CAPTURES / "2026-10-13_retest"
    journals = {}
    for j in glob.glob(str(retest / "journal" / "*.log")):
        for mt in re.finditer(r"AF_Prob=([\d.]+)%", Path(j).read_text()):
            journals.setdefault(mt.group(1), set()).add(Path(j).stem.split("_")[0])
    auto = {}
    for f in sorted(glob.glob(str(retest / "auto_af_windows" / "*.npz"))):
        z = np.load(f)
        runs = journals.get(f"{100 * float(z['af_probability']):.2f}", {"unattributed"})
        auto.setdefault("/".join(sorted(runs)), []).append(z["peaks"])
    for k, v in sorted(auto.items()):
        sets[f"10-13 auto-captured AF windows, run {k}"] = v

    res = {}
    for name, pk in sets.items():
        row = {}
        for vname, kw in VARIANTS.items():
            F, fixed = features(pk, kw)
            pr = np.array([clf.predict_from_features(f)[0] for f in F])
            row[vname] = {"n": len(pk), "af_calls": int((pr >= TAU).sum()),
                          "corrected": int(fixed.sum())}
        res[name] = row
        print(f"\n{name}")
        for vname, q in row.items():
            print(f"   {vname:<20} AF calls {q['af_calls']:>3}/{q['n']:<3} "
                  f"windows corrected {q['corrected']}")
    return res


def main() -> None:
    out = {"protocol": "see module docstring", "deployed_operating_point":
           {"tau": TAU, "consensus": f"{K}-of-{N}", "subject_flag_rate": FLAG_T},
           "mimic": part_a(), "device": part_b()}
    path = ML_DIR / "training" / "runs" / "ectopic_filter_eval.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
