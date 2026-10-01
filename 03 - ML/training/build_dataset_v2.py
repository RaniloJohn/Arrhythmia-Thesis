"""
Dataset builder v2 — corrects the defects found in the 2026-10-01 training audit.

What changed versus `build_dataset.py`, and why:

1. **Group keys are file-scoped.** DeepBeat's `parameters[:, 2]` restarts between
   `validate.npz` and `test.npz`: ID 149 is a 2015 AF recording in one file and a 2017
   non-AF recording in the other. v1 regrouped by the bare ID and merged different people
   under one label-contradictory pseudo-subject. Keys here are `db_<file>_<id>`.

2. **Temporal redundancy is removed.** DeepBeat ships 25 s parent windows at a **1 second
   stride** across **8 simultaneous channels** (`parameters[:, 1]`, 'a'..'h'), so
   consecutive windows share 96% of their samples. v1 treated all 536,399 parents as
   independent and reported 3,725 hours of recording; the true unique wall-clock content is
   **20.0 hours**. This builder selects parents at a >= `stride_s` spacing per
   (group, channel) so emitted windows never overlap in time.

3. **SQI gating is scale-aware.** `SignalQualityAssessor` forces `pi = 0` whenever the raw
   DC mean is below 1000 (a MAX30102 "probe off" test). Dataset signals are normalised
   floats, so that branch fired on every window, capping every SQI at 0.6 and making the
   `< 0.50` gate a pure skew/kurtosis filter. Here PI is skipped explicitly for file-based
   data (`dc_floor=None`) and the gate is applied **identically to every split**.

Window contract is unchanged and still matches `edge_inference/runner.py`:
1000 samples @ 100 Hz, `detrend_ppg` -> Butterworth 0.5-5 Hz -> `zscore_normalize`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from signal_processing.filter import ButterBandpassFilter, detrend_ppg, zscore_normalize
from training.resample import to_100hz

FS = 100.0
WIN_SAMPLES = 1000
WIN_SECONDS = 10.0


# --------------------------------------------------------------------------- utils


def _epoch_seconds(values: np.ndarray) -> np.ndarray:
    """Parse DeepBeat's mixed timestamp column (pandas Timestamp or string) to epoch s."""
    import pandas as pd

    parsed = pd.to_datetime(
        [str(v).strip() for v in values], format="mixed", errors="coerce", utc=False
    )
    # Force nanosecond resolution: pandas may return us-resolution, which silently
    # scaled the stride by 1000x on the first attempt at this.
    return np.asarray(parsed.astype("datetime64[ns]").astype("int64")) / 1e9


def preprocess_window(raw: np.ndarray, fs_in: float, bp: ButterBandpassFilter) -> Optional[np.ndarray]:
    """Resample to 100 Hz, then apply the exact production DSP chain. None if unusable."""
    sig = np.asarray(raw, dtype=np.float64).ravel()
    if sig.size == 0 or not np.all(np.isfinite(sig)):
        sig = np.nan_to_num(sig, nan=np.nan)
        finite = np.isfinite(sig)
        if finite.sum() < max(8, 0.5 * sig.size):
            return None
        # linear interpolation across sensor dropout
        idx = np.arange(sig.size)
        sig = np.interp(idx, idx[finite], sig[finite])

    if abs(fs_in - FS) > 1e-6:
        sig = to_100hz(sig, fs_in)
    if sig.size < WIN_SAMPLES:
        return None

    win = sig[:WIN_SAMPLES]
    if np.std(win) < 1e-9:  # flatline
        return None
    out = zscore_normalize(bp.apply(detrend_ppg(win)))
    return None if not np.all(np.isfinite(out)) else out.astype(np.float32)


def window_sqi(filtered: np.ndarray) -> float:
    """
    Scale-invariant quality score for file-based PPG.

    Mirrors `SignalQualityAssessor`'s skewness/kurtosis criteria but omits the perfusion
    index, which is undefined for signals that are not raw MAX30102 counts. Documented
    rather than silently zeroed, which is what v1 did.
    """
    from scipy import stats

    skew = float(stats.skew(filtered))
    kurt = float(stats.kurtosis(filtered))
    score = 1.0
    if skew < -0.8 or skew > 3.0:
        score -= 0.4
    if kurt < -1.2 or kurt > 10.0:
        score -= 0.4
    return float(max(0.0, min(1.0, score)))


# ------------------------------------------------------------------- deepbeat build


def build_deepbeat(data_dir: Path, stride_s: float, min_sqi: float) -> Dict[str, np.ndarray]:
    bp = ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=FS, order=4)
    X: List[np.ndarray] = []
    y: List[int] = []
    groups: List[str] = []
    sqis: List[float] = []
    dropped = 0

    for fname in ("validate.npz", "test.npz"):
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not present")
            continue
        tag = fpath.stem
        d = np.load(fpath, allow_pickle=True)
        sig_all = d["signal"]
        rhythm = d["rhythm"]
        labels = np.argmax(rhythm, axis=1) if rhythm.ndim == 2 else rhythm.astype(int)
        params = d["parameters"]
        subj = np.array([str(v).strip() for v in params[:, 2]])
        chan = np.array([str(v).strip() for v in params[:, 1]])
        tsec = _epoch_seconds(params[:, 0])

        fs_in = sig_all.shape[1] / 25.0  # 800 samples per 25 s parent -> 32 Hz

        for s in sorted(set(subj), key=lambda v: int(v) if v.isdigit() else v):
            gid = f"db_{tag}_{s}"
            for c in sorted(set(chan)):
                m = (subj == s) & (chan == c)
                if not m.any():
                    continue
                idx = np.where(m)[0]
                order = np.argsort(tsec[idx], kind="mergesort")
                idx = idx[order]
                last = -np.inf
                for i in idx:
                    t = tsec[i]
                    if not np.isfinite(t) or t - last < stride_s:
                        continue
                    win = preprocess_window(sig_all[i], fs_in, bp)
                    if win is None:
                        dropped += 1
                        continue
                    q = window_sqi(win)
                    if q < min_sqi:
                        dropped += 1
                        continue
                    last = t
                    X.append(win)
                    y.append(int(labels[i]))
                    groups.append(gid)
                    sqis.append(q)
        print(f"  {fname}: cumulative kept={len(X)}")

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int8),
        "group": np.asarray(groups),
        "sqi": np.asarray(sqis, dtype=np.float32),
        "dataset": np.array(["deepbeat"] * len(X)),
        "_dropped": dropped,
    }


# ---------------------------------------------------------------------- mimic build


def build_mimic(processed_dir: Path, min_sqi: float) -> Dict[str, np.ndarray]:
    """
    Reuse the already-correct MIMIC build. It is non-overlapping by construction
    (120 x 10 s windows per 20-minute recording) and used the same DSP chain; only the
    group key and the quality score are recomputed here.
    """
    src = processed_dir / "mimic_external_val.npz"
    d = np.load(src, allow_pickle=True)
    X, y, sid = d["X"], d["y"].astype(np.int8), d["group"] if "group" in d else d["subject_id"]
    keep, sqis = [], []
    for i in range(len(X)):
        q = window_sqi(X[i])
        if q >= min_sqi:
            keep.append(i)
            sqis.append(q)
    keep = np.asarray(keep, dtype=int)
    return {
        "X": X[keep],
        "y": y[keep],
        "group": np.array([f"mimic_{str(s)}" for s in np.asarray(sid)[keep]]),
        "sqi": np.asarray(sqis, dtype=np.float32),
        "dataset": np.array(["mimic"] * len(keep)),
        "_dropped": len(X) - len(keep),
    }


# ---------------------------------------------------------------------------- main


def summarise(name: str, d: Dict[str, np.ndarray]) -> Dict[str, object]:
    g = d["group"]
    y = d["y"]
    per: Dict[str, Dict[str, float]] = {}
    for gid in sorted(set(g.tolist())):
        m = g == gid
        per[gid] = {"n": int(m.sum()), "af_pct": round(100 * float(y[m].mean()), 2)}
    mixed = sum(1 for v in per.values() if 0.0 < v["af_pct"] < 100.0)
    af_groups = sum(1 for v in per.values() if v["af_pct"] >= 50.0)
    print(
        f"{name}: {len(y)} windows | {len(per)} groups "
        f"({af_groups} AF / {len(per) - af_groups} non-AF) | "
        f"AF windows {100 * float(y.mean()):.1f}% | mixed-label groups {mixed} | "
        f"dropped {d['_dropped']}"
    )
    return {
        "windows": int(len(y)),
        "groups": len(per),
        "af_groups": af_groups,
        "af_window_pct": round(100 * float(y.mean()), 2),
        "mixed_label_groups": mixed,
        "dropped": int(d["_dropped"]),
        "per_group": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild AF datasets with corrected keying and no temporal redundancy.")
    ap.add_argument("--deepbeat-dir", type=Path, default=ML_DIR / "data" / "deepbeat")
    ap.add_argument("--processed-dir", type=Path, default=ML_DIR / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ML_DIR / "data" / "processed" / "combined_v2.npz")
    ap.add_argument("--stride-s", type=float, default=WIN_SECONDS,
                    help="Minimum seconds between emitted windows of the same channel (default 10 = non-overlapping)")
    ap.add_argument("--min-sqi", type=float, default=0.5)
    args = ap.parse_args()

    print("Building DeepBeat (file-scoped groups, non-overlapping windows)...")
    db = build_deepbeat(args.deepbeat_dir, args.stride_s, args.min_sqi)
    print("Building MIMIC...")
    mi = build_mimic(args.processed_dir, args.min_sqi)

    report = {
        "deepbeat": summarise("DeepBeat", db),
        "mimic": summarise("MIMIC   ", mi),
    }

    out = {
        k: np.concatenate([db[k], mi[k]]) for k in ("X", "y", "group", "sqi", "dataset")
    }
    assert len(set(db["group"].tolist()) & set(mi["group"].tolist())) == 0, "group key collision"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    report["combined"] = {
        "windows": int(len(out["y"])),
        "groups": len(set(out["group"].tolist())),
        "af_window_pct": round(100 * float(out["y"].mean()), 2),
    }
    (args.out.with_suffix(".report.json")).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}  ({len(out['y'])} windows, "
          f"{len(set(out['group'].tolist()))} groups)")


if __name__ == "__main__":
    main()
