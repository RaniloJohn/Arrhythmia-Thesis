"""
Dense-stride MIMIC build for CNN training.

The evaluation set stays non-overlapping (10 s stride, 120 windows per 20-minute recording)
so reported metrics are never inflated by redundant windows — that was a v1 defect. But a
1.03M-to-74k parameter CNN cannot be fit from 4,189 windows, so the *training* set is built
with a short stride. Overlap between training windows is harmless here because every split
in this project is patient-disjoint: an overlapping window can only ever leak into another
window of the same recording, which is already on the same side of the split.

Emits two arrays, and `train_v3.py` uses them for their stated purposes only:
  * `X_train` / stride `--train-stride` (default 2 s)  -> fitting
  * `X_eval`  / stride 10 s, non-overlapping           -> every reported metric
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from signal_processing.filter import ButterBandpassFilter, detrend_ppg, zscore_normalize
from training.datasets.mimic_perform import MimicPerformAFLoader
from training.resample import to_100hz

FS = 100.0
WIN = 1000


def windows(sig: np.ndarray, stride: int, bp: ButterBandpassFilter):
    for s in range(0, len(sig) - WIN + 1, stride):
        w = sig[s:s + WIN]
        if np.std(w) < 1e-9:
            continue
        out = zscore_normalize(bp.apply(detrend_ppg(w)))
        if np.all(np.isfinite(out)):
            yield out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stride", type=float, default=2.0, help="seconds between training windows")
    ap.add_argument("--out", type=Path, default=ML_DIR / "data" / "processed" / "mimic_dense.npz")
    args = ap.parse_args()

    bp = ButterBandpassFilter(lowcut=0.5, highcut=5.0, fs=FS, order=4)
    Xtr, ytr, gtr, Xev, yev, gev = [], [], [], [], [], []

    for rec in MimicPerformAFLoader().subjects():
        sig = np.asarray(rec.signal, dtype=np.float64)
        finite = np.isfinite(sig)
        if not finite.all():
            idx = np.arange(sig.size)
            sig = np.interp(idx, idx[finite], sig[finite])
        if abs(rec.fs - FS) > 1e-6:
            sig = to_100hz(sig, rec.fs)

        for w in windows(sig, int(round(args.train_stride * FS)), bp):
            Xtr.append(w); ytr.append(rec.label); gtr.append(rec.subject_id)
        for w in windows(sig, WIN, bp):
            Xev.append(w); yev.append(rec.label); gev.append(rec.subject_id)
        print(f"  {rec.subject_id}: label={rec.label} train={len(Xtr)} eval={len(Xev)}", flush=True)

    out = dict(
        X_train=np.asarray(Xtr, dtype=np.float32), y_train=np.asarray(ytr, dtype=np.int8),
        g_train=np.asarray(gtr),
        X_eval=np.asarray(Xev, dtype=np.float32), y_eval=np.asarray(yev, dtype=np.int8),
        g_eval=np.asarray(gev),
    )
    np.savez_compressed(args.out, **out)
    print(f"\ntrain {out['X_train'].shape} ({100*out['y_train'].mean():.1f}% AF) | "
          f"eval {out['X_eval'].shape} ({100*out['y_eval'].mean():.1f}% AF) | "
          f"{len(set(out['g_train'].tolist()))} subjects")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
