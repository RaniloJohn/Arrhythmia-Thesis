"""
1D-CNN training v2 — trains the frozen Chapter 2 topology under the evaluation protocol
established in `baseline_ibi.py`, and reports against that baseline rather than in isolation.

Changes from `train.py`, each traceable to a 2026-10-01 audit finding:

* **No window caps.** v1 shipped weights from a `--max-train-windows` smoke run that used
  4 of 12 groups (4.9% of windows) and exported `best_epoch: 1`. There is no cap here.
* **Patient-disjoint k-fold CV is the headline metric**, not a single split, because the
  cohort has 35 subjects and a single split's subject AUROC has a very wide interval.
* **Early stopping tracks subject-level AUROC**, not window AUROC. Labels are constant
  within a recording, so window metrics overstate the effective sample size (n=35, not
  n=4,189) and a window-optimal checkpoint is not subject-optimal.
* **Augmentation deliberately destroys recording identity** while preserving interval
  structure: circular shift, amplitude scaling, additive noise, segment masking and time
  reversal. AF's signature is interval irregularity, which is invariant to all five, so
  these push the network toward rhythm and away from per-subject morphology — the shortcut
  that made the v1 model score below chance on its own training data.
* **Training cohort is MIMIC PERform AF**, on measured grounds. See the module docstring of
  `build_dataset_v2.py` and the 2026-10-01 audit: DeepBeat as downloaded contains 20.0 h of
  unique signal (2.4 h of AF across 11 recordings) sampled at 32 Hz, where one sample is
  31 ms against the 50 ms pNN50 criterion, and a 3-parameter irregularity model scores at
  chance on it (subject AUROC 0.465). MIMIC is 125 Hz, 35 subjects, and reaches subject
  AUROC 0.914 from irregularity features alone. DeepBeat is retained as a cross-sensor
  robustness cohort, which is what it can honestly support.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import torch
import torch.nn as nn

from training.torch_model import Arrhythmia1DCNN
from training.baseline_ibi import auroc, subject_level

SEED = 42


# --------------------------------------------------------------------- augmentation


def augment(xb: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """
    Rhythm-preserving, identity-destroying augmentation. Every transform here leaves the
    inter-beat interval sequence intact (or reverses it, which preserves irregularity),
    so it cannot erase the AF signal while it does erase per-recording morphology cues.
    """
    b, _, n = xb.shape
    out = xb.clone()

    # circular time shift — removes any absolute phase cue
    for i in range(b):
        out[i] = torch.roll(out[i], int(rng.integers(0, n)), dims=-1)

    # amplitude scaling — teaches scale invariance beyond the z-score
    scale = torch.tensor(rng.uniform(0.8, 1.25, size=(b, 1, 1)), dtype=out.dtype)
    out = out * scale

    # additive noise — suppresses reliance on fine morphology
    sigma = torch.tensor(rng.uniform(0.0, 0.15, size=(b, 1, 1)), dtype=out.dtype)
    out = out + torch.randn_like(out) * sigma

    # segment masking — simulates motion dropout
    for i in range(b):
        if rng.random() < 0.3:
            w = int(rng.integers(50, 150))
            s = int(rng.integers(0, max(1, n - w)))
            out[i, :, s:s + w] = 0.0

    # time reversal — interval irregularity is symmetric under reversal
    flip = rng.random(b) < 0.5
    if flip.any():
        idx = np.where(flip)[0]
        out[idx] = torch.flip(out[idx], dims=[-1])

    return out


# ------------------------------------------------------------------------- one model


def predict(model: nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1)
            out[i:i + batch] = torch.sigmoid(model(xb, return_logits=True)).squeeze(-1).numpy()
    return out


def fit_one(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray, gva: np.ndarray,
    epochs: int, patience: int, lr: float, weight_decay: float,
    batch_size: int, seed: int, verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Arrhythmia1DCNN(input_length=Xtr.shape[1], dropout=0.5)

    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    Xt = torch.from_numpy(Xtr).float().unsqueeze(1)
    yt = torch.from_numpy(ytr).float().unsqueeze(-1)

    best = {"subject_auroc": -1.0, "epoch": 0, "state": None}
    history: List[Dict[str, float]] = []
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), batch_size):
            sel = perm[i:i + batch_size]
            xb = augment(Xt[sel], rng)
            opt.zero_grad()
            loss = crit(model(xb, return_logits=True), yt[sel])
            loss.backward()
            opt.step()
            tot += float(loss) * len(sel)

        pv = predict(model, Xva)
        ys, ps = subject_level(yva, pv, gva)
        sa = auroc(ys, ps)
        wa = auroc(yva, pv)
        sa = 0.0 if not np.isfinite(sa) else sa
        history.append({"epoch": ep, "train_loss": round(tot / len(Xt), 4),
                        "val_subject_auroc": round(float(sa), 4),
                        "val_window_auroc": round(float(wa), 4)})
        sched.step(sa)
        if sa > best["subject_auroc"]:
            best = {"subject_auroc": float(sa), "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if verbose:
            print(f"    ep{ep:>3} loss {history[-1]['train_loss']:.4f} "
                  f"val_subj_auroc {sa:.4f} (best {best['subject_auroc']:.4f} @ep{best['epoch']})",
                  flush=True)
        if bad >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"],
                   "best_val_subject_auroc": round(best["subject_auroc"], 4),
                   "epochs_run": len(history), "history": history}


def inner_split(groups: np.ndarray, y: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Group-disjoint, label-stratified inner validation split for early stopping."""
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups.tolist()))
    lab = {g: int(round(float(y[groups == g].mean()))) for g in gs}
    va: List[str] = []
    for cls in (0, 1):
        cand = [g for g in gs if lab[g] == cls]
        rng.shuffle(cand)
        k = max(1, int(round(frac * len(cand))))
        va += cand[:k]
    va_mask = np.isin(groups, va)
    return ~va_mask, va_mask


# ------------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ML_DIR / "data" / "processed" / "combined_v2.npz")
    ap.add_argument("--cohort", type=str, default="mimic", choices=["mimic", "deepbeat", "combined"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-weights", type=Path, default=ML_DIR / "model" / "weights" / "cnn_af_v2.npz")
    ap.add_argument("--runs-dir", type=Path, default=ML_DIR / "training" / "runs")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = np.load(args.data, allow_pickle=True)
    X, y, g, ds = d["X"], d["y"].astype(int), d["group"], d["dataset"]
    if args.cohort != "combined":
        m = ds == args.cohort
        X, y, g, ds = X[m], y[m], g[m], ds[m]
    print(f"Cohort '{args.cohort}': {len(y)} windows, {len(set(g.tolist()))} groups, "
          f"{100 * y.mean():.1f}% AF windows")

    from sklearn.model_selection import GroupKFold

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_dir / f"{stamp}_train_v2_{args.cohort}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, object] = {"cohort": args.cohort, "n_windows": int(len(y)),
                                 "n_groups": len(set(g.tolist()))}

    # ---- patient-disjoint CV: this is the reported performance estimate
    print(f"\n[CV] {args.folds}-fold patient-disjoint cross-validation")
    oof = np.zeros(len(y))
    folds: List[Dict[str, object]] = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups=g), 1):
        itr, iva = inner_split(g[tr], y[tr], frac=0.25, seed=SEED + k)
        print(f"  fold {k}: train {int(itr.sum())} / inner-val {int(iva.sum())} / test {len(te)} windows")
        model, info = fit_one(
            X[tr][itr], y[tr][itr], X[tr][iva], y[tr][iva], g[tr][iva],
            args.epochs, args.patience, args.lr, args.weight_decay,
            args.batch_size, SEED + k, verbose=False,
        )
        oof[te] = predict(model, X[te])
        ys, ps = subject_level(y[te], oof[te], g[te])
        fa = auroc(ys, ps)
        folds.append({"fold": k, "test_groups": sorted(set(g[te].tolist())),
                      "subject_auroc": None if not np.isfinite(fa) else round(float(fa), 4),
                      "window_auroc": round(float(auroc(y[te], oof[te])), 4), **
                      {kk: vv for kk, vv in info.items() if kk != "history"}})
        print(f"    -> fold subject AUROC {folds[-1]['subject_auroc']} "
              f"window AUROC {folds[-1]['window_auroc']} (best ep {info['best_epoch']})")

    ys, ps = subject_level(y, oof, g)
    pooled = float(auroc(ys, ps))
    fold_sas = [f["subject_auroc"] for f in folds if f["subject_auroc"] is not None]
    results["cv"] = {
        "folds": folds,
        "pooled_subject_auroc": round(pooled, 4),
        "pooled_window_auroc": round(float(auroc(y, oof)), 4),
        "fold_subject_auroc_mean": round(float(np.mean(fold_sas)), 4),
        "fold_subject_auroc_std": round(float(np.std(fold_sas)), 4),
    }
    print(f"\n  pooled out-of-fold subject AUROC {pooled:.4f}  "
          f"(per-fold {np.mean(fold_sas):.3f} +/- {np.std(fold_sas):.3f})")
    np.savez_compressed(run_dir / "oof_predictions.npz", y=y, p=oof, group=g, dataset=ds)

    # ---- deployed model: all cohort subjects, inner split only for early stopping
    print("\n[FINAL] fitting deployment model on all cohort subjects")
    itr, iva = inner_split(g, y, frac=0.25, seed=SEED)
    model, info = fit_one(X[itr], y[itr], X[iva], y[iva], g[iva],
                          args.epochs, args.patience, args.lr, args.weight_decay,
                          args.batch_size, SEED, verbose=True)
    results["final"] = {kk: vv for kk, vv in info.items() if kk != "history"}
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "final.pt")
    (run_dir / "final_history.json").write_text(json.dumps(info["history"], indent=1), encoding="utf-8")

    # ---- cross-sensor robustness on the other cohort
    other = "deepbeat" if args.cohort == "mimic" else "mimic"
    dall = np.load(args.data, allow_pickle=True)
    mo = dall["dataset"] == other
    if mo.any():
        po = predict(model, dall["X"][mo])
        yo = dall["y"][mo].astype(int)
        go = dall["group"][mo]
        yso, pso = subject_level(yo, po, go)
        results["cross_sensor"] = {
            "cohort": other,
            "subject_auroc": round(float(auroc(yso, pso)), 4),
            "window_auroc": round(float(auroc(yo, po)), 4),
            "n_subjects": int(len(yso)),
        }
        print(f"\n[CROSS-SENSOR] {other}: subject AUROC "
              f"{results['cross_sensor']['subject_auroc']} (n={len(yso)})")

    (run_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nWrote {run_dir}")
    print("Next: calibrate_threshold_v2.py on the out-of-fold predictions, then export weights.")


if __name__ == "__main__":
    main()
