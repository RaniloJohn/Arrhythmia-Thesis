"""
Architecture study: compares temporal-aggregation heads and receptive fields under one
patient-disjoint CV protocol, against the IBI baseline measured by `baseline_ibi.py`.

Produces the evidence for amending the Chapter 2 topology. Early stopping tracks inner-val
**window** AUROC rather than subject AUROC: an inner split holds only ~7 subjects, so
subject AUROC there is far too noisy to stop on (it was what made `train_v2.py` select
epoch 1-2 and report 0.47).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from training.torch_model_v2 import VARIANTS, build
from training.train_v2 import augment, inner_split, predict
from training.baseline_ibi import auroc, subject_level

SEED = 42


def fit(model: nn.Module, Xtr, ytr, Xva, yva, gva, epochs, patience, lr, wd, bs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    Xt = torch.from_numpy(Xtr).float().unsqueeze(1)
    yt = torch.from_numpy(ytr).float().unsqueeze(-1)

    best_score, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]
            opt.zero_grad()
            crit(model(augment(Xt[sel], rng), return_logits=True), yt[sel]).backward()
            opt.step()
        pv = predict(model, Xva)
        score = auroc(yva, pv)              # window AUROC: smooth enough to stop on
        score = 0.0 if not np.isfinite(score) else float(score)
        sched.step(score)
        if score > best_score:
            best_score, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ML_DIR / "data" / "processed" / "combined_v2.npz")
    ap.add_argument("--cohort", default="mimic")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", type=Path, default=ML_DIR / "training" / "runs" / "arch_sweep.json")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    m = d["dataset"] == args.cohort
    X, y, g = d["X"][m], d["y"][m].astype(int), d["group"][m]
    print(f"{args.cohort}: {len(y)} windows, {len(set(g.tolist()))} subjects, {100*y.mean():.1f}% AF\n")

    from sklearn.model_selection import GroupKFold

    results: Dict[str, object] = {"cohort": args.cohort, "folds": args.folds,
                                  "ibi_baseline_subject_auroc": 0.970}
    rows = []
    for name in args.variants.split(","):
        probe = build(name)
        rf, npar = probe.receptive_field, probe.n_params()
        print(f"=== {name}: receptive field {rf} samples ({rf/100:.2f} s), {npar:,} params")
        oof = np.zeros(len(y))
        per_fold = []
        t0 = time.time()
        for k, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups=g), 1):
            itr, iva = inner_split(g[tr], y[tr], frac=0.25, seed=SEED + k)
            mdl, ep, sc = fit(build(name), X[tr][itr], y[tr][itr], X[tr][iva], y[tr][iva],
                              g[tr][iva], args.epochs, args.patience, args.lr,
                              args.weight_decay, args.batch_size, SEED + k)
            oof[te] = predict(mdl, X[te])
            ys, ps = subject_level(y[te], oof[te], g[te])
            fa = auroc(ys, ps)
            per_fold.append(None if not np.isfinite(fa) else round(float(fa), 4))
            print(f"    fold {k}: subject AUROC {per_fold[-1]}  (stopped ep {ep})")
        ys, ps = subject_level(y, oof, g)
        pooled = float(auroc(ys, ps))
        valid = [v for v in per_fold if v is not None]
        rec = {"receptive_field_samples": rf, "receptive_field_s": round(rf / 100, 3),
               "n_params": npar, "pooled_subject_auroc": round(pooled, 4),
               "pooled_window_auroc": round(float(auroc(y, oof)), 4),
               "fold_subject_auroc": per_fold,
               "fold_mean": round(float(np.mean(valid)), 4),
               "fold_std": round(float(np.std(valid)), 4),
               "minutes": round((time.time() - t0) / 60, 1)}
        results[name] = rec
        rows.append((name, rec))
        np.savez_compressed(args.out.parent / f"oof_{name}.npz", y=y, p=oof, group=g)
        print(f"  -> pooled subject AUROC {pooled:.4f} "
              f"(per-fold {rec['fold_mean']:.3f} +/- {rec['fold_std']:.3f}) "
              f"[{rec['minutes']} min]\n")

    print(f"{'variant':<16}{'RF (s)':>8}{'params':>12}{'subj AUROC':>12}{'fold mean':>11}")
    for name, r in sorted(rows, key=lambda kv: -kv[1]["pooled_subject_auroc"]):
        print(f"{name:<16}{r['receptive_field_s']:>8.2f}{r['n_params']:>12,}"
              f"{r['pooled_subject_auroc']:>12.3f}{r['fold_mean']:>11.3f}")
    print(f"{'IBI baseline':<16}{'-':>8}{12:>12,}{0.970:>12.3f}{'-':>11}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
