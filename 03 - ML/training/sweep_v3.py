"""
Architecture sweep v3 — dense training windows, non-overlapping evaluation windows.

Fits on `mimic_dense.npz:X_train` (2 s stride, 20,860 windows) and reports exclusively on
`X_eval` (10 s stride, non-overlapping, 4,200 windows). Splits are patient-disjoint, so
training-window overlap cannot leak across the split, while every reported number comes from
redundancy-free windows.

Early stopping uses inner-validation window AUROC; subject AUROC on a ~7-subject inner split
is too noisy to stop on.
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


def fit(model: nn.Module, Xtr, ytr, Xva, yva, epochs, patience, lr, wd, bs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    Xt = torch.from_numpy(Xtr).float().unsqueeze(1)
    yt = torch.from_numpy(ytr).float().unsqueeze(-1)

    best, state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]
            opt.zero_grad()
            crit(model(augment(Xt[sel], rng), return_logits=True), yt[sel]).backward()
            opt.step()
        sc = auroc(yva, predict(model, Xva))
        sc = 0.0 if not np.isfinite(sc) else float(sc)
        sched.step(sc)
        if sc > best:
            best, best_ep, bad = sc, ep, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        model.load_state_dict(state)
    return model, best_ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ML_DIR / "data" / "processed" / "mimic_dense.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--variants", default="ch2_flatten,xwide_statpool,xxwide_statpool,huge_statpool,huge_gap,deep_statpool")
    ap.add_argument("--out", type=Path, default=ML_DIR / "training" / "runs" / "arch_sweep_v3.json")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Xtr_all, ytr_all, gtr_all = d["X_train"], d["y_train"].astype(int), d["g_train"]
    Xev, yev, gev = d["X_eval"], d["y_eval"].astype(int), d["g_eval"]
    subjects = np.array(sorted(set(gev.tolist())))
    sub_lab = np.array([int(round(float(yev[gev == s].mean()))) for s in subjects])
    print(f"train {Xtr_all.shape} | eval {Xev.shape} | {len(subjects)} subjects "
          f"({sub_lab.sum()} AF / {len(sub_lab) - sub_lab.sum()} non-AF)\n")

    from sklearn.model_selection import StratifiedKFold

    results: Dict[str, object] = {"ibi_baseline_subject_auroc": 0.970,
                                 "n_subjects": int(len(subjects))}
    rows = []
    for name in args.variants.split(","):
        probe = build(name)
        rf, npar = probe.receptive_field, probe.n_params()
        print(f"=== {name}: RF {rf} samples ({rf/100:.2f} s), {npar:,} params")
        oof = np.zeros(len(yev))
        t0 = time.time()
        per_fold: List[object] = []
        # stratify folds by subject label so every fold has both classes
        for k, (tr_s, te_s) in enumerate(
                StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
                .split(subjects, sub_lab), 1):
            tr_subj, te_subj = set(subjects[tr_s]), set(subjects[te_s])
            mtr = np.isin(gtr_all, list(tr_subj))
            mev = np.isin(gev, list(te_subj))
            itr, iva = inner_split(gtr_all[mtr], ytr_all[mtr], frac=0.2, seed=SEED + k)
            mdl, ep = fit(build(name), Xtr_all[mtr][itr], ytr_all[mtr][itr],
                          Xtr_all[mtr][iva], ytr_all[mtr][iva],
                          args.epochs, args.patience, args.lr, args.weight_decay,
                          args.batch_size, SEED + k)
            oof[mev] = predict(mdl, Xev[mev])
            ys, ps = subject_level(yev[mev], oof[mev], gev[mev])
            fa = auroc(ys, ps)
            per_fold.append(None if not np.isfinite(fa) else round(float(fa), 4))
            print(f"    fold {k}: subject AUROC {per_fold[-1]} (stopped ep {ep})", flush=True)

        ys, ps = subject_level(yev, oof, gev)
        pooled = float(auroc(ys, ps))
        valid = [v for v in per_fold if v is not None]
        rec = {"receptive_field_s": round(rf / 100, 3), "n_params": npar,
               "pooled_subject_auroc": round(pooled, 4),
               "pooled_window_auroc": round(float(auroc(yev, oof)), 4),
               "fold_subject_auroc": per_fold,
               "fold_mean": round(float(np.mean(valid)), 4),
               "fold_std": round(float(np.std(valid)), 4),
               "minutes": round((time.time() - t0) / 60, 1)}
        results[name] = rec
        rows.append((name, rec))
        np.savez_compressed(args.out.parent / f"oof_v3_{name}.npz", y=yev, p=oof, group=gev)
        print(f"  -> pooled subject AUROC {pooled:.4f} "
              f"(fold {rec['fold_mean']:.3f} +/- {rec['fold_std']:.3f}) [{rec['minutes']} min]\n")

    print(f"{'variant':<18}{'RF (s)':>8}{'params':>12}{'subj AUROC':>12}{'fold mean':>11}")
    for name, r in sorted(rows, key=lambda kv: -kv[1]["pooled_subject_auroc"]):
        print(f"{name:<18}{r['receptive_field_s']:>8.2f}{r['n_params']:>12,}"
              f"{r['pooled_subject_auroc']:>12.3f}{r['fold_mean']:>11.3f}")
    print(f"{'IBI baseline':<18}{'-':>8}{12:>12,}{0.970:>12.3f}{'-':>11}")

    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
