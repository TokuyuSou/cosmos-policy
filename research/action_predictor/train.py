"""Train one action-predictor variant and save it to a run directory.

Example:
    uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/train.py \
        --data-dir research/data/pnp_counter_to_stove \
        --out research/results/dense/conv --img-mode conv --img-views primary --epochs 120
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    VIEW_NAME_TO_IDX,
    ChunkDataset,
    Normalizers,
    build_samples,
    fit_normalizers,
    list_success_episodes,
    split_episode_files,
)
from model import ActionPredictor, count_params


def gaussian_nll(mean, log_std, target):
    inv_var = torch.exp(-2.0 * log_std)
    return (0.5 * ((target - mean) ** 2 * inv_var + 2.0 * log_std)).mean()


def physical_rmse(pred_phys, target_phys):
    """RMSE over all dims and per-dim (np arrays, shape (...,7))."""
    err = pred_phys - target_phys
    overall = float(np.sqrt((err ** 2).mean()))
    per_dim = np.sqrt((err ** 2).reshape(-1, err.shape[-1]).mean(0))
    return overall, per_dim


@torch.no_grad()
def evaluate_split(model, loader, norm: Normalizers, device):
    model.eval()
    preds, targs, baselines, anchors = [], [], [], []
    for b in loader:
        prev, state, img = b["prev_actions"].to(device), b["state"].to(device), b["future_img"].to(device)
        mean, _ = model(prev, state, img)
        pred_norm = b["anchor"].to(device).unsqueeze(1) + mean  # anchor + residual
        pred_phys = norm.denorm_actions(pred_norm).cpu().numpy()
        preds.append(pred_phys)
        targs.append(b["target_phys"].numpy())
        baselines.append(b["cosmos_remaining"].numpy())
        anchors.append(norm.denorm_actions(b["anchor"]).numpy())
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    baselines = np.concatenate(baselines)
    # repeat-last baseline (anchor action held for 16 steps)
    rl = np.repeat(np.concatenate(anchors)[:, None, :], preds.shape[1], axis=1)
    out = {}
    out["predictor"], out["predictor_per_dim"] = physical_rmse(preds, targs)
    out["cosmos_remaining"], out["cosmos_remaining_per_dim"] = physical_rmse(baselines, targs)
    out["repeat_last"], out["repeat_last_per_dim"] = physical_rmse(rl, targs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-mode", default="conv", choices=["none", "mean", "meanstd", "grid", "conv", "full"])
    ap.add_argument("--img-views", default="primary", help="comma-separated: wrist,primary,secondary")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--nll-weight", type=float, default=0.5)
    ap.add_argument("--mse-weight", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    views = [VIEW_NAME_TO_IDX[v.strip()] for v in args.img_views.split(",") if v.strip()]

    files = list_success_episodes(args.data_dir)
    assert len(files) >= 4, f"Not enough successful episodes: {len(files)}"
    train_files, val_files = split_episode_files(files, args.val_frac, args.seed)
    train_samples = build_samples(train_files, views)
    val_samples = build_samples(val_files, views)
    norm = fit_normalizers(train_samples)
    print(f"episodes: {len(train_files)} train / {len(val_files)} val | "
          f"samples: {len(train_samples)} train / {len(val_samples)} val | views={views}", flush=True)

    train_ds = ChunkDataset(train_samples, norm)
    val_ds = ChunkDataset(val_samples, norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = ActionPredictor(
        n_views=len(views), img_mode=args.img_mode, width=args.width,
        depth=args.depth, heads=args.heads, dropout=args.dropout, grid=args.grid,
    ).to(args.device)
    n_params = count_params(model)
    print(f"img_mode={args.img_mode} params={n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    best = {"epoch": -1, "predictor": float("inf")}
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for b in train_loader:
            prev, state, img = b["prev_actions"].to(args.device), b["state"].to(args.device), b["future_img"].to(args.device)
            tr = b["target_residual"].to(args.device)
            mean, log_std = model(prev, state, img)
            loss = args.nll_weight * gaussian_nll(mean, log_std, tr) + args.mse_weight * ((mean - tr) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * prev.shape[0]
        sched.step()
        val = evaluate_split(model, val_loader, norm, args.device)
        history.append({"epoch": epoch, "train_loss": tot / len(train_ds), **{k: v for k, v in val.items() if not k.endswith("per_dim")}})
        if val["predictor"] < best["predictor"]:
            best = {"epoch": epoch, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val.items()}}
            torch.save({"model": model.state_dict(), "args": vars(args), "views": views}, os.path.join(args.out, "ckpt.pt"))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"ep {epoch:3d} loss={tot/len(train_ds):.4f} "
                  f"val_pred_rmse={val['predictor']:.4f} cosmos_rem={val['cosmos_remaining']:.4f} "
                  f"repeat_last={val['repeat_last']:.4f}", flush=True)

    np.savez(os.path.join(args.out, "normalizers.npz"), **norm.to_dict())
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "n_params": n_params, "views": views,
                   "n_train_ep": len(train_files), "n_val_ep": len(val_files),
                   "n_train_samples": len(train_samples), "n_val_samples": len(val_samples)}, f, indent=2)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"history": history, "best": best}, f, indent=2)
    print(f"BEST epoch={best['epoch']} predictor_rmse={best['predictor']:.4f} "
          f"(cosmos_remaining={best['cosmos_remaining']:.4f}, repeat_last={best['repeat_last']:.4f})", flush=True)


if __name__ == "__main__":
    main()
