"""Evaluate a trained action predictor: RMSE vs the ground-truth next-16 actions,
compared against the Cosmos-Policy 'remaining 16 actions' baseline (and repeat-last).

RMSE is computed in physical action space on the held-out (val) episodes — the SAME
episode split used in training (rebuilt deterministically from the saved config).

Example:
    uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/evaluate.py --run research/results/dense/conv \
        --data-dir research/data/pnp_counter_to_stove
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ChunkDataset, Normalizers, build_samples, list_success_episodes, split_episode_files
from model import ActionPredictor
from train import evaluate_split

# RoboCasa OSC_POSE action layout: [dpos(3), drot(3), gripper(1)]
DIM_GROUPS = {"pos": [0, 1, 2], "rot": [3, 4, 5], "gripper": [6]}


def grouped(per_dim):
    per_dim = np.asarray(per_dim)
    return {g: float(np.sqrt((per_dim[idx] ** 2).mean())) for g, idx in DIM_GROUPS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with ckpt.pt/config.json/normalizers.npz")
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    a = cfg["args"]
    views = cfg["views"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = list_success_episodes(args.data_dir)
    _, val_files = split_episode_files(files, a["val_frac"], a["seed"])
    val_samples = build_samples(val_files, views)
    norm = Normalizers.from_dict(dict(np.load(os.path.join(args.run, "normalizers.npz"))))
    val_loader = DataLoader(ChunkDataset(val_samples, norm), batch_size=256, shuffle=False)

    model = ActionPredictor(
        n_views=len(views), img_mode=a["img_mode"], width=a["width"],
        depth=a["depth"], heads=a["heads"], dropout=a["dropout"], grid=a["grid"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run, "ckpt.pt"), map_location=device)["model"])

    res = evaluate_split(model, val_loader, norm, device)
    report = {
        "run": args.run,
        "img_mode": a["img_mode"],
        "img_views": a["img_views"],
        "n_params_M": round(cfg["n_params"] / 1e6, 2),
        "n_val_episodes": len(val_files),
        "n_val_samples": len(val_samples),
        "rmse_overall": {k: round(float(res[k]), 5) for k in ["predictor", "cosmos_remaining", "repeat_last"]},
        "rmse_grouped": {
            "predictor": {k: round(v, 5) for k, v in grouped(res["predictor_per_dim"]).items()},
            "cosmos_remaining": {k: round(v, 5) for k, v in grouped(res["cosmos_remaining_per_dim"]).items()},
            "repeat_last": {k: round(v, 5) for k, v in grouped(res["repeat_last_per_dim"]).items()},
        },
        "rmse_per_dim": {
            "predictor": [round(float(x), 5) for x in res["predictor_per_dim"]],
            "cosmos_remaining": [round(float(x), 5) for x in res["cosmos_remaining_per_dim"]],
            "repeat_last": [round(float(x), 5) for x in res["repeat_last_per_dim"]],
        },
    }
    with open(os.path.join(args.run, "eval.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
