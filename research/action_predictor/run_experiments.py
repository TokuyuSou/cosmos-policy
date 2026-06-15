"""Train + evaluate all future-image pooling variants, parallelized across GPUs,
then aggregate RESULTS.md.

Each variant's training is a lightweight job (small model, no Cosmos/robosuite), so we
run them concurrently, round-robin across --gpus. Then evaluate each and aggregate.

Run after data collection finishes:
    uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/run_experiments.py \
        --data-dir research/data/pnp_counter_to_stove_dense \
        --results-dir research/results/dense --gpus 0,1,2 --epochs 150
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def launch(script, run_dir, gpu, extra):
    os.makedirs(run_dir, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, os.path.join(HERE, script)] + extra
    log = open(os.path.join(run_dir, f"{os.path.splitext(script)[0]}.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_all(procs, phase):
    rc = {}
    for mode, p in procs:
        rc[mode] = p.wait()
    print(f"[{phase}] exit codes: {rc}", flush=True)
    bad = [m for m, c in rc.items() if c != 0]
    if bad:
        print(f"[{phase}] WARNING failed: {bad}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", default="research/results/dense")
    ap.add_argument("--modes", default="none,mean,meanstd,grid,conv,full")
    ap.add_argument("--img-views", default="primary")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    os.makedirs(args.results_dir, exist_ok=True)

    # Phase 1: train all variants in parallel (round-robin GPU).
    print(f"Training {len(modes)} variants across GPUs {gpus} ...", flush=True)
    train_procs = []
    for k, mode in enumerate(modes):
        run_dir = os.path.join(args.results_dir, mode)
        extra = ["--data-dir", args.data_dir, "--out", run_dir,
                 "--img-mode", mode, "--img-views", args.img_views,
                 "--epochs", str(args.epochs), "--width", str(args.width),
                 "--depth", str(args.depth), "--heads", str(args.heads),
                 "--dropout", str(args.dropout), "--seed", str(args.seed)]
        gpu = gpus[k % len(gpus)]
        print(f"  train {mode} -> GPU{gpu}", flush=True)
        train_procs.append((mode, launch("train.py", run_dir, gpu, extra)))
        time.sleep(2)
    wait_all(train_procs, "train")

    # Phase 2: evaluate all variants in parallel (round-robin GPU).
    eval_procs = []
    for k, mode in enumerate(modes):
        run_dir = os.path.join(args.results_dir, mode)
        extra = ["--run", run_dir, "--data-dir", args.data_dir]
        gpu = gpus[k % len(gpus)]
        eval_procs.append((mode, launch("evaluate.py", run_dir, gpu, extra)))
        time.sleep(1)
    wait_all(eval_procs, "eval")

    # Phase 3: aggregate.
    reports = []
    for mode in modes:
        ej = os.path.join(args.results_dir, mode, "eval.json")
        if os.path.exists(ej):
            reports.append(json.load(open(ej)))
    if reports:
        write_results_md(args, reports)
    else:
        print("No eval.json found; nothing to aggregate.", flush=True)


def write_results_md(args, reports):
    out = os.path.join(args.results_dir, "RESULTS.md")
    ref = reports[0]
    cr = ref["rmse_overall"]["cosmos_remaining"]
    crg = ref["rmse_grouped"]["cosmos_remaining"]
    rl = ref["rmse_overall"]["repeat_last"]
    rlg = ref["rmse_grouped"]["repeat_last"]
    lines = []
    lines.append("# Action Predictor — RMSE vs Cosmos Policy (PnPCounterToStove, PandaOmron)\n")
    lines.append(f"- Data dir: `{args.data_dir}`")
    lines.append(f"- Val episodes: {ref['n_val_episodes']} | val samples (call pairs): {ref['n_val_samples']}")
    lines.append(f"- Architecture: width={args.width}, depth={args.depth}, heads={args.heads}, dropout={args.dropout}")
    lines.append(f"- Future-image view(s): {args.img_views}\n")
    lines.append("RMSE in physical action space on held-out successful episodes. "
                 "Target = actually-executed next 16 actions (`chunk[i+1][:16]`). **Lower is better.**\n")
    lines.append("## Baselines (model-independent)\n")
    lines.append("| baseline | overall | pos | rot | gripper |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| **Cosmos remaining 16** | {cr:.4f} | {crg['pos']:.4f} | {crg['rot']:.4f} | {crg['gripper']:.4f} |")
    lines.append(f"| repeat-last action | {rl:.4f} | {rlg['pos']:.4f} | {rlg['rot']:.4f} | {rlg['gripper']:.4f} |")
    lines.append("\n## Action predictor (by future-image pooling mode)\n")
    lines.append("| img_mode | params (M) | overall | pos | rot | gripper | beats Cosmos-rem? |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in reports:
        o = r["rmse_overall"]["predictor"]
        g = r["rmse_grouped"]["predictor"]
        beat = "yes" if o < cr else "-"
        lines.append(f"| {r['img_mode']} | {r['n_params_M']} | {o:.4f} | {g['pos']:.4f} | {g['rot']:.4f} | {g['gripper']:.4f} | {beat} |")
    best = min(reports, key=lambda r: r["rmse_overall"]["predictor"])
    lines.append(f"\n**Best predictor:** `{best['img_mode']}` "
                 f"(overall RMSE {best['rmse_overall']['predictor']:.4f} vs Cosmos-remaining {cr:.4f}).\n")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
