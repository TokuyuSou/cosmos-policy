"""Launcher: spawn N collection processes across GPUs (intra-GPU parallelism by
running several processes per GPU). Each child is the (unmodified) collector run on a
disjoint episode range; outputs share one dir (filenames keyed by episode index).

Example (largest safe parallelism for ~7.6GB/proc on 48GB GPUs):
    HF_TOKEN=... uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/launch_collect.py \
        --collector collect_data_dense.py --total-episodes 150 \
        --gpus 0,1,2 --procs-per-gpu 5 --query-stride 4 \
        --out-dir research/data/pnp_counter_to_stove_dense
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def split_ranges(total: int, n: int):
    """Split [0,total) into n contiguous, near-even (start, count) ranges."""
    base, rem = divmod(total, n)
    ranges, s = [], 0
    for k in range(n):
        cnt = base + (1 if k < rem else 0)
        if cnt == 0:
            continue
        ranges.append((s, cnt))
        s += cnt
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collector", default="collect_data_dense.py",
                    choices=["collect_data.py", "collect_data_dense.py"])
    ap.add_argument("--task", default="PnPCounterToStove")
    ap.add_argument("--total-episodes", type=int, default=150)
    ap.add_argument("--episode-offset", type=int, default=0,
                    help="first episode index (default 0). Use to APPEND new episodes to an existing "
                         "dataset without overwriting: --episode-offset 150 collects [150, 150+total). "
                         "Episode index seeds the scene, so new indices = new (non-duplicate) scenes.")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--procs-per-gpu", type=int, default=5)
    ap.add_argument("--query-stride", type=int, default=4)
    ap.add_argument("--out-dir", default="research/data/pnp_counter_to_stove_dense")
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--denoising-steps", type=int, default=5)
    ap.add_argument("--no-vla-shadow", action="store_true",
                    help="dense only: skip shadow VLA calls; store zeroed chunk/future_* (proprio+images only)")
    ap.add_argument("--stagger-sec", type=float, default=12.0,
                    help="delay between launches so model-load spikes don't collide")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    nproc = len(gpus) * args.procs_per_gpu
    ranges = split_ranges(args.total_episodes, nproc)
    os.makedirs(args.out_dir, exist_ok=True)
    collector = os.path.join(HERE, args.collector)
    is_dense = "dense" in args.collector

    print(f"Plan: {len(ranges)} processes over GPUs {gpus} ({args.procs_per_gpu}/GPU), "
          f"collector={args.collector}, stride={args.query_stride if is_dense else 'n/a'}", flush=True)
    procs = []
    for k, (start, cnt) in enumerate(ranges):
        gpu = gpus[k % len(gpus)]  # round-robin: consecutive launches hit different GPUs
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # MuJoCo EGL ignores CUDA_VISIBLE_DEVICES; if MUJOCO_EGL_DEVICE_ID is unset it
        # defaults to EGL device 0 (= physical GPU2 here), so ALL camera rendering piles
        # on GPU2 and OOMs/bottlenecks under parallelism. Set it = CUDA id (robosuite's
        # required pairing, binding_utils.py asserts it is a substring of CUDA_VISIBLE_DEVICES).
        # With even per-GPU distribution this spreads rendering across all GPUs.
        env["MUJOCO_EGL_DEVICE_ID"] = gpu
        abs_start = args.episode_offset + start  # absolute episode index (append-safe)
        cmd = [sys.executable, collector, "--task", args.task,
               "--episode-start", str(abs_start), "--num-episodes", str(cnt),
               "--out-dir", args.out_dir, "--seed", str(args.seed),
               "--denoising-steps", str(args.denoising_steps)]
        if is_dense:
            cmd += ["--query-stride", str(args.query_stride)]
            if args.no_vla_shadow:
                cmd += ["--no-vla-shadow"]
        log_path = os.path.join(args.out_dir, f"log_gpu{gpu}_ep{abs_start:04d}-{abs_start + cnt - 1:04d}.log")
        log = open(log_path, "w")
        print(f"  proc{k:02d}: GPU{gpu} episodes [{abs_start},{abs_start + cnt}) -> {log_path}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(args.stagger_sec)

    print(f"All {len(procs)} launched. Waiting for completion...", flush=True)
    rc = [p.wait() for p in procs]
    print(f"exit codes: {rc}", flush=True)
    print("ALL OK" if not any(rc) else f"SOME FAILED: {rc}", flush=True)


if __name__ == "__main__":
    main()
