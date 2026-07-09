"""Parallel closed-loop evaluation across GPUs.

Splits episodes into disjoint ranges, runs run_closed_loop_eval.py workers concurrently
(round-robin over GPUs, with the MuJoCo EGL device fix so rendering spreads across GPUs),
then aggregates per-skip-rate success into closed_loop_eval.json.

Run:
    HF_TOKEN=... uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/run_closed_loop_parallel.py \
        --policy retrieval --data-dir research/data/pnp_counter_to_stove_dense_img \
        --task PnPCounterToStove --skip-policy even --skip-rates 0.2,0.4,0.6 \
        --total-episodes 24 --gpus 0,1,2 --procs-per-gpu 4 \
        --out research/results/closed_loop/ct_stove
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def save_run_config(out_dir, args):
    """Reproducibility: persist the exact invocation + ALL hyperparameters (run_config.json) and a
    runnable re-run script (rerun.sh) next to the results, so any eval can be reproduced from its out dir."""
    cfg = {"argv": sys.argv, "python": sys.executable, "cwd": os.getcwd(),
           "utc": datetime.datetime.utcnow().isoformat() + "Z", "hyperparams": vars(args)}
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    rerun = os.path.join(out_dir, "rerun.sh")
    with open(rerun, "w") as f:
        f.write("#!/bin/bash\n# auto-generated exact re-run (launch from a robocasa-capable env, e.g. via `uv run`)\n"
                "set -u\ncd " + shlex.quote(os.getcwd()) + "\n"
                + " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv) + "\n")
    os.chmod(rerun, 0o755)


def split_ranges(total, n):
    base, rem = divmod(total, n)
    out, s = [], 0
    for k in range(n):
        c = base + (1 if k < rem else 0)
        if c:
            out.append((s, c))
            s += c
    return out


def aggregate(out_dir, tags):
    parts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "part_ep*.json")))]
    agg = []
    for tag in tags:
        ns = ne = sk = ca = 0
        for p in parts:
            for r in p["by_skip_rate"]:  # one entry per setting (skip-rate)
                if r.get("setting") == tag:
                    ns += r["n_success"]; ne += r["n_episodes"]
                    sk += r["total_skips"]; ca += r["total_calls"]
        agg.append({"setting": tag, "success_rate": ns / max(1, ne),
                    "n_success": ns, "n_episodes": ne,
                    "effective_skip_rate": sk / max(1, sk + ca)})
    return parts, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="predictor", choices=["predictor", "retrieval", "tmt"])
    ap.add_argument("--run", help="trained predictor run dir (--policy predictor)")
    ap.add_argument("--data-dir", help="cached episodes for the dictionary (retrieval / tmt)")
    ap.add_argument("--key", default="prev_state", help="retrieval lookup key")
    ap.add_argument("--knn", type=int, default=1, help="retrieval: nearest neighbours to combine")
    ap.add_argument("--fused-encoder", default="",
                    help="retrieval FUSED key: multimodal fused encoder ckpt (image+proprio+prev). When set, the "
                         "retrieval key is this encoder's embedding -- a learned REPLACEMENT for the N1 key. "
                         "Requires --state-source actual_next_proprio. Default off = plain N1 retrieval.")
    ap.add_argument("--tmt-encoder", default="", help="trained TMT encoder .pt (--policy tmt)")
    ap.add_argument("--tmt-w", type=float, default=-1.0,
                    help="--policy tmt: override the learned block weight w (<=0 = learned)")
    ap.add_argument("--state-source", default="actual_next_proprio",
                    help="retrieval: state source. actual_next_proprio = the real, locally-sensed self-state "
                         "at a skip (deployable; matches all recent baselines).")
    ap.add_argument("--cache-episodes", type=int, default=0,
                    help="retrieval/tmt: first N success episodes as the cache (0 = default train split)")
    ap.add_argument("--skip-policy", default="even", choices=["even", "random"])
    ap.add_argument("--sim", default="robocasa", choices=["robocasa", "libero"],
                    help="simulator backend forwarded to each worker (sim.py)")
    ap.add_argument("--task", default="PnPCounterToStove")
    ap.add_argument("--total-episodes", type=int, default=24)
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--skip-rates", default="0.2,0.4,0.6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--procs-per-gpu", type=int, default=4)
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--skip-seed", type=int, default=0)
    ap.add_argument("--stagger-sec", type=float, default=12.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    save_run_config(args.out, args)  # persist hyperparameters + rerun.sh before launching workers
    for old in glob.glob(os.path.join(args.out, "part_ep*.json")):
        os.remove(old)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    ranges = split_ranges(args.total_episodes, len(gpus) * args.procs_per_gpu)
    # tags must match run_closed_loop_eval.build_settings
    pref = "even" if args.skip_policy == "even" else "skip"
    tags = [f"{pref}{sr}" for sr in args.skip_rates.split(",") if sr != ""]

    procs = []
    for k, (off, cnt) in enumerate(ranges):
        gpu = gpus[k % len(gpus)]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["MUJOCO_EGL_DEVICE_ID"] = gpu  # spread MuJoCo rendering across GPUs (see launch_collect.py)
        policy_args = ["--policy", args.policy]
        if args.policy == "predictor":
            policy_args += ["--run", args.run]
        elif args.policy == "tmt":
            assert args.tmt_encoder, "--policy tmt requires --tmt-encoder"
            policy_args += ["--data-dir", args.data_dir, "--cache-episodes", str(args.cache_episodes),
                            "--state-source", args.state_source, "--tmt-encoder", args.tmt_encoder,
                            "--tmt-w", str(args.tmt_w)]
        else:  # retrieval
            policy_args += ["--data-dir", args.data_dir, "--cache-episodes", str(args.cache_episodes),
                            "--key", args.key, "--knn", str(args.knn), "--state-source", args.state_source]
            if args.fused_encoder:  # fused-encoder key (off by default = plain N1 retrieval)
                policy_args += ["--fused-encoder", args.fused_encoder]
        sweep_args = ["--skip-policy", args.skip_policy, "--skip-rates", args.skip_rates]
        cmd = [sys.executable, os.path.join(HERE, "run_closed_loop_eval.py"),
               "--sim", args.sim, *policy_args, *sweep_args, "--task", args.task,
               "--episode-start", str(args.episode_start + off), "--num-episodes", str(cnt),
               "--out", args.out, "--seed", str(args.seed), "--skip-seed", str(args.skip_seed)]
        log = open(os.path.join(args.out, f"worker_gpu{gpu}_ep{args.episode_start + off}.log"), "w")
        print(f"  worker{k}: GPU{gpu} episodes [{args.episode_start + off},{args.episode_start + off + cnt})", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(args.stagger_sec)
    rc = [p.wait() for p in procs]
    print(f"workers exit codes: {rc}", flush=True)

    parts, agg = aggregate(args.out, tags)
    report = {"run": args.run or args.data_dir, "policy": args.policy, "skip_policy": args.skip_policy,
              "task": args.task, "img_mode": parts[0]["img_mode"] if parts else None,
              "state_source": parts[0]["state_source"] if parts else None,
              "total_episodes": args.total_episodes, "by_skip_rate": agg}
    with open(os.path.join(args.out, "closed_loop_eval.json"), "w") as f:
        json.dump(report, f, indent=2)
    # Also save each setting's aggregated results under ITS video dir.
    for r in agg:
        vdir = os.path.join(args.out, "videos", r["setting"])
        os.makedirs(vdir, exist_ok=True)
        rec = {"settings": {"run": args.run or args.data_dir, "img_mode": report["img_mode"],
                            "state_source": report["state_source"], "task": args.task,
                            "setting": r["setting"], "total_episodes": args.total_episodes,
                            "episode_start": args.episode_start, "skip_seed": args.skip_seed,
                            "gpus": args.gpus, "procs_per_gpu": args.procs_per_gpu},
               "results": r}
        with open(os.path.join(vdir, "eval.json"), "w") as f:
            json.dump(rec, f, indent=2)
    print("\n=== success vs effective-skip ===", flush=True)
    for r in agg:
        print(f"  {r['setting']:>10}  eff_skip={r['effective_skip_rate']:.2f}  "
              f"success={r['success_rate']:.3f} ({r['n_success']}/{r['n_episodes']})", flush=True)


if __name__ == "__main__":
    main()
