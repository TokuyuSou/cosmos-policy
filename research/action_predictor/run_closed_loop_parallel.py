"""Parallel closed-loop evaluation across GPUs.

Splits episodes into disjoint ranges, runs run_closed_loop_eval.py workers concurrently
(round-robin over GPUs, with the MuJoCo EGL device fix so rendering spreads across GPUs),
then aggregates per-skip-rate success into closed_loop_eval.json.

Run:
    HF_TOKEN=... uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/run_closed_loop_parallel.py \
        --run research/results/dense/none --task PnPCounterToStove \
        --total-episodes 24 --skip-rates 0,0.2,0.3 --gpus 0,1,2 --procs-per-gpu 4 \
        --out research/results/closed_loop/dense_none
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
    runnable re-run script (rerun.sh) next to the results, so any eval can be reproduced from its out dir.
    Written for every run; `vars(args)` is the complete hyperparameter set forwarded to the workers."""
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
            for r in p["by_skip_rate"]:  # one entry per setting (random skip-rate or gate-K)
                if r.get("setting") == tag:
                    ns += r["n_success"]; ne += r["n_episodes"]
                    sk += r["total_skips"]; ca += r["total_calls"]
        agg.append({"setting": tag, "success_rate": ns / max(1, ne),
                    "n_success": ns, "n_episodes": ne,
                    "effective_skip_rate": sk / max(1, sk + ca)})
    return parts, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="predictor",
                    choices=["predictor", "retrieval", "retrieval_oracle", "fusion"])
    ap.add_argument("--run", help="trained predictor run dir (--policy predictor / fusion)")
    ap.add_argument("--data-dir", help="cached episodes for the dictionary (retrieval[_oracle])")
    ap.add_argument("--key", default="prev_state", help="retrieval lookup key")
    ap.add_argument("--knn", type=int, default=1, help="retrieval: nearest neighbours to average")
    ap.add_argument("--fusion-k", type=int, default=10,
                    help="fusion: state-nearest cache chunks the predictor chooses among")
    ap.add_argument("--consensus-encoder", default="",
                    help="retrieval CONSENSUS: R3M encoder ckpt (skip executes the N1-top-K ∩ encoder-top-K entry, "
                         "nearest by N1; falls back to N1 top-1). Default off = plain N1 retrieval.")
    ap.add_argument("--consensus-k", type=int, default=50, help="retrieval consensus: per-key top-K (default 50)")
    ap.add_argument("--fused-encoder", default="",
                    help="retrieval FUSED key: multimodal fused encoder ckpt (image+proprio+prev). When set, the "
                         "retrieval key is this encoder's embedding -- a learned REPLACEMENT for the N1 key. "
                         "Requires --state-source actual_next_proprio. Default off = plain N1 retrieval.")
    ap.add_argument("--state-source", default="actual_next_proprio",
                    help="retrieval: state source. actual_next_proprio = the real, locally-sensed self-state "
                         "at a skip (deployable; matches all recent baselines). "
                         "vla_future_proprio = the VLA's cached +H-step prediction (legacy).")
    ap.add_argument("--metric", default="l2", choices=["l2", "norm"], help="retrieval_oracle: NN distance")
    ap.add_argument("--cache-episodes", type=int, default=0,
                    help="retrieval/rmsegate: first N success episodes as the cache (0 = default). "
                         "Pin both to the same N for a consistent gate-vs-retrieval experiment.")
    ap.add_argument("--skip-policy", default="random",
                    choices=["random", "even", "gate", "gripevent", "rmsegate", "distgate", "disagree", "driftgate",
                             "oraclermse", "fusiongate", "envelope"])
    ap.add_argument("--gate-ks", default="50,150,400", help="N1 top-K thresholds (--skip-policy gate)")
    ap.add_argument("--fusiongate-ks", default="",
                    help="fusiongate: gate corroboration neighbourhood K(s), DECOUPLED from --fusion-k "
                         "(comma-separated ints, or 'auto' to offline-calibrate; empty = use --fusion-k). Swept x --skip-rates.")
    ap.add_argument("--fusiongate-skip-ceiling", type=float, default=0.6,
                    help="fusiongate 'auto' target: pick the smallest gate_K whose held-out gate-pass coverage "
                         ">= this (= the max effective skip it permits). Set > 0.5 to sweep past 50% skip.")
    ap.add_argument("--grip-ckpt", default="", help="trained grip-event model dir (--skip-policy gripevent)")
    ap.add_argument("--grip-qs", default="0.2,0.3,0.4,0.5,0.6,0.7,0.85", help="gripevent skip-rate quantiles")
    ap.add_argument("--rmse-qs", default="0.2,0.3,0.4,0.5,0.6", help="rmsegate skip-rate quantiles")
    ap.add_argument("--rmse-k", type=int, default=10, help="rmsegate: k for the k-NN RMSE lookup")
    ap.add_argument("--rmse-dict-dir", default="", help="rmsegate: RMSE-dict data dir (default = --data-dir)")
    ap.add_argument("--rmse-dict-start", type=int, default=-1, help="rmsegate: keep dict episodes with index >= this")
    ap.add_argument("--dist-qs", default="0.2,0.3,0.4,0.5,0.6,0.8,1.0", help="distgate: d1-quantile sweep")
    ap.add_argument("--disagree-qs", default="0.2,0.3,0.4,0.5,0.6,0.8,1.0", help="disagree: gate-quantile sweep")
    ap.add_argument("--envelope-qs", default="0.1,0.2,0.3,0.5", help="envelope: score-quantile sweep (needs --data-dir)")
    ap.add_argument("--drift-qs", default="0.3,0.5,0.7,0.85,1.0", help="driftgate: obs-drift distance-quantile sweep")
    ap.add_argument("--oracle-rmse-taus", default="0.5,0.75,1.0,1.25,1.5",
                    help="oraclermse: absolute per-dim z-scored RMSE thresholds, all 7 dims incl. gripper "
                         "(skip iff RMSE(N1 top-1, live VLA) < tau)")
    ap.add_argument("--drift-img-dir", default="research/data/pnp_counter_to_stove_dense_img",
                    help="driftgate: episodes WITH images for the observation cache (default = dense_img)")
    ap.add_argument("--drift-views", default="primary,wrist", help="driftgate: views for the obs embedding")
    ap.add_argument("--disagree-k", type=int, default=10, help="disagree: K neighbours for the disagreement signal")
    ap.add_argument("--disagree-w-grip", type=float, default=1.0, help="disagree: weight on z(grip) (0 = pure disagree)")
    ap.add_argument("--max-skips", type=int, default=-1,
                    help="consecutive-skip drift budget forwarded to workers (<0 = unlimited; e.g. 2). "
                         "Tags get a '_b{n}' suffix so budgeted runs aggregate separately.")
    ap.add_argument("--task", default="PnPCounterToStove")
    ap.add_argument("--total-episodes", type=int, default=24)
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--skip-rates", default="0,0.2,0.3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--procs-per-gpu", type=int, default=4)
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--skip-seed", type=int, default=0)
    ap.add_argument("--stagger-sec", type=float, default=12.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    save_run_config(args.out, args)  # persist hyperparameters + rerun.sh before launching workers
    # fusiongate 'auto': offline-calibrate gate_K ONCE in the parent (CPU) so the tag set and every worker
    # use the SAME concrete K -- avoids 12x recalibration and keeps aggregation tags known up front.
    if args.skip_policy == "fusiongate" and "auto" in [x.strip() for x in args.fusiongate_ks.split(",")]:
        cenv = dict(os.environ); cenv["CUDA_VISIBLE_DEVICES"] = ""  # calibrate on CPU, leave GPUs for the eval
        cmd = [sys.executable, os.path.join(HERE, "calibrate_fusiongate.py"),
               "--run", args.run, "--data-dir", args.data_dir, "--fusion-k", str(args.fusion_k),
               "--key", args.key, "--state-source", args.state_source,
               "--cache-episodes", str(args.cache_episodes), "--skip-ceiling", str(args.fusiongate_skip_ceiling),
               "--out", os.path.join(args.out, "fusiongate_calib.json"), "--print-k"]
        kline = [ln for ln in subprocess.check_output(cmd, env=cenv, text=True).splitlines()
                 if ln.startswith("KSTAR=")]
        kstar = kline[-1].split("=", 1)[1].strip()
        args.fusiongate_ks = ",".join(kstar if x.strip() == "auto" else x.strip()
                                      for x in args.fusiongate_ks.split(",") if x.strip())
        print(f"[fusiongate] auto-calibrated gate_K = {kstar} (skip_ceiling={args.fusiongate_skip_ceiling}) "
              f"-> --fusiongate-ks {args.fusiongate_ks}", flush=True)
    for old in glob.glob(os.path.join(args.out, "part_ep*.json")):
        os.remove(old)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    ranges = split_ranges(args.total_episodes, len(gpus) * args.procs_per_gpu)
    mxs = args.max_skips if args.max_skips >= 0 else None
    bsuf = f"_b{mxs}" if mxs is not None else ""  # must match run_closed_loop_eval.build_settings tags
    if args.skip_policy == "gate":
        tags = [f"gateK{k}{bsuf}" for k in args.gate_ks.split(",") if k != ""]
    elif args.skip_policy == "gripevent":
        tags = [f"gripQ{q}{bsuf}" for q in args.grip_qs.split(",") if q != ""]
    elif args.skip_policy == "rmsegate":
        tags = [f"rmseQ{q}{bsuf}" for q in args.rmse_qs.split(",") if q != ""]
    elif args.skip_policy == "distgate":
        tags = [f"distQ{q}{bsuf}" for q in args.dist_qs.split(",") if q != ""]
    elif args.skip_policy == "disagree":
        tags = [f"disagQ{q}{bsuf}" for q in args.disagree_qs.split(",") if q != ""]
    elif args.skip_policy == "envelope":
        tags = [f"envQ{q}{bsuf}" for q in args.envelope_qs.split(",") if q != ""]
    elif args.skip_policy == "driftgate":
        tags = [f"driftQ{q}{bsuf}" for q in args.drift_qs.split(",") if q != ""]
    elif args.skip_policy == "oraclermse":
        tags = [f"oracleRmseT{tau}{bsuf}" for tau in args.oracle_rmse_taus.split(",") if tau != ""]
    elif args.skip_policy == "fusiongate":
        # gate_k resolves to --fusion-k when --fusiongate-ks is empty (matches run_closed_loop_eval).
        gks = [int(x) for x in args.fusiongate_ks.split(",") if x.strip()] or [int(args.fusion_k)]
        tags = [f"fgateK{gk}P{sr}{bsuf}" for gk in gks for sr in args.skip_rates.split(",") if sr != ""]
    elif args.skip_policy == "even":
        tags = [f"even{sr}{bsuf}" for sr in args.skip_rates.split(",") if sr != ""]
    else:
        tags = [f"skip{sr}{bsuf}" for sr in args.skip_rates.split(",") if sr != ""]

    procs = []
    for k, (off, cnt) in enumerate(ranges):
        gpu = gpus[k % len(gpus)]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["MUJOCO_EGL_DEVICE_ID"] = gpu  # spread MuJoCo rendering across GPUs (see launch_collect.py)
        policy_args = ["--policy", args.policy]
        if args.policy == "predictor":
            policy_args += ["--run", args.run]
        elif args.policy == "fusion":
            policy_args += ["--run", args.run, "--data-dir", args.data_dir,
                            "--cache-episodes", str(args.cache_episodes), "--key", args.key,
                            "--state-source", args.state_source, "--fusion-k", str(args.fusion_k)]
        else:
            policy_args += ["--data-dir", args.data_dir, "--cache-episodes", str(args.cache_episodes)]
            policy_args += (["--metric", args.metric] if args.policy == "retrieval_oracle"
                            else ["--key", args.key, "--knn", str(args.knn), "--state-source", args.state_source])
            if args.policy == "retrieval" and args.consensus_encoder:  # consensus retrieval (off by default)
                policy_args += ["--consensus-encoder", args.consensus_encoder, "--consensus-k", str(args.consensus_k)]
            if args.policy == "retrieval" and args.fused_encoder:  # fused-encoder key (off by default)
                policy_args += ["--fused-encoder", args.fused_encoder]
        if args.skip_policy == "gate":
            sweep_args = ["--skip-policy", "gate", "--gate-ks", args.gate_ks, "--data-dir", args.data_dir]
        elif args.skip_policy == "gripevent":
            sweep_args = ["--skip-policy", "gripevent", "--grip-ckpt", args.grip_ckpt, "--grip-qs", args.grip_qs]
        elif args.skip_policy == "rmsegate":
            sweep_args = ["--skip-policy", "rmsegate", "--rmse-qs", args.rmse_qs, "--rmse-k", str(args.rmse_k),
                          "--data-dir", args.data_dir, "--rmse-dict-start", str(args.rmse_dict_start)]
            if args.rmse_dict_dir:
                sweep_args += ["--rmse-dict-dir", args.rmse_dict_dir]
        elif args.skip_policy == "distgate":
            sweep_args = ["--skip-policy", "distgate", "--dist-qs", args.dist_qs, "--data-dir", args.data_dir]
        elif args.skip_policy == "disagree":
            sweep_args = ["--skip-policy", "disagree", "--disagree-qs", args.disagree_qs,
                          "--disagree-k", str(args.disagree_k), "--disagree-w-grip", str(args.disagree_w_grip),
                          "--data-dir", args.data_dir]
        elif args.skip_policy == "envelope":
            # envelope needs --data-dir (demo episodes for sigma + calibration) even for --policy predictor
            sweep_args = ["--skip-policy", "envelope", "--envelope-qs", args.envelope_qs,
                          "--data-dir", args.data_dir, "--state-source", args.state_source]
        elif args.skip_policy == "driftgate":
            sweep_args = ["--skip-policy", "driftgate", "--drift-qs", args.drift_qs,
                          "--drift-img-dir", args.drift_img_dir, "--drift-views", args.drift_views,
                          "--data-dir", args.data_dir]
        elif args.skip_policy == "oraclermse":
            sweep_args = ["--skip-policy", "oraclermse", "--oracle-rmse-taus", args.oracle_rmse_taus]
        elif args.skip_policy == "fusiongate":
            # the gate reuses the fusion policy (already forwarded via policy_args); sweep gate_k x random prob.
            # args.fusiongate_ks is already a concrete K here ('auto' was resolved in the parent above).
            sweep_args = ["--skip-policy", "fusiongate", "--skip-rates", args.skip_rates,
                          "--fusiongate-ks", args.fusiongate_ks,
                          "--fusiongate-skip-ceiling", str(args.fusiongate_skip_ceiling)]
        elif args.skip_policy == "even":
            sweep_args = ["--skip-policy", "even", "--skip-rates", args.skip_rates]
        else:
            sweep_args = ["--skip-rates", args.skip_rates]
        cmd = [sys.executable, os.path.join(HERE, "run_closed_loop_eval.py"),
               *policy_args, *sweep_args, "--task", args.task,
               "--episode-start", str(args.episode_start + off), "--num-episodes", str(cnt),
               "--out", args.out, "--seed", str(args.seed), "--skip-seed", str(args.skip_seed),
               "--max-skips", str(args.max_skips)]
        log = open(os.path.join(args.out, f"worker_gpu{gpu}_ep{args.episode_start + off}.log"), "w")
        print(f"  worker{k}: GPU{gpu} episodes [{args.episode_start + off},{args.episode_start + off + cnt})", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(args.stagger_sec)
    rc = [p.wait() for p in procs]
    print(f"workers exit codes: {rc}", flush=True)

    parts, agg = aggregate(args.out, tags)
    report = {"run": args.run or args.data_dir, "policy": args.policy, "skip_policy": args.skip_policy,
              "task": args.task, "img_mode": parts[0]["img_mode"] if parts else None,
              "state_source": parts[0]["state_source"] if parts else None, "max_skips": mxs,
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
