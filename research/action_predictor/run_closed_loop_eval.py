"""Closed-loop evaluation: measure the success rate of a skip-time policy under a SkipPolicy
(even or random skipping) at one or more skip rates, on RoboCasa (or LIBERO via --sim).

Three deployable policies share the predict_chunk API the closed-loop runner expects:
    predictor  -- a trained action-predictor net (--run).
    retrieval  -- dictionary lookup keyed on locally-available features; N1 (prev+proprio) key by
                  default, or a learned FUSED-encoder embedding key with --fused-encoder.
    tmt        -- Transition-Metric-Transformer retrieval key (--tmt-encoder); the runner threads the
                  previous decision's frames into the query.

Run:
    CUDA_VISIBLE_DEVICES=0 HF_TOKEN=... \
    uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/run_closed_loop_eval.py \
        --policy retrieval --data-dir research/data/pnp_counter_to_stove_dense_img \
        --task PnPCounterToStove --skip-policy even --skip-rates 0.2,0.4,0.6 \
        --episode-start 5000 --num-episodes 12 --out research/results/closed_loop/ct_stove
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cosmos_policy.experiments.robot.cosmos_utils import get_model, init_t5_text_embeddings_cache, load_dataset_stats
from cosmos_policy.utils.utils import set_seed_everywhere

from closed_loop import run_closed_loop_episode
from common import NUM_OPEN_LOOP_STEPS
from sim import get_backend
from predictor_policy import PredictorPolicy
from retrieval_policy import RetrievalPolicy
from tmt_policy import TMTRetrievalPolicy
from reference_servo_tmt_policy import ReferenceServoTMTPolicy
from skip_policy import make_skip_policy


def build_policy(args):
    """Build the skip-time policy (all share the predict_chunk API closed_loop expects)."""
    if args.policy == "retrieval":
        assert args.data_dir, "--data-dir (the cached episodes) is required for --policy retrieval"
        return RetrievalPolicy(args.data_dir, key=args.key, k=args.knn, state_source=args.state_source,
                               cache_episodes=(args.cache_episodes or None), fused_encoder=args.fused_encoder)
    if args.policy == "tmt":
        assert args.data_dir, "--data-dir (the cached episodes) is required for --policy tmt"
        assert args.tmt_encoder, "--tmt-encoder (the trained TMT .pt) is required for --policy tmt"
        return TMTRetrievalPolicy(args.data_dir, args.tmt_encoder, state_source=args.state_source,
                                  cache_episodes=(args.cache_episodes or None), w=args.tmt_w)
    if args.policy == "reference_servo_tmt":
        assert args.data_dir, "--data-dir is required for --policy reference_servo_tmt"
        assert args.tmt_encoder, "--tmt-encoder is required for --policy reference_servo_tmt"
        assert args.servo_model, "--servo-model is required for --policy reference_servo_tmt"
        return ReferenceServoTMTPolicy(
            args.data_dir, args.tmt_encoder, args.servo_model,
            state_source=args.state_source,
            cache_episodes=(args.cache_episodes or None), w=args.tmt_w,
            tmt_period_steps=NUM_OPEN_LOOP_STEPS,
            local_replan_steps=args.local_replan_steps,
        )
    assert args.run, "--run (trained predictor dir) is required for --policy predictor"
    return PredictorPolicy(args.run)


def build_settings(args):
    """List of (skip_policy, tag, extra) to evaluate: one entry per --skip-rates value.

    even   -- deterministic maximally-even pattern (Bresenham/error-diffusion; skips never cluster ->
              minimal open-loop drift). No seed needed.
    random -- fixed-rate stochastic skipping (seeded by --skip-seed).
    """
    if args.skip_policy == "even":
        return [(make_skip_policy("even", float(sr)), f"even{sr}", {"skip_rate_target": float(sr)})
                for sr in args.skip_rates.split(",") if sr != ""]
    return [(make_skip_policy("random", float(sr), seed=args.skip_seed), f"skip{sr}",
             {"skip_rate_target": float(sr)})
            for sr in args.skip_rates.split(",") if sr != ""]


def eval_setting(cfg, cosmos, dataset_stats, predictor, policy, tag, extra, task, ep_start, n_ep, out_dir,
                 backend=None, local_replan_steps=0):
    video_dir = os.path.join(out_dir, "videos", tag)  # save every episode's rollout video
    succ, skips, calls, episodes, traces, hit_rows = 0, 0, 0, [], [], []
    for ep in range(ep_start, ep_start + n_ep):
        r = run_closed_loop_episode(cfg, cosmos, dataset_stats, predictor, policy, task, ep,
                                    video_dir=video_dir, backend=backend,
                                    local_replan_steps=(local_replan_steps or None))
        succ += int(r["success"])
        skips += r["n_skip"]
        calls += r["n_call"]
        episodes.append({"ep": ep, "success": r["success"], "length": r["length"],
                         "n_call": r["n_call"], "n_skip": r["n_skip"],
                         "n_local_replan": r.get("n_local_replan", 0)})
        print(f"  [{tag}] ep{ep} success={r['success']} len={r['length']} "
              f"calls={r['n_call']} skips={r['n_skip']}", flush=True)
        for d in r.get("trace", []):
            traces.append({"ep": ep, **d})
        for h in r.get("hit_images", []):
            hit_rows.append({"ep": ep, **h})
    n_dec = skips + calls
    effective_local_steps = int(local_replan_steps or NUM_OPEN_LOOP_STEPS)
    result = {"setting": tag, **extra, "success_rate": succ / max(1, n_ep),
              "n_success": succ, "n_episodes": n_ep, "total_skips": skips, "total_calls": calls,
              "effective_skip_rate": skips / max(1, n_dec),
              "local_replan_steps": effective_local_steps, "episodes": episodes}
    # Save full settings + results for THIS eval next to its videos (range-specific = parallel-safe).
    os.makedirs(video_dir, exist_ok=True)
    # per-decision skip log (where + skip/call + gate score/threshold), one JSON per line
    with open(os.path.join(video_dir, f"skip_trace_ep{ep_start:04d}-{ep_start + n_ep - 1:04d}.jsonl"), "w") as f:
        for d in traces:
            f.write(json.dumps(d) + "\n")
    if hit_rows:  # DUMP_HIT_IMAGES: live decision-point frame + matched-cache provenance per skip (for cache_hit_viz.py)
        _g = lambda k: np.array([(h[k] if h.get(k) is not None else -1) for h in hit_rows])
        np.savez_compressed(
            os.path.join(video_dir, f"hits_ep{ep_start:04d}-{ep_start + n_ep - 1:04d}.npz"),
            live_image=np.stack([h["live_image"] for h in hit_rows]).astype(np.uint8),
            ep=_g("ep"), decision_idx=_g("decision_idx"), step=_g("step"),
            src_ep=_g("src_ep"), src_imgidx=_g("src_imgidx"), src_t=_g("src_t"),
            cache_idx=_g("cache_idx"), dist=np.array([h["dist"] for h in hit_rows], dtype=np.float32))
    record = {"settings": {"run": predictor.run_dir, "img_mode": predictor.img_mode,
                           "state_source": predictor.state_source, "task": task, "setting": tag, **extra,
                           "local_replan_steps": effective_local_steps,
                           "episode_start": ep_start, "num_episodes": n_ep, "seed": cfg.seed,
                           "num_denoising_steps_action": cfg.num_denoising_steps_action},
              "results": result}
    with open(os.path.join(video_dir, f"eval_ep{ep_start:04d}-{ep_start + n_ep - 1:04d}.json"), "w") as f:
        json.dump(record, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="predictor",
                    choices=["predictor", "retrieval", "tmt", "reference_servo_tmt"],
                    help="predictor = trained net (--run); retrieval = dictionary lookup (--data-dir; N1 key, "
                         "or fused-encoder key with --fused-encoder); tmt = transition-metric-transformer "
                         "retrieval key (--data-dir + --tmt-encoder); reference_servo_tmt = persistent "
                         "TMT@16 + Ridge@4 (--servo-model)")
    ap.add_argument("--run", help="trained predictor run dir (required for --policy predictor)")
    ap.add_argument("--data-dir", help="cached episodes for the retrieval dictionary (retrieval / tmt)")
    ap.add_argument("--key", default="prev_state", help="retrieval lookup key: prev|prev_state|prev_state_img")
    ap.add_argument("--knn", type=int, default=1, help="retrieval: number of nearest neighbours to combine")
    ap.add_argument("--fused-encoder", default="",
                    help="retrieval FUSED key: path to a multimodal fused encoder ckpt (image+proprio+prev). "
                         "When set, the retrieval key is this encoder's embedding -- a learned REPLACEMENT for "
                         "the N1 (prev_state) key. Requires --state-source actual_next_proprio. Default off.")
    ap.add_argument("--tmt-encoder", default="",
                    help="trained TMT encoder .pt (--policy tmt): transition-transformer retrieval key; "
                         "the runner threads the previous decision's frames into the query")
    ap.add_argument("--tmt-w", type=float, default=-1.0,
                    help="--policy tmt: override the learned block weight w (val-calibrated, e.g. 0.5); <=0 = learned")
    ap.add_argument("--servo-model", default="",
                    help="frozen Reference-Servo Ridge4 .npz (--policy reference_servo_tmt)")
    ap.add_argument("--local-replan-steps", type=int, default=0,
                    help="within a skipped 16-step block, re-query every N steps (0 = once per block)")
    ap.add_argument("--state-source", default="actual_next_proprio",
                    help="retrieval: state feature source. actual_next_proprio = the real, locally-sensed "
                         "self-state at a skip (deployable; matches all recent baselines).")
    ap.add_argument("--cache-episodes", type=int, default=0,
                    help="retrieval/tmt: use the first N success episodes as the cache (0 = the default "
                         "train split)")
    ap.add_argument("--skip-policy", default="even", choices=["even", "random"],
                    help="even = deterministic MAXIMALLY-EVEN pattern at the --skip-rates rate "
                         "(Bresenham/error-diffusion; skips never cluster -> minimal open-loop drift); "
                         "random = fixed-rate stochastic skipping (seeded by --skip-seed)")
    ap.add_argument("--sim", default="robocasa", choices=["robocasa", "libero"],
                    help="simulator backend (sim.py): robocasa task name, or libero '<suite>:<task_id>'")
    ap.add_argument("--task", default="PnPCounterToStove")
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--num-episodes", type=int, default=12)
    ap.add_argument("--skip-rates", default="0.2,0.4,0.6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--skip-seed", type=int, default=0)
    ap.add_argument("--denoising-steps", type=int, default=5)
    args = ap.parse_args()

    if args.policy == "reference_servo_tmt":
        assert args.local_replan_steps == 4, (
            "--policy reference_servo_tmt requires --local-replan-steps 4"
        )

    os.makedirs(args.out, exist_ok=True)
    backend = get_backend(args.sim)
    cfg = backend.build_cfg(args.task, args.seed, args.episode_start + args.num_episodes, args.denoising_steps)
    set_seed_everywhere(cfg.seed)
    backend.validate_cfg(cfg)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    cosmos, _ = get_model(cfg)
    predictor = build_policy(args)
    print(f"policy={args.policy} src={predictor.run_dir} | img_mode={predictor.img_mode} "
          f"state_source={predictor.state_source}", flush=True)

    settings = build_settings(args)
    print(f"skip_policy={args.skip_policy} | settings={[t for _, t, _ in settings]}", flush=True)
    results = [eval_setting(cfg, cosmos, dataset_stats, predictor, pol, tag, extra, args.task,
                            args.episode_start, args.num_episodes, args.out, backend=backend,
                            local_replan_steps=args.local_replan_steps)
               for pol, tag, extra in settings]
    report = {"run": predictor.run_dir, "policy": args.policy, "skip_policy": args.skip_policy,
              "task": args.task, "img_mode": predictor.img_mode, "state_source": predictor.state_source,
              "local_replan_steps": int(args.local_replan_steps or NUM_OPEN_LOOP_STEPS),
              "episodes": [args.episode_start, args.episode_start + args.num_episodes],
              "by_skip_rate": results}  # key kept for backward-compat; holds one entry per setting
    # Range-specific filename so parallel workers (disjoint episode ranges) don't clobber.
    fname = f"part_ep{args.episode_start:04d}-{args.episode_start + args.num_episodes - 1:04d}.json"
    with open(os.path.join(args.out, fname), "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== success vs effective-skip ===", flush=True)
    for r in results:
        print(f"  {r['setting']:>10}  eff_skip={r['effective_skip_rate']:.2f}  "
              f"success={r['success_rate']:.3f} ({r['n_success']}/{r['n_episodes']})", flush=True)


if __name__ == "__main__":
    main()
