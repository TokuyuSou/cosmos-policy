"""Closed-loop evaluation: measure success rate of a trained predictor under a SkipPolicy
(random skipping for now) at one or more skip rates, on RoboCasa.

Run:
    CUDA_VISIBLE_DEVICES=0 HF_TOKEN=... \
    uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/run_closed_loop_eval.py \
        --run research/results/dense/none --task PnPCounterToStove \
        --episode-start 0 --num-episodes 12 --skip-rates 0,0.2,0.3 \
        --out research/results/closed_loop/dense_none
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import get_model, init_t5_text_embeddings_cache, load_dataset_stats
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import validate_config
from cosmos_policy.utils.utils import set_seed_everywhere

from closed_loop import run_closed_loop_episode
from collect_data import build_cfg
from predictor_policy import PredictorPolicy
from retrieval_policy import FusionRetrievalPolicy, OracleRetrievalPolicy, RetrievalPolicy
from skip_policy import (
    DisagreeGateSkipPolicy,
    DistGateSkipPolicy,
    EnvelopeGateSkipPolicy,
    FusionGateSkipPolicy,
    GateSkipPolicy,
    OracleRmseGateSkipPolicy,
    calibrate_fusion_gate_k,
    make_skip_policy,
)


def build_policy(args):
    """Build the skip-time policy (all share the predict_chunk / lookup API closed_loop expects).

    predictor        : trained neural net.
    retrieval        : dictionary lookup keyed on locally-available features (deployable).
    retrieval_oracle : at a skip, run the VLA and look up the nearest cached action window to
                       its predicted chunk (diagnostic upper bound; --metric l2|norm).
    fusion           : predictor-directed retrieval -- run the predictor, then execute the K
                       state-nearest cache chunk closest to its output (--run + --data-dir).
    """
    if args.policy == "fusion":
        assert args.run, "--run (trained predictor dir) is required for --policy fusion"
        assert args.data_dir, "--data-dir (the cached episodes) is required for --policy fusion"
        return FusionRetrievalPolicy(args.run, args.data_dir, K=args.fusion_k, key=args.key,
                                     state_source=args.state_source, cache_episodes=(args.cache_episodes or None))
    if args.policy == "retrieval":
        assert args.data_dir, "--data-dir (the cached episodes) is required for --policy retrieval"
        return RetrievalPolicy(args.data_dir, key=args.key, k=args.knn, state_source=args.state_source,
                               cache_episodes=(args.cache_episodes or None),
                               consensus_encoder=args.consensus_encoder, consensus_k=args.consensus_k,
                               fused_encoder=args.fused_encoder)
    if args.policy == "retrieval_oracle":
        assert args.data_dir, "--data-dir (the cached episodes) is required for --policy retrieval_oracle"
        return OracleRetrievalPolicy(args.data_dir, metric=args.metric)
    assert args.run, "--run (trained predictor dir) is required for --policy predictor"
    return PredictorPolicy(args.run)


def resolve_max_skips(args):
    """--max-skips < 0 (default) = unlimited (None); >= 0 = consecutive-skip drift budget."""
    return args.max_skips if (args.max_skips is not None and args.max_skips >= 0) else None


def build_settings(args, predictor):
    """List of (skip_policy, tag, extra) to evaluate. random -> sweep --skip-rates; gate -> --gate-ks.

    A drift budget (--max-skips >= 0) appends '_b{n}' to each setting tag and records max_skips in extra,
    so budgeted runs aggregate/save separately from unbudgeted ones.
    """
    mxs = resolve_max_skips(args)
    bsuf, bex = (f"_b{mxs}" if mxs is not None else ""), {"max_skips": mxs}
    if args.skip_policy == "gate":
        assert args.data_dir, "--data-dir (cached episodes) is required for --skip-policy gate"
        return [(GateSkipPolicy(predictor, args.data_dir, int(k)), f"gateK{k}{bsuf}", {"gate_k": int(k), **bex})
                for k in args.gate_ks.split(",") if k != ""]
    if args.skip_policy == "gripevent":
        assert args.grip_ckpt, "--grip-ckpt (trained grip-event model dir) is required for --skip-policy gripevent"
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gripper_event"))
        from grip_skip_policy import GripEventSkipPolicy
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return [(GripEventSkipPolicy(args.grip_ckpt, float(q), device=dev), f"gripQ{q}{bsuf}", {"q": float(q), **bex})
                for q in args.grip_qs.split(",") if q != ""]
    if args.skip_policy == "rmsegate":
        assert args.data_dir, "--data-dir (the dense_img episodes) is required for --skip-policy rmsegate"
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "retrieval_error"))
        from rmse_skip_policy import RetrievalErrorSkipPolicy
        ce = args.cache_episodes                                      # 0 -> data_dir's train split (= retrieval default)
        ddir = args.rmse_dict_dir or None                            # dict source (default = data_dir)
        dstart = args.rmse_dict_start if args.rmse_dict_start >= 0 else None
        return [(RetrievalErrorSkipPolicy(args.data_dir, float(q), cache_episodes=ce, k=args.rmse_k,
                                          dict_data_dir=ddir, dict_start=dstart),
                 f"rmseQ{q}{bsuf}", {"q": float(q), "rmse_k": args.rmse_k, "cache_episodes": ce,
                                     "dict_dir": ddir, "dict_start": dstart, **bex})
                for q in args.rmse_qs.split(",") if q != ""]
    if args.skip_policy == "distgate":
        assert args.data_dir, "--data-dir (the cached episodes) is required for --skip-policy distgate"
        return [(DistGateSkipPolicy(args.data_dir, float(q), key=args.key, knn=args.knn,
                                    state_source=args.state_source, cache_episodes=args.cache_episodes),
                 f"distQ{q}{bsuf}", {"q": float(q), **bex})
                for q in args.dist_qs.split(",") if q != ""]
    if args.skip_policy == "driftgate":
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "retrieval_oracle"))
        from drift_skip_policy import ObsDriftSkipPolicy
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        idir = args.drift_img_dir or args.data_dir  # observation (image) source; default = dense_img
        # split seed defaults to 0 inside the gate, matching the retrieval cache / distgate convention
        return [(ObsDriftSkipPolicy(idir, float(q), views=args.drift_views, device=dev),
                 f"driftQ{q}{bsuf}", {"q": float(q), "views": args.drift_views, "drift_img_dir": idir, **bex})
                for q in args.drift_qs.split(",") if q != ""]
    if args.skip_policy == "oraclermse":
        assert args.policy == "retrieval", "--skip-policy oraclermse gates the N1 top-1, so needs --policy retrieval"
        return [(OracleRmseGateSkipPolicy(predictor, float(tau)), f"oracleRmseT{tau}{bsuf}", {"tau": float(tau), **bex})
                for tau in args.oracle_rmse_taus.split(",") if tau != ""]
    if args.skip_policy == "disagree":
        assert args.data_dir, "--data-dir (the cached episodes) is required for --skip-policy disagree"
        return [(DisagreeGateSkipPolicy(args.data_dir, float(q), key=args.key, knn=args.knn,
                                        state_source=args.state_source, cache_episodes=args.cache_episodes,
                                        K=args.disagree_k, w_grip=args.disagree_w_grip),
                 f"disagQ{q}{bsuf}", {"q": float(q), "K": args.disagree_k, "w_grip": args.disagree_w_grip, **bex})
                for q in args.disagree_qs.split(",") if q != ""]
    if args.skip_policy == "envelope":
        assert args.data_dir, "--data-dir (the demo episodes for sigma + calibration) is required for --skip-policy envelope"
        return [(EnvelopeGateSkipPolicy(predictor, args.data_dir, float(q),
                                        state_source=args.state_source, cache_episodes=args.cache_episodes),
                 f"envQ{q}{bsuf}", {"q": float(q), **bex})
                for q in args.envelope_qs.split(",") if q != ""]
    if args.skip_policy == "fusiongate":
        assert args.policy == "fusion", "--skip-policy fusiongate gates the fusion policy, so needs --policy fusion"
        # gate_k (corroboration neighbourhood) is DECOUPLED from fusion's --fusion-k. Tokens are ints or "auto"
        # (offline-calibrated: smallest gate_k with coverage >= --fusiongate-skip-ceiling); empty -> fusion.K.
        toks = [x.strip() for x in args.fusiongate_ks.split(",") if x.strip()] or [str(int(predictor.K))]
        gks = []
        for tok in toks:
            if tok.lower() == "auto":
                kstar, cov, nval = calibrate_fusion_gate_k(predictor, args.data_dir, args.fusiongate_skip_ceiling)
                print(f"[fusiongate] auto-calibrated gate_K={kstar} (skip_ceiling={args.fusiongate_skip_ceiling}, "
                      f"coverage={cov[kstar]:.3f}, n_val={nval})", flush=True)
                gks.append(kstar)
            else:
                gks.append(int(tok))
        gks = list(dict.fromkeys(gks))  # dedup, preserve order
        return [(FusionGateSkipPolicy(predictor, float(sr), gate_k=gk, seed=args.skip_seed),
                 f"fgateK{gk}P{sr}{bsuf}",
                 {"gate_k": gk, "fusion_k": int(predictor.K), "skip_rate_target": float(sr), **bex})
                for gk in gks for sr in args.skip_rates.split(",") if sr != ""]
    if args.skip_policy == "even":
        # deterministic maximally-even pattern at each --skip-rates value (anti-clustering; no seed needed)
        return [(make_skip_policy("even", float(sr)), f"even{sr}{bsuf}",
                 {"skip_rate_target": float(sr), **bex})
                for sr in args.skip_rates.split(",") if sr != ""]
    return [(make_skip_policy("random", float(sr), seed=args.skip_seed), f"skip{sr}{bsuf}",
             {"skip_rate_target": float(sr), **bex})
            for sr in args.skip_rates.split(",") if sr != ""]


def eval_setting(cfg, cosmos, dataset_stats, predictor, policy, tag, extra, task, ep_start, n_ep, out_dir,
                 max_skips=None):
    video_dir = os.path.join(out_dir, "videos", tag)  # save every episode's rollout video
    succ, skips, calls, episodes, traces, hit_rows = 0, 0, 0, [], [], []
    for ep in range(ep_start, ep_start + n_ep):
        r = run_closed_loop_episode(cfg, cosmos, dataset_stats, predictor, policy, task, ep,
                                    collect_dagger=False, video_dir=video_dir, max_skips=max_skips)
        succ += int(r["success"])
        skips += r["n_skip"]
        calls += r["n_call"]
        episodes.append({"ep": ep, "success": r["success"], "length": r["length"],
                         "n_call": r["n_call"], "n_skip": r["n_skip"]})
        print(f"  [{tag}] ep{ep} success={r['success']} len={r['length']} "
              f"calls={r['n_call']} skips={r['n_skip']}", flush=True)
        for d in r.get("trace", []):
            traces.append({"ep": ep, **d})
        for h in r.get("hit_images", []):
            hit_rows.append({"ep": ep, **h})
    n_dec = skips + calls
    result = {"setting": tag, **extra, "success_rate": succ / max(1, n_ep),
              "n_success": succ, "n_episodes": n_ep, "total_skips": skips, "total_calls": calls,
              "effective_skip_rate": skips / max(1, n_dec), "episodes": episodes}
    # Save full settings + results for THIS eval next to its videos (range-specific = parallel-safe).
    os.makedirs(video_dir, exist_ok=True)
    # per-decision skip log (where + which gate + skip/call + gate score/threshold), one JSON per line
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
                           "episode_start": ep_start, "num_episodes": n_ep, "seed": cfg.seed,
                           "num_denoising_steps_action": cfg.num_denoising_steps_action},
              "results": result}
    with open(os.path.join(video_dir, f"eval_ep{ep_start:04d}-{ep_start + n_ep - 1:04d}.json"), "w") as f:
        json.dump(record, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="predictor",
                    choices=["predictor", "retrieval", "retrieval_oracle", "fusion"],
                    help="predictor = trained net; retrieval = dictionary lookup; "
                         "retrieval_oracle = look up the nearest cached window to the VLA's own chunk; "
                         "fusion = predictor picks the closest of the K state-nearest cache chunks "
                         "(--run + --data-dir)")
    ap.add_argument("--run", help="trained predictor run dir (required for --policy predictor)")
    ap.add_argument("--data-dir", help="cached episodes for the retrieval dictionary (retrieval[_oracle])")
    ap.add_argument("--key", default="prev_state", help="retrieval lookup key: prev|prev_state|prev_state_img")
    ap.add_argument("--knn", type=int, default=1, help="retrieval: number of nearest neighbours to average")
    ap.add_argument("--fusion-k", type=int, default=10,
                    help="fusion: number of state-nearest cache chunks the predictor chooses among "
                         "(executes the candidate whose chunk is closest to the predictor output)")
    ap.add_argument("--consensus-encoder", default="",
                    help="retrieval CONSENSUS: path to an R3M action-metric encoder ckpt. When set, a skip "
                         "executes the cache entry in BOTH the N1 top-K and the encoder top-K (nearest by N1), "
                         "falling back to N1 top-1 if the intersection is empty. Default off = plain N1 retrieval.")
    ap.add_argument("--consensus-k", type=int, default=50,
                    help="retrieval consensus: per-key top-K for the N1 and encoder short-lists (default 50)")
    ap.add_argument("--fused-encoder", default="",
                    help="retrieval FUSED key: path to a multimodal fused encoder ckpt (image+proprio+prev). "
                         "When set, the retrieval key is this encoder's embedding -- a learned REPLACEMENT for "
                         "the N1 (prev_state) key. Requires --state-source actual_next_proprio. Default off.")
    ap.add_argument("--state-source", default="actual_next_proprio",
                    help="retrieval: state feature source. actual_next_proprio = the real, locally-sensed "
                         "self-state at a skip (deployable; matches all recent baselines). "
                         "vla_future_proprio = the VLA's cached +H-step prediction (legacy).")
    ap.add_argument("--metric", default="l2", choices=["l2", "norm"],
                    help="retrieval_oracle: NN distance (l2 = physical, norm = per-dim z-scored)")
    ap.add_argument("--cache-episodes", type=int, default=0,
                    help="retrieval/rmsegate: use the first N success episodes as the cache (0 = retrieval's "
                         "default train split; rmsegate defaults to 78). Pin both to the same N for a "
                         "consistent gate-vs-retrieval experiment.")
    ap.add_argument("--skip-policy", default="random",
                    choices=["random", "even", "gate", "gripevent", "rmsegate", "distgate", "disagree", "driftgate",
                             "oraclermse", "fusiongate", "envelope"],
                    help="random = fixed rate (stochastic); even = deterministic MAXIMALLY-EVEN pattern at the "
                         "--skip-rates rate (Bresenham/error-diffusion; skips never cluster -> minimal open-loop "
                         "drift, the anti-clustering counterpart of random); gate = N1-corroborated; gripevent = grip-event detector gate; "
                         "rmsegate = predicted N1-retrieval-error gate (k-NN over an RMSE dictionary); "
                         "distgate = out-of-support gate (skip iff nearest cache-key distance d1 < tau); "
                         "disagree = neighbour-disagreement gate (skip iff z(disagree)+w*z(grip) < tau); "
                         "driftgate = obs-drift gate (skip iff nearest cached-observation DINO distance < tau); "
                         "oraclermse = oracle upper bound: skip iff TRUE RMSE(N1 top-1, live VLA chunk) < tau "
                         "(VLA always run, no compute saved; requires --policy retrieval); "
                         "fusiongate = fusion N1-agreement VETO + random: skip iff the predictor's L1 pick is "
                         "within fusion's K state-nearest AND a coin (prob --skip-rates) lands, else CALL the "
                         "VLA (requires --policy fusion; K = --fusion-k); "
                         "envelope = learning-free success-envelope gate: skip iff the predictor chunk stays "
                         "within tau*sigma of the last executed action on every continuous dim/step AND the "
                         "gripper switch does not flip (sigma from demos; tau = q-quantile, --envelope-qs)")
    ap.add_argument("--gate-ks", default="50,150,400", help="N1 top-K thresholds to sweep (--skip-policy gate)")
    ap.add_argument("--fusiongate-ks", default="",
                    help="fusiongate: gate corroboration neighbourhood K(s), DECOUPLED from --fusion-k "
                         "(comma-separated ints, or 'auto' to offline-calibrate; empty = use --fusion-k). Larger "
                         "K -> more gate-pass / skip-eligible. Swept x --skip-rates; tags = fgateK{k}P{sr}.")
    ap.add_argument("--fusiongate-skip-ceiling", type=float, default=0.6,
                    help="fusiongate 'auto' calibration target: pick the SMALLEST gate_K whose held-out gate-pass "
                         "coverage >= this (= the max effective skip the gate then permits). Set > 0.5 to be able "
                         "to sweep --skip-rates past 50% skip (sweep skip-rates toward 1.0 to realise it).")
    ap.add_argument("--grip-ckpt", default="", help="trained grip-event model dir (--skip-policy gripevent)")
    ap.add_argument("--grip-qs", default="0.2,0.3,0.4,0.5,0.6,0.7,0.85",
                    help="gripevent: skip-rate quantiles to sweep (tau = q-quantile of train grip probs)")
    ap.add_argument("--rmse-qs", default="0.2,0.3,0.4,0.5,0.6",
                    help="rmsegate: skip-rate quantiles to sweep (tau = q-quantile of dict predicted errors)")
    ap.add_argument("--dist-qs", default="0.2,0.3,0.4,0.5,0.6,0.8,1.0",
                    help="distgate: quantiles to sweep (tau = q-quantile of held-out d1; q ~= in-support skip rate)")
    ap.add_argument("--disagree-qs", default="0.2,0.3,0.4,0.5,0.6,0.8,1.0",
                    help="disagree: quantiles to sweep (tau = q-quantile of held-out gate; q ~= in-distribution skip rate)")
    ap.add_argument("--envelope-qs", default="0.1,0.2,0.3,0.5",
                    help="envelope: quantiles to sweep (tau = q-quantile of held-out envelope score; q ~= in-distribution "
                         "skip fraction). Needs --data-dir (demo episodes for sigma + calibration).")
    ap.add_argument("--drift-qs", default="0.3,0.5,0.7,0.85,1.0",
                    help="driftgate: quantiles to sweep (tau = q-quantile of held-out obs-drift distance)")
    ap.add_argument("--oracle-rmse-taus", default="0.5,0.75,1.0,1.25,1.5",
                    help="oraclermse: ABSOLUTE per-dim z-scored RMSE thresholds to sweep (all 7 dims, gripper "
                         "included; skip iff RMSE(N1 top-1, live VLA chunk) < tau). Smaller tau -> fewer skips. "
                         "eff_skip is measured, not assumed.")
    ap.add_argument("--drift-img-dir", default="research/data/pnp_counter_to_stove_dense_img",
                    help="driftgate: episodes WITH images for the observation cache (default = the dense_img twin "
                         "of --data-dir; the retrieval/value path is unchanged, only the skip gate uses images)")
    ap.add_argument("--drift-views", default="primary,wrist",
                    help="driftgate: camera views forming the observation embedding (subset of primary,wrist)")
    ap.add_argument("--disagree-k", type=int, default=10,
                    help="disagree: K nearest cache chunks whose mean defines the neighbour-disagreement signal")
    ap.add_argument("--disagree-w-grip", type=float, default=1.0,
                    help="disagree: weight w on the z(grip) term (gate = z(disagree)+w*z(grip); 0 = pure disagree)")
    ap.add_argument("--rmse-k", type=int, default=10, help="rmsegate: k for the k-NN RMSE-dictionary lookup")
    ap.add_argument("--rmse-dict-dir", default="",
                    help="rmsegate: data dir for the RMSE dictionary (default = --data-dir). Use a dir with "
                         "disjoint scenes (e.g. newly-collected episodes) to enlarge the dict.")
    ap.add_argument("--rmse-dict-start", type=int, default=-1,
                    help="rmsegate: keep only dict episodes with index >= this (-1 = off). E.g. 150 = use only "
                         "the new ep150+ scenes, disjoint from a cache built on ep0-149.")
    ap.add_argument("--max-skips", type=int, default=-1,
                    help="consecutive-skip drift budget: force a VLA call after this many skips in a row "
                         "(<0 = unlimited/legacy; e.g. 2 = at most 2 skips before a forced cloud call). "
                         "Applies to any --skip-policy; tags get a '_b{n}' suffix.")
    ap.add_argument("--task", default="PnPCounterToStove")
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--num-episodes", type=int, default=12)
    ap.add_argument("--skip-rates", default="0,0.2,0.3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--skip-seed", type=int, default=0)
    ap.add_argument("--denoising-steps", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = build_cfg(args.task, args.seed, args.episode_start + args.num_episodes, args.denoising_steps)
    set_seed_everywhere(cfg.seed)
    validate_config(cfg)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    cosmos, _ = get_model(cfg)
    predictor = build_policy(args)
    print(f"policy={args.policy} src={predictor.run_dir} | img_mode={predictor.img_mode} "
          f"state_source={predictor.state_source}", flush=True)

    settings = build_settings(args, predictor)
    print(f"skip_policy={args.skip_policy} | settings={[t for _, t, _ in settings]}", flush=True)
    results = [eval_setting(cfg, cosmos, dataset_stats, predictor, pol, tag, extra, args.task,
                            args.episode_start, args.num_episodes, args.out,
                            max_skips=resolve_max_skips(args)) for pol, tag, extra in settings]
    report = {"run": predictor.run_dir, "policy": args.policy, "skip_policy": args.skip_policy,
              "task": args.task, "img_mode": predictor.img_mode, "state_source": predictor.state_source,
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
