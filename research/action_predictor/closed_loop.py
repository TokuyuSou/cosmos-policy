"""Closed-loop rollout: run a RoboCasa episode where, at each 16-step decision point, a
SkipPolicy chooses cloud (Cosmos) vs local (action predictor). Reuses the official eval
primitives so cloud behavior matches real Cosmos Policy. Does not modify existing files.

Cache model (deployment-faithful): only a REAL cloud call updates the cached
future-proprio / future-image latent. On a skip the predictor consumes those cached
cloud outputs plus locally-available prev-actions / current-proprio.

DAgger relabeling (optional): at a skip, also query the VLA at the visited state for the
EXPERT action chunk (shadow; does NOT update the cache or change execution). These
(on-policy inputs -> expert action) pairs train the predictor on its own state distribution.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch

# When set, each retrieval skip also buffers the live decision-point frame + the matched cache provenance,
# so an episode can be analyzed frame-by-frame (see cache_hit_viz.py). Off by default (no overhead in normal runs).
_DUMP_HITS = os.environ.get("DUMP_HIT_IMAGES") == "1"

from cosmos_policy.experiments.robot.cosmos_utils import get_action
from cosmos_policy.experiments.robot.robocasa.robocasa_utils import save_rollout_video
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS,
    create_robocasa_env,
    prepare_observation,
)
from cosmos_policy.utils.utils import set_seed_everywhere

from collect_data import NUM_STEPS_WAIT
from common import ACTION_DIM, NUM_OPEN_LOOP_STEPS, PROPRIO_DIM, extract_vec_from_latent_frame


def extract_cloud_outputs(ret):
    """From a get_action() return dict, extract (future_proprio[9], future_img[3,16,28,28])."""
    gl = ret["generated_latent"].detach().to(torch.float32).cpu().numpy()[0]  # (C,T,H,W)
    li = ret["latent_indices"]
    fp = extract_vec_from_latent_frame(gl[:, li["future_proprio_latent_idx"]], PROPRIO_DIM).astype(np.float32)
    vi = [li["future_wrist_image_latent_idx"], li["future_image_latent_idx"], li["future_image2_latent_idx"]]
    fimg = gl[:, vi].transpose(1, 0, 2, 3).astype(np.float32)  # (3,16,28,28)
    return fp, fimg


def _cosmos_chunk(cfg, model, dataset_stats, obs, lang):
    """Run Cosmos once; return (chunk[32,7], future_proprio[9], future_img[3,16,28,28])."""
    ret = get_action(
        cfg, model, dataset_stats, prepare_observation(obs, cfg.flip_images), lang,
        seed=cfg.seed, randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,
    )
    chunk = np.asarray(ret["actions"], dtype=np.float32)
    fp, fimg = extract_cloud_outputs(ret)
    return chunk, fp, fimg


def run_closed_loop_episode(cfg, cosmos_model, dataset_stats, predictor, skip_policy, task,
                            episode_idx, collect_dagger=False, deterministic_reset=True, video_dir=None,
                            max_skips=None):
    """Run one closed-loop episode. Returns a dict with success/length/counts and (if
    collect_dagger) `dagger` = list of {prev, cur_proprio, cached_fp, cached_fimg, target,
    cosmos_remaining} recorded at skip decision points.

    deterministic_reset: reseed the global RNG from `episode_idx` right before env.reset() so
      the scene/object placement is FULLY fixed by episode_idx (the env reset otherwise depends
      on the global RNG state; verified). This makes train/eval episode sets reproducible/disjoint.
    video_dir: if set, save an mp4 of the rollout for this episode there.
    """
    H = NUM_OPEN_LOOP_STEPS
    seed = cfg.seed * episode_idx * 256 if cfg.deterministic else None
    env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
    if deterministic_reset and seed is not None:
        set_seed_everywhere(seed)  # pin scene/object placement to episode_idx (independent of run/order)
    env.reset()
    lang = env.get_ep_meta()["lang"]
    max_steps = TASK_MAX_STEPS.get(task, 500)

    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))

    realized, queue = [], deque()
    cached_fp, cached_fimg = None, None
    success, n_call, n_skip, decision_idx = False, 0, 0, 0
    consec = 0  # consecutive skips so far; drift budget forces a VLA call once it reaches max_skips
    dagger = []
    trace = []  # per-decision skip log (gate-agnostic): where + which gate + skip/call + gate's score/threshold
    hits = []   # (DUMP_HIT_IMAGES) per-skip {step, live_image, src_ep, src_imgidx, ...} for cache-hit viz
    rp, rs, rw = [], [], []  # replay images for video (when video_dir)
    skip_policy.reset()

    for t in range(max_steps):
        if video_dir is not None:
            vob = prepare_observation(obs, cfg.flip_images)
            rp.append(vob["primary_image"]); rs.append(vob["secondary_image"]); rw.append(vob["wrist_image"])
        if len(queue) == 0:  # decision point
            ob = prepare_observation(obs, cfg.flip_images)
            cur_proprio = ob["proprio"].astype(np.float32)
            can_skip = (cached_fp is not None) and (len(realized) >= H)
            ctx = {"decision_idx": decision_idx, "step": t, "cached_future_proprio": cached_fp,
                   "cached_future_img": cached_fimg, "current_proprio": cur_proprio,
                   "current_image": ob.get("primary_image"), "current_wrist": ob.get("wrist_image"),
                   "prev_actions": np.stack(realized[-H:]) if len(realized) >= H else None}
            # drift budget: once we've skipped max_skips times in a row, force a VLA call (don't even
            # consult the gate) -- caps worst-case open-loop drift to max_skips*H executed steps.
            budget_block = bool(max_skips is not None and consec >= max_skips)
            # Oracle gates (needs_vla) compare the VLA's fresh chunk to the skip-time chunk, so the VLA
            # must run BEFORE the skip decision. Compute it once here and REUSE it on a CALL (no 2nd query);
            # on a skip it is discarded (cache untouched -- a skip never updates the cache, like every gate).
            vla = None  # (chunk[32,7], future_proprio[9], future_img[3,16,28,28]) when precomputed
            if can_skip and not budget_block and getattr(skip_policy, "needs_vla", False):
                vla = _cosmos_chunk(cfg, cosmos_model, dataset_stats, obs, lang)
                ctx["vla_chunk"] = vla[0]
            if can_skip and not budget_block:
                do_skip = bool(skip_policy.decide(ctx))
                _info = getattr(skip_policy, "last", None)
            else:
                do_skip, _info = False, None  # gate not consulted (no cache/history yet, or budget exhausted)
            # gate-agnostic skip log: where + which gate + skip/call + gate score, plus drift-budget state
            te = {"decision_idx": decision_idx, "step": t, "can_skip": bool(can_skip),
                  "skip": bool(do_skip), "gate": getattr(skip_policy, "name", "?"),
                  "consec_skips": int(consec), "max_skips": max_skips, "budget_block": budget_block,
                  **(_info if isinstance(_info, dict) else {})}
            trace.append(te)
            if do_skip:
                prev = np.stack(realized[-H:])  # (16,7)
                if getattr(predictor, "oracle_query", False):
                    # ORACLE: run the VLA only to FORM the query (cache untouched, no compute
                    # saved); execute the nearest cached action window to the VLA's own chunk.
                    vla_chunk, _, _ = _cosmos_chunk(cfg, cosmos_model, dataset_stats, obs, lang)
                    pred = predictor.lookup(vla_chunk[:H])  # (16,7)
                else:
                    pred = predictor.predict_chunk(prev, cur_proprio, cached_fp, cached_fimg,
                                                   current_image=ctx["current_image"],
                                                   current_wrist=ctx["current_wrist"])  # (16,7)
                if collect_dagger:  # shadow expert label at the visited state (cache untouched)
                    exp_chunk, _, _ = _cosmos_chunk(cfg, cosmos_model, dataset_stats, obs, lang)
                    dagger.append(dict(
                        prev=prev.copy(), cur_proprio=cur_proprio.copy(),
                        cached_fp=cached_fp.copy(), cached_fimg=cached_fimg.copy(),
                        target=exp_chunk[:H].copy(), cosmos_remaining=exp_chunk[H : 2 * H].copy(),
                    ))
                m = getattr(predictor, "last_match", None)   # which cache frame was executed at this skip
                if isinstance(m, dict):
                    te.update({"hit_cache_idx": m.get("cache_idx"), "hit_src_ep": m.get("src_ep"),
                               "hit_src_imgidx": m.get("src_imgidx"), "hit_src_t": m.get("src_t"),
                               "hit_dist": m.get("dist")})
                    if _DUMP_HITS and ctx["current_image"] is not None:
                        hits.append({"decision_idx": int(decision_idx), "step": int(t),
                                     "live_image": np.asarray(ctx["current_image"]).copy(), **m})
                queue.extend(pred[i] for i in range(H))
                n_skip += 1; consec += 1
            else:
                chunk, fp, fimg = vla if vla is not None else _cosmos_chunk(cfg, cosmos_model, dataset_stats, obs, lang)
                cached_fp, cached_fimg = fp, fimg  # only REAL calls update the cache
                queue.extend(chunk[i] for i in range(H))
                n_call += 1; consec = 0  # any VLA call resets the drift budget
            decision_idx += 1

        a = queue.popleft()
        realized.append(np.asarray(a, dtype=np.float32))
        if a.shape[-1] == ACTION_DIM and env.action_dim == 12:
            a = np.concatenate([a, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
        obs, _, _, _ = env.step(a)
        if env._check_success():
            success = True
            break
    env.close()
    if video_dir is not None:
        import os
        os.makedirs(video_dir, exist_ok=True)
        save_rollout_video(rp, rs, rw, episode_idx, success=success, task_description=lang,
                           rollout_data_dir=video_dir, log_file=None)
    n_dec = n_call + n_skip
    return dict(success=success, length=len(realized), lang=lang, n_call=n_call, n_skip=n_skip,
                skip_rate=(n_skip / n_dec if n_dec else 0.0), dagger=dagger, trace=trace, hit_images=hits)
