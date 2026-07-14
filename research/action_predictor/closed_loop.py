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

from common import NUM_OPEN_LOOP_STEPS
from sim import get_backend


def _cosmos_chunk(cfg, model, dataset_stats, obs, lang, backend):
    """Run Cosmos once; return (chunk, future_proprio[9], future_img[3,16,28,28])."""
    ret = get_action(
        cfg, model, dataset_stats, backend.prepare_obs(obs, cfg), lang,
        seed=cfg.seed, randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,
    )
    chunk = np.asarray(ret["actions"], dtype=np.float32)
    fp, fimg = backend.extract_cloud_outputs(ret)
    return chunk, fp, fimg


def run_closed_loop_episode(cfg, cosmos_model, dataset_stats, predictor, skip_policy, task,
                            episode_idx, collect_dagger=False, deterministic_reset=True, video_dir=None,
                            max_skips=None, backend=None, local_replan_steps=None, ensemble_m=None,
                            record_obs_stride=None):
    """Run one closed-loop episode. Returns a dict with success/length/counts and (if
    collect_dagger) `dagger` = list of {prev, cur_proprio, cached_fp, cached_fimg, target,
    cosmos_remaining} recorded at skip decision points.

    deterministic_reset: reseed the global RNG from `episode_idx` right before env.reset() so
      the scene/object placement is FULLY fixed by episode_idx (the env reset otherwise depends
      on the global RNG state; verified). This makes train/eval episode sets reproducible/disjoint.
    video_dir: if set, save an mp4 of the rollout for this episode there.
    record_obs_stride: if set (e.g. 4), record the live observation (proprio + primary/wrist RGB)
      every this many env steps -- the SAME per-stride obs cadence the dense demo collectors use --
      and return `obs_records` (list of {t, proprio, primary, wrist}), `realized_actions` (T,7)
      and `exec_src` (T,) uint8 (per executed step: 0 = VLA chunk, 1 = local/retrieved chunk) in
      the result, so a rollout can be saved as a demo-schema episode (rollout_cache_collect.py).
      None (default) = no recording, result keys absent, behaviour unchanged.
    """
    H = NUM_OPEN_LOOP_STEPS
    # LOCAL replan period: within an H-step SKIP block the local predictor re-queries every k_local steps
    # using fresh obs, while the VLA cadence + the skip schedule (gate is consulted once per H block) stay
    # on the H grid. Default (None/0/H) reproduces the legacy single-retrieval-per-block behaviour exactly.
    k_local = local_replan_steps or H
    assert 1 <= k_local <= H, f"local_replan_steps must be in [1, {H}], got {local_replan_steps}"
    required_local = getattr(predictor, "required_local_replan_steps", None)
    if required_local is not None and k_local != int(required_local):
        raise ValueError(
            f"{type(predictor).__name__} requires local_replan_steps={required_local}, got {k_local}"
        )
    if k_local < H:
        assert not getattr(predictor, "oracle_query", False), \
            "local_replan_steps<H is unsupported for the oracle_query predictor (it runs the VLA per skip)"
        if getattr(predictor, "needs_prev_frame", False):
            assert H % k_local == 0, (
                f"needs_prev_frame local replanning requires a period dividing H={H}, got {k_local}"
            )
    backend = backend or get_backend("robocasa")
    env, lang, max_steps = backend.make_env(cfg, episode_idx, reseed_before_reset=deterministic_reset)

    obs = None
    for _ in range(backend.NUM_STEPS_WAIT):
        obs, _, _, _ = backend.step(env, backend.dummy_action(env, cfg))

    realized, queue = [], deque()
    src_queue = deque()  # parallel to `queue`: 0 = VLA-issued action, 1 = locally-predicted/retrieved
    exec_src = []        # per executed step (parallel to `realized`)
    obs_records = []     # (record_obs_stride) per-stride {t, proprio, primary, wrist}
    cached_fp, cached_fimg = None, None
    success, n_call, n_skip, decision_idx = False, 0, 0, 0
    consec = 0  # consecutive skips so far; drift budget forces a VLA call once it reaches max_skips
    local_left = 0   # steps left in the current local-skip block to sub-replan (0 = not mid local block)
    n_local = 0      # count of finer local re-queries (only when k_local < H; 0 in the legacy path)
    dagger = []
    trace = []  # per-decision skip log (gate-agnostic): where + which gate + skip/call + gate's score/threshold
    hits = []   # (DUMP_HIT_IMAGES) per-skip {step, live_image, src_ep, src_imgidx, ...} for cache-hit viz
    rp, rs, rw = [], [], []  # replay images for video (when video_dir)
    skip_policy.reset()
    if hasattr(predictor, "reset"):   # stateful predictors (e.g. tracked retrieval): clear per-episode state
        predictor.reset()             # (no existing predictor defines reset -> no behaviour change)
    ens = None   # temporal-ensembling hook: off in the standard eval, so the guarded ens.* paths below are inert

    # obs frames of the PREVIOUS gate decision (16 executed steps ago at every gate decision): passed
    # ONLY to policies that declare ``needs_prev_frame`` (e.g. the TMT transition encoder); all existing
    # policies never see them -> no behaviour change. None at the first decision (policy falls back).
    prev_dec_img = prev_dec_wri = None
    fbuf = {}   # step -> (primary, wrist) at the local cadence for needs_prev_frame policies
    for t in range(max_steps):
        prepared_ob = None
        if video_dir is not None:
            vob = backend.prepare_obs(obs, cfg)
            rp.append(vob["primary_image"]); rs.append(vob.get("secondary_image")); rw.append(vob["wrist_image"])
        if record_obs_stride and t % record_obs_stride == 0:  # obs AT step t, before executing action t
            rob = backend.prepare_obs(obs, cfg)
            obs_records.append(dict(
                t=t, proprio=rob["proprio"].astype(np.float32),
                primary=(None if rob.get("primary_image") is None
                         else np.ascontiguousarray(rob["primary_image"]).astype(np.uint8)),
                wrist=(None if rob.get("wrist_image") is None
                       else np.ascontiguousarray(rob["wrist_image"]).astype(np.uint8))))
        # A local query at t needs the real frame from t-H. Capture the cadence even while a VLA block
        # owns the action queue; otherwise the first skipped block after a VLA call has holes in fbuf.
        if k_local < H and getattr(predictor, "needs_prev_frame", False) and t % k_local == 0:
            fob = prepared_ob = backend.prepare_obs(obs, cfg)
            if fob.get("primary_image") is not None:
                fbuf[t] = (np.asarray(fob["primary_image"]).copy(),
                           np.asarray(fob["wrist_image"]).copy())
                fbuf.pop(t - 2 * H, None)
        if len(queue) == 0 and local_left > 0:  # mid local-skip block: re-query LOCAL (no gate, no VLA)
            ob = prepared_ob if prepared_ob is not None else backend.prepare_obs(obs, cfg)
            cur_proprio = ob["proprio"].astype(np.float32)
            prev = np.stack(realized[-H:])
            _pf = {}
            if getattr(predictor, "needs_prev_frame", False):
                pi, pw = fbuf.get(t - H, (None, None))
                _pf = {"prev_image": pi, "prev_wrist": pw}
                if ob.get("primary_image") is not None:
                    fbuf[t] = (np.asarray(ob["primary_image"]).copy(), np.asarray(ob["wrist_image"]).copy())
                    fbuf.pop(t - 2 * H, None)
            pred = predictor.predict_chunk(prev, cur_proprio, cached_fp, cached_fimg,
                                           current_image=ob.get("primary_image"),
                                           current_wrist=ob.get("wrist_image"), **_pf)  # (16,7); execute first n
            n = min(k_local, local_left); queue.extend(pred[i] for i in range(n)); local_left -= n
            src_queue.extend([1] * n)
            n_local += 1
            if ens is not None:
                ens.add(t, pred)  # (16,7) local re-query chunk issued at step t
        if len(queue) == 0 and local_left == 0:  # gate decision point (fresh H-step block)
            ob = prepared_ob if prepared_ob is not None else backend.prepare_obs(obs, cfg)
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
                vla = _cosmos_chunk(cfg, cosmos_model, dataset_stats, obs, lang, backend)
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
                    vla_chunk, _, _ = _cosmos_chunk(cfg, cosmos_model, dataset_stats, obs, lang, backend)
                    pred = predictor.lookup(vla_chunk[:H])  # (16,7)
                else:
                    _pf = {}
                    if getattr(predictor, "needs_prev_frame", False):
                        pi, pw = fbuf.get(t - H, (prev_dec_img, prev_dec_wri))
                        _pf = {"prev_image": pi, "prev_wrist": pw}
                    pred = predictor.predict_chunk(prev, cur_proprio, cached_fp, cached_fimg,
                                                   current_image=ctx["current_image"],
                                                   current_wrist=ctx["current_wrist"], **_pf)  # (16,7)
                if collect_dagger:  # shadow expert label at the visited state (cache untouched)
                    exp_chunk, _, _ = _cosmos_chunk(cfg, cosmos_model, dataset_stats, obs, lang, backend)
                    _dpi, _dpw = (fbuf.get(t - H, (prev_dec_img, prev_dec_wri))
                                  if getattr(predictor, "needs_prev_frame", False) else (None, None))
                    dagger.append(dict(
                        prev=prev.copy(), cur_proprio=cur_proprio.copy(),
                        cached_fp=cached_fp.copy(), cached_fimg=cached_fimg.copy(),
                        target=exp_chunk[:H].copy(), cosmos_remaining=exp_chunk[H : 2 * H].copy(),
                        # live decision-point frames: needed to train the RERANKER on-policy (it scores
                        # candidates from the observation). None-safe; existing collectors ignore extra keys.
                        current_image=(np.asarray(ctx["current_image"]).copy() if ctx["current_image"] is not None else None),
                        current_wrist=(np.asarray(ctx["current_wrist"]).copy() if ctx["current_wrist"] is not None else None),
                        # PREVIOUS decision-point frames (needs_prev_frame policies, e.g. TMT): lets an offline
                        # tool recompute the exact transition query embedding. None when unused.
                        prev_image=(np.asarray(_dpi).copy() if _dpi is not None else None),
                        prev_wrist=(np.asarray(_dpw).copy() if _dpw is not None else None),
                    ))
                m = getattr(predictor, "last_match", None)   # which cache frame was executed at this skip
                if isinstance(m, dict):
                    te.update({"hit_cache_idx": m.get("cache_idx"), "hit_src_ep": m.get("src_ep"),
                               "hit_src_imgidx": m.get("src_imgidx"), "hit_src_t": m.get("src_t"),
                               "hit_dist": m.get("dist"), "hit_mode": m.get("mode")})
                    if _DUMP_HITS and ctx["current_image"] is not None:
                        hits.append({"decision_idx": int(decision_idx), "step": int(t),
                                     "live_image": np.asarray(ctx["current_image"]).copy(), **m})
                n = min(k_local, H); queue.extend(pred[i] for i in range(n)); local_left = H - n
                src_queue.extend([1] * n)
                n_skip += 1; consec += 1
                if ens is not None:
                    ens.add(t, pred)  # (16,7) retrieved/predicted skip chunk issued at step t
            else:
                chunk, fp, fimg = vla if vla is not None else _cosmos_chunk(cfg, cosmos_model, dataset_stats, obs, lang, backend)
                cached_fp, cached_fimg = fp, fimg  # only REAL calls update the cache
                queue.extend(chunk[i] for i in range(H))
                src_queue.extend([0] * H)
                n_call += 1; consec = 0  # any VLA call resets the drift budget
                if hasattr(predictor, "notify_vla_call"):  # stateful predictors: the VLA block makes any
                    predictor.notify_vla_call()            # committed/tracked demo stale (no-op otherwise)
                if ens is not None:
                    ens.add(t, chunk[:H])  # (16,7) VLA chunk issued at step t (same buffer as skip chunks)
            if getattr(predictor, "needs_prev_frame", False):  # keep copies only when a policy consumes them
                prev_dec_img = (None if ctx["current_image"] is None else np.asarray(ctx["current_image"]).copy())
                prev_dec_wri = (None if ctx["current_wrist"] is None else np.asarray(ctx["current_wrist"]).copy())
                if prev_dec_img is not None:
                    fbuf[t] = (prev_dec_img, prev_dec_wri)
                    fbuf.pop(t - 2 * H, None)
            decision_idx += 1

        a = queue.popleft()
        exec_src.append(src_queue.popleft())
        if ens is not None:  # replace the committed action by the recency-weighted ensemble of overlapping chunks
            ea = ens.action(t)
            if ea is not None:
                a = ea
        realized.append(np.asarray(a, dtype=np.float32))
        obs, _, done, info = backend.step(env, np.asarray(a, dtype=np.float32))
        if backend.is_success(env, done, info):
            success = True
            break
    env.close()
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        backend.save_video(rp, rs, rw, episode_idx, success, lang, video_dir)
    n_dec = n_call + n_skip
    out = dict(success=success, length=len(realized), lang=lang, n_call=n_call, n_skip=n_skip,
               skip_rate=(n_skip / n_dec if n_dec else 0.0), dagger=dagger, trace=trace, hit_images=hits,
               n_local_replan=n_local, local_replan_steps=k_local,
               ensemble_m=(float(ensemble_m) if ens is not None else None))
    if record_obs_stride:  # rollout-cache collection: the executed trajectory + per-stride obs
        out["realized_actions"] = np.stack(realized).astype(np.float32)   # (T,7)
        out["exec_src"] = np.array(exec_src, dtype=np.uint8)              # (T,) 0=VLA, 1=local
        out["obs_records"] = obs_records
    return out
