"""Dense ("shadow query") data collection.

Same rollout as collect_data.py — execution still happens in 16-step open-loop blocks
driven by REAL Cosmos calls, so the executed trajectory (and success) is IDENTICAL to
the non-dense run for a given seed. ADDITIONALLY, every `--query-stride` steps we issue a
*record-only* (shadow) Cosmos query and log its features. Shadow queries never affect
execution; they only densify the (features -> next-16-actions) training pairs.

For a query recorded at timestep tau, a training sample uses decision point d = tau + 16:
    prev_actions = realized_actions[tau   : tau+16]   # the 16 actually-executed actions
    vla_state    = future_proprio_norm(tau)           # VLA-predicted self-state at tau
    future_img   = future_img_latent(tau)             # Cosmos future-image latent at tau
    target       = realized_actions[tau+16: tau+32]    # actually-executed next 16
    baseline     = chunk(tau)[16:32]                   # Cosmos own remaining 16

At stride=16 this reduces EXACTLY to the non-dense dataset (verified separately).

Run:
    CUDA_VISIBLE_DEVICES=0 HF_TOKEN=... \
    uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/collect_data_dense.py \
        --task PnPCounterToStove --episode-start 0 --num-episodes 50 \
        --query-stride 4 --out-dir research/data/pnp_counter_to_stove_dense
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    TASK_MAX_STEPS,
    create_robocasa_env,
    prepare_observation,
    validate_config,
)
from cosmos_policy.utils.utils import set_seed_everywhere

from collect_data import CKPT, NUM_STEPS_WAIT, build_cfg  # reuse verified config
from common import ACTION_DIM, NUM_OPEN_LOOP_STEPS, PROPRIO_DIM, extract_vec_from_latent_frame


def _u8(img):
    """Raw camera frame -> contiguous uint8 (H,W,3); None stays None if a camera is absent."""
    return None if img is None else np.ascontiguousarray(img).astype(np.uint8)


def _record_from_ret(ret):
    """Extract (chunk, future_proprio, future_img) from a get_action return dict."""
    chunk = np.asarray(ret["actions"], dtype=np.float32)  # (32,7)
    gl = ret["generated_latent"].detach().to(torch.float32).cpu().numpy()[0]  # (C,T,H,W)
    li = ret["latent_indices"]
    fut_proprio = extract_vec_from_latent_frame(gl[:, li["future_proprio_latent_idx"]], PROPRIO_DIM)
    view_idx = [li["future_wrist_image_latent_idx"], li["future_image_latent_idx"], li["future_image2_latent_idx"]]
    fut_img = gl[:, view_idx].transpose(1, 0, 2, 3)  # (3,16,28,28)
    return chunk, fut_proprio.astype(np.float32), fut_img.astype(np.float16)


def collect_episode(cfg, model, dataset_stats, task, episode_idx, stride, no_vla_shadow=False):
    seed = cfg.seed * episode_idx * 256 if cfg.deterministic else None
    env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
    env.reset()
    lang = env.get_ep_meta()["lang"]
    max_steps = TASK_MAX_STEPS.get(task, 500)

    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))

    realized_actions = []  # executed 7-dim action per step
    queries = []  # list of (t, chunk, future_proprio, future_img)
    action_queue: deque = deque()
    success = False
    zeros_tmpl = None  # (chunk, future_proprio, future_img) zero templates when --no-vla-shadow
    for t in range(max_steps):
        observation = prepare_observation(obs, cfg.flip_images)
        ret = None
        if len(action_queue) == 0:  # REAL call drives execution (every 16 steps)
            ret = get_action(
                cfg, model, dataset_stats, observation, lang,
                seed=cfg.seed, randomize_seed=False,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=False,
            )
            real_chunk = np.asarray(ret["actions"], dtype=np.float32)
            action_queue.extend([real_chunk[i] for i in range(cfg.num_open_loop_steps)])
            if no_vla_shadow and zeros_tmpl is None:  # capture zero templates from the real call (no extra VLA)
                _c, _fp, _fi = _record_from_ret(ret)
                zeros_tmpl = (np.zeros_like(_c), np.zeros_like(_fp), np.zeros_like(_fi))
        # SHADOW query for recording at the dense cadence (reuse real call at block boundaries)
        if t % stride == 0:
            if no_vla_shadow:
                chunk, fp, fimg = zeros_tmpl  # no extra VLA call; zeroed prediction fields (proprio+images only)
            else:
                if ret is None:
                    ret = get_action(
                        cfg, model, dataset_stats, observation, lang,
                        seed=cfg.seed, randomize_seed=False,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=False,
                    )
                chunk, fp, fimg = _record_from_ret(ret)
            queries.append(dict(t=t, chunk=chunk, cur_proprio=observation["proprio"].astype(np.float32),
                                future_proprio_norm=fp, future_img_latent=fimg,
                                # raw current-frame RGB (no VLA needed at deploy time) for observation keys
                                cur_image=_u8(observation["primary_image"]),
                                cur_wrist_image=_u8(observation["wrist_image"])))

        action = action_queue.popleft()
        realized_actions.append(np.asarray(action, dtype=np.float32))  # 7-dim
        if action.shape[-1] == ACTION_DIM and env.action_dim == 12:
            action = np.concatenate([action, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
        obs, _, _, _ = env.step(action)
        if env._check_success():
            success = True
            break
    env.close()
    return success, len(realized_actions), lang, np.stack(realized_actions), queries


def save_episode(out_dir, task, episode_idx, success, length, lang, realized, queries, stride):
    os.makedirs(out_dir, exist_ok=True)
    # Drop trailing queries that cannot form a full target (need tau+32 <= length).
    queries = [q for q in queries if q["t"] + NUM_OPEN_LOOP_STEPS * 2 <= length]
    if len(queries) == 0:
        return 0
    extra = {}  # raw RGB frames (saved only if the cameras were available)
    if queries[0].get("cur_image") is not None:
        extra["cur_image"] = np.stack([q["cur_image"] for q in queries])  # (Q,H,W,3) uint8
    if queries[0].get("cur_wrist_image") is not None:
        extra["cur_wrist_image"] = np.stack([q["cur_wrist_image"] for q in queries])  # (Q,H,W,3) uint8
    np.savez_compressed(
        os.path.join(out_dir, f"ep{episode_idx:04d}_success={int(success)}.npz"),
        task=task, episode_idx=episode_idx, success=success, length=length, lang=lang, stride=stride,
        realized_actions=realized.astype(np.float32),  # (T,7)
        query_t=np.array([q["t"] for q in queries], dtype=np.int32),
        chunk=np.stack([q["chunk"] for q in queries]),  # (Q,32,7)
        cur_proprio=np.stack([q["cur_proprio"] for q in queries]),  # (Q,9)
        future_proprio_norm=np.stack([q["future_proprio_norm"] for q in queries]),  # (Q,9)
        future_img_latent=np.stack([q["future_img_latent"] for q in queries]),  # (Q,3,16,28,28) fp16
        **extra,
    )
    return len(queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCounterToStove")
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--query-stride", type=int, default=4, help="record a (shadow) query every K steps; must divide 16")
    ap.add_argument("--out-dir", default="research/data/pnp_counter_to_stove_dense")
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--denoising-steps", type=int, default=5)
    ap.add_argument("--no-vla-shadow", action="store_true",
                    help="skip the extra shadow VLA calls; store ZEROED chunk/future_* (keep proprio+images+"
                         "realized actions only). Execution still uses the real VLA every 16 steps.")
    args = ap.parse_args()
    assert 16 % args.query_stride == 0, "query-stride must divide 16 so shadow records align with real calls"

    cfg = build_cfg(args.task, args.seed, args.episode_start + args.num_episodes, args.denoising_steps)
    set_seed_everywhere(cfg.seed)
    validate_config(cfg)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(cfg)

    n_success = 0
    for ep in range(args.episode_start, args.episode_start + args.num_episodes):
        t0 = time.time()
        success, length, lang, realized, queries = collect_episode(
            cfg, model, dataset_stats, args.task, ep, args.query_stride, args.no_vla_shadow)
        nq = save_episode(args.out_dir, args.task, ep, success, length, lang, realized, queries, args.query_stride)
        n_success += int(success)
        print(f"[ep {ep}] success={success} len={length} queries={nq} lang='{lang}' ({time.time()-t0:.1f}s)", flush=True)
    print(f"DONE: {n_success}/{args.num_episodes} successful. stride={args.query_stride}. Saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
