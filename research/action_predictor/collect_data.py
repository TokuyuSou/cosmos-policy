"""Collect Cosmos Policy rollouts on a RoboCasa task and log per-VLA-call signals
for training a lightweight action predictor.

Reuses the official eval primitives (env creation, observation prep, get_action) so
the rollouts match real Cosmos Policy behavior. Does NOT modify any existing file.

Run (one GPU, one episode range):
    CUDA_VISIBLE_DEVICES=0 HF_TOKEN=... \
    uv run --extra cu128 --group robocasa --python 3.10 \
        python research/action_predictor/collect_data.py \
        --task PnPCounterToStove --episode-start 0 --num-episodes 50 \
        --out-dir research/data/pnp_counter_to_stove
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
    PolicyEvalConfig,
    create_robocasa_env,
    prepare_observation,
    validate_config,
)
from cosmos_policy.utils.utils import set_seed_everywhere

from common import (
    ACTION_DIM,
    CHUNK_SIZE,
    NUM_OPEN_LOOP_STEPS,
    PROPRIO_DIM,
    extract_vec_from_latent_frame,
)

# Cosmos RoboCasa pretrained checkpoint (auto-downloaded / cached).
CKPT = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
NUM_STEPS_WAIT = 10  # env stabilization steps before the policy loop (matches run_episode)


def build_cfg(task: str, seed: int, num_episodes: int, denoising_steps: int) -> PolicyEvalConfig:
    """Mirror the ROBOCASA.md inference config (deterministic)."""
    return PolicyEvalConfig(
        suite="robocasa",
        config="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ckpt_path=CKPT,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{CKPT}/robocasa_dataset_statistics.json",
        t5_text_embeddings_path=f"{CKPT}/robocasa_t5_embeddings.pkl",
        use_third_person_image=True,
        num_third_person_images=2,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        chunk_size=CHUNK_SIZE,
        num_open_loop_steps=NUM_OPEN_LOOP_STEPS,
        num_denoising_steps_action=denoising_steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        deterministic=True,
        randomize_seed=False,
        use_variance_scale=False,
        task_name=task,
        num_trials_per_task=num_episodes,
        seed=seed,
    )


def collect_episode(cfg, model, dataset_stats, task: str, episode_idx: int):
    """Run one rollout; return (success, length, lang, list-of-per-call-dicts)."""
    # Deterministic env seed: same formula as run_task in run_robocasa_eval.py.
    seed = cfg.seed * episode_idx * 256 if cfg.deterministic else None
    env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
    env.reset()
    task_description = env.get_ep_meta()["lang"]
    max_steps = TASK_MAX_STEPS.get(task, 500)

    # Stabilize the scene with dummy zero actions (objects settle) before querying.
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))

    calls = []
    action_queue: deque = deque()
    success = False
    length = 0
    for t in range(max_steps):
        observation = prepare_observation(obs, cfg.flip_images)
        if len(action_queue) == 0:
            ret = get_action(
                cfg,
                model,
                dataset_stats,
                observation,
                task_description,
                seed=cfg.seed,
                randomize_seed=False,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=False,  # we read latents directly
            )
            chunk = np.asarray(ret["actions"], dtype=np.float32)  # (32, 7), physical scale
            gl = ret["generated_latent"].detach().to(torch.float32).cpu().numpy()[0]  # (C,T,H,W)
            li = ret["latent_indices"]
            fut_proprio = extract_vec_from_latent_frame(
                gl[:, li["future_proprio_latent_idx"]], PROPRIO_DIM
            )  # (9,) normalized
            view_idx = [
                li["future_wrist_image_latent_idx"],
                li["future_image_latent_idx"],
                li["future_image2_latent_idx"],
            ]
            fut_img = gl[:, view_idx].transpose(1, 0, 2, 3)  # (3, C=16, H=28, W=28)
            calls.append(
                dict(
                    call_t=t,
                    chunk=chunk,
                    cur_proprio=observation["proprio"].astype(np.float32),
                    future_proprio_norm=fut_proprio.astype(np.float32),
                    future_img_latent=fut_img.astype(np.float16),
                )
            )
            action_queue.extend([chunk[i] for i in range(cfg.num_open_loop_steps)])

        action = action_queue.popleft()
        if action.shape[-1] == ACTION_DIM and env.action_dim == 12:
            action = np.concatenate([action, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
        obs, _, _, _ = env.step(action)
        length += 1
        if env._check_success():
            success = True
            break
    env.close()
    return success, length, task_description, calls


def save_episode(out_dir: str, task: str, episode_idx: int, success: bool, length: int, lang: str, calls):
    os.makedirs(out_dir, exist_ok=True)
    if len(calls) == 0:
        return None
    np.savez_compressed(
        os.path.join(out_dir, f"ep{episode_idx:04d}_success={int(success)}.npz"),
        task=task,
        episode_idx=episode_idx,
        success=success,
        length=length,
        lang=lang,
        call_t=np.array([c["call_t"] for c in calls], dtype=np.int32),
        chunk=np.stack([c["chunk"] for c in calls]),  # (n,32,7)
        cur_proprio=np.stack([c["cur_proprio"] for c in calls]),  # (n,9)
        future_proprio_norm=np.stack([c["future_proprio_norm"] for c in calls]),  # (n,9)
        future_img_latent=np.stack([c["future_img_latent"] for c in calls]),  # (n,3,16,28,28) fp16
    )
    return len(calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PnPCounterToStove")
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--out-dir", default="research/data/pnp_counter_to_stove")
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--denoising-steps", type=int, default=5)
    args = ap.parse_args()

    cfg = build_cfg(args.task, args.seed, args.episode_start + args.num_episodes, args.denoising_steps)
    set_seed_everywhere(cfg.seed)
    validate_config(cfg)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(cfg)

    n_success = 0
    for episode_idx in range(args.episode_start, args.episode_start + args.num_episodes):
        t0 = time.time()
        success, length, lang, calls = collect_episode(cfg, model, dataset_stats, args.task, episode_idx)
        n_calls = save_episode(args.out_dir, args.task, episode_idx, success, length, lang, calls)
        n_success += int(success)
        print(
            f"[ep {episode_idx}] success={success} len={length} calls={n_calls} "
            f"lang='{lang}' ({time.time() - t0:.1f}s)",
            flush=True,
        )
    print(f"DONE: {n_success}/{args.num_episodes} successful. Saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
