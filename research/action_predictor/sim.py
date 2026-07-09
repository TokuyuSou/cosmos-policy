"""Simulator backend: the ONLY place that knows RoboCasa- vs LIBERO-specific glue, so the rest of the
pipeline (encoder, retrieval, dataset, skip policies, plotting, cache-hit viz) stays simulator-agnostic.

Select one with ``get_backend("robocasa"|"libero")``. Every pipeline entry point takes ``--sim`` (default
``robocasa`` -> existing behaviour is byte-for-byte unchanged). Both backends expose ONE interface:

    CKPT, NUM_STEPS_WAIT
    build_cfg(task, seed, n_episodes, denoising) -> cfg
    validate_cfg(cfg)
    make_env(cfg, episode_idx, reseed_before_reset=False) -> (env, lang, max_steps)
    dummy_action(env, cfg)                              # action for the NUM_STEPS_WAIT warm-up
    prepare_obs(obs, cfg) -> {primary_image, wrist_image, proprio[, secondary_image]}
    step(env, action) -> (obs, reward, done, info)      # handles any robot-specific action padding
    is_success(env, done, info) -> bool
    extract_cloud_outputs(ret) -> (future_proprio[9], future_img[3,16,28,28])   # VLA latents (unused by fused)
    save_video(primary, secondary, wrist, idx, success, lang, out_dir)

Heavy sim imports are LAZY (inside methods) so importing this module -- or selecting one backend --
never forces loading the other simulator's dependencies (robocasa vs libero are different uv groups).
"""
from __future__ import annotations

import numpy as np

from common import ACTION_DIM, CHUNK_SIZE, NUM_OPEN_LOOP_STEPS, PROPRIO_DIM, extract_vec_from_latent_frame

NUM_STEPS_WAIT = 10  # env-stabilization steps before the policy loop (matches both run_episode loops)
IMG_RES = 224        # stored decision-point frame size (R3M input); robocasa renders this natively


class RobocasaBackend:
    """Wraps the official RoboCasa eval primitives verbatim -> behaviour identical to the pre-refactor code."""
    name = "robocasa"
    NUM_STEPS_WAIT = NUM_STEPS_WAIT
    CKPT = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"

    def build_cfg(self, task, seed, num_episodes, denoising_steps):
        from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import PolicyEvalConfig
        return PolicyEvalConfig(
            suite="robocasa", config="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
            ckpt_path=self.CKPT, config_file="cosmos_policy/config/config.py",
            dataset_stats_path=f"{self.CKPT}/robocasa_dataset_statistics.json",
            t5_text_embeddings_path=f"{self.CKPT}/robocasa_t5_embeddings.pkl",
            use_third_person_image=True, num_third_person_images=2, use_wrist_image=True, num_wrist_images=1,
            use_proprio=True, normalize_proprio=True, unnormalize_actions=True, trained_with_image_aug=True,
            use_jpeg_compression=True, flip_images=True, chunk_size=CHUNK_SIZE,
            num_open_loop_steps=NUM_OPEN_LOOP_STEPS, num_denoising_steps_action=denoising_steps,
            num_denoising_steps_future_state=1, num_denoising_steps_value=1, deterministic=True,
            randomize_seed=False, use_variance_scale=False, task_name=task,
            num_trials_per_task=num_episodes, seed=seed,
        )

    def validate_cfg(self, cfg):
        from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import validate_config
        validate_config(cfg)

    def make_env(self, cfg, episode_idx, reseed_before_reset=False):
        from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import TASK_MAX_STEPS, create_robocasa_env
        from cosmos_policy.utils.utils import set_seed_everywhere
        seed = cfg.seed * episode_idx * 256 if cfg.deterministic else None
        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
        if reseed_before_reset and seed is not None:
            set_seed_everywhere(seed)  # pin scene/object placement to episode_idx (independent of run/order)
        env.reset()
        return env, env.get_ep_meta()["lang"], TASK_MAX_STEPS.get(cfg.task_name, 500)

    def dummy_action(self, env, cfg):
        return np.zeros(env.action_spec[0].shape)

    def prepare_obs(self, obs, cfg):
        from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import prepare_observation
        return prepare_observation(obs, cfg.flip_images)

    def step(self, env, action):
        if action.shape[-1] == ACTION_DIM and env.action_dim == 12:   # robocasa mobile base: pad to 12
            action = np.concatenate([action, np.array([0.0, 0.0, 0.0, 0.0, -1.0])])
        return env.step(action)

    def is_success(self, env, done, info):
        return bool(env._check_success())

    def extract_cloud_outputs(self, ret):
        import torch
        gl = ret["generated_latent"].detach().to(torch.float32).cpu().numpy()[0]  # (C,T,H,W)
        li = ret["latent_indices"]
        fp = extract_vec_from_latent_frame(gl[:, li["future_proprio_latent_idx"]], PROPRIO_DIM).astype(np.float32)
        vi = [li["future_wrist_image_latent_idx"], li["future_image_latent_idx"], li["future_image2_latent_idx"]]
        fimg = gl[:, vi].transpose(1, 0, 2, 3).astype(np.float32)  # (3,16,28,28)
        return fp, fimg

    def save_video(self, primary, secondary, wrist, idx, success, lang, out_dir):
        from cosmos_policy.experiments.robot.robocasa.robocasa_utils import save_rollout_video
        save_rollout_video(primary, secondary, wrist, idx, success=success,
                           task_description=lang, rollout_data_dir=out_dir, log_file=None)


class LiberoBackend:
    """LIBERO equivalent. ``task`` is "<suite>:<task_id>" (e.g. "libero_spatial:0"); an episode index is an
    init-state index within that task. The fused flow never uses the VLA's future-image latent, so
    ``extract_cloud_outputs`` returns robocasa-shaped outputs (real future-proprio, zeroed future-image),
    keeping the saved-npz schema identical."""
    name = "libero"
    NUM_STEPS_WAIT = NUM_STEPS_WAIT
    CKPT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"

    def __init__(self):
        self.suite, self.task_id, self._suite_obj, self._init_states = None, None, None, None

    @staticmethod
    def _parse(task):
        suite, _, tid = task.partition(":")
        return suite, int(tid or 0)

    def build_cfg(self, task, seed, num_episodes, denoising_steps):
        from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
        self.suite, self.task_id = self._parse(task)
        return PolicyEvalConfig(
            suite="libero", model_family="cosmos", config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path=self.CKPT, config_file="cosmos_policy/config/config.py",
            dataset_stats_path=f"{self.CKPT}/libero_dataset_statistics.json",
            t5_text_embeddings_path=f"{self.CKPT}/libero_t5_embeddings.pkl",
            use_third_person_image=True, num_third_person_images=1, use_wrist_image=True, num_wrist_images=1,
            use_proprio=True, normalize_proprio=True, unnormalize_actions=True, trained_with_image_aug=True,
            use_jpeg_compression=True, flip_images=True, chunk_size=NUM_OPEN_LOOP_STEPS,
            num_open_loop_steps=NUM_OPEN_LOOP_STEPS, num_denoising_steps_action=denoising_steps,
            num_denoising_steps_future_state=1, num_denoising_steps_value=1, deterministic=True,
            task_suite_name=self.suite, num_trials_per_task=num_episodes, seed=seed,
        )

    def validate_cfg(self, cfg):
        from cosmos_policy.experiments.robot.libero.run_libero_eval import validate_config
        validate_config(cfg)

    def _task_suite(self):
        if self._suite_obj is None:
            from libero.libero import benchmark
            self._suite_obj = benchmark.get_benchmark_dict()[self.suite]()
            self._init_states = self._suite_obj.get_task_init_states(self.task_id)
        return self._suite_obj

    def make_env(self, cfg, episode_idx, reseed_before_reset=False):
        from cosmos_policy.experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS
        from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
        ts = self._task_suite()
        env, lang = get_libero_env(ts.get_task(self.task_id), cfg.model_family, resolution=cfg.env_img_res)
        env.reset()
        env.set_init_state(self._init_states[episode_idx % len(self._init_states)])  # init-state = episode index
        return env, lang, TASK_MAX_STEPS[cfg.task_suite_name]

    def dummy_action(self, env, cfg):
        from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action
        return np.asarray(get_libero_dummy_action(cfg.model_family), dtype=np.float32)

    def prepare_obs(self, obs, cfg):
        import cv2
        from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation
        o = prepare_observation(obs, IMG_RES, cfg.flip_images)  # {primary_image, wrist_image, proprio}
        for k in ("primary_image", "wrist_image"):  # render is 256px -> resize to the R3M/store resolution
            o[k] = cv2.resize(np.ascontiguousarray(o[k]), (IMG_RES, IMG_RES), interpolation=cv2.INTER_AREA)
        return o

    def step(self, env, action):
        return env.step(np.asarray(action).tolist())  # libero env wants a python list; action_dim is 7 (no pad)

    def is_success(self, env, done, info):
        return bool(done)

    def extract_cloud_outputs(self, ret):
        import torch
        gl = ret["generated_latent"].detach().to(torch.float32).cpu().numpy()[0]
        li = ret["latent_indices"]
        fp = (extract_vec_from_latent_frame(gl[:, li["future_proprio_latent_idx"]], PROPRIO_DIM).astype(np.float32)
              if "future_proprio_latent_idx" in li else np.zeros(PROPRIO_DIM, np.float32))
        fimg = np.zeros((3, NUM_OPEN_LOOP_STEPS, 28, 28), np.float32)  # unused by fused; keep robocasa shape
        return fp, fimg

    def save_video(self, primary, secondary, wrist, idx, success, lang, out_dir):
        # The libero helper (libero_utils.save_rollout_video) hardcodes ./rollouts/<date>/ at the REPO ROOT
        # and takes no output dir. Write the mp4 under the eval's out_dir instead (out_dir lives under
        # research/**/results/, which is git-ignored) so rollouts stay with their eval results and never
        # pollute the repo. Mirrors how RobocasaBackend passes rollout_data_dir=out_dir.
        import os, imageio
        vdir = os.path.join(out_dir, "rollouts"); os.makedirs(vdir, exist_ok=True)
        desc = lang.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:40]
        mp4 = os.path.join(vdir, f"episode={idx}--success={success}--task={desc}.mp4")
        with imageio.get_writer(mp4, fps=30) as w:
            for img in primary:
                w.append_data(img)


def get_backend(name="robocasa"):
    name = (name or "robocasa").lower()
    if name == "robocasa":
        return RobocasaBackend()
    if name == "libero":
        return LiberoBackend()
    raise ValueError(f"unknown --sim {name!r} (choose: robocasa | libero)")
