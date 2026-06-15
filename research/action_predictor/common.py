"""Shared constants and helpers for the action-predictor research code."""

from __future__ import annotations

import numpy as np

# RoboCasa / Cosmos Policy constants (verified against the codebase).
ACTION_DIM = 7  # Cosmos manipulation action dim (env pads to 12 with mobile-base [0,0,0,0,-1]).
PROPRIO_DIM = 9  # gripper_qpos(2) + eef_pos(3) + eef_quat(4)
CHUNK_SIZE = 32  # actions predicted per VLA call
NUM_OPEN_LOOP_STEPS = 16  # actions actually executed before re-query
LATENT_C, LATENT_H, LATENT_W = 16, 28, 28  # Wan VAE latent frame shape

# Future-image latent views stored, in this order.
FUTURE_IMG_VIEWS = ("wrist", "primary", "secondary")


def extract_vec_from_latent_frame(frame_chw: np.ndarray, dim: int) -> np.ndarray:
    """Invert the latent injection (replace_latent_with_proprio / _with_action_chunk).

    Low-dim modalities (proprio/action) are written by tiling the flattened vector
    row-major across the (C*H*W) volume. So we recover the vector by reshaping the
    flattened frame into repeated copies and averaging over them.

    Args:
        frame_chw: (C, H, W) latent frame for the modality.
        dim: dimensionality of the underlying vector (e.g. 9 for proprio).

    Returns:
        (dim,) float32 vector (in the model's normalized space).
    """
    flat = np.asarray(frame_chw, dtype=np.float32).reshape(-1)
    n = flat.shape[0] // dim
    return flat[: n * dim].reshape(n, dim).mean(axis=0)
