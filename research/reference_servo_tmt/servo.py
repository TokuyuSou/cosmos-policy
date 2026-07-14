"""Small, testable action-space servo operators for Reference-Servo TMT.

All corrections are applied in the action z-score space used by the TMT
metric.  The gripper channel is deliberately copied from the retrieved
reference: a continuous correction must not average open/close commands.
"""
from __future__ import annotations

import numpy as np


ARM = np.arange(6)
POS = np.arange(3)


def normalize_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(actions, np.float32) - mean) / std


def denormalize_actions(actions_z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray(actions_z, np.float32) * std + mean


def constant_bias_target(
    reference_z: np.ndarray,
    target_z: np.ndarray,
    dims: np.ndarray | tuple[int, ...],
    horizon: int = 4,
) -> np.ndarray:
    """Least-squares shared bias for ``reference[:, :horizon, dims]``.

    A single vector is shared by the first ``horizon`` actions.  This is the
    simplest action-space analogue of a short-horizon Cartesian servo twist.
    During ridge fitting this is the supervised correction target; the future
    target is never used by the deployed policy.
    """
    dims = np.asarray(dims, dtype=np.int64)
    error = target_z[:, :horizon, dims] - reference_z[:, :horizon, dims]
    return error.mean(axis=1).astype(np.float32)


def limit_l2(bias: np.ndarray, max_norm: float | None) -> np.ndarray:
    """Bound each correction vector without changing its direction."""
    bias = np.asarray(bias, np.float32)
    if max_norm is None or not np.isfinite(max_norm):
        return bias
    norm = np.linalg.norm(bias, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norm, 1e-8))
    return bias * scale


def apply_constant_bias(
    reference_z: np.ndarray,
    bias: np.ndarray,
    dims: np.ndarray | tuple[int, ...],
    horizon: int = 4,
) -> np.ndarray:
    """Copy a reference chunk and add one shared bias to its first actions."""
    dims = np.asarray(dims, dtype=np.int64)
    out = np.asarray(reference_z, np.float32).copy()
    # Two-step indexing avoids NumPy advanced-index axis reordering.
    first = out[:, :horizon]
    first[..., dims] += np.asarray(bias, np.float32)[:, None, :]
    out[:, :horizon] = first
    return out


def quaternion_relative_rotvec(query_xyzw: np.ndarray, reference_xyzw: np.ndarray) -> np.ndarray:
    """Relative rotation ``query * inverse(reference)`` as a 3-D rotvec."""
    q = np.asarray(query_xyzw, np.float64)
    r = np.asarray(reference_xyzw, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    r = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    # q * conj(r), for xyzw quaternions.
    qv, qw = q[..., :3], q[..., 3:]
    rv, rw = -r[..., :3], r[..., 3:]
    vector = qw * rv + rw * qv + np.cross(qv, rv)
    scalar = qw * rw - np.sum(qv * rv, axis=-1, keepdims=True)
    # q and -q encode the same rotation; use the short arc.
    flip = scalar < 0
    vector = np.where(flip, -vector, vector)
    scalar = np.where(flip, -scalar, scalar)
    sin_half = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(scalar, 0.0, 1.0))
    axis = vector / np.maximum(sin_half, 1e-12)
    return (axis * angle).astype(np.float32)
