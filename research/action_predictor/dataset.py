"""Dataset for the action predictor.

Builds (input, target) pairs from consecutive VLA calls within successful episodes:

    input  = { prev_actions = chunk[i][:16],            # last 16 executed actions
               vla_state    = future_proprio_norm[i],   # VLA-predicted self-state
               future_img   = future_img_latent[i] }    # Cosmos future-image latent(s)
    target = chunk[i+1][:16]                             # actually-executed next 16
    cosmos_remaining = chunk[i][16:32]                   # baseline (Cosmos own remaining 16)
    anchor = chunk[i][15] repeated 16x                   # repeat-last (residual decode)

Splitting is by episode (no leakage). Normalizers are fit on the train split only.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from common import ACTION_DIM, NUM_OPEN_LOOP_STEPS, PROPRIO_DIM

VIEW_NAME_TO_IDX = {"wrist": 0, "primary": 1, "secondary": 2}


def list_success_episodes(data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, "ep*_success=1.npz")))


def split_episode_files(files: List[str], val_frac: float, seed: int):
    """Deterministic episode-level split (shared by train and eval)."""
    files = sorted(files)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * val_frac)))
    val_idx = set(perm[:n_val].tolist())
    train = [f for k, f in enumerate(files) if k not in val_idx]
    val = [f for k, f in enumerate(files) if k in val_idx]
    return train, val


@dataclass
class Sample:
    prev_actions: np.ndarray  # (16, 7)
    vla_state: np.ndarray  # (9,)
    future_img: np.ndarray  # (V, 16, 28, 28) fp32, selected views
    target: np.ndarray  # (16, 7)
    cosmos_remaining: np.ndarray  # (16, 7)
    anchor: np.ndarray  # (7,) last executed action


def build_samples(files: List[str], view_idx: List[int]) -> List[Sample]:
    """Build training samples. Auto-detects dense vs non-dense schema per file."""
    samples: List[Sample] = []
    H = NUM_OPEN_LOOP_STEPS
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "realized_actions" in d.files:
            samples.extend(_samples_from_dense(d, view_idx, H))
        else:
            samples.extend(_samples_from_calls(d, view_idx, H))
    return samples


def _samples_from_calls(d, view_idx, H) -> List[Sample]:
    """Non-dense: one sample per consecutive VLA-call pair (i, i+1)."""
    chunk = d["chunk"].astype(np.float32)  # (n,32,7)
    fp = d["future_proprio_norm"].astype(np.float32)  # (n,9)
    fimg = d["future_img_latent"].astype(np.float32)  # (n,3,16,28,28)
    out = []
    for i in range(chunk.shape[0] - 1):
        out.append(Sample(
            prev_actions=chunk[i, :H], vla_state=fp[i], future_img=fimg[i, view_idx],
            target=chunk[i + 1, :H], cosmos_remaining=chunk[i, H : 2 * H], anchor=chunk[i, H - 1],
        ))
    return out


def _samples_from_dense(d, view_idx, H) -> List[Sample]:
    """Dense: for each recorded query at tau, decision point d=tau+16.

    prev_actions = realized[tau:tau+16], target = realized[tau+16:tau+32].
    This reduces EXACTLY to _samples_from_calls when stride=16.
    """
    realized = d["realized_actions"].astype(np.float32)  # (T,7)
    qt = d["query_t"]
    chunk = d["chunk"].astype(np.float32)  # (Q,32,7)
    fp = d["future_proprio_norm"].astype(np.float32)  # (Q,9)
    fimg = d["future_img_latent"].astype(np.float32)  # (Q,3,16,28,28)
    T = realized.shape[0]
    out = []
    for q, tau in enumerate(qt):
        tau = int(tau)
        if tau + 2 * H > T:
            continue
        out.append(Sample(
            prev_actions=realized[tau : tau + H],
            vla_state=fp[q],
            future_img=fimg[q, view_idx],
            target=realized[tau + H : tau + 2 * H],
            cosmos_remaining=chunk[q, H : 2 * H],
            anchor=realized[tau + H - 1],
        ))
    return out


@dataclass
class Normalizers:
    act_mean: np.ndarray
    act_std: np.ndarray
    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    img_mean: np.ndarray  # (16,) per-channel
    img_std: np.ndarray  # (16,)

    def to_dict(self):
        return {k: getattr(self, k) for k in ["act_mean", "act_std", "proprio_mean", "proprio_std", "img_mean", "img_std"]}

    @staticmethod
    def from_dict(d):
        return Normalizers(**{k: np.asarray(d[k], dtype=np.float32) for k in d})

    def denorm_actions(self, a_norm):
        """Map normalized actions back to physical scale (np or torch)."""
        if isinstance(a_norm, torch.Tensor):
            m = torch.as_tensor(self.act_mean, device=a_norm.device, dtype=a_norm.dtype)
            s = torch.as_tensor(self.act_std, device=a_norm.device, dtype=a_norm.dtype)
            return a_norm * s + m
        return a_norm * self.act_std + self.act_mean


def fit_normalizers(samples: List[Sample]) -> Normalizers:
    acts = np.concatenate([np.concatenate([s.prev_actions, s.target], axis=0) for s in samples], axis=0)  # (.,7)
    prop = np.stack([s.vla_state for s in samples], axis=0)  # (.,9)
    imgs = np.stack([s.future_img for s in samples], axis=0)  # (.,V,16,28,28)
    eps = 1e-6
    return Normalizers(
        act_mean=acts.mean(0),
        act_std=acts.std(0) + eps,
        proprio_mean=prop.mean(0),
        proprio_std=prop.std(0) + eps,
        img_mean=imgs.mean(axis=(0, 1, 3, 4)),  # per-channel (16,)
        img_std=imgs.std(axis=(0, 1, 3, 4)) + eps,
    )


class ChunkDataset(Dataset):
    def __init__(self, samples: List[Sample], norm: Normalizers):
        self.samples = samples
        self.norm = norm

    def __len__(self):
        return len(self.samples)

    def _na(self, a):  # normalize actions
        return (a - self.norm.act_mean) / self.norm.act_std

    def __getitem__(self, k):
        s = self.samples[k]
        prev = self._na(s.prev_actions)  # (16,7)
        state = (s.vla_state - self.norm.proprio_mean) / self.norm.proprio_std  # (9,)
        img = (s.future_img - self.norm.img_mean[None, :, None, None]) / self.norm.img_std[None, :, None, None]
        anchor = self._na(s.anchor)  # (7,)
        target = self._na(s.target)  # (16,7)
        residual = target - anchor[None, :]  # (16,7) residual-from-repeat-last
        return {
            "prev_actions": torch.from_numpy(prev).float(),
            "state": torch.from_numpy(state).float(),
            "future_img": torch.from_numpy(img).float(),  # (V,16,28,28)
            "anchor": torch.from_numpy(anchor).float(),
            "target_residual": torch.from_numpy(residual).float(),
            "target_norm": torch.from_numpy(target).float(),
            "target_phys": torch.from_numpy(s.target).float(),  # (16,7) physical
            "cosmos_remaining": torch.from_numpy(s.cosmos_remaining).float(),  # (16,7) physical baseline
        }
