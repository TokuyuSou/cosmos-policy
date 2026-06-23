"""Dataset for the action predictor.

Builds (input, target) pairs from VLA calls within successful episodes. For a sample
at decision point d (= the moment we'd re-query the VLA):

    input  = { prev_actions = the last 16 executed actions (ending at d),
               state        = a self-state vector (see STATE_SOURCES below),
               future_img   = Cosmos future-image latent(s) cached from the last call }
    target = the actually-executed next 16 actions (from d)
    cosmos_remaining = the last call's own remaining 16 actions  (baseline)
    anchor = last executed action repeated 16x                    (repeat-last residual decode)

The `state` input is pluggable via STATE_SOURCES so inputs are easy to switch/extend:
  - "vla_future_proprio"  : the VLA-predicted future proprio (normalized; the +32-step
                            prediction made at the last call). [default]
  - "actual_next_proprio" : the ACTUAL proprio at the decision point d, i.e. the real
                            self-state after executing the last call's 16 steps (physical).

A Sample stores every requested state vector in `Sample.states`; a sample is only kept if
ALL requested sources are available, so different state sources can be compared on an
IDENTICAL sample set. Splitting is by episode (no leakage); normalizers fit on train only.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from common import NUM_OPEN_LOOP_STEPS

VIEW_NAME_TO_IDX = {"wrist": 0, "primary": 1, "secondary": 2}

DEFAULT_STATE_SOURCE = "vla_future_proprio"
STATE_SOURCES = ("vla_future_proprio", "actual_next_proprio")


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
    states: Dict[str, np.ndarray]  # state_source -> (9,)
    future_img: np.ndarray  # (V, 16, 28, 28) fp32, selected views
    target: np.ndarray  # (16, 7)
    cosmos_remaining: np.ndarray  # (16, 7)
    anchor: np.ndarray  # (7,) last executed action
    image: Optional[np.ndarray] = None  # (H,W,3) uint8 primary RGB at the decision point (if with_image)
    wrist: Optional[np.ndarray] = None  # (H,W,3) uint8 wrist RGB at the decision point (if with_image)
    obs_emb: Optional[np.ndarray] = None  # (D,) precomputed observation embedding (attached post-hoc)
    # provenance: which cache frame this sample is (for retrieval-hit visualization / analysis)
    src_ep: Optional[int] = None       # source episode number (from ep<N>_success=1.npz)
    src_imgidx: Optional[int] = None   # index into that episode's cur_image array (the decision-point frame)
    src_t: Optional[int] = None        # decision-point timestep within the episode


def _norm_sources(state_sources: Sequence[str]) -> List[str]:
    srcs = list(state_sources) if state_sources else [DEFAULT_STATE_SOURCE]
    for s in srcs:
        if s not in STATE_SOURCES:
            raise ValueError(f"unknown state_source {s!r}; choose from {STATE_SOURCES}")
    return srcs


def build_samples(files: List[str], view_idx: List[int],
                  state_sources: Sequence[str] = (DEFAULT_STATE_SOURCE,),
                  with_image: bool = False) -> List[Sample]:
    """Build training samples. Auto-detects dense vs non-dense schema per file.

    Only samples for which every requested state source is available are kept (so multiple
    sources share an identical sample set). `with_image` additionally attaches the raw RGB
    frame at each decision point (Sample.image), if the file was collected with it.
    """
    srcs = _norm_sources(state_sources)
    samples: List[Sample] = []
    H = NUM_OPEN_LOOP_STEPS
    for f in files:
        d = np.load(f, allow_pickle=True)
        try:                                # ep<N>_success=1.npz -> N (provenance for retrieval-hit analysis)
            ep_id = int(os.path.basename(f).split("ep")[1].split("_")[0])
        except (IndexError, ValueError):
            ep_id = None
        if "realized_actions" in d.files:
            samples.extend(_samples_from_dense(d, view_idx, H, srcs, with_image, ep_id))
        else:
            samples.extend(_samples_from_calls(d, view_idx, H, srcs, with_image, ep_id))
    return samples


def _samples_from_calls(d, view_idx, H, srcs, with_image=False, ep_id=None) -> List[Sample]:
    """Non-dense: one sample per consecutive VLA-call pair (i, i+1)."""
    chunk = d["chunk"].astype(np.float32)  # (n,32,7)
    fp = d["future_proprio_norm"].astype(np.float32)  # (n,9)
    cur = d["cur_proprio"].astype(np.float32)  # (n,9)
    fimg = d["future_img_latent"].astype(np.float32)  # (n,3,16,28,28)
    img = d["cur_image"] if (with_image and "cur_image" in d.files) else None
    out = []
    for i in range(chunk.shape[0] - 1):
        states = {}
        for s in srcs:
            if s == "vla_future_proprio":
                states[s] = fp[i]
            elif s == "actual_next_proprio":
                states[s] = cur[i + 1]  # proprio observed at the next call = state at decision point
        out.append(Sample(
            prev_actions=chunk[i, :H], states=states, future_img=fimg[i, view_idx],
            target=chunk[i + 1, :H], cosmos_remaining=chunk[i, H : 2 * H], anchor=chunk[i, H - 1],
            image=(img[i + 1] if img is not None else None),  # frame at the decision point
            src_ep=ep_id, src_imgidx=i + 1, src_t=i + 1,
        ))
    return out


def _samples_from_dense(d, view_idx, H, srcs, with_image=False, ep_id=None) -> List[Sample]:
    """Dense: for each recorded query at tau, decision point d = tau + 16."""
    realized = d["realized_actions"].astype(np.float32)  # (T,7)
    qt = d["query_t"]
    chunk = d["chunk"].astype(np.float32)  # (Q,32,7)
    fp = d["future_proprio_norm"].astype(np.float32)  # (Q,9)
    cur = d["cur_proprio"].astype(np.float32)  # (Q,9)
    fimg = d["future_img_latent"].astype(np.float32)  # (Q,3,16,28,28)
    cur_img = d["cur_image"] if (with_image and "cur_image" in d.files) else None  # (Q,H,W,3) uint8
    cur_wrist = d["cur_wrist_image"] if (with_image and "cur_wrist_image" in d.files) else None
    T = realized.shape[0]
    t2idx = {int(t): k for k, t in enumerate(qt)}
    out = []
    for q, tau in enumerate(qt):
        tau = int(tau)
        if tau + 2 * H > T:
            continue
        d_idx = t2idx.get(tau + H)  # query recorded at the decision point d = tau+16
        states, ok = {}, True
        for s in srcs:
            if s == "vla_future_proprio":
                states[s] = fp[q]
            elif s == "actual_next_proprio":
                if d_idx is None:  # no record at the decision point
                    ok = False
                    break
                states[s] = cur[d_idx]
        if not ok:
            continue
        out.append(Sample(
            prev_actions=realized[tau : tau + H], states=states, future_img=fimg[q, view_idx],
            target=realized[tau + H : tau + 2 * H], cosmos_remaining=chunk[q, H : 2 * H],
            anchor=realized[tau + H - 1],
            image=(cur_img[d_idx] if (cur_img is not None and d_idx is not None) else None),
            wrist=(cur_wrist[d_idx] if (cur_wrist is not None and d_idx is not None) else None),
            src_ep=ep_id, src_imgidx=d_idx, src_t=tau + H,
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


def fit_normalizers(samples: List[Sample], state_source: str = DEFAULT_STATE_SOURCE) -> Normalizers:
    acts = np.concatenate([np.concatenate([s.prev_actions, s.target], axis=0) for s in samples], axis=0)  # (.,7)
    prop = np.stack([s.states[state_source] for s in samples], axis=0)  # (.,9)
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
    def __init__(self, samples: List[Sample], norm: Normalizers, state_source: str = DEFAULT_STATE_SOURCE):
        self.samples = samples
        self.norm = norm
        self.state_source = state_source

    def __len__(self):
        return len(self.samples)

    def _na(self, a):  # normalize actions
        return (a - self.norm.act_mean) / self.norm.act_std

    def __getitem__(self, k):
        s = self.samples[k]
        prev = self._na(s.prev_actions)  # (16,7)
        state = (s.states[self.state_source] - self.norm.proprio_mean) / self.norm.proprio_std  # (9,)
        img = (s.future_img - self.norm.img_mean[None, :, None, None]) / self.norm.img_std[None, :, None, None]
        anchor = self._na(s.anchor)  # (7,)
        target = self._na(s.target)  # (16,7)
        residual = target - anchor[None, :]  # (16,7) residual-from-repeat-last
        out = {
            "prev_actions": torch.from_numpy(prev).float(),
            "state": torch.from_numpy(state).float(),
            "future_img": torch.from_numpy(img).float(),  # (V,16,28,28)
            "anchor": torch.from_numpy(anchor).float(),
            "target_residual": torch.from_numpy(residual).float(),
            "target_norm": torch.from_numpy(target).float(),
            "target_phys": torch.from_numpy(s.target).float(),  # (16,7) physical
            "cosmos_remaining": torch.from_numpy(s.cosmos_remaining).float(),  # (16,7) physical baseline
        }
        if s.obs_emb is not None:
            out["obs_emb"] = torch.from_numpy(s.obs_emb).float()  # (D,) observation embedding
        if s.image is not None:  # raw frames for the end-to-end spatial-vision path (CHW uint8)
            out["image"] = torch.from_numpy(np.ascontiguousarray(s.image)).permute(2, 0, 1)
            out["wrist"] = torch.from_numpy(np.ascontiguousarray(s.wrist)).permute(2, 0, 1)
        return out
