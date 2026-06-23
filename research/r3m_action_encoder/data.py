"""Data prep for the R3M action-metric encoder.

For every decision point d (a frame with a recorded image) we keep:
    primary, wrist : (224,224,3) uint8 RGB of the third-person / wrist camera observed at d
    act            : (16,7) the actually-executed next-16 action chunk  (= the metric LABEL)
    proprio (9,), prev (16,7), ep : cheap baseline key features + episode id

The action-chunk DISTANCE between two frames is the target: we want the encoder to place
frames whose executed chunks are close (far) close (far) in embedding space.

Unlike the DINO `action_encoder`, the backbone is FINE-TUNED, so we cannot precompute features.
Instead the raw uint8 images are collected once and cached as memory-mapped .npy files; training
reads slices on demand (cheap, ~4 GB on disk, near-zero RAM).

Episode-level train/val/test split (no leakage). Action z-score stats are fit on train only.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

H = 16          # NUM_OPEN_LOOP_STEPS: actions executed before re-query (matches the policy / action_encoder)
EPS = 1e-6


def list_success_episodes(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "ep*_success=1.npz")))


def collect(files):
    """Per decision point d=tau+H with a recorded frame: images, proprio, prev(16,7), act(16,7), ep.

    Mirrors retrieval_error.learn_gate.collect but always keeps images and avoids importing the
    heavy DINO stack. A frame is kept only if it has a full H-step history AND a full H-step future
    (so every action-chunk has the same length and the chunk distance is well defined).
    """
    o = {k: [] for k in ("primary", "wrist", "proprio", "prev", "act", "ep")}
    for ei, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        r = d["realized_actions"].astype(np.float32)
        qt = d["query_t"]
        cur = d["cur_proprio"].astype(np.float32)
        ci, wi = d["cur_image"], d["cur_wrist_image"]
        T = len(r)
        t2 = {int(t): k for k, t in enumerate(qt)}      # decision timestep -> image row
        for tau in qt:
            tau = int(tau)
            if tau + 2 * H > T:
                continue
            dpt = tau + H
            di = t2.get(dpt)
            if di is None:
                continue
            hist = r[dpt - H:dpt]
            o["primary"].append(ci[di]); o["wrist"].append(wi[di])
            o["proprio"].append(cur[di]); o["prev"].append(hist)
            o["act"].append(r[dpt:dpt + H]); o["ep"].append(ei)
    return {
        "primary": np.stack(o["primary"]).astype(np.uint8),
        "wrist": np.stack(o["wrist"]).astype(np.uint8),
        "proprio": np.stack(o["proprio"]).astype(np.float32),
        "prev": np.stack(o["prev"]).astype(np.float32),
        "act": np.stack(o["act"]).astype(np.float32),
        "ep": np.array(o["ep"], dtype=np.int64),
    }


def _split(ep, seed, val_frac, test_frac):
    eps = np.unique(ep)
    rng = np.random.RandomState(seed)
    rng.shuffle(eps)
    n = len(eps)
    n_te = max(1, int(round(n * test_frac)))
    n_va = max(1, int(round(n * val_frac)))
    te_e = set(eps[:n_te].tolist())
    va_e = set(eps[n_te:n_te + n_va].tolist())
    te = np.array([e in te_e for e in ep])
    va = np.array([e in va_e for e in ep])
    return ~(te | va), va, te


@dataclass
class Data:
    primary: np.ndarray   # (N,224,224,3) uint8, memory-mapped
    wrist: np.ndarray     # (N,224,224,3) uint8, memory-mapped
    act: np.ndarray       # (N,16,7) executed next-16 chunk = metric label
    proprio: np.ndarray   # (N,9)
    prev: np.ndarray      # (N,16,7)
    ep: np.ndarray        # (N,) episode index (globally unique across tasks)
    tr: np.ndarray        # bool (N,) train mask
    va: np.ndarray        # bool val mask
    te: np.ndarray        # bool test mask
    am: np.ndarray        # (7,) action mean (train, pooled prev+act; global pooled for multi-task)
    asd: np.ndarray       # (7,) action std
    # multi-task fields (single-task: one task, populated trivially -> existing code unaffected)
    task: np.ndarray = None       # (N,) task id per sample
    task_am: np.ndarray = None    # (T,7) per-task action mean
    task_asd: np.ndarray = None   # (T,7) per-task action std
    tasks: tuple = ()             # task names, indexed by task id


class _ConcatView:
    """Read-only drop-in for a stacked (N,...) array backed by a LIST of per-task arrays (mmaps).
    Supports int / slice / index-array access exactly like the underlying arrays, so the Dataset and
    eval code that does ``primary[i]`` or ``primary[i:j]`` work unchanged on multi-task data without
    copying every task's images into one file."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.offsets = np.concatenate([[0], np.cumsum([len(a) for a in arrays])])
        self.dtype = arrays[0].dtype
        self.shape = (int(self.offsets[-1]),) + tuple(arrays[0].shape[1:])

    def __len__(self):
        return int(self.offsets[-1])

    def _locate(self, i):
        t = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return t, int(i - self.offsets[t])

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            t, j = self._locate(idx)
            return self.arrays[t][j]
        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(len(self)))
        idx = np.asarray(idx)
        out = np.empty((len(idx),) + self.shape[1:], dtype=self.dtype)
        owner = np.searchsorted(self.offsets, idx, side="right") - 1
        for t in np.unique(owner):
            m = owner == t
            out[m] = self.arrays[t][idx[m] - self.offsets[t]]
        return out


def load(data_dir, seed=0, val_frac=0.15, test_frac=0.15, cache_dir=None, rebuild=False) -> Data:
    here = os.path.dirname(os.path.abspath(__file__))
    name = os.path.basename(os.path.normpath(data_dir))
    cache_dir = cache_dir or os.path.join(here, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    pri_p = os.path.join(cache_dir, f"{name}_primary.npy")
    wri_p = os.path.join(cache_dir, f"{name}_wrist.npy")
    meta_p = os.path.join(cache_dir, f"{name}_meta.npz")

    if not (os.path.exists(pri_p) and os.path.exists(wri_p) and os.path.exists(meta_p)) or rebuild:
        files = list_success_episodes(data_dir)
        c = collect(files)
        np.save(pri_p, c["primary"]); np.save(wri_p, c["wrist"])
        np.savez(meta_p, act=c["act"], proprio=c["proprio"], prev=c["prev"], ep=c["ep"])
        print(f"[data] {len(files)} ep / {len(c['act'])} decision points cached -> {cache_dir}")

    primary = np.load(pri_p, mmap_mode="r")
    wrist = np.load(wri_p, mmap_mode="r")
    m = np.load(meta_p)
    act, proprio, prev, ep = m["act"], m["proprio"], m["prev"], m["ep"]

    tr, va, te = _split(ep, seed, val_frac, test_frac)
    pooled = np.concatenate([prev[tr].reshape(-1, 7), act[tr].reshape(-1, 7)], 0)
    am = pooled.mean(0).astype(np.float32)
    asd = (pooled.std(0) + EPS).astype(np.float32)
    print(f"[data] {name} split (seed {seed}): train {tr.sum()} / val {va.sum()} / test {te.sum()} "
          f"({len(np.unique(ep[tr]))}/{len(np.unique(ep[va]))}/{len(np.unique(ep[te]))} ep)")
    return Data(primary, wrist, act.astype(np.float32), proprio.astype(np.float32),
                prev.astype(np.float32), ep, tr, va, te, am, asd,
                task=np.zeros(len(act), np.int64), task_am=am[None].copy(), task_asd=asd[None].copy(),
                tasks=(name,))


def load_multi(data_dirs, seed=0, val_frac=0.15, test_frac=0.15, cache_dir=None, rebuild=False) -> Data:
    """Load several task dirs and combine into ONE Data with per-sample task ids and per-task action
    stats. Reuses the single-task ``load`` per dir (so caching is unchanged), keeps images as per-task
    mmaps behind a `_ConcatView` (no giant combined copy), and splits each task independently so every
    task is represented in train/val/test."""
    ds = [load(d, seed, val_frac, test_frac, cache_dir, rebuild) for d in data_dirs]
    cat = lambda key: np.concatenate([getattr(d, key) for d in ds])
    ep_parts, off = [], 0
    for d in ds:
        ep_parts.append(d.ep + off); off += int(d.ep.max()) + 1     # globally-unique episode ids
    ep = np.concatenate(ep_parts)
    task = np.concatenate([np.full(len(d.act), i, np.int64) for i, d in enumerate(ds)])
    task_am = np.stack([d.am for d in ds]); task_asd = np.stack([d.asd for d in ds])
    pooled = np.concatenate([np.concatenate([d.prev[d.tr].reshape(-1, 7), d.act[d.tr].reshape(-1, 7)], 0)
                             for d in ds], 0)
    am = pooled.mean(0).astype(np.float32); asd = (pooled.std(0) + EPS).astype(np.float32)
    tasks = tuple(d.tasks[0] for d in ds)
    print(f"[multi] {len(tasks)} tasks {tasks} | total {len(task)} dp | per-task train "
          f"{[int(d.tr.sum()) for d in ds]}")
    return Data(_ConcatView([d.primary for d in ds]), _ConcatView([d.wrist for d in ds]),
                cat("act"), cat("proprio"), cat("prev"), ep, cat("tr"), cat("va"), cat("te"),
                am, asd, task, task_am, task_asd, tasks)
