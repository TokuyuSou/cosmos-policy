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


def list_failure_episodes(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "ep*_success=0.npz")))


def collect(files):
    """Per decision point d=tau+H with a recorded frame: images, proprio, prev(16,7), act(16,7), ep.

    Mirrors retrieval_error.learn_gate.collect but always keeps images and avoids importing the
    heavy DINO stack. A frame is kept only if it has a full H-step history AND a full H-step future
    (so every action-chunk has the same length and the chunk distance is well defined).
    """
    o = {k: [] for k in ("primary", "wrist", "proprio", "prev", "act", "ep", "prog")}
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
            o["prog"].append(dpt / T)                   # normalized task progress (decision step / rollout length)
    return {
        "primary": np.stack(o["primary"]).astype(np.uint8),
        "wrist": np.stack(o["wrist"]).astype(np.uint8),
        "proprio": np.stack(o["proprio"]).astype(np.float32),
        "prev": np.stack(o["prev"]).astype(np.float32),
        "act": np.stack(o["act"]).astype(np.float32),
        "ep": np.array(o["ep"], dtype=np.int64),
        "prog": np.array(o["prog"], dtype=np.float32),
    }


def recompute_prog(data_dir):
    """Normalized progress (t/T) per decision point, in the SAME order as ``collect`` -- used to upgrade
    caches built before ``prog`` was stored (re-reads only the cheap action arrays, NOT the images)."""
    prog = []
    for f in list_success_episodes(data_dir):
        d = np.load(f, allow_pickle=True)
        r = d["realized_actions"]
        qt = d["query_t"]
        T = len(r)
        t2 = {int(t) for t in qt}
        for tau in qt:
            tau = int(tau)
            if tau + 2 * H > T or (tau + H) not in t2:
                continue
            prog.append((tau + H) / T)
    return np.array(prog, dtype=np.float32)


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
    task_pm: np.ndarray = None    # (T,9) per-task proprio mean (for per-task INPUT standardization)
    task_psd: np.ndarray = None   # (T,9) per-task proprio std
    prog: np.ndarray = None       # (N,) normalized task progress t/T per decision point (for the progress head)


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
    prog = m["prog"] if "prog" in m.files else recompute_prog(data_dir)  # upgrade pre-prog caches on the fly
    assert len(prog) == len(act), f"prog/act misaligned ({len(prog)} vs {len(act)}); rebuild cache"

    tr, va, te = _split(ep, seed, val_frac, test_frac)
    pooled = np.concatenate([prev[tr].reshape(-1, 7), act[tr].reshape(-1, 7)], 0)
    am = pooled.mean(0).astype(np.float32)
    asd = (pooled.std(0) + EPS).astype(np.float32)
    pm = proprio[tr].mean(0).astype(np.float32)            # per-task proprio stats (train split)
    psd = (proprio[tr].std(0) + EPS).astype(np.float32)
    print(f"[data] {name} split (seed {seed}): train {tr.sum()} / val {va.sum()} / test {te.sum()} "
          f"({len(np.unique(ep[tr]))}/{len(np.unique(ep[va]))}/{len(np.unique(ep[te]))} ep)")
    return Data(primary, wrist, act.astype(np.float32), proprio.astype(np.float32),
                prev.astype(np.float32), ep, tr, va, te, am, asd,
                task=np.zeros(len(act), np.int64), task_am=am[None].copy(), task_asd=asd[None].copy(),
                tasks=(name,), task_pm=pm[None].copy(), task_psd=psd[None].copy(),
                prog=prog.astype(np.float32))


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
                am, asd, task, task_am, task_asd, tasks,
                task_pm=np.stack([d.task_pm[0] for d in ds]), task_psd=np.stack([d.task_psd[0] for d in ds]),
                prog=cat("prog"))


def append_onpolicy(data: Data, onpolicy_dirs, val_frac=0.2, seed=0) -> Data:
    """DAgger dataset aggregation for the fused encoder: append on-policy rollout records (from
    fused_dagger_collect.py) to the demo ``Data`` as EXTRA episodes, each frame labelled by its VLA expert
    chunk (the metric target). Single-task only.

    Masks are set so the eval stays DEPLOY-FAITHFUL while the loss trains on the aggregate:
      - ``tr``          = demo TRAIN split only  -> the retrieval db (== deploy cache) AND the input-norm stats.
      - ``train_pool``  = demo train  UNION  on-policy TRAIN episodes  -> the batch sampler's pool (the loss).
      - ``va`` == ``te`` = on-policy VAL episodes -> queries; val/test RMSE@1 retrieves them against the demo db,
        i.e. exactly the closed-loop distribution (embed a drifted state, retrieve the nearest DEMO chunk).

    On-policy frames get fresh per-rollout-episode ids (unique across rounds) so the supcon cross-episode
    positives are well-defined. Demo action/proprio stats (am/asd/task_*) are kept UNCHANGED (the deployed
    encoder's frame). ``onpolicy_dirs`` is a list (or comma-string) of collect output dirs (aggregated)."""
    if isinstance(onpolicy_dirs, str):
        onpolicy_dirs = [d for d in onpolicy_dirs.split(",") if d.strip()]
    prev, pro, tgt, pri, wri, epid = [], [], [], [], [], []
    next_ep = int(data.ep.max()) + 1                                  # on-policy ep ids continue after demos
    seen = {}                                                         # (dir_index, rollout_ep) -> global ep id
    for di, d in enumerate(onpolicy_dirs):
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            if len(z["prev"]) == 0:
                continue
            reps = z["ep"] if "ep" in z.files else np.zeros(len(z["prev"]), np.int64)
            gids = np.empty(len(reps), np.int64)
            for j, re in enumerate(reps.tolist()):
                key = (di, int(re))
                if key not in seen:
                    seen[key] = next_ep; next_ep += 1
                gids[j] = seen[key]
            prev.append(z["prev"].astype(np.float32)); pro.append(z["cur_proprio"].astype(np.float32))
            tgt.append(z["target"].astype(np.float32)); pri.append(z["primary"]); wri.append(z["wrist"])
            epid.append(gids)
    assert prev, f"no on-policy records found under {onpolicy_dirs}"
    op_prev = np.concatenate(prev); op_pro = np.concatenate(pro); op_tgt = np.concatenate(tgt)
    op_pri = np.concatenate(pri); op_wri = np.concatenate(wri); op_ep = np.concatenate(epid)
    Nd, Nop = len(data.act), len(op_tgt)

    # episode-disjoint on-policy train/val split
    uep = np.unique(op_ep); rng = np.random.RandomState(seed); rng.shuffle(uep)
    n_val = max(1, int(round(len(uep) * val_frac)))
    val_ep = set(uep[:n_val].tolist())
    op_is_val = np.array([e in val_ep for e in op_ep])

    F = lambda n: np.zeros(n, bool)
    tr = np.concatenate([data.tr, F(Nop)])                           # eval db + norm = demo train split
    train_pool = np.concatenate([data.tr, ~op_is_val])               # loss pool = demo train UNION on-policy train
    va = np.concatenate([F(Nd), op_is_val]); te = va.copy()          # queries = on-policy val (vs demo db)
    prog = np.concatenate([data.prog, np.zeros(Nop, np.float32)])    # on-policy prog unused (no --progress)

    nd = Data(_ConcatView([data.primary, op_pri]), _ConcatView([data.wrist, op_wri]),
              np.concatenate([data.act, op_tgt]).astype(np.float32),
              np.concatenate([data.proprio, op_pro]).astype(np.float32),
              np.concatenate([data.prev, op_prev]).astype(np.float32),
              np.concatenate([data.ep, op_ep]), tr, va, te,
              data.am, data.asd, np.zeros(Nd + Nop, np.int64),
              data.task_am, data.task_asd, data.tasks,
              task_pm=data.task_pm, task_psd=data.task_psd, prog=prog)
    nd.train_pool = train_pool                                       # consumed by train.py's batch sampler
    print(f"[dagger] +{Nop} on-policy frames / {len(uep)} eps ({len(uep)-n_val} train / {n_val} val) "
          f"from {len(onpolicy_dirs)} round(s) | demo train db {int(data.tr.sum())} | total {Nd + Nop}")
    return nd


def append_failures(data: Data, data_dir, cache_dir=None, tail_drop=0) -> Data:
    """Append FAILURE-episode decision points (ep*_success=0.npz, same dir) as extra QUERY-side rows.

    The retrieval encoder's supervision target is VLA PARITY, not task success (the cache replaces VLA
    calls; full-VLA is the ceiling), so failed rollouts carry equally valid per-row VLA teacher chunks --
    and they are the natural supply of the off-support states where retrieval collapses at high skip.

    Masks keep everything else frozen (same discipline as ``append_onpolicy``):
      - ``tr``/``va``/``te`` stay False on the new rows -> the retrieval DB (keys), all input-norm stats
        and the val/test benchmark are IDENTICAL to the success-only run.
      - ``fail`` (bool, N; plain attribute) marks anchor-eligible appended rows; the trainer unions them
        into its KD/IDM/EFF anchor pool. Single-task only.
    ``tail_drop`` excludes the last K decision rows of each failure episode from the anchor pool
    (timeout flailing); the rows stay in the arrays so post/prev row indexing is unaffected."""
    assert len(data.tasks) == 1, "append_failures: single-task Data only"
    here = os.path.dirname(os.path.abspath(__file__))
    name = os.path.basename(os.path.normpath(data_dir))
    cache_dir = cache_dir or os.path.join(here, "cache")
    pri_p = os.path.join(cache_dir, f"{name}_fail_primary.npy")
    wri_p = os.path.join(cache_dir, f"{name}_fail_wrist.npy")
    meta_p = os.path.join(cache_dir, f"{name}_fail_meta.npz")
    files = list_failure_episodes(data_dir)
    assert files, f"no failure episodes (ep*_success=0.npz) under {data_dir}"
    if not (os.path.exists(pri_p) and os.path.exists(wri_p) and os.path.exists(meta_p)):
        c = collect(files)
        np.save(pri_p, c["primary"]); np.save(wri_p, c["wrist"])
        np.savez(meta_p, act=c["act"], proprio=c["proprio"], prev=c["prev"], ep=c["ep"], prog=c["prog"])
        print(f"[failq] {len(files)} failure ep / {len(c['act'])} decision points cached -> {cache_dir}")
    pri_f = np.load(pri_p, mmap_mode="r")
    wri_f = np.load(wri_p, mmap_mode="r")
    m = np.load(meta_p)
    act_f, pro_f, prev_f, ep_f, prog_f = m["act"], m["proprio"], m["prev"], m["ep"], m["prog"]
    assert len(m["ep"]) == 0 or len(np.unique(m["ep"])) <= len(files)

    keep = np.ones(len(act_f), bool)
    if tail_drop > 0:
        for e in np.unique(ep_f):
            r = np.where(ep_f == e)[0]
            keep[r[-tail_drop:]] = False
    Nd, Nf = len(data.act), len(act_f)
    F = lambda n: np.zeros(n, bool)
    nd = Data(_ConcatView([data.primary, pri_f]), _ConcatView([data.wrist, wri_f]),
              np.concatenate([data.act, act_f]).astype(np.float32),
              np.concatenate([data.proprio, pro_f]).astype(np.float32),
              np.concatenate([data.prev, prev_f]).astype(np.float32),
              np.concatenate([data.ep, ep_f + int(data.ep.max()) + 1]),   # fresh episode ids
              np.concatenate([data.tr, F(Nf)]), np.concatenate([data.va, F(Nf)]),
              np.concatenate([data.te, F(Nf)]), data.am, data.asd,
              task=np.concatenate([data.task, np.zeros(Nf, np.int64)]),
              task_am=data.task_am, task_asd=data.task_asd, tasks=data.tasks,
              task_pm=data.task_pm, task_psd=data.task_psd,
              prog=np.concatenate([data.prog, prog_f.astype(np.float32)]))
    nd.fail = np.concatenate([F(Nd), keep])                          # TMT: failure = anchor-only (never a key)
    nd.train_pool = nd.tr | nd.fail                                  # fused sampler pool = success-train U failures
    print(f"[failq] +{Nf} failure decision points from {len(files)} failed eps "
          f"({int(keep.sum())} anchor-eligible, tail_drop={tail_drop}) | db/norms/val/test unchanged")
    return nd
