"""Offline evaluation: does the fine-tuned R3M embedding's distance match action-chunk distance,
and does it retrieve better chunks than raw R3M / the current proprio+prev key?

Protocol (no leakage): database = TRAIN decision points (values = their next-16 chunks); queries =
held-out TEST points. For each query, k-NN by the candidate key -> retrieved chunk -> z-scored
non-gripper RMSE vs the true next-16 chunk (the SAME metric the gate code uses). Lower = better.

Reported keys: random, proprio_prev (current pipeline key), frozen_r3m (raw backbone features),
learned (ours), learned+proprio_prev (does the image ADD?), oracle_action (lower bound).
Also: neighbourhood Spearman (cross-episode) and within-episode Spearman -- the literal target
"embedding distance corresponds to action distance".
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "action_encoder"))
from eval_retrieval import _neigh_spearman, _retr_rmse, _spearman, _std  # noqa: E402
from losses import action_feat  # noqa: E402

from model import R3MBackbone  # noqa: E402

EPS = 1e-6


def _to_chw(hwc_uint8):
    """(k,224,224,3) uint8 np -> (k,3,224,224) uint8 tensor."""
    return torch.from_numpy(np.ascontiguousarray(hwc_uint8)).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def embed_all(model, data, device, bs=512):
    """L2-normalized embedding for ALL decision points -> (N, out_dim) float32."""
    model.eval()
    out = []
    for i in range(0, len(data.act), bs):
        p = _to_chw(data.primary[i:i + bs]).to(device)
        w = _to_chw(data.wrist[i:i + bs]).to(device)
        out.append(model(p, w).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def embed_all_mm(model, data, device, bs=512):
    """L2-normalized fused embedding (image + proprio + prev) for ALL decision points -> (N, out_dim).
    Counterpart of ``embed_all`` for the MultiModalActionMetricEncoder (forward needs prev+proprio)."""
    model.eval()
    out = []
    for i in range(0, len(data.act), bs):
        p = _to_chw(data.primary[i:i + bs]).to(device)
        w = _to_chw(data.wrist[i:i + bs]).to(device)
        prev = torch.as_tensor(data.prev[i:i + bs], device=device)
        pro = torch.as_tensor(data.proprio[i:i + bs], device=device)
        out.append(model(p, w, prev, pro).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def raw_r3m_feats(data, device, bs=512, backbone="resnet18"):
    """Concatenated raw (frozen) R3M features for both views -> (N, 2*feat_dim). The no-training baseline."""
    bb = R3MBackbone(backbone, init="r3m", freeze="frozen").to(device).eval()
    out = []
    for i in range(0, len(data.act), bs):
        p = _to_chw(data.primary[i:i + bs]).to(device)
        w = _to_chw(data.wrist[i:i + bs]).to(device)
        out.append(torch.cat([bb(p), bb(w)], -1).cpu().numpy())
    del bb
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def within_episode_spearman(emb, data, mask, min_frames=8, seed=0, am=None, asd=None):
    """Mean over episodes (in ``mask``) of Spearman( embedding-distance, action-distance ) across all
    within-episode frame pairs. This is the headline "distance correspondence" metric.
    ``am``/``asd`` override the action normalization (pass a task's own stats for per-task eval)."""
    am = data.am if am is None else am
    asd = data.asd if asd is None else asd
    feats = action_feat(data.act, am, asd).reshape(len(data.act), -1)
    vals = []
    for e in np.unique(data.ep[mask]):
        ix = np.where(mask & (data.ep == e))[0]
        if len(ix) < min_frames:
            continue
        ed = np.linalg.norm(emb[ix][:, None] - emb[ix][None], axis=2)
        ad = np.linalg.norm(feats[ix][:, None] - feats[ix][None], axis=2)
        iu = np.triu_indices(len(ix), k=1)
        vals.append(_spearman(ed[iu], ad[iu]))
    return float(np.mean(vals))


def task_rmse1(emb, data, t, device, raw=None):
    """Per-task retrieval RMSE@1 (db=train_t, query=test_t, task t's own action stats): learned vs the
    proprio_prev key (and raw frozen-R3M if given). Used in the multi-task report."""
    dbm, qm = data.tr & (data.task == t), data.te & (data.task == t)
    am, asd = data.task_am[t], data.task_asd[t]
    db_act, q_act = data.act[dbm], data.act[qm]
    pf = data.prev.reshape(len(data.prev), -1)
    pp_db = np.concatenate([_std(data.proprio[dbm], data.proprio[qm])[0], _std(pf[dbm], pf[qm])[0]], 1)
    pp_q = np.concatenate([_std(data.proprio[dbm], data.proprio[qm])[1], _std(pf[dbm], pf[qm])[1]], 1)
    r = {"learned": _retr_rmse(emb[dbm], emb[qm], db_act, q_act, am, asd, (1,), device)[0][1],
         "proprio_prev": _retr_rmse(pp_db, pp_q, db_act, q_act, am, asd, (1,), device)[0][1]}
    if raw is not None:
        rd, rq = _std(raw[dbm], raw[qm])
        r["frozen_r3m"] = _retr_rmse(rd, rq, db_act, q_act, am, asd, (1,), device)[0][1]
    return r


def evaluate(emb, data, device, raw=None, ks=(1, 5), split=("tr", "te")):
    """Build all keys and return printable rows + scalar summary. ``emb`` = learned embedding for all
    points; ``raw`` = optional raw-R3M features for all points (frozen baseline)."""
    dbm, qm = getattr(data, split[0]), getattr(data, split[1])
    am, asd = data.am, data.asd
    db_act, q_act = data.act[dbm], data.act[qm]
    prev_flat = data.prev.reshape(len(data.prev), -1)

    pp_db = np.concatenate([_std(data.proprio[dbm], data.proprio[qm])[0],
                            _std(prev_flat[dbm], prev_flat[qm])[0]], 1)
    pp_q = np.concatenate([_std(data.proprio[dbm], data.proprio[qm])[1],
                           _std(prev_flat[dbm], prev_flat[qm])[1]], 1)
    le_db_s, le_q_s = _std(emb[dbm], emb[qm])

    reps = {
        "proprio_prev": (pp_db, pp_q),
        "learned": (emb[dbm], emb[qm]),
        "learned+proprio_prev": (np.concatenate([le_db_s, pp_db], 1), np.concatenate([le_q_s, pp_q], 1)),
    }
    if raw is not None:
        rd, rq = _std(raw[dbm], raw[qm])
        reps["frozen_r3m"] = (rd, rq)
        reps["frozen_r3m+proprio_prev"] = (np.concatenate([rd, pp_db], 1), np.concatenate([rq, pp_q], 1))

    rows = []
    rng = np.random.RandomState(0)
    ridx = rng.randint(0, len(db_act), size=len(q_act))
    e = ((db_act[ridx] - q_act) / asd)[..., :6]
    rows.append({"rep": "random", "dim": 0, "rmse": {1: float(np.sqrt((e ** 2).mean(axis=(1, 2))).mean())},
                 "neigh_spearman": 0.0})

    order = ["proprio_prev", "frozen_r3m", "frozen_r3m+proprio_prev", "learned", "learned+proprio_prev"]
    for name in [n for n in order if n in reps]:
        kdb, kq = reps[name]
        rmse, _ = _retr_rmse(kdb, kq, db_act, q_act, am, asd, ks, device)
        sp = _neigh_spearman(kdb, kq, db_act, q_act, am, asd, device)
        rows.append({"rep": name, "dim": int(kdb.shape[1]), "rmse": rmse, "neigh_spearman": sp})

    af_db = action_feat(db_act, am, asd).reshape(len(db_act), -1)
    af_q = action_feat(q_act, am, asd).reshape(len(q_act), -1)
    rmse, _ = _retr_rmse(af_db, af_q, db_act, q_act, am, asd, ks, device)
    rows.append({"rep": "oracle_action", "dim": int(af_db.shape[1]), "rmse": rmse, "neigh_spearman": 1.0})
    return rows


def fmt(rows, ks=(1, 5)):
    head = f"{'representation':28s}{'dim':>6}" + "".join(f"{'RMSE@' + str(k):>10}" for k in ks) + f"{'neigh_sp':>10}"
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = "".join(f"{r['rmse'].get(k, float('nan')):>10.4f}" for k in ks)
        lines.append(f"{r['rep']:28s}{r['dim']:>6}{cells}{r['neigh_spearman']:>10.3f}")
    return "\n".join(lines)


def rmse_at(rows, rep, k=1):
    for r in rows:
        if r["rep"] == rep:
            return r["rmse"].get(k, float("nan"))
    return float("nan")
