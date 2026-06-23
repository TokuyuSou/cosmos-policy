"""Offline retrieval evaluation: does an action-aligned image embedding retrieve better action
chunks than frozen DINO / the current proprio+prev key?

Protocol (clean, no leakage): database = TRAIN-episode decision points (values = their next-16
chunks); queries = held-out TEST-episode decision points. For each query, 1-NN (and k-NN) by the
candidate key -> retrieved chunk -> z-scored non-gripper RMSE vs the query's true next-16 chunk
(the SAME metric the gate code uses). Lower = better.

Reported representations:
    random                 chance upper bound
    proprio_prev           the CURRENT pipeline key (z-scored proprio + prev-16-actions)
    frozen_dino            raw DINOv2 CLS (primary+wrist), standardized -- the encoder baseline
    learned                our action-aligned embedding (native cosine metric)
    learned+proprio_prev   fusion -- does the image embedding ADD to the cheap key?
    frozen_dino+proprio_prev   fusion with raw DINO (isolates the value of *learning*)
    oracle_action          NN by the TRUE action distance (lower bound on retrieval RMSE)

Also reports neighbourhood Spearman: mean over sampled queries of Spearman(key-distance,
action-distance) across the DB -- how well the key's geometry ranks the DB by action similarity.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gripper_event"))
from grip_event_model import _rankdata_avg  # noqa: E402

from losses import action_feat, n_act_dims  # noqa: E402

EPS = 1e-6


def _spearman(a, b):
    ra, rb = _rankdata_avg(np.asarray(a, float)), _rankdata_avg(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + EPS))


def _retr_rmse(db_key, q_key, db_act, q_act, am, asd, ks, device):
    """For each k in ks: 1/k-NN retrieved chunk -> mean z-scored RMSE over queries (non-grip, or all 7 dims
    incl. gripper when losses.INCLUDE_GRIP)."""
    D = torch.cdist(torch.as_tensor(q_key, device=device), torch.as_tensor(db_key, device=device))
    kmax = max(ks)
    idx = D.topk(kmax, largest=False).indices.cpu().numpy()  # (Nq, kmax) nearest db indices
    out = {}
    for k in ks:
        ret = db_act[idx[:, :k]].mean(axis=1)                # (Nq,16,7) averaged chunk (matches k-NN policy)
        e = ((ret - q_act) / asd)[..., :n_act_dims()]        # z-scored residual over the configured action dims
        rmse = np.sqrt((e ** 2).mean(axis=(1, 2)))
        out[k] = float(rmse.mean())
    return out, idx[:, 0]


def _neigh_spearman(db_key, q_key, db_act, q_act, am, asd, device, n_q=300, n_db=2000, seed=0):
    rng = np.random.RandomState(seed)
    qi = rng.choice(len(q_key), min(n_q, len(q_key)), replace=False)
    di = rng.choice(len(db_key), min(n_db, len(db_key)), replace=False)
    kd = torch.cdist(torch.as_tensor(q_key[qi], device=device),
                     torch.as_tensor(db_key[di], device=device)).cpu().numpy()  # (nq,ndb) key dist
    af_q = action_feat(q_act[qi], am, asd).reshape(len(qi), -1)
    af_d = action_feat(db_act[di], am, asd).reshape(len(di), -1)
    ad = np.linalg.norm(af_q[:, None, :] - af_d[None, :, :], axis=2)            # (nq,ndb) action dist
    return float(np.mean([_spearman(kd[r], ad[r]) for r in range(len(qi))]))


def _std(train_block, *blocks):
    m, s = train_block.mean(0), train_block.std(0) + EPS
    return [(b - m) / s for b in (train_block, *blocks)]


@torch.no_grad()
def embed_all(head, cls_p, cls_w, device, bs=4096):
    head.eval()
    out = []
    for i in range(0, len(cls_p), bs):
        v = [torch.as_tensor(cls_p[i:i + bs], device=device), torch.as_tensor(cls_w[i:i + bs], device=device)]
        out.append(head(v).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def compare(data, learned_emb, device, ks=(1, 5), split=("tr", "te")):
    """Build all representations and return a metrics dict + ordered rows for printing.

    db = data.<split[0]> (default train), queries = data.<split[1]> (default test).
    learned_emb: (N, D) embedding for ALL decision points (already L2-normalized).
    """
    dbm = getattr(data, split[0])
    qm = getattr(data, split[1])
    am, asd = data.am, data.asd
    db_act, q_act = data.act[dbm], data.act[qm]

    dino = np.concatenate([data.cls_p, data.cls_w], 1)
    prev_flat = data.prev.reshape(len(data.prev), -1)

    # standardized blocks (stats fit on DB/train rows of each block)
    pp_db, pp_q = None, None
    pp_blocks_db, pp_blocks_q = [], []
    for blk in (data.proprio, prev_flat):
        sd, sq = _std(blk[dbm], blk[qm])
        pp_blocks_db.append(sd)
        pp_blocks_q.append(sq)
    pp_db = np.concatenate(pp_blocks_db, 1)
    pp_q = np.concatenate(pp_blocks_q, 1)

    dino_db, dino_q = _std(dino[dbm], dino[qm])
    le_db_s, le_q_s = _std(learned_emb[dbm], learned_emb[qm])  # standardized learned (for fair fusion)

    reps = {
        "proprio_prev": (pp_db, pp_q),
        "frozen_dino": (dino_db, dino_q),
        "learned": (learned_emb[dbm], learned_emb[qm]),                       # native cosine (unit-norm)
        "learned+proprio_prev": (np.concatenate([le_db_s, pp_db], 1), np.concatenate([le_q_s, pp_q], 1)),
        "frozen_dino+proprio_prev": (np.concatenate([dino_db, pp_db], 1), np.concatenate([dino_q, pp_q], 1)),
    }

    rows = []
    # chance baseline
    rng = np.random.RandomState(0)
    ridx = rng.randint(0, len(db_act), size=len(q_act))
    e = ((db_act[ridx] - q_act) / asd)[..., :6]
    rows.append({"rep": "random", "dim": 0, "rmse": {1: float(np.sqrt((e ** 2).mean(axis=(1, 2))).mean())},
                 "neigh_spearman": 0.0})

    for name, (kdb, kq) in reps.items():
        rmse, _ = _retr_rmse(kdb, kq, db_act, q_act, am, asd, ks, device)
        sp = _neigh_spearman(kdb, kq, db_act, q_act, am, asd, device)
        rows.append({"rep": name, "dim": int(kdb.shape[1]), "rmse": rmse, "neigh_spearman": sp})

    # oracle: retrieve by the true action distance itself (lower bound)
    af_db = action_feat(db_act, am, asd).reshape(len(db_act), -1)
    af_q = action_feat(q_act, am, asd).reshape(len(q_act), -1)
    rmse, _ = _retr_rmse(af_db, af_q, db_act, q_act, am, asd, ks, device)
    rows.append({"rep": "oracle_action", "dim": int(af_db.shape[1]), "rmse": rmse, "neigh_spearman": 1.0})

    return rows


def fmt(rows, ks=(1, 5)):
    head = f"{'representation':26s}{'dim':>6}" + "".join(f"{'RMSE@'+str(k):>10}" for k in ks) + f"{'neigh_sp':>10}"
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = "".join(f"{r['rmse'].get(k, float('nan')):>10.4f}" for k in ks)
        lines.append(f"{r['rep']:26s}{r['dim']:>6}{cells}{r['neigh_spearman']:>10.3f}")
    return "\n".join(lines)
