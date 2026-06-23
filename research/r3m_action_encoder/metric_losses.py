"""Losses that shape embedding distance to match executed-action-chunk distance.

`action_feat` / `label_dist` / `rnc_loss` are reused verbatim from the sibling `action_encoder`
package (same action-distance label = z-scored RMSE over the non-gripper dims of the next-16 chunk,
so results are directly comparable). We add `corr_loss`, which optimises the target property
*directly*: maximise the correlation between pairwise embedding distance and pairwise action
distance within a batch.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "action_encoder"))
from losses import action_feat, label_dist, rnc_loss  # noqa: E402,F401  (reused)


def _offdiag(m):
    """Strictly-upper-triangular entries of a (B,B) matrix as a 1-D vector."""
    B = m.shape[0]
    iu = torch.triu_indices(B, B, offset=1, device=m.device)
    return m[iu[0], iu[1]]


def softnn_loss(z, ld, temp=0.3):
    """Soft nearest-neighbour (NCA-style) retrieval loss -- the DIRECT surrogate for retrieval RMSE@1.

    corr/rnc shape the embedding so distance correlates with action distance *globally* (great Spearman)
    but say nothing about whether the SINGLE nearest neighbour is good -- which is exactly what RMSE@1
    measures and what the proprio_prev key wins on. This loss closes that gap: for each anchor it forms
    a softmax over the other batch items by embedding proximity (sharp -> the neighbour the embedding
    would actually retrieve) and minimises the EXPECTED action distance of that soft neighbour. As
    temp->0 it becomes the true top-1 action distance; finite temp keeps the gradient informative.

    Args:
        z:    (B,D) L2-normalized embeddings.
        ld:   (B,B) action distances (label).
        temp: softmax temperature on embedding distance (smaller = sharper = closer to literal top-1).
    """
    B = z.shape[0]
    ed = torch.cdist(z, z)                                    # (B,B) embedding distance
    mask = torch.eye(B, dtype=torch.bool, device=z.device)
    logits = (-ed / temp).masked_fill(mask, torch.finfo(ed.dtype).min)   # exclude self
    w = torch.softmax(logits, dim=1)                         # (B,B) retrieval weights per anchor
    return (w * ld).sum(dim=1).mean()                        # expected retrieved action distance


def top1nca_loss(z, ld, ep, temp=0.2):
    """CROSS-EPISODE soft-nearest-neighbour (NCA) loss -- the deploy-faithful TOP-1 surrogate.

    At deploy, retrieval uses ONLY the top-1 cache neighbour, and ALWAYS cross-episode (the query is a
    held-out episode; the cache is the train episodes). corr/rnc instead optimise the *global*
    distance<->action correspondence over ALL pairs, dominated by trivial WITHIN-episode temporal
    neighbours (adjacent frames, near-identical) that deploy never retrieves. This loss targets exactly
    what deploy does: for each anchor, softmax over OTHER-EPISODE batch points by embedding proximity
    (= the neighbour the embedding would retrieve) and minimise its EXPECTED action distance. temp->0
    -> the literal top-1 action distance.

    Masking same-episode is the key difference from ``softnn``: it removes the trivial within-episode
    neighbours (no useful gradient + collapse risk) and matches the cross-episode deploy setting.

    Args:
        z:    (B,D) L2-normalized embeddings.
        ld:   (B,B) action distances (label).
        ep:   (B,) episode id per sample (same-episode candidates are masked out).
        temp: softmax temperature on embedding distance (smaller = sharper = closer to literal top-1).
    """
    ed = torch.cdist(z, z)                                    # (B,B) embedding distance
    same = ep[:, None] == ep[None, :]                        # same-episode (incl. self) -> masked
    logits = (-ed / temp).masked_fill(same, torch.finfo(ed.dtype).min)
    w = torch.softmax(logits, dim=1)                         # (B,B) CROSS-episode retrieval weights
    return (w * ld).sum(dim=1).mean()                        # expected cross-episode retrieved action distance


def supcon_loss(z, ld, ep, k_pos=8, temp=0.1):
    """Supervised-contrastive (SupCon) retrieval loss for CONTINUOUS action labels, CROSS-EPISODE.

    The stable, canonical alternative to the high-variance soft-argmin NCA. For each anchor, the
    ``k_pos`` action-NEAREST OTHER-EPISODE points are POSITIVES (the demos we want to retrieve); all
    other other-episode points are negatives. InfoNCE (cosine-sim / temp) pulls positives embedding-near
    and pushes negatives -> action-near demos become the embedding-near (retrievable) ones. A SET of
    positives (not a single soft-argmin) keeps the gradient low-variance; same-episode points are masked
    (trivial temporal neighbours, and deploy retrieval is always cross-episode).

    Args:
        z:     (B,D) L2-normalized embeddings.
        ld:    (B,B) action distances (label).
        ep:    (B,) episode id (same-episode candidates masked out).
        k_pos: # action-nearest cross-episode positives per anchor.
        temp:  InfoNCE temperature.
    """
    sim = (z @ z.t()) / temp                                     # (B,B) cosine sim / temp (z is L2-normed)
    same = ep[:, None] == ep[None, :]                           # same-episode (incl self) -> not candidates
    neg_inf = torch.finfo(sim.dtype).min
    sim = sim.masked_fill(same, neg_inf)                        # denominator over cross-episode candidates only
    logZ = torch.logsumexp(sim, dim=1, keepdim=True)            # (B,1)
    log_prob = sim - logZ                                       # (B,B)
    pos_idx = ld.masked_fill(same, float("inf")).topk(k_pos, dim=1, largest=False).indices  # k action-nearest x-ep
    pos = torch.zeros_like(sim, dtype=torch.bool).scatter_(1, pos_idx, True)
    return (-(log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def corr_loss(z, ld):
    """1 - Pearson( ||z_i - z_j||, action_dist(i,j) ) over all distinct pairs in the batch.

    Directly targets "embedding distance corresponds to action distance" (the stated objective),
    rather than the rank-only surrogate of RNC.

    Args:
        z:  (B,D) L2-normalized embeddings.
        ld: (B,B) action distances.
    """
    ed = _offdiag(torch.cdist(z, z))
    ad = _offdiag(ld)
    ed = ed - ed.mean()
    ad = ad - ad.mean()
    corr = (ed * ad).sum() / (ed.norm() * ad.norm() + 1e-8)
    return 1.0 - corr
