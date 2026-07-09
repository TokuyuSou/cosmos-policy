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


def supcon_loss(z, ld, ep, k_pos=8, temp=0.1, neg_margin=0.0, pos_thresh=None):
    """Supervised-contrastive (SupCon) retrieval loss for CONTINUOUS action labels, CROSS-EPISODE.

    The stable, canonical alternative to the high-variance soft-argmin NCA. For each anchor the
    POSITIVES are the OTHER-EPISODE points that are action-near (the demos we want to retrieve); all
    other other-episode points are negatives. InfoNCE (cosine-sim / temp) pulls positives embedding-near
    and pushes negatives -> action-near demos become the embedding-near (retrievable) ones. A SET of
    positives (not a single soft-argmin) keeps the gradient low-variance; same-episode points are masked
    (trivial temporal neighbours, and deploy retrieval is always cross-episode).

    Positive selection has two modes (mutually exclusive, chosen by ``pos_thresh``):
      * top-k  (``pos_thresh is None``, DEFAULT): the ``k_pos`` action-NEAREST other-episode points.
        A fixed COUNT per anchor -- this path is byte-identical to the original loss.
      * radius (``pos_thresh`` set): ALL other-episode points within action distance ``pos_thresh``
        (a radius in ``label_dist`` units, typically produced by ``calibrate_supcon_thresh`` so the
        average count matches the top-k it replaces). The count is now ADAPTIVE -- dense neighbourhoods
        keep more positives, sparse ones fewer, so a fixed k neither dilutes dense anchors with far
        "positives" nor starves them of genuinely-near ones. The single nearest other-episode point is
        always kept as a positive so every anchor contributes a gradient. ``k_pos`` is ignored here.

    Args:
        z:     (B,D) L2-normalized embeddings.
        ld:    (B,B) action distances (label).
        ep:    (B,) episode id (same-episode candidates masked out).
        k_pos: # action-nearest cross-episode positives per anchor (top-k mode only).
        temp:  InfoNCE temperature.
        neg_margin: action-distance DEAD-ZONE (default 0.0 = off -> identical to the original loss).
            When >0, candidates that are neither positive nor same-episode but whose action distance is
            within ``neg_margin`` of the positive boundary (the k-th positive distance in top-k mode, or
            ``pos_thresh`` in radius mode) are dropped from the denominator (neither pulled nor pushed) --
            a safety band so borderline "almost a good continuation" points (relevant once state-confusable
            hard negatives are mined into the batch) are not penalized as negatives. Positives unaffected.
        pos_thresh: None -> top-k mode (uses ``k_pos``). float -> radius mode (ignores ``k_pos``): the
            action-distance radius within which cross-episode points count as positives.
    """
    sim = (z @ z.t()) / temp                                     # (B,B) cosine sim / temp (z is L2-normed)
    same = ep[:, None] == ep[None, :]                           # same-episode (incl self) -> not candidates
    neg_inf = torch.finfo(sim.dtype).min
    sim = sim.masked_fill(same, neg_inf)                        # denominator over cross-episode candidates only
    ld_xep = ld.masked_fill(same, float("inf"))
    if pos_thresh is None:                                      # --- top-k positives (default, unchanged) ---
        pos_idx = ld_xep.topk(k_pos, dim=1, largest=False).indices  # k action-nearest x-ep
        pos = torch.zeros_like(sim, dtype=torch.bool).scatter_(1, pos_idx, True)
        boundary = ld_xep.gather(1, pos_idx[:, -1:]) if neg_margin > 0.0 else None  # (B,1) k-th positive dist
    else:                                                       # --- action-distance RADIUS positives ---
        pos = ld_xep <= pos_thresh                             # every x-ep point within the radius is positive
        pos.scatter_(1, ld_xep.argmin(1, keepdim=True), True)  # always keep the single nearest x-ep point
        boundary = pos_thresh                                  # dead-zone (if any) measured from the radius
    if neg_margin > 0.0:                                        # dead-zone just beyond the positives
        dead = (~pos) & (~same) & (ld <= boundary + neg_margin)  # borderline negatives -> drop from denominator
        sim = sim.masked_fill(dead, neg_inf)
    logZ = torch.logsumexp(sim, dim=1, keepdim=True)            # (B,1)
    log_prob = sim - logZ                                       # (B,B)
    return (-(log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def calibrate_supcon_thresh(feat, ep, k_pos, task=None, sample=1024, chunk=256, device="cpu", seed=0):
    """Calibrate the SupCon radius-mode threshold, in ``label_dist`` (action-distance) units.

    Picks the radius so the AVERAGE number of cross-episode positives per anchor equals ``k_pos`` -- i.e.
    the radius neighbourhood carries the same total positive budget as the top-k it replaces (~k_pos*B per
    batch), but the per-anchor count becomes ADAPTIVE (dense neighbourhoods keep more, sparse fewer).
    Concretely: pool the cross-episode action distances from a random sample of anchors and take the value
    below which ``k_pos * n_anchors`` of them fall (so mean positives/anchor == k_pos). Choosing the mean
    directly -- rather than the per-anchor k-th-NN distance -- is robust to distance concentration, where a
    per-anchor-k-th-NN radius over-counts on average because the count-vs-radius curve is steep.

    Same-episode points are excluded (deploy retrieval is always cross-episode). With ``task`` given,
    candidates are restricted to the anchor's task (matches the same-task-batch supcon setting, and keeps
    the radius meaningful since ``label_dist`` is computed in each task's own z-scored action space).

    Args:
        feat:   (N,F) z-scored action-chunk features (``action_feat`` output) -- the SAME space
                ``label_dist`` operates on, so the returned radius is directly comparable to ``ld``.
        ep:     (N,) episode ids (numpy or tensor).
        k_pos:  target average #positives per anchor (the top-k count this radius replaces).
        task:   (N,) optional task ids; when given, candidates are the anchor's task only.
        sample: #random anchors to estimate from (caps the O(N*sample) cost + pooled-distance memory).
        chunk:  #anchors per distance-matrix chunk (bounds peak memory).
    Returns:
        float radius in ``label_dist`` units.
    """
    feat_t = torch.as_tensor(feat, device=device, dtype=torch.float32)
    ep_t = torch.as_tensor(ep, device=device)
    task_t = None if task is None else torch.as_tensor(task, device=device)
    N, F = feat_t.shape
    g = torch.Generator().manual_seed(int(seed))
    anc = torch.randperm(N, generator=g)[:min(sample, N)].to(device)
    dists = []
    for a in anc.split(chunk):                                   # chunk anchors to bound the (chunk,N) matrix
        d = torch.cdist(feat_t[a], feat_t) / (F ** 0.5)         # (chunk,N) == label_dist rows
        d = d.masked_fill(ep_t[a][:, None] == ep_t[None, :], float("inf"))       # cross-episode only
        if task_t is not None:
            d = d.masked_fill(task_t[a][:, None] != task_t[None, :], float("inf"))  # same-task only
        d = d.reshape(-1)
        dists.append(d[torch.isfinite(d)])                     # pooled valid anchor-candidate distances
    dists = torch.cat(dists)
    rank = min(len(dists), max(1, int(round(k_pos * len(anc)))))  # mean positives/anchor == k_pos
    return float(dists.kthvalue(rank).values)


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


def distmatch_loss(z, ld, scale, weight="sammon", beta=1.0, eps=0.1, normalize=False):
    """Distance-CALIBRATED metric regression: make ||z_i - z_j|| match the action distance VALUE itself.

    rnc uses only the RANK of ld; corr uses only the linear CORRELATION with ld (affine-invariant, so it is
    blind to absolute scale/offset -- any monotone-linear rescaling of the embedding is optimal). This loss
    is the missing piece: it regresses the embedding distance onto ``ld / scale`` DIRECTLY, so the learned
    metric is calibrated in action-distance units. Two payoffs:
      * the retrieval hit-distance ``min_j ||z_q - z_j||`` becomes an estimate of the retrieved chunk's
        action error (times ``scale``) -- a directly-usable per-step failure/uncertainty signal, not just
        a monotone proxy;
      * pinning the absolute geometry (not just its ordering/correlation) regularizes the global structure,
        which can steady RMSE@5 / neighbourhood-Spearman.

    ``z`` is L2-normalized so ``ed`` lives in [0, 2]; ``scale`` maps action distance into that window
    (calibrate it with ``calibrate_distmatch_scale`` so a typical action distance lands mid-range). Pairs
    whose scaled target exceeds the sphere's reach are handled by the robust Huber term + the near-pair
    weighting rather than dominating the loss.

    Args:
        z:      (B,D) L2-normalized embeddings.
        ld:     (B,B) action distances (label).
        scale:  action-distance units per unit embedding distance (target = ld/scale). See calibrator.
        weight: 'sammon' -> weight each pair by 1/(ld+eps): emphasize NEAR pairs (the retrieval
                neighbourhood; classic Sammon stress). 'uniform' -> plain mean over pairs (global MDS).
        beta:   Huber/smooth-L1 transition point on the distance residual (robust to the few far pairs a
                bounded ``ed`` cannot reach).
        eps:    floor in the Sammon weight (also caps the up-weighting of the very nearest pairs).
        normalize: match STANDARDIZED distances instead of absolute ones -- this is the CORR objective
                (scale/offset-invariant). Each of ``ed``/``ld`` is batch-standardized (mean 0, std 1) before
                the residual, so ``scale`` is ignored; with weight="uniform" this equals ``corr_loss`` up to
                a constant (min ||std(ed)-std(ld)||^2 = 2(1-Pearson)), and with weight="sammon" it is a
                near-pair-weighted corr. normalize=False (default) keeps the absolute value-matching above.
    """
    ad = _offdiag(ld)                                                      # raw action distances (Sammon weight)
    ed = _offdiag(torch.cdist(z, z))
    if normalize:                                                          # match standardized distances == corr
        a = (ad - ad.mean()) / (ad.std() + 1e-8)
        e = (ed - ed.mean()) / (ed.std() + 1e-8)
        res = torch.nn.functional.smooth_l1_loss(e, a, beta=beta, reduction="none")
    else:                                                                  # match absolute (calibrated) values
        res = torch.nn.functional.smooth_l1_loss(ed, ad / scale, beta=beta, reduction="none")
    if weight == "sammon":
        w = 1.0 / (ad + eps)
        res = res * (w / w.mean())                                         # mean-1 normalized -> scale-stable
    return res.mean()


def calibrate_distmatch_scale(feat, ep, task=None, quantile=0.5, target=1.0,
                              sample=1024, chunk=256, device="cpu", seed=0):
    """Calibrate ``distmatch_loss``'s ``scale`` so a typical action distance maps to a mid-range embedding
    distance (keeps targets inside the L2-normalized sphere's [0,2] reach, no magic number).

    Returns ``scale = quantile(cross-episode ld) / target`` -- i.e. the ``quantile`` (default median) of the
    cross-episode action-distance distribution is mapped to embedding distance ``target`` (default 1.0). The
    pooling (cross-episode, optionally same-task) mirrors ``calibrate_supcon_thresh`` so both calibrators
    see the same distance space.

    Args:
        feat:     (N,F) z-scored action-chunk features (``action_feat`` output).
        ep:       (N,) episode ids.
        task:     (N,) optional task ids; when given, candidates are the anchor's task only.
        quantile: which cross-episode action-distance quantile maps to ``target`` (0.5 = median).
        target:   embedding distance the quantile maps to (mid-range in [0,2]).
    Returns:
        float scale (action-distance units per unit embedding distance).
    """
    feat_t = torch.as_tensor(feat, device=device, dtype=torch.float32)
    ep_t = torch.as_tensor(ep, device=device)
    task_t = None if task is None else torch.as_tensor(task, device=device)
    N, F = feat_t.shape
    g = torch.Generator().manual_seed(int(seed))
    anc = torch.randperm(N, generator=g)[:min(sample, N)].to(device)
    dists = []
    for a in anc.split(chunk):
        d = torch.cdist(feat_t[a], feat_t) / (F ** 0.5)
        d = d.masked_fill(ep_t[a][:, None] == ep_t[None, :], float("inf"))
        if task_t is not None:
            d = d.masked_fill(task_t[a][:, None] != task_t[None, :], float("inf"))
        d = d.reshape(-1)
        dists.append(d[torch.isfinite(d)])
    dists = torch.cat(dists)
    rank = min(len(dists), max(1, int(round(float(quantile) * len(dists)))))
    return float(dists.kthvalue(rank).values / target)
