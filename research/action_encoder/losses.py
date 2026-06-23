"""Rank-N-Contrast (Zha et al., NeurIPS 2023) for continuous targets, applied to action chunks.

The ranking signal (the "label") is the action-chunk distance between two decision points:
the z-scored RMSE over the NON-gripper dims (0..5) of the actually-executed next-16 actions.
RNC then shapes the image embedding so that pairs whose action chunks are CLOSE have higher
embedding similarity than pairs whose chunks are FARTHER -- i.e. the embedding's nearest
neighbours are the action nearest neighbours, which is exactly what retrieval needs.

Why RNC (and not plain InfoNCE / triplet): the action distance is CONTINUOUS, so there is no
clean positive/negative threshold; RNC contrasts by *rank* (for anchor i and a closer sample j,
every sample k that is farther-or-equal than j is a negative), needing no threshold and creating
no false negatives among action-similar frames.
"""
from __future__ import annotations

import torch

NON_GRIP = slice(0, 6)  # action dims 0..5 = pos(3)+rot(3); dim 6 = gripper (excluded by default, see README)
INCLUDE_GRIP = False    # global toggle (set by the trainer): when True, action_feat + the retrieval RMSE use
                        # ALL 7 dims (gripper included). Default False keeps every existing result unchanged.


def n_act_dims() -> int:
    """Number of action dims action_feat / the retrieval RMSE use: 7 (incl. gripper) iff INCLUDE_GRIP, else 6."""
    return 7 if INCLUDE_GRIP else 6


def action_feat(act, am, asd):
    """(...,16,7) physical chunk -> (...,16*D) z-scored feature; D = n_act_dims() (6 non-grip, or 7 incl. gripper).

    Euclidean distance over this feature equals the z-scored action RMSE * sqrt(16*D), so it is
    order-equivalent to that RMSE -- the same target metric the gate code uses. torch or numpy.
    """
    an = (act - am) / asd
    return an[..., :n_act_dims()].reshape(*act.shape[:-2], -1)


def label_dist(feat):
    """(B,F) action features -> (B,B) RMSE distances (euclidean / sqrt(F))."""
    return torch.cdist(feat, feat) / (feat.shape[1] ** 0.5)


def rnc_loss(z, ld, tau=2.0):
    """Rank-N-Contrast loss.

    Args:
        z:   (B,D) L2-normalized embeddings.
        ld:  (B,B) label (action) distances, ld[i,i]=0.
        tau: temperature.

    Similarity = -euclidean(z_i,z_k)/tau (the paper's default). For anchor i and positive j, the
    denominator sums over all k!=i with ld[i,k] >= ld[i,j] (the samples ranked no closer than j).
    """
    B = z.shape[0]
    sim = -torch.cdist(z, z) / tau                                  # (B,B)
    ge = ld.unsqueeze(1) >= ld.unsqueeze(2)                         # ge[i,j,k] = ld[i,k] >= ld[i,j]
    idx = torch.arange(B, device=z.device)
    ge[idx, :, idx] = False                                         # exclude k=i from every denominator
    S = sim.unsqueeze(1).masked_fill(~ge, float("-inf"))           # (i,j,k) = sim[i,k] if valid else -inf
    denom = torch.logsumexp(S, dim=2)                              # (B,B) logsumexp over k
    logprob = sim - denom                                          # (B,B), use the j-th column as positive
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    logprob = logprob.masked_fill(eye, 0.0)                        # drop the j=i term
    return -(logprob.sum() / (B * (B - 1)))


def nca_loss(z, ld, k=16, tau=0.1):
    """Neighbourhood InfoNCE (NCA / SupCon-with-kNN-positives): pull each anchor toward its k
    action-NEAREST samples, push from all others. Unlike RNC (global rank), this targets the LOCAL
    neighbourhood -- i.e. top-1 / top-k retrieval, the metric the policy actually uses.

    Args:
        z:   (B,D) L2-normalized embeddings.
        ld:  (B,B) action distances.
        k:   number of action-nearest positives per anchor.
        tau: temperature for cosine similarity.
    """
    B = z.shape[0]
    sim = z @ z.t() / tau                                          # (B,B) cosine/tau
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))                      # exclude self from the denominator
    logZ = torch.logsumexp(sim, dim=1, keepdim=True)              # (B,1) over all j!=i
    ld2 = ld.masked_fill(eye, float("inf"))                        # exclude self when picking positives
    pos = ld2.topk(min(k, B - 1), largest=False, dim=1).indices   # (B,k) k action-nearest
    logp = sim.gather(1, pos) - logZ                              # (B,k) log-prob of each positive
    return -logp.mean()


if __name__ == "__main__":  # self-test: aligned embedding -> ~0 loss; anti-aligned/random -> larger
    torch.manual_seed(0)
    B, D = 64, 8
    lab = torch.randn(B, 1)
    ld = torch.cdist(lab, lab)
    z_aligned = torch.nn.functional.normalize(torch.cat([lab, torch.zeros(B, D - 1)], 1) + 1e-3 * torch.randn(B, D), dim=1)
    z_random = torch.nn.functional.normalize(torch.randn(B, D), dim=1)
    print(f"loss(aligned)={rnc_loss(z_aligned, ld):.4f}  loss(random)={rnc_loss(z_random, ld):.4f}  "
          f"(expect aligned < random)")
