"""Dictionary-lookup ("retrieval") policy: a drop-in alternative to PredictorPolicy that, at
a skip decision point, returns the next-16 actions of the NEAREST entry in a database built
from the predictor's train-split samples -- instead of running a neural net.

It has the same `predict_chunk(...)` signature as PredictorPolicy, so it plugs straight into
`closed_loop.run_closed_loop_episode` (and `run_closed_loop_eval.py --policy retrieval`) with no
change to the closed-loop code.

Database (built once at init, from the SAME train split the predictor uses -> fair head-to-head):
    key   = normalized features available at a decision point: prev_actions(16x7) [+ state(9)]
            [+ cached future-image latent]  -- chosen by `key`
    value = that sample's actually-executed next-16 actions (physical, (16,7))
At a skip, the identical key is built from the live (prev_actions, state, cached_img) and the
value of the nearest key (Euclidean) is returned; k>1 averages the k nearest values.

This is the closed-loop, NON-oracle counterpart of research/retrieval_oracle/ (which queried
with the ground-truth answer): here the query is built only from information available online.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from common import NUM_OPEN_LOOP_STEPS
from dataset import (
    DEFAULT_STATE_SOURCE,
    VIEW_NAME_TO_IDX,
    build_samples,
    fit_normalizers,
    list_success_episodes,
    split_episode_files,
)

KEY_CHOICES = ("prev", "prev_state", "prev_state_img")  # which features form the lookup key
METRIC_CHOICES = ("l2", "norm")  # oracle NN distance: physical L2 vs per-dim z-scored L2


class RetrievalPolicy:
    def __init__(self, data_dir: str, key: str = "prev_state", k: int = 1,
                 state_source: str = DEFAULT_STATE_SOURCE, img_views: str = "primary",
                 val_frac: float = 0.15, seed: int = 0, cache_episodes: int | None = None,
                 consensus_encoder: str = "", consensus_k: int = 50, fused_encoder: str = "",
                 combine: str = "mean", combine_beta: float = 0.15, combine_tau: float = 0.0,
                 combine_grip: str = "top1", fused_renorm: bool = False):
        assert key in KEY_CHOICES, f"key must be one of {KEY_CHOICES}"
        assert k >= 1
        self.key, self.k, self.state_source = key, int(k), state_source
        # How the k nearest cache VALUES are combined into the executed chunk (see _knn_value):
        #   'mean'    -- legacy: top-1 if k==1 else UNIFORM mean of the k nearest (DEFAULT, unchanged).
        #   'softknn' -- kernel-weighted (Nadaraya-Watson) average over the k nearest, weights
        #                w_j ∝ exp(-(d_j - d_min)/(beta*mean_k d)). Offline this beats top-1 by ~16%
        #                z-RMSE because the metric ranks neighbours well but can't SELECT the best one,
        #                so distance-weighted averaging cuts variance. Gripper is kept discrete and an
        #                optional mode-guard (combine_tau>0) drops action-outlier neighbours first.
        assert combine in ("mean", "softknn", "mode"), "combine must be 'mean', 'softknn' or 'mode'"
        assert combine_grip in ("top1", "mean"), "combine_grip must be 'top1' or 'mean'"
        self.combine = combine
        self.combine_beta = float(combine_beta)
        self.combine_tau = float(combine_tau)      # >0 enables the mode-guard (radius = tau * median pair-dist)
        self.combine_grip = combine_grip           # 'top1' keeps the nearest neighbour's discrete gripper
        # Optional per-skip retrieval diagnostics (env RETR_LOG=<prefix>): logs, for EVERY predict_chunk,
        # the top-1 key distance (OOD signal), shortlist mean distance, effective #neighbours, and the
        # executed-chunk change (softknn vs top1). One JSON line per skip -> <prefix>_<pid>.jsonl. Off by
        # default (no env) -> zero overhead, no behaviour change. Used to compare ONLINE query geometry to
        # the offline test distribution (does the -16% offline gain survive the closed-loop state shift?).
        self._logpath = os.environ.get("RETR_LOG")
        self._logf = None
        self._nskip_logged = 0
        self.use_state = "state" in key
        self.use_img = "img" in key
        self.views = [VIEW_NAME_TO_IDX[v.strip()] for v in img_views.split(",") if v.strip()] if self.use_img else []
        # CONSENSUS retrieval (optional): also retrieve by a learned R3M encoder key and, at a skip, take the
        # entry in BOTH the N1 top-K and the encoder top-K -- nearest by N1 -- falling back to N1 top-1 if the
        # intersection is empty. Off by default (consensus_encoder="") -> identical to plain N1 retrieval.
        self.consensus = bool(consensus_encoder)
        self.consensus_k = int(consensus_k)
        # FUSED retrieval (optional): use a learned multimodal encoder (image + proprio + prev) embedding
        # as the SOLE lookup key -- a drop-in REPLACEMENT for the N1 (prev_state) key. The cache keys are
        # that encoder's embedding of every cache decision point; at a skip the live frames+prev+proprio
        # are embedded the same way and the nearest entry's value is returned. Off by default (="").
        self.fused = bool(fused_encoder)
        assert not (self.fused and self.consensus), "fused_encoder and consensus_encoder are mutually exclusive"
        # Option A re-normalization (fused only): re-fit the encoder's prev/proprio z-score to THIS cache's
        # own stats, overriding the checkpoint's baked training stats. Off by default -> behaviour identical
        # (the encoder uses its saved stats). On => train/deploy normalization match, so a multi-task or
        # unseen-task encoder sees inputs in the DEPLOY task's own frame (see obs_embed.fit_encoder_norm).
        self.fused_renorm = bool(fused_renorm)
        if self.fused:
            assert state_source == "actual_next_proprio", (
                "fused_encoder requires --state-source actual_next_proprio "
                "(the encoder was trained on the ACTUAL decision-point proprio)")

        # Cache = same train split as the action-predictor eval (default), OR the first
        # `cache_episodes` success episodes when set (to match a gate built on that same cache).
        need_img = self.consensus or self.fused
        # data_dir may be a comma-separated LIST of dirs (mixed multi-task cache): pool each dir's cache
        # files (its train split, or first cache_episodes). A single dir reproduces the original behaviour.
        self.data_dirs = [d for d in str(data_dir).split(",") if d]
        cache_files = []
        for d in self.data_dirs:
            fs = list_success_episodes(d)
            cache_files += (fs[:cache_episodes] if cache_episodes else split_episode_files(fs, val_frac, seed)[0])
        samples = build_samples(cache_files, self.views, [state_source], with_image=need_img)
        with warnings.catch_warnings():  # benign empty-slice stats for unused (image) modality
            warnings.simplefilter("ignore", RuntimeWarning)
            self.norm = fit_normalizers(samples, state_source)
        self.values = np.stack([s.target for s in samples]).astype(np.float32)  # (N,16,7) physical
        # provenance of every cache entry (which real frame it is) -> exposed via self.last_match on each
        # predict_chunk, for retrieval-hit visualization / analysis. Parallel to self.values/self.keys.
        self.src_ep = [s.src_ep for s in samples]
        self.src_imgidx = [s.src_imgidx for s in samples]
        self.src_t = [s.src_t for s in samples]
        self.last_match = None

        # Attributes PredictorPolicy exposes, so the runner's record-keeping works unchanged.
        self.run_dir = data_dir

        if self.fused:  # learned multimodal key REPLACES N1: cache embeddings + keep the encoder for live queries
            import hashlib
            import torch
            from obs_embed import compute_fused_emb, load_fused_encoder
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = load_fused_encoder(fused_encoder, self.device)
            rn = ""
            if self.fused_renorm:  # Option A: override baked stats with this cache's (deploy task's) stats.
                from obs_embed import fit_encoder_norm   # applied IN PLACE -> both cache + live query use it.
                self.encoder.set_norm(*fit_encoder_norm(samples, state_source))
                rn = "_renorm"      # namespace the embedding cache so a renormed/non-renormed run never collide
            cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            dn = (os.path.basename(os.path.normpath(self.data_dirs[0])) if len(self.data_dirs) == 1 else
                  f"mix{len(self.data_dirs)}_" + hashlib.md5(",".join(self.data_dirs).encode()).hexdigest()[:8])
            et = os.path.splitext(os.path.basename(fused_encoder))[0]
            self.keys = compute_fused_emb(samples, self.encoder, self.device, state_source,
                                          os.path.join(cdir, f"fusedemb_{dn}_{et}{rn}_retrcache.npz")).astype(np.float32)
            self.img_mode = f"fused[{et}{rn}]:k{k}{self._combine_suffix()}"
            self.n_rollout = 0
            self.n_demo = len(self.keys)      # demo-prefix length
            self.rollout_gate_tau = None      # off-support gate threshold (None = gate off; kept for _gated_dist)
            return

        self.keys = np.stack([self._key(s.prev_actions, s.states[state_source], s.future_img)
                              for s in samples]).astype(np.float32)  # (N, D)
        self.img_mode = f"retrieval:{key}:k{k}{self._combine_suffix()}"
        if self.consensus:  # build the encoder cache keys (kept loaded for the live query at each skip)
            import torch
            from obs_embed import attach_obs_emb, load_corr_encoder
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = load_corr_encoder(consensus_encoder, self.device)
            cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            dn, et = os.path.basename(os.path.normpath(data_dir)), os.path.splitext(os.path.basename(consensus_encoder))[0]
            attach_obs_emb(samples, self.encoder, self.device, os.path.join(cdir, f"obsemb_{dn}_{et}_retrcache.npz"))
            self.emb_keys = np.stack([s.obs_emb for s in samples]).astype(np.float32)  # (N, D_emb) encoder cache keys
            self.img_mode = f"consensus[{et}:K{self.consensus_k}]+{self.img_mode}"

        self.n_dagger = 0

    def _combine_suffix(self) -> str:
        """Tag for img_mode/logging. Empty for the legacy 'mean' path so existing run names are unchanged."""
        if self.combine == "mean":
            return ""
        if self.combine == "mode":
            return f":mode(b{self.combine_beta})"
        g = "" if self.combine_grip == "top1" else "_gripmean"
        t = "" if self.combine_tau <= 0 else f"_t{self.combine_tau}"
        return f":softknn(b{self.combine_beta}{t}{g})"

    def _key(self, prev_actions, state, future_img) -> np.ndarray:
        """Flatten the selected, per-modality-normalized features into one vector."""
        n = self.norm
        parts = [((prev_actions - n.act_mean) / n.act_std).reshape(-1)]  # (112,)
        if self.use_state:
            parts.append((state - n.proprio_mean) / n.proprio_std)  # (9,)
        if self.use_img:
            z = (future_img - n.img_mean[None, :, None, None]) / n.img_std[None, :, None, None]
            parts.append(z.reshape(-1))  # (V*16*28*28,)
        return np.concatenate(parts).astype(np.float32)

    def _query_key(self, prev_actions, current_proprio, cached_future_proprio, cached_future_img) -> np.ndarray:
        """Build the lookup key from the live closed-loop inputs (shared by predict_chunk and the fusion policy)."""
        state = current_proprio if self.state_source == "actual_next_proprio" else cached_future_proprio
        img = (np.asarray(cached_future_img, np.float32)[self.views] if self.use_img and cached_future_img is not None
               else np.zeros((len(self.views), 16, 28, 28), np.float32))
        return self._key(np.asarray(prev_actions, np.float32), np.asarray(state, np.float32), img)

    def _gated_dist(self, dist):
        """OFF-SUPPORT gate for the expanded cache (fused + rollout_cache + rollout_gate_q only).
        If the query is IN-SUPPORT of the demo cache (demo-prefix top-1 distance <= gate tau), restrict
        retrieval to the demo prefix -- the demos already cover it, and rollout entries there can only
        act as near-tie distractors (measured: executed rollout chunks differ from the nearest-demo
        chunk by ~1.6x the metric's intrinsic action resolution). Off-support queries keep the FULL
        cache (that is what the expansion is for). No-op (returns dist unchanged) when the gate is off."""
        if self.rollout_gate_tau is not None and dist[:self.n_demo].min() <= self.rollout_gate_tau:
            return dist[:self.n_demo]
        return dist

    def _record_match(self, j, dist):
        """Record the executed cache entry j (its real-frame provenance + key distance) in self.last_match,
        so the closed-loop trace can log exactly which cached frame was hit at each skip."""
        j = int(j)
        self.last_match = {"cache_idx": j, "src_ep": self.src_ep[j], "src_imgidx": self.src_imgidx[j],
                           "src_t": self.src_t[j], "dist": float(dist[j])}
        return j

    def _knn_value(self, dist) -> np.ndarray:
        """Combine the k nearest cache values for a query whose key-distance to every cache entry is
        ``dist`` (N,). Returns the executed chunk (16,7) physical and sets self.last_match to the
        executed top-1 (nearest) entry's provenance. Behaviour is selected by self.combine:

          'mean'    : top-1 if k==1, else the UNIFORM mean of the k nearest -- the legacy path, byte-
                      identical to the previous code.
          'softknn' : kernel-weighted (Nadaraya-Watson) average of the k nearest. Weights
                      w_j ∝ exp(-(d_j - d_min)/(beta * mean_k d)) trust nearer neighbours more (the
                      embedding ranks neighbours well); averaging cuts the single-draw variance of
                      top-1. The discrete gripper dim is taken from the top-1 (never averaged) unless
                      combine_grip=='mean'; combine_tau>0 first drops neighbours whose action chunk is
                      farther than tau * (median pairwise action-dist) from the top-1 (mode-guard).
          'mode'    : consensus-mode -- EXECUTE the single REAL top-k chunk at the trusted-density peak
                      (no averaging). The conditional MEAN (softknn) minimises z-RMSE to one demo but
                      blends modes and dilutes decisive actions; the conditional MODE keeps a coherent,
                      full-commitment, on-manifold demo chunk. See _mode_value.
        """
        if self.k == 1:
            j = self._record_match(dist.argmin(), dist)
            self._log_skip(dist, np.array([j]), None, None)
            return self.values[j].copy()
        idx = np.argpartition(dist, self.k)[: self.k]            # k nearest (unordered)
        near = int(np.argmin(dist[idx]))                        # local pos of the top-1 within idx
        if self.combine == "mean":
            self._record_match(idx[near], dist)                 # provenance = the executed top-1
            self._log_skip(dist, idx, None, None)
            return self.values[idx].mean(axis=0).astype(np.float32)
        if self.combine == "mode":
            return self._mode_value(dist, idx, near)            # consensus-mode (real chunk; see below)
        # ---- soft-kNN fusion ----
        self._record_match(idx[near], dist)                     # provenance = the executed top-1
        dv = dist[idx].astype(np.float64)                       # (k,) embedding distances of the shortlist
        V = self.values[idx]                                    # (k,16,7) physical candidate chunks
        keep = np.ones(len(idx), dtype=bool)
        if self.combine_tau > 0:                                # mode-guard: prune action-outlier neighbours
            af = ((V[..., :6] - self.norm.act_mean[:6]) / self.norm.act_std[:6]).reshape(len(V), -1)
            d0 = np.sqrt(((af - af[near][None]) ** 2).mean(1))  # action-dist of each neighbour to top-1
            iu = np.triu_indices(len(V), k=1)
            pair = np.sqrt(((af[:, None, :] - af[None, :, :]) ** 2).mean(2))[iu]
            r = float(np.median(pair)) + 1e-9
            keep = d0 <= self.combine_tau * r
        h = self.combine_beta * dv.mean() + 1e-9
        w = np.exp(-(dv - dv.min()) / h) * keep
        w = (w / w.sum()).astype(np.float32)
        out = (w[:, None, None] * V).sum(axis=0).astype(np.float32)   # (16,7)
        if self.combine_grip == "top1":
            out[:, 6] = V[near, :, 6]                           # keep the nearest neighbour's discrete gripper
        self._log_skip(dist, idx, out, V[near], w=w)
        return out

    def _mode_value(self, dist, idx, near):
        """Consensus-mode retrieval: among the k-nearest REAL cache chunks, execute the one at the
        trusted-density peak (the conditional MODE), as a single coherent demo chunk -- no averaging,
        so full commitment is kept (softknn's MEAN dilutes decisive actions; the MODE does not).

          w_j  = exp(-(d_j - d_min)/(beta * mean_k d))      # embedding-trust weight (encoder geometry)
          A_jl = z-scored action-chunk distance (non-grip)  # disagreement between candidates j, l
          h    = median off-diagonal A                       # per-query bandwidth
          rho_j= Σ_l w_l * exp(-½ (A_jl/h)²)                 # trusted-neighbour density at candidate j
          execute V[argmax_j rho_j]                          # the REAL chunk at the consensus peak
        """
        dv = dist[idx].astype(np.float64)                       # (k,) shortlist embedding distances
        V = self.values[idx]                                    # (k,16,7) real candidate chunks
        h = self.combine_beta * dv.mean() + 1e-9
        w = np.exp(-(dv - dv.min()) / h); w = w / w.sum()       # embedding-trust weights
        af = (V[..., :6] / self.norm.act_std[:6]).reshape(len(V), -1)   # z-scored non-grip features (mean cancels)
        A = np.sqrt(((af[:, None, :] - af[None, :, :]) ** 2).mean(-1))  # (k,k) z-RMSE action distance
        iu = np.triu_indices(len(V), k=1)
        hb = float(np.median(A[iu])) + 1e-9                     # per-query bandwidth (median pairwise)
        dens = (w[None, :] * np.exp(-0.5 * (A / hb) ** 2)).sum(axis=1)  # (k,) trusted local density
        m = int(dens.argmax())
        self._record_match(idx[m], dist)                        # provenance = the EXECUTED (mode) chunk
        self._log_skip(dist, idx, V[m].astype(np.float32), V[near], w=w)
        return V[m].copy().astype(np.float32)

    def _log_skip(self, dist, idx, sk_chunk, t1_chunk, w=None):
        """Append one per-skip diagnostic record (only when env RETR_LOG is set). Records the geometry of
        THIS live query so the online distribution can be compared to the offline test distribution:
          d_top1     : key distance to the nearest cache entry (OOD signal -- larger online => state drift)
          d_meank    : mean key distance over the k-shortlist
          n_eff      : effective #neighbours of the softknn kernel (1/Σw²; ~1 => collapsed to top1)
          chunk_diff : z-RMSE between the executed softknn chunk and the top1 chunk (continuous dims),
                       early(0-3) and late(8-15) -- how much execution actually changed at this skip
        """
        if not getattr(self, "_logpath", None):
            return
        if getattr(self, "_logf", None) is None:
            self._logf = open(f"{self._logpath}_{os.getpid()}.jsonl", "a")
        dv = dist[idx].astype(np.float64)
        rec = {"d_top1": float(dv.min()), "d_meank": float(dv.mean()), "k": int(len(idx)),
               "n_eff": (float(1.0 / np.sum(w ** 2)) if w is not None else 1.0)}
        if sk_chunk is not None and t1_chunk is not None:
            d = ((sk_chunk - t1_chunk) / self.norm.act_std)[..., :6]
            rec["chunk_diff_early"] = float(np.sqrt((d[:4] ** 2).mean()))
            rec["chunk_diff_late"] = float(np.sqrt((d[8:] ** 2).mean()))
        self._logf.write(json.dumps(rec) + "\n")
        self._logf.flush()
        self._nskip_logged += 1

    def predict_chunk(self, prev_actions, current_proprio, cached_future_proprio,
                      cached_future_img, current_image=None, current_wrist=None) -> np.ndarray:
        """Return the next-16 action chunk (physical, (16,7)) by nearest-neighbour lookup. With the consensus
        option, ``current_image``/``current_wrist`` form the live encoder key; otherwise they are unused.
        Side effect: sets self.last_match = provenance of the executed (top-1) cache entry."""
        if self.fused:  # learned multimodal embedding key (image+proprio+prev) -> nearest value
            assert current_image is not None and current_wrist is not None, \
                "fused retrieval needs live primary+wrist frames (current_image/current_wrist)"
            q = self._embed_live_fused(current_image, current_wrist, prev_actions, current_proprio)
            dist = np.linalg.norm(self.keys - q[None, :], axis=1)  # (N,) fused-embedding distances
            return self._knn_value(self._gated_dist(dist))
        q = self._query_key(prev_actions, current_proprio, cached_future_proprio, cached_future_img)
        dist = np.linalg.norm(self.keys - q[None, :], axis=1)  # (N,) N1 distances
        if self.consensus and current_image is not None and current_wrist is not None:
            return self.values[self._record_match(
                self._consensus_idx(dist, current_image, current_wrist), dist)].copy()  # top-1 (consensus)
        return self._knn_value(dist)

    def _embed_live(self, image, wrist) -> np.ndarray:
        """Encoder embedding of the live 224x224 primary+wrist frames (same preprocessing as PredictorPolicy)."""
        import torch
        p = torch.from_numpy(np.ascontiguousarray(image).astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        w = torch.from_numpy(np.ascontiguousarray(wrist).astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encoder(p, w)[0].cpu().numpy().astype(np.float32)

    def _embed_live_fused(self, image, wrist, prev_actions, proprio) -> np.ndarray:
        """Fused-encoder embedding of the live decision point: primary+wrist frames + prev_actions(16,7) +
        proprio(9). Inputs are physical (the encoder z-scores prev/proprio internally via its own buffers)."""
        import torch
        p = torch.from_numpy(np.ascontiguousarray(image).astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        w = torch.from_numpy(np.ascontiguousarray(wrist).astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        prev = torch.from_numpy(np.asarray(prev_actions, np.float32)).unsqueeze(0).to(self.device)   # (1,16,7)
        pro = torch.from_numpy(np.asarray(proprio, np.float32)).unsqueeze(0).to(self.device)          # (1,9)
        with torch.no_grad():
            return self.encoder(p, w, prev, pro)[0].cpu().numpy().astype(np.float32)

    def _consensus_idx(self, d_n1, image, wrist) -> int:
        """N1 top-K intersect encoder top-K -> the member nearest by N1; fall back to N1 top-1 if empty."""
        K = min(self.consensus_k, d_n1.shape[0])
        n1_top = np.argpartition(d_n1, K - 1)[:K]
        d_enc = np.linalg.norm(self.emb_keys - self._embed_live(image, wrist)[None, :], axis=1)
        enc_top = np.argpartition(d_enc, K - 1)[:K]
        inter = np.intersect1d(n1_top, enc_top, assume_unique=True)
        if inter.size:
            return int(inter[np.argmin(d_n1[inter])])  # N1's best among the encoder-corroborated candidates
        return int(d_n1.argmin())                       # no consensus -> plain N1 top-1


class OracleRetrievalPolicy:
    """Closed-loop ORACLE counterpart of research/retrieval_oracle/ (which queried with the
    ground-truth future). A closed-loop rollout has no ground-truth future, so at a skip the
    VLA is run and ITS predicted chunk[:16] is used as the query into a dictionary of all
    length-16 action windows from the cached episodes; the nearest window is EXECUTED.

    This is a diagnostic upper bound, not a deployable policy: it still calls the VLA at every
    skip (no compute saved). It measures whether cached action windows can behaviourally
    substitute for the VLA's own chunks -- i.e. whether the small open-loop action-RMSE gap
    survives closed-loop compounding.

    `oracle_query = True` signals closed_loop.run_closed_loop_episode to run the VLA at a skip
    and call `lookup(vla_chunk[:16])` instead of `predict_chunk(...)`.
    """

    oracle_query = True

    def __init__(self, data_dir: str, metric: str = "l2", val_frac: float = 0.15, seed: int = 0):
        assert metric in METRIC_CHOICES, f"metric must be one of {METRIC_CHOICES}"
        H = NUM_OPEN_LOOP_STEPS
        train_files, _ = split_episode_files(list_success_episodes(data_dir), val_frac, seed)
        trajs = [np.load(f, allow_pickle=True)["realized_actions"].astype(np.float32) for f in train_files]
        # All stride-1 length-H windows of executed actions = the cached "action dictionary".
        self.windows = np.concatenate(
            [np.lib.stride_tricks.sliding_window_view(r, H, axis=0).transpose(0, 2, 1)
             for r in trajs if r.shape[0] >= H], axis=0).astype(np.float32)  # (N, H, 7) physical
        act_std = np.concatenate(trajs, axis=0).std(axis=0).astype(np.float32) + 1e-6  # (7,)
        self.scale = np.ones(7, np.float32) if metric == "l2" else act_std  # per-dim divisor
        self.keys = (self.windows / self.scale).reshape(self.windows.shape[0], -1)  # (N, H*7)
        self.metric = metric

        # Attributes the runner records (drop-in with PredictorPolicy / RetrievalPolicy).
        self.run_dir = data_dir
        self.img_mode = f"retrieval_oracle:{metric}"
        self.state_source = "vla_chunk_query"

    def lookup(self, query_chunk) -> np.ndarray:
        """Nearest cached action window to the VLA's predicted chunk. query_chunk: (H,7) physical."""
        q = (np.asarray(query_chunk, np.float32) / self.scale).reshape(-1)  # (H*7,)
        dist = np.linalg.norm(self.keys - q[None, :], axis=1)  # (N,)
        return self.windows[int(dist.argmin())].copy()  # (H,7)


class FusionRetrievalPolicy:
    """Predictor-directed retrieval ("retrieval offers options, the predictor chooses"): at a skip, run the
    local predictor to get its action-chunk estimate p, take the K state-nearest cache chunks (the SAME N1
    dictionary RetrievalPolicy uses), and EXECUTE the candidate chunk closest to p. This keeps a real,
    coherent demo chunk (no averaging) while letting the predictor select the right mode among the options.

    Drop-in `predict_chunk(...)` policy -- identical interface to PredictorPolicy / RetrievalPolicy -- so it
    composes with ANY skip gate (e.g. `--skip-policy random`) with NO closed-loop changes. Ported from
    research/skip_v2 (FusionPolicy) but DECOUPLED from the fusion agreement gate: the executed-chunk selection
    lives entirely in predict_chunk and carries no gating, so the gate is whatever SkipPolicy it is paired with.

    Composition (not inheritance) keeps each piece deployable on its own: a PredictorPolicy (the chooser) and
    a RetrievalPolicy (the dictionary + key construction), reused verbatim so the cache/key matches the plain
    N1 retrieval exactly.
    """

    def __init__(self, run_dir: str, data_dir: str, K: int = 10, key: str = "prev_state",
                 state_source: str = DEFAULT_STATE_SOURCE, cache_episodes: int | None = None,
                 val_frac: float = 0.15, seed: int = 0):
        from predictor_policy import PredictorPolicy
        self.predictor = PredictorPolicy(run_dir)
        self.retr = RetrievalPolicy(data_dir, key=key, k=1, state_source=state_source,
                                    cache_episodes=cache_episodes, val_frac=val_frac, seed=seed)
        self.K = max(1, int(K))

        # Attributes the runner records (drop-in with PredictorPolicy / RetrievalPolicy).
        self.run_dir = run_dir
        self.img_mode = f"fusion:{self.predictor.img_mode}+{self.retr.img_mode}:K{self.K}"
        self.state_source = self.predictor.state_source

    def predict_chunk(self, prev_actions, current_proprio, cached_future_proprio,
                      cached_future_img, current_image=None, current_wrist=None) -> np.ndarray:
        """Execute the K-nearest cache chunk closest to the predictor's output. Returns (16,7) physical.
        Live frames are forwarded to the predictor (used only if it is an obs-emb predictor)."""
        p = self.predictor.predict_chunk(prev_actions, current_proprio, cached_future_proprio,
                                         cached_future_img, current_image=current_image, current_wrist=current_wrist)
        q = self.retr._query_key(prev_actions, current_proprio, cached_future_proprio, cached_future_img)
        dist = np.linalg.norm(self.retr.keys - q[None, :], axis=1)  # (N,) state-space distance
        K = min(self.K, dist.shape[0])
        topk = np.argpartition(dist, K - 1)[:K]                     # K nearest (unordered)
        cand = self.retr.values[topk]                              # (K,16,7) real coherent options
        dp = np.sqrt(((cand - p[None]) ** 2).reshape(K, -1).mean(1))  # each candidate's mean dist to p
        return cand[int(dp.argmin())].copy()                       # predictor-chosen, coherent chunk


if __name__ == "__main__":  # smoke test: build a DB and run one lookup (no simulator needed)
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data/pnp_counter_to_stove_dense")
    ap.add_argument("--key", default="prev_state", choices=KEY_CHOICES)
    ap.add_argument("--knn", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    pol = RetrievalPolicy(args.data_dir, key=args.key, k=args.knn)
    print(f"[realistic] DB: {pol.keys.shape[0]} entries, key dim={pol.keys.shape[1]} "
          f"({args.key}, k={args.knn}) in {time.time() - t0:.1f}s")
    out = pol.predict_chunk(
        prev_actions=np.zeros((16, 7), np.float32), current_proprio=np.zeros(9, np.float32),
        cached_future_proprio=np.zeros(9, np.float32),
        cached_future_img=np.zeros((3, 16, 28, 28), np.float32),
    )
    print(f"[realistic] predict_chunk -> shape {out.shape}, dtype {out.dtype}  (expect (16, 7) float32)")

    for metric in METRIC_CHOICES:
        t0 = time.time()
        ora = OracleRetrievalPolicy(args.data_dir, metric=metric)
        ret = ora.lookup(np.zeros((16, 7), np.float32))
        print(f"[oracle:{metric}] DB: {ora.windows.shape[0]} windows, key dim={ora.keys.shape[1]} "
              f"in {time.time() - t0:.1f}s | lookup -> shape {ret.shape}  (expect (16, 7))")
