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
                 consensus_encoder: str = "", consensus_k: int = 50, fused_encoder: str = ""):
        assert key in KEY_CHOICES, f"key must be one of {KEY_CHOICES}"
        assert k >= 1
        self.key, self.k, self.state_source = key, int(k), state_source
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
        if self.fused:
            assert state_source == "actual_next_proprio", (
                "fused_encoder requires --state-source actual_next_proprio "
                "(the encoder was trained on the ACTUAL decision-point proprio)")

        # Cache = same train split as the action-predictor eval (default), OR the first
        # `cache_episodes` success episodes when set (to match a gate built on that same cache).
        need_img = self.consensus or self.fused
        files = list_success_episodes(data_dir)
        cache_files = files[:cache_episodes] if cache_episodes else split_episode_files(files, val_frac, seed)[0]
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
            import os
            import torch
            from obs_embed import compute_fused_emb, load_fused_encoder
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = load_fused_encoder(fused_encoder, self.device)
            cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            dn = os.path.basename(os.path.normpath(data_dir))
            et = os.path.splitext(os.path.basename(fused_encoder))[0]
            self.keys = compute_fused_emb(samples, self.encoder, self.device, state_source,
                                          os.path.join(cdir, f"fusedemb_{dn}_{et}_retrcache.npz")).astype(np.float32)
            self.img_mode = f"fused[{et}]:k{k}"
            return

        self.keys = np.stack([self._key(s.prev_actions, s.states[state_source], s.future_img)
                              for s in samples]).astype(np.float32)  # (N, D)
        self.img_mode = f"retrieval:{key}:k{k}"
        if self.consensus:  # build the encoder cache keys (kept loaded for the live query at each skip)
            import os
            import torch
            from obs_embed import attach_obs_emb, load_corr_encoder
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = load_corr_encoder(consensus_encoder, self.device)
            cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            dn, et = os.path.basename(os.path.normpath(data_dir)), os.path.splitext(os.path.basename(consensus_encoder))[0]
            attach_obs_emb(samples, self.encoder, self.device, os.path.join(cdir, f"obsemb_{dn}_{et}_retrcache.npz"))
            self.emb_keys = np.stack([s.obs_emb for s in samples]).astype(np.float32)  # (N, D_emb) encoder cache keys
            self.img_mode = f"consensus[{et}:K{self.consensus_k}]+{self.img_mode}"

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

    def _record_match(self, j, dist):
        """Record the executed cache entry j (its real-frame provenance + key distance) in self.last_match,
        so the closed-loop trace can log exactly which cached frame was hit at each skip."""
        j = int(j)
        self.last_match = {"cache_idx": j, "src_ep": self.src_ep[j], "src_imgidx": self.src_imgidx[j],
                           "src_t": self.src_t[j], "dist": float(dist[j])}
        return j

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
            if self.k == 1:
                return self.values[self._record_match(dist.argmin(), dist)].copy()
            idx = np.argpartition(dist, self.k)[: self.k]
            self._record_match(idx[np.argmin(dist[idx])], dist)  # top-1 among the averaged k (for provenance)
            return self.values[idx].mean(axis=0).astype(np.float32)
        q = self._query_key(prev_actions, current_proprio, cached_future_proprio, cached_future_img)
        dist = np.linalg.norm(self.keys - q[None, :], axis=1)  # (N,) N1 distances
        if self.consensus and current_image is not None and current_wrist is not None:
            return self.values[self._record_match(
                self._consensus_idx(dist, current_image, current_wrist), dist)].copy()  # top-1 (consensus)
        if self.k == 1:
            return self.values[self._record_match(dist.argmin(), dist)].copy()
        idx = np.argpartition(dist, self.k)[: self.k]  # k nearest (unordered)
        self._record_match(idx[np.argmin(dist[idx])], dist)
        return self.values[idx].mean(axis=0).astype(np.float32)

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
