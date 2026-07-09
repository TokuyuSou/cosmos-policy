"""Closed-loop retrieval policy for the Transition Metric Transformer (TMT) encoder.

Drop-in ``predict_chunk(...)`` policy (same surface as RetrievalPolicy): the TMT embedding REPLACES the
retrieval key -- top-1 nearest cache chunk under the learned transition metric is executed. Composes
RetrievalPolicy VERBATIM for the cache values/provenance so the executed chunks and match logging are
byte-identical machinery; only the KEY changes.

TMT needs the transition triple, so this policy declares ``needs_prev_frame = True`` and the closed-loop
runner passes the PREVIOUS gate decision's frames (16 executed steps back -- exactly the training-time
row spacing). First decision of an episode has none -> falls back to the current frames (zero-change),
matching training's episode-start fallback.

KEY (cache) side is deploy-faithful to offline training: per cache row, current + prev-row Theia tokens
(prev row = same episode, src_t-16; fallback self), plus EFFECT tokens from the post row (src_t+16;
missing -> zeroed delta). All norm stats come baked from the checkpoint buffers. Tokens and key
embeddings are cached to disk, signature-checked (sample count + image bytes + ckpt mtime).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os

import numpy as np
import torch

from dataset import build_samples, list_success_episodes, split_episode_files
from retrieval_policy import RetrievalPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
R3M = os.path.join(HERE, "..", "r3m_action_encoder")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TMTEncoder = _load_module("tmt_model_policy", os.path.join(R3M, "model_tmt.py")).TMTEncoder
TheiaBackbone = _load_module("r3m_metric_model_tmt", os.path.join(R3M, "model.py")).TheiaBackbone


@torch.no_grad()
def theia_grid_tokens(bb, primary_chw, wrist_chw, grid):
    """Frozen-Theia COARSE-SPATIAL scene tokens: each view's 14x14 token grid is adaptive-avg-pooled to
    grid x grid, then primary|wrist are concatenated. ``primary_chw``/``wrist_chw``: (B,3,224,224)
    uint8/float[0,255] on ``bb``'s device -> (B, 2*grid*grid, fd).

    The two views are run through the frozen ViT in a SINGLE batched forward (stacked to (2B,...) then
    split) instead of two sequential passes: the ViT is LayerNorm-only + frozen/eval, so a batched pass is
    numerically identical to per-view passes but launches one kernel set instead of two. Falls back to
    per-view passes if the two views differ in spatial shape."""
    g2 = grid * grid
    if primary_chw.shape == wrist_chw.shape:                # common case: batch both views in ONE ViT forward
        b = primary_chw.shape[0]
        t = bb.tokens(torch.cat([primary_chw, wrist_chw], dim=0))   # (2B,196,fd)
        views = (t[:b], t[b:])
    else:                                                   # differing view shapes -> keep separate passes
        views = (bb.tokens(primary_chw), bb.tokens(wrist_chw))
    outs = []
    for t in views:
        b, n, c = t.shape
        hw = int(round(n ** 0.5))
        t = t.transpose(1, 2).reshape(b, c, hw, hw)         # (B,fd,14,14)
        t = torch.nn.functional.adaptive_avg_pool2d(t, grid)
        outs.append(t.reshape(b, c, g2).transpose(1, 2))    # (B,g2,fd)
    return torch.cat(outs, dim=1)                           # (B, 2*g2, fd)  (primary then wrist -> view_ids 0/1)


def _to_chw(hwc_uint8):
    """(H,W,3) uint8 -> (1,3,H,W) uint8 tensor."""
    return torch.from_numpy(np.ascontiguousarray(hwc_uint8).astype(np.uint8)).permute(2, 0, 1).unsqueeze(0)


def load_tmt(ckpt_path, device):
    """Build a TMTEncoder with the architecture INFERRED from the checkpoint shapes (nhead, not inferable,
    from the sibling metrics json; default 8) and load it; frozen + eval. Returns (model, g_img, g_eff, fd)."""
    sd = torch.load(ckpt_path, map_location=device)
    fd = int(sd["img_proj.weight"].shape[1])
    d_model = int(sd["img_proj.weight"].shape[0])
    half_dim = int(sd["out_proj.weight"].shape[0])
    n_img = int(sd["img_pos"].shape[1])
    n_eff = int(sd["eff_pos"].shape[1])
    prev_steps = int(sd["act_pos"].shape[1])
    act_dim = int(sd["act_proj.weight"].shape[1])
    proprio_dim = int(sd["pro_proj.weight"].shape[1])
    ffn = int(sd["enc.layers.0.linear1.weight"].shape[0])
    layers = len({k.split(".")[2] for k in sd if k.startswith("enc.layers.")})
    nhead = 8
    meta = os.path.join(os.path.dirname(ckpt_path),
                        "metrics_" + os.path.basename(ckpt_path).replace(".pt", ".json"))
    if os.path.exists(meta):
        nhead = int(json.load(open(meta)).get("args", {}).get("nhead", nhead))
    m = TMTEncoder(fd, n_img, n_eff, act_dim=act_dim, prev_steps=prev_steps, proprio_dim=proprio_dim,
                   d_model=d_model, nhead=nhead, layers=layers, ffn=ffn, dropout=0.0,
                   half_dim=half_dim, eff_scene="effscene_head.weight" in sd).to(device)
    m.load_state_dict(sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    g_img = int(round((n_img / 2) ** 0.5))
    g_eff = int(round((n_eff / 2) ** 0.5))
    assert 2 * g_img ** 2 == n_img and 2 * g_eff ** 2 == n_eff, (n_img, n_eff)
    return m, g_img, g_eff, fd


class TMTRetrievalPolicy:
    needs_prev_frame = True    # closed_loop threads the previous gate decision's frames into predict_chunk

    def __init__(self, data_dir: str, tmt_ckpt: str, state_source: str = "actual_next_proprio",
                 cache_episodes: int | None = None, val_frac: float = 0.15, seed: int = 0,
                 w: float = -1.0):
        """``w`` > 0 overrides the checkpoint's learned block weight (fraction of squared retrieval
        distance carried by the transformer block). The learned w optimizes the KD softmax, not top-1;
        a val-calibrated w (e.g. 0.5) is systematically better for retrieval. <=0 = keep learned."""
        assert state_source == "actual_next_proprio", (
            "tmt policy requires --state-source actual_next_proprio (trained on decision-point proprio)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_source = state_source

        # cache values + provenance machinery, byte-identical to plain retrieval (key goes unused)
        self.retr = RetrievalPolicy(data_dir, key="prev_state", k=1, state_source=state_source,
                                    cache_episodes=cache_episodes, val_frac=val_frac, seed=seed)

        self.model, self.g_img, self.g_eff, self.fd = load_tmt(tmt_ckpt, self.device)
        self.w = float(w)
        if self.w > 0:
            assert 0.0 < self.w < 1.0, f"--tmt-w must be in (0,1), got {w}"
            with torch.no_grad():
                self.model.w_logit.fill_(math.log(self.w / (1.0 - self.w)))
        self.bb = TheiaBackbone(TheiaBackbone.MODEL_BY_DIM.get(self.fd, TheiaBackbone.DEFAULT_MODEL),
                                freeze="frozen").to(self.device).eval()

        # parallel cache WITH images, same files/order as self.retr (alignment contract)
        data_dirs = [d for d in str(data_dir).split(",") if d]
        cache_files = []
        for d in data_dirs:
            fs = list_success_episodes(d)
            cache_files += (fs[:cache_episodes] if cache_episodes else split_episode_files(fs, val_frac, seed)[0])
        samples = build_samples(cache_files, [0], [state_source], with_image=True)
        assert len(samples) == len(self.retr.values), \
            f"cache misalignment vs RetrievalPolicy ({len(samples)} vs {len(self.retr.values)})"
        assert [s.src_ep for s in samples] == self.retr.src_ep and [s.src_t for s in samples] == self.retr.src_t, \
            "cache order mismatch vs RetrievalPolicy"

        # transition/effect row maps by (episode, env-step): prev row = src_t-16 (fallback self = zero
        # change), post row = src_t+16 (missing -> -1 = zeroed effect). 16 env steps == the executed chunk
        # == the offline training spacing (dense rows are 4 steps apart, ROWS_PER_CHUNK=4).
        pos = {(s.src_ep, s.src_t): i for i, s in enumerate(samples)}
        self.prev_i = np.array([pos.get((s.src_ep, s.src_t - 16), i) for i, s in enumerate(samples)])
        self.post_i = np.array([pos.get((s.src_ep, s.src_t + 16), -1) for i, s in enumerate(samples)])

        et = os.path.splitext(os.path.basename(tmt_ckpt))[0] + (f"_w{self.w:g}" if self.w > 0 else "")
        dn = (os.path.basename(os.path.normpath(data_dirs[0])) if len(data_dirs) == 1 else
              "mix_" + hashlib.md5(",".join(data_dirs).encode()).hexdigest()[:8])
        cdir = os.path.join(HERE, "cache")
        tok_img = self._tokens(samples, self.g_img, os.path.join(cdir, f"tmt_tok_g{self.g_img}_{dn}.npz"))
        tok_eff = (tok_img if self.g_eff == self.g_img else
                   self._tokens(samples, self.g_eff, os.path.join(cdir, f"tmt_tok_g{self.g_eff}_{dn}.npz")))
        self.keys = self._key_embeddings(samples, tok_img, tok_eff,
                                         os.path.join(cdir, f"tmt_keys_{dn}_{et}.npz"), tmt_ckpt)

        self.run_dir = tmt_ckpt
        self.img_mode = f"tmt[{et}]:g{self.g_img}/e{self.g_eff}:top1"
        self.last_match = None

    # ------------------------------------------------------------------ cache-side precompute
    @torch.no_grad()
    def _tokens(self, samples, grid, cache_path, bs=128):
        sig = hashlib.md5(f"{len(samples)}|g{grid}|{self.fd}".encode()
                          + samples[0].image.tobytes()[:4096] + samples[-1].image.tobytes()[:4096]).hexdigest()[:12]
        if os.path.exists(cache_path):
            z = np.load(cache_path)
            if str(z["sig"]) == sig and len(z["tok"]) == len(samples):
                print(f"[tmt] loaded token cache {cache_path} ({len(samples)} samples)")
                return z["tok"].astype(np.float32)
        out = np.zeros((len(samples), 2 * grid * grid, self.fd), np.float32)
        for i in range(0, len(samples), bs):
            ch = samples[i:i + bs]
            p = torch.from_numpy(np.stack([s.image for s in ch]).astype(np.uint8)).permute(0, 3, 1, 2).to(self.device)
            w = torch.from_numpy(np.stack([s.wrist for s in ch]).astype(np.uint8)).permute(0, 3, 1, 2).to(self.device)
            out[i:i + bs] = theia_grid_tokens(self.bb, p, w, grid).cpu().numpy()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, tok=out, sig=sig)
        print(f"[tmt] computed + cached tokens {cache_path} {out.shape}")
        return out

    @torch.no_grad()
    def _key_embeddings(self, samples, tok_img, tok_eff, cache_path, ckpt_path, bs=256):
        sig = hashlib.md5(f"{len(samples)}|{os.path.getmtime(ckpt_path)}|{ckpt_path}".encode()
                          + samples[0].image.tobytes()[:4096] + samples[-1].image.tobytes()[:4096]).hexdigest()[:12]
        if os.path.exists(cache_path):
            z = np.load(cache_path)
            if str(z["sig"]) == sig and len(z["emb"]) == len(samples):
                print(f"[tmt] loaded key cache {cache_path} ({len(samples)} keys)")
                return z["emb"].astype(np.float32)
        prev = np.stack([s.prev_actions for s in samples]).astype(np.float32)
        pro = np.stack([s.states[self.state_source] for s in samples]).astype(np.float32)
        ok = self.post_i >= 0
        p0 = np.where(ok, self.post_i, np.arange(len(samples)))
        eff_s = (tok_eff[p0] - tok_eff) * ok[:, None, None]
        eff_p = (pro[p0] - pro) * ok[:, None]
        out = []
        for i in range(0, len(samples), bs):
            sl = slice(i, min(i + bs, len(samples)))
            t = lambda a: torch.as_tensor(a[sl], dtype=torch.float32, device=self.device)
            z, _ = self.model.embed(t(tok_img), torch.as_tensor(tok_img[self.prev_i[sl]], dtype=torch.float32,
                                                                device=self.device),
                                    t(prev), t(pro), eff_s=t(eff_s), eff_p=t(eff_p))
            out.append(z.float().cpu().numpy())
        emb = np.concatenate(out).astype(np.float32)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, emb=emb, sig=sig)
        print(f"[tmt] computed + cached keys {cache_path} {emb.shape}")
        return emb

    # ------------------------------------------------------------------ live query
    def reset(self):
        pass  # stateless: prev frames come from the runner

    @torch.no_grad()
    def _query_tokens(self, current_image, current_wrist, prev_image=None, prev_wrist=None):
        """Frozen-Theia g_img tokens for the query's current (+ previous) frame -> (cur_tok, prv_tok), each
        (1, n_img, fd). BOTH frames' 4 images (cur primary|wrist, prev primary|wrist) go through the ViT in a
        SINGLE batched forward: the trunk is frozen/eval + LayerNorm-only, so this is numerically identical to
        separate passes but replaces 4 batch-1 launches with one batch-4 launch (~4x less Theia time -- the
        dominant deploy cost). Prev-frame fallback = current frames (training's episode-start convention)."""
        if prev_image is not None and prev_wrist is not None:
            p = torch.cat([_to_chw(current_image), _to_chw(prev_image)], dim=0).to(self.device)  # (2,3,H,W)
            w = torch.cat([_to_chw(current_wrist), _to_chw(prev_wrist)], dim=0).to(self.device)
            tok = theia_grid_tokens(self.bb, p, w, self.g_img)             # (2,n_img,fd) -- ONE ViT forward, 4 imgs
            return tok[:1], tok[1:2]
        cur = theia_grid_tokens(self.bb, _to_chw(current_image).to(self.device),
                                _to_chw(current_wrist).to(self.device), self.g_img)            # (1,n_img,fd)
        return cur, cur                                        # first decision: zero-change fallback

    @torch.no_grad()
    def _dists_from_tokens(self, cur, prv, prev_actions, current_proprio) -> np.ndarray:
        """(N,) distances from the LIVE query embedding (built from precomputed Theia tokens) to all cache
        keys (QUERY mode, no effect input)."""
        pa = torch.from_numpy(np.asarray(prev_actions, np.float32))[None].to(self.device)
        pr = torch.from_numpy(np.asarray(current_proprio, np.float32))[None].to(self.device)
        z, _ = self.model.embed(cur, prv, pa, pr)              # QUERY mode (no effect input)
        q = z[0].float().cpu().numpy()
        return np.linalg.norm(self.keys - q[None, :], axis=1)

    @torch.no_grad()
    def _query_dists(self, current_image, current_wrist, prev_actions, current_proprio,
                     prev_image=None, prev_wrist=None) -> np.ndarray:
        """(N,) distances from the LIVE query embedding to all cache keys (QUERY mode; prev-frame
        fallback = current frames, matching training's episode-start convention)."""
        cur, prv = self._query_tokens(current_image, current_wrist, prev_image, prev_wrist)
        return self._dists_from_tokens(cur, prv, prev_actions, current_proprio)

    @torch.no_grad()
    def predict_chunk(self, prev_actions, current_proprio, cached_future_proprio, cached_future_img,
                      current_image=None, current_wrist=None, prev_image=None, prev_wrist=None) -> np.ndarray:
        assert current_image is not None and current_wrist is not None, \
            "tmt policy needs live primary+wrist frames"
        dist = self._query_dists(current_image, current_wrist, prev_actions, current_proprio,
                                 prev_image, prev_wrist)
        best = int(np.argmin(dist))
        self.retr._record_match(best, dist)
        self.last_match = self.retr.last_match
        return self.retr.values[best].copy()


if __name__ == "__main__":  # smoke test: build on a real cache + one lookup (no simulator needed)
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data/pnp_sink_to_counter_dense_img")
    ap.add_argument("--tmt", default="../r3m_action_encoder/results/pnp_sink_to_counter/tmt_v1.pt")
    args = ap.parse_args()
    t0 = time.time()
    pol = TMTRetrievalPolicy(args.data_dir, args.tmt)
    print(f"[tmt] cache {pol.keys.shape} | g_img {pol.g_img} g_eff {pol.g_eff} fd {pol.fd} "
          f"| prev-fallback {int((pol.prev_i == np.arange(len(pol.prev_i))).sum())} rows "
          f"| no-post {int((pol.post_i < 0).sum())} rows | built in {time.time()-t0:.0f}s")
    z = np.zeros((224, 224, 3), np.uint8)
    for tag, pimg in (("first-decision (no prev)", None), ("with prev frame", z)):
        out = pol.predict_chunk(np.zeros((16, 7), np.float32), np.zeros(9, np.float32),
                                np.zeros(9, np.float32), np.zeros((3, 16, 28, 28), np.float32),
                                current_image=z, current_wrist=z, prev_image=pimg, prev_wrist=pimg)
        print(f"[tmt] {tag}: -> {out.shape} {out.dtype} match={pol.last_match}")
