"""Attach precomputed observation embeddings (the R3M action-metric encoder) to predictor samples.

The encoder (research/r3m_action_encoder, `encoder_corr.pt`) maps each decision-point frame pair
(primary `cur_image` + wrist `cur_wrist_image`) to a 128-d L2-normalized vector whose distances
match executed-action-chunk distances. We run it once over the samples (with image+wrist attached
via `build_samples(..., with_image=True)`), cache the result, and store it on each `Sample.obs_emb`
so the predictor can consume it as a frozen input token.

The encoder stays FROZEN here: it was already trained for the action-metric objective, and freezing
keeps the none-vs-obs comparison a clean isolation of the embedding's value.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R3M_MODEL_PY = os.path.join(HERE, "..", "r3m_action_encoder", "model.py")


def _load_metric_model_module():
    """Load r3m_action_encoder/model.py under a unique module name (avoids clashing with
    action_predictor's own `model` module)."""
    spec = importlib.util.spec_from_file_location("r3m_metric_model", R3M_MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_metric_model = _load_metric_model_module()
ActionMetricEncoder = _metric_model.ActionMetricEncoder
TheiaBackbone = _metric_model.TheiaBackbone


def load_corr_encoder(ckpt_path, device, backbone="resnet18", out_dim=128):
    """Build the ActionMetricEncoder and load the trained (corr) weights; frozen + eval.

    Built with ``share_backbone=False`` so BOTH shared and separate-backbone checkpoints load
    correctly: a shared checkpoint stores identical primary/wrist weights (its state_dict carries
    both prefixes), so loading it into a separate model reproduces the shared model's outputs exactly
    (verified bit-identical); a separate checkpoint loads its distinct per-view weights as intended.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    # Infer the image-backbone family from the keys (Theia stores its ViT under `*_bb.vit.*`); R3M
    # checkpoints have no such key and load exactly as before.
    if any(k.startswith("primary_bb.vit.") for k in sd):
        feat_dim = int(sd["primary_bb.vit.embeddings.patch_embeddings.projection.weight"].shape[0])
        img_backbone = "theia"
        theia_model = TheiaBackbone.MODEL_BY_DIM.get(feat_dim, TheiaBackbone.DEFAULT_MODEL)
    else:
        img_backbone, theia_model = "r3m", None
    theia_reduce = "attn" if "primary_bb.pool.q" in sd else "mean"
    attn_queries = int(sd["primary_bb.pool.q"].shape[0]) if theia_reduce == "attn" else 4
    enc = ActionMetricEncoder(backbone, init="r3m", freeze="frozen", out_dim=out_dim,
                              share_backbone=False, img_backbone=img_backbone,
                              theia_model=theia_model, theia_reduce=theia_reduce,
                              attn_queries=attn_queries).to(device)
    enc.load_state_dict(sd)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def _load_state_encoder(mod, sd, device):
    """Build a StateActionMetricEncoder (prev+proprio only, no image) from its state_dict shapes and load
    it. Frozen + eval. Dispatched from ``load_fused_encoder`` when the ckpt has no image branch."""
    Enc = mod.StateActionMetricEncoder
    state_dim = int(sd["state_ln.weight"].shape[0])
    hidden = int(sd["head.0.weight"].shape[0])        # head = Linear(state_dim,hidden)->GELU->Dropout->Linear(hidden,out)
    out_dim = int(sd["head.3.weight"].shape[0])
    state_mode = ("recency" if "state_enc.pool_q" in sd
                  else "temporal" if "state_enc.tok.weight" in sd else "mlp")
    act_dim = int(sd["state_enc.act_mean"].shape[0])
    proprio_dim = int(sd["state_enc.pro_mean"].shape[0])
    prev_steps = (int(sd["state_enc.pool_bias"].shape[0]) if state_mode == "recency"
                  else int(sd["state_enc.pos"].shape[1]) - 1 if state_mode == "temporal"
                  else (int(sd["state_enc.net.0.weight"].shape[1]) - proprio_dim) // act_dim)
    enc = Enc(out_dim=out_dim, state_dim=state_dim, hidden=hidden, state_mode=state_mode,
              prev_steps=prev_steps, act_dim=act_dim, proprio_dim=proprio_dim).to(device)
    enc.load_state_dict(sd)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_fused_encoder(ckpt_path, device):
    """Load the multimodal FUSED encoder (image + proprio + prev -> one embedding) from a checkpoint,
    with its architecture INFERRED from the state_dict shapes (robust to any fused variant: residual or
    concat fusion, mlp or temporal state encoder, any widths). Frozen + eval.

    Built with ``share_backbone=False`` for the same reason as ``load_corr_encoder``: a shared-backbone
    checkpoint stores identical primary/wrist weights under both prefixes, so loading into a separate
    model reproduces the shared output exactly; a separate checkpoint loads its per-view weights.
    """
    spec = importlib.util.spec_from_file_location("r3m_metric_model", R3M_MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Enc = mod.MultiModalActionMetricEncoder
    sd = torch.load(ckpt_path, map_location="cpu")
    # STATE-ONLY encoder (no image branch) has no `img_ln` -> dispatch to its loader. It shares the same
    # (primary, wrist, prev, proprio) forward signature (images ignored), so the rest of the deploy path
    # (compute_fused_emb / fit_encoder_norm / live query) consumes it unchanged.
    if "img_ln.weight" not in sd:
        return _load_state_encoder(mod, sd, device)
    # OBS-STATE encoder (image + proprio, NO prev): IMAGE-base residual with a dynamic gate. Identified by
    # `pro_proj` (its proprio-correction projection); the fused-residual encoder uses `state_proj` instead.
    # Drop-in: same (primary, wrist, prev, proprio) forward with prev IGNORED, so compute_fused_emb /
    # fit_encoder_norm / the live query consume it unchanged. Dims inferred from the state_dict shapes.
    if "pro_proj.weight" in sd:
        out_dim = int(sd["img_proj.weight"].shape[0])
        img_dim = int(sd["img_ln.weight"].shape[0])
        state_dim = int(sd["state_ln.weight"].shape[0])
        act_dim = int(sd["state_enc.act_mean"].shape[0])
        proprio_dim = int(sd["state_enc.pro_mean"].shape[0])
        gate_dyn = "gate_net.0.weight" in sd
        gate_rank = int(sd["gate_net.0.weight"].shape[0]) if gate_dyn else 16
        enc = mod.ObsStateActionMetricEncoder(
            backbone="resnet18", init="r3m", freeze="frozen", out_dim=out_dim, img_dim=img_dim,
            state_dim=state_dim, share_backbone=False, mod_dropout=0.0, proprio_dim=proprio_dim,
            act_dim=act_dim, gate_dyn=gate_dyn, gate_rank=gate_rank).to(device)
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, \
            f"unexpected/missing keys loading obsstate encoder: missing={missing[:4]} unexpected={unexpected[:4]}"
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        return enc
    # ResidualImageEncoder nests its frozen base under `state.*`; its closed-loop forward needs the same
    # init-frame threading as the init-diff encoders -> not wired yet.
    if any(k.startswith("state.") for k in sd):
        raise NotImplementedError(
            "residual-image encoder (frozen state base + image residual) — closed-loop loading not wired yet.")
    # Image-backbone family is INFERRED from the tensor keys (no metadata): Theia stores its inner ViT
    # under `*_bb.vit.*`; an R3M/ResNet ckpt has no such key, so old checkpoints take the r3m path
    # unchanged. For Theia the feat_dim (-> which tiny/small/base variant) is read from the patch-embed.
    if any(k.startswith("primary_bb.vit.") for k in sd):
        feat_dim = int(sd["primary_bb.vit.embeddings.patch_embeddings.projection.weight"].shape[0])
        img_backbone = "theia"
        theia_model = mod.TheiaBackbone.MODEL_BY_DIM.get(feat_dim, mod.TheiaBackbone.DEFAULT_MODEL)
    else:
        feat_dim, img_backbone, theia_model = 512, "r3m", None   # R3M ResNet-18 backbone feat_dim
    # Init-diff encoders need the episode-initial frame, which compute_fused_emb does not yet supply -> fail
    # loudly rather than silently mis-load. (Offline eval goes through train.py, not this path.) Version A
    # (InitDiffFusedEncoder) widens img_head to (2+ndiff)*feat_dim; Version B (TokenChangeFusedEncoder) keeps
    # 2*feat_dim but adds a `change_pool` module.
    n_img_streams = int(sd["img_head.0.weight"].shape[0]) // feat_dim
    if n_img_streams != 2 or any(k.startswith("change_pool.") for k in sd):
        raise NotImplementedError(
            "init-diff fused encoder (Version A/B) — closed-loop loading is not wired yet "
            "(needs the episode-initial frame threaded into compute_fused_emb).")
    # token aggregation: an attn-pool head stores `*_bb.pool.q` (n_query, feat_dim); absent => mean/max
    theia_reduce = "attn" if "primary_bb.pool.q" in sd else "mean"
    attn_queries = int(sd["primary_bb.pool.q"].shape[0]) if theia_reduce == "attn" else 4
    fusion = "residual" if "gate" in sd else "concat"
    state_mode = ("recency" if "state_enc.pool_q" in sd
                  else "temporal" if "state_enc.tok.weight" in sd else "mlp")
    img_dim = int(sd["img_ln.weight"].shape[0])
    state_dim = int(sd["state_ln.weight"].shape[0])
    out_dim = int(sd["state_proj.weight"].shape[0] if fusion == "residual" else sd["fuse.3.weight"].shape[0])
    hidden = int(sd["fuse.0.weight"].shape[0]) if fusion == "concat" else 256
    act_dim = int(sd["state_enc.act_mean"].shape[0])
    proprio_dim = int(sd["state_enc.pro_mean"].shape[0])
    prev_steps = (int(sd["state_enc.pool_bias"].shape[0]) if state_mode == "recency"
                  else int(sd["state_enc.pos"].shape[1]) - 1 if state_mode == "temporal"
                  else (int(sd["state_enc.net.0.weight"].shape[1]) - proprio_dim) // act_dim)
    gate_dyn = any(k.startswith("gate_net.") for k in sd)  # input-dependent gate present?
    gate_rank = int(sd["gate_net.0.weight"].shape[0]) if gate_dyn else 64
    enc = Enc(backbone="resnet18", init="r3m", freeze="frozen", out_dim=out_dim, img_dim=img_dim,
              state_dim=state_dim, hidden=hidden, fusion=fusion, state_mode=state_mode, mod_dropout=0.0,
              share_backbone=False, prev_steps=prev_steps, act_dim=act_dim, proprio_dim=proprio_dim,
              gate_dyn=gate_dyn, gate_rank=gate_rank,
              img_backbone=img_backbone, theia_model=theia_model,
              theia_reduce=theia_reduce, attn_queries=attn_queries).to(device)
    # A ProgressActionMetricEncoder checkpoint carries extra `prog_head.*` weights used only at training
    # time; deploy needs only the embedding (forward is identical), so load non-strict and drop them.
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not missing and all(k.startswith("prog_head") for k in unexpected), \
        f"unexpected/missing keys loading fused encoder: missing={missing[:4]} unexpected={unexpected[:4]}"
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def _chw(hwc_uint8_list):
    """list of (H,W,3) uint8 -> (B,3,H,W) uint8 tensor."""
    arr = np.stack(hwc_uint8_list).astype(np.uint8)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def attach_obs_emb(samples, encoder, device, cache_path=None, bs=512):
    """Compute (or load cached) obs embeddings for each sample and set `sample.obs_emb`.

    Requires samples built with `with_image=True` (so `image`/`wrist` are populated). The cache is
    keyed by sample count + a hash of the first/last image bytes, so a stale cache cannot be reused
    for a different sample set.
    """
    assert all(s.image is not None and s.wrist is not None for s in samples), \
        "samples must be built with with_image=True to attach obs embeddings"

    sig = hashlib.md5(
        f"{len(samples)}".encode()
        + samples[0].image.tobytes()[:4096] + samples[-1].image.tobytes()[:4096]
    ).hexdigest()[:12]
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        if z["sig"] == sig and len(z["emb"]) == len(samples):
            for s, e in zip(samples, z["emb"]):
                s.obs_emb = e.astype(np.float32)
            print(f"[obs_emb] loaded cache {cache_path} ({len(samples)} samples)")
            return z["emb"].shape[1]

    embs = []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        p = _chw([s.image for s in chunk]).to(device)
        w = _chw([s.wrist for s in chunk]).to(device)
        embs.append(encoder(p, w).cpu().numpy().astype(np.float32))
    emb = np.concatenate(embs)
    for s, e in zip(samples, emb):
        s.obs_emb = e
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, emb=emb, sig=sig)
        print(f"[obs_emb] computed + cached {cache_path} ({len(samples)} samples, dim {emb.shape[1]})")
    return emb.shape[1]


def fit_encoder_norm(samples, state_source):
    """Compute the fused encoder's input z-score stats (action stats for prev, proprio stats) FROM a set
    of cache samples -- the deploy-time analog of the per-task train stats in r3m ``data.load`` (Option A
    re-normalization). Returns ``(act_mean, act_std, pro_mean, pro_std)`` (7,7,9,9 float32), ready to pass
    straight to ``encoder.set_norm(*...)``.

    Mirrors training EXACTLY: action stats pool ``prev_actions`` (16,7) AND the executed ``target`` chunk
    (16,7) per dim; proprio stats are over ``states[state_source]`` (9,); std uses the same 1e-6 floor.
    Recomputing these from the deploy cache (instead of using the checkpoint's baked training stats) makes
    the encoder see prev/proprio in the DEPLOY task's own normalized frame -- matching how a per-task /
    multi-task encoder was trained, and adapting an unseen task's input scale."""
    EPS = 1e-6
    prev = np.stack([s.prev_actions for s in samples]).reshape(-1, 7)
    tgt = np.stack([s.target for s in samples]).reshape(-1, 7)
    pooled = np.concatenate([prev, tgt], 0).astype(np.float32)
    am = pooled.mean(0).astype(np.float32)
    asd = (pooled.std(0) + EPS).astype(np.float32)
    pro = np.stack([s.states[state_source] for s in samples]).astype(np.float32)
    pm = pro.mean(0).astype(np.float32)
    psd = (pro.std(0) + EPS).astype(np.float32)
    return am, asd, pm, psd


@torch.no_grad()
def compute_fused_emb(samples, encoder, device, state_source, cache_path=None, bs=256):
    """Return (N, D) FUSED embeddings (image + proprio + prev) for the cache samples, computing them
    once or loading a matching cache. Each sample contributes its decision-point primary+wrist frames,
    ``prev_actions`` (16x7) and ``states[state_source]`` proprio (9,) -- the exact inputs the encoder
    was trained on. Requires samples built with ``with_image=True``. The cache signature includes the
    state_source so a key change cannot reuse a stale cache.
    """
    assert all(s.image is not None and s.wrist is not None for s in samples), \
        "samples must be built with with_image=True to compute fused embeddings"
    sig = hashlib.md5(
        f"{len(samples)}|{state_source}".encode()
        + samples[0].image.tobytes()[:4096] + samples[-1].image.tobytes()[:4096]
    ).hexdigest()[:12]
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        if z["sig"] == sig and len(z["emb"]) == len(samples):
            print(f"[fused_emb] loaded cache {cache_path} ({len(samples)} samples)")
            return z["emb"].astype(np.float32)

    embs = []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        p = _chw([s.image for s in chunk]).to(device)
        w = _chw([s.wrist for s in chunk]).to(device)
        prev = torch.from_numpy(np.stack([s.prev_actions for s in chunk]).astype(np.float32)).to(device)
        pro = torch.from_numpy(np.stack([s.states[state_source] for s in chunk]).astype(np.float32)).to(device)
        embs.append(encoder(p, w, prev, pro).cpu().numpy().astype(np.float32))
    emb = np.concatenate(embs)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, emb=emb, sig=sig)
        print(f"[fused_emb] computed + cached {cache_path} ({len(samples)} samples, dim {emb.shape[1]})")
    return emb
