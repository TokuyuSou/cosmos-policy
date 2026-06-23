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


def _load_metric_encoder_cls():
    """Load ActionMetricEncoder from r3m_action_encoder/model.py under a unique module name
    (avoids clashing with action_predictor's own `model` module)."""
    spec = importlib.util.spec_from_file_location("r3m_metric_model", R3M_MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ActionMetricEncoder


ActionMetricEncoder = _load_metric_encoder_cls()


def load_corr_encoder(ckpt_path, device, backbone="resnet18", out_dim=128):
    """Build the ActionMetricEncoder and load the trained (corr) weights; frozen + eval.

    Built with ``share_backbone=False`` so BOTH shared and separate-backbone checkpoints load
    correctly: a shared checkpoint stores identical primary/wrist weights (its state_dict carries
    both prefixes), so loading it into a separate model reproduces the shared model's outputs exactly
    (verified bit-identical); a separate checkpoint loads its distinct per-view weights as intended.
    """
    enc = ActionMetricEncoder(backbone, init="r3m", freeze="frozen", out_dim=out_dim,
                              share_backbone=False).to(device)
    sd = torch.load(ckpt_path, map_location="cpu")
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
    fusion = "residual" if "gate" in sd else "concat"
    state_mode = "temporal" if "state_enc.tok.weight" in sd else "mlp"
    img_dim = int(sd["img_ln.weight"].shape[0])
    state_dim = int(sd["state_ln.weight"].shape[0])
    out_dim = int(sd["state_proj.weight"].shape[0] if fusion == "residual" else sd["fuse.3.weight"].shape[0])
    hidden = int(sd["fuse.0.weight"].shape[0]) if fusion == "concat" else 256
    act_dim = int(sd["state_enc.act_mean"].shape[0])
    proprio_dim = int(sd["state_enc.pro_mean"].shape[0])
    prev_steps = ((int(sd["state_enc.net.0.weight"].shape[1]) - proprio_dim) // act_dim
                  if state_mode == "mlp" else int(sd["state_enc.pos"].shape[1]) - 1)
    enc = Enc(backbone="resnet18", init="r3m", freeze="frozen", out_dim=out_dim, img_dim=img_dim,
              state_dim=state_dim, hidden=hidden, fusion=fusion, state_mode=state_mode, mod_dropout=0.0,
              share_backbone=False, prev_steps=prev_steps, act_dim=act_dim, proprio_dim=proprio_dim).to(device)
    enc.load_state_dict(sd)
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
