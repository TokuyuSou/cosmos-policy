"""R3M ResNet encoder for the action-metric task: two RGB views -> one L2-normalized embedding
whose distances match executed-action-chunk distances.

Design choices (see README for the rationale):
- Backbone: ResNet-18 initialised from R3M (manipulation-pretrained, time-contrastive). Cheap to
  fine-tune end-to-end -- the whole point of moving off DINO. Weights are loaded from a local
  checkpoint, no network needed.
- Views: a SINGLE shared-weight backbone encodes primary and wrist independently (R3M generalises
  across viewpoints; sharing halves the parameters and regularises). The 512-d per-view features
  are concatenated and a small fusion MLP learns the view weighting + projects to the metric space.
- Output: L2-normalized, so Euclidean distance is monotone in cosine -- the metric the loss and the
  retrieval policy use.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# R3M ResNet-18 backbone weights (torchvision state_dict), vendored in this dir so the encoder is
# self-contained. Source: HuggingFace surajnair/r3m-18 (convnet.* -> torchvision resnet18).
R3M_RESNET18 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r3m_resnet18.pt")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_FEAT_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


class R3MBackbone(nn.Module):
    """ResNet backbone (fc removed) producing a (B, feat_dim) feature from (B,3,224,224) images.

    ``init`` is 'r3m' (load the cached R3M state_dict), 'imagenet', or a checkpoint path.
    ``freeze`` selects which layers stay trainable: 'frozen' (none), 'layer4', 'layer34', 'none' (all).
    The R3M preprocessing (/255 then ImageNet-normalize) is applied INSIDE forward so callers just
    pass uint8/float images in [0,255].
    """

    def __init__(self, backbone="resnet18", init="r3m", freeze="layer4"):
        super().__init__()
        net = getattr(models, backbone)(weights=None)
        ckpt = R3M_RESNET18 if init == "r3m" else (init if init not in ("imagenet",) else None)
        if init == "imagenet":
            enum = (models.ResNet50_Weights.IMAGENET1K_V2 if backbone == "resnet50"
                    else getattr(models, f"ResNet{backbone[6:]}_Weights").IMAGENET1K_V1)
            net = getattr(models, backbone)(weights=enum)
        elif ckpt is not None:
            sd = torch.load(ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            missing, unexpected = net.load_state_dict(sd, strict=False)
            assert not unexpected, f"unexpected R3M keys: {unexpected[:5]}"
            assert set(missing) <= {"fc.weight", "fc.bias"}, f"missing backbone keys: {missing[:5]}"
        net.fc = nn.Identity()
        self.net = net
        self.feat_dim = _FEAT_DIM[backbone]

        trainable = {
            "frozen": (),
            "layer4": ("layer4",),
            "layer34": ("layer3", "layer4"),
            "none": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
        }[freeze]
        for name, p in self.net.named_parameters():
            p.requires_grad = any(name.startswith(t) for t in trainable)
        self.freeze = freeze
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def set_trainable_frozen(self):
        for p in self.net.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Re-apply the configured ``freeze`` mode (used after a frozen head-warmup)."""
        trainable = {"frozen": (), "layer4": ("layer4",), "layer34": ("layer3", "layer4"),
                     "none": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")}[self.freeze]
        for name, p in self.net.named_parameters():
            p.requires_grad = any(name.startswith(t) for t in trainable)

    def trunk_parameters(self):
        """The pretrained/fine-tuned backbone weights (get the small bb-lr). For ResNet that is the whole
        module; the method exists so the trainer can split trunk (bb-lr) from any auxiliary head (head-lr)
        uniformly across backbones -- see TheiaBackbone, whose attn-pool head must train at the head-lr."""
        return self.net.parameters()

    def forward(self, x):  # (B,3,224,224) uint8/float in [0,255] -> (B, feat_dim)
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        return self.net(x)


class TokenAttnPool(nn.Module):
    """Learned attention pooling over ViT spatial tokens, as a RESIDUAL on top of mean-pool:
    ``out = mean(tokens) + gamma * attn_pool(tokens)``.

    ``n_query`` learned queries each attention-pool the (B,N,dim) tokens (capturing distinct spatial
    aspects -- e.g. gripper / object / contact region); their outputs are concatenated and projected back
    to ``dim``. ``gamma`` is a single scalar initialised to 0, so at init ``out == mean-pool`` EXACTLY
    (byte-identical to ``reduce='mean'``); the attention refinement only switches on as training opens
    gamma. This floor-preserving design mirrors the residual image gate in MultiModalActionMetricEncoder
    -- it cannot regress below the proven mean-pool model, only add Theia's discarded local detail."""

    def __init__(self, dim, n_query=4):
        super().__init__()
        self.dim, self.n_query = dim, n_query
        self.q = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(n_query * dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))   # residual weight; 0 => out == mean-pool at init

    def forward(self, tok):                          # (B, N, dim) spatial tokens -> (B, dim)
        mean = tok.mean(1)
        k, v = self.k(tok), self.v(tok)              # (B, N, dim)
        scores = torch.einsum("qd,bnd->bqn", self.q, k) / self.dim ** 0.5  # (B, n_query, N)
        attn = scores.softmax(dim=-1)                # attention over the N tokens
        pooled = torch.einsum("bqn,bnd->bqd", attn, v).reshape(tok.shape[0], -1)  # (B, n_query*dim)
        return mean + self.gamma * self.proj(pooled)


class TheiaBackbone(nn.Module):
    """Theia distilled-VFM (DeiT-tiny) trunk -> (B, feat_dim) from (B,3,R,R) images in [0,255].

    Drop-in replacement for ``R3MBackbone`` (same public surface: ``feat_dim``;
    ``forward((B,3,R,R)[0,255])->(B,feat_dim)``; ``set_trainable_frozen`` / ``unfreeze``), so the
    fused/image encoders and the training loop use it unchanged. Theia's per-token spatial features
    carry far more local information than R3M's global-average-pooled ResNet.

    Preprocessing (resize->[0,1]->normalize) is replicated GPU-side and the inner HF ViT is called
    directly -- ~35x faster than the model's CPU image-processor, differentiable, and verified to match
    ``forward_feature`` to <1% relative error. Normalization constants (mean=std=[0.5,0.5,0.5] for
    theia-tiny, NOT ImageNet) and the resize target are read from the model's own processor so they
    can never drift; they are baked into buffers so the saved checkpoint is self-contained.

    ``freeze=='frozen'`` keeps the whole trunk fixed (the default for a pretrained ViT on small data);
    anything else fine-tunes only the last transformer block + the final LayerNorm.

    ``reduce`` chooses how the 196 spatial tokens become one per-view vector: 'mean' (GAP, the faithful
    drop-in) | 'max' | 'attn' (a learned residual attention-pool head -- TokenAttnPool -- that exploits
    Theia's local detail; ``attn_queries`` sets its number of pooling queries). The trunk is untouched in
    all cases; 'attn' only adds a small trainable head (trained at the head-lr, see ``trunk_parameters``).
    """

    DEFAULT_MODEL = "theaiinstitute/theia-tiny-patch16-224-cddsv"
    MODEL_BY_DIM = {192: "theaiinstitute/theia-tiny-patch16-224-cddsv",   # feat_dim -> ckpt
                    384: "theaiinstitute/theia-small-patch16-224-cddsv",
                    768: "theaiinstitute/theia-base-patch16-224-cddsv"}

    def __init__(self, model_name=DEFAULT_MODEL, freeze="frozen", reduce="mean", attn_queries=4):
        super().__init__()
        from transformers import AutoModel
        theia = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        deit = theia.backbone                 # DeiT wrapper: .model = inner ViTModel, .processor
        self.vit = deit.model                 # HF ViTModel (pooler already nn.Identity())
        proc = deit.processor
        self.feat_dim = int(self.vit.config.hidden_size)   # 192 (tiny)/384/768
        self.res = int(proc.size["height"])                # 224 (direct resize, no center-crop)
        self.model_name = model_name
        self.reduce = reduce                  # 'mean' (GAP) | 'max' | 'attn' (learned residual pool)
        self.freeze = freeze
        self.pool = TokenAttnPool(self.feat_dim, attn_queries) if reduce == "attn" else None
        # Bake the OFFICIAL processor normalization (mean=std=[0.5,0.5,0.5]) into buffers, so the saved
        # checkpoint is self-contained and the constants can never drift from Theia's preprocessing.
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1))
        self.vit.eval()
        self.unfreeze()

    def trunk_parameters(self):
        """ONLY the pretrained ViT weights (get the small bb-lr). The attn-pool head (if any) is excluded
        so the trainer routes that fresh head to the head-lr -- see R3MBackbone.trunk_parameters."""
        return self.vit.parameters()

    def set_trainable_frozen(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Re-apply the configured ``freeze`` mode: 'frozen' = whole trunk fixed; anything else
        fine-tunes the last transformer block + the final LayerNorm (used after a frozen head-warmup)."""
        for p in self.vit.parameters():
            p.requires_grad = False
        if self.freeze != "frozen":
            for p in self.vit.encoder.layer[-1].parameters():
                p.requires_grad = True
            for p in self.vit.layernorm.parameters():
                p.requires_grad = True

    def train(self, mode=True):
        """Keep a fully-frozen trunk in eval() (deterministic; no dropout/stochastic depth) even when
        the parent encoder .train()s -- otherwise the embeddings would be non-deterministic."""
        super().train(mode)
        if self.freeze == "frozen":
            self.vit.eval()
        return self

    def tokens(self, x):                       # (B,3,R,R) uint8/float [0,255] -> (B, H*W, C) spatial tokens
        # GPU-native replica of Theia's HF processor: resize->[0,1]->normalize (mean/std=0.5).
        x = F.interpolate(x.float(), size=(self.res, self.res), mode="bilinear",
                          align_corners=False, antialias=True)     # antialias matches PIL BILINEAR
        x = (x / 255.0 - self.mean) / self.std
        tok = self.vit(pixel_values=x, interpolate_pos_encoding=False).last_hidden_state  # (B,1+H*W,C)
        return tok[:, 1:]                      # drop CLS -> (B,H*W,C) spatial tokens (the pre-pool output)

    def forward(self, x):                      # (B,3,R,R) uint8/float [0,255] -> (B, feat_dim) pooled
        tok = self.tokens(x)
        if self.reduce == "attn":
            return self.pool(tok)
        return tok.mean(1) if self.reduce == "mean" else tok.amax(1)


class TokenChangePool(nn.Module):
    """Version-B change-guided attention pool over the PRIMARY view's spatial tokens. Pools the current
    tokens with attention over per-token ``[T_cur ; LayerNorm(T_cur - T_init)]`` (so each location carries
    'what is here' AND 'how it changed'), with the attention additionally BIASED toward high-change
    locations (a learnable scalar on the per-token change magnitude). As a RESIDUAL on mean-pool:

        z = mean(T_cur) + gamma * proj( attn_pool([T_cur ; LN(T_cur - T_init)]) )

    ``gamma`` init 0 -> ``z == mean(T_cur)`` (the no-diff floor), so B is a strict generalization that can
    only add the change-guided refinement. ``bias`` init 0 -> attention starts content-only, then learns
    to weight changed regions. Unlike Version A (pool current & initial SEPARATELY with a SHARED pool, then
    subtract), here the difference is taken at the TOKEN level and the pool itself sees WHERE it changed,
    and this pool is DEDICATED to the primary view (not shared with the wrist)."""

    def __init__(self, dim, n_query=4):
        super().__init__()
        self.dim = dim
        self.dnorm = nn.LayerNorm(dim)
        self.q = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.k = nn.Linear(2 * dim, dim)
        self.v = nn.Linear(2 * dim, dim)
        self.bias = nn.Parameter(torch.zeros(1))    # learnable weight on the change-magnitude attention prior
        self.proj = nn.Linear(n_query * dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))   # residual weight; 0 => z == mean(T_cur) at init

    def forward(self, T_c, T_0):                     # (B,N,dim),(B,N,dim) -> (B,dim)
        mean = T_c.mean(1)
        delta = T_c - T_0
        U = torch.cat([T_c, self.dnorm(delta)], dim=-1)               # (B,N,2*dim): current + normalized change
        k, v = self.k(U), self.v(U)                                   # (B,N,dim)
        chg = delta.norm(dim=-1)                                      # (B,N) raw per-token change magnitude
        chg = (chg - chg.mean(1, keepdim=True)) / (chg.std(1, keepdim=True) + 1e-6)   # per-image standardized
        scores = torch.einsum("qd,bnd->bqn", self.q, k) / self.dim ** 0.5 + self.bias * chg.unsqueeze(1)
        attn = scores.softmax(dim=-1)                                 # attention over the N tokens
        pooled = torch.einsum("bqn,bnd->bqd", attn, v).reshape(T_c.shape[0], -1)      # (B, n_query*dim)
        return mean + self.gamma * self.proj(pooled)


def make_backbone(img_backbone="r3m", backbone="resnet18", init="r3m", freeze="layer4", theia_model=None,
                  theia_reduce="mean", attn_queries=4):
    """Build the per-view image trunk. ``img_backbone`` selects the FAMILY ('r3m' = the ResNet-18
    R3M/ImageNet backbone, the default => existing behavior unchanged; 'theia' = the distilled-VFM ViT).
    ``backbone`` (resnet arch) / ``init`` only apply to 'r3m'; ``theia_model`` (HF id), ``theia_reduce``
    (token aggregation: mean|max|attn) and ``attn_queries`` only apply to 'theia'."""
    if img_backbone == "theia":
        return TheiaBackbone(theia_model or TheiaBackbone.DEFAULT_MODEL, freeze=freeze,
                             reduce=theia_reduce, attn_queries=attn_queries)
    if img_backbone == "r3m":
        return R3MBackbone(backbone, init=init, freeze=freeze)
    raise ValueError(f"unknown img_backbone={img_backbone!r} (expected 'r3m' or 'theia')")


class ActionMetricEncoder(nn.Module):
    """Two views -> shared R3M backbone -> concat -> fusion MLP -> L2-normalized embedding."""

    def __init__(self, backbone="resnet18", init="r3m", freeze="layer4",
                 out_dim=128, hidden=512, dropout=0.1, share_backbone=True,
                 img_backbone="r3m", theia_model=None, theia_reduce="mean", attn_queries=4):
        super().__init__()
        bb = lambda: make_backbone(img_backbone, backbone, init, freeze, theia_model,
                                   theia_reduce, attn_queries)
        self.primary_bb = bb()
        self.wrist_bb = self.primary_bb if share_backbone else bb()
        self.share_backbone = share_backbone
        fd = self.primary_bb.feat_dim
        self.head = nn.Sequential(
            nn.LayerNorm(2 * fd), nn.Linear(2 * fd, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def backbones(self):
        return (self.primary_bb,) if self.share_backbone else (self.primary_bb, self.wrist_bb)

    def head_warmup(self, on: bool):
        """Freeze (on) / restore (off) the backbone(s) -- used to warm up the random head first."""
        for bb in self.backbones():
            bb.set_trainable_frozen() if on else bb.unfreeze()

    def forward(self, primary, wrist):
        fp = self.primary_bb(primary)
        fw = self.wrist_bb(wrist)
        z = self.head(torch.cat([fp, fw], dim=-1))
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# Multi-modal fusion encoder: image + proprio + prev-actions -> one embedding.
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """Encode the cheap proprioceptive key (proprio + prev-action history) into a feature.

    Inputs are z-scored INSIDE the module using registered buffers (action stats for ``prev``,
    proprio stats for ``proprio``) so the encoder is self-contained at deploy time. Two encoders:

    - ``mlp``: flatten z-scored prev (16x7) + proprio (9) -> MLP. Matches the proprio_prev retrieval
      key's input exactly; the 0.554 RMSE@1 baseline proves this input already carries the state
      signal, so this is the robust default on limited data.
    - ``temporal``: a tiny Transformer over the 16 prev-action steps (each a 7-d token, + a learned
      CLS token and positional embedding); its CLS output is concatenated with an MLP of proprio.
      The "proper" sequence encoder -- captures momentum/trend with an explicit temporal inductive
      bias; tested as a potential upgrade.
    - ``recency``: a recency-pooling DUAL-STREAM encoder, motivated by a training-free probe
      (``probe_state_features.py``): for the next-chunk target the signal is dominated by the MOST
      RECENT actions (the last step alone ~= all 16 for RMSE@1; far-past steps add ranking NOISE),
      while proprio is a weak-but-complementary phase context. So the flat ``mlp`` (16 steps equally
      weighted) wastes capacity on the noisy far-past and the ``temporal`` Transformer pools all 16
      with no recency prior. ``recency`` instead: (1) attention-POOLING over the step tokens with a
      learned per-step (recency) bias -> a summary that learns HOW FAR back to look; (2) a raw skip of
      the last ``recent`` actions (default 3) projected directly in, guaranteeing the proven dominant
      feature (probe: proprio+last3 ~= the 121-d key) regardless of what the pooling learns; (3) a
      DEDICATED proprio MLP so the 9-d pose is not drowned by the 112-d prev. The three streams are
      concatenated -> MLP. Pooling-only (single query) keeps it data-efficient vs self-attention.
    - ``proprio``: proprio-ONLY (prev-actions IGNORED) -- a small MLP on the 9-d self-state. Used by
      ObsStateActionMetricEncoder to force the image to be the representation; the state side then
      carries only the current pose (a weak phase context), never the action history.
    """

    def __init__(self, prev_steps=16, act_dim=7, proprio_dim=9, out_dim=256,
                 mode="mlp", dropout=0.1, d_model=64, nhead=4, layers=2, recent=3):
        super().__init__()
        self.mode = mode
        self.prev_steps, self.act_dim, self.proprio_dim = prev_steps, act_dim, proprio_dim
        # z-score buffers (set via set_norm; default identity so the module is usable untrained)
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))
        self.register_buffer("pro_mean", torch.zeros(proprio_dim))
        self.register_buffer("pro_std", torch.ones(proprio_dim))

        if mode == "mlp":
            din = prev_steps * act_dim + proprio_dim
            self.net = nn.Sequential(
                nn.Linear(din, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim), nn.GELU(),
            )
        elif mode == "temporal":
            self.tok = nn.Linear(act_dim, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.zeros(1, prev_steps + 1, d_model))
            enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                             dropout=dropout, batch_first=True, activation="gelu")
            self.tf = nn.TransformerEncoder(enc, layers)
            self.pro = nn.Sequential(nn.Linear(proprio_dim, d_model), nn.GELU())
            self.net = nn.Sequential(nn.Linear(2 * d_model, out_dim), nn.GELU(), nn.Dropout(dropout))
        elif mode == "recency":
            # internal width scales with the branch width (no new required ctor args); ~2-2.5x mlp params.
            dh = max(64, out_dim // 4)
            self.dh = dh
            self.recent = min(int(recent), prev_steps)        # raw last-N skip window (probe sweet spot ~3)
            # (1) attention pooling over the step tokens (single learned query + recency score bias)
            self.step_embed = nn.Linear(act_dim, dh)
            self.pool_k = nn.Linear(dh, dh)
            self.pool_v = nn.Linear(dh, dh)
            self.pool_q = nn.Parameter(torch.randn(dh) * 0.02)
            # per-step additive bias on the pooling logits, init as a mild recency ramp (recent steps higher)
            self.pool_bias = nn.Parameter(torch.linspace(0.0, 1.0, prev_steps))
            # (2) raw last-`recent` actions, projected straight in (lossless recency skip)
            self.recent_proj = nn.Linear(self.recent * act_dim, dh)
            # (3) dedicated proprio MLP (so the 9-d pose is not drowned)
            self.pro_enc = nn.Sequential(nn.Linear(proprio_dim, dh), nn.GELU(), nn.Linear(dh, dh), nn.GELU())
            # fuse the three streams -> out_dim (same output contract as the other modes; caller LayerNorms)
            self.fuse_net = nn.Sequential(
                nn.Linear(3 * dh, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim), nn.GELU())
        elif mode == "proprio":
            # proprio-ONLY self-state (prev IGNORED): a small MLP on the 9-d pose. Same output contract
            # as the other modes (the caller LayerNorms the result).
            self.net = nn.Sequential(
                nn.Linear(proprio_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim), nn.GELU())
        else:
            raise ValueError(mode)
        self.out_dim = out_dim

    def set_norm(self, am, asd, pm, psd):
        for buf, val in (("act_mean", am), ("act_std", asd), ("pro_mean", pm), ("pro_std", psd)):
            getattr(self, buf).copy_(torch.as_tensor(val, dtype=torch.float32))

    def _forward_recency(self, prev, proprio):  # prev (B,L,act_dim) z-scored, proprio (B,proprio_dim) z-scored
        H = self.step_embed(prev)                                       # (B,L,dh) step tokens
        scores = (self.pool_k(H) @ self.pool_q) / self.dh ** 0.5 + self.pool_bias   # (B,L) per-step logits
        alpha = scores.softmax(dim=-1)                                  # (B,L) recency-weighted attention
        pooled = (alpha.unsqueeze(-1) * self.pool_v(H)).sum(dim=1)      # (B,dh) attention-pooled summary
        recent = self.recent_proj(prev[:, -self.recent:].reshape(prev.shape[0], -1))  # (B,dh) raw last-N skip
        pro = self.pro_enc(proprio)                                     # (B,dh) dedicated proprio path
        return self.fuse_net(torch.cat([pooled, recent, pro], dim=-1))  # (B,out_dim)

    def forward(self, prev, proprio):           # prev (B,16,7), proprio (B,9)
        proprio = (proprio - self.pro_mean) / self.pro_std
        if self.mode == "proprio":              # self-state only -- prev is IGNORED
            return self.net(proprio)
        prev = (prev - self.act_mean) / self.act_std
        if self.mode == "mlp":
            x = torch.cat([prev.flatten(1), proprio], dim=-1)
            return self.net(x)
        if self.mode == "recency":
            return self._forward_recency(prev, proprio)
        tok = self.tok(prev)                                    # (B,16,d)
        cls = self.cls.expand(prev.shape[0], -1, -1)            # (B,1,d)
        h = torch.cat([cls, tok], dim=1) + self.pos             # (B,17,d)
        h = self.tf(h)[:, 0]                                    # CLS output (B,d)
        return self.net(torch.cat([h, self.pro(proprio)], dim=-1))


class MultiModalActionMetricEncoder(nn.Module):
    """Image (primary+wrist via shared R3M) + state (proprio+prev) -> one L2-normalized embedding
    whose distance matches executed-action-chunk distance.

    Fusion = per-branch LayerNorm -> [modality dropout] -> concat -> MLP -> L2-normalize. Modality
    dropout (train only) zeros a whole branch per sample so each modality must be independently
    useful -- this is what stops the fusion from collapsing onto the dominant state branch (which
    alone already beats the image), the failure mode that made naive concat no better than the key.
    """

    def __init__(self, backbone="resnet18", init="r3m", freeze="layer4", out_dim=128,
                 img_dim=256, state_dim=256, hidden=256, dropout=0.1, share_backbone=True,
                 state_mode="mlp", mod_dropout=0.3, fusion="concat", prev_steps=16, act_dim=7,
                 proprio_dim=9, gate_dyn=False, gate_rank=64, img_backbone="r3m", theia_model=None,
                 theia_reduce="mean", attn_queries=4, gate_init=-2.0):
        super().__init__()
        bb = lambda: make_backbone(img_backbone, backbone, init, freeze, theia_model,
                                   theia_reduce, attn_queries)
        self.primary_bb = bb()
        self.wrist_bb = self.primary_bb if share_backbone else bb()
        self.share_backbone = share_backbone
        fd = self.primary_bb.feat_dim
        self.img_head = nn.Sequential(
            nn.LayerNorm(2 * fd), nn.Linear(2 * fd, img_dim), nn.GELU(), nn.Dropout(dropout))
        self.state_enc = StateEncoder(prev_steps, act_dim, proprio_dim, state_dim,
                                      mode=state_mode, dropout=dropout)
        self.img_ln = nn.LayerNorm(img_dim)
        self.state_ln = nn.LayerNorm(state_dim)
        self.fusion = fusion
        if fusion == "concat":
            self.fuse = nn.Sequential(
                nn.Linear(img_dim + state_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, out_dim))
        elif fusion == "residual":
            # State is the BASE of the embedding; image is a gated CORRECTION added to it. The gate
            # starts at sigmoid(gate_init) (default -2 -> ~0.12) so the metric begins as the learned-state
            # metric (>= proprio_prev on the same input) and only adds image where it helps -- a floor
            # near the baseline with upside, the structure most likely to *clearly* beat the key. Raise
            # gate_init (e.g. 0 -> 0.5) to start the image with MORE weight when it is under-used.
            self.state_proj = nn.Linear(state_dim, out_dim)
            self.img_proj = nn.Linear(img_dim, out_dim)
            self.gate = nn.Parameter(torch.full((out_dim,), float(gate_init)))
            # Optional INPUT-DEPENDENT gate: a small low-rank MLP on [state, img] produces a per-sample,
            # per-dim DELTA added to the static gate BEFORE the sigmoid. Near-identity init (tiny final
            # weights + zero bias) => at init the gate == the static gate, so this is a strict
            # generalization that cannot regress the proven model, yet lets the image weight adapt to the
            # situation. The final weights are small but NON-zero so gradient still flows to the first layer.
            if gate_dyn:
                self.gate_net = nn.Sequential(
                    nn.Linear(img_dim + state_dim, gate_rank), nn.GELU(),
                    nn.Linear(gate_rank, out_dim))
                nn.init.normal_(self.gate_net[-1].weight, std=1e-3)
                nn.init.zeros_(self.gate_net[-1].bias)
        else:
            raise ValueError(fusion)
        self.gate_dyn = bool(gate_dyn) and fusion == "residual"
        self.mod_dropout = mod_dropout
        self.out_dim = out_dim

    def backbones(self):
        return (self.primary_bb,) if self.share_backbone else (self.primary_bb, self.wrist_bb)

    def head_warmup(self, on: bool):
        for bb in self.backbones():
            bb.set_trainable_frozen() if on else bb.unfreeze()

    def set_norm(self, am, asd, pm, psd):
        self.state_enc.set_norm(am, asd, pm, psd)

    def _modality_dropout(self, img, state):
        """Zero a whole branch per sample (train only); never both (keep state when both drawn)."""
        if not self.training or self.mod_dropout <= 0:
            return img, state
        B = img.shape[0]
        p = self.mod_dropout
        drop_img = torch.rand(B, 1, device=img.device) < p
        drop_state = torch.rand(B, 1, device=img.device) < p
        drop_state = drop_state & ~(drop_img & drop_state)   # if both drawn, keep state (the strong branch)
        return img * (~drop_img), state * (~drop_state)

    def _img_bb_batchable(self):
        """Whether both views can share ONE backbone forward: true iff the two image backbones are the same
        module or hold byte-identical weights+buffers (the deploy loader builds share_backbone=False but a
        shared-trained checkpoint fills both prefixes identically). Computed once, cached."""
        b = getattr(self, "_img_bb_batchable_cache", None)
        if b is None:
            if self.wrist_bb is self.primary_bb:
                b = True
            else:
                sa, sw = self.primary_bb.state_dict(), self.wrist_bb.state_dict()
                b = (list(sa.keys()) == list(sw.keys())
                     and all(torch.equal(sa[k], sw[k]) for k in sa))
            self._img_bb_batchable_cache = b
        return b

    def _image_feat(self, primary, wrist):
        """Per-view pooled backbone features concatenated -> img_head input (B, 2*feat_dim). Factored out
        so subclasses (InitDiffFusedEncoder) can extend it WITHOUT touching the fusion. Behaviour-identical
        to the original inline ``cat([primary_bb(primary), wrist_bb(wrist)])``.

        At EVAL, when the two backbones share weights, both views are run in a SINGLE batched forward
        (stacked to (2B,...) then split). In eval the ResNet BatchNorm uses its frozen running stats, so a
        batched pass is numerically identical to per-view passes -- it only avoids a second kernel launch.
        The TRAIN path is left byte-identical (guarded by ``self.training``) so batch statistics can never
        change; a non-shared backbone also keeps the original per-view passes."""
        if (not self.training) and self._img_bb_batchable() and primary.shape == wrist.shape:
            b = primary.shape[0]
            f = self.primary_bb(torch.cat([primary, wrist], dim=0))   # ONE forward over 2B views
            return torch.cat([f[:b], f[b:]], dim=-1)
        return torch.cat([self.primary_bb(primary), self.wrist_bb(wrist)], dim=-1)

    def _fuse(self, fimg, fstate):
        """[modality-dropout] -> concat/residual fusion -> L2-normalize. Factored out so subclasses reuse
        the EXACT fusion (residual gate / concat / mod-dropout) and only change how ``fimg`` is built."""
        fimg, fstate = self._modality_dropout(fimg, fstate)
        if self.fusion == "concat":
            z = self.fuse(torch.cat([fstate, fimg], dim=-1))
        else:  # residual: state base + gated image correction
            g = self.gate
            if self.gate_dyn:  # input-dependent per-sample, per-dim delta on the gate
                g = g + self.gate_net(torch.cat([fstate, fimg], dim=-1))
            z = self.state_proj(fstate) + torch.sigmoid(g) * self.img_proj(fimg)
        return F.normalize(z, dim=-1)

    def forward(self, primary, wrist, prev, proprio):
        fimg = self.img_ln(self.img_head(self._image_feat(primary, wrist)))
        fstate = self.state_ln(self.state_enc(prev, proprio))
        return self._fuse(fimg, fstate)


class InitDiffFusedEncoder(MultiModalActionMetricEncoder):
    """Version A of the CHANGE-GUIDED image branch: the image feature includes, per configured view, the
    feature-level DIFFERENCE of the current and the episode-INITIAL pooled backbone feature
    (``current - initial``). The metric then sees the CHANGE from the initial scene (manipulation
    progress: object moved / grasped / placed), not only the static current scene 'atmosphere'.

    The proprio_prev-style STATE branch and the residual (additive) fusion are INHERITED UNCHANGED -- the
    image (current ⊕ diff) is *added* to the state base exactly like the parent's residual gate. Only the
    image-feature extractor and ``img_head`` input width change.

    Drop-in extension of MultiModalActionMetricEncoder: same ctor + ``set_norm`` / ``backbones`` /
    ``head_warmup`` surface; forward gains optional ``primary_init`` / ``wrist_init`` (the episode-initial
    frames). With an init frame == None the corresponding diff stream is ZERO -> the encoder degrades to
    current-only (and is bit-identical to passing init == current). ``init_diff_views`` lists which views
    contribute a diff stream (A default: ``("primary",)`` -- the third-person camera, whose patches are
    world-aligned; the WRIST is a moving camera so its initial-frame diff is ill-posed and omitted).

    Built so Version B (token-level change-guided pool) only needs to override ``_view_feat`` / the per-view
    extractor, NOT this class's fusion or data path."""

    DIFFABLE = ("primary", "wrist")

    def __init__(self, *args, init_diff_views=("primary",), **kwargs):
        super().__init__(*args, **kwargs)
        self.init_diff_views = tuple(init_diff_views)
        assert all(v in self.DIFFABLE for v in self.init_diff_views), \
            f"init_diff_views must be subset of {self.DIFFABLE}, got {self.init_diff_views}"
        fd = self.primary_bb.feat_dim
        img_dim = self.img_head[1].out_features            # reuse the parent's img_head width + dropout
        drop_p = self.img_head[3].p
        n_streams = 2 + len(self.init_diff_views)           # primary_cur, wrist_cur, + one diff per view
        self.img_head = nn.Sequential(
            nn.LayerNorm(n_streams * fd), nn.Linear(n_streams * fd, img_dim), nn.GELU(), nn.Dropout(drop_p))

    def _bb(self, view):
        return self.primary_bb if view == "primary" else self.wrist_bb

    def _image_feat(self, primary, wrist, primary_init=None, wrist_init=None):
        cur = {"primary": self.primary_bb(primary), "wrist": self.wrist_bb(wrist)}
        init = {"primary": primary_init, "wrist": wrist_init}
        streams = [cur["primary"], cur["wrist"]]            # always: current per view
        for v in self.init_diff_views:                      # append the CHANGE (current - initial) per view
            x0 = init[v]
            streams.append(cur[v] - self._bb(v)(x0) if x0 is not None else torch.zeros_like(cur[v]))
        return torch.cat(streams, dim=-1)

    def forward(self, primary, wrist, prev, proprio, primary_init=None, wrist_init=None):
        fimg = self.img_ln(self.img_head(self._image_feat(primary, wrist, primary_init, wrist_init)))
        fstate = self.state_ln(self.state_enc(prev, proprio))
        return self._fuse(fimg, fstate)


class TokenChangeFusedEncoder(MultiModalActionMetricEncoder):
    """Version B: the PRIMARY view's image feature is a DEDICATED TokenChangePool over its spatial tokens
    (``z_prim = mean(T_cur) + gamma * change_pool(T_cur, T_init)``), so the difference is taken at the TOKEN
    level and the pool attends to WHERE the scene changed -- instead of Version A's pool-then-subtract with
    a pool SHARED across current/initial/wrist. The wrist (moving camera) stays current-only via the
    backbone pool. ``img_feat = [z_prim ; z_wrist]`` (2*feat_dim), so ``img_head`` and the residual fusion
    are the INHERITED base widths -- only the primary feature extractor changes.

    Theia-only (needs spatial ``tokens``). Drop-in: same forward signature + surface as the other fused
    encoders; ``gamma`` init 0 -> ``z_prim == mean(T_cur)`` (the no-diff floor). ``primary_init`` None or ==
    current -> zero change (bit-identical), so it degrades gracefully."""

    def __init__(self, *args, attn_queries=4, **kwargs):
        super().__init__(*args, attn_queries=attn_queries, **kwargs)
        assert hasattr(self.primary_bb, "tokens"), \
            "TokenChangeFusedEncoder (Version B) needs a token-producing backbone -- use --img-backbone theia"
        self.change_pool = TokenChangePool(self.primary_bb.feat_dim, n_query=attn_queries)

    def forward(self, primary, wrist, prev, proprio, primary_init=None, wrist_init=None):
        T_c = self.primary_bb.tokens(primary)
        T_0 = self.primary_bb.tokens(primary_init) if primary_init is not None else T_c   # init None -> zero change
        z_prim = self.change_pool(T_c, T_0)               # change-aware primary feature (B, feat_dim)
        z_wrist = self.wrist_bb(wrist)                     # current-only pooled wrist feature
        fimg = self.img_ln(self.img_head(torch.cat([z_prim, z_wrist], dim=-1)))
        fstate = self.state_ln(self.state_enc(prev, proprio))
        return self._fuse(fimg, fstate)


class ResidualImageEncoder(nn.Module):
    """Boosting-style residual encoder: a FROZEN state-only base ``z_N1`` (a StateActionMetricEncoder) plus
    a trainable image residual --  ``z = z_N1 + sigmoid(gate) * img_proj(image_feature)`` then L2-normalize.

    ONLY the image branch + ``img_proj`` + ``gate`` train; the state base is frozen (and kept in eval()).
    Because the base is fixed and already strong, the metric loss's gradient concentrates on the points the
    state base gets WRONG (the state-confusable residual) -- so the image is forced to learn that residual,
    rather than staying dormant as it does under joint training (where the state can absorb the easy signal).
    Pair with ``--hard-neg`` to flood each batch with the state-confusable decoys the image must separate.

    ``image_mode``: 'attn' (plain pool of the current primary+wrist) | 'tokenchange' (Version-B dedicated
    token change-pool on the primary view -> uses the episode-initial frame). Same public surface
    (``backbones`` / ``head_warmup`` / ``set_norm``) and the same ``(primary, wrist, prev, proprio[,
    primary_init, wrist_init])`` forward as the other fused encoders, so the training loop / eval embedders
    consume it unchanged. General over Theia variant / share_backbone / n_query."""

    def __init__(self, state_base, out_dim=128, img_dim=256, dropout=0.1, share_backbone=True,
                 image_mode="attn", img_backbone="theia", theia_model=None, theia_reduce="attn",
                 attn_queries=4, gate_init=0.0):
        super().__init__()
        assert image_mode in ("attn", "tokenchange"), image_mode
        assert state_base.out_dim == out_dim, \
            f"state base out_dim {state_base.out_dim} != encoder out_dim {out_dim}"
        self.state = state_base                          # frozen base (z_N1)
        for p in self.state.parameters():
            p.requires_grad = False
        self.state.eval()
        self.image_mode = image_mode
        bb = lambda: make_backbone(img_backbone, "resnet18", "r3m", "frozen", theia_model,
                                   theia_reduce, attn_queries)
        self.primary_bb = bb()
        self.wrist_bb = self.primary_bb if share_backbone else bb()
        self.share_backbone = share_backbone
        fd = self.primary_bb.feat_dim
        if image_mode == "tokenchange":
            assert hasattr(self.primary_bb, "tokens"), "image_mode=tokenchange needs a token backbone (theia)"
            self.change_pool = TokenChangePool(fd, n_query=attn_queries)
        self.img_head = nn.Sequential(
            nn.LayerNorm(2 * fd), nn.Linear(2 * fd, img_dim), nn.GELU(), nn.Dropout(dropout))
        self.img_ln = nn.LayerNorm(img_dim)
        self.img_proj = nn.Linear(img_dim, out_dim)
        self.gate = nn.Parameter(torch.full((out_dim,), float(gate_init)))
        self.out_dim = out_dim

    def backbones(self):
        return (self.primary_bb,) if self.share_backbone else (self.primary_bb, self.wrist_bb)

    def head_warmup(self, on: bool):
        for bb in self.backbones():
            bb.set_trainable_frozen() if on else bb.unfreeze()

    def set_norm(self, am, asd, pm, psd):
        self.state.set_norm(am, asd, pm, psd)            # the frozen base operates on the deploy data's norm

    def train(self, mode=True):
        super().train(mode)
        self.state.eval()                                # frozen base stays deterministic (no dropout)
        return self

    def _img_feat(self, primary, wrist, primary_init=None):
        z_wrist = self.wrist_bb(wrist)
        if self.image_mode == "tokenchange":
            T_c = self.primary_bb.tokens(primary)
            T_0 = self.primary_bb.tokens(primary_init) if primary_init is not None else T_c   # init None -> 0 change
            z_prim = self.change_pool(T_c, T_0)
        else:
            z_prim = self.primary_bb(primary)
        return torch.cat([z_prim, z_wrist], dim=-1)

    def forward(self, primary, wrist, prev, proprio, primary_init=None, wrist_init=None):
        with torch.no_grad():                            # base frozen -> no grad/graph through it
            z_n1 = self.state.features(prev, proprio)
        fimg = self.img_ln(self.img_head(self._img_feat(primary, wrist, primary_init)))
        z = z_n1 + torch.sigmoid(self.gate) * self.img_proj(fimg)
        return F.normalize(z, dim=-1)


class ProgressActionMetricEncoder(MultiModalActionMetricEncoder):
    """Fused encoder + an auxiliary TASK-PROGRESS head. Motivation (see probe_progress.py): the ideal
    (oracle) retrieval already pulls chunks from ~the same normalized progress t/T (pearson 0.82-0.99),
    so injecting progress as an explicit, low-noise structural signal should pull the LEARNED retrieval
    toward that oracle behaviour (retrieve from the same phase) and sharpen the embedding.

    The head reads the FINAL L2-normalized embedding ``z`` (the exact vector retrieval compares) and is
    deliberately SHALLOW (one hidden layer) so progress must be (near-)linearly encoded in z's geometry
    -- i.e. it shows up in the retrieval DISTANCE -- rather than being decoded by a deep head that could
    hide it. ``forward`` is INHERITED UNCHANGED (returns only z), so this is a drop-in for eval/deploy;
    the head is used only in training via ``progress(z)``. Trained with a Huber (smooth-L1) regression
    to the absolute progress label (oracle uses absolute-progress similarity), weighted by lambda.
    """

    def __init__(self, *args, prog_hidden=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.prog_head = nn.Sequential(
            nn.Linear(self.out_dim, prog_hidden), nn.GELU(), nn.Linear(prog_hidden, 1))

    def progress(self, z):
        """Predict normalized task progress in [0,1] from the (L2-normalized) embedding z. (B,out_dim)->(B,)."""
        return torch.sigmoid(self.prog_head(z)).squeeze(-1)


class ObsStateActionMetricEncoder(nn.Module):
    """Observation (primary+wrist via shared R3M) + self-state PROPRIO (NO prev-actions) -> one
    L2-normalized embedding whose distance matches executed-action-chunk distance.

    The IMAGE is the BASE of the embedding; proprio is a GATED correction added on top:

        z = img_proj(f_img)  +  gate(x) * pro_proj(f_pro)          (then L2-normalize)

    Motivation. The fused encoder's residual makes proprio+prev the BASE and the image a small
    (sigmoid(-2)=0.12) correction, so the image stays under-used. Dropping prev-actions AND flipping the
    residual so the IMAGE is the base forces the metric geometry to be vision-driven -- proprio only
    refines the phase/pose the image cannot read precisely (gripper aperture, absolute eef pose).

    Dynamic gate (simple + effective). ``gate = sigmoid(b + s(x))``:
      - ``b``: a per-dim STATIC gate (which dims generally need proprio, e.g. the gripper); ``b`` ==
        ``gate_init`` at init (default 0.0 -> weight 0.5).
      - ``s(x)``: a SCALAR, image-conditioned shift broadcast over all dims -- "how much does THIS frame
        need proprio". A tiny low-rank MLP on ``f_img`` ONLY (proprio is always present, so gating on it
        would just learn to always-open). Its last layer is ~0-init, so at start ``s(x)~=0`` and
        ``gate == sigmoid(b)`` -- the proven static gate (FLOOR-PRESERVING). ``gate_dyn=False`` = the
        static-gate control.

    DROP-IN for MultiModalActionMetricEncoder: identical ``forward(primary, wrist, prev, proprio)`` --
    ``prev`` is accepted but IGNORED (self-state = proprio only) -- plus the same ``backbones()`` /
    ``head_warmup`` / ``set_norm`` / ``out_dim`` surface, so train.py, ``eval.embed_all_mm`` and
    ``obs_embed`` consume it unchanged (mirrors how StateActionMetricEncoder ignores the image args).
    """

    def __init__(self, backbone="resnet18", init="r3m", freeze="layer4", out_dim=128,
                 img_dim=256, state_dim=128, dropout=0.1, share_backbone=True, mod_dropout=0.0,
                 prev_steps=16, act_dim=7, proprio_dim=9, gate_init=0.0, gate_dyn=True, gate_rank=16,
                 img_backbone="r3m", theia_model=None, theia_reduce="mean", attn_queries=4):
        super().__init__()
        bb = lambda: make_backbone(img_backbone, backbone, init, freeze, theia_model,
                                   theia_reduce, attn_queries)
        self.primary_bb = bb()
        self.wrist_bb = self.primary_bb if share_backbone else bb()
        self.share_backbone = share_backbone
        fd = self.primary_bb.feat_dim
        # image branch -- identical to the fused encoder's (shared R3M -> concat -> head)
        self.img_head = nn.Sequential(
            nn.LayerNorm(2 * fd), nn.Linear(2 * fd, img_dim), nn.GELU(), nn.Dropout(dropout))
        self.img_ln = nn.LayerNorm(img_dim)
        # self-state branch -- proprio ONLY (StateEncoder in proprio mode ignores prev)
        self.state_enc = StateEncoder(prev_steps, act_dim, proprio_dim, state_dim,
                                      mode="proprio", dropout=dropout)
        self.state_ln = nn.LayerNorm(state_dim)
        # fusion: image base + gated proprio correction
        self.img_proj = nn.Linear(img_dim, out_dim)
        self.pro_proj = nn.Linear(state_dim, out_dim)
        self.gate = nn.Parameter(torch.full((out_dim,), float(gate_init)))   # per-dim static gate b
        self.gate_dyn = bool(gate_dyn)
        if self.gate_dyn:            # scalar, image-conditioned shift s(x); ~0-init keeps gate == sigmoid(b)
            self.gate_net = nn.Sequential(
                nn.Linear(img_dim, gate_rank), nn.GELU(), nn.Linear(gate_rank, 1))
            nn.init.normal_(self.gate_net[-1].weight, std=1e-3)
            nn.init.zeros_(self.gate_net[-1].bias)
        self.mod_dropout = mod_dropout
        self.out_dim = out_dim

    def backbones(self):
        return (self.primary_bb,) if self.share_backbone else (self.primary_bb, self.wrist_bb)

    def head_warmup(self, on: bool):
        for bb in self.backbones():
            bb.set_trainable_frozen() if on else bb.unfreeze()

    def set_norm(self, am, asd, pm, psd):
        self.state_enc.set_norm(am, asd, pm, psd)

    def _gate(self, f_img):
        g = self.gate                                    # (out_dim,)
        if self.gate_dyn:
            g = g + self.gate_net(f_img)                 # (out_dim,) + (B,1) -> (B,out_dim) broadcast
        return torch.sigmoid(g)

    def _proprio_dropout(self, f_pro):
        """Zero the PROPRIO correction per sample (train only); the image base is NEVER dropped -- it is
        the workhorse the metric is being forced onto. mod_dropout=0 (default) => no-op."""
        if not self.training or self.mod_dropout <= 0:
            return f_pro
        keep = (torch.rand(f_pro.shape[0], 1, device=f_pro.device) >= self.mod_dropout).float()
        return f_pro * keep

    def forward(self, primary, wrist, prev, proprio):   # prev accepted but IGNORED (self-state = proprio only)
        f_img = self.img_ln(self.img_head(torch.cat([self.primary_bb(primary), self.wrist_bb(wrist)], dim=-1)))
        f_pro = self._proprio_dropout(self.state_ln(self.state_enc(prev, proprio)))
        z = self.img_proj(f_img) + self._gate(f_img) * self.pro_proj(f_pro)
        return F.normalize(z, dim=-1)


class StateActionMetricEncoder(nn.Module):
    """prev-actions + proprio -> one L2-normalized embedding, with NO image. The LEARNED counterpart of
    the fixed ``proprio_prev`` retrieval key (it sees the same inputs but fits the action-metric), and
    the ``z_N1`` base for the residual fused design.

    DROP-IN for ``MultiModalActionMetricEncoder``: identical ``forward(primary, wrist, prev, proprio)``
    signature -- the image args are ACCEPTED BUT IGNORED -- so the training loop, ``eval.embed_all_mm``
    and ``obs_embed.compute_fused_emb`` call it unchanged; plus the same ``backbones()`` / ``head_warmup``
    / ``set_norm`` surface. It has no image backbone, so ``backbones()`` is empty (all params train at the
    head-lr) and ``head_warmup`` is a no-op. The state branch z-scores prev/proprio internally via its
    own buffers (``set_norm``), exactly like the fused encoder, so the data path is identical.
    """

    def __init__(self, out_dim=128, state_dim=256, hidden=None, dropout=0.1, state_mode="mlp",
                 prev_steps=16, act_dim=7, proprio_dim=9):
        super().__init__()
        hidden = int(hidden) if hidden else state_dim
        self.state_enc = StateEncoder(prev_steps, act_dim, proprio_dim, state_dim,
                                      mode=state_mode, dropout=dropout)
        self.state_ln = nn.LayerNorm(state_dim)
        self.head = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim))
        self.out_dim = out_dim

    def backbones(self):
        return ()                      # no image backbone -> trunk_parameters loop is empty; all params -> head-lr

    def head_warmup(self, on: bool):
        pass                           # no image backbone to freeze/unfreeze during warmup

    def set_norm(self, am, asd, pm, psd):
        self.state_enc.set_norm(am, asd, pm, psd)

    def features(self, prev, proprio):
        """The PRE-normalize 256-d head output (the state metric before L2-normalize). Used as the frozen
        ``z_N1`` base in ResidualImageEncoder, so the image residual is added before re-normalizing."""
        return self.head(self.state_ln(self.state_enc(prev, proprio)))

    def forward(self, primary, wrist, prev, proprio):   # primary/wrist accepted but IGNORED (drop-in signature)
        return F.normalize(self.features(prev, proprio), dim=-1)


def load_state_only_encoder(path, map_location="cpu"):
    """Build + load a StateActionMetricEncoder from a checkpoint, inferring all dims from the saved shapes
    (state_mode mlp/temporal/recency, widths, prev_steps). Used to supply the frozen base of a
    ResidualImageEncoder. Returns the loaded encoder (in train mode; caller freezes/eval()s as needed)."""
    sd = torch.load(path, map_location=map_location)
    state_dim = int(sd["state_ln.weight"].shape[0])
    hidden = int(sd["head.0.weight"].shape[0])          # head = Linear(state_dim,hidden)->...->Linear(hidden,out)
    out_dim = int(sd["head.3.weight"].shape[0])
    mode = ("recency" if "state_enc.pool_q" in sd
            else "temporal" if "state_enc.tok.weight" in sd else "mlp")
    act_dim = int(sd["state_enc.act_mean"].shape[0])
    proprio_dim = int(sd["state_enc.pro_mean"].shape[0])
    prev_steps = (int(sd["state_enc.pool_bias"].shape[0]) if mode == "recency"
                  else int(sd["state_enc.pos"].shape[1]) - 1 if mode == "temporal"
                  else (int(sd["state_enc.net.0.weight"].shape[1]) - proprio_dim) // act_dim)
    enc = StateActionMetricEncoder(out_dim=out_dim, state_dim=state_dim, hidden=hidden, state_mode=mode,
                                   prev_steps=prev_steps, act_dim=act_dim, proprio_dim=proprio_dim)
    enc.load_state_dict(sd)
    return enc
