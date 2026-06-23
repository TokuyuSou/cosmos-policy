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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

R3M_RESNET18 = "/workspace/openpi/checkpoints/r3m/r3m_resnet18.pt"

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

    def forward(self, x):  # (B,3,224,224) uint8/float in [0,255] -> (B, feat_dim)
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        return self.net(x)


class ActionMetricEncoder(nn.Module):
    """Two views -> shared R3M backbone -> concat -> fusion MLP -> L2-normalized embedding."""

    def __init__(self, backbone="resnet18", init="r3m", freeze="layer4",
                 out_dim=128, hidden=512, dropout=0.1, share_backbone=True):
        super().__init__()
        self.primary_bb = R3MBackbone(backbone, init, freeze)
        self.wrist_bb = self.primary_bb if share_backbone else R3MBackbone(backbone, init, freeze)
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
    """

    def __init__(self, prev_steps=16, act_dim=7, proprio_dim=9, out_dim=256,
                 mode="mlp", dropout=0.1, d_model=64, nhead=4, layers=2):
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
        else:
            raise ValueError(mode)
        self.out_dim = out_dim

    def set_norm(self, am, asd, pm, psd):
        for buf, val in (("act_mean", am), ("act_std", asd), ("pro_mean", pm), ("pro_std", psd)):
            getattr(self, buf).copy_(torch.as_tensor(val, dtype=torch.float32))

    def forward(self, prev, proprio):           # prev (B,16,7), proprio (B,9)
        prev = (prev - self.act_mean) / self.act_std
        proprio = (proprio - self.pro_mean) / self.pro_std
        if self.mode == "mlp":
            x = torch.cat([prev.flatten(1), proprio], dim=-1)
            return self.net(x)
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
                 proprio_dim=9):
        super().__init__()
        self.primary_bb = R3MBackbone(backbone, init, freeze)
        self.wrist_bb = self.primary_bb if share_backbone else R3MBackbone(backbone, init, freeze)
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
            # starts near zero (init -2 -> sigmoid ~0.12) so the metric begins as the learned-state
            # metric (>= proprio_prev on the same input) and only adds image where it helps -- a floor
            # near the baseline with upside, the structure most likely to *clearly* beat the key.
            self.state_proj = nn.Linear(state_dim, out_dim)
            self.img_proj = nn.Linear(img_dim, out_dim)
            self.gate = nn.Parameter(torch.full((out_dim,), -2.0))
        else:
            raise ValueError(fusion)
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

    def forward(self, primary, wrist, prev, proprio):
        fimg = self.img_head(torch.cat([self.primary_bb(primary), self.wrist_bb(wrist)], dim=-1))
        fimg = self.img_ln(fimg)
        fstate = self.state_ln(self.state_enc(prev, proprio))
        fimg, fstate = self._modality_dropout(fimg, fstate)
        if self.fusion == "concat":
            z = self.fuse(torch.cat([fstate, fimg], dim=-1))
        else:  # residual: state base + gated image correction
            z = self.state_proj(fstate) + torch.sigmoid(self.gate) * self.img_proj(fimg)
        return F.normalize(z, dim=-1)
