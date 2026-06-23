"""Lightweight action predictor: predict the next 16 actions from the last 16
executed actions + VLA-predicted self-state + Cosmos future-image latent.

Token-based Transformer encoder with learnable query tokens, predicting a
residual from a repeat-last anchor (mean + log_std per action dim).

`img_mode` selects how the future-image latent (V, 16, 28, 28) is consumed:
  none     - ignore the future image (ablation)
  mean     - per-channel global mean  -> 1 token        (lightest)
  meanstd  - per-channel mean & std   -> 1 token
  grid     - adaptive g x g spatial pool -> V*g*g tokens (cheap spatial)
  conv     - small CNN over the latent -> 16 tokens     (expressive)
  full     - flatten the whole latent  -> 1 token       ("use all")
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision import models

from common import ACTION_DIM, LATENT_C, LATENT_H, LATENT_W, PROPRIO_DIM

_R3M_RESNET18 = "/workspace/openpi/checkpoints/r3m/r3m_resnet18.pt"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SpatialVisionEncoder(nn.Module):
    """Shared R3M ResNet-18 -> a small spatial token GRID per view (end-to-end trainable top block).

    Both views (primary, wrist) pass through the SAME backbone; their tokens occupy distinct slots in
    the predictor's sequence, so the predictor's learned positional embedding gives each token its
    (camera, grid-cell) identity -- the multi-camera ACT recipe. By default only ``layer4`` + the 1x1
    projection train; the R3M-pretrained lower blocks stay frozen (cheap, stable). Forward takes raw
    uint8 (B,3,224,224) frames and applies the R3M preprocessing (/255 + ImageNet-normalize) inside.
    """

    def __init__(self, width, grid=4, init="r3m", freeze="layer4"):
        super().__init__()
        net = models.resnet18(weights=None)
        sd = torch.load(_R3M_RESNET18 if init in ("r3m", "", None) else init, map_location="cpu")
        sd = {k[len("primary_bb.net."):]: v for k, v in sd.items() if k.startswith("primary_bb.net.")} or sd
        net.load_state_dict(sd, strict=False)  # fc is absent in the R3M checkpoint
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3, net.layer4)
        for p in self.stem.parameters():
            p.requires_grad = (freeze == "all")
        if freeze == "layer4":
            for p in self.stem[7].parameters():   # stem[7] = layer4
                p.requires_grad = True
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj = nn.Conv2d(512, width, 1)
        self.norm = nn.LayerNorm(width)
        self.grid, self.n_tokens = grid, 2 * grid * grid
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _tokens(self, x):                          # (B,3,224,224) uint8/[0,255] -> (B, grid*grid, width)
        x = (x.float() / 255.0 - self.mean) / self.std
        f = self.pool(self.proj(self.stem(x)))     # (B,width,g,g)
        return self.norm(f.flatten(2).transpose(1, 2))

    def forward(self, primary, wrist):             # -> (B, 2*grid*grid, width)
        return torch.cat([self._tokens(primary), self._tokens(wrist)], dim=1)


def mlp_proj(in_dim: int, width: int) -> nn.Module:
    """LayerNorm -> Linear -> GELU -> Linear projection to `width`."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, width),
        nn.GELU(),
        nn.Linear(width, width),
    )


class ImageEncoder(nn.Module):
    """Encode future-image latent (B, V, 16, 28, 28) -> (B, n_tokens, width)."""

    def __init__(self, mode: str, n_views: int, width: int, grid: int = 4):
        super().__init__()
        self.mode = mode
        self.n_views = n_views
        self.grid = grid
        C = LATENT_C
        if mode == "none":
            self.n_tokens = 0
        elif mode == "mean":
            self.proj = mlp_proj(n_views * C, width)
            self.n_tokens = 1
        elif mode == "meanstd":
            self.proj = mlp_proj(n_views * C * 2, width)
            self.n_tokens = 1
        elif mode == "grid":
            self.proj = mlp_proj(C, width)
            self.n_tokens = n_views * grid * grid
        elif mode == "conv":
            self.cnn = nn.Sequential(
                nn.Conv2d(n_views * C, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(grid),
            )
            self.proj = mlp_proj(128, width)
            self.n_tokens = grid * grid
        elif mode == "full":
            self.proj = mlp_proj(n_views * C * LATENT_H * LATENT_W, width)
            self.n_tokens = 1
        else:
            raise ValueError(f"unknown img_mode={mode}")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = img.shape
        if self.mode == "none":
            return img.new_zeros(B, 0, 0)
        if self.mode == "mean":
            feat = img.mean(dim=(3, 4)).reshape(B, V * C)
            return self.proj(feat).unsqueeze(1)
        if self.mode == "meanstd":
            m = img.mean(dim=(3, 4))
            s = img.std(dim=(3, 4))
            feat = torch.cat([m, s], dim=-1).reshape(B, V * C * 2)
            return self.proj(feat).unsqueeze(1)
        if self.mode == "grid":
            pooled = nn.functional.adaptive_avg_pool2d(img.reshape(B * V, C, H, W), self.grid)  # (B*V,C,g,g)
            pooled = pooled.reshape(B, V, C, self.grid * self.grid).permute(0, 1, 3, 2).reshape(B, V * self.grid * self.grid, C)
            return self.proj(pooled)  # (B, V*g*g, width)
        if self.mode == "conv":
            fmap = self.cnn(img.reshape(B, V * C, H, W))  # (B,128,g,g)
            tok = fmap.flatten(2).transpose(1, 2)  # (B, g*g, 128)
            return self.proj(tok)
        if self.mode == "full":
            return self.proj(img.reshape(B, -1)).unsqueeze(1)


class ActionPredictor(nn.Module):
    def __init__(
        self,
        n_views: int = 1,
        img_mode: str = "primary_conv",
        pred_h: int = 16,
        prev_h: int = 16,
        width: int = 384,
        depth: int = 6,
        heads: int = 6,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        grid: int = 4,
        use_obs_emb: bool = False,
        obs_dim: int = 128,
        vis_mode: str = "none",
        vis_grid: int = 4,
        vis_init: str = "r3m",
        vis_freeze: str = "layer4",
    ):
        super().__init__()
        self.pred_h = pred_h
        self.prev_h = prev_h
        self.width = width

        self.act_proj = mlp_proj(ACTION_DIM, width)  # shared over prev-action tokens
        self.state_proj = mlp_proj(PROPRIO_DIM, width)
        self.img_enc = ImageEncoder(img_mode, n_views, width, grid=grid)
        # Optional precomputed observation-embedding token (e.g. the R3M action-metric encoder):
        # one global vector per frame -> one token, fused by attention like any other modality.
        self.obs_proj = mlp_proj(obs_dim, width) if use_obs_emb else None
        # Optional end-to-end SPATIAL vision: a grid of R3M tokens per view that the query tokens
        # attend to (ACT-style). Takes raw frames; runs the backbone inside forward.
        self.vis = SpatialVisionEncoder(width, vis_grid, vis_init, vis_freeze) if vis_mode == "spatial" else None
        self.query = nn.Parameter(torch.randn(pred_h, width) * 0.02)

        n_vis = self.vis.n_tokens if self.vis is not None else 0
        n_tokens = prev_h + 1 + self.img_enc.n_tokens + (1 if use_obs_emb else 0) + n_vis + pred_h
        self.pos = nn.Parameter(torch.randn(n_tokens, width) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=ffn_mult * width,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head_mean = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ACTION_DIM))
        self.head_logstd = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ACTION_DIM))

    def forward(self, prev_actions, state, future_img, obs_emb=None, image=None, wrist=None):
        B = prev_actions.shape[0]
        prev_tok = self.act_proj(prev_actions)  # (B,prev_h,width)
        state_tok = self.state_proj(state).unsqueeze(1)  # (B,1,width)
        tokens = [prev_tok, state_tok]
        if self.img_enc.n_tokens > 0:
            tokens.append(self.img_enc(future_img))
        if self.obs_proj is not None:
            assert obs_emb is not None, "model built with use_obs_emb=True but obs_emb not provided"
            tokens.append(self.obs_proj(obs_emb).unsqueeze(1))  # (B,1,width) observation-embedding token
        if self.vis is not None:
            assert image is not None and wrist is not None, "vis_mode='spatial' needs image & wrist frames"
            tokens.append(self.vis(image, wrist))  # (B, 2*g*g, width) spatial vision tokens
        tokens.append(self.query.unsqueeze(0).expand(B, -1, -1))  # (B,pred_h,width)
        x = torch.cat(tokens, dim=1) + self.pos.unsqueeze(0)
        x = self.encoder(x)
        q = x[:, -self.pred_h :]  # query outputs
        mean = self.head_mean(q)  # (B,pred_h,7) residual mean (normalized action space)
        log_std = self.head_logstd(q).clamp(-5.0, 2.0)
        return mean, log_std


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
