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

import torch
import torch.nn as nn

from common import ACTION_DIM, LATENT_C, LATENT_H, LATENT_W, PROPRIO_DIM


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
    ):
        super().__init__()
        self.pred_h = pred_h
        self.prev_h = prev_h
        self.width = width

        self.act_proj = mlp_proj(ACTION_DIM, width)  # shared over prev-action tokens
        self.state_proj = mlp_proj(PROPRIO_DIM, width)
        self.img_enc = ImageEncoder(img_mode, n_views, width, grid=grid)
        self.query = nn.Parameter(torch.randn(pred_h, width) * 0.02)

        n_tokens = prev_h + 1 + self.img_enc.n_tokens + pred_h
        self.pos = nn.Parameter(torch.randn(n_tokens, width) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=ffn_mult * width,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head_mean = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ACTION_DIM))
        self.head_logstd = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ACTION_DIM))

    def forward(self, prev_actions, state, future_img):
        B = prev_actions.shape[0]
        prev_tok = self.act_proj(prev_actions)  # (B,prev_h,width)
        state_tok = self.state_proj(state).unsqueeze(1)  # (B,1,width)
        tokens = [prev_tok, state_tok]
        if self.img_enc.n_tokens > 0:
            tokens.append(self.img_enc(future_img))
        tokens.append(self.query.unsqueeze(0).expand(B, -1, -1))  # (B,pred_h,width)
        x = torch.cat(tokens, dim=1) + self.pos.unsqueeze(0)
        x = self.encoder(x)
        q = x[:, -self.pred_h :]  # query outputs
        mean = self.head_mean(q)  # (B,pred_h,7) residual mean (normalized action space)
        log_std = self.head_logstd(q).clamp(-5.0, 2.0)
        return mean, log_std


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
