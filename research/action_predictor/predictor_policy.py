"""Load a trained action predictor and run it for closed-loop control.

Handles input normalization, the configured state source, future-image views, and the
repeat-last residual decode. Inputs come from what is available at a skip decision point:
  - prev_actions      : last 16 executed actions (LOCAL, always available)
  - current_proprio   : robot proprio at the decision point (LOCAL)
  - cached_future_*    : the last CLOUD call's future-proprio / future-image latent (CACHED)
The state source (from the run's config) selects which proprio the model consumes.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from dataset import DEFAULT_STATE_SOURCE, Normalizers
from model import ActionPredictor

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_ckpt(path: str, run_dir: str) -> str:
    """Resolve a (possibly relative) checkpoint path recorded in a run's config: try as-is, then
    relative to this module's dir (the cwd the path was authored against), then relative to run_dir."""
    for cand in (path, os.path.join(HERE, path), os.path.join(run_dir, path)):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"obs-emb checkpoint not found: {path!r} (tried as-is / under {HERE} / under {run_dir})")


class PredictorPolicy:
    def __init__(self, run_dir: str, device: str | None = None):
        self.run_dir = run_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        a = cfg["args"]
        self.img_mode = a["img_mode"]
        self.views = cfg["views"]
        self.state_source = cfg.get("state_source", DEFAULT_STATE_SOURCE)
        self.norm = Normalizers.from_dict(dict(np.load(os.path.join(run_dir, "normalizers.npz"))))
        # Observation-embedding predictor: if the run was trained with --obs-emb-ckpt, rebuild the
        # frozen R3M encoder + the extra input token. Detected purely from the saved config, so the
        # runner / fusion policy need no special-casing.
        self.use_obs_emb = bool(a.get("obs_emb_ckpt", ""))
        self.obs_dim = 128
        self.model = ActionPredictor(
            n_views=len(self.views), img_mode=a["img_mode"], width=a["width"],
            depth=a["depth"], heads=a["heads"], dropout=a["dropout"], grid=a["grid"],
            use_obs_emb=self.use_obs_emb, obs_dim=self.obs_dim,
        ).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=self.device)["model"])
        self.model.eval()
        self.encoder = None
        if self.use_obs_emb:
            from obs_embed import load_corr_encoder
            self.encoder = load_corr_encoder(_resolve_ckpt(a["obs_emb_ckpt"], run_dir), self.device)

    @torch.no_grad()
    def _obs_emb(self, current_image, current_wrist):
        """Live frame pair -> (1, obs_dim) embedding, with the SAME preprocessing as training:
        ascontiguous uint8 (matches collect_data._u8) -> encoder (/255 + ImageNet-norm at 224)."""
        assert current_image is not None and current_wrist is not None, \
            "obs-emb predictor requires current_image and current_wrist at the decision point"
        p = np.ascontiguousarray(current_image).astype(np.uint8)
        w = np.ascontiguousarray(current_wrist).astype(np.uint8)
        assert p.shape[:2] == (224, 224) and w.shape[:2] == (224, 224), \
            f"expected 224x224 frames (as in training), got {p.shape} / {w.shape}"
        pt = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0).to(self.device)
        wt = torch.from_numpy(w).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.encoder(pt, wt)  # (1, obs_dim), L2-normalized

    @torch.no_grad()
    def predict_chunk(self, prev_actions, current_proprio, cached_future_proprio, cached_future_img,
                      current_image=None, current_wrist=None) -> np.ndarray:
        """Return the next-16 action chunk in physical scale, shape (16, 7). ``current_image`` /
        ``current_wrist`` are the live decision-point frames, used only by an obs-emb predictor."""
        n = self.norm
        state_vec = current_proprio if self.state_source == "actual_next_proprio" else cached_future_proprio
        prev_n = (prev_actions - n.act_mean) / n.act_std  # (16,7)
        state_n = (state_vec - n.proprio_mean) / n.proprio_std  # (9,)
        if cached_future_img is not None:
            img = np.asarray(cached_future_img)[self.views]  # (V,16,28,28)
        else:  # img_mode == "none": value is ignored by the model
            img = np.zeros((len(self.views), 16, 28, 28), dtype=np.float32)
        img_n = (img - n.img_mean[None, :, None, None]) / n.img_std[None, :, None, None]

        prev_t = torch.from_numpy(prev_n).float().unsqueeze(0).to(self.device)
        state_t = torch.from_numpy(state_n).float().unsqueeze(0).to(self.device)
        img_t = torch.from_numpy(img_n).float().unsqueeze(0).to(self.device)
        obs_t = self._obs_emb(current_image, current_wrist) if self.use_obs_emb else None
        mean, _ = self.model(prev_t, state_t, img_t, obs_t)  # residual (1,16,7)
        anchor_n = prev_n[-1]  # (7,)
        pred_n = anchor_n[None, :] + mean[0].cpu().numpy()  # (16,7)
        return (pred_n * n.act_std + n.act_mean).astype(np.float32)
