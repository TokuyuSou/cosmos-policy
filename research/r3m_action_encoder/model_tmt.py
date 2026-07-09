"""Transition Metric Transformer (TMT) -- from-scratch fused retrieval encoder.

One EARLY-FUSION transformer replaces the pooled-branches + gated-addition design. Inputs per decision
point: previous action chunk (16x7), current proprio (9), current Theia patch tokens (2 views) and the
PREVIOUS decision point's Theia tokens (the obs where the prev chunk started, 4 dense rows back) -- a full
transition triple (o_{t-1}, a_prev, o_t). Attention computes change, cross-view (wrist<->primary) and
action-conditioned spatial relations directly, instead of pooling each branch to one vector first.

ASYMMETRIC towers, shared weights: the KEY (cache) side appends EFFECT tokens (post-execution deltas:
z-scored dproprio + scaled dTheia grid tokens, both on disk) -- "what picking this row accomplishes".
The QUERY (live) side has no effect input; the listwise training aligns it with desirable keys
(predicted-desired-effect matching, the surviving path from the effect-reranker probes).

BLOCKED FLOOR output (no additive cancellation): z = [ sqrt(1-w)*zs_hat ; sqrt(w)*zx_hat ] with zs_hat a
state-only MLP branch (z-scored prev+proprio) and zx_hat the transformer readout, each L2-normalized, w a
learned scalar starting small -- so the metric starts as (learned) state-only and the transformer earns
its share; state-confusable candidates give the state branch ~no gradient, forcing the transformer to
carry exactly the discrimination the state cannot (the diagnosed 85% error mass).

Aux heads (free dense supervision, both use only allowed inputs):
- IDM: predict a_prev from a state-masked forward (images only) -- grounds patch features in action space.
- EFF: predict own realized post-execution dproprio from the query forward -- shapes the query embedding
  into a desired-effect predictor.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TMTEncoder(nn.Module):
    def __init__(self, fd, n_img, n_eff, act_dim=7, prev_steps=16, proprio_dim=9,
                 d_model=256, nhead=8, layers=4, ffn=1024, dropout=0.1, half_dim=128, w_init=0.15,
                 lang_dim=0, eff_scene=False):
        """``fd`` Theia feature dim; ``n_img`` tokens per TIME SLICE (2 views * g_img^2); ``n_eff``
        delta-scene tokens (2 views * g_eff^2)."""
        super().__init__()
        self.fd, self.n_img, self.n_eff = fd, n_img, n_eff
        self.prev_steps, self.act_dim, self.proprio_dim = prev_steps, act_dim, proprio_dim

        # ---- state floor branch (z-scored prev+proprio -> half_dim, L2-normalized outside) ----
        sdim = prev_steps * act_dim + proprio_dim
        self.state_mlp = nn.Sequential(nn.Linear(sdim, 512), nn.GELU(), nn.Dropout(dropout),
                                       nn.Linear(512, 512), nn.GELU(), nn.Linear(512, half_dim))

        # ---- token projections (LN on raw Theia features first: scale-frees the frozen trunk) ----
        self.img_ln = nn.LayerNorm(fd)
        self.img_proj = nn.Linear(fd, d_model)
        self.act_proj = nn.Linear(act_dim, d_model)
        self.pro_proj = nn.Linear(proprio_dim, d_model)
        self.effs_ln = nn.LayerNorm(fd)
        self.effs_proj = nn.Linear(fd, d_model)
        self.effp_proj = nn.Linear(proprio_dim, d_model)

        # ---- learned embeddings: readout token, positions, time (cur/prev), modality types ----
        def _p(*shape):
            return nn.Parameter(torch.randn(*shape) * 0.02)
        self.emb_tok = _p(1, 1, d_model)
        self.img_pos = _p(1, n_img, d_model)          # shared across the two time slices
        self.time_emb = _p(2, 1, d_model)             # 0=current, 1=previous
        self.act_pos = _p(1, prev_steps, d_model)
        self.eff_pos = _p(1, n_eff, d_model)
        self.type_emb = _p(5, 1, d_model)             # img / act / pro / eff_scene / eff_pro

        layer = nn.TransformerEncoderLayer(d_model, nhead, ffn, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.out_proj = nn.Linear(d_model, half_dim)

        self.idm_head = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, prev_steps * act_dim))
        self.eff_head = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, proprio_dim))

        # C2: optional [LANG] instruction token (frozen-T5 embedding -> 1 token in BOTH towers).
        # lang_dim=0 (default) creates NOTHING -> old checkpoints keep their exact key set.
        self.lang_dim = int(lang_dim)
        if self.lang_dim > 0:
            self.lang_proj = nn.Linear(self.lang_dim, d_model)
            self.lang_type = _p(1, 1, d_model)

        # EFF++ (dense desired-effect prediction, QUERY side): n_eff learned [EFFPRED] read-out tokens that
        # (attention-masked so they never perturb z) predict this row's per-cell post-execution Δscene
        # (tok_eff[post]-tok_eff[cur])/dss -- the exact quantity the KEY tower ingests. eff_scene=False
        # (default) creates NOTHING -> old checkpoints keep their exact key set; the metric z is unchanged.
        self.eff_scene = bool(eff_scene)
        if self.eff_scene:
            self.effpred_tok = _p(1, 1, d_model)            # shared learned base (per-cell position = eff_pos)
            self.effpred_type = _p(1, 1, d_model)           # new modality type (query-side effect predictor)
            self.effscene_head = nn.Linear(d_model, fd)     # per-token d_model -> Theia feature (Δscene/dss)

        self.w_logit = nn.Parameter(torch.tensor(float(math.log(w_init / (1 - w_init)))))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

        for name in ("am", "asd", "pm", "psd", "dpm", "dps"):
            self.register_buffer(name, torch.zeros(act_dim if name in ("am", "asd") else proprio_dim))
        self.register_buffer("dss", torch.ones(fd))    # delta-scene per-dim scale

    def set_norm(self, am, asd, pm, psd):
        for n, v in zip(("am", "asd", "pm", "psd"), (am, asd, pm, psd)):
            getattr(self, n).copy_(torch.as_tensor(np.asarray(v), dtype=torch.float32))

    def set_task_norm(self, tam, tasd, tpm, tpsd):
        """C1: PER-TASK input z-score stats ((T,7),(T,7),(T,9),(T,9)). Registered LAZILY so single-task /
        old checkpoints keep their exact state-dict keys; used only when a ``task`` id tensor is passed."""
        for n, v in (("task_am", tam), ("task_asd", tasd), ("task_pm", tpm), ("task_psd", tpsd)):
            self.register_buffer(n, torch.as_tensor(np.asarray(v), dtype=torch.float32,
                                                    device=self.am.device))

    def _norm_stats(self, task):
        """(am, asd, pm, psd) -- per-row task stats when ``task`` ids given and set_task_norm was called,
        else the global buffers (bit-identical legacy path)."""
        if task is None or not hasattr(self, "task_am"):
            return self.am, self.asd, self.pm, self.psd
        return (self.task_am[task][:, None, :], self.task_asd[task][:, None, :],
                self.task_pm[task], self.task_psd[task])

    def set_eff_norm(self, dpm, dps, dss):
        for n, v in zip(("dpm", "dps", "dss"), (dpm, dps, dss)):
            getattr(self, n).copy_(torch.as_tensor(np.asarray(v), dtype=torch.float32))

    # ---------------------------------------------------------------- trunk
    def _trunk(self, img_cur, img_prev, prev=None, proprio=None, eff_s=None, eff_p=None,
               task=None, lang=None, effscene=False):
        """(B, n_img, fd) x2 [+ state (B,16,7)/(B,9)] [+ effect (B,n_eff,fd)/(B,9)] -> EMB readout (B, d).
        prev/proprio None = state-masked forward (IDM); eff_* None = query mode. ``task`` (B,) selects
        per-task input stats (C1); ``lang`` (B, lang_dim) appends the instruction token (C2).
        ``effscene`` (EFF++, query side): also append n_eff [EFFPRED] tokens, attention-masked so the
        existing tokens (incl. the readout) never attend to them -> the readout is BIT-IDENTICAL to the
        no-EFFPRED forward (so train-z == deploy-z); returns (readout, per-cell Δscene prediction)."""
        B = img_cur.shape[0]
        ic = self.img_proj(self.img_ln(img_cur)) + self.img_pos + self.time_emb[0] + self.type_emb[0]
        ip = self.img_proj(self.img_ln(img_prev)) + self.img_pos + self.time_emb[1] + self.type_emb[0]
        toks = [self.emb_tok.expand(B, -1, -1), ic, ip]
        if prev is not None:
            am, asd, pm, psd = self._norm_stats(task)
            a = self.act_proj((prev - am) / asd) + self.act_pos + self.type_emb[1]
            p = self.pro_proj((proprio - pm) / psd)[:, None] + self.type_emb[2]
            toks += [a, p]
        if lang is not None:
            toks.append(self.lang_proj(lang)[:, None, :] + self.lang_type)
        if eff_s is not None:
            es = self.effs_proj(self.effs_ln(eff_s / self.dss)) + self.eff_pos + self.type_emb[3]
            ep = self.effp_proj((eff_p - self.dpm) / self.dps)[:, None] + self.type_emb[4]
            toks += [es, ep]
        seq = torch.cat(toks, dim=1)
        if not (effscene and self.eff_scene):
            return self.enc(seq)[:, 0]
        S0 = seq.shape[1]                                                    # existing tokens
        ep_tok = (self.effpred_tok + self.eff_pos + self.effpred_type).expand(B, -1, -1)
        full = torch.cat([seq, ep_tok], dim=1)
        # additive float mask: existing tokens (incl. readout) get a large-negative score on the [EFFPRED]
        # columns -> their attention weight there is 0 (exp(-1e9)=0), so the readout z is bit-identical to
        # the no-[EFFPRED] forward. [EFFPRED] rows are unmasked (attend to all). A large FINITE negative
        # (not -inf) avoids the online-softmax NaN in the flash/mem-efficient SDPA backend.
        mask = torch.zeros(full.shape[1], full.shape[1], device=full.device, dtype=torch.float32)
        mask[:S0, S0:] = -1e9
        out = self.enc(full, mask=mask)
        return out[:, 0], self.effscene_head(out[:, S0:])                   # (B, d), (B, n_eff, fd)

    # ---------------------------------------------------------------- public
    def embed(self, img_cur, img_prev, prev, proprio, eff_s=None, eff_p=None, task=None, lang=None,
              ret_effscene=False):
        """Retrieval embedding. eff_* given = KEY (cache) mode; absent = QUERY (live) mode.
        Returns (z (B, 2*half_dim) unit-norm, h (B, d) trunk readout for aux heads); with
        ``ret_effscene`` also the EFF++ per-cell Δscene prediction (B, n_eff, fd). z is unchanged either
        way (the [EFFPRED] tokens are attention-masked out of the readout)."""
        if ret_effscene:
            h, eff_scene_pred = self._trunk(img_cur, img_prev, prev, proprio, eff_s, eff_p,
                                            task=task, lang=lang, effscene=True)
        else:
            h = self._trunk(img_cur, img_prev, prev, proprio, eff_s, eff_p, task=task, lang=lang)
        zx = F.normalize(self.out_proj(h), dim=-1)
        am, asd, pm, psd = self._norm_stats(task)
        sflat = torch.cat([((prev - am) / asd).reshape(len(prev), -1),
                           (proprio - pm) / psd], dim=-1)
        zs = F.normalize(self.state_mlp(sflat), dim=-1)
        w = torch.sigmoid(self.w_logit)
        z = torch.cat([(1 - w).sqrt() * zs, w.sqrt() * zx], dim=-1)
        return (z, h, eff_scene_pred) if ret_effscene else (z, h)

    def idm(self, img_cur, img_prev):
        """State-masked forward -> predicted z-scored prev chunk (B, 16*7)."""
        return self.idm_head(self._trunk(img_cur, img_prev))

    def eff_pred(self, h):
        """Query readout -> predicted z-scored own dproprio (B, 9)."""
        return self.eff_head(h)

    def scores(self, zq, zk):
        """(B,D) queries x (B,M,D) keys -> (B,M) similarity logits (cosine * learned scale). The cap
        ``scale_max`` (plain attribute, default 100 = legacy) exists because a target sharper than the
        geometry can express turns logit_scale into a runaway ratchet (observed 10->64 with oracle-
        injected targets): capping keeps the CE pressure on the GEOMETRY instead."""
        return self.logit_scale.clamp(max=float(np.log(getattr(self, "scale_max", 100.0)))).exp() \
            * (zq[:, None, :] * zk).sum(-1)
