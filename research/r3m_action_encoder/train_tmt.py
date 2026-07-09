"""Train + offline-evaluate the Transition Metric Transformer (TMT) retrieval encoder.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/r3m_action_encoder/train_tmt.py \
        --data research/data/pnp_sink_to_counter_dense_img

Standalone: does NOT touch train.py/model.py/eval.py. Reuses data.py (same dense_img rows/splits) and
action_reranker/common.py (Theia token cache, N1/fused candidate building, post-row effect utilities).

Training = listwise oracle KD (the signal proven to extract image information on state-confusable sets):
per anchor, candidates = N1 top-M (state-confusable) + current-fused top-M (today's hard negatives) +
fresh random cross-episode rows; target = softmax(-action_dist/tau0). Aux: IDM (predict prev chunk from
the image pair alone) + EFF (predict own post-execution dproprio). Selection by VAL retrieval RMSE@1.

Offline benchmark IDENTICAL to eval.py's TEST table (db=train keys, query=test, top-1 / top-5-mean chunk,
non-grip z-scored action RMSE); proprio_prev and the current fused encoder are re-evaluated in-run as
sanity baselines (expect ~0.5595 / ~0.4848 on sink test).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch

from model_tmt import TMTEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "..", "action_reranker")

_spec = importlib.util.spec_from_file_location("ar_common", os.path.join(AR, "common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def prev_index(ep, rows=4):
    """(N,) index of the row where each row's PREV chunk started (same episode, ``rows`` rows back),
    else the row itself (zero-change fallback, mirrors TokenChange's graceful degradation)."""
    N = len(ep)
    pi = np.arange(N) - rows
    ok = (pi >= 0) & (ep[np.maximum(pi, 0)] == ep)
    return np.where(ok, pi, np.arange(N))


def episode_aug_tokens(data, grid, var_idx, seed, cache, device, bs=128):
    """EPISODE-CONSISTENT augmented Theia grid tokens: ONE random camera shift/scale + color transform is
    drawn PER (episode, view) and applied to EVERY frame of that episode, then frozen-Theia tokens are
    pooled exactly like precompute_theia_tokens. This simulates a fresh episode fingerprint (slightly
    different camera pose / lighting) while preserving intra-episode dynamics, so prev-frame and
    post-effect token DELTAS stay consistent within a variant. The rng seed excludes ``grid`` and episodes
    are iterated in a fixed order, so the g_img and g_eff caches of the same variant share identical
    per-episode transforms (trunk tokens and effect deltas correspond)."""
    if os.path.exists(cache):
        return np.load(cache)["tok"]
    import torch.nn.functional as Fnn
    rng = np.random.RandomState(seed * 100003 + var_idx * 977)
    bb = C.TheiaBackbone(C.TheiaBackbone.DEFAULT_MODEL, freeze="frozen").to(device).eval()
    g2 = grid * grid
    out = np.zeros((len(data.act), 2 * g2, bb.feat_dim), np.float32)

    def transform(x, p):                                     # x (b,3,H,W) float [0,255]
        th = torch.tensor([[1 / p["sc"], 0, p["tx"]], [0, 1 / p["sc"], p["ty"]]],
                          dtype=torch.float32, device=device)
        gr = Fnn.affine_grid(th[None].expand(len(x), -1, -1), x.shape, align_corners=False)
        x = Fnn.grid_sample(x, gr, padding_mode="border", align_corners=False)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * p["ct"] + m                            # contrast (mean-preserving)
        gray = x.mean(1, keepdim=True)
        x = (x - gray) * p["sa"] + gray                      # saturation
        return (x * p["br"]).clamp(0, 255)                   # brightness

    with torch.no_grad():
        for e in np.unique(data.ep):
            rows = np.where(data.ep == e)[0]
            prm = {view: dict(sc=rng.uniform(0.92, 1.08), tx=rng.uniform(-0.1, 0.1), ty=rng.uniform(-0.1, 0.1),
                              br=rng.uniform(0.85, 1.15), ct=rng.uniform(0.85, 1.15), sa=rng.uniform(0.85, 1.15))
                   for view in ("primary", "wrist")}
            for i in range(0, len(rows), bs):
                sl = rows[i:i + bs]
                toks = []
                for view, arr in (("primary", data.primary), ("wrist", data.wrist)):
                    x = torch.as_tensor(np.ascontiguousarray(arr[sl]), device=device).permute(0, 3, 1, 2).float()
                    t = bb.tokens(transform(x, prm[view]))
                    b_, n, c_ = t.shape
                    hw = int(round(n ** 0.5))
                    t = t.transpose(1, 2).reshape(b_, c_, hw, hw)
                    t = torch.nn.functional.adaptive_avg_pool2d(t, grid).reshape(b_, c_, g2).transpose(1, 2)
                    toks.append(t)
                out[sl] = torch.cat(toks, dim=1).cpu().numpy()
    del bb
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, tok=out)
    return out


def build_lang_embeddings(dirs, data, t5_pkl):
    """(N, 1024) mean-pooled frozen-T5 instruction embedding per row (episode-level). Episode ids follow
    data.collect's enumerate(files) order per dir with load_multi's cumulative offsets (reconstructed
    exactly: next offset = max global ep of this task + 1). Instructions missing from the VLA's T5 cache
    fall back to the task mean. Padding token rows (zero norm) are excluded from the mean-pool."""
    import pickle
    with open(t5_pkl, "rb") as f:
        cache = pickle.load(f)
    embs = np.zeros((len(data.act), 1024), np.float32)
    n_miss, off = 0, 0
    for ti, d in enumerate(dirs):
        files = C.D.list_success_episodes(d)
        vecs, rows_by_ep = [], []
        for ei, fp in enumerate(files):
            lang = str(np.load(fp, allow_pickle=True)["lang"])
            t = cache.get(lang)
            v = None
            if t is not None:
                e = torch.as_tensor(t, dtype=torch.float32).reshape(-1, t.shape[-1])
                v = e[e.norm(dim=-1) > 1e-6].mean(0).numpy()
            vecs.append(v)
            rows_by_ep.append(np.where(data.ep == ei + off)[0])
        found = [v for v in vecs if v is not None]
        assert found, f"no instruction of {d} found in {t5_pkl}"
        fb = np.mean(found, 0)
        for v, rows in zip(vecs, rows_by_ep):
            if v is None:
                n_miss += 1
            embs[rows] = v if v is not None else fb
        task_rows = np.where(data.task == ti)[0]
        assert set(np.unique(data.ep[task_rows])) <= {ei + off for ei in range(len(files))}, \
            "episode-id mapping mismatch (collect/file order changed?)"
        off = int(data.ep[task_rows].max()) + 1
    return embs, n_miss


def merge_candidates(*cand_arrays):
    """Row-wise union of candidate index arrays (each (N, Mi), pad -1) -> padded (N, Mmax)."""
    N = cand_arrays[0].shape[0]
    merged = []
    for i in range(N):
        u = np.unique(np.concatenate([c[i] for c in cand_arrays]))
        merged.append(u[u >= 0])
    M = max(len(m) for m in merged)
    out = np.full((N, M), -1, np.int64)
    for i, m in enumerate(merged):
        out[i, :len(m)] = m
    return out


class Embedder:
    """GPU-resident tensors + batched TMT embedding for arbitrary row index sets."""

    def __init__(self, data, tok_img, tok_eff, prev_i, post, device):
        self.dev = device
        f16 = lambda a: torch.as_tensor(a, dtype=torch.float16, device=device)
        f32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        if tok_img.ndim == 3:                                     # no augmentation -> single clean variant
            tok_img, tok_eff = tok_img[None], tok_eff[None]
        self.V = len(tok_img)                                     # variants (index 0 = CLEAN; eval uses 0)
        self.tok = f16(tok_img)                                   # (V, N, 2g2, fd) scene tokens
        self.prev_i = torch.as_tensor(prev_i, device=device)
        self.prev = f32(data.prev)
        self.pro = f32(data.proprio)
        self.act = f32(data.act)
        ok = post >= 0
        p0 = np.where(ok, post, np.arange(len(post)))
        okm = torch.as_tensor(ok, device=device)[:, None, None]
        self.eff_s = torch.stack([f16(tok_eff[v][p0] - tok_eff[v]) * okm for v in range(self.V)])
        self.eff_p = f32((data.proprio[p0] - data.proprio) * ok[:, None])
        self.task = None    # (N,) int64 on device -> per-task input norms (C1); set by main
        self.lang = None    # (N, 1024) fp16 on device -> [LANG] token (C2); set by main

    def tok_at(self, idx_t, v=None):
        """(cur, prev) fp32 scene tokens for rows ``idx_t`` under per-row variant ``v`` (None = clean)."""
        vv = 0 if v is None else v
        return self.tok[vv, idx_t].float(), self.tok[vv, self.prev_i[idx_t]].float()

    def batch(self, idx_t, key_mode, v=None):
        """Row indices (B,) tensor -> kwargs for model.embed (fp32 upcast of fp16 token gathers).
        ``v`` (B,) per-row augmentation variant indices; None = clean (all eval/deploy paths)."""
        cur, prv = self.tok_at(idx_t, v)
        kw = dict(img_cur=cur, img_prev=prv, prev=self.prev[idx_t], proprio=self.pro[idx_t])
        if self.task is not None:
            kw["task"] = self.task[idx_t]
        if self.lang is not None:
            kw["lang"] = self.lang[idx_t].float()
        if key_mode:
            es = self.eff_s[0 if v is None else v, idx_t]
            kw.update(eff_s=es.float(), eff_p=self.eff_p[idx_t])
        return kw

    @torch.no_grad()
    def embed_rows(self, model, rows, key_mode, bs=256):
        model.eval()
        out = []
        for i in range(0, len(rows), bs):
            idx = torch.as_tensor(rows[i:i + bs], device=self.dev)
            out.append(model.embed(**self.batch(idx, key_mode))[0].float().cpu())
        return torch.cat(out).numpy()


def retr_rmse(db_key, q_key, db_act, q_act, asd, device, ks=(1, 5), nd=7, ts=None):
    """eval.py's _retr_rmse verbatim: k-NN retrieved (k-averaged) chunk -> mean z-scored action RMSE over
    the first ``nd`` dims (7 = grip included, matching every published --include-grip TEST table).
    ``ts`` restricts the RMSE to the FIRST ts chunk steps (short-horizon benchmark); None = all 16."""
    Dm = torch.cdist(torch.as_tensor(q_key, device=device), torch.as_tensor(db_key, device=device))
    idx = Dm.topk(max(ks), largest=False).indices.cpu().numpy()
    if ts:
        db_act, q_act = db_act[:, :ts], q_act[:, :ts]
    out = {}
    for k in ks:
        ret = db_act[idx[:, :k]].mean(axis=1)
        e = ((ret - q_act) / asd)[..., :nd]
        out[k] = float(np.sqrt((e ** 2).mean(axis=(1, 2))).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fused-ckpt", default="", help="fused encoder defining the hard-negative recall "
                    "(default results/<task>/encoder_fused_corrsupcon_k8_grip.pt)")
    ap.add_argument("--g-img", type=int, default=7, help="Theia token grid per view for the trunk")
    ap.add_argument("--g-eff", type=int, default=4, help="Theia token grid per view for effect deltas")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--half-dim", type=int, default=128)
    ap.add_argument("--w-init", type=float, default=0.15)
    ap.add_argument("--n1-m", type=int, default=16)
    ap.add_argument("--fused-m", type=int, default=16, help="0 = no fused-encoder negatives (no trained "
                    "fused ckpt needed for TRAINING; operationally self-contained)")
    ap.add_argument("--self-m", type=int, default=0, help="SELF-MINED hard negatives: each epoch, each train "
                    "anchor's top-M cross-episode neighbours under the CURRENT model (query-vs-key) join the "
                    "hard pool. Adaptive difficulty, no external encoder; safe under listwise-KD (graded "
                    "targets absorb accidental near-positives)")
    ap.add_argument("--cand-keep", type=int, default=16, help="hard candidates SUBSAMPLED per step from "
                    "the merged N1+fused pool (stochastic hard-negative set)")
    ap.add_argument("--rand-m", type=int, default=8, help="fresh random cross-episode negatives per step")
    ap.add_argument("--target-steps", type=int, default=0,
                    help="KD relevance horizon: rank candidates by the action distance of the FIRST k "
                         "chunk steps only (0 = all 16, legacy). NOTE: unlike the old fused --target-steps,"
                         " the BENCHMARK (val selection + final val/test RMSE@1) stays FULL-16-step, so "
                         "results remain directly comparable to tmt_v2_d192L3 -- this tests whether a "
                         "shorter TRAINING horizon improves full-chunk retrieval.")
    ap.add_argument("--tau0", type=float, default=0.3, help="oracle soft-target temperature (reranker-proven)")
    ap.add_argument("--oracle-m", type=int, default=0,
                    help="inject this many GLOBAL cross-episode oracle positives (action-distance top-8, "
                         "softmax(-d/tau0)-weighted sample) into each anchor's KD candidate set -- the KD "
                         "target then explicitly teaches scoring the TRUE best rows high (they are often "
                         "absent from N1/self-mined pools). 0 = off.")
    ap.add_argument("--lam-align", type=float, default=0.0,
                    help="QUERY-QUERY oracle alignment: batch = seeds + their sampled global-oracle "
                         "partners; penalize squared embedding distance between paired QUERY embeddings "
                         "(the geometry listwise-KD never constrains; episode islands violate it). 0 = off.")
    ap.add_argument("--align-delta", type=float, default=0.0,
                    help="HINGE floor for the query-query oracle alignment: only pull pairs FARTHER than "
                         "this embedding distance (calibrated ~p50 of healthy oracle-pair distances, e.g. "
                         "0.25) -- transitive collapse becomes structurally impossible. 0 = legacy "
                         "attract-to-zero (mathematically identical to the original form).")
    ap.add_argument("--scale-max", type=float, default=100.0,
                    help="cap on exp(logit_scale) (100 = legacy). Use ~15 with sharp injected targets to "
                         "block the scale ratchet (unexpressible sharpness must not be faked by scale).")
    ap.add_argument("--lam-sym", type=float, default=0.0,
                    help="SYMMETRIC listwise KD: each unique KEY in the step also ranks the batch QUERIES "
                         "(CLIP-style reverse direction; oracle target = softmax(-action_dist/tau0) over "
                         "anchors, same-episode pairs masked). Constrains the key->query direction that "
                         "forward KD leaves free. 0 = off (skipped entirely -> legacy behavior).")
    ap.add_argument("--lam-corr", type=float, default=0.0,
                    help="GLOBAL distance-matching regularizer: maximize the Pearson correlation between "
                         "full-z embedding distances and action-chunk distances over all cross-episode "
                         "(anchor x unique-candidate) pairs in the step (the query-query/key-key geometry "
                         "listwise-KD leaves free; the fused encoder's corr loss precedent). 0 = off "
                         "(computation skipped entirely -> bit-identical legacy behavior).")
    ap.add_argument("--lam-idm", type=float, default=0.5)
    ap.add_argument("--lam-eff", type=float, default=0.5)
    ap.add_argument("--lam-eff-scene", type=float, default=0.0,
                    help="EFF++ (dense desired-effect): weight of the QUERY-side per-cell Δscene prediction "
                         "(Huber, delta=1). The query grows n_eff attention-masked [EFFPRED] tokens that "
                         "predict this row's post-execution (tok_eff[post]-tok_eff[cur])/dss -- the exact "
                         "quantity the KEY tower ingests. 0 = off (bit-identical: no params, no forward "
                         "change). Suggested 0.25-0.5.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16, help="anchors per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-grip", action="store_true", help="drop the grip dim from the KD relevance "
                    "(default: include grip, matching the deployed fused encoder)")
    ap.add_argument("--pertask-norm", action="store_true",
                    help="C1: per-task input z-scoring (task_am/asd/pm/psd from data) + per-task asd in "
                         "the KD relevance; default off = pooled stats (naive)")
    ap.add_argument("--lang-token", action="store_true",
                    help="C2: append one [LANG] token (frozen-T5 instruction embedding, per episode) to "
                         "BOTH towers; default off")
    ap.add_argument("--t5-pkl", default="/home/cao/.cache/huggingface/models--nvidia--Cosmos-Policy-"
                    "RoboCasa-Predict2-2B/snapshots/4b2a04c80d97202f86127ebec80461e8016ec1dc/"
                    "robocasa_t5_embeddings.pkl", help="VLA T5 instruction-embedding cache (pickle)")
    ap.add_argument("--aug-variants", type=int, default=0,
                    help="EPISODE-CONSISTENT image augmentation: precompute this many augmented Theia-token "
                         "copies (camera shift/scale + color per episode&view) and sample {clean,augs} "
                         "per row per step during training. Eval/deploy always uses clean. 0 = off "
                         "(bit-identical rng stream to previous runs).")
    ap.add_argument("--rows-per-chunk", type=int, default=4,
                    help="dense rows per 16-step chunk = 16 // collection stride (4 for stride-4 data, "
                         "16 for stride-1 *_s1 data); sets the prev-frame and post-effect row offsets")
    ap.add_argument("--train-substride", type=int, default=1,
                    help="keep every k-th within-episode row for TRAIN anchors/DB/mining (1 = all rows; "
                         "e.g. 4 on stride-1 data = stride-4-equivalent training density)")
    ap.add_argument("--eval-substride", type=int, default=1,
                    help="keep every k-th within-episode row for VAL/TEST queries (fix across conditions "
                         "so query sets are identical; DB density follows --train-substride)")
    ap.add_argument("--train-ep-cap", type=int, default=0,
                    help="cap the number of TRAINING episodes (val/test fixed); 0 = use all. Data-quantity "
                         "control -- a smaller cap is a seeded subset of a larger one.")
    ap.add_argument("--fail-anchors", action="store_true",
                    help="add FAILURE-episode rows (ep*_success=0.npz in --data) as extra KD/IDM/EFF "
                         "ANCHORS (query side only). The supervision target is VLA parity, so failed "
                         "rollouts carry valid per-row VLA teacher chunks AND the off-support states "
                         "where retrieval collapses at high skip. DB keys, norm stats and the val/test "
                         "benchmark stay success-only -> numbers directly comparable to legacy runs.")
    ap.add_argument("--fail-tail-drop", type=int, default=0,
                    help="with --fail-anchors: exclude the last K decision rows of each failure episode "
                         "from the anchor pool (timeout flailing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="tmt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    include_grip = not args.no_grip

    multi = "," in args.data
    dirs = [d for d in args.data.split(",") if d]
    if multi:  # NAIVE multi-task: pooled norms, no task/language conditioning; candidates/mining/eval all SAME-TASK
        data = C.D.load_multi(dirs, seed=args.seed)
        name = "multi_" + "+".join(sorted(os.path.basename(d.rstrip("/")).replace("_dense_img", "")
                                          .replace("pnp_", "") for d in dirs))
    else:
        data = C.load_data(args.data, seed=args.seed)
        name = os.path.basename(args.data.rstrip("/")).replace("_dense_img", "").replace("_dense", "")
    # DATA-QUANTITY control: keep val/test (and all norm stats below) FIXED, subsample only the TRAINING
    # episodes -> the ONLY variable is how many episodes the metric+cache are built from. The retained set
    # is seeded, so a smaller cap is a subset of a larger one (nested), and pm/psd are still fit on the
    # capped tr below (== what a real smaller collection would see). Tokens/fused caches are per-row (full
    # dataset), so they are shared across caps with no rebuild.
    if args.train_ep_cap > 0:
        tr_eps = np.unique(data.ep[data.tr])
        keep = np.sort(np.random.RandomState(args.seed).permutation(tr_eps)[:args.train_ep_cap])
        data.tr = data.tr & np.isin(data.ep, keep)
        print(f"[tmt] TRAIN-EP-CAP {args.train_ep_cap}: kept {len(keep)}/{len(tr_eps)} train eps "
              f"-> {int(data.tr.sum())} train rows (val/test unchanged)")
    fused_ckpt = args.fused_ckpt or os.path.join(HERE, "results", name, "encoder_fused_corrsupcon_k8_grip.pt")

    if args.train_substride > 1 or args.eval_substride > 1:
        pos_in_ep = np.zeros(len(data.ep), np.int64)   # within-episode row position (rows are contiguous)
        for e in np.unique(data.ep):
            r = np.where(data.ep == e)[0]
            pos_in_ep[r] = np.arange(len(r))
        if args.train_substride > 1:
            data.tr = data.tr & (pos_in_ep % args.train_substride == 0)
        if args.eval_substride > 1:
            data.va = data.va & (pos_in_ep % args.eval_substride == 0)
            data.te = data.te & (pos_in_ep % args.eval_substride == 0)
        print(f"[tmt] substride train={args.train_substride} eval={args.eval_substride} -> "
              f"tr {int(data.tr.sum())} / va {int(data.va.sum())} / te {int(data.te.sum())} rows")
    if args.fail_anchors:  # append AFTER split/caps (masks frozen) and BEFORE token precompute (rows final)
        assert not multi and args.oracle_m == 0 and args.lam_align == 0 and args.aug_variants == 0 \
            and args.fused_m == 0, "--fail-anchors v1: single-task, no oracle/align/aug/fused negatives"
        data = C.D.append_failures(data, args.data, tail_drop=args.fail_tail_drop)
    tok_tag = f"{name}_failq" if args.fail_anchors else name           # extended row set -> own token cache
    tok_img, fd, _ = C.precompute_theia_tokens(data, device, None, args.g_img,
                                               cache=os.path.join(AR, "cache", f"theia_tok_{tok_tag}_g{args.g_img}.npz"))
    tok_eff, _, _ = C.precompute_theia_tokens(data, device, None, args.g_eff,
                                              cache=os.path.join(AR, "cache", f"theia_tok_{tok_tag}_g{args.g_eff}.npz"))
    if args.aug_variants > 0:
        vs_img = [tok_img] + [episode_aug_tokens(data, args.g_img, v, args.seed,
                              os.path.join(AR, "cache", f"theia_tok_{name}_g{args.g_img}_aug{v}_s{args.seed}.npz"),
                              device) for v in range(args.aug_variants)]
        vs_eff = [tok_eff] + [episode_aug_tokens(data, args.g_eff, v, args.seed,
                              os.path.join(AR, "cache", f"theia_tok_{name}_g{args.g_eff}_aug{v}_s{args.seed}.npz"),
                              device) for v in range(args.aug_variants)]
        tok_img, tok_eff = np.stack(vs_img), np.stack(vs_eff)
        print(f"[tmt] episode-consistent augmentation: clean + {args.aug_variants} variants (V={len(vs_img)})")
    post = C.post_index(data.ep, rows=args.rows_per_chunk)
    prev_i = prev_index(data.ep, rows=args.rows_per_chunk)
    dpm, dps, dss = C.effect_norm_stats(data, tok_eff if tok_eff.ndim == 3 else tok_eff[0], post)
    E = Embedder(data, tok_img, tok_eff, prev_i, post, device)

    # ---- hard-negative candidate pools (anchors -> train db, cross-episode) ----
    tr_rows = np.where(data.tr)[0]
    if args.fail_anchors:  # anchors = success train rows + failure rows; db/keys stay success train only
        anchor_mask = data.tr | data.fail
        anchor_rows = np.where(anchor_mask)[0]
        assert (anchor_rows == np.concatenate([tr_rows, np.where(data.fail)[0]])).all()  # ascending order
    else:
        anchor_mask, anchor_rows = data.tr, tr_rows
    _, n1_c = C.build_candidates(data, anchor_mask, data.tr, args.n1_m, device)
    femb = None
    if os.path.exists(fused_ckpt) and not args.fail_anchors:  # baseline report (and negatives iff --fused-m > 0)
        femb = C.compute_fused_embeddings(data, fused_ckpt, device,
                                          cache=os.path.join(AR, "cache", f"fusedemb_{name}_"
                                                f"{os.path.splitext(os.path.basename(fused_ckpt))[0]}.npz"))
    if args.fused_m > 0:
        assert femb is not None, f"--fused-m>0 needs {fused_ckpt}"
        _, fu_c = C.build_candidates(data, data.tr, data.tr, args.fused_m, device, key_emb=femb)
        cand_pool = merge_candidates(n1_c, fu_c)                   # (Ntr, <=n1_m+fused_m)
    else:
        cand_pool = merge_candidates(n1_c)
    mined_c = np.full((len(anchor_rows), max(args.self_m, 1)), -1, np.int64)  # refreshed per epoch when --self-m
    asd_t = torch.as_tensor(data.asd, device=device)
    pm, psd = data.proprio[data.tr].mean(0), data.proprio[data.tr].std(0) + 1e-6

    lang_embs, lang_dim = None, 0
    if args.lang_token:
        lang_embs, n_miss = build_lang_embeddings(dirs, data, args.t5_pkl)
        lang_dim = lang_embs.shape[1]
        print(f"[tmt] C2 lang token: per-episode T5 embeddings ready (dim {lang_dim}, "
              f"{n_miss} instruction(s) missing -> task-mean fallback)")
    n_img, n_eff = tok_img.shape[-2], tok_eff.shape[-2]        # robust to the (V, N, ...) augmented stack
    model = TMTEncoder(fd, n_img, n_eff, act_dim=data.prev.shape[2], prev_steps=data.prev.shape[1],
                       proprio_dim=data.proprio.shape[1], d_model=args.d_model, nhead=args.nhead,
                       layers=args.layers, ffn=args.ffn, dropout=args.dropout, half_dim=args.half_dim,
                       w_init=args.w_init, lang_dim=lang_dim, eff_scene=args.lam_eff_scene > 0).to(device)
    model.scale_max = args.scale_max
    model.set_norm(data.am, data.asd, pm, psd)
    model.set_eff_norm(dpm, dps, dss)
    if args.pertask_norm:
        model.set_task_norm(data.task_am, data.task_asd, data.task_pm, data.task_psd)
        E.task = torch.as_tensor(data.task, device=device)
        print(f"[tmt] C1 per-task input norms: {len(data.tasks)} task(s)")
    if args.lang_token:
        E.lang = torch.as_tensor(lang_embs, dtype=torch.float16, device=device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[tmt] {name} N={len(data.act)} (tr {len(tr_rows)}) | img {n_img}x2t fd={fd} eff {n_eff} "
          f"| d={args.d_model} L{args.layers} h{args.nhead} halfdim={args.half_dim} | {n_par/1e6:.2f}M params")
    print(f"[tmt] cand pool/anchor: median {int(np.median((cand_pool >= 0).sum(1)))} "
          f"(n1 {args.n1_m} + fused {args.fused_m} merged) | keep {args.cand_keep} + rand {args.rand_m}/step "
          f"| tau0={args.tau0} grip={include_grip} | lam_idm={args.lam_idm} lam_eff={args.lam_eff}")

    # ---- in-run baselines on the SAME benchmark (sanity: must reproduce known numbers) ----
    va_rows, te_rows = np.where(data.va)[0], np.where(data.te)[0]
    pf = data.prev.reshape(len(data.prev), -1).astype(np.float32)
    pk = np.concatenate([(data.proprio - pm) / psd,
                         (pf - pf[tr_rows].mean(0)) / (pf[tr_rows].std(0) + 1e-6)], 1).astype(np.float32)
    nd = 7 if include_grip else 6
    TS = args.target_steps or None
    if TS:
        print(f"[tmt] SHORT-HORIZON benchmark: loss target AND val/test RMSE over the first {TS} chunk steps")
    for nm, key in ([] if multi else [("proprio_prev", pk)] + ([("fused[cur]", femb)] if femb is not None else [])):
        r_va = retr_rmse(key[tr_rows], key[va_rows], data.act[tr_rows], data.act[va_rows], data.asd, device, nd=nd, ts=TS)
        r_te = retr_rmse(key[tr_rows], key[te_rows], data.act[tr_rows], data.act[te_rows], data.asd, device, nd=nd, ts=TS)
        print(f"  [baseline] {nm:13s} val RMSE@1={r_va[1]:.4f} @5={r_va[5]:.4f} | "
              f"test RMSE@1={r_te[1]:.4f} @5={r_te[5]:.4f}")

    def eval_split(zdb, zq, q_rows):
        """(mean-metric dict, per-task rmse1 dict|None, print suffix). Multi: per-task DB + per-task asd
        (numbers directly comparable to each single-task run); selection = task-mean RMSE@1."""
        if not multi:
            return retr_rmse(zdb, zq, data.act[tr_rows], data.act[q_rows], data.asd, device, nd=nd, ts=TS), None, ""
        per = {}
        for ti, tn in enumerate(data.tasks):
            dm = data.task[tr_rows] == ti
            qm = data.task[q_rows] == ti
            per[tn] = retr_rmse(zdb[dm], zq[qm], data.act[tr_rows[dm]], data.act[q_rows[qm]],
                                data.task_asd[ti], device, nd=nd, ts=TS)
        r = {k: float(np.mean([v[k] for v in per.values()])) for k in (1, 5)}
        sfx = " |" + "".join(f" {tn.replace('pnp_', '')[:14]}:{v[1]:.4f}" for tn, v in per.items())
        return r, {tn: v[1] for tn, v in per.items()}, sfx

    task_tr_rows = [tr_rows[data.task[tr_rows] == ti] for ti in range(len(getattr(data, "tasks", ("x",))))]
    orc_idx = orc_w = None
    if args.oracle_m > 0 or args.lam_align > 0:
        K_OR = 8
        orc_idx = np.zeros((len(tr_rows), K_OR), np.int64)
        orc_w = np.zeros((len(tr_rows), K_OR), np.float32)
        act_tr = torch.as_tensor(data.act[tr_rows], device=device)
        for i in range(0, len(tr_rows), 256):
            sl = slice(i, min(i + 256, len(tr_rows)))
            da_o = C.act_dist_t(act_tr.unsqueeze(0), act_tr[sl].unsqueeze(1), asd_t, include_grip)
            bad = torch.as_tensor((data.ep[tr_rows][None, :] == data.ep[tr_rows[sl]][:, None])
                                  | (data.task[tr_rows][None, :] != data.task[tr_rows[sl]][:, None]),
                                  device=device)
            da_o.masked_fill_(bad, float("inf"))
            v, ix = da_o.topk(K_OR, largest=False)
            orc_idx[sl] = tr_rows[ix.cpu().numpy()]
            orc_w[sl] = torch.softmax(-v / args.tau0, dim=1).float().cpu().numpy()
        pos_of = np.full(len(data.act), -1, np.int64)
        pos_of[tr_rows] = np.arange(len(tr_rows))
        print(f"[tmt] GLOBAL ORACLE POSITIVES ready: cross-ep same-task top-{K_OR} per train row "
              f"(inject {args.oracle_m}/anchor into KD; align lam={args.lam_align})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.fail_anchors:
        print(f"[tmt] FAIL-ANCHORS: {len(anchor_rows)} anchors = {len(tr_rows)} success-train "
              f"+ {len(anchor_rows) - len(tr_rows)} failure rows (db/val/test success-only)")
    steps_per_epoch = max(1, len(anchor_rows) // args.batch)
    total, warm = args.epochs * steps_per_epoch, args.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    asd_task = (torch.as_tensor(np.asarray(data.task_asd), dtype=torch.float32, device=device)
                if (multi and args.pertask_norm) else None)
    rng = np.random.RandomState(args.seed)
    M = args.cand_keep + args.rand_m + args.oracle_m
    best = {"rmse1": 1e9, "epoch": -1, "state": None}
    hist = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(anchor_rows))
        sums = np.zeros(8)
        t0 = time.time()
        for i in range(0, steps_per_epoch * args.batch, args.batch):
            b = order[i:i + args.batch]
            if args.lam_align > 0:      # batch = seeds + their sampled global-oracle partners (query-query pairs)
                half = len(b) // 2
                seeds = b[:half]
                part = np.array([rng.choice(orc_idx[s], p=orc_w[s] / orc_w[s].sum()) for s in seeds])
                b = np.concatenate([seeds, pos_of[part]])
                assert (b >= 0).all(), "oracle partner not a train row"
            a_glob = anchor_rows[b]
            # per-anchor candidates: subsample the merged hard pool + fresh cross-ep randoms
            cmat = np.empty((len(b), M), np.int64)
            for r, (bi, ai) in enumerate(zip(b, a_glob)):
                pool = np.unique(np.concatenate([cand_pool[bi], mined_c[bi]]))
                pool = pool[pool >= 0]
                hard = rng.choice(pool, min(args.cand_keep, len(pool)), replace=False)
                if args.oracle_m > 0:
                    pw = orc_w[bi] / orc_w[bi].sum()
                    osel = orc_idx[bi][rng.choice(len(orc_idx[bi]), min(args.oracle_m, len(orc_idx[bi])),
                                                  replace=False, p=pw)]
                    hard = np.unique(np.concatenate([hard, osel]))
                n_rand = M - len(hard)
                tpool = task_tr_rows[data.task[ai]]                     # SAME-TASK rows (deploy-faithful)
                rand = tpool[rng.randint(0, len(tpool), n_rand * 3)]
                rand = rand[data.ep[rand] != data.ep[ai]][:n_rand]
                while len(rand) < n_rand:  # pathological same-ep streak
                    extra = tpool[rng.randint(0, len(tpool), n_rand)]
                    rand = np.concatenate([rand, extra[data.ep[extra] != data.ep[ai]]])[:n_rand]
                cmat[r] = np.concatenate([hard, rand])
            uniq, inv = np.unique(cmat, return_inverse=True)
            a_t = torch.as_tensor(a_glob, device=device)
            u_t = torch.as_tensor(uniq, device=device)
            v_a = torch.as_tensor(rng.randint(0, E.V, len(a_glob)), device=device) if E.V > 1 else None
            v_u = torch.as_tensor(rng.randint(0, E.V, len(uniq)), device=device) if E.V > 1 else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.lam_eff_scene > 0:
                    zq, h, eff_scene_pred = model.embed(**E.batch(a_t, key_mode=False, v=v_a),
                                                        ret_effscene=True)
                else:
                    zq, h = model.embed(**E.batch(a_t, key_mode=False, v=v_a))
                zu, _ = model.embed(**E.batch(u_t, key_mode=True, v=v_u))
                zk = zu[torch.as_tensor(inv, device=device)].view(len(b), M, -1)
                scores = model.scores(zq, zk)                                     # (B, M)
                rel_asd = (asd_task[torch.as_tensor(data.task[a_glob], device=device)][:, None, None, :]
                           if asd_task is not None else asd_t)
                rel = -C.act_dist_t(E.act[torch.as_tensor(cmat, device=device)],
                                    E.act[a_t].unsqueeze(1), rel_asd, include_grip,
                                    ksteps=(args.target_steps or None))
                target = torch.softmax(rel / args.tau0, dim=1)
                l_kd = -(target * torch.log_softmax(scores, dim=1)).sum(1).mean()
                ent = -(target * target.clamp(min=1e-12).log()).sum(1).mean()   # CE floor (target entropy)
                # IDM: images-only forward predicts the (z-scored) prev chunk
                idm = model.idm(*E.tok_at(a_t, v_a))
                tgt_idm = ((E.prev[a_t] - model.am) / model.asd).reshape(len(b), -1)
                l_idm = torch.nn.functional.mse_loss(idm, tgt_idm)
                # EFF: query readout predicts own realized (z-scored) dproprio where a post row exists
                ok = torch.as_tensor(post[a_glob] >= 0, device=device)
                ep_t = (E.eff_p[a_t] - model.dpm) / model.dps
                l_eff = (((model.eff_pred(h) - ep_t) ** 2).mean(1) * ok).sum() / ok.sum().clamp(min=1)
                # EFF++: query [EFFPRED] tokens predict own per-cell Δscene (tok_eff[post]-tok_eff[cur])/dss
                l_effscene = torch.zeros((), device=device)
                if args.lam_eff_scene > 0:
                    es_tgt = E.eff_s[0, a_t].float() / model.dss                    # (B, n_eff, fd); 0 where no post
                    es_per = torch.nn.functional.huber_loss(eff_scene_pred.float(), es_tgt,
                                                            reduction="none", delta=1.0).mean((1, 2))
                    l_effscene = (es_per * ok).sum() / ok.sum().clamp(min=1)
                l_corr = torch.zeros((), device=device)
                l_sym = torch.zeros((), device=device)
                if args.lam_corr > 0 or args.lam_sym > 0:
                    da = C.act_dist_t(E.act[u_t].unsqueeze(0), E.act[a_t].unsqueeze(1),
                                      rel_asd, include_grip)                       # (B, U) action distances
                    vm = torch.as_tensor(data.ep[a_glob][:, None] != data.ep[uniq][None, :], device=device)
                if args.lam_corr > 0:
                    dz = torch.cdist(zq.float(), zu.float())                       # (B, U) metric distances
                    x = dz[vm]; y = da[vm]
                    x = x - x.mean(); y = y - y.mean()
                    l_corr = 1.0 - (x * y).sum() / (x.norm() * y.norm() + 1e-8)
                if args.lam_sym > 0:
                    S = model.logit_scale.clamp(max=np.log(100.0)).exp() * (zq @ zu.T)   # (B, U) logits
                    neg = torch.finfo(S.dtype).min
                    tgt = torch.softmax((-da / args.tau0).masked_fill(~vm, neg), dim=0)  # queries per key
                    logp = torch.log_softmax(S.masked_fill(~vm, neg), dim=0)
                    col_ok = vm.sum(0) >= 2
                    l_sym = -((tgt * logp).sum(0))[col_ok].mean()
                l_align = torch.zeros((), device=device)
                if args.lam_align > 0:
                    hf = len(b) // 2
                    pd = (zq[:hf] - zq[hf:2 * hf]).norm(dim=-1)
                    l_align = (torch.relu(pd - args.align_delta) ** 2).mean()
                loss = (l_kd + args.lam_idm * l_idm + args.lam_eff * l_eff
                        + args.lam_corr * l_corr + args.lam_sym * l_sym + args.lam_align * l_align
                        + args.lam_eff_scene * l_effscene)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            sums += [float(l_kd), float(l_kd - ent), float(l_idm), float(l_eff), float(l_corr), float(l_sym), float(l_align), float(l_effscene)]
        zdb = E.embed_rows(model, tr_rows, key_mode=True)
        zva = E.embed_rows(model, va_rows, key_mode=False)
        r, per, sfx = eval_split(zdb, zva, va_rows)
        s = sums / steps_per_epoch
        w = float(torch.sigmoid(model.w_logit))
        hist.append({"epoch": epoch, "kd": s[0], "kl": s[1], "idm": s[2], "eff": s[3], "dcorr": 1 - s[4], "sym": s[5], "align": s[6], "effscene": s[7],
                     "val_rmse1": r[1], "val_rmse5": r[5], "w": w, **({"per_task": per} if per else {})})
        print(f"  ep{epoch:03d} kd={s[0]:.4f} kl={s[1]:.4f} idm={s[2]:.4f} eff={s[3]:.4f} es={s[7]:.4f} corr={1 - s[4]:.3f} sym={s[5]:.3f} aln={s[6]:.3f} lr={opt.param_groups[0]['lr']:.1e} "
              f"| VAL RMSE@1={r[1]:.4f} @5={r[5]:.4f}{sfx} | w={w:.3f} scale={float(model.logit_scale.exp()):.1f} "
              f"({time.time()-t0:.0f}s)")
        if args.self_m > 0:
            ztr_q = E.embed_rows(model, anchor_rows, key_mode=False)
            dq = torch.cdist(torch.as_tensor(ztr_q, device=device), torch.as_tensor(zdb, device=device))
            dq.masked_fill_(torch.as_tensor((data.ep[anchor_rows][:, None] == data.ep[tr_rows][None, :])
                                            | (data.task[anchor_rows][:, None] != data.task[tr_rows][None, :]),
                                            device=device), float("inf"))
            mined_c = tr_rows[dq.topk(args.self_m, largest=False).indices.cpu().numpy()]
        if r[1] < best["rmse1"]:
            best = {"rmse1": r[1], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    zdb = E.embed_rows(model, tr_rows, key_mode=True)
    res = {}
    for split, rows in (("val", va_rows), ("test", te_rows)):
        zq = E.embed_rows(model, rows, key_mode=False)
        r, per, sfx = eval_split(zdb, zq, rows)
        res[split] = {"rmse1": r[1], "rmse5": r[5], **({"per_task": per} if per else {})}
        print(f"\n=== {split.upper()} (db=train keys, query={split}; retrieval RMSE) ===")
        print(f"  TMT RMSE@1={r[1]:.4f} RMSE@5={r[5]:.4f}{sfx}  (best val epoch {best['epoch']})")

    out_dir = os.path.join(HERE, "results", name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{args.tag}.json"), "w") as f:
        json.dump({"args": vars(args), "best_epoch": best["epoch"], **res, "history": hist}, f, indent=2)
    torch.save(best["state"], os.path.join(out_dir, f"{args.tag}.pt"))
    print(f"\nsaved -> {out_dir}/{args.tag}.pt , metrics_{args.tag}.json")


if __name__ == "__main__":
    main()
