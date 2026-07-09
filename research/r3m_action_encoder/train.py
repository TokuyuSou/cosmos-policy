"""Fine-tune the R3M action-metric encoder.

    CUDA_VISIBLE_DEVICES=1 uv run --extra cu128 --python 3.10 \
        python research/r3m_action_encoder/train.py [--loss rnc|corr] [...]

Pipeline: load cached images -> episode-aware batches (each = a few episodes x N frames, so a batch
holds many within-episode pairs spanning the full action-distance range, plus cross-episode pairs)
-> backbone fine-tune with a frozen head-warmup then discriminative LRs -> select the epoch with the
best VAL neighbourhood Spearman (the "distance corresponds to action distance" metric) -> report the
TEST retrieval table + within-episode Spearman, and save the best weights.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import data as D
import eval as E
from metric_losses import (action_feat, calibrate_distmatch_scale, calibrate_supcon_thresh, corr_loss,
                           distmatch_loss, label_dist, rnc_loss, softnn_loss, supcon_loss, top1nca_loss)
from model import (ActionMetricEncoder, InitDiffFusedEncoder, MultiModalActionMetricEncoder,
                   ObsStateActionMetricEncoder, ProgressActionMetricEncoder, ResidualImageEncoder,
                   StateActionMetricEncoder, TokenChangeFusedEncoder, load_state_only_encoder)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "data", "pnp_counter_to_stove_dense_img")


class FrameDS(Dataset):
    """Returns (primary CHW uint8, wrist CHW uint8, global index). Augmentation (train only) is
    applied per view; NO horizontal flip -- flipping would invert the left/right action labels.

    The index handed to ``__getitem__`` is ENCODED as ``epoch * n + i`` (see EpisodeBatchSampler) so the
    augmentation RNG can be seeded deterministically from (epoch, frame i). This makes the exact
    augmentation INDEPENDENT of the dataloader worker count -- so workers can be scaled up freely to feed
    the GPU without changing the training behaviour (same per-(epoch,frame) aug distribution, reproducible)."""

    def __init__(self, data, aug, base_seed=0):
        self.d = data
        self.aug = aug
        self.n = len(data.act)
        self.base_seed = int(base_seed)

    def __len__(self):
        return self.n

    def _chw(self, arr, i):
        return torch.from_numpy(np.ascontiguousarray(arr[i])).permute(2, 0, 1)

    def __getitem__(self, enc):
        ep_id, i = divmod(int(enc), self.n)            # decode (epoch, frame) from the encoded index
        p, w = self._chw(self.d.primary, i), self._chw(self.d.wrist, i)
        if self.aug is not None:
            # seed per (base_seed, epoch, frame) so aug is deterministic & worker-count-independent;
            # fork_rng(devices=[]) isolates the CPU RNG (aug is CPU-only) and restores it afterwards.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((self.base_seed * 1000003 + ep_id) * 1000003 + i)
                p, w = self.aug(p), self.aug(w)        # primary then wrist -> independent draws (as before)
        return p, w, i


def make_aug():
    return v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    ])


class EpisodeBatchSampler:
    """Each batch = ``eps_per_batch`` random train episodes x ``frames_per_ep`` random frames.

    Multi-task (``task`` given and >1 task present): every batch is drawn from a SINGLE task and tasks
    are balanced (equal #batches/task, round-robin order) -- so the pairwise loss optimises the fine
    WITHIN-task action structure and isn't dominated by easy cross-task separation. Single-task keeps
    the original behaviour exactly. With ``mix=True`` (multi-task only) batches instead pool ALL tasks so
    the loss sees cross-task pairs (the trainer then z-scores each chunk by its OWN task's stats)."""

    def __init__(self, ep, mask, eps_per_batch, frames_per_ep, seed=0, task=None, n=None, mix=False):
        idx = np.where(mask)[0]
        self.n = int(n) if n is not None else len(ep)   # frame-count base for the epoch*n+i index encoding
        self.by_ep, ep_task = {}, {}
        for i in idx:
            e = int(ep[i])
            self.by_ep.setdefault(e, []).append(int(i))
            if task is not None:
                ep_task[e] = int(task[i])
        self.ep_ids = [e for e, v in self.by_ep.items() if len(v) > 0]
        self.E, self.F, self.seed, self.epoch = eps_per_batch, frames_per_ep, seed, 0
        self.task_eps = {}
        for e in self.ep_ids:
            self.task_eps.setdefault(ep_task.get(e, 0), []).append(e)
        self.multi = len(self.task_eps) > 1
        self.mix = bool(mix) and self.multi          # mixed multi-task: pool ALL tasks per batch (cross-task)
        if self.multi and not self.mix:
            self.task_list = sorted(self.task_eps)
            self.n_per_task = max(1, (len(idx) // len(self.task_list)) // (eps_per_batch * frames_per_ep))
            self.n_batches = self.n_per_task * len(self.task_list)
        else:
            self.n_batches = max(1, len(idx) // (eps_per_batch * frames_per_ep))

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n_batches

    def _draw(self, rng, eps):
        batch = []
        for e in rng.choice(eps, min(self.E, len(eps)), replace=False):
            pool = self.by_ep[e]
            batch += rng.choice(pool, self.F, replace=len(pool) < self.F).tolist()
        return batch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        off = self.epoch * self.n                        # encode epoch into each index (epoch*n + i)
        if self.multi and not self.mix:                  # same-task batches (one task per batch)
            order = [t for t in self.task_list for _ in range(self.n_per_task)]
            rng.shuffle(order)
            for t in order:
                yield [off + i for i in self._draw(rng, self.task_eps[t])]
        else:                                            # single-task OR mixed multi-task: pool all episodes
            for _ in range(self.n_batches):
                yield [off + i for i in self._draw(rng, self.ep_ids)]


def val_metric(emb, data, device, is_multi, target_steps=None):
    """Given a precomputed embedding, return the model-selection metric (higher=better) + an aux value.
    Multi-task: MEAN over tasks of within-task val within-episode Spearman (the headline metric, with
    each task's own action stats). Single-task: the original cross-pool val neighbourhood Spearman.
    ``target_steps`` (single-task only) restricts the action-distance metric to the first N chunk steps
    (None or >= chunk length = full chunk = unchanged)."""
    if is_multi:
        sps, rmses = [], []
        for t in range(len(data.tasks)):
            mtr, mva = data.tr & (data.task == t), data.va & (data.task == t)
            sps.append(E.within_episode_spearman(emb, data, mva, am=data.task_am[t], asd=data.task_asd[t]))
            # per-task val RMSE@1: query=val_t retrieves the action-nearest in db=train_t (t's own stats)
            r, _ = E._retr_rmse(emb[mtr], emb[mva], data.act[mtr], data.act[mva],
                                data.task_am[t], data.task_asd[t], (1,), device)
            rmses.append(r[1])
        return float(np.mean(sps)), {"per_task_sp": sps, "rmse1": float(np.mean(rmses))}
    # Selection only needs the LEARNED rep -> compute just that, skipping the fixed proprio_prev / oracle /
    # random / learned+proprio_prev baselines that E.evaluate() recomputes every epoch (those don't affect
    # selection). Identical learned RMSE@1 + neigh-Spearman -> same chosen checkpoint; the full TEST table
    # is still produced once at the end via E.evaluate().
    dbm, qm = data.tr, data.va
    db_act, q_act = data.act[dbm], data.act[qm]
    if target_steps is not None and target_steps < db_act.shape[1]:   # restrict metric to first-N steps
        db_act, q_act = db_act[:, :target_steps], q_act[:, :target_steps]
    rmse, _ = E._retr_rmse(emb[dbm], emb[qm], db_act, q_act, data.am, data.asd, (1,), device)
    sp = E._neigh_spearman(emb[dbm], emb[qm], db_act, q_act, data.am, data.asd, device)
    return sp, rmse[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--loss", choices=["rnc", "corr", "softnn", "corr+softnn", "top1nca", "corr+top1nca",
                                       "supcon", "corr+supcon", "distmatch"], default="rnc",
                    help="supcon = SupCon-style CROSS-EPISODE contrastive (k action-nearest other-episode "
                         "positives; stable multi-positive retrieval loss); corr+supcon = corr (global) + "
                         "supcon (local top-1). softnn/top1nca = soft-argmin NCA variants (high-variance). "
                         "distmatch = calibrated distance-VALUE regression (embed distance == action distance, "
                         "not just rank/correlation); usually composed onto a base via --distmatch-w.")
    ap.add_argument("--softnn-temp", type=float, default=0.3, help="softnn softmax temperature")
    ap.add_argument("--softnn-w", type=float, default=1.0, help="softnn weight in corr+softnn")
    ap.add_argument("--top1nca-temp", type=float, default=0.2, help="top1nca softmax temperature (smaller=sharper top-1)")
    ap.add_argument("--top1nca-w", type=float, default=1.0, help="top1nca weight in corr+top1nca")
    ap.add_argument("--supcon-k", type=int, default=8, help="supcon: # action-nearest cross-episode positives "
                    "(--supcon-pos topk); also the target avg #positives when auto-calibrating --supcon-pos thresh")
    ap.add_argument("--supcon-temp", type=float, default=0.1, help="supcon InfoNCE temperature")
    ap.add_argument("--supcon-w", type=float, default=1.0, help="supcon weight in corr+supcon")
    ap.add_argument("--supcon-pos", choices=["topk", "thresh"], default="topk",
                    help="supcon positive selection: 'topk' (the --supcon-k action-nearest cross-episode "
                         "points, a fixed COUNT -- default, unchanged) or 'thresh' (ALL cross-episode points "
                         "within a calibrated action-distance RADIUS -- adaptive count; dense neighbourhoods "
                         "keep more positives, sparse fewer, and the single nearest is always kept).")
    ap.add_argument("--supcon-thresh", type=float, default=None,
                    help="--supcon-pos thresh: action-distance radius (label_dist units). Default None => "
                         "auto-calibrated on the train pool so avg #positives/anchor ~= --supcon-k (no magic number).")
    ap.add_argument("--distmatch-w", type=float, default=0.0,
                    help="weight of the calibrated distance-VALUE matching term (distmatch_loss) ADDED to the "
                         "--loss base (0 = off, default -> base unchanged). Makes ||z_i-z_j|| match the action "
                         "distance value itself (calibrated hit-distance), beyond rnc's rank / corr's correlation. "
                         "--loss distmatch runs it standalone.")
    ap.add_argument("--distmatch-scale", type=float, default=None,
                    help="distmatch: action-distance units per unit embedding distance (target = ld/scale). "
                         "Default None => auto-calibrated so the median cross-ep action distance maps to "
                         "--distmatch-target embedding distance.")
    ap.add_argument("--distmatch-weight", choices=["sammon", "uniform"], default="sammon",
                    help="distmatch pair weighting: 'sammon' (1/(ld+eps), emphasize near pairs = the retrieval "
                         "neighbourhood) or 'uniform' (global MDS).")
    ap.add_argument("--distmatch-normalize", action="store_true",
                    help="distmatch: match STANDARDIZED distances (the CORR objective; scale/offset-invariant) "
                         "instead of absolute values. weight=uniform => equals --loss corr; weight=sammon => "
                         "near-pair-weighted corr. --distmatch-scale is ignored in this mode.")
    ap.add_argument("--distmatch-beta", type=float, default=1.0, help="distmatch Huber transition on the distance residual")
    ap.add_argument("--distmatch-quantile", type=float, default=0.5,
                    help="distmatch scale calibration: which cross-ep action-distance quantile maps to --distmatch-target")
    ap.add_argument("--distmatch-target", type=float, default=1.0,
                    help="distmatch scale calibration: embedding distance the quantile maps to (mid-range in [0,2])")
    ap.add_argument("--hard-neg", action="store_true",
                    help="state-aware HARD-NEGATIVE mining (fused/supcon only): append each anchor's "
                         "proprio+prev nearest cross-episode train points to the batch so the supcon "
                         "denominator contains the state-confusable decoys the image must separate "
                         "(see hardneg.py). OFF (default) => batches are byte-identical to the baseline.")
    ap.add_argument("--hard-neg-m", type=int, default=8, help="hard-neg: #neighbors appended per chosen anchor")
    ap.add_argument("--hard-neg-k", type=int, default=32, help="hard-neg: size of each anchor's precomputed state-neighbor pool")
    ap.add_argument("--hard-neg-p", type=float, default=0.5, help="hard-neg: target fraction of anchors augmented (curriculum ceiling)")
    ap.add_argument("--hard-neg-ramp", type=int, default=30, help="hard-neg: epochs to ramp p from 0 (after warmup) to its target")
    ap.add_argument("--hard-neg-cap", type=int, default=128, help="hard-neg: max extra rows appended per batch (bounds batch growth)")
    ap.add_argument("--hard-neg-margin", type=float, default=0.0,
                    help="hard-neg: supcon action-distance dead-zone (0=off). Drops borderline negatives "
                         "just beyond the positives from the denominator; positives unaffected.")
    ap.add_argument("--anchor-batch", action="store_true",
                    help="ORACLE-ANCHORED batches (supcon losses, single-task): each batch = anchor groups "
                         "of [anchor + n_pos GLOBAL cross-ep action-nearest positives (soft-sampled) + "
                         "n_dec mined state-confusable decoys + n_rand random]. Fixes the measured defect "
                         "that batch-local supcon positives overlap the true global top-8 by only 6% (1.59x "
                         "action-farther) and decoys are ~absent. Loss/model/ckpt format UNCHANGED; "
                         "OFF (default) => batches byte-identical to the legacy sampler.")
    ap.add_argument("--anchor-pos-k", type=int, default=16, help="anchor-batch: global positive pool size")
    ap.add_argument("--anchor-pos-n", type=int, default=8, help="anchor-batch: #positives sampled per anchor")
    ap.add_argument("--anchor-dec-k", type=int, default=16, help="anchor-batch: mined decoy pool size")
    ap.add_argument("--anchor-dec-n", type=int, default=6, help="anchor-batch: #decoys sampled per anchor")
    ap.add_argument("--anchor-rand-n", type=int, default=5, help="anchor-batch: #random rows per anchor group")
    ap.add_argument("--anchor-pos-temp", type=float, default=0.15,
                    help="anchor-batch: positive sampling weight ∝ exp(-action_dist/temp) over the top-k pool "
                         "(listwise-soft; <=0 = uniform)")
    ap.add_argument("--mix-tasks", action="store_true",
                    help="multi-task only: draw each batch from ALL tasks pooled (cross-task pairs in the loss, "
                         "per-sample action z-scoring) instead of the default same-task batches; no effect single-task")
    ap.add_argument("--select-by", choices=["spearman", "rmse1"], default=None,
                    help="checkpoint-selection metric (val). Default: rmse1 for single-task fused runs "
                         "(the stated goal is to beat proprio_prev RMSE@1), spearman otherwise.")
    ap.add_argument("--backbone", default="resnet18", help="r3m img-backbone ONLY: the ResNet arch "
                    "(resnet18/34/50). Ignored when --img-backbone theia.")
    ap.add_argument("--img-backbone", choices=["r3m", "theia"], default="r3m",
                    help="image-trunk FAMILY: 'r3m' (ResNet-18 R3M init, the default => unchanged) or "
                         "'theia' (distilled-VFM ViT; richer spatial tokens, used frozen on small data). "
                         "Selecting 'theia' with --freeze left at its R3M default (layer4) auto-switches "
                         "freeze->frozen and namespaces the output tag (theia_...).")
    ap.add_argument("--theia-model", default=None,
                    help="HF id for --img-backbone theia (default theia-tiny-patch16-224-cddsv; "
                         "small/base variants auto-size the image head from their feat_dim).")
    ap.add_argument("--theia-reduce", choices=["mean", "max", "attn"], default="mean",
                    help="theia only: how the 196 spatial tokens become one per-view vector. 'mean' (GAP, "
                         "the faithful drop-in, default) | 'max' | 'attn' (learned residual attention-pool "
                         "head that exploits Theia's local detail; trunk stays frozen, only the head trains).")
    ap.add_argument("--theia-pool-queries", type=int, default=4,
                    help="--theia-reduce attn: number of pooling queries (distinct spatial aspects pooled).")
    ap.add_argument("--freeze", default="layer4", choices=["frozen", "layer4", "layer34", "none"])
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--arch", choices=["image", "fused", "state", "initdiff", "residual", "obsstate"], default="image",
                    help="image: the original images-only encoder. fused: also encode proprio+prev "
                         "and fuse them with the image into one embedding (MultiModalActionMetricEncoder). "
                         "state: prev+proprio ONLY, no image (StateActionMetricEncoder) -- the learned "
                         "counterpart of the proprio_prev key. initdiff: fused + the image branch also "
                         "encodes current-minus-INITIAL-frame feature diffs (InitDiffFusedEncoder, Version A). "
                         "residual: FROZEN state-only base (--state-ckpt) + a trainable image RESIDUAL "
                         "(ResidualImageEncoder) -- the image learns only what the state base gets wrong. "
                         "obsstate: image + proprio (NO prev-actions), IMAGE-base residual with a dynamic "
                         "gate (ObsStateActionMetricEncoder) -- forces the image to be the representation, "
                         "proprio only a gated correction. Uses --gate-dyn/--gate-rank/--gate-init/--mod-dropout.")
    ap.add_argument("--state-ckpt", default=None,
                    help="--arch residual: path to the trained state-only encoder (.pt) used as the frozen z_N1 base.")
    ap.add_argument("--image-mode", choices=["attn", "tokenchange"], default="tokenchange",
                    help="--arch residual: image branch -- 'attn' (plain Theia pool of current frames) or "
                         "'tokenchange' (Version-B dedicated token change-pool on the primary view, the default).")
    ap.add_argument("--init-diff-views", default="primary",
                    help="--arch initdiff (mode feat / Version A): comma list of views that contribute a "
                         "current-minus-initial diff stream (default 'primary' = the world-aligned "
                         "third-person cam; the wrist is a moving cam so its initial diff is omitted).")
    ap.add_argument("--init-diff-mode", choices=["feat", "token"], default="feat",
                    help="--arch initdiff: 'feat' (Version A, InitDiffFusedEncoder -- pool then subtract on "
                         "the pooled feature, shared pool) | 'token' (Version B, TokenChangeFusedEncoder -- "
                         "a DEDICATED token-level change-guided pool on the primary view; Theia only).")
    ap.add_argument("--state-enc", choices=["mlp", "temporal", "recency"], default="mlp",
                    help="fused only: state-branch encoder over proprio+prev. 'mlp' flat-input; 'temporal' "
                         "tiny Transformer over the 16 prev steps; 'recency' recency-pooling dual-stream "
                         "(attention-pool + raw last-3 skip + dedicated proprio MLP; see StateEncoder).")
    ap.add_argument("--progress", action="store_true",
                    help="fused only: add an auxiliary TASK-PROGRESS head (ProgressActionMetricEncoder) and "
                         "regress the embedding to normalized progress t/T (Huber). Injects the oracle-confirmed "
                         "same-progress structure into the retrieval embedding. Off by default (encoder unchanged).")
    ap.add_argument("--progress-weight", type=float, default=5.0,
                    help="weight lambda of the progress (smooth-L1) loss added to the metric loss.")
    ap.add_argument("--progress-hidden", type=int, default=128, help="progress head hidden width (shallow on purpose).")
    ap.add_argument("--fusion", choices=["concat", "residual"], default="concat",
                    help="fused only: concat->MLP, or residual (learned-state base + gated image "
                         "correction; floor near proprio_prev with image upside).")
    ap.add_argument("--gate-dyn", action="store_true",
                    help="fused residual only: make the image gate INPUT-DEPENDENT -- a small low-rank MLP "
                         "on [state,img] adds a per-sample/per-dim delta to the static gate (near-identity "
                         "init, so it starts == the static-gate model and can only adapt from there).")
    ap.add_argument("--gate-rank", type=int, default=64, help="--gate-dyn: bottleneck width of the gate MLP")
    ap.add_argument("--gate-init", type=float, default=-2.0,
                    help="fused residual only: initial image-gate logit (image weight = sigmoid(gate_init); "
                         "default -2.0 -> ~0.12). Raise (e.g. 0.0 -> 0.5) to start the image with more weight.")
    ap.add_argument("--mod-dropout", type=float, default=0.3,
                    help="fused only: per-sample probability of zeroing a whole branch in training "
                         "(prevents the fusion collapsing onto the dominant state branch). 0 = off.")
    ap.add_argument("--img-dim", type=int, default=256, help="fused only: image-branch feature width")
    ap.add_argument("--state-dim", type=int, default=256, help="fused only: state-branch feature width")
    ap.add_argument("--dropout", type=float, default=0.1, help="fused only: head/branch dropout")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3, help="frozen-backbone head-warmup epochs")
    ap.add_argument("--eps-per-batch", type=int, default=6)
    ap.add_argument("--frames-per-ep", type=int, default=40)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bb-lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--separate-backbones", action="store_true",
                    help="use a SEPARATE R3M encoder per view (primary vs wrist) instead of sharing one. "
                         "Under --freeze layer4 this shares the frozen trunk's R3M init but trains a "
                         "per-view layer4 (and keeps per-view BN stats) -- view-specialised, same "
                         "trainable-param budget per view, training-friendly.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32,
                    help="dataloader workers for CPU image augmentation. Augmentation is seeded per "
                         "(epoch, frame) so the result is INDEPENDENT of this value -- raise it to feed the "
                         "GPU on many-core hosts (the original GPU-starvation fix), lower it if oversubscribed.")
    ap.add_argument("--include-grip", action="store_true",
                    help="include the gripper dim in the action-distance TARGET (corr/rnc loss) AND the val "
                         "RMSE/Spearman metrics, i.e. use all 7 action dims instead of the 6 non-grip ones.")
    ap.add_argument("--target-steps", type=int, default=16,
                    help="horizon (# of the next action steps) used as the action-distance TARGET/metric: the "
                         "loss label, val selection, and the TEST report are all computed over the FIRST "
                         "target-steps of the 16-step chunk (the retrieval VALUE/executed chunk is unchanged; "
                         "this only changes what distance the embedding is trained+scored to match). Default 16 "
                         "= the full chunk = byte-identical to the current behavior. Single-task only.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--init-ckpt", default="", help="warm-start: load this encoder state_dict before training "
                    "(DAgger fine-tune). Empty = train from scratch (default).")
    ap.add_argument("--onpolicy-dirs", default="", help="comma-separated fused_dagger_collect output dirs to "
                    "aggregate as extra on-policy episodes (DAgger dataset aggregation). Empty = demos only.")
    ap.add_argument("--onpolicy-val-frac", type=float, default=0.2, help="fraction of on-policy EPISODES held "
                    "out for val/model-selection (queries retrieved against the demo db).")
    ap.add_argument("--fail-anchors", action="store_true",
                    help="add FAILURE-episode rows (ep*_success=0.npz in --data) to the TRAINING POOL "
                         "(the batch sampler), each labelled by its recorded VLA chunk. The retrieval "
                         "target is VLA parity (not task success), so failed rollouts carry valid teacher "
                         "chunks AND the off-support states where retrieval collapses at high skip. The "
                         "deploy cache, input-norm stats and val/test benchmark stay success-only "
                         "(tr/va/te unchanged) -> numbers directly comparable to the success-only run. "
                         "NOTE: unlike TMT's asymmetric listwise-KD (failure = anchor-only), fused's "
                         "episode-batched SupCon is symmetric, so failure rows also act as batch keys/"
                         "negatives -- the faithful analog that leaves the validated loss untouched.")
    ap.add_argument("--fail-tail-drop", type=int, default=0,
                    help="with --fail-anchors: drop the last K decision rows of each failure episode from "
                         "the training pool (timeout flailing); rows stay in the arrays so indexing holds.")
    args = ap.parse_args()
    # Theia is a pretrained ViT used FROZEN on this small dataset: 'layer4' is an R3M-specific freeze
    # mode, so if the user left freeze at its R3M default, auto-switch to 'frozen' (still overridable).
    if args.img_backbone == "theia" and args.freeze == "layer4":
        args.freeze = "frozen"
        print("  [img-backbone=theia] freeze left at R3M default -> auto-switched to 'frozen'")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or (f"{args.arch}_{args.image_mode}_{args.loss}" if args.arch == "residual"
                       else f"{args.arch}_{args.state_enc}_{args.loss}"
                       if args.arch in ("fused", "state", "initdiff")
                       else f"{args.arch}_{args.loss}" if args.arch == "obsstate" else args.loss)
    if args.img_backbone == "theia" and not args.tag:      # namespace so theia never clobbers the R3M baseline ckpt
        tag = f"theia_{args.theia_reduce + '_' if args.theia_reduce != 'mean' else ''}{tag}"
    import losses as _losses                       # set the gripper toggle for action_feat + the RMSE metric
    _losses.INCLUDE_GRIP = bool(args.include_grip)

    data_dirs = [d for d in args.data.split(",") if d.strip()]
    data = D.load(data_dirs[0], seed=args.seed) if len(data_dirs) == 1 \
        else D.load_multi(data_dirs, seed=args.seed)
    if args.onpolicy_dirs:                              # DAgger: append on-policy rollout frames as extra episodes
        assert len(data_dirs) == 1, "--onpolicy-dirs is single-task only"
        data = D.append_onpolicy(data, args.onpolicy_dirs, args.onpolicy_val_frac, args.seed)
    if args.fail_anchors:                              # add failure-episode rows to the training pool (see help)
        assert len(data_dirs) == 1 and not args.onpolicy_dirs, "--fail-anchors: single-task, no --onpolicy-dirs"
        assert args.arch in ("fused", "state", "image"), "--fail-anchors v1: fused/state/image arch (no init/residual)"
        assert not (args.hard_neg or args.anchor_batch), "--fail-anchors v1: no --hard-neg/--anchor-batch " \
            "(their neighbor mining assumes the pool == data.tr)"
        data = D.append_failures(data, data_dirs[0], tail_drop=args.fail_tail_drop)
    is_multi = len(data.tasks) > 1
    mix = bool(args.mix_tasks) and is_multi            # mixed multi-task batches (cross-task) vs same-task batches
    FULL_STEPS = data.act.shape[1]                     # 16-step chunk
    K_TARGET = int(args.target_steps)
    assert 1 <= K_TARGET <= FULL_STEPS, f"--target-steps must be in [1,{FULL_STEPS}]"
    if K_TARGET < FULL_STEPS:
        assert not is_multi, "--target-steps is single-task only"
        print(f"  [target-steps] action-distance metric = FIRST {K_TARGET}/{FULL_STEPS} chunk steps "
              f"(loss label + val selection + TEST report); retrieval value/executed chunk unchanged")
    am_t = torch.tensor(data.task_am, device=device)   # (T,7) per-task action stats; loss uses the batch's task
    asd_t = torch.tensor(data.task_asd, device=device)

    # SupCon radius mode: calibrate the action-distance threshold that replaces the fixed top-k positives.
    supcon_thresh = None                               # None => supcon uses top-k (default; byte-identical)
    if args.loss in ("supcon", "corr+supcon") and args.supcon_pos == "thresh":
        if args.supcon_thresh is not None:             # user-set radius: skip calibration
            supcon_thresh = float(args.supcon_thresh)
            print(f"  [supcon-thresh] positives within action-distance radius={supcon_thresh:.4f} "
                  f"(user-set); --supcon-k ignored")
        else:                                          # auto-calibrate on the train pool (same units as ld)
            trm = getattr(data, "train_pool", data.tr)
            _act = torch.as_tensor(data.act[trm][:, :K_TARGET], device=device)          # (M,K,7) target chunk
            _ti = torch.as_tensor(data.task[trm], device=device)
            _feat = action_feat(_act, am_t[_ti].unsqueeze(1), asd_t[_ti].unsqueeze(1))  # per-task z-scored
            supcon_thresh = calibrate_supcon_thresh(_feat, data.ep[trm], args.supcon_k,
                                                    task=data.task[trm], device=device, seed=args.seed)
            print(f"  [supcon-thresh] calibrated action-distance radius={supcon_thresh:.4f} "
                  f"(~{args.supcon_k} cross-ep positives/anchor on train; now adaptive per anchor)")

    # distmatch: calibrate the action-distance -> embedding-distance scale (unused in --distmatch-normalize mode).
    distmatch_scale = None
    dm_active = (args.loss == "distmatch") or (args.distmatch_w > 0.0)
    if dm_active and args.distmatch_normalize:
        print(f"  [distmatch] normalize mode: matching STANDARDIZED distances (corr objective, "
              f"weight={args.distmatch_weight}); scale unused")
    elif dm_active:
        if args.distmatch_scale is not None:
            distmatch_scale = float(args.distmatch_scale)
            print(f"  [distmatch] scale={distmatch_scale:.4f} (user-set)")
        else:
            trm = getattr(data, "train_pool", data.tr)
            _act = torch.as_tensor(data.act[trm][:, :K_TARGET], device=device)
            _ti = torch.as_tensor(data.task[trm], device=device)
            _feat = action_feat(_act, am_t[_ti].unsqueeze(1), asd_t[_ti].unsqueeze(1))
            distmatch_scale = calibrate_distmatch_scale(_feat, data.ep[trm], task=data.task[trm],
                                                        quantile=args.distmatch_quantile,
                                                        target=args.distmatch_target, device=device, seed=args.seed)
            print(f"  [distmatch] calibrated scale={distmatch_scale:.4f} "
                  f"(q{args.distmatch_quantile:g} cross-ep action dist -> embed dist {args.distmatch_target:g}; "
                  f"weight={args.distmatch_weight})")

    ds = FrameDS(data, None if args.no_aug else make_aug(), base_seed=args.seed)
    sampler = EpisodeBatchSampler(data.ep, getattr(data, "train_pool", data.tr),
                                  args.eps_per_batch, args.frames_per_ep, args.seed,
                                  task=data.task, n=len(data.act), mix=mix)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=(4 if args.workers > 0 else None))

    fused = args.arch == "fused"
    state_only = args.arch == "state"
    init_diff = args.arch == "initdiff"
    residual = args.arch == "residual"
    obsstate = args.arch == "obsstate"
    # whether the forward/embed path must supply each point's episode-INITIAL frame (token-level change)
    uses_init = init_diff or (residual and args.image_mode == "tokenchange")
    needs_state = fused or state_only or init_diff or residual or obsstate  # path consuming prev/proprio (set_norm + embed_all_mm)
    assert not (args.progress and not fused), "--progress is fused-only (the progress head reads the fused embedding)"
    init_idx = E.episode_initial_idx(data) if uses_init else None   # each point -> its episode's t~0 frame row
    if init_diff or residual:
        assert not (is_multi and not mix), f"--arch {args.arch}: per-task multi-task embedding not yet supported (single-task or --mix-tasks)"
    if residual:
        assert args.state_ckpt, "--arch residual requires --state-ckpt <trained state-only encoder .pt>"
        state_base = load_state_only_encoder(args.state_ckpt)        # frozen z_N1 base (dims inferred from ckpt)
        model = ResidualImageEncoder(
            state_base, out_dim=args.out_dim, img_dim=args.img_dim, dropout=args.dropout,
            share_backbone=not args.separate_backbones, image_mode=args.image_mode,
            img_backbone=args.img_backbone, theia_model=args.theia_model, theia_reduce=args.theia_reduce,
            attn_queries=args.theia_pool_queries, gate_init=args.gate_init).to(device)
        pm = data.proprio[data.tr].mean(0)
        psd = data.proprio[data.tr].std(0) + D.EPS
        model.set_norm(data.am, data.asd, pm, psd)
    elif init_diff:
        idkw = dict(
            out_dim=args.out_dim, img_dim=args.img_dim, state_dim=args.state_dim, dropout=args.dropout,
            share_backbone=not args.separate_backbones, state_mode=args.state_enc, mod_dropout=args.mod_dropout,
            fusion=args.fusion, gate_dyn=args.gate_dyn, gate_rank=args.gate_rank,
            prev_steps=data.prev.shape[1], act_dim=data.prev.shape[2], proprio_dim=data.proprio.shape[1],
            img_backbone=args.img_backbone, theia_model=args.theia_model, theia_reduce=args.theia_reduce,
            attn_queries=args.theia_pool_queries, gate_init=args.gate_init)
        if args.init_diff_mode == "token":           # Version B: dedicated token-level change-guided pool (Theia)
            model = TokenChangeFusedEncoder(args.backbone, "r3m", args.freeze, **idkw).to(device)
        else:                                        # Version A: feature-level pool-then-diff
            model = InitDiffFusedEncoder(args.backbone, "r3m", args.freeze, **idkw,
                                         init_diff_views=tuple(v for v in args.init_diff_views.split(",") if v)).to(device)
        pm = data.proprio[data.tr].mean(0)
        psd = data.proprio[data.tr].std(0) + D.EPS
        model.set_norm(data.am, data.asd, pm, psd)
    elif fused:
        Enc = ProgressActionMetricEncoder if args.progress else MultiModalActionMetricEncoder
        kw = dict(prog_hidden=args.progress_hidden) if args.progress else {}
        model = Enc(
            args.backbone, "r3m", args.freeze, out_dim=args.out_dim, img_dim=args.img_dim,
            state_dim=args.state_dim, dropout=args.dropout, share_backbone=not args.separate_backbones,
            state_mode=args.state_enc, mod_dropout=args.mod_dropout, fusion=args.fusion,
            gate_dyn=args.gate_dyn, gate_rank=args.gate_rank,
            prev_steps=data.prev.shape[1], act_dim=data.prev.shape[2],
            proprio_dim=data.proprio.shape[1], img_backbone=args.img_backbone,
            theia_model=args.theia_model, theia_reduce=args.theia_reduce,
            attn_queries=args.theia_pool_queries, gate_init=args.gate_init, **kw).to(device)
        pm = data.proprio[data.tr].mean(0)
        psd = data.proprio[data.tr].std(0) + D.EPS
        model.set_norm(data.am, data.asd, pm, psd)   # action stats for prev, proprio stats for proprio
    elif obsstate:                                   # image + proprio (NO prev): IMAGE-base residual + dynamic gate
        model = ObsStateActionMetricEncoder(
            args.backbone, "r3m", args.freeze, out_dim=args.out_dim, img_dim=args.img_dim,
            state_dim=args.state_dim, dropout=args.dropout, share_backbone=not args.separate_backbones,
            mod_dropout=args.mod_dropout, prev_steps=data.prev.shape[1], act_dim=data.prev.shape[2],
            proprio_dim=data.proprio.shape[1], gate_init=args.gate_init, gate_dyn=args.gate_dyn,
            gate_rank=args.gate_rank, img_backbone=args.img_backbone, theia_model=args.theia_model,
            theia_reduce=args.theia_reduce, attn_queries=args.theia_pool_queries).to(device)
        pm = data.proprio[data.tr].mean(0)
        psd = data.proprio[data.tr].std(0) + D.EPS
        model.set_norm(data.am, data.asd, pm, psd)
    elif state_only:                                 # prev+proprio only (no image): learned proprio_prev / z_N1 base
        model = StateActionMetricEncoder(
            out_dim=args.out_dim, state_dim=args.state_dim, dropout=args.dropout, state_mode=args.state_enc,
            prev_steps=data.prev.shape[1], act_dim=data.prev.shape[2],
            proprio_dim=data.proprio.shape[1]).to(device)
        pm = data.proprio[data.tr].mean(0)
        psd = data.proprio[data.tr].std(0) + D.EPS
        model.set_norm(data.am, data.asd, pm, psd)
    else:
        model = ActionMetricEncoder(args.backbone, "r3m", args.freeze, out_dim=args.out_dim,
                                    share_backbone=not args.separate_backbones,
                                    img_backbone=args.img_backbone, theia_model=args.theia_model,
                                    theia_reduce=args.theia_reduce,
                                    attn_queries=args.theia_pool_queries).to(device)
    if args.init_ckpt:                                   # DAgger warm-start: load the previous round's encoder
        sd = torch.load(args.init_ckpt, map_location=device)
        sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  [warm-start] loaded {args.init_ckpt} (missing {len(missing)}, unexpected {len(unexpected)})")
    # Generic head/backbone split (works for both encoders): the backbone TRUNK(s) get the small bb-lr,
    # everything else (head / state-encoder / fusion / any backbone aux head like the Theia attn-pool)
    # gets head-lr. trunk_parameters() excludes the attn-pool head so that fresh head trains at head-lr.
    bb_params = [p for bb in model.backbones() for p in bb.trunk_parameters()]
    bb_ids = {id(p) for p in bb_params}
    # exclude frozen params (e.g. ResidualImageEncoder's frozen state base) from the head group
    head_params = [p for p in model.parameters() if id(p) not in bb_ids and p.requires_grad]
    if not needs_state:
        embed = lambda: E.embed_all(model, data, device)
    elif uses_init:                                  # init-frame path: pass each point's episode-initial frame
        embed = lambda: E.embed_all_mm_initdiff(model, data, device, init_idx)
    elif is_multi and not mix:                       # per-task input standardization (set_norm per task at embed)
        embed = lambda: E.embed_all_mm_pertask(model, data, device)
    else:                                            # single-task, or mixed multi-task (one global input norm)
        embed = lambda: E.embed_all_mm(model, data, device)
    # bb group omitted when there is no backbone (state-only) -- AdamW rejects an empty param group.
    opt_groups = [{"params": head_params, "lr": args.head_lr, "weight_decay": 1e-2}]
    if bb_params:
        opt_groups.append({"params": bb_params, "lr": args.bb_lr, "weight_decay": 1e-4})
    opt = AdamW(opt_groups)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = {  # all take (z, ld, ep); ep (episode ids) used only by the cross-episode top1nca term
        "rnc": lambda z, ld, ep: rnc_loss(z, ld, args.tau),
        "corr": lambda z, ld, ep: corr_loss(z, ld),
        "softnn": lambda z, ld, ep: softnn_loss(z, ld, args.softnn_temp),
        "corr+softnn": lambda z, ld, ep: corr_loss(z, ld) + args.softnn_w * softnn_loss(z, ld, args.softnn_temp),
        "top1nca": lambda z, ld, ep: top1nca_loss(z, ld, ep, args.top1nca_temp),
        "corr+top1nca": lambda z, ld, ep: corr_loss(z, ld) + args.top1nca_w * top1nca_loss(z, ld, ep, args.top1nca_temp),
        "supcon": lambda z, ld, ep: supcon_loss(z, ld, ep, args.supcon_k, args.supcon_temp, args.hard_neg_margin, supcon_thresh),
        "corr+supcon": lambda z, ld, ep: corr_loss(z, ld) + args.supcon_w * supcon_loss(z, ld, ep, args.supcon_k, args.supcon_temp, args.hard_neg_margin, supcon_thresh),
        "distmatch": lambda z, ld, ep: distmatch_loss(z, ld, distmatch_scale, args.distmatch_weight, args.distmatch_beta, normalize=args.distmatch_normalize),
    }[args.loss]
    if args.distmatch_w > 0.0 and args.loss != "distmatch":     # compose: ADD distance-matching onto the base
        _base_loss_fn = loss_fn
        loss_fn = lambda z, ld, ep: _base_loss_fn(z, ld, ep) + args.distmatch_w * distmatch_loss(
            z, ld, distmatch_scale, args.distmatch_weight, args.distmatch_beta, normalize=args.distmatch_normalize)
    select_by = args.select_by or ("rmse1" if (needs_state and not is_multi) else "spearman")

    n_tr = sum(p.numel() for p in head_params + bb_params if p.requires_grad)
    if residual:
        arch_info = f"(FROZEN-state base + image-residual:{args.image_mode}, gate_init={args.gate_init}) "
    elif fused or init_diff:
        idtag = f",initdiff:{args.init_diff_mode}" + (f"({args.init_diff_views})" if init_diff and args.init_diff_mode == "feat" else "")
        arch_info = f"(state={args.state_enc},mod_drop={args.mod_dropout}{idtag if init_diff else ''}) "
    elif state_only:
        arch_info = f"(state={args.state_enc}) "
    elif obsstate:
        arch_info = (f"(image-base + proprio, gate_dyn={args.gate_dyn}, gate_init={args.gate_init}, "
                     f"mod_drop={args.mod_dropout}) ")
    else:
        arch_info = " "
    print(f"=== action-metric encoder | img_backbone={args.img_backbone}"
          f"{('('+(args.theia_model or 'tiny')+')') if args.img_backbone == 'theia' else ''} | arch={args.arch}"
          f"{arch_info}"
          f"loss={args.loss} freeze={args.freeze} "
          f"backbones={'separate' if args.separate_backbones else 'shared'} "
          f"tasks={len(data.tasks)}{('(mixed batches)' if mix else '(same-task batches)') if is_multi else ''} "
          f"batch={args.eps_per_batch}x{args.frames_per_ep} | action_dims={_losses.n_act_dims()}"
          f"{'(incl. gripper)' if args.include_grip else '(non-grip)'} | trainable~{n_tr/1e6:.1f}M ===")
    sp0, _ = val_metric(embed(), data, device, is_multi, target_steps=K_TARGET)
    print(f"  [init/frozen-head] val {'mean-within-ep' if is_multi else 'neigh'}_sp={sp0:.3f}")

    # Selection metric: lower val RMSE@1 (the stated goal) or higher val Spearman, per select_by.
    best = {"score": (1e9 if select_by == "rmse1" else -2.0), "sp": -2.0, "rmse1": float("nan"),
            "epoch": -1, "state": None}
    is_better = (lambda new, cur: new < cur) if select_by == "rmse1" else (lambda new, cur: new > cur)
    print(f"  [select-by={select_by}]")
    hist = []
    model.head_warmup(True)
    for ep in range(args.epochs):
        if ep == args.warmup:
            model.head_warmup(False)  # unfreeze configured backbone layers (no-op for state-only: no backbone)
            if not state_only:
                print(f"  [epoch {ep}] backbone unfrozen ({args.freeze})")
        sampler.set_epoch(ep)
        model.train()
        t0 = time.time()
        tot = torch.zeros((), device=device)   # accumulate on-GPU; one host sync per epoch (not per batch)
        for p, w, idx in loader:
            p, w = p.to(device, non_blocking=True), w.to(device, non_blocking=True)
            idxn = idx.numpy()
            if needs_state:
                if is_multi and not mix:             # same-task batch -> standardize inputs by THIS task's stats
                    bt0 = int(data.task[idxn[0]])
                    model.set_norm(data.task_am[bt0], data.task_asd[bt0],
                                   data.task_pm[bt0], data.task_psd[bt0])
                prev_b = torch.as_tensor(data.prev[idxn], device=device)
                pro_b = torch.as_tensor(data.proprio[idxn], device=device)
                if uses_init:                        # pair each point with its episode-initial frame (un-augmented)
                    ii = init_idx[idxn]
                    pi = E._to_chw(data.primary[ii]).to(device, non_blocking=True)
                    wi = E._to_chw(data.wrist[ii]).to(device, non_blocking=True)
                    z = model(p, w, prev_b, pro_b, primary_init=pi, wrist_init=wi)
                else:
                    z = model(p, w, prev_b, pro_b)
            else:
                z = model(p, w)
            act = torch.as_tensor(data.act[idxn][:, :K_TARGET], device=device)  # (B,K,7) metric-target chunk
            if mix:                                                       # batch spans tasks -> per-sample z-scoring
                ti = torch.as_tensor(data.task[idxn], device=device)
                am_b, asd_b = am_t[ti].unsqueeze(1), asd_t[ti].unsqueeze(1)   # (B,1,7), broadcast over the 16 steps
            else:
                bt = int(data.task[idxn[0]])                              # same-task batch -> its single action stats
                am_b, asd_b = am_t[bt], asd_t[bt]
            ld = label_dist(action_feat(act, am_b, asd_b))                # (B,B) action-chunk distances
            ep_b = torch.as_tensor(data.ep[idxn], device=device)          # episode ids (for cross-episode top1nca)
            loss = loss_fn(z, ld, ep_b)
            if args.progress:                                             # auxiliary task-progress regression on z
                prog_b = torch.as_tensor(data.prog[idxn], device=device)
                loss = loss + args.progress_weight * torch.nn.functional.smooth_l1_loss(model.progress(z), prog_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()                # no per-batch .item() sync -> lets prefetch overlap compute
        tot = float(tot)
        sched.step()
        sp, aux = val_metric(embed(), data, device, is_multi, target_steps=K_TARGET)
        val_rmse1 = (aux["rmse1"] if is_multi else float(aux))   # multi-task: MEAN per-task val RMSE@1
        score = val_rmse1 if select_by == "rmse1" else sp
        hist.append({"epoch": ep, "loss": tot / len(loader), "val_sel_sp": sp, "val_rmse1": val_rmse1})
        if ep % 2 == 0 or ep == args.epochs - 1:
            extra = (f"val_RMSE@1={val_rmse1:.4f} per-task_sp={[round(x, 2) for x in aux['per_task_sp']]}"
                     if is_multi else f"val_RMSE@1={aux:.4f}")
            print(f"  ep{ep:03d} loss={tot / len(loader):.4f} val_sp={sp:.3f} {extra} ({time.time() - t0:.0f}s)")
        if is_better(score, best["score"]):
            best = {"score": score, "sp": sp, "rmse1": val_rmse1, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    _bm = (f"RMSE@1={best['rmse1']:.4f}" if select_by == "rmse1" else f"sp={best['sp']:.3f}")
    print(f"\n  best (by {select_by}) val {_bm} (sp={best['sp']:.3f}) @ epoch {best['epoch']}")
    model.load_state_dict(best["state"])
    emb = embed()
    raw = E.raw_r3m_feats(data, device, backbone=args.backbone)
    raw_std = E._std(raw, raw)[0]
    if args.progress:                          # report progress-head quality + learned retrieved-progress correspondence
        from eval_retrieval import _spearman
        with torch.no_grad():
            phat = model.progress(torch.as_tensor(emb[data.te], device=device)).cpu().numpy()
        p_te = data.prog[data.te]
        r2 = 1.0 - np.mean((phat - p_te) ** 2) / (np.var(p_te) + 1e-9)
        pear_head = float(np.corrcoef(phat, p_te)[0, 1])
        # learned-key retrieved-progress (test->train top-1) vs the oracle ceiling (probe_progress: ~0.82-0.99)
        Dq = torch.cdist(torch.as_tensor(emb[data.te], device=device), torch.as_tensor(emb[data.tr], device=device))
        j = Dq.argmin(1).cpu().numpy()
        pear_retr = float(np.corrcoef(p_te, data.prog[data.tr][j])[0, 1])
        print(f"\n[progress] head: R2={r2:.3f} pearson(p_hat,p)={pear_head:.3f} | "
              f"learned-retrieval progress pearson(p_q,p_retr)={pear_retr:.3f}  "
              f"mean|Δp|={np.mean(np.abs(p_te - data.prog[data.tr][j])):.3f}")
        _prog_metrics = {"head_r2": float(r2), "head_pearson": pear_head,
                         "retrieved_progress_pearson": pear_retr}
    # Results live under results/<task>/ (the dataset basename w/o the _dense[_img] suffix; multi-task -> multiN).
    _names = [os.path.basename(d.rstrip("/")).replace("_dense_img", "").replace("_dense", "") for d in data_dirs]
    task_dir = _names[0] if len(_names) == 1 else f"multi{len(_names)}"
    out_dir = os.path.join(HERE, "results", task_dir)
    os.makedirs(out_dir, exist_ok=True)
    report = {"args": vars(args), "select_by": select_by, "best_epoch": best["epoch"],
              "best_val_sp": best["sp"], "best_val_rmse1": best["rmse1"], "history": hist,
              "pertask_norm": bool(is_multi and not mix)}
    if args.progress:
        report["progress_metrics"] = _prog_metrics

    if is_multi:
        print("\n=== per-task TEST: within-episode Spearman (frozen R3M -> learned) | retrieval RMSE@1 ===")
        per_task = {}
        for t, name in enumerate(data.tasks):
            mte = data.te & (data.task == t)
            kw = dict(am=data.task_am[t], asd=data.task_asd[t])
            we = E.within_episode_spearman(emb, data, mte, **kw)
            we_tr = E.within_episode_spearman(emb, data, data.tr & (data.task == t), **kw)
            raw_we = E.within_episode_spearman(raw_std, data, mte, **kw)
            r1 = E.task_rmse1(emb, data, t, device, raw=raw)
            per_task[name] = {"within_ep_frozen": raw_we, "within_ep_learned": we, "within_ep_learned_train": we_tr,
                              "rmse1": r1}
            print(f"  {name:24s} within-ep {raw_we:.3f} -> {we:.3f} (train {we_tr:.3f}) | "
                  f"RMSE@1 learned {r1['learned']:.3f} vs proprio_prev {r1['proprio_prev']:.3f} "
                  f"vs frozen {r1['frozen_r3m']:.3f}")
        mwe = float(np.mean([v["within_ep_learned"] for v in per_task.values()]))
        mwe_f = float(np.mean([v["within_ep_frozen"] for v in per_task.values()]))
        print(f"\n  MEAN within-episode Spearman (test): frozen_r3m={mwe_f:.3f} -> learned={mwe:.3f}")
        report.update({"per_task": per_task, "mean_within_ep_learned_test": mwe,
                       "mean_within_ep_frozen_test": mwe_f})
    else:
        de = data                                          # TEST report on the target horizon (first-K steps)
        if K_TARGET < FULL_STEPS:
            from dataclasses import replace
            de = replace(data, act=np.ascontiguousarray(np.asarray(data.act[:, :K_TARGET])))
        rows = E.evaluate(emb, de, device, raw=raw, ks=(1, 5), split=("tr", "te"))
        we_tr = E.within_episode_spearman(emb, de, de.tr)
        we_te = E.within_episode_spearman(emb, de, de.te)
        raw_we_te = E.within_episode_spearman(raw_std, de, de.te)
        print("\n=== TEST (db=train, query=test; non-grip z-scored retrieval RMSE, lower=better) ===")
        print(E.fmt(rows, ks=(1, 5)))
        print(f"\nwithin-episode Spearman:  frozen_r3m(test)={raw_we_te:.3f}  "
              f"learned(test)={we_te:.3f}  learned(train)={we_tr:.3f}")
        report.update({"test_rows": rows, "within_ep_spearman": {"frozen_r3m_test": raw_we_te,
                       "learned_test": we_te, "learned_train": we_tr}})

    if is_multi:
        # Per-task-trained encoders are deployed with a per-task re-norm (set the buffers to the target
        # task's stats). Save a DETERMINISTIC fallback = global pooled stats (best["state"] would otherwise
        # carry whichever task's stats the last embed/train batch left in the buffers).
        g_pm = data.proprio[data.tr].mean(0).astype(np.float32)
        g_psd = (data.proprio[data.tr].std(0) + D.EPS).astype(np.float32)
        model.set_norm(data.am, data.asd, g_pm, g_psd)
        save_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        save_state = best["state"]
    torch.save(save_state, os.path.join(out_dir, f"encoder_{tag}.pt"))
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out_dir}/encoder_{tag}.pt , metrics_{tag}.json")


if __name__ == "__main__":
    main()
