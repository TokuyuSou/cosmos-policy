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
from metric_losses import action_feat, corr_loss, label_dist, rnc_loss, softnn_loss, supcon_loss, top1nca_loss
from model import ActionMetricEncoder, MultiModalActionMetricEncoder

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
    the original behaviour exactly."""

    def __init__(self, ep, mask, eps_per_batch, frames_per_ep, seed=0, task=None, n=None):
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
        if self.multi:
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
        if not self.multi:
            for _ in range(self.n_batches):
                yield [off + i for i in self._draw(rng, self.ep_ids)]
        else:
            order = [t for t in self.task_list for _ in range(self.n_per_task)]
            rng.shuffle(order)
            for t in order:
                yield [off + i for i in self._draw(rng, self.task_eps[t])]


def val_metric(emb, data, device, is_multi):
    """Given a precomputed embedding, return the model-selection metric (higher=better) + an aux value.
    Multi-task: MEAN over tasks of within-task val within-episode Spearman (the headline metric, with
    each task's own action stats). Single-task: the original cross-pool val neighbourhood Spearman."""
    if is_multi:
        sps = [E.within_episode_spearman(emb, data, data.va & (data.task == t),
                                         am=data.task_am[t], asd=data.task_asd[t])
               for t in range(len(data.tasks))]
        return float(np.mean(sps)), sps
    # Selection only needs the LEARNED rep -> compute just that, skipping the fixed proprio_prev / oracle /
    # random / learned+proprio_prev baselines that E.evaluate() recomputes every epoch (those don't affect
    # selection). Identical learned RMSE@1 + neigh-Spearman -> same chosen checkpoint; the full TEST table
    # is still produced once at the end via E.evaluate().
    dbm, qm = data.tr, data.va
    db_act, q_act = data.act[dbm], data.act[qm]
    rmse, _ = E._retr_rmse(emb[dbm], emb[qm], db_act, q_act, data.am, data.asd, (1,), device)
    sp = E._neigh_spearman(emb[dbm], emb[qm], db_act, q_act, data.am, data.asd, device)
    return sp, rmse[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--loss", choices=["rnc", "corr", "softnn", "corr+softnn", "top1nca", "corr+top1nca",
                                       "supcon", "corr+supcon"], default="rnc",
                    help="supcon = SupCon-style CROSS-EPISODE contrastive (k action-nearest other-episode "
                         "positives; stable multi-positive retrieval loss); corr+supcon = corr (global) + "
                         "supcon (local top-1). softnn/top1nca = soft-argmin NCA variants (high-variance).")
    ap.add_argument("--softnn-temp", type=float, default=0.3, help="softnn softmax temperature")
    ap.add_argument("--softnn-w", type=float, default=1.0, help="softnn weight in corr+softnn")
    ap.add_argument("--top1nca-temp", type=float, default=0.2, help="top1nca softmax temperature (smaller=sharper top-1)")
    ap.add_argument("--top1nca-w", type=float, default=1.0, help="top1nca weight in corr+top1nca")
    ap.add_argument("--supcon-k", type=int, default=8, help="supcon: # action-nearest cross-episode positives")
    ap.add_argument("--supcon-temp", type=float, default=0.1, help="supcon InfoNCE temperature")
    ap.add_argument("--supcon-w", type=float, default=1.0, help="supcon weight in corr+supcon")
    ap.add_argument("--select-by", choices=["spearman", "rmse1"], default=None,
                    help="checkpoint-selection metric (val). Default: rmse1 for single-task fused runs "
                         "(the stated goal is to beat proprio_prev RMSE@1), spearman otherwise.")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--freeze", default="layer4", choices=["frozen", "layer4", "layer34", "none"])
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--arch", choices=["image", "fused"], default="image",
                    help="image: the original images-only encoder. fused: also encode proprio+prev "
                         "and fuse them with the image into one embedding (MultiModalActionMetricEncoder).")
    ap.add_argument("--state-enc", choices=["mlp", "temporal"], default="mlp",
                    help="fused only: state-branch encoder over proprio+prev (mlp flat-input, or a tiny "
                         "Transformer over the 16 prev-action steps).")
    ap.add_argument("--fusion", choices=["concat", "residual"], default="concat",
                    help="fused only: concat->MLP, or residual (learned-state base + gated image "
                         "correction; floor near proprio_prev with image upside).")
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
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or (f"{args.arch}_{args.state_enc}_{args.loss}" if args.arch == "fused" else args.loss)
    import losses as _losses                       # set the gripper toggle for action_feat + the RMSE metric
    _losses.INCLUDE_GRIP = bool(args.include_grip)

    data_dirs = [d for d in args.data.split(",") if d.strip()]
    data = D.load(data_dirs[0], seed=args.seed) if len(data_dirs) == 1 \
        else D.load_multi(data_dirs, seed=args.seed)
    is_multi = len(data.tasks) > 1
    am_t = torch.tensor(data.task_am, device=device)   # (T,7) per-task action stats; loss uses the batch's task
    asd_t = torch.tensor(data.task_asd, device=device)

    ds = FrameDS(data, None if args.no_aug else make_aug(), base_seed=args.seed)
    sampler = EpisodeBatchSampler(data.ep, data.tr, args.eps_per_batch, args.frames_per_ep, args.seed,
                                  task=data.task, n=len(data.act))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=(4 if args.workers > 0 else None))

    fused = args.arch == "fused"
    if fused:
        model = MultiModalActionMetricEncoder(
            args.backbone, "r3m", args.freeze, out_dim=args.out_dim, img_dim=args.img_dim,
            state_dim=args.state_dim, dropout=args.dropout, share_backbone=not args.separate_backbones,
            state_mode=args.state_enc, mod_dropout=args.mod_dropout, fusion=args.fusion,
            prev_steps=data.prev.shape[1], act_dim=data.prev.shape[2],
            proprio_dim=data.proprio.shape[1]).to(device)
        pm = data.proprio[data.tr].mean(0)
        psd = data.proprio[data.tr].std(0) + D.EPS
        model.set_norm(data.am, data.asd, pm, psd)   # action stats for prev, proprio stats for proprio
    else:
        model = ActionMetricEncoder(args.backbone, "r3m", args.freeze, out_dim=args.out_dim,
                                    share_backbone=not args.separate_backbones).to(device)
    # Generic head/backbone split (works for both encoders): the R3M backbone(s) get the small bb-lr,
    # everything else (head / state-encoder / fusion) gets head-lr.
    bb_params = [p for bb in model.backbones() for p in bb.parameters()]
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    embed = (lambda: E.embed_all_mm(model, data, device)) if fused else \
        (lambda: E.embed_all(model, data, device))
    opt = AdamW([{"params": head_params, "lr": args.head_lr, "weight_decay": 1e-2},
                 {"params": bb_params, "lr": args.bb_lr, "weight_decay": 1e-4}])
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = {  # all take (z, ld, ep); ep (episode ids) used only by the cross-episode top1nca term
        "rnc": lambda z, ld, ep: rnc_loss(z, ld, args.tau),
        "corr": lambda z, ld, ep: corr_loss(z, ld),
        "softnn": lambda z, ld, ep: softnn_loss(z, ld, args.softnn_temp),
        "corr+softnn": lambda z, ld, ep: corr_loss(z, ld) + args.softnn_w * softnn_loss(z, ld, args.softnn_temp),
        "top1nca": lambda z, ld, ep: top1nca_loss(z, ld, ep, args.top1nca_temp),
        "corr+top1nca": lambda z, ld, ep: corr_loss(z, ld) + args.top1nca_w * top1nca_loss(z, ld, ep, args.top1nca_temp),
        "supcon": lambda z, ld, ep: supcon_loss(z, ld, ep, args.supcon_k, args.supcon_temp),
        "corr+supcon": lambda z, ld, ep: corr_loss(z, ld) + args.supcon_w * supcon_loss(z, ld, ep, args.supcon_k, args.supcon_temp),
    }[args.loss]
    select_by = args.select_by or ("rmse1" if (fused and not is_multi) else "spearman")

    n_tr = sum(p.numel() for p in head_params + bb_params if p.requires_grad)
    print(f"=== R3M action-metric encoder | arch={args.arch}"
          f"{f'(state={args.state_enc},mod_drop={args.mod_dropout})' if fused else ''} "
          f"loss={args.loss} freeze={args.freeze} "
          f"backbones={'separate' if args.separate_backbones else 'shared'} "
          f"tasks={len(data.tasks)}{'(same-task batches)' if is_multi else ''} "
          f"batch={args.eps_per_batch}x{args.frames_per_ep} | action_dims={_losses.n_act_dims()}"
          f"{'(incl. gripper)' if args.include_grip else '(non-grip)'} | trainable~{n_tr/1e6:.1f}M ===")
    sp0, _ = val_metric(embed(), data, device, is_multi)
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
            model.head_warmup(False)  # unfreeze configured backbone layers
            print(f"  [epoch {ep}] backbone unfrozen ({args.freeze})")
        sampler.set_epoch(ep)
        model.train()
        t0 = time.time()
        tot = torch.zeros((), device=device)   # accumulate on-GPU; one host sync per epoch (not per batch)
        for p, w, idx in loader:
            p, w = p.to(device, non_blocking=True), w.to(device, non_blocking=True)
            idxn = idx.numpy()
            if fused:
                prev_b = torch.as_tensor(data.prev[idxn], device=device)
                pro_b = torch.as_tensor(data.proprio[idxn], device=device)
                z = model(p, w, prev_b, pro_b)
            else:
                z = model(p, w)
            bt = int(data.task[idxn[0]])                                   # batch is same-task -> its action stats
            act = torch.as_tensor(data.act[idxn], device=device)          # (B,16,7) executed chunk
            ld = label_dist(action_feat(act, am_t[bt], asd_t[bt]))        # (B,B) action-chunk distances
            ep_b = torch.as_tensor(data.ep[idxn], device=device)          # episode ids (for cross-episode top1nca)
            loss = loss_fn(z, ld, ep_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()                # no per-batch .item() sync -> lets prefetch overlap compute
        tot = float(tot)
        sched.step()
        sp, aux = val_metric(embed(), data, device, is_multi)
        val_rmse1 = (None if is_multi else float(aux))
        score = val_rmse1 if select_by == "rmse1" else sp
        hist.append({"epoch": ep, "loss": tot / len(loader), "val_sel_sp": sp, "val_rmse1": val_rmse1})
        if ep % 2 == 0 or ep == args.epochs - 1:
            extra = f"per-task={[round(x, 2) for x in aux]}" if is_multi else f"val_RMSE@1={aux:.4f}"
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
    # Results live under results/<task>/ (the dataset basename w/o the _dense[_img] suffix; multi-task -> multiN).
    _names = [os.path.basename(d.rstrip("/")).replace("_dense_img", "").replace("_dense", "") for d in data_dirs]
    task_dir = _names[0] if len(_names) == 1 else f"multi{len(_names)}"
    out_dir = os.path.join(HERE, "results", task_dir)
    os.makedirs(out_dir, exist_ok=True)
    report = {"args": vars(args), "select_by": select_by, "best_epoch": best["epoch"],
              "best_val_sp": best["sp"], "best_val_rmse1": best["rmse1"], "history": hist}

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
        rows = E.evaluate(emb, data, device, raw=raw, ks=(1, 5), split=("tr", "te"))
        we_tr = E.within_episode_spearman(emb, data, data.tr)
        we_te = E.within_episode_spearman(emb, data, data.te)
        raw_we_te = E.within_episode_spearman(raw_std, data, data.te)
        print("\n=== TEST (db=train, query=test; non-grip z-scored retrieval RMSE, lower=better) ===")
        print(E.fmt(rows, ks=(1, 5)))
        print(f"\nwithin-episode Spearman:  frozen_r3m(test)={raw_we_te:.3f}  "
              f"learned(test)={we_te:.3f}  learned(train)={we_tr:.3f}")
        report.update({"test_rows": rows, "within_ep_spearman": {"frozen_r3m_test": raw_we_te,
                       "learned_test": we_te, "learned_train": we_tr}})

    torch.save(best["state"], os.path.join(out_dir, f"encoder_{tag}.pt"))
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out_dir}/encoder_{tag}.pt , metrics_{tag}.json")


if __name__ == "__main__":
    main()
