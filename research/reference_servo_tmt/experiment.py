#!/usr/bin/env python3
"""Train and offline-evaluate persistent TMT@16 + Ridge@4 on CabToCounter.

The script is intentionally independent of the training/evaluation entrypoints.
It reuses a frozen TMT checkpoint and writes only token/embedding caches,
offline metrics, and the episode-disjoint deployable ridge checkpoint here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = RESEARCH.parent
R3M = RESEARCH / "r3m_action_encoder"
ACTION_PREDICTOR = RESEARCH / "action_predictor"
for path in (str(R3M), str(ACTION_PREDICTOR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data import load as load_data  # noqa: E402
try:  # package import (closed-loop verifier) and direct-script execution
    from .servo import (  # type: ignore  # noqa: E402
        ARM, POS, apply_constant_bias, constant_bias_target, denormalize_actions,
        limit_l2, normalize_actions, quaternion_relative_rotvec,
    )
except ImportError:
    from servo import (  # noqa: E402
        ARM, POS, apply_constant_bias, constant_bias_target, denormalize_actions,
        limit_l2, normalize_actions, quaternion_relative_rotvec,
    )
import tmt_policy as TMT_POLICY  # noqa: E402


def prev_index(episodes, rows=4):
    """Return the same-episode row ``rows`` positions earlier, else self."""
    episodes = np.asarray(episodes)
    candidate = np.arange(len(episodes)) - int(rows)
    safe = np.maximum(candidate, 0)
    valid = (candidate >= 0) & (episodes[safe] == episodes)
    return np.where(valid, candidate, np.arange(len(episodes)))


def post_index(episodes, rows=4):
    """Return the same-episode row ``rows`` positions later, else ``-1``."""
    episodes = np.asarray(episodes)
    candidate = np.arange(len(episodes)) + int(rows)
    safe = np.minimum(candidate, len(episodes) - 1)
    valid = (candidate < len(episodes)) & (episodes[safe] == episodes)
    return np.where(valid, candidate, -1)


def load_or_create_token_caches(data, requests, feature_dim, device, batch_size=128):
    """Load Theia grid-token caches, computing any missing/stale grids from raw RGB."""
    loaded = {}
    missing = []
    for path, grid in requests:
        path = Path(path)
        signature = hashlib.sha256(
            f"{len(data.act)}|{grid}|{feature_dim}".encode()
            + np.asarray(data.primary[0]).tobytes()[:4096]
            + np.asarray(data.primary[-1]).tobytes()[:4096]
            + np.asarray(data.wrist[0]).tobytes()[:4096]
            + np.asarray(data.wrist[-1]).tobytes()[:4096]
        ).hexdigest()[:20]
        if path.exists():
            cached = np.load(path)
            expected = (len(data.act), 2 * grid * grid, feature_dim)
            if ("signature" in cached.files and str(cached["signature"]) == signature
                    and cached["tok"].shape == expected):
                loaded[path] = cached["tok"]
                continue
        missing.append((path, int(grid), signature))

    if missing:
        model_name = TMT_POLICY.TheiaBackbone.MODEL_BY_DIM.get(
            feature_dim, TMT_POLICY.TheiaBackbone.DEFAULT_MODEL
        )
        backbone = TMT_POLICY.TheiaBackbone(model_name, freeze="frozen").to(device).eval()
        output = {
            path: np.empty((len(data.act), 2 * grid * grid, feature_dim), np.float32)
            for path, grid, _ in missing
        }
        with torch.no_grad():
            for start in range(0, len(data.act), batch_size):
                stop = min(start + batch_size, len(data.act))
                primary = torch.from_numpy(
                    np.ascontiguousarray(data.primary[start:stop])
                ).permute(0, 3, 1, 2).to(device)
                wrist = torch.from_numpy(
                    np.ascontiguousarray(data.wrist[start:stop])
                ).permute(0, 3, 1, 2).to(device)
                for path, grid, _ in missing:
                    output[path][start:stop] = TMT_POLICY.theia_grid_tokens(
                        backbone, primary, wrist, grid
                    ).float().cpu().numpy()
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for path, _, signature in missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, tok=output[path], signature=signature)
            loaded[path] = output[path]
            print(f"[tokens] computed {path} {output[path].shape}", flush=True)
    return [loaded[Path(path)] for path, _ in requests]


class Embedder:
    """Minimal clean-token TMT embedder used to build the ridge dataset."""

    def __init__(self, data, image_tokens, effect_tokens, previous, post, device):
        self.device = device
        self.image_tokens = torch.as_tensor(image_tokens, dtype=torch.float16, device=device)
        self.effect_tokens = torch.as_tensor(effect_tokens, dtype=torch.float16, device=device)
        self.previous = torch.as_tensor(previous, device=device)
        self.prev_action = torch.as_tensor(data.prev, dtype=torch.float32, device=device)
        self.proprio = torch.as_tensor(data.proprio, dtype=torch.float32, device=device)

        valid = post >= 0
        post_or_self = np.where(valid, post, np.arange(len(post)))
        self.effect_scene = (
            torch.as_tensor(
                effect_tokens[post_or_self] - effect_tokens,
                dtype=torch.float16,
                device=device,
            )
            * torch.as_tensor(valid, device=device)[:, None, None]
        )
        self.effect_proprio = torch.as_tensor(
            (data.proprio[post_or_self] - data.proprio) * valid[:, None],
            dtype=torch.float32,
            device=device,
        )

    def batch(self, rows, key_mode):
        kwargs = {
            "img_cur": self.image_tokens[rows].float(),
            "img_prev": self.image_tokens[self.previous[rows]].float(),
            "prev": self.prev_action[rows],
            "proprio": self.proprio[rows],
        }
        if key_mode:
            kwargs.update(
                eff_s=self.effect_scene[rows].float(),
                eff_p=self.effect_proprio[rows],
            )
        return kwargs

    @torch.no_grad()
    def embed_rows(self, model, rows, key_mode, batch_size=256):
        model.eval()
        output = []
        for start in range(0, len(rows), batch_size):
            batch_rows = torch.as_tensor(rows[start : start + batch_size], device=self.device)
            output.append(model.embed(**self.batch(batch_rows, key_mode))[0].float().cpu())
        return torch.cat(output).numpy()


@dataclass
class Metrics:
    rmse7: float
    arm_rmse6: float
    pos_rmse3: float
    rot_rmse3: float
    grip_rmse1: float
    query_rmse7_p90: float
    grip_sign_mismatch: float
    action_outside_unit: float
    boundary_jump: float
    target_boundary_jump: float
    boundary_ratio: float


def evaluate(pred_z, target_z, prev_z, action_mean, action_std, horizon=4) -> Metrics:
    pred_z = np.asarray(pred_z)[:, :horizon]
    target_z = np.asarray(target_z)[:, :horizon]

    def rmse(dims):
        # Match train_tmt.retr_rmse and all published offline TMT tables:
        # compute an RMSE per query, then average queries equally.
        per_query_group = np.sqrt(
            ((pred_z[..., dims] - target_z[..., dims]) ** 2).mean(axis=(1, 2))
        )
        return float(per_query_group.mean())

    per_query = np.sqrt(((pred_z - target_z) ** 2).mean(axis=(1, 2)))
    pred = denormalize_actions(pred_z, action_mean, action_std)
    mismatch = np.signbit(pred[..., 6]) != np.signbit(
        denormalize_actions(target_z, action_mean, action_std)[..., 6]
    )
    pred_jump = np.sqrt(((pred_z[:, 0, :6] - prev_z[:, -1, :6]) ** 2).mean(axis=1))
    true_jump = np.sqrt(((target_z[:, 0, :6] - prev_z[:, -1, :6]) ** 2).mean(axis=1))
    natural = float(true_jump.mean())
    boundary = float(pred_jump.mean())
    return Metrics(
        rmse7=rmse(np.arange(7)),
        arm_rmse6=rmse(ARM),
        pos_rmse3=rmse(POS),
        rot_rmse3=rmse(np.arange(3, 6)),
        grip_rmse1=rmse(np.array([6])),
        query_rmse7_p90=float(np.quantile(per_query, 0.9)),
        grip_sign_mismatch=float(mismatch.mean()),
        action_outside_unit=float((np.abs(pred) > 1.0 + 1e-6).mean()),
        boundary_jump=boundary,
        target_boundary_jump=natural,
        boundary_ratio=boundary / max(natural, 1e-12),
    )


def paired_episode_comparison(baseline_z, pred_z, target_z, episode, horizon=4, seed=0):
    """Paired improvement with an episode-level bootstrap confidence interval."""
    base = np.sqrt(((baseline_z[:, :horizon] - target_z[:, :horizon]) ** 2).mean(axis=(1, 2)))
    pred = np.sqrt(((pred_z[:, :horizon] - target_z[:, :horizon]) ** 2).mean(axis=(1, 2)))
    unique = np.unique(episode)
    by_episode = np.asarray([base[episode == e].mean() - pred[episode == e].mean() for e in unique])
    rng = np.random.RandomState(seed)
    sampled = by_episode[rng.randint(0, len(unique), size=(10000, len(unique)))].mean(axis=1)
    return {
        "absolute_rmse7_reduction": float(base.mean() - pred.mean()),
        "relative_rmse7_reduction": float(1.0 - pred.mean() / base.mean()),
        "episodes_improved": int((by_episode > 0).sum()),
        "episodes_total": int(len(unique)),
        "episode_mean_reduction_ci95": [float(x) for x in np.quantile(sampled, (0.025, 0.975))],
        "episode_reductions": {str(int(e)): float(d) for e, d in zip(unique, by_episode)},
    }


def nearest(query, key, query_rows, database_rows, ep, exclude_same_episode=False, block=512):
    """Exact asymmetric TMT top-1 in bounded CPU memory."""
    query_rows = np.asarray(query_rows, np.int64)
    database_rows = np.asarray(database_rows, np.int64)
    db = torch.from_numpy(key[database_rows]).float()
    result = []
    distance = []
    for start in range(0, len(query_rows), block):
        rows = query_rows[start : start + block]
        dist = torch.cdist(torch.from_numpy(query[rows]).float(), db)
        if exclude_same_episode:
            mask = torch.from_numpy(ep[rows, None] == ep[database_rows][None, :])
            dist.masked_fill_(mask, float("inf"))
        value, local = dist.min(dim=1)
        result.append(database_rows[local.numpy()])
        distance.append(value.numpy())
    return np.concatenate(result), np.concatenate(distance).astype(np.float32)


def successor_index(ep: np.ndarray) -> np.ndarray:
    """Next dataset row in the same episode, or ``-1`` at episode end."""
    ep = np.asarray(ep)
    out = np.full(len(ep), -1, np.int64)
    valid = np.flatnonzero(ep[:-1] == ep[1:])
    out[valid] = valid + 1
    return out


def persist_from_fresh(
    fresh_reference: np.ndarray,
    fresh_distance: np.ndarray,
    query_rows: np.ndarray,
    ep: np.ndarray,
    period_rows: int = 4,
):
    """Use fresh TMT only every ``period_rows`` and phase-advance in between.

    The inputs contain fresh top-1 results for every row for convenient offline
    vectorization.  Non-anchor results are never read.  If a committed
    reference reaches the end of its demonstration, the current row is
    re-anchored early rather than repeating an invalid action.
    """
    fresh_reference = np.asarray(fresh_reference, np.int64)
    fresh_distance = np.asarray(fresh_distance, np.float32)
    query_rows = np.asarray(query_rows, np.int64)
    assert len(fresh_reference) == len(fresh_distance) == len(query_rows)
    assert period_rows >= 1
    successor = successor_index(ep)
    reference = np.full(len(query_rows), -1, np.int64)
    distance = np.full(len(query_rows), np.nan, np.float32)
    retrieved = np.zeros(len(query_rows), bool)
    fallback = np.zeros(len(query_rows), bool)

    previous_query = -1
    current_reference = -1
    anchor_distance = np.nan
    position = 0
    for i, query_row in enumerate(query_rows):
        new_episode = i == 0 or ep[query_row] != ep[previous_query] or query_row != previous_query + 1
        if new_episode:
            position = 0
            current_reference = -1
        scheduled = position % period_rows == 0
        if not scheduled and current_reference >= 0:
            current_reference = int(successor[current_reference])
        missing_successor = not scheduled and current_reference < 0
        if scheduled or missing_successor:
            current_reference = int(fresh_reference[i])
            anchor_distance = float(fresh_distance[i])
            retrieved[i] = True
            fallback[i] = missing_successor
        reference[i] = current_reference
        distance[i] = anchor_distance
        previous_query = int(query_row)
        position += 1
    assert (reference >= 0).all() and np.isfinite(distance).all()
    return reference, distance, retrieved, fallback


def reference_continuity(reference_rows, query_rows, ep):
    """Reference episode switches between adjacent rows of one query episode."""
    reference_rows = np.asarray(reference_rows)
    query_rows = np.asarray(query_rows)
    adjacent = (ep[query_rows[1:]] == ep[query_rows[:-1]]) & (query_rows[1:] == query_rows[:-1] + 1)
    if not adjacent.any():
        return 0.0
    switched = ep[reference_rows[1:]] != ep[reference_rows[:-1]]
    return float(switched[adjacent].mean())


def reference_break_rate(reference_rows, query_rows, ep):
    """Fraction not equal to the expected next row on adjacent query windows."""
    reference_rows = np.asarray(reference_rows)
    query_rows = np.asarray(query_rows)
    adjacent = (ep[query_rows[1:]] == ep[query_rows[:-1]]) & (query_rows[1:] == query_rows[:-1] + 1)
    if not adjacent.any():
        return 0.0
    expected = successor_index(ep)[reference_rows[:-1]]
    broken = reference_rows[1:] != expected
    return float(broken[adjacent].mean())


def embedding_cache(data, checkpoint: Path, img_tokens: Path, eff_tokens: Path, cache: Path, device: str):
    model, g_img, g_eff, feature_dim = TMT_POLICY.load_tmt(str(checkpoint), device)
    img, eff = load_or_create_token_caches(
        data, [(img_tokens, g_img), (eff_tokens, g_eff)], feature_dim, device
    )
    signature = hashlib.sha256(
        f"{checkpoint.resolve()}|{checkpoint.stat().st_mtime_ns}|{img_tokens.stat().st_mtime_ns}|"
        f"{eff_tokens.stat().st_mtime_ns}|{len(data.act)}".encode()
    ).hexdigest()[:20]
    if cache.exists():
        z = np.load(cache)
        if str(z["signature"]) == signature and len(z["query"]) == len(data.act):
            print(f"[embedding] loaded {cache}", flush=True)
            return z["query"], z["key"]

    assert img.shape[0] == eff.shape[0] == len(data.act)
    assert img.shape[1] == 2 * g_img * g_img and eff.shape[1] == 2 * g_eff * g_eff
    post = post_index(data.ep, rows=4)
    before = prev_index(data.ep, rows=4)
    embedder = Embedder(data, img, eff, before, post, device)
    rows = np.arange(len(data.act))
    print(f"[embedding] computing {len(rows)} query/key embeddings on {device}", flush=True)
    query = embedder.embed_rows(model, rows, key_mode=False).astype(np.float32)
    key = embedder.embed_rows(model, rows, key_mode=True).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, query=query, key=key, signature=signature)
    print(f"[embedding] saved {cache}", flush=True)
    return query, key


def ridge_features(data, action_z, prev_z, query_rows, reference_rows, hit_distance, kind):
    qpro = data.proprio[query_rows]
    rpro = data.proprio[reference_rows]
    # First five proprio values are Euclidean (gripper2 + eef xyz); the
    # quaternion is represented as a sign-invariant relative rotvec.
    euclidean = (qpro[:, :5] - rpro[:, :5]) / data.task_psd[0, :5]
    rotation = quaternion_relative_rotvec(qpro[:, 5:9], rpro[:, 5:9])
    previous_last = prev_z[query_rows, -1] - prev_z[reference_rows, -1]
    previous_mean4 = prev_z[query_rows, -4:].mean(1) - prev_z[reference_rows, -4:].mean(1)
    reference_first4 = action_z[reference_rows, :4, :6].mean(1)
    base_without_distance = np.concatenate(
        [euclidean, rotation, previous_last, previous_mean4, reference_first4], axis=1
    ).astype(np.float32)
    if kind == "state_no_distance":
        return base_without_distance
    base = np.concatenate([base_without_distance, hit_distance[:, None]], axis=1).astype(np.float32)
    if kind == "state":
        return base
    raise ValueError(kind)


class Ridge:
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def fit(self, x, y):
        self.mean = x.mean(0)
        self.std = x.std(0)
        self.std[self.std < 1e-6] = 1.0
        xs = (x - self.mean) / self.std
        self.ymean = y.mean(0)
        yc = y - self.ymean
        # SVD form is stable for both the 29-D and 285-D feature variants.
        u, s, vt = np.linalg.svd(xs, full_matrices=False)
        self.weight = (vt.T * (s / (s * s + self.alpha))) @ (u.T @ yc)
        return self

    def predict(self, x):
        return ((x - self.mean) / self.std) @ self.weight + self.ymean


def save_ridge_model(model, model_path, alpha, max_norm, dims, features, data, **metadata):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        feature_mean=model.mean.astype(np.float32),
        feature_std=model.std.astype(np.float32),
        target_mean=model.ymean.astype(np.float32),
        weight=model.weight.astype(np.float32),
        alpha=np.float32(alpha),
        max_bias_norm=np.float32(max_norm),
        correction_dims=np.asarray(dims, np.int64),
        feature_kind=np.asarray(features),
        action_mean=data.am,
        action_std=data.asd,
        proprio_input_std=data.task_psd[0],
    )
    payload.update(metadata)
    np.savez_compressed(model_path, **payload)
    return str(model_path.resolve())


def fit_persistent_twist_report(
    data,
    action_z,
    prev_z,
    query,
    key,
    train_rows,
    val_rows,
    test_rows,
    deploy_db,
    fresh_test_reference,
    fresh_test_distance,
    result,
    model_path,
    period_rows=4,
):
    """Train/evaluate TMT@16 + state-servo@4 under the exact periodic contract."""

    def persistent_pairs(rows, database, exclude_same_episode=False):
        fresh_reference, fresh_distance = nearest(
            query, key, rows, database, data.ep, exclude_same_episode
        )
        persistent = persist_from_fresh(
            fresh_reference, fresh_distance, rows, data.ep, period_rows
        )
        return persistent, fresh_reference

    (train_reference, train_distance, _, _), _ = persistent_pairs(
        train_rows, train_rows, True
    )
    (val_reference, val_distance, _, _), _ = persistent_pairs(
        val_rows, train_rows
    )
    dims = ARM
    y_train = constant_bias_target(action_z[train_reference], action_z[train_rows], dims)
    candidates = []
    # The current TMT embedding is intentionally absent between anchors.
    # Carrying the anchor distance is deployable; omitting it is even cheaper.
    for features in ("state_no_distance", "state"):
        x_train = ridge_features(
            data, action_z, prev_z, train_rows, train_reference, train_distance, features
        )
        x_val = ridge_features(
            data, action_z, prev_z, val_rows, val_reference, val_distance, features
        )
        for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
            model = Ridge(alpha).fit(x_train, y_train)
            raw = model.predict(x_val)
            for max_norm in (0.2, 0.3, 0.5, 0.8, math.inf):
                bias = limit_l2(raw, max_norm)
                pred = apply_constant_bias(action_z[val_reference], bias, dims)
                metrics = evaluate(pred, action_z[val_rows], prev_z[val_rows], data.am, data.asd)
                candidates.append((metrics.rmse7, features, alpha, max_norm))
    _, features, alpha, max_norm = min(candidates)

    # Freeze validation choices, then refit using all non-test episodes.
    fit_rows = np.concatenate([train_rows, val_rows])
    (fit_reference, fit_distance, _, _), _ = persistent_pairs(fit_rows, deploy_db, True)
    x_fit = ridge_features(
        data, action_z, prev_z, fit_rows, fit_reference, fit_distance, features
    )
    y_fit = constant_bias_target(action_z[fit_reference], action_z[fit_rows], dims)
    model = Ridge(alpha).fit(x_fit, y_fit)

    test_reference, test_distance, retrieved, fallback = persist_from_fresh(
        fresh_test_reference, fresh_test_distance, test_rows, data.ep, period_rows
    )
    x_test = ridge_features(
        data, action_z, prev_z, test_rows, test_reference, test_distance, features
    )
    bias = limit_l2(model.predict(x_test), max_norm)
    pred = apply_constant_bias(action_z[test_reference], bias, dims)
    persistent_baseline = action_z[test_reference]
    metrics = asdict(evaluate(pred, action_z[test_rows], prev_z[test_rows], data.am, data.asd))
    metrics.update(
        features=features,
        alpha=float(alpha),
        max_bias_norm=None if not np.isfinite(max_norm) else float(max_norm),
        feature_dim=int(x_fit.shape[1]),
        fit_rows=int(len(fit_rows)),
        bias_l2_p50=float(np.quantile(np.linalg.norm(bias, axis=1), 0.5)),
        bias_l2_p90=float(np.quantile(np.linalg.norm(bias, axis=1), 0.9)),
        paired_vs_persistent_baseline=paired_episode_comparison(
            persistent_baseline, pred, action_z[test_rows], data.ep[test_rows]
        ),
        paired_vs_fresh_tmt4=paired_episode_comparison(
            action_z[fresh_test_reference], pred, action_z[test_rows], data.ep[test_rows]
        ),
        model_file=save_ridge_model(
            model, model_path, alpha, max_norm, dims, features, data,
            tmt_period_steps=np.int64(4 * period_rows),
            local_replan_steps=np.int64(4),
        ),
    )
    result["persistent_tmt16_baseline"] = asdict(evaluate(
        persistent_baseline, action_z[test_rows], prev_z[test_rows], data.am, data.asd
    ))
    result["persistent_tmt16_twist_ridge4"] = metrics
    result["persistent_tmt16_twist_ridge4_selection"] = {
        "criterion": "validation first-4 rmse7 under the persistent-reference contract",
        "num_candidates": len(candidates),
        "best_validation_rmse7": float(min(candidates)[0]),
    }
    result["persistent_protocol"] = {
        "local_replan_steps": 4,
        "tmt_period_steps": int(4 * period_rows),
        "test_action_windows": int(len(test_rows)),
        "tmt_retrievals": int(retrieved.sum()),
        "scheduled_retrievals": int(sum(
            math.ceil((data.ep[test_rows] == e).sum() / period_rows)
            for e in np.unique(data.ep[test_rows])
        )),
        "early_retrievals_at_reference_end": int(fallback.sum()),
        "retrieval_fraction": float(retrieved.mean()),
        "retrieval_reduction_vs_tmt4": float(1.0 - retrieved.mean()),
        "fresh_reference_episode_switch_rate": reference_continuity(
            fresh_test_reference, test_rows, data.ep
        ),
        "persistent_reference_episode_switch_rate": reference_continuity(
            test_reference, test_rows, data.ep
        ),
        "fresh_reference_break_rate": reference_break_rate(
            fresh_test_reference, test_rows, data.ep
        ),
        "persistent_reference_break_rate": reference_break_rate(
            test_reference, test_rows, data.ep
        ),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(RESEARCH / "data" / "pnp_cab_to_counter_dense_img_orig150"))
    p.add_argument("--checkpoint", default=str(
        R3M / "results" / "pnp_cab_to_counter_orig150" / "tmt_d192L3_failq.pt"))
    p.add_argument("--img-tokens", default=str(
        HERE / "cache" / "cab_theia_g7.npz"))
    p.add_argument("--eff-tokens", default=str(
        HERE / "cache" / "cab_theia_g4.npz"))
    p.add_argument("--embedding-cache", default=str(HERE / "cache" / "cab_tmt_embeddings.npz"))
    p.add_argument("--output", default=str(HERE / "results" / "cab_seed0" / "metrics.json"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    data = load_data(args.data, seed=args.seed, val_frac=0.15, test_frac=0.15)
    query, key = embedding_cache(data, Path(args.checkpoint), Path(args.img_tokens),
                                 Path(args.eff_tokens), Path(args.embedding_cache), args.device)
    train_rows, val_rows, test_rows = (np.flatnonzero(m) for m in (data.tr, data.va, data.te))
    deploy_db = np.concatenate([train_rows, val_rows])
    test_reference, test_distance = nearest(query, key, test_rows, deploy_db, data.ep)
    offline_reference, _ = nearest(query, key, test_rows, train_rows, data.ep)
    action_z = normalize_actions(data.act, data.am, data.asd)
    prev_z = normalize_actions(data.prev, data.am, data.asd)
    output = Path(args.output)
    result = {
        "experiment": "reference_servo_tmt_cab",
        "protocol": {
            "seed": args.seed,
            "train_rows": int(len(train_rows)),
            "val_rows": int(len(val_rows)),
            "test_rows": int(len(test_rows)),
            "train_episodes": int(len(np.unique(data.ep[train_rows]))),
            "val_episodes": int(len(np.unique(data.ep[val_rows]))),
            "test_episodes": int(len(np.unique(data.ep[test_rows]))),
            "retrieval_database": "train+val success rows",
            "evaluation_horizon": 4,
            "cache_row_environment_steps": 4,
            "checkpoint": str(Path(args.checkpoint).resolve()),
        },
        "baseline": asdict(evaluate(action_z[test_reference], action_z[test_rows], prev_z[test_rows],
                                    data.am, data.asd)),
        "published_protocol_train_only_baseline": asdict(evaluate(
            action_z[offline_reference], action_z[test_rows], prev_z[test_rows], data.am, data.asd
        )),
    }
    fit_persistent_twist_report(
        data, action_z, prev_z, query, key, train_rows, val_rows, test_rows,
        deploy_db, test_reference, test_distance, result,
        output.parent / "persistent_twist_ridge.npz", period_rows=4,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
