#!/usr/bin/env python3
"""Real-checkpoint verification for ReferenceServoTMTPolicy.

This is intentionally simulator-free. It checks exact offline/live feature and
action parity, then exercises one fresh + three phase-advance calls with real
CabToCounter cache rows and RGB frames.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = RESEARCH.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import build_samples, list_success_episodes, split_episode_files  # noqa: E402
from reference_servo_tmt_policy import ReferenceServoTMTPolicy  # noqa: E402
from research.reference_servo_tmt.experiment import ridge_features  # noqa: E402
from research.reference_servo_tmt.servo import (  # noqa: E402
    apply_constant_bias,
    denormalize_actions,
    limit_l2,
    normalize_actions,
)


def load_r3m_data_module():
    path = RESEARCH / "r3m_action_encoder" / "data.py"
    name = "reference_servo_verify_data"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def global_row_map(files):
    """Map action_predictor provenance ``(real episode id, src_t)`` to R3M rows."""
    mapping = {}
    row = 0
    for file_index, file in enumerate(files):
        payload = np.load(file, allow_pickle=True)
        realized = payload["realized_actions"]
        query_t = payload["query_t"]
        available = {int(t) for t in query_t}
        try:
            episode_id = int(os.path.basename(file).split("ep")[1].split("_")[0])
        except (IndexError, ValueError) as error:
            raise ValueError(f"cannot parse episode id from {file}") from error
        for tau in query_t:
            tau = int(tau)
            decision = tau + 16
            if tau + 32 <= len(realized) and decision in available:
                mapping[(episode_id, decision)] = row
                row += 1
    return mapping, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(RESEARCH / "data" / "pnp_cab_to_counter_dense_img_orig150"))
    parser.add_argument("--tmt", default=str(
        RESEARCH / "r3m_action_encoder" / "results" / "pnp_cab_to_counter_orig150" /
        "tmt_d192L3_failq.pt"
    ))
    parser.add_argument("--servo", default=str(
        RESEARCH / "reference_servo_tmt" / "results" / "cab_seed0" /
        "persistent_twist_ridge.npz"
    ))
    args = parser.parse_args()

    policy = ReferenceServoTMTPolicy(args.data, args.tmt, args.servo, local_replan_steps=4)
    r3m_data = load_r3m_data_module()
    data = r3m_data.load(args.data, seed=0, val_frac=0.15, test_frac=0.15)
    files = r3m_data.list_success_episodes(args.data)
    row_map, count = global_row_map(files)
    assert count == len(data.act), (count, len(data.act))

    query_row = int(np.flatnonzero(data.te)[0])
    reference_idx = 0
    provenance = (int(policy.retr.src_ep[reference_idx]), int(policy.retr.src_t[reference_idx]))
    reference_row = row_map[provenance]
    action_z = normalize_actions(data.act, data.am, data.asd)
    prev_z = normalize_actions(data.prev, data.am, data.asd)
    offline_feature = ridge_features(
        data, action_z, prev_z, np.asarray([query_row]), np.asarray([reference_row]),
        np.zeros(1, np.float32),
        policy.feature_kind,
    )[0]
    live_feature = policy._raw_features(data.prev[query_row], data.proprio[query_row], reference_idx)
    feature_error = float(np.max(np.abs(offline_feature - live_feature)))
    assert feature_error < 2e-5, feature_error

    standardized = (offline_feature - policy.feature_mean) / policy.feature_std
    bias = standardized @ policy.weight + policy.target_mean
    bias = limit_l2(bias[None], policy.max_bias_norm)[0]
    expected_z = apply_constant_bias(
        action_z[reference_row : reference_row + 1], bias[None], policy.correction_dims
    )[0]
    expected = denormalize_actions(expected_z, policy.action_mean, policy.action_std)
    policy.last_match = {}
    live = policy._correct_reference(data.prev[query_row], data.proprio[query_row], reference_idx)
    action_error = float(np.max(np.abs(expected - live)))
    assert action_error < 2e-5, action_error
    print(f"[parity] feature max_abs={feature_error:.3g} action max_abs={action_error:.3g}")

    cache_files = split_episode_files(list_success_episodes(args.data), 0.15, 0)[0]
    samples = build_samples(cache_files, [0], ["actual_next_proprio"], with_image=True)
    start = next(
        i for i in range(4, len(samples) - 3)
        if all(
            samples[i + j].src_ep == samples[i].src_ep and
            samples[i + j].src_t == samples[i].src_t + 4 * j
            for j in range(-4, 4)
        )
    )
    policy.reset()
    before = dict(policy.track_stats)
    outputs = []
    for j in range(4):
        sample = samples[start + j]
        previous_frame = samples[start + j - 4]
        output = policy.predict_chunk(
            sample.prev_actions, sample.states["actual_next_proprio"], None, None,
            current_image=sample.image, current_wrist=sample.wrist,
            prev_image=previous_frame.image, prev_wrist=previous_frame.wrist,
        )
        outputs.append(output)
        print(
            f"[cycle] call={j + 1} mode={policy.last_match['mode']} "
            f"cache_idx={policy.last_match['cache_idx']} bias={policy.last_prediction['bias_norm']:.4f}"
        )
    delta = {key: int(value) - int(before[key]) for key, value in policy.track_stats.items()}
    assert delta == {"fresh": 1, "advance": 3, "advance_missing": 0, "ridge_calls": 4}, delta
    assert all(output.shape == (16, 7) and np.isfinite(output).all() for output in outputs)
    print("REFERENCE-SERVO REAL CHECKPOINT VERIFICATION PASSED")


if __name__ == "__main__":
    main()
