"""Fast tests for the opt-in Reference-Servo TMT closed-loop adapter."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reference_servo_tmt_policy import ReferenceServoTMTPolicy  # noqa: E402
from tmt_policy import _phase_advance_map  # noqa: E402
import closed_loop as closed_loop_runner  # noqa: E402


class FakeRetrieval:
    def __init__(self):
        self.values = np.stack([
            np.stack([np.full(7, 100 * row + step, np.float32) for step in range(16)])
            for row in range(6)
        ])
        self.src_ep = [0, 0, 0, 0, 0, 1]
        self.src_t = [0, 4, 8, 12, 16, 0]
        self.src_imgidx = list(range(6))
        self.last_match = None

    def _record_match(self, index, distance):
        index = int(index)
        self.last_match = {
            "cache_idx": index,
            "src_ep": self.src_ep[index],
            "src_t": self.src_t[index],
            "src_imgidx": self.src_imgidx[index],
            "dist": float(distance[index]),
        }


def fake_policy():
    policy = ReferenceServoTMTPolicy.__new__(ReferenceServoTMTPolicy)
    policy.local_replan_steps = 4
    policy.tmt_period_steps = 16
    policy.period_queries = 4
    policy.output_horizon = 16
    policy.retr = FakeRetrieval()
    policy.keys = np.zeros((6, 4), np.float32)
    policy.next_i = _phase_advance_map(policy.retr.src_ep, policy.retr.src_t, 4)
    policy.cache_prev = np.zeros((6, 16, 7), np.float32)
    policy.cache_proprio = np.zeros((6, 9), np.float32)
    policy.cache_proprio[:, 8] = 1.0
    policy.feature_kind = "state_no_distance"
    policy.servo_type = "ridge"
    policy.feature_mean = np.zeros(28, np.float32)
    policy.feature_std = np.ones(28, np.float32)
    policy.weight = np.zeros((28, 6), np.float32)
    policy.target_mean = np.ones(6, np.float32)
    policy.max_bias_norm = np.inf
    policy.correction_dims = np.arange(6)
    policy.action_mean = np.zeros(7, np.float32)
    policy.action_std = np.ones(7, np.float32)
    policy.proprio_input_std = np.ones(9, np.float32)
    policy.track_stats = {"fresh": 0, "advance": 0, "advance_missing": 0, "ridge_calls": 0}
    policy._reference_idx = -1
    policy._period_position = 0
    policy._anchor_distance = np.nan
    policy.last_prediction = None
    policy.last_match = None
    policy._distance_queue = [
        np.array([0.1, 0.4, 0.5, 0.6, 0.7, 0.8], np.float32),
        np.array([0.8, 0.7, 0.6, 0.5, 0.1, 0.4], np.float32),
    ]
    policy._query_dists = lambda *args, **kwargs: policy._distance_queue.pop(0)
    return policy


def inputs():
    proprio = np.zeros(9, np.float32)
    proprio[8] = 1.0
    image = np.zeros((2, 2, 3), np.uint8)
    return dict(
        prev_actions=np.zeros((16, 7), np.float32),
        current_proprio=proprio,
        current_image=image,
        current_wrist=image,
        prev_image=image,
        prev_wrist=image,
    )


def test_periodic_selection_and_correction():
    policy = fake_policy()
    outputs = [policy.predict_chunk(**inputs()) for _ in range(5)]
    # One visual TMT lookup, three exact +4 cache successors, then a new
    # lookup at the next 16-step block boundary.
    assert [int(x[0, 0] - 1) for x in outputs] == [0, 100, 200, 300, 400]
    assert policy.track_stats == {"fresh": 2, "advance": 3, "advance_missing": 0, "ridge_calls": 5}
    assert len(policy._distance_queue) == 0
    # The same 6-D bias modifies only the executed first four actions.
    np.testing.assert_allclose(outputs[0][:4, :6], policy.retr.values[0, :4, :6] + 1)
    np.testing.assert_allclose(outputs[0][4:, :6], policy.retr.values[0, 4:, :6])
    np.testing.assert_array_equal(outputs[0][:, 6], policy.retr.values[0, :, 6])
    assert policy.last_match["mode"] == "fresh"  # fifth call starts the next TMT16 block

    policy = fake_policy()
    policy.predict_chunk(**inputs())
    policy.predict_chunk(**inputs())
    assert policy.last_match["mode"] == "advance" and policy.last_match["cache_idx"] == 1


def test_missing_successor_requeries_and_vla_resets():
    policy = fake_policy()
    policy._distance_queue = [
        np.array([0.8, 0.7, 0.6, 0.5, 0.1, 0.4], np.float32),
        np.array([0.1, 0.4, 0.5, 0.6, 0.7, 0.8], np.float32),
        np.array([0.8, 0.7, 0.6, 0.5, 0.1, 0.4], np.float32),
    ]
    first = policy.predict_chunk(**inputs())   # fresh row 4, whose +4 successor is missing
    second = policy.predict_chunk(**inputs())  # safe early re-query -> row 0
    assert int(first[0, 0] - 1) == 400
    assert int(second[0, 0] - 1) == 0
    assert policy.track_stats["advance_missing"] == 1
    assert policy.last_match["mode"] == "advance_missing_requery"
    policy.notify_vla_call()
    policy.predict_chunk(**inputs())
    assert policy.last_match["mode"] == "fresh"


def test_build_policy_is_opt_in_and_wired():
    import run_closed_loop_eval as runner

    made = {}

    class StubPolicy:
        def __init__(self, *args, **kwargs):
            made["args"], made["kwargs"] = args, kwargs

    old = runner.ReferenceServoTMTPolicy
    runner.ReferenceServoTMTPolicy = StubPolicy
    try:
        args = SimpleNamespace(
            policy="reference_servo_tmt", data_dir="cache", tmt_encoder="tmt.pt",
            servo_model="servo.npz", state_source="actual_next_proprio",
            cache_episodes=0, tmt_w=-1.0, local_replan_steps=4,
        )
        runner.build_policy(args)
    finally:
        runner.ReferenceServoTMTPolicy = old
    assert made["args"] == ("cache", "tmt.pt", "servo.npz")
    assert made["kwargs"]["tmt_period_steps"] == 16
    assert made["kwargs"]["local_replan_steps"] == 4


class FakeEnv:
    def close(self):
        pass


class FakeBackend:
    NUM_STEPS_WAIT = 0

    def __init__(self):
        self.step_count = 0

    def make_env(self, cfg, episode_idx, reseed_before_reset=True):
        return FakeEnv(), "instruction", 32

    def dummy_action(self, env, cfg):
        return np.zeros(7, np.float32)

    def prepare_obs(self, obs, cfg):
        step = 0 if obs is None else int(obs["step"])
        image = np.full((2, 2, 3), step, np.uint8)
        proprio = np.zeros(9, np.float32)
        proprio[8] = 1.0
        return {"proprio": proprio, "primary_image": image, "wrist_image": image.copy()}

    def step(self, env, action):
        self.step_count += 1
        return {"step": self.step_count}, 0.0, False, {}

    def is_success(self, env, done, info):
        return False


class AlwaysSkip:
    name = "always"
    needs_vla = False
    last = None

    def reset(self):
        pass

    def decide(self, ctx):
        return True


def fake_cosmos(*args, **kwargs):
    return (
        np.zeros((32, 7), np.float32),
        np.zeros(9, np.float32),
        np.zeros((3, 16, 28, 28), np.float32),
    )


def test_closed_loop_runner_executes_one_fresh_plus_three_advances(monkeypatch):
    policy = fake_policy()
    monkeypatch.setattr(closed_loop_runner, "_cosmos_chunk", fake_cosmos)
    cfg = SimpleNamespace(seed=0, num_denoising_steps_action=1)
    result = closed_loop_runner.run_closed_loop_episode(
        cfg, None, None, policy, AlwaysSkip(), "task", 0,
        deterministic_reset=False, backend=FakeBackend(), local_replan_steps=4,
    )
    assert result["n_skip"] == 1 and result["n_local_replan"] == 3
    assert policy.track_stats == {
        "fresh": 1, "advance": 3, "advance_missing": 0, "ridge_calls": 4,
    }
