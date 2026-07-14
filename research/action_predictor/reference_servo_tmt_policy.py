"""Closed-loop adapter for the split-rate Reference-Servo TMT policy.

The policy is opt-in and leaves :class:`TMTRetrievalPolicy` untouched.  It
performs one global TMT lookup per 16-step skip block, advances along the same
demonstration at each four-step local query, and applies a frozen low-dimensional
ridge-servo correction to the next four actions.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.reference_servo_tmt.servo import (  # noqa: E402
    denormalize_actions,
    limit_l2,
    normalize_actions,
    quaternion_relative_rotvec,
)
from tmt_policy import TMTRetrievalPolicy, _phase_advance_map  # noqa: E402


def resolve_servo_checkpoint(path: str | os.PathLike) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "persistent_twist_ridge.npz"
    if not candidate.is_file():
        raise FileNotFoundError(f"Reference-Servo checkpoint not found: {candidate}")
    return candidate.resolve()


class ReferenceServoTMTPolicy(TMTRetrievalPolicy):
    """TMT@16 reference selection plus a state-conditioned 6-D servo@4.

    A returned chunk is always ``(16, 7)``, but only its first four actions are
    corrected and the runner is required to re-query after those four actions.
    The gripper channel and the remaining reference suffix are preserved.
    """

    needs_prev_frame = True
    oracle_query = False
    output_horizon = 16
    required_local_replan_steps = 4

    def __init__(
        self,
        data_dir: str,
        tmt_ckpt: str,
        servo_checkpoint: str,
        state_source: str = "actual_next_proprio",
        cache_episodes: int | None = None,
        val_frac: float = 0.15,
        seed: int = 0,
        w: float = -1.0,
        tmt_period_steps: int = 16,
        local_replan_steps: int = 4,
    ):
        if "," in str(data_dir):
            raise ValueError("Reference-Servo TMT currently requires one task cache directory")
        if cache_episodes not in (None, 0):
            raise ValueError(
                "the frozen servo was fitted for the default train cache; custom --cache-episodes is unsupported"
            )
        if w > 0:
            raise ValueError("the frozen servo was fitted with the TMT checkpoint's learned w; do not override --tmt-w")
        if int(local_replan_steps) != self.required_local_replan_steps:
            raise ValueError(
                f"Reference-Servo TMT requires local_replan_steps={self.required_local_replan_steps}"
            )
        if int(tmt_period_steps) != 16:
            raise ValueError("this frozen servo checkpoint was trained for tmt_period_steps=16")

        super().__init__(
            data_dir=data_dir,
            tmt_ckpt=tmt_ckpt,
            state_source=state_source,
            cache_episodes=None,
            val_frac=val_frac,
            seed=seed,
            w=w,
        )
        self.local_replan_steps = int(local_replan_steps)
        self.tmt_period_steps = int(tmt_period_steps)
        self.period_queries = self.tmt_period_steps // self.local_replan_steps
        self.next_i = _phase_advance_map(
            self.retr.src_ep, self.retr.src_t, self.local_replan_steps
        )
        if float((self.next_i >= 0).mean()) <= 0.5:
            raise RuntimeError("cache has insufficient same-episode +4-step successor coverage")

        # RetrievalPolicy's prev_state key is exactly [normalized prev(16x7),
        # normalized proprio(9)]. Decode it once to recover the cache-side
        # physical inputs needed by the servo feature contract.
        expected_key_dim = self.output_horizon * 7 + 9
        if self.retr.keys.shape != (len(self.retr.values), expected_key_dim):
            raise RuntimeError(
                f"unexpected cache key shape {self.retr.keys.shape}; expected "
                f"({len(self.retr.values)},{expected_key_dim})"
            )
        prev_key = self.retr.keys[:, : self.output_horizon * 7].reshape(-1, self.output_horizon, 7)
        pro_key = self.retr.keys[:, self.output_horizon * 7 :]
        self.cache_prev = (
            prev_key * self.retr.norm.act_std[None, None] + self.retr.norm.act_mean[None, None]
        ).astype(np.float32)
        self.cache_proprio = (
            pro_key * self.retr.norm.proprio_std[None] + self.retr.norm.proprio_mean[None]
        ).astype(np.float32)

        self.servo_checkpoint = resolve_servo_checkpoint(servo_checkpoint)
        self._load_servo(self.servo_checkpoint)
        self.run_dir = f"{tmt_ckpt}|{self.servo_checkpoint}"
        self.img_mode = (
            f"reference-servo-tmt[{Path(tmt_ckpt).stem}]:"
            f"tmt{self.tmt_period_steps}/ridge{self.local_replan_steps}"
        )
        self.track_stats = {"fresh": 0, "advance": 0, "advance_missing": 0, "ridge_calls": 0}
        self._reference_idx = -1
        self._period_position = 0
        self._anchor_distance = np.nan
        self.last_prediction = None

    def _load_servo(self, checkpoint: Path):
        z = np.load(checkpoint, allow_pickle=False)
        self.servo_type = str(z["servo_type"]) if "servo_type" in z.files else "ridge"
        if self.servo_type != "ridge":
            raise ValueError(
                f"Reference-Servo Ridge4 requires a ridge checkpoint, got {self.servo_type!r}"
            )
        required = {
            "feature_mean", "feature_std", "target_mean", "max_bias_norm",
            "correction_dims", "feature_kind", "action_mean", "action_std",
            "proprio_input_std", "tmt_period_steps", "local_replan_steps",
        }
        required.add("weight")
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"servo checkpoint is missing fields: {sorted(missing)}")
        self.feature_mean = z["feature_mean"].astype(np.float32)
        self.feature_std = z["feature_std"].astype(np.float32)
        self.target_mean = z["target_mean"].astype(np.float32)
        self.max_bias_norm = float(z["max_bias_norm"])
        self.correction_dims = z["correction_dims"].astype(np.int64)
        self.feature_kind = str(z["feature_kind"])
        self.action_mean = z["action_mean"].astype(np.float32)
        self.action_std = z["action_std"].astype(np.float32)
        self.proprio_input_std = z["proprio_input_std"].astype(np.float32)

        expected_dims = {
            "state_no_distance": 28,
            "state": 29,
        }
        if self.feature_kind not in expected_dims:
            raise ValueError(f"unsupported servo feature kind {self.feature_kind!r}")
        expected_dim = expected_dims[self.feature_kind]
        if self.feature_mean.shape != (expected_dim,) or self.feature_std.shape != (expected_dim,):
            raise ValueError("servo feature normalization shape mismatch")
        if self.target_mean.shape != (6,):
            raise ValueError("servo target mean shape mismatch")
        self.weight = z["weight"].astype(np.float32)
        if self.weight.shape != (expected_dim, 6):
            raise ValueError("servo ridge weight shape mismatch")
        if not np.array_equal(self.correction_dims, np.arange(6)):
            raise ValueError("closed-loop adapter expects a six-dimensional arm twist correction")
        if self.action_mean.shape != (7,) or self.action_std.shape != (7,):
            raise ValueError("servo action normalization must have seven channels")
        if self.proprio_input_std.shape != (9,):
            raise ValueError("servo proprio_input_std must have nine channels")
        if int(z["tmt_period_steps"]) != self.tmt_period_steps:
            raise ValueError("servo checkpoint TMT period does not match the policy")
        if int(z["local_replan_steps"]) != self.local_replan_steps:
            raise ValueError("servo checkpoint local replan period does not match the policy")
        model_am = self.model.am.detach().float().cpu().numpy()
        model_asd = self.model.asd.detach().float().cpu().numpy()
        model_psd = self.model.psd.detach().float().cpu().numpy()
        if not np.array_equal(self.action_mean, model_am) or not np.array_equal(self.action_std, model_asd):
            raise ValueError("servo and TMT action normalization differ")
        if not np.array_equal(self.proprio_input_std, model_psd):
            raise ValueError("servo and TMT proprio normalization differ")
        finite_arrays = [
            self.feature_mean, self.feature_std, self.target_mean,
            self.action_mean, self.action_std, self.proprio_input_std,
        ]
        finite_arrays.append(self.weight)
        if not all(np.isfinite(x).all() for x in finite_arrays):
            raise ValueError("servo checkpoint contains non-finite parameters")
        if ((self.feature_std <= 0).any() or (self.action_std <= 0).any()
                or (self.proprio_input_std <= 0).any()):
            raise ValueError("servo checkpoint contains non-positive scales")

    def reset(self):
        self._reference_idx = -1
        self._period_position = 0
        self._anchor_distance = np.nan
        self.last_prediction = None
        self.last_match = None

    def notify_vla_call(self):
        # A cloud block interrupts the local reference trajectory. The next
        # skip must therefore start with a fresh visual TMT lookup.
        self.reset()

    def _fresh_reference(
        self,
        prev_actions,
        current_proprio,
        current_image,
        current_wrist,
        prev_image,
        prev_wrist,
        mode,
    ) -> int:
        if current_image is None or current_wrist is None:
            raise ValueError("Reference-Servo TMT requires current primary and wrist RGB images")
        dist = self._query_dists(
            current_image, current_wrist, prev_actions, current_proprio, prev_image, prev_wrist
        )
        chosen = int(np.argmin(dist))
        self._anchor_distance = float(dist[chosen])
        self.retr._record_match(chosen, dist)
        self.retr.last_match.update({"mode": mode, "reference_advanced": False})
        self.last_match = self.retr.last_match
        self.track_stats["fresh"] += 1
        return chosen

    def _advanced_reference(self, chosen: int):
        self.retr.last_match = {
            "cache_idx": int(chosen),
            "src_ep": self.retr.src_ep[chosen],
            "src_imgidx": self.retr.src_imgidx[chosen],
            "src_t": self.retr.src_t[chosen],
            # Current TMT is intentionally not evaluated at an intermediate
            # query. This is the last anchor's distance and is diagnostic only.
            "dist": float(self._anchor_distance),
            "mode": "advance",
            "reference_advanced": True,
        }
        self.last_match = self.retr.last_match
        self.track_stats["advance"] += 1

    def _raw_features(self, prev_actions, current_proprio, reference_idx):
        previous = np.asarray(prev_actions, np.float32)
        proprio = np.asarray(current_proprio, np.float32)
        if previous.shape != (self.output_horizon, 7):
            raise ValueError(f"prev_actions must be (16,7), got {previous.shape}")
        if proprio.shape != (9,):
            raise ValueError(f"current_proprio must be (9,), got {proprio.shape}")
        reference_prev = self.cache_prev[reference_idx]
        reference_proprio = self.cache_proprio[reference_idx]
        previous_z = normalize_actions(previous, self.action_mean, self.action_std)
        reference_prev_z = normalize_actions(reference_prev, self.action_mean, self.action_std)
        reference_action_z = normalize_actions(
            self.retr.values[reference_idx], self.action_mean, self.action_std
        )
        euclidean = (proprio[:5] - reference_proprio[:5]) / self.proprio_input_std[:5]
        rotation = quaternion_relative_rotvec(
            proprio[None, 5:9], reference_proprio[None, 5:9]
        )[0]
        previous_last = previous_z[-1] - reference_prev_z[-1]
        previous_mean4 = previous_z[-4:].mean(0) - reference_prev_z[-4:].mean(0)
        reference_first4 = reference_action_z[:4, :6].mean(0)
        parts = [euclidean, rotation, previous_last, previous_mean4, reference_first4]
        if self.feature_kind == "state":
            parts.append(np.asarray([self._anchor_distance], np.float32))
        features = np.concatenate(parts).astype(np.float32)
        if features.shape != self.feature_mean.shape or not np.isfinite(features).all():
            raise RuntimeError(f"invalid servo features: shape={features.shape}")
        return features

    def _predict_bias(self, features):
        """Predict one bounded 6-D correction with the selected frozen head."""
        features = np.asarray(features, np.float32)
        standardized = (features - self.feature_mean) / self.feature_std
        bias = standardized @ self.weight + self.target_mean
        return limit_l2(np.asarray(bias, np.float32)[None], self.max_bias_norm)[0]

    def _correct_reference(self, prev_actions, current_proprio, reference_idx):
        features = self._raw_features(prev_actions, current_proprio, reference_idx)
        bias = self._predict_bias(features)
        reference_z = normalize_actions(
            self.retr.values[reference_idx], self.action_mean, self.action_std
        )
        output_z = reference_z.copy()
        output_z[: self.local_replan_steps, self.correction_dims] += bias[None]
        output = denormalize_actions(output_z, self.action_mean, self.action_std).astype(np.float32)
        if output.shape != (self.output_horizon, 7) or not np.isfinite(output).all():
            raise RuntimeError("Reference-Servo produced an invalid action chunk")
        self.track_stats["ridge_calls"] += 1
        self.last_prediction = {
            "reference_idx": int(reference_idx),
            "period_position": int(self._period_position),
            "bias": bias.copy(),
            "bias_norm": float(np.linalg.norm(bias)),
        }
        self.last_match.update({
            "servo_bias_norm": self.last_prediction["bias_norm"],
            "servo_period_position": int(self._period_position),
        })
        return output

    def predict_chunk(
        self,
        prev_actions,
        current_proprio,
        cached_future_proprio=None,
        cached_future_img=None,
        *,
        current_image=None,
        current_wrist=None,
        prev_image=None,
        prev_wrist=None,
    ):
        del cached_future_proprio, cached_future_img
        scheduled = self._reference_idx < 0 or self._period_position == 0
        if scheduled:
            self._reference_idx = self._fresh_reference(
                prev_actions, current_proprio, current_image, current_wrist,
                prev_image, prev_wrist, mode="fresh",
            )
        else:
            advanced = int(self.next_i[self._reference_idx])
            if advanced < 0:
                self.track_stats["advance_missing"] += 1
                self._reference_idx = self._fresh_reference(
                    prev_actions, current_proprio, current_image, current_wrist,
                    prev_image, prev_wrist, mode="advance_missing_requery",
                )
            else:
                self._reference_idx = advanced
                self._advanced_reference(advanced)

        output = self._correct_reference(prev_actions, current_proprio, self._reference_idx)
        self._period_position = (self._period_position + 1) % self.period_queries
        return output


__all__ = ["ReferenceServoTMTPolicy", "resolve_servo_checkpoint"]
