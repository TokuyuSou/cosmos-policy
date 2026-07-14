from __future__ import annotations

import unittest

import numpy as np

from experiment import persist_from_fresh
from servo import (
    apply_constant_bias,
    constant_bias_target,
    quaternion_relative_rotvec,
)


class ServoTest(unittest.TestCase):
    def test_constant_bias_is_exact_and_preserves_gripper(self):
        reference = np.zeros((2, 16, 7), np.float32)
        target = reference.copy()
        target[:, :4, :3] = np.array([[1, 2, 3], [-1, 0.5, 0.25]])[:, None]
        target[:, :4, 6] = 1.0
        bias = constant_bias_target(reference, target, (0, 1, 2))
        pred = apply_constant_bias(reference, bias, (0, 1, 2))
        np.testing.assert_allclose(pred[:, :4, :3], target[:, :4, :3])
        np.testing.assert_array_equal(pred[..., 6], reference[..., 6])

    def test_quaternion_relative_rotation(self):
        identity = np.array([[0, 0, 0, 1]], np.float32)
        half = np.sqrt(0.5)
        z90 = np.array([[0, 0, half, half]], np.float32)
        got = quaternion_relative_rotvec(z90, identity)
        np.testing.assert_allclose(got, [[0, 0, np.pi / 2]], atol=1e-6)
        # Quaternion sign must not alter the rotation.
        np.testing.assert_allclose(quaternion_relative_rotvec(-z90, identity), got, atol=1e-6)

    def test_persistent_reference_retrieves_once_per_four_rows(self):
        ep = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        query_rows = np.arange(5)
        fresh = np.array([5, 6, 7, 8, 9])
        distance = np.arange(5, dtype=np.float32) + 0.5
        reference, carried_distance, retrieved, fallback = persist_from_fresh(
            fresh, distance, query_rows, ep, period_rows=4
        )
        np.testing.assert_array_equal(reference, [5, 6, 7, 8, 9])
        np.testing.assert_allclose(carried_distance, [0.5, 0.5, 0.5, 0.5, 4.5])
        np.testing.assert_array_equal(retrieved, [True, False, False, False, True])
        self.assertFalse(fallback.any())


if __name__ == "__main__":
    unittest.main()
