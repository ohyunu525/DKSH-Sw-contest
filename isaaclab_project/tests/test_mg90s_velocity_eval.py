"""The Velocity pass rule must reject a stationary policy."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mg90s_velocity_eval import velocity_pass


class VelocityEvalTests(unittest.TestCase):
    def test_stationary_robot_cannot_pass(self):
        result = dict(
            falls=0, out_of_bounds=0, max_effort_ratio=0.2,
            zero_speed_error_mps=0.005352, zero_yaw_error_rps=0.034245,
            mean_speed_error_mps=0.005352, mean_yaw_error_rps=0.034245,
        )
        self.assertFalse(velocity_pass(result))

    def test_improvement_on_both_axes_is_required(self):
        result = dict(
            falls=0, out_of_bounds=0, max_effort_ratio=0.2,
            zero_speed_error_mps=0.02, zero_yaw_error_rps=0.04,
            mean_speed_error_mps=0.006, mean_yaw_error_rps=0.02,
        )
        self.assertTrue(velocity_pass(result))
        result['mean_yaw_error_rps'] = 0.04
        self.assertFalse(velocity_pass(result))
        result['mean_yaw_error_rps'] = 0.02
        result['zero_speed_error_mps'] = 0.001
        self.assertFalse(velocity_pass(result))


if __name__ == '__main__':
    unittest.main()
