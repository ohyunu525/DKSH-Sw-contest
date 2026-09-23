"""Check the default Nav2 envelope against the CAD6 v3 stance budget."""

import math
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (ROOT / 'ros2/config/nav2_low_resource.yaml').read_text(encoding='utf-8')


def scalar(name):
    match = re.search(r'^\s*' + re.escape(name) + r':\s*([-+\d.]+)\s*$', CONFIG, re.MULTILINE)
    if match is None:
        raise AssertionError(f'missing Nav2 parameter: {name}')
    return float(match.group(1))


class Nav2Cad6ContractTests(unittest.TestCase):
    def test_controller_and_smoother_fit_stance_budget(self):
        self.assertLessEqual(scalar('vx_max'), 0.050)
        self.assertGreaterEqual(scalar('vx_min'), -0.025)
        self.assertLessEqual(scalar('vy_max'), 0.025)
        self.assertLessEqual(scalar('wz_max'), 0.08)
        urdf = ET.parse(ROOT / 'isaaclab_project/assets/spiderbot_variants/spiderbot_6leg/spiderbot_6leg.urdf')
        hip = urdf.find("joint[@name='leg_0_hip_joint']")
        hip_radius = float(hip.find('origin').get('xyz').split()[0])
        # Conservative upper bound for the 0.0282 m hip-local stance radius.
        local_radius = 0.030
        max_foot_speed = math.hypot(scalar('vx_max'), scalar('vy_max')) + scalar('wz_max') * (hip_radius + local_radius)
        self.assertLess(max_foot_speed * 0.6 * 5 / 6, 0.025)
        self.assertLessEqual(scalar('max_rotational_vel'), scalar('wz_max'))

    def test_progress_requirement_is_feasible_at_commanded_speed(self):
        self.assertIn('plugin: nav2_controller::PoseProgressChecker', CONFIG)
        self.assertLess(
            scalar('required_movement_radius'),
            scalar('vx_max') * scalar('movement_time_allowance'),
        )
        self.assertLess(
            scalar('required_movement_angle'),
            scalar('wz_max') * scalar('movement_time_allowance'),
        )


if __name__ == '__main__':
    unittest.main()
