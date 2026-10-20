"""Load configs using Isaac Lab's real configclass, with external assets doubled."""

import ast
from copy import deepcopy
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class Asset:
    def __init__(self):
        self.spawn = types.SimpleNamespace(usd_path="test", activate_contact_sensors=False)
        self.init_state = types.SimpleNamespace(pos=(0, 0, 0.12), joint_pos={})
        self.actuators = {"test": "torque-limited"}
        self.prim_path = "{ENV_REGEX_NS}/Robot"

    def copy(self):
        return deepcopy(self)

    def replace(self, **kwargs):
        result = self.copy()
        result.__dict__.update(kwargs)
        return result


def load_cfg():
    # The config decorator itself uses only stdlib. Its dict conversion imports
    # are irrelevant here; importing isaaclab.utils normally starts Omni modules.
    decorator_file = ROOT / "IsaacLab/source/isaaclab/isaaclab/utils/configclass.py"
    tree = ast.parse(decorator_file.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "dict")]
    decorator_module = types.ModuleType("_joint_cfg_decorator_test")
    module = types.ModuleType("_joint_cfg_test")
    with patch.dict(sys.modules, {decorator_module.__name__: decorator_module, module.__name__: module}):
        exec(compile(tree, str(decorator_file), "exec"), decorator_module.__dict__)
        configclass = decorator_module.configclass

        @configclass
        class Base:
            action_space = 24
            robot = Asset()
            environment_preset = "flat"

        @configclass
        class Policy:
            init_noise_std = 0.35

        @configclass
        class Runner:
            policy = Policy()

        path = ROOT / "isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/joint_angle_env_cfg.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        module.__dict__.update(
            math=math, configclass=configclass, SPIDERBOT_CFG=Asset(), cad_spiderbot_cfg=lambda _: Asset(),
            SpiderNavigationEnvCfg=Base, SpiderNavigationPPORunnerCfg=Runner,
            mg90s_robot_cfg=lambda: Asset().replace(prim_path="/World/envs/env_.*/Robot"), MG90S_NO_LOAD_SPEED=10.47198,
        )
        exec(compile(tree, str(path), "exec"), module.__dict__)
        return module, Runner()


class JointAngleConfigurationTests(unittest.TestCase):
    def test_all_variants_instantiate_and_preserve_own_limits_and_actuators(self):
        module, _ = load_cfg()
        for name in ("JointAngleNavigationEnvCfg", "Cad8JointAngleEnvCfg", "Cad6JointAngleEnvCfg", "MG90SJointAngleEnvCfg", "MG90SCad8JointAngleEnvCfg", "MG90SCad6JointAngleEnvCfg"):
            with self.subTest(name=name):
                cfg = getattr(module, name)()
                self.assertEqual(cfg.robot.prim_path, "/World/envs/env_.*/Robot")
                self.assertEqual(cfg.action_semantics, "absolute_joint_angles_v1")
                self.assertEqual(cfg.action_space, 18 if "Cad6" in name else 24)
                self.assertGreater(cfg.target_rate_limit, 0)
                self.assertLess(cfg.target_rate_limit, cfg.servo_velocity_limit)
                for lower, upper in cfg.joint_angle_ranges.values():
                    self.assertLess(lower, upper)
                if "Cad" in name:
                    self.assertEqual(cfg.joint_angle_ranges["tibia"][0], -0.4)
                if "MG90S" in name:
                    self.assertEqual(cfg.target_speed, 0.02)

    def test_policy_noise_and_mutable_ranges_do_not_modify_parent_or_other_instances(self):
        module, original = load_cfg()
        policy = module.JointAnglePPORunnerCfg()
        self.assertEqual(policy.policy.init_noise_std, 0.12)
        self.assertEqual(original.policy.init_noise_std, 0.35)
        first, second = module.Cad6JointAngleEnvCfg(), module.Cad6JointAngleEnvCfg()
        first.joint_angle_ranges["hip"] = (0, 1)
        self.assertEqual(second.joint_angle_ranges["hip"], (-1.4, 1.4))


if __name__ == "__main__":
    unittest.main()
