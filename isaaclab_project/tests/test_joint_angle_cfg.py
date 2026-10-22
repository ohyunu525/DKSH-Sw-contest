"""Load configs using Isaac Lab's real configclass, with external assets doubled."""

import ast
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"
SERVO_SPEC = importlib.util.spec_from_file_location("_joint_cfg_servos", FOLDER / "servo_specs.py")
SERVO = importlib.util.module_from_spec(SERVO_SPEC)
SERVO_SPEC.loader.exec_module(SERVO)


class Asset:
    def __init__(self):
        self.spawn = types.SimpleNamespace(usd_path="test", activate_contact_sensors=False,
                                           articulation_props=types.SimpleNamespace(enabled_self_collisions=False))
        self.init_state = types.SimpleNamespace(pos=(0, 0, 0.12), joint_pos={})
        self.actuators = {"test": "torque-limited"}
        self.prim_path = "{ENV_REGEX_NS}/Robot"

    def copy(self):
        return deepcopy(self)

    def replace(self, **kwargs):
        result = self.copy()
        result.__dict__.update(kwargs)
        return result


def cad_asset(legs):
    robot = Asset()
    name = "onshape_20260928" if legs == 6 else "spiderbot_8leg"
    robot.spawn.usd_path = str(ROOT / f"isaaclab_project/assets/spiderbot_variants/{name}/{name}.usd")
    robot.init_state.pos = (0, 0, 0.1275 if legs == 6 else 0.12004)
    robot.init_state.joint_pos = {".*_joint": 0.0}
    return robot


def servo_factories():
    namespace = {name: getattr(SERVO, name) for name in vars(SERVO) if name.isupper()}
    namespace.update(SPIDERBOT_CFG=Asset(), DCMotorCfg=types.SimpleNamespace,
                     STANCE_DEPTH=0.19, STANCE_FEMUR=0.65, STANCE_TIBIA=0.75,
                     RC920DMG_STALL_TORQUE=SERVO.RC920DMG_STALL_TORQUE_NM_5V,
                     RC920DMG_NO_LOAD_SPEED=SERVO.RC920DMG_NO_LOAD_SPEED_RAD_S_5V)
    tree = ast.parse((FOLDER / "mg90s_env_cfg.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("mixed_servo_robot_cfg", "historical_mg90s_robot_cfg")]
    exec(compile(tree, "mg90s_env_cfg.py", "exec"), namespace)
    return namespace


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
            math=math, Path=Path, configclass=configclass, SPIDERBOT_CFG=Asset(), cad_spiderbot_cfg=cad_asset,
            SpiderNavigationEnvCfg=Base, SpiderNavigationPPORunnerCfg=Runner,
            mg90s_robot_cfg=servo_factories()["historical_mg90s_robot_cfg"],
            mixed_servo_robot_cfg=servo_factories()["mixed_servo_robot_cfg"],
        )
        module.__dict__.update({name: getattr(SERVO, name) for name in vars(SERVO) if name.isupper()})
        exec(compile(tree, str(path), "exec"), module.__dict__)
        return module, Runner()


class JointAngleConfigurationTests(unittest.TestCase):
    def test_all_variants_instantiate_and_preserve_own_limits_and_actuators(self):
        module, _ = load_cfg()
        for name in ("JointAngleNavigationEnvCfg", "Cad8JointAngleEnvCfg", "Cad6JointAngleEnvCfg", "MG90SJointAngleEnvCfg", "MG90SCad8JointAngleEnvCfg", "MG90SCad6JointAngleEnvCfg", "MG996RHipCad6JointAngleEnvCfg"):
            with self.subTest(name=name):
                cfg = getattr(module, name)()
                self.assertEqual(cfg.robot.prim_path, "/World/envs/env_.*/Robot")
                self.assertEqual(cfg.action_semantics, "absolute_joint_angles_v1")
                self.assertEqual(cfg.action_space, 18 if "Cad6" in name else 24)
                self.assertGreater(cfg.target_rate_limit, 0)
                speed = cfg.servo_velocity_limit
                self.assertLess(cfg.target_rate_limit, min(speed) if isinstance(speed, tuple) else speed)
                for lower, upper in cfg.joint_angle_ranges.values():
                    self.assertLess(lower, upper)
                if name in ("Cad8JointAngleEnvCfg", "MG90SCad8JointAngleEnvCfg", "MG90SCad6JointAngleEnvCfg"):
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
        self.assertEqual(second.joint_angle_ranges["hip"], (-math.pi / 2, math.pi / 2))

    def test_current_cad6_uses_main_zero_limits_foot_frames_and_mixed_actuators(self):
        module, _ = load_cfg()
        cfg = module.MG996RHipCad6JointAngleEnvCfg()
        asset = ROOT / "isaaclab_project/assets/spiderbot_variants/onshape_20260928/robot.training.urdf"
        robot = ET.parse(asset).getroot()
        self.assertEqual(Path(cfg.robot.spawn.usd_path), asset.with_name("onshape_20260928.usd"))
        self.assertEqual(cfg.robot.init_state.pos, (0, 0, 0.1275))
        self.assertEqual(cfg.robot.init_state.joint_pos, {".*_joint": 0.0})
        self.assertTrue(cfg.use_asset_joint_limits)
        self.assertFalse(cfg.enable_self_collisions)
        for leg in range(6):
            foot = robot.find(f"joint[@name='leg_{leg}_foot_fixed']/origin")
            self.assertEqual(cfg.sole_offset[leg], tuple(map(float, foot.get("xyz").split())))
            for part in ("hip", "femur", "tibia"):
                limit = robot.find(f"joint[@name='leg_{leg}_{part}_joint']/limit")
                self.assertEqual(cfg.joint_angle_ranges[part], (float(limit.get("lower")), float(limit.get("upper"))))
                name = "mg996r_4v8_hip" if part == "hip" else "rc920dmg_5v_leg"
                actuator = cfg.robot.actuators[name]
                self.assertAlmostEqual(actuator.effort_limit_sim, float(limit.get("effort")), places=5)
                self.assertAlmostEqual(actuator.velocity_limit_sim, float(limit.get("velocity")), places=7)
        self.assertIsNone(cfg.servo_model)

    def test_historical_mg90s_keeps_its_original_geometry_and_motor_model(self):
        module, _ = load_cfg()
        cfg = module.MG90SCad6JointAngleEnvCfg()
        self.assertEqual(Path(cfg.robot.spawn.usd_path).parts[-2:], ("spiderbot_6leg", "spiderbot_6leg.usd"))
        self.assertEqual(set(cfg.robot.actuators), {"mg90s_4v8"})
        self.assertAlmostEqual(cfg.robot.actuators["mg90s_4v8"].effort_limit_sim, 0.132389775)


if __name__ == "__main__":
    unittest.main()
