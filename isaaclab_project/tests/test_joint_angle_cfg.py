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
GEOMETRY_SPEC = importlib.util.spec_from_file_location("_joint_cfg_feet", FOLDER / "joint_angle_foot_geometry.py")
GEOMETRY = importlib.util.module_from_spec(GEOMETRY_SPEC)
GEOMETRY_SPEC.loader.exec_module(GEOMETRY)


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
    namespace.update(DCMotorCfg=types.SimpleNamespace)
    tree = ast.parse((FOLDER / "robot_actuators.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "mixed_servo_actuators"]
    exec(compile(tree, "robot_actuators.py", "exec"), namespace)
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
            algorithm = Policy()

        path = ROOT / "isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation/joint_angle_env_cfg.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        module.__dict__.update(
            math=math, Path=Path, configclass=configclass, SPIDERBOT_CFG=Asset(), cad_spiderbot_cfg=cad_asset,
            SpiderNavigationEnvCfg=Base, SpiderNavigationPPORunnerCfg=Runner,
            mixed_servo_actuators=servo_factories()["mixed_servo_actuators"],
            ONSHAPE_FOOT_SUPPORT_OFFSETS=GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS,
        )
        module.__dict__.update({name: getattr(SERVO, name) for name in vars(SERVO) if name.isupper()})
        exec(compile(tree, str(path), "exec"), module.__dict__)
        return module, Runner()


class JointAngleConfigurationTests(unittest.TestCase):
    def test_all_variants_instantiate_and_preserve_own_limits_and_actuators(self):
        module, _ = load_cfg()
        for name in ("JointAngleNavigationEnvCfg", "Cad8JointAngleEnvCfg", "Cad6JointAngleEnvCfg", "MG996RHipCad6JointAngleEnvCfg"):
            with self.subTest(name=name):
                cfg = getattr(module, name)()
                self.assertEqual(cfg.robot.prim_path, "/World/envs/env_.*/Robot")
                self.assertEqual(cfg.action_semantics, "absolute_joint_angles_v2")
                self.assertEqual(cfg.action_space, 18 if "Cad6" in name else 24)
                self.assertGreater(cfg.target_rate_limit, 0)
                self.assertEqual(cfg.observation_space, 140 if "Cad6" in name else 166)
                self.assertEqual(cfg.joint_reset_noise, 0)
                self.assertGreaterEqual(cfg.goal_min_distance, 0.5)
                self.assertLess(cfg.speed_cost_scale, 0)
                self.assertEqual(cfg.behavior_profile, "revised_v3")
                self.assertEqual(cfg.reward_contract, "signed_progress_huber_speed_v3")
                self.assertEqual(cfg.contact_contract, "support_surface_foot_geometry_v3")
                self.assertEqual((cfg.exploration_initial_std, cfg.exploration_min_std, cfg.exploration_max_std), (0.08, 0.03, 0.30))
                speed = cfg.servo_velocity_limit
                self.assertLess(cfg.target_rate_limit, min(speed) if isinstance(speed, tuple) else speed)
                for lower, upper in cfg.joint_angle_ranges.values():
                    self.assertLess(lower, upper)
                if name == "Cad8JointAngleEnvCfg":
                    self.assertEqual(cfg.joint_angle_ranges["tibia"][0], -0.4)

    def test_policy_noise_and_mutable_ranges_do_not_modify_parent_or_other_instances(self):
        module, original = load_cfg()
        policy = module.JointAnglePPORunnerCfg()
        self.assertEqual(policy.policy.init_noise_std, 0.08)
        self.assertEqual(policy.algorithm.entropy_coef, 0.001)
        self.assertEqual(policy.algorithm.gamma, 0.9995)
        self.assertEqual(policy.num_steps_per_env, 128)
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
        self.assertEqual(cfg.robot.init_state.pos, (0, 0, 0.1375))
        self.assertEqual(cfg.robot.init_state.joint_pos, {".*_joint": 0.0})
        self.assertTrue(cfg.use_asset_joint_limits)
        self.assertTrue(cfg.enable_self_collisions)
        self.assertGreater(cfg.robot_width, 0.556867)
        self.assertEqual(len(cfg.foot_support_offsets), 6)
        self.assertTrue(all(len(points) == 137 for points in cfg.foot_support_offsets))
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

    def test_mixed_cad6_preserves_tuning_without_historical_class_inheritance(self):
        module, _ = load_cfg()
        cfg = module.MG996RHipCad6JointAngleEnvCfg()
        self.assertEqual(module.MG996RHipCad6JointAngleEnvCfg.__bases__, (module.Cad6JointAngleEnvCfg,))
        self.assertEqual((cfg.minimum_base_height, cfg.target_speed, cfg.speed_tracking_sigma,
                          cfg.episode_length_s), (0.06, 0.02, 0.015, 240.0))
        self.assertEqual((cfg.target_rate_limit, cfg.target_smoothing), (1.5, 0.35))
        self.assertEqual((cfg.goal_min_distance, cfg.goal_max_distance, cfg.goal_radius), (0.5, 1.5, 0.08))
        self.assertEqual((cfg.robot_width, cfg.robot_height), (0.56, 0.24))
        self.assertEqual(cfg.actuator_contract, "mg996r_4v8_rc920dmg_5v_v1")
        self.assertEqual(cfg.foot_support_offsets, GEOMETRY.ONSHAPE_FOOT_SUPPORT_OFFSETS)
        self.assertEqual((cfg.sole_contact_height_tolerance, cfg.foot_contact_height_tolerance), (0.012, 0.004))
        self.assertEqual(cfg.servo_velocity_limit,
                         (SERVO.MG996R_NO_LOAD_SPEED_RAD_S, SERVO.RC920DMG_NO_LOAD_SPEED_RAD_S_5V,
                          SERVO.RC920DMG_NO_LOAD_SPEED_RAD_S_5V) * 6)

    def test_mixed_actuator_factory_preserves_physical_limits_and_returns_fresh_configs(self):
        factory = servo_factories()["mixed_servo_actuators"]
        actuators = factory()
        self.assertEqual(set(actuators), {"mg996r_4v8_hip", "rc920dmg_5v_leg"})
        expected = (
            ("mg996r_4v8_hip", [".*_hip_joint"], 0.9218251, 0.691368825, math.pi / 3 / 0.19),
            ("rc920dmg_5v_leg", [".*_femur_joint", ".*_tibia_joint"], 1.8632635, 1.397447625, math.pi / 3 / 0.16),
        )
        for name, joints, stall, limit, speed in expected:
            actuator = actuators[name]
            self.assertEqual(actuator.joint_names_expr, joints)
            self.assertAlmostEqual(actuator.saturation_effort, stall)
            self.assertAlmostEqual(actuator.effort_limit, limit)
            self.assertAlmostEqual(actuator.effort_limit_sim, limit)
            self.assertAlmostEqual(actuator.velocity_limit, speed)
            self.assertAlmostEqual(actuator.velocity_limit_sim, speed)
            self.assertEqual((actuator.stiffness, actuator.damping, actuator.armature, actuator.friction),
                             (3.0, 0.06, 0.0002, 0.0))
        actuators["mg996r_4v8_hip"].stiffness = 99.0
        self.assertEqual(factory()["mg996r_4v8_hip"].stiffness, 3.0)

    def test_registry_keeps_current_robots_and_excludes_removed_servo_tasks(self):
        registrations = []
        gym = types.ModuleType("gymnasium")
        gym.register = lambda **kwargs: registrations.append(kwargs)
        with patch.dict(sys.modules, {"gymnasium": gym}):
            exec(compile((FOLDER / "__init__.py").read_text(encoding="utf-8"), "task_registry", "exec"),
                 {"__name__": "spider_navigation"})
        task_ids = {item["id"] for item in registrations}
        self.assertEqual(task_ids, {
            "Isaac-DKSH-Spider-JointAngles-v2", "Isaac-DKSH-Spider-CAD8-JointAngles-v2",
            "Isaac-DKSH-Spider-CAD6-JointAngles-v2", "Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2",
            "Isaac-DKSH-Spider-Navigation-Direct-v0", "Isaac-DKSH-Spider-CAD8-Navigation-Direct-v0",
            "Isaac-DKSH-Spider-CAD6-Navigation-Direct-v0",
        })


if __name__ == "__main__":
    unittest.main()
