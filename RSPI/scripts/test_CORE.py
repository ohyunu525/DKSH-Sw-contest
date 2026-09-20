"""Pure Python contract checks between the Pi runner and CAD6 training code."""

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import CORE


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "isaaclab_project/source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation"


def class_constants(path: Path, class_name: str) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    values = {}
    for node in definition.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


class CoreTrainingContractTests(unittest.TestCase):
    def test_training_dimensions_ranges_and_action_transform(self):
        cad6 = TASK_DIR / "mg90s_cad6.py"
        base = class_constants(TASK_DIR / "mg90s_env_cfg.py", "MG90SWalkEnvCfg")
        walk = class_constants(cad6, "MG90SCad6EnvCfg")
        sprint = class_constants(cad6, "MG90SCad6SprintEnvCfg")
        self.assertEqual((walk["observation_space"], walk["action_space"]),
                         (CORE.OBSERVATIONS, CORE.ACTIONS))
        self.assertEqual((base["action_scale"], base["action_smoothing"]),
                         (CORE.ACTION_SCALE_RAD, CORE.ACTION_SMOOTHING))
        self.assertEqual((walk["command_speed_min"], walk["command_speed_max"]),
                         (CORE.PROFILES["walk"].min_speed_mps, CORE.PROFILES["walk"].max_speed_mps))
        self.assertEqual((sprint["command_speed_min"], sprint["command_speed_max"]),
                         (CORE.PROFILES["sprint"].min_speed_mps, CORE.PROFILES["sprint"].max_speed_mps))

    def test_observation_positions_match_training_concat_order(self):
        state = CORE.sample_state()
        state["root_lin_vel_b"] = [1.0, 2.0, 3.0]
        state["root_ang_vel_b"] = [4.0, 5.0, 6.0]
        state["joint_pos"][0] += 0.25
        state["joint_vel"][0] = 2.0
        state["filtered_actions"][0] = -0.4
        state["phase"] = 0.25
        observation = CORE.build_observation(state)
        self.assertEqual(len(observation), 68)
        self.assertEqual(observation[:6], [1, 2, 3, 4, 5, 6])
        self.assertEqual((observation[8], observation[10], observation[11]), (-1.0, 0.0, 0.0))
        self.assertAlmostEqual(observation[9], 0.0045)
        self.assertEqual(observation[12], 0.25)
        self.assertAlmostEqual(observation[30], 0.2)
        self.assertEqual(observation[48], -0.4)
        self.assertAlmostEqual(observation[66], 1.0)
        self.assertAlmostEqual(observation[67], 0.0)

    def test_untrained_motion_commands_are_rejected(self):
        state = CORE.sample_state()
        state["command"] = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            CORE.build_observation(state)
        state["command"] = [0.0045, 0.0, 0.1]
        with self.assertRaises(ValueError):
            CORE.build_observation(state)
        self.assertEqual(len(CORE.build_observation(CORE.sample_state("sprint"), "sprint")), 68)

    def test_action_clamp_and_smoothing(self):
        action = CORE.process_action([2.0, -2.0] + [0.0] * 16, [0.5] * 18)
        self.assertEqual(action["action_clipped"][:2], [1.0, -1.0])
        self.assertAlmostEqual(action["filtered_actions_next"][0], 0.6)
        self.assertAlmostEqual(action["filtered_actions_next"][1], 0.2)
        self.assertAlmostEqual(action["joint_residual_rad"][0], 0.021)

    def test_model_task_must_match_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run_metadata.json"
            metadata = {"task": CORE.PROFILES["walk"].task, "observations": 68,
                        "actions": 18, "legs": 6, "status": "completed",
                        "mass_kg": CORE.EXPECTED_SIM_MASS_KG}
            path.write_text(json.dumps(metadata), encoding="utf-8")
            CORE._validate_metadata(path, "walk")
            metadata["mass_kg"] = 2.31362
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                CORE._validate_metadata(path, "walk")
            metadata["mass_kg"] = CORE.EXPECTED_SIM_MASS_KG
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                CORE._validate_metadata(path, "sprint")
            metadata["task"] = "Isaac-DKSH-Spider-CAD6-Navigation-Direct-v0"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                CORE._validate_metadata(path, "walk")

    def test_cli_records_processed_action_without_motor_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = io.StringIO()
            log = Path(temp) / "core.jsonl"
            with patch.object(CORE, "infer_once", return_value=[2.0] + [0.0] * 17):
                with redirect_stdout(output):
                    code = CORE.main(["--model", str(Path(temp) / "policy.onnx"), "--log", str(log)])
            self.assertEqual(code, 0)
            record = json.loads(output.getvalue())
            self.assertEqual(record["status"], "inferred")
            self.assertEqual(record["action_clipped"][0], 1.0)
            self.assertAlmostEqual(record["filtered_actions_next"][0], 0.2)
            self.assertAlmostEqual(record["joint_residual_rad"][0], 0.007)
            self.assertNotIn("joint_target_rad", record)
            self.assertEqual(json.loads(log.read_text(encoding="utf-8"))["status"], "inferred")


if __name__ == "__main__":
    unittest.main()
