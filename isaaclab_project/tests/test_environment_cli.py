"""Environment selection and checkpoint routing tests without an Isaac Sim runtime."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("dksh_environment_cli", SCRIPTS / "environment_cli.py")
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def make_spec():
    return types.SimpleNamespace(kwargs={
        "env_cfg_entry_point": Mock(side_effect=lambda: types.SimpleNamespace(
            seed=42, environment_preset="flat", environment_difficulty=0.5
        )),
        "rsl_rl_cfg_entry_point": Mock(side_effect=lambda: types.SimpleNamespace(
            seed=42, experiment_name="robot"
        )),
    })


class EnvironmentArgumentsTests(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        CLI.add_environment_args(self.parser)

    def test_default_settings_preserve_flat_and_config_seed(self):
        args = self.parser.parse_args([])
        self.assertEqual((args.environment, args.difficulty, args.seed), ("flat", 0.5, None))

    def test_all_presets_and_difficulty_boundaries(self):
        for preset in CLI.ENVIRONMENTS:
            for difficulty in (0, 0.5, 1):
                with self.subTest(preset=preset, difficulty=difficulty):
                    args = self.parser.parse_args([
                        f"--environment={preset}", f"--difficulty={difficulty}", "--seed=7"
                    ])
                    self.assertEqual((args.environment, args.difficulty, args.seed), (preset, difficulty, 7))

    def test_invalid_options_fail_before_simulator_start(self):
        for option in (
            "--environment=unknown", "--difficulty=nan", "--difficulty=inf", "--difficulty=-0.1",
            "--difficulty=1.1", "--difficulty=invalid", "--seed=-1", "--seed=2147483648", "--seed=1.5",
        ):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parser.parse_args([option])
                self.assertEqual(error.exception.code, 2)


class RunnerConfigurationTests(unittest.TestCase):
    def test_train_and_play_accept_flags_and_hydra_aliases(self):
        for training in (True, False):
            for use_hydra in (True, False):
                with self.subTest(training=training, use_hydra=use_hydra):
                    spec = make_spec()
                    args = [
                        "env.environment_preset=mixed", "env.environment_difficulty=0.8",
                        "env.seed=7", "agent.experiment_name=chosen",
                    ] if use_hydra else [
                        "--environment=mixed", "--difficulty=0.8", "--seed=7", "--experiment_name=chosen",
                    ]
                    fake_gym = types.SimpleNamespace(spec=lambda task: spec)
                    with patch.dict(sys.modules, {"gymnasium": fake_gym}):
                        remaining = CLI.configure_runner(args + ["--task=Example", "--num_envs=4"], training=training)
                    cfg = spec.kwargs["env_cfg_entry_point"]()
                    agent = spec.kwargs["rsl_rl_cfg_entry_point"]()
                    self.assertEqual((cfg.environment_preset, cfg.environment_difficulty, cfg.seed), ("mixed", 0.8, 7))
                    self.assertEqual((agent.experiment_name, agent.seed), ("chosen_obstacles", 7))
                    self.assertIn("--task=Example", remaining)
                    self.assertIn("--num_envs=4", remaining)
                    # The lazy factories above own these values for both paths.
                    # Re-injecting them makes Isaac Lab 2.1 reject the training
                    # command before the registered environment config exists.
                    self.assertNotIn("--seed=7", remaining)
                    self.assertFalse(any(arg.startswith("env.") for arg in remaining))

    def test_config_is_lazy_and_unselected_tasks_are_untouched(self):
        selected, untouched = make_spec(), make_spec()
        original_env = selected.kwargs["env_cfg_entry_point"]
        original_agent = selected.kwargs["rsl_rl_cfg_entry_point"]
        untouched_entries = untouched.kwargs.copy()
        fake_gym = types.SimpleNamespace(spec={"Selected": selected, "Other": untouched}.__getitem__)
        with patch.dict(sys.modules, {"gymnasium": fake_gym}):
            CLI.configure_runner(["--task=Selected", "--environment=narrow"], training=True)
        original_env.assert_not_called()
        original_agent.assert_not_called()
        self.assertEqual(untouched.kwargs, untouched_entries)
        self.assertEqual(selected.kwargs["env_cfg_entry_point"]().environment_preset, "narrow")

    def test_default_flat_preserves_experiment_and_seed(self):
        spec = make_spec()
        with patch.dict(sys.modules, {"gymnasium": types.SimpleNamespace(spec=lambda task: spec)}):
            CLI.configure_runner([], training=True)
        self.assertEqual(spec.kwargs["rsl_rl_cfg_entry_point"]().experiment_name, "robot")
        self.assertEqual(spec.kwargs["env_cfg_entry_point"]().seed, 42)

    def test_obstacle_experiment_suffix_is_applied_once(self):
        self.assertEqual(CLI.experiment_for_environment("robot", "narrow"), "robot_obstacles")
        self.assertEqual(CLI.experiment_for_environment("robot_obstacles", "mixed"), "robot_obstacles")
        self.assertEqual(CLI.experiment_for_environment("robot", "flat"), "robot")

    def test_nonflat_rejects_published_flat_checkpoint(self):
        fake_gym = types.SimpleNamespace(spec=Mock())
        with patch.dict(sys.modules, {"gymnasium": fake_gym}), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                CLI.configure_runner(["--environment=mixed", "--use_pretrained_checkpoint"], training=False)
        fake_gym.spec.assert_not_called()


class EvaluationCheckpointTests(unittest.TestCase):
    def setUp(self):
        # Import only the resolver: importing evaluate.py itself starts AppLauncher.
        path = SCRIPTS / "evaluate.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_checkpoint")
        self.args = types.SimpleNamespace(
            checkpoint="", task="Isaac-DKSH-Spider-Navigation-Direct-v0", environment="flat"
        )
        namespace = {"Path": Path, "__file__": str(path), "args_cli": self.args}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        self.resolve = namespace["_resolve_checkpoint"]
        self.cfg = types.SimpleNamespace(experiment_name="dksh_spider_navigation")

    def test_only_baseline_flat_may_use_bundled_checkpoint(self):
        with patch.object(Path, "glob", return_value=[]), patch.object(Path, "is_file", return_value=True):
            self.assertEqual(self.resolve(self.cfg).name, "balance_baseline.pt")
            for preset in CLI.ENVIRONMENTS[1:]:
                with self.subTest(preset=preset):
                    self.args.environment = preset
                    self.cfg.experiment_name = "dksh_spider_navigation_obstacles"
                    with self.assertRaisesRegex(FileNotFoundError, "_obstacles"):
                        self.resolve(self.cfg)
            self.args.environment = "flat"
            self.args.task = "Isaac-DKSH-Spider-CAD6-Navigation-Direct-v0"
            with self.assertRaises(FileNotFoundError):
                self.resolve(self.cfg)

    def test_newest_checkpoint_is_searched_in_selected_experiment_only(self):
        older, newer = Mock(spec=Path), Mock(spec=Path)
        older.stat.return_value.st_mtime = 10
        newer.stat.return_value.st_mtime = 20
        self.cfg.experiment_name = "dksh_spider_navigation_obstacles"
        self.args.environment = "mixed"
        with patch.object(Path, "glob", autospec=True, return_value=[older, newer]) as glob:
            self.assertIs(self.resolve(self.cfg), newer)
        self.assertEqual(glob.call_args.args[0], Path.cwd() / "logs" / "rsl_rl" / self.cfg.experiment_name)

    def test_explicit_checkpoint_must_exist(self):
        self.args.checkpoint = "selected_model.pt"
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                self.resolve(self.cfg)
        with patch.object(Path, "is_file", return_value=True):
            self.assertEqual(self.resolve(self.cfg), Path("selected_model.pt").resolve())


if __name__ == "__main__":
    unittest.main()
