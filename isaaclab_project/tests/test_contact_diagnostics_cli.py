"""Contact diagnostic configuration without simulator imports or file writes."""

import argparse
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contact_diagnostics_cli_under_test", PROJECT / "scripts/contact_diagnostics_cli.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class ContactDiagnosticsArgumentsTests(unittest.TestCase):
    def arguments(self, *arguments, mode="check", task="Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2"):
        parser = argparse.ArgumentParser()
        CLI.add_contact_diagnostics_args(parser)
        args = parser.parse_args(arguments)
        args.mode, args.task = mode, task
        return args

    def test_training_always_configures_bounded_basic_snapshots_inside_new_run(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "logs/new run"
            cfg = NS()
            path = CLI.apply_contact_diagnostics_cfg(cfg, self.arguments(mode="train"), project_root=root, run_dir=run)
            self.assertEqual(path, run / "contact_failures.jsonl")
            self.assertEqual(cfg.contact_diagnostics_path, str(path))
            self.assertFalse(cfg.contact_diagnostics_pairs)
            self.assertEqual((cfg.contact_diagnostics_max_records, cfg.contact_diagnostics_per_step), (64, 4))
            self.assertFalse(run.exists())  # Writer creates files only after an actual diagnostic event.

    def test_check_evaluate_and_play_do_not_add_sensors_or_files_by_default(self):
        with TemporaryDirectory() as folder:
            for mode in ("check", "evaluate", "play"):
                cfg = NS(contact_diagnostics_path="old", contact_diagnostics_pairs=True)
                result = CLI.apply_contact_diagnostics_cfg(cfg, self.arguments(mode=mode), project_root=folder)
                self.assertIsNone(result)
                self.assertIsNone(cfg.contact_diagnostics_path)
                self.assertFalse(cfg.contact_diagnostics_pairs)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_pair_capture_check_gets_a_unique_log_separate_from_existing_models(self):
        with TemporaryDirectory() as folder:
            paths = []
            for _ in range(2):
                cfg = NS()
                path = CLI.apply_contact_diagnostics_cfg(cfg, self.arguments("--contact_diagnostics"), project_root=folder)
                paths.append(path)
                self.assertTrue(cfg.contact_diagnostics_pairs)
                self.assertTrue(path.is_relative_to(Path(folder) / "logs/contact_diagnostics"))
                self.assertEqual(path.name, "contact_failures.jsonl")
                self.assertFalse(path.exists())
            self.assertNotEqual(paths[0], paths[1])

    def test_explicit_output_alone_enables_basic_snapshots_and_preserves_spaces(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "diagnostic log/failures.jsonl"
            cfg = NS()
            args = self.arguments(f"--contact_diagnostics_output={output}",
                                  "--contact_diagnostics_max_records=9", "--contact_diagnostics_per_step=2")
            path = CLI.apply_contact_diagnostics_cfg(cfg, args, project_root=folder)
            self.assertEqual(path, output)
            self.assertFalse(cfg.contact_diagnostics_pairs)
            self.assertEqual((cfg.contact_diagnostics_max_records, cfg.contact_diagnostics_per_step), (9, 2))
            self.assertFalse(path.parent.exists())

    def test_pair_capture_and_output_can_be_combined(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "pairs.jsonl"
            cfg = NS()
            path = CLI.apply_contact_diagnostics_cfg(cfg, self.arguments("--contact_diagnostics", f"--contact_diagnostics_output={output}"),
                                                     project_root=folder)
            self.assertEqual(path, output)
            self.assertTrue(cfg.contact_diagnostics_pairs)

    def test_nonpositive_and_noninteger_limits_are_rejected_by_parser(self):
        for option in ("--contact_diagnostics_max_records=0", "--contact_diagnostics_max_records=-1",
                       "--contact_diagnostics_per_step=0", "--contact_diagnostics_per_step=1.5"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.arguments(option)

    def test_explicit_diagnostics_reject_legacy_tasks_and_directory_outputs(self):
        args = self.arguments("--contact_diagnostics", task="Isaac-DKSH-Spider-Navigation-Direct-v0")
        with self.assertRaisesRegex(ValueError, "JointAngles-v2"):
            CLI.validate_contact_diagnostics_args(args)
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "not a directory"):
                CLI.apply_contact_diagnostics_cfg(NS(), self.arguments(f"--contact_diagnostics_output={folder}"), project_root=folder)
        with self.assertRaisesRegex(ValueError, "name a JSONL file"):
            CLI.validate_contact_diagnostics_args(self.arguments("--contact_diagnostics_output=  "))

    def test_runner_and_smoke_configure_diagnostics_before_creating_environment(self):
        for name in ("joint_angle_runner.py", "smoke_env.py"):
            tree = ast.parse((PROJECT / "scripts" / name).read_text(encoding="utf-8"))
            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
            configure = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Name) and node.func.id == "apply_contact_diagnostics_cfg")
            create = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                          and node.func.value.id == "gym" and node.func.attr == "make")
            self.assertLess(configure, create)


if __name__ == "__main__":
    unittest.main()
