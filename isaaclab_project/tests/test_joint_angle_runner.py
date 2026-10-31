"""Evaluation failures and actual optimizer-bound exploration without Isaac Sim."""

import importlib.util
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

SPEC = importlib.util.spec_from_file_location("joint_angle_runner_utils", Path(__file__).resolve().parents[1] / "scripts/joint_angle_runner_utils.py")
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)
CHECKPOINT_SPEC = importlib.util.spec_from_file_location("joint_angle_checkpoint", Path(__file__).resolve().parents[1] / "scripts/joint_angle_checkpoint.py")
CHECKPOINT = importlib.util.module_from_spec(CHECKPOINT_SPEC)
CHECKPOINT_SPEC.loader.exec_module(CHECKPOINT)


class StdConstraintTests(unittest.TestCase):
    def make_policy(self, logarithmic=False, value=0.12):
        policy = torch.nn.Module()
        name = "log_std" if logarithmic else "std"
        parameter = torch.nn.Parameter(torch.full((18,), math.log(value) if logarithmic else value))
        policy.register_parameter(name, parameter)
        return policy, parameter

    def test_direct_std_is_projected_after_real_optimizer_updates(self):
        policy, parameter = self.make_policy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
        constraint = UTILS.PolicyStdConstraint(policy, optimizer, initial=0.08, minimum=0.03, maximum=0.30, initialize=True)
        self.assertTrue(torch.allclose(parameter, torch.full((18,), 0.08)))
        optimizer.zero_grad()
        (-parameter.sum()).backward()
        optimizer.step()
        self.assertTrue((parameter <= 0.30).all())
        optimizer.zero_grad()
        parameter.sum().backward()  # The unconstrained step would make std negative.
        optimizer.step()
        self.assertTrue(torch.allclose(parameter, torch.full((18,), 0.03)))
        constraint.close()

    def test_log_std_is_supported_and_resume_does_not_reset_valid_std(self):
        policy, parameter = self.make_policy(logarithmic=True, value=0.05)
        optimizer = torch.optim.Adam(policy.parameters(), lr=5.0)
        constraint = UTILS.PolicyStdConstraint(policy, optimizer, initial=0.08, minimum=0.03, maximum=0.30)
        self.assertAlmostEqual(constraint.values()[0], 0.05, places=6)
        optimizer.zero_grad()
        (-parameter.sum()).backward()
        optimizer.step()
        self.assertLessEqual(max(constraint.values()), 0.300001)
        # The same parameter governs both sampling and log-probability evaluation.
        distribution = torch.distributions.Normal(torch.zeros(18), parameter.exp())
        sample = distribution.sample()
        self.assertTrue(torch.isfinite(distribution.log_prob(sample)).all())
        constraint.close()

    def test_nonfinite_optimizer_result_fails_instead_of_saving_a_broken_policy(self):
        policy, parameter = self.make_policy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=1)
        constraint = UTILS.PolicyStdConstraint(policy, optimizer, initial=0.08, minimum=0.03, maximum=0.30, initialize=True)
        parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            optimizer.step()
        constraint.close()

    def test_explicit_resume_reset_clears_only_std_adam_state(self):
        for logarithmic in (False, True):
            with self.subTest(logarithmic=logarithmic):
                policy, parameter = self.make_policy(logarithmic=logarithmic, value=0.01)
                policy.register_parameter("weight", torch.nn.Parameter(torch.tensor([2.0])))
                optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
                (parameter.sum() + policy.weight.sum()).backward()
                optimizer.step()
                weight_before = policy.weight.detach().clone()
                weight_state = optimizer.state[policy.weight]
                weight_step = weight_state["step"].clone()
                self.assertIn(parameter, optimizer.state)
                constraint = UTILS.PolicyStdConstraint(
                    policy, optimizer, initial=0.08, minimum=0.03, maximum=0.30,
                    initialize=True, reset_optimizer_state=True,
                )
                self.assertTrue(constraint.optimizer_state_cleared)
                self.assertNotIn(parameter, optimizer.state)
                self.assertIs(optimizer.state[policy.weight], weight_state)
                torch.testing.assert_close(weight_state["step"], weight_step)
                torch.testing.assert_close(policy.weight, weight_before)
                self.assertAlmostEqual(constraint.values()[0], 0.08, places=6)
                constraint.close()

    def test_default_resume_clamps_collapsed_std_but_preserves_adam_state(self):
        policy, parameter = self.make_policy(value=0.01)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        parameter.sum().backward()
        optimizer.step()
        state = optimizer.state[parameter]
        before = parameter.detach().clone()
        constraint = UTILS.PolicyStdConstraint(policy, optimizer, initial=0.08, minimum=0.03, maximum=0.30)
        self.assertIs(optimizer.state[parameter], state)
        self.assertFalse(constraint.optimizer_state_cleared)
        self.assertAlmostEqual(constraint.before_values[0], before[0].item(), places=7)
        self.assertAlmostEqual(constraint.after_values[0], 0.03, places=7)
        constraint.close()

    def test_invalid_bounds_are_rejected_before_training(self):
        for bounds in ((0.08, 0, 0.3), (float("nan"), 0.03, 0.3), (0.01, 0.03, 0.3),
                       (0.08, 0.03, 0.04), (0.08, 0.03, float("inf"))):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                UTILS.validate_exploration_bounds(*bounds)


class EvaluationTests(unittest.TestCase):
    def metrics(self, count=1, progress=0.02, goal=0, fallen=0, out_of_bounds=0):
        result = {key: torch.zeros(count) for key in UTILS.EvaluationAccumulator.METRICS}
        result.update(progress_m=torch.full((count,), progress), travel_distance_m=torch.full((count,), abs(progress)),
                      forward_speed_mps=torch.full((count,), progress), mechanical_power_w=torch.ones(count),
                      goal=torch.full((count,), goal), fallen=torch.full((count,), fallen),
                      out_of_bounds=torch.full((count,), out_of_bounds))
        return result

    def report(self, accumulator):
        return accumulator.report(target_speed=0.02, minimum_speed_fraction=0.5, dt=1)

    def test_incomplete_goal_flags_never_count_as_completed_success(self):
        accumulator = UTILS.EvaluationAccumulator(1)
        accumulator.update(self.metrics(goal=1), torch.tensor([False]), 1)
        report = self.report(accumulator)
        self.assertEqual(report["episodes"]["goal"], 0)
        self.assertEqual(report["incomplete_episode_count"], 1)
        self.assertFalse(report["locomotion_pass"])

    def test_slow_goal_reaching_is_rejected_even_at_100_percent_success(self):
        accumulator = UTILS.EvaluationAccumulator(1)
        for index in range(100):
            accumulator.update(self.metrics(progress=0.0001, goal=int(index == 99)), torch.tensor([index == 99]), 1)
        report = self.report(accumulator)
        self.assertEqual(report["success_rate"], 1)
        self.assertLess(report["sustained_progress_speed_mps"], 0.01)
        self.assertFalse(report["locomotion_pass"])

    def test_rocking_signed_progress_is_not_replaced_by_travel_distance(self):
        accumulator = UTILS.EvaluationAccumulator(1)
        accumulator.update(self.metrics(progress=0.2), torch.tensor([False]), 1)
        accumulator.update(self.metrics(progress=-0.2, goal=1), torch.tensor([True]), 1)
        report = self.report(accumulator)
        self.assertAlmostEqual(report["travel_distance_m"], 0.4, places=6)
        self.assertAlmostEqual(report["completed_progress_m"], 0, places=6)
        self.assertFalse(report["locomotion_pass"])

    def test_all_environments_must_complete_successes_and_failures_take_precedence(self):
        accumulator = UTILS.EvaluationAccumulator(2)
        accumulator.update(self.metrics(count=2, goal=1), torch.tensor([True, False]), 1)
        self.assertFalse(self.report(accumulator)["locomotion_pass"])
        metrics = self.metrics(count=2, goal=1)
        metrics["fallen"][1] = 1
        accumulator.update(metrics, torch.tensor([False, True]), 1)
        report = self.report(accumulator)
        self.assertEqual(report["episodes"]["goal"], 1)
        self.assertEqual(report["episodes"]["fallen"], 1)
        self.assertFalse(report["locomotion_pass"])

    def test_valid_sustained_completed_episodes_can_pass_after_independent_resets(self):
        accumulator = UTILS.EvaluationAccumulator(2)
        for repeat in range(2):
            for index in range(50):
                accumulator.update(self.metrics(count=2, goal=int(index == 49)), torch.tensor([index == 49] * 2), 1)
        report = self.report(accumulator)
        self.assertEqual(report["episodes"]["goal"], 4)
        self.assertTrue(report["locomotion_pass"])
        self.assertEqual(report["incomplete_episode_count"], 0)

    def test_femur_contact_during_travel_is_rejected_after_clean_goal_stance(self):
        accumulator = UTILS.EvaluationAccumulator(1)
        for index in range(100):
            metrics = self.metrics(progress=.004, goal=int(index == 99))
            metrics["body_contact"][:] = int(index < 90)
            metrics["non_distal_support_fraction"] = torch.zeros(1)
            accumulator.update(metrics, torch.tensor([index == 99]), .02)
        report = accumulator.report(target_speed=.2, minimum_speed_fraction=.5, dt=.02)
        self.assertAlmostEqual(report["body_contact_fraction"], .9)
        self.assertEqual(report["success_rate"], 1)
        self.assertFalse(report["locomotion_pass"])
        self.assertTrue(any("body_contact_fraction" in reason for reason in report["failure_reasons"]))

    def test_clean_episodes_cannot_dilute_one_environments_bad_contact(self):
        accumulator = UTILS.EvaluationAccumulator(2)
        metrics = self.metrics(count=2, goal=1)
        metrics["non_distal_support_fraction"] = torch.tensor([0., .1])
        accumulator.update(metrics, torch.ones(2, dtype=torch.bool), 1)
        for _ in range(20):
            accumulator.update(self.metrics(count=2, goal=1), torch.ones(2, dtype=torch.bool), 1)
        report = accumulator.report(target_speed=.02, minimum_speed_fraction=.5, dt=1,
                                    max_non_distal_support_fraction=.05)
        self.assertFalse(report["locomotion_pass"])
        self.assertTrue(any("non_distal_support_fraction" in reason for reason in report["failure_reasons"]))

    def test_strict_contact_requires_complete_per_episode_diagnostics(self):
        for missing_name in ("non_distal_support_fraction", "non_distal_collision_fraction"):
            with self.subTest(missing_name=missing_name):
                accumulator = UTILS.EvaluationAccumulator(1)
                for index in range(2):
                    metrics = self.metrics(goal=index)
                    metrics["non_distal_support_fraction"] = torch.zeros(1)
                    metrics["non_distal_collision_fraction"] = torch.zeros(1)
                    if index == 0:
                        del metrics[missing_name]
                    accumulator.update(metrics, torch.tensor([index == 1]), 1)
                report = accumulator.report(target_speed=.02, minimum_speed_fraction=.5, dt=1,
                                            require_strict_contact=True)
                self.assertFalse(report["locomotion_pass"])
                self.assertTrue(any(missing_name in reason for reason in report["failure_reasons"]))
        accumulator = UTILS.EvaluationAccumulator(1)
        metrics = self.metrics(goal=1)
        metrics.update(non_distal_support_fraction=torch.zeros(1), non_distal_collision_fraction=torch.zeros(1))
        accumulator.update(metrics, torch.tensor([True]), 1)
        self.assertTrue(accumulator.report(target_speed=.02, minimum_speed_fraction=.5, dt=1,
                                           require_strict_contact=True)["locomotion_pass"])

    def test_collision_geometry_is_rejected_even_with_zero_reported_link_forces(self):
        accumulator = UTILS.EvaluationAccumulator(1)
        metrics = self.metrics(goal=1)
        metrics.update(non_distal_support_fraction=torch.zeros(1), non_distal_collision_fraction=torch.ones(1))
        accumulator.update(metrics, torch.tensor([True]), 1)
        report = self.report(accumulator)
        self.assertFalse(report["locomotion_pass"])
        self.assertEqual(report["non_distal_collision_fraction"], 1)

    def test_pending_bad_contacts_cannot_be_hidden_by_earlier_clean_success(self):
        accumulator = UTILS.EvaluationAccumulator(1)
        accumulator.update(self.metrics(goal=1), torch.tensor([True]), 1)
        metrics = self.metrics()
        metrics["body_contact"][:] = 1
        accumulator.update(metrics, torch.tensor([False]), 1)
        self.assertFalse(self.report(accumulator)["locomotion_pass"])

    def test_strict_pending_episode_must_also_have_complete_contact_evidence(self):
        accumulator = UTILS.EvaluationAccumulator(1)
        metrics = self.metrics(goal=1)
        metrics.update(non_distal_support_fraction=torch.zeros(1), non_distal_collision_fraction=torch.zeros(1))
        accumulator.update(metrics, torch.tensor([True]), 1)
        accumulator.update(self.metrics(), torch.tensor([False]), 1)
        self.assertFalse(accumulator.report(target_speed=.02, minimum_speed_fraction=.5, dt=1,
                                            require_strict_contact=True)["locomotion_pass"])

    def test_one_fast_environment_cannot_hide_another_environment_that_never_completes(self):
        accumulator = UTILS.EvaluationAccumulator(2)
        for _ in range(2):
            accumulator.update(self.metrics(count=2, goal=1), torch.tensor([True, False]), 1)
        report = self.report(accumulator)
        self.assertEqual(report["episodes"]["goal"], 2)
        self.assertEqual(report["successful_environment_count"], 1)
        self.assertFalse(report["locomotion_pass"])

    def test_evaluation_distance_seed_and_duration_do_not_inherit_short_training_goals(self):
        self.assertEqual(UTILS.effective_seed("train", None), 42)
        self.assertEqual(UTILS.effective_seed("evaluate", None), 73)
        self.assertEqual(UTILS.effective_seed("evaluate", 99), 99)
        cfg = NS(evaluation_goal_distance=1.0, evaluation_episode_length_s=90, goal_min_distance=0.08,
                 goal_max_distance=0.25, goal_radius=0.04, episode_length_s=180)
        UTILS.configure_evaluation(cfg, "flat")
        self.assertEqual((cfg.goal_min_distance, cfg.goal_max_distance, cfg.goal_radius), (1.0, 1.0, 0.08))
        self.assertEqual(cfg.episode_length_s, 90)
        UTILS.configure_evaluation(cfg, "stairs")
        self.assertGreaterEqual(cfg.episode_length_s, 240)

    def test_smoke_accepts_actor_only_v2_and_checks_legacy_critic_when_requested(self):
        actor = torch.zeros((4, 140))
        UTILS.validate_observation_batch({"policy": actor}, num_envs=4, actor_size=140, critic_size=0)
        critic = torch.zeros((4, 156))
        UTILS.validate_observation_batch({"policy": actor, "critic": critic}, num_envs=4, actor_size=140, critic_size=156)
        critic[0, 0] = 1
        with self.assertRaises(AssertionError):
            UTILS.validate_observation_batch({"policy": actor, "critic": critic}, num_envs=4, actor_size=140, critic_size=156)

    def test_optional_motion_diagnostics_are_reported_without_changing_acceptance(self):
        accumulator = UTILS.EvaluationAccumulator(1)
        metrics = self.metrics(goal=1)
        metrics["target_excursion_rad"] = torch.tensor([0.12])
        accumulator.update(metrics, torch.tensor([True]), 1)
        report = self.report(accumulator)
        self.assertAlmostEqual(report["target_excursion_rad"], 0.12, places=6)
        self.assertTrue(report["locomotion_pass"])

    def test_report_preserves_actual_minimum_collision_clearance_across_steps_and_environments(self):
        accumulator = UTILS.EvaluationAccumulator(2)
        first = self.metrics(count=2)
        first.update(min_non_distal_clearance_m=torch.tensor([.04, .02]),
                     strict_contact_failure=torch.tensor([0., 0.]))
        accumulator.update(first, torch.zeros(2, dtype=torch.bool), 1)
        last = self.metrics(count=2, goal=1)
        last.update(min_non_distal_clearance_m=torch.tensor([-.003, .08]),
                    strict_contact_failure=torch.tensor([1., 0.]))
        accumulator.update(last, torch.ones(2, dtype=torch.bool), 1)
        report = self.report(accumulator)
        self.assertAlmostEqual(report["min_non_distal_clearance_m"], -.003)
        self.assertEqual(report["strict_contact_failure_fraction"], .25)
        self.assertNotIn("strict_contact_failure", report)
        self.assertFalse(report["locomotion_pass"])

    def test_diagnostics_remain_averaged_but_non_distal_contacts_block_acceptance(self):
        accumulator = UTILS.EvaluationAccumulator(2)
        metrics = self.metrics(count=2, goal=1)
        diagnostics = {
            "distal_support_fraction": [0.5, 1.0],
            "non_distal_support_fraction": [0.0, 0.5],
            "mean_knee_ground_clearance_m": [0.02, 0.04],
            "knee_ground_clearance_violation_fraction": [0.5, 0.0],
        }
        for name, values in diagnostics.items():
            metrics[name] = torch.tensor(values)
        accumulator.update(metrics, torch.tensor([True, True]), 1)
        # An old profile may omit every new optional metric.
        accumulator.update(self.metrics(count=2, goal=1), torch.tensor([True, True]), 1)
        report = self.report(accumulator)
        for name, values in diagnostics.items():
            self.assertAlmostEqual(report[name], sum(values) / len(values), places=6)
        self.assertFalse(report["locomotion_pass"])


class ResumeIntegrationTests(unittest.TestCase):
    def test_real_resume_records_clamp_or_std_only_reset_and_preserves_weights_and_normalizer(self):
        source = Path(__file__).resolve().parents[1] / "scripts/joint_angle_runner.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                     and node.name in ("train", "write_json", "toe_button_sensor_metadata")], type_ignores=[])
        for reset in (False, True):
            with self.subTest(reset=reset), TemporaryDirectory() as directory:
                root = Path(directory)
                policy = torch.nn.Module()
                policy.register_parameter("std", torch.nn.Parameter(torch.full((18,), 0.01)))
                policy.register_parameter("weight", torch.nn.Parameter(torch.tensor([2.0])))
                policy.register_buffer("normalizer_mean", torch.tensor([3.0]))
                optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
                (policy.std.sum() + policy.weight.sum()).backward()
                optimizer.step()
                with torch.no_grad():
                    policy.std.fill_(0.01)
                checkpoint = root / "model_350.pt"
                torch.save({"model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                            "iter": 350}, checkpoint)
                weight_before = policy.weight.detach().clone()
                normalizer_before = policy.normalizer_mean.clone()
                weight_state_before = {key: value.clone() for key, value in optimizer.state[policy.weight].items()}
                base = NS(num_envs=1, step_dt=0.02, completed={"goal": 0}, _joint_ids=list(range(18)),
                          _robot=NS(joint_names=[f"joint_{index}" for index in range(18)]),
                          _angle_limits=torch.tensor([[[-1.57, 1.57]] * 18]), _servo_velocity_limits=torch.ones((1, 18)))
                cfg = NS(exploration_reference_saturation_probability=0.01, exploration_initial_std=0.08,
                         exploration_min_std=0.03, exploration_max_std=0.30, behavior_profile="legacy_v2",
                         target_rate_limit=1.5, target_smoothing=0.35, seed=42, action_space=18, observation_space=140,
                         target_speed=0.20, speed_tracking_sigma=0.015, progress_reward_scale=25.0, speed_cost_scale=-0.50,
                         distal_contact_height_tolerance=0.004, proximal_contact_height_tolerance=0.008,
                         knee_ground_clearance_m=0.03, knee_ground_clearance_cost_scale=-5.0,
                         non_distal_support_cost_scale=-15.0,
                         non_distal_ground_clearance_m=0.008, non_distal_contact_height_tolerance_m=0.001)
                args = NS(task="Isaac-DKSH-MG996R-Hip-CAD6-JointAngles-v2", environment="flat", difficulty=0.5,
                          max_iterations=1, reset_exploration_std=reset, allow_behavior_transfer=False)
                log = root / "resumed"
                log.mkdir()
                runner = NS(alg=NS(policy=policy, optimizer=optimizer), current_learning_iteration=0)

                def load(path):
                    saved = torch.load(path, weights_only=True)
                    policy.load_state_dict(saved["model_state_dict"])
                    optimizer.load_state_dict(saved["optimizer_state_dict"])
                    runner.current_learning_iteration = saved["iter"]

                def learn(**kwargs):
                    runner.current_learning_iteration += kwargs["num_learning_iterations"]
                    torch.save({"model_state_dict": policy.state_dict()}, log / f"model_{runner.current_learning_iteration}.pt")

                runner.load, runner.learn = load, learn
                namespace = {"Path": Path, "torch": torch, "json": json, "hashlib": hashlib, "args": args,
                             "ACTION_SEMANTICS": CHECKPOINT.ACTION_SEMANTICS, "SCHEMA_CONTRACTS": CHECKPOINT.SCHEMA_CONTRACTS,
                             "BEHAVIOR_PROFILES": CHECKPOINT.BEHAVIOR_PROFILES, "EXPLORATION_PROFILE": CHECKPOINT.EXPLORATION_PROFILE,
                             "behavior_profile_from_metadata": CHECKPOINT.behavior_profile_from_metadata,
                             "expected_cad_asset_sha256": CHECKPOINT.expected_cad_asset_sha256,
                             "cad_report": None, "dump_yaml": Mock(),
                             "PolicyStdConstraint": UTILS.PolicyStdConstraint,
                             "course_configuration": UTILS.course_configuration,
                             "CurriculumCheckpointState": UTILS.CurriculumCheckpointState}
                exec(compile(functions, str(source), "exec"), namespace)
                control = types.ModuleType("dksh_isaaclab.tasks.direct.spider_navigation.joint_angle_control")
                control.exploration_std = Mock(return_value=0.0211844)
                parent_metadata = CHECKPOINT.BEHAVIOR_PROFILES["legacy_v2"] | {"exploration_std_cap": 0.0211844}
                with patch.dict(sys.modules, {control.__name__: control}), contextlib.redirect_stdout(io.StringIO()):
                    namespace["train"](NS(unwrapped=base), runner, cfg, NS(), log, checkpoint, parent_metadata)
                metadata = json.loads((log / "run_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "completed")
                self.assertAlmostEqual(metadata["exploration_std_before"][0], 0.01, places=6)
                self.assertAlmostEqual(metadata["exploration_std_after"][0], 0.08 if reset else 0.03, places=6)
                self.assertEqual(metadata["exploration_std_reset_applied"], reset)
                self.assertEqual(metadata["exploration_std_optimizer_state_cleared"], reset)
                self.assertTrue(metadata["exploration_bounds_changed"])
                self.assertFalse(metadata["behavior_transfer_applied"])
                self.assertEqual(metadata["reward_contract"], parent_metadata["reward_contract"])
                self.assertEqual(metadata["exploration_std_cap"], 0.30)
                self.assertEqual(metadata["cad_asset_sha256"], CHECKPOINT.expected_cad_asset_sha256(args.task))
                for name in ("distal_contact_height_tolerance", "proximal_contact_height_tolerance",
                             "knee_ground_clearance_m", "knee_ground_clearance_cost_scale",
                             "non_distal_support_cost_scale", "non_distal_ground_clearance_m",
                             "non_distal_contact_height_tolerance_m"):
                    self.assertEqual(metadata[name], getattr(cfg, name))
                self.assertEqual(metadata["learning_rate_before"], 0.001)
                self.assertEqual(metadata["learning_rate_after"], 0.001)
                self.assertEqual(metadata["learning_rate_final"], 0.001)
                self.assertEqual(policy.std in optimizer.state, not reset)
                torch.testing.assert_close(policy.weight, weight_before)
                torch.testing.assert_close(policy.normalizer_mean, normalizer_before)
                for key, expected in weight_state_before.items():
                    torch.testing.assert_close(optimizer.state[policy.weight][key], expected)


if __name__ == "__main__":
    unittest.main()
