"""Evaluation failures and actual optimizer-bound exploration without Isaac Sim."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

SPEC = importlib.util.spec_from_file_location("joint_angle_runner_utils", Path(__file__).resolve().parents[1] / "scripts/joint_angle_runner_utils.py")
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)


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
        constraint = UTILS.PolicyStdConstraint(policy, optimizer, 0.021, initialize=True)
        self.assertTrue(torch.allclose(parameter, torch.full((18,), 0.021)))
        optimizer.zero_grad()
        (-parameter.sum()).backward()  # The unconstrained step would increase std to 1.021.
        optimizer.step()
        self.assertTrue((parameter <= 0.021).all())
        optimizer.zero_grad()
        parameter.sum().backward()  # The unconstrained step would make std negative.
        optimizer.step()
        self.assertTrue((parameter > 0).all())
        constraint.close()

    def test_log_std_is_supported_and_resume_does_not_reset_valid_std(self):
        policy, parameter = self.make_policy(logarithmic=True, value=0.01)
        optimizer = torch.optim.Adam(policy.parameters(), lr=5.0)
        constraint = UTILS.PolicyStdConstraint(policy, optimizer, 0.021, initialize=False)
        self.assertAlmostEqual(constraint.values()[0], 0.01, places=6)
        optimizer.zero_grad()
        (-parameter.sum()).backward()
        optimizer.step()
        self.assertLessEqual(max(constraint.values()), 0.021001)
        # The same parameter governs both sampling and log-probability evaluation.
        distribution = torch.distributions.Normal(torch.zeros(18), parameter.exp())
        sample = distribution.sample()
        self.assertTrue(torch.isfinite(distribution.log_prob(sample)).all())
        constraint.close()

    def test_nonfinite_optimizer_result_fails_instead_of_saving_a_broken_policy(self):
        policy, parameter = self.make_policy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=1)
        constraint = UTILS.PolicyStdConstraint(policy, optimizer, 0.021, initialize=True)
        parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            optimizer.step()
        constraint.close()


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


if __name__ == "__main__":
    unittest.main()
