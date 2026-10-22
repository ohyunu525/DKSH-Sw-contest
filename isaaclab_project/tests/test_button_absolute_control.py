"""Check digital switches and absolute commands without starting Isaac Sim."""
import ast
import importlib.util
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

TASK_DIR = Path(__file__).resolve().parents[1] / 'source/dksh_isaaclab/dksh_isaaclab/tasks/direct/spider_navigation'


def load(name):
    spec = importlib.util.spec_from_file_location(name, TASK_DIR / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load('joint_angle_control')
B = load('button_contact')


class ButtonAbsoluteTests(unittest.TestCase):
    def test_deployment_output_is_radians_and_includes_observation_normalization(self):
        actor = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            actor.weight.copy_(torch.eye(2))
        normalizer = torch.nn.BatchNorm1d(2)
        normalizer.running_mean.copy_(torch.tensor([1., 2.]))
        normalizer.running_var.fill_(1 - normalizer.eps)
        model = C.AbsoluteAnglePolicy(actor, normalizer, torch.tensor([[[-1., 1.], [-0.4, 1.4]]]))
        probe = torch.tensor([[1., 2.]])
        expected = torch.tensor([[0., 0.5]])
        torch.testing.assert_close(model(probe), expected)
        # Regression: ONNX restores the parent mode recursively. A training-mode
        # parent would turn BatchNorm/EmpiricalNormalization updates back on.
        if importlib.util.find_spec('onnx') is not None:
            torch.onnx.export(model, probe, io.BytesIO(), opset_version=11)
        torch.testing.assert_close(model(probe), expected)
        traced = torch.jit.trace(model, probe)
        torch.testing.assert_close(traced(probe), expected)

    def test_absolute_endpoints_asymmetric_limits_and_no_accumulation(self):
        limits = torch.tensor([[[-1., 1.], [-0.4, 1.4], [-1.4, 0.8]]])
        torch.testing.assert_close(C.action_to_angles(torch.tensor([[-1., 0., 1.]]), limits),
                                   torch.tensor([[-1., 0.5, 0.8]]))
        target = torch.zeros(1, 3)
        actions = torch.tensor([[0.3, 0., -0.2]])
        for _ in range(300):
            target = C.bounded_angle_targets(actions, target, limits, dt=0.02, rate_limit=1.5, smoothing=0.15)
        torch.testing.assert_close(target, C.action_to_angles(actions, limits))
        previous = target.clone()
        for _ in range(100):
            target = C.bounded_angle_targets(actions, target, limits, dt=0.02, rate_limit=1.5, smoothing=0.15)
        torch.testing.assert_close(target, previous)

    def test_rate_limit_and_bad_components_hold_previous(self):
        limits = torch.tensor([-math.pi / 2, math.pi / 2]).expand(1, 3, 2)
        previous = torch.tensor([[0.3, -0.4, 0.]])
        target = C.bounded_angle_targets(torch.tensor([[float('nan'), float('inf'), 1.]]),
                                         previous, limits, dt=0.02, rate_limit=1.5, smoothing=0.15)
        torch.testing.assert_close(target, torch.tensor([[0.3, -0.4, 0.03]]))

    def test_stair_step_uses_leg_order_without_any_gait_or_phase(self):
        tree = ast.parse((TASK_DIR / 'mg90s_stairs.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MG90SCad6StairsEnv')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_pre_physics_step')
        namespace = {'torch': torch, 'bounded_angle_targets': C.bounded_angle_targets,
                     'normalize_angles': C.normalize_angles}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<absolute-step>', 'exec'), namespace)
        order = torch.arange(18).flip(0)
        fake = SimpleNamespace(_previous_x=torch.zeros(1), _episode_origins=torch.zeros(1, 3),
                               _robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(1, 3))),
                               extras={}, _previous_actions=torch.zeros(1, 18), _actions=torch.zeros(1, 18),
                               joint_angle_targets_rad=torch.zeros(1, 18), _targets=torch.zeros(1, 18),
                               _angle_limits=torch.tensor([-1., 1.]).expand(1, 18, 2), _joint_ids=order,
                               step_dt=0.02, cfg=SimpleNamespace(target_rate_limit=1.5, target_smoothing=1.),
                               _filtered_actions=torch.zeros(1, 18), _age=torch.zeros(1), _lidar_age=torch.zeros(1))
        actions = torch.linspace(-1, 1, 18)[None]
        namespace['_pre_physics_step'](fake, actions)
        torch.testing.assert_close(fake._targets[:, order], actions.clamp(-0.03, 0.03))

    def test_buttons_debounce_hysteresis_leg_independence_and_reset(self):
        bank = B.ButtonBank((2, 6), 'cpu', 0.5, 0.25, 0.04)
        force = torch.zeros(2, 6)
        force[0, 2] = 0.6
        self.assertEqual(float(bank.update(force, 0.02).sum()), 0.)
        values = bank.update(force, 0.02)
        self.assertEqual(float(values.sum()), 1.)
        self.assertEqual(float(values[0, 2]), 1.)
        force[0, 2] = 0.3
        self.assertEqual(float(bank.update(force, 0.1)[0, 2]), 1.)
        self.assertEqual(float(bank.update(torch.zeros_like(force), 0.02)[0, 2]), 1.)
        self.assertEqual(float(bank.update(torch.zeros_like(force), 0.02)[0, 2]), 0.)
        bank.update(force.fill_(1), 0.1)
        bank.reset(torch.tensor([0]))
        self.assertFalse(bank.state[0].any())
        self.assertTrue(bank.state[1].all())

    def test_button_bounce_cannot_accumulate_nonconsecutive_pulses(self):
        bank = B.ButtonBank((1, 6), 'cpu', 0.5, 0.25, 0.04)
        for _ in range(10):
            bank.update(torch.ones(1, 6), 0.02)
            bank.update(torch.zeros(1, 6), 0.02)
        self.assertFalse(bank.state.any())


if __name__ == '__main__':
    unittest.main()
