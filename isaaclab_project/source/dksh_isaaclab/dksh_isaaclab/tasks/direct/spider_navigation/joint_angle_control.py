"""Absolute motor targets adapted from OYW/Develop/MLForLegs, d970d488.

Public actions and targets are leg-major: leg_0 hip/femur/tibia, then leg_1, ...
Angles are radians relative to the assembled CAD zero, never gait residuals.
"""
import copy
import math
import torch

ACTION_SEMANTICS = 'absolute_joint_angles_v1'


def joint_names(leg_count):
    return [f'leg_{leg}_{part}_joint' for leg in range(leg_count) for part in ('hip', 'femur', 'tibia')]


def normalize_angles(angles, limits):
    lower, upper = limits.unbind(-1)
    return 2 * (angles - lower) / (upper - lower).clamp_min(1e-6) - 1


def action_to_angles(actions, limits):
    """Map [-1, 1] to each joint's absolute working range in radians."""
    lower, upper = limits.unbind(-1)
    return lower + 0.5 * (actions.clamp(-1, 1) + 1) * (upper - lower)


def bounded_angle_targets(actions, previous_targets, limits, *, dt, rate_limit, smoothing):
    """A constant action converges to a constant angle; no reference or increment."""
    if not (math.isfinite(dt) and dt > 0 and math.isfinite(rate_limit) and rate_limit > 0):
        raise ValueError('dt and rate_limit must be finite and positive')
    if not math.isfinite(smoothing) or not 0 < smoothing <= 1:
        raise ValueError('smoothing must be in (0, 1]')
    safe = torch.where(torch.isfinite(actions), actions, normalize_angles(previous_targets, limits))
    desired = action_to_angles(safe, limits)
    delta = ((desired - previous_targets) * smoothing).clamp(-rate_limit * dt, rate_limit * dt)
    lower, upper = limits.unbind(-1)
    return (previous_targets + delta).clamp(lower, upper)


class AbsoluteAnglePolicy(torch.nn.Module):
    """Deployment model output is absolute radians; filtering remains in the controller."""

    def __init__(self, actor, normalizer, limits):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu().eval()
        self.normalizer = copy.deepcopy(normalizer).cpu().eval()
        self.register_buffer('limits', limits.detach().cpu().clone())
        # ONNX restores the parent's mode recursively; keep the whole wrapper in
        # eval mode so exporting cannot re-enable running-stat updates.
        self.eval()

    def forward(self, observations):
        return action_to_angles(self.actor(self.normalizer(observations)), self.limits)
