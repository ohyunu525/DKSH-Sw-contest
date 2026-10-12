"""Conservative joint-space wave references for the September Onshape export.

The exported zero pose is the assembled stance: femur down and tibia outward.
Unlike the legacy CAD6 model, this module intentionally works in the normalized
Onshape joint coordinates instead of reusing legacy link-length IK constants.
"""

from __future__ import annotations

import math

import torch


ROOT_HEIGHT_M = 0.1275
STANCE_RADIUS_M = 0.285
FOOT_VERTICAL_GAIN_M_PER_RAD = 0.2745
# CAD axes are measured at the assembled zero.  The following conservative
# Jacobian gains convert a small Cartesian reference into position targets;
# they are deliberately below the +/- 90 degree hard limits.
# Zero-pose CAD lever arms: hip to foot is about 0.1796 m and the radial
# derivative of counter-rotated femur/tibia is about 0.1000 m per radian.
# Convert body-frame travel once, rather than multiplying it by an arbitrary 4.
HIP_TRAVEL_GAIN = STANCE_RADIUS_M / 0.1796
LEG_TRAVEL_GAIN = FOOT_VERTICAL_GAIN_M_PER_RAD / 0.1000
SWING_JOINT_GAIN = 8.0
WAVE_OFFSETS = (0, 3, 1, 4, 2, 5)
TRIPOD_OFFSETS = (0, 3, 0, 3, 0, 3)


def _phase_terms(phase: torch.Tensor, offsets: tuple[int, ...], duty: float) -> tuple[torch.Tensor, torch.Tensor]:
    offset = torch.tensor(offsets, device=phase.device, dtype=phase.dtype) / 6
    cycle = (phase[:, None] + offset).remainder(1)
    swing = ((cycle - duty) / (1 - duty)).clamp(0, 1)
    travel = torch.where(cycle < duty, 0.5 - cycle / duty, -0.5 + swing.square() * (3 - 2 * swing))
    return swing, travel


def joint_positions_for_twist(
    phase: torch.Tensor,
    command: torch.Tensor,
    period: float,
    lift: float,
    stride_limit: float,
    *,
    offsets: tuple[int, ...] = WAVE_OFFSETS,
    duty: float = 5 / 6,
) -> torch.Tensor:
    """Return normalized ``[hip, femur, tibia]`` targets for each Onshape leg.

    Hip displacement is the tangential projection of the desired foot travel.
    Positive femur and tibia offsets raise the exported foot point. They do
    not guarantee that all attached tibia collision shapes leave the floor.
    """

    if command.ndim != 2 or command.shape[1] != 3:
        raise ValueError("command must have shape [num_envs, 3]")
    angles = math.pi / 6 + torch.arange(6, device=phase.device, dtype=phase.dtype) * (math.pi / 3)
    radial = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    tangent = torch.stack((-torch.sin(angles), torch.cos(angles)), dim=-1)
    rest = STANCE_RADIUS_M * radial
    swing, travel = _phase_terms(phase, offsets, duty)
    vx, vy, yaw_rate = command.unbind(-1)
    foot_velocity = torch.stack(
        (vx[:, None] - yaw_rate[:, None] * rest[:, 1], vy[:, None] + yaw_rate[:, None] * rest[:, 0]),
        dim=-1,
    )
    displacement = foot_velocity * (period * duty)
    scale = (stride_limit / displacement.norm(dim=-1).clamp_min(1e-12)).clamp(max=1.0)
    displacement = displacement * scale[..., None]
    # A planted foot travels opposite to the desired body command.  Hip yaw
    # supplies tangential travel; femur/tibia supply radial travel.  The
    # opposite tibia contribution avoids treating the two serial pivots as a
    # single virtual slider.
    # travel decreases during stance. CAD-positive hip rotates CCW, so a
    # positive body tangential command needs a decreasing hip angle to hold
    # the planted world foot fixed. Do not negate that decreasing reference.
    hip = (displacement * tangent).sum(dim=-1) * travel / STANCE_RADIUS_M * HIP_TRAVEL_GAIN
    # The radial reference uses the same decreasing stance travel. Keeping
    # an extra minus here would make translation fight the corrected hips.
    radial_travel = (displacement * radial).sum(dim=-1) * travel / FOOT_VERTICAL_GAIN_M_PER_RAD
    lift_target = lift * torch.sin(math.pi * swing).square() / FOOT_VERTICAL_GAIN_M_PER_RAD
    femur = radial_travel * LEG_TRAVEL_GAIN + lift_target * SWING_JOINT_GAIN
    tibia = -radial_travel * LEG_TRAVEL_GAIN + lift_target * SWING_JOINT_GAIN
    return torch.stack((hip, femur, tibia), dim=-1)


def wave_joint_positions(
    phase: torch.Tensor, speed: torch.Tensor, period: float, lift: float, stride_limit: float, *,
    offsets: tuple[int, ...] = WAVE_OFFSETS, duty: float = 5 / 6,
) -> torch.Tensor:
    command = torch.stack((speed, torch.zeros_like(speed), torch.zeros_like(speed)), dim=-1)
    return joint_positions_for_twist(phase, command, period, lift, stride_limit, offsets=offsets, duty=duty)


def blended_wave_joint_positions(
    phase: torch.Tensor, speed: torch.Tensor, period: float, lift: float, stride_limit: float, tripod_weight: torch.Tensor,
) -> torch.Tensor:
    wave = wave_joint_positions(phase, speed, period, lift, stride_limit)
    tripod = wave_joint_positions(phase, speed, period, lift, stride_limit, offsets=TRIPOD_OFFSETS, duty=0.55)
    return torch.lerp(wave, tripod, tripod_weight[:, None, None])


def blended_directional_joint_positions(
    phase: torch.Tensor, command: torch.Tensor, period: float, lift: float, stride_limit: float, tripod_weight: torch.Tensor,
) -> torch.Tensor:
    wave = joint_positions_for_twist(phase, command, period, lift, stride_limit)
    tripod = joint_positions_for_twist(phase, command, period, lift, stride_limit, offsets=TRIPOD_OFFSETS, duty=0.55)
    return torch.lerp(wave, tripod, tripod_weight[:, None, None])
