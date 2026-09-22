"""Six-leg CAD kinematics using the URDF joint origins and tibia collision sole.

Angles are relative to the assembled CAD pose. These differ from primitive links.
The sole target is the bottom centre of the tibia collision box at joint zero.
"""
import math
import torch

HIP_X, HIP_Z = 0.036987004, 0.001525013
FEMUR_X, FEMUR_Z = 0.029238866, -0.029238866
SOLE_X, SOLE_Z = 0.016120809, -0.036284219 - 0.088092423 / 2
L1, L2 = math.hypot(FEMUR_X, FEMUR_Z), math.hypot(SOLE_X, SOLE_Z)
ALPHA1, ALPHA2 = math.atan2(-FEMUR_Z, FEMUR_X), math.atan2(-SOLE_Z, SOLE_X)
STANCE_FEMUR, STANCE_TIBIA = 0.60, -0.20
STANCE_RADIUS = HIP_X + L1 * math.cos(ALPHA1 + STANCE_FEMUR) + L2 * math.cos(ALPHA2 + STANCE_FEMUR + STANCE_TIBIA)
STANCE_Z = HIP_Z - L1 * math.sin(ALPHA1 + STANCE_FEMUR) - L2 * math.sin(ALPHA2 + STANCE_FEMUR + STANCE_TIBIA)


WAVE_OFFSETS = (0, 3, 1, 4, 2, 5)
TRIPOD_OFFSETS = (0, 3, 0, 3, 0, 3)


def foot_targets(phase, speed, period=2.4, lift=0.004, stride_limit=0.012, *, offsets=WAVE_OFFSETS, duty=5 / 6):
    """Return reachable CAD6 targets for one gait pattern.

    ``offsets`` is expressed in sixths of a cycle so the default remains the
    original wave gait.  Tripod uses alternating (0, 2, 4) and (1, 3, 5)
    legs, with a shorter stance fraction for higher cadence locomotion.
    """
    angles = torch.arange(6, device=phase.device, dtype=phase.dtype) * (math.pi / 3)
    offsets = torch.tensor(offsets, device=phase.device, dtype=phase.dtype) / 6
    p = (phase[:, None] + offsets).remainder(1)
    swing = ((p - duty) / (1 - duty)).clamp(0, 1)
    travel = torch.where(p < duty, 0.5 - p / duty, -0.5 + swing.square() * (3 - 2 * swing))
    stride = (speed * period * duty).clamp(0, stride_limit)[:, None]
    x = STANCE_RADIUS * torch.cos(angles)[None, :] + stride * travel
    y = STANCE_RADIUS * torch.sin(angles)[None, :].expand_as(x)
    z = STANCE_Z + lift * torch.sin(math.pi * swing).square()
    return torch.stack((x, y, z), -1)


def blended_foot_targets(phase, speed, period, lift, stride_limit, tripod_weight):
    """Blend wave and tripod targets continuously during speed transitions."""
    wave = foot_targets(phase, speed, period, lift, stride_limit)
    tripod = foot_targets(
        phase, speed, period, lift, stride_limit, offsets=TRIPOD_OFFSETS, duty=0.55
    )
    return torch.lerp(wave, tripod, tripod_weight[:, None, None])


def feasible_velocity_command(command, period, stride_limit):
    """Scale the whole twist to fit every foot's wave-stance travel budget.

    Uniform scaling preserves the requested translation/rotation ratio. Wave
    duty is conservative for both endpoints of the wave/tripod blend.
    """
    angles = torch.arange(6, device=command.device, dtype=command.dtype) * (math.pi / 3)
    vx, vy, yaw = command.unbind(-1)
    foot_x = vx[:, None] + yaw[:, None] * STANCE_RADIUS * torch.sin(angles)
    foot_y = vy[:, None] - yaw[:, None] * STANCE_RADIUS * torch.cos(angles)
    travel = torch.stack((foot_x, foot_y), -1).norm(dim=-1).amax(dim=-1) * period * (5 / 6)
    scale = (stride_limit / travel.clamp_min(1e-12)).clamp(max=1.0)
    return command * scale[:, None]


def directional_foot_targets(
    phase, command, period=2.4, lift=0.004, stride_limit=0.012, *, offsets=WAVE_OFFSETS, duty=5 / 6
):
    """Return CAD6 targets for body-frame ``[vx, vy, yaw_rate]`` commands.

    During stance each foot moves opposite the commanded body twist. The
    per-foot travel is norm-limited so combined translation and rotation do
    not exceed the kinematic envelope verified for the forward gait.
    """
    if command.ndim != 2 or command.shape[1] != 3:
        raise ValueError("command must have shape [num_envs, 3]")
    angles = torch.arange(6, device=phase.device, dtype=phase.dtype) * (math.pi / 3)
    offsets = torch.tensor(offsets, device=phase.device, dtype=phase.dtype) / 6
    p = (phase[:, None] + offsets).remainder(1)
    swing = ((p - duty) / (1 - duty)).clamp(0, 1)
    travel = torch.where(p < duty, 0.5 - p / duty, -0.5 + swing.square() * (3 - 2 * swing))

    rest_x = STANCE_RADIUS * torch.cos(angles)[None, :]
    rest_y = STANCE_RADIUS * torch.sin(angles)[None, :]
    vx, vy, yaw_rate = command.unbind(-1)
    # The CAD hip axes use the opposite rotation sign to the ROS body yaw
    # convention; invert the rotational tangent so positive angular.z yields
    # positive measured root yaw in PhysX.
    foot_velocity_x = vx[:, None] + yaw_rate[:, None] * rest_y
    foot_velocity_y = vy[:, None] - yaw_rate[:, None] * rest_x
    displacement = torch.stack((foot_velocity_x, foot_velocity_y), dim=-1) * (period * duty)
    scale = (stride_limit / displacement.norm(dim=-1).clamp_min(1e-12)).clamp(max=1.0)
    displacement = displacement * scale[..., None]

    x = rest_x + displacement[..., 0] * travel
    y = rest_y + displacement[..., 1] * travel
    moving = (command.abs().amax(dim=-1) > 1e-9).to(phase.dtype)
    z = STANCE_Z + lift * moving[:, None] * torch.sin(math.pi * swing).square()
    return torch.stack((x, y, z), -1)


def blended_directional_foot_targets(phase, command, period, lift, stride_limit, tripod_weight):
    """Blend directional wave and tripod targets continuously."""
    wave = directional_foot_targets(phase, command, period, lift, stride_limit)
    tripod = directional_foot_targets(
        phase, command, period, lift, stride_limit, offsets=TRIPOD_OFFSETS, duty=0.55
    )
    return torch.lerp(wave, tripod, tripod_weight[:, None, None])


def inverse_kinematics(feet):
    angles = torch.arange(6, device=feet.device, dtype=feet.dtype) * (math.pi / 3)
    x, y, z = feet.unbind(-1)
    hip = (torch.atan2(y, x) - angles + math.pi).remainder(2 * math.pi) - math.pi
    reach = torch.sqrt(x.square() + y.square()) - HIP_X
    depth = HIP_Z - z
    cosine = (reach.square() + depth.square() - L1**2 - L2**2) / (2 * L1 * L2)
    bend = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
    femur = torch.atan2(depth, reach) - torch.atan2(L2 * torch.sin(bend), L1 + L2 * torch.cos(bend)) - ALPHA1
    tibia = bend - ALPHA2 + ALPHA1
    return torch.stack((hip, femur, tibia), -1)


def forward_kinematics(joints):
    angles = torch.arange(6, device=joints.device, dtype=joints.dtype) * (math.pi / 3)
    hip, femur, tibia = joints.unbind(-1)
    reach = HIP_X + L1 * torch.cos(ALPHA1 + femur) + L2 * torch.cos(ALPHA2 + femur + tibia)
    z = HIP_Z - L1 * torch.sin(ALPHA1 + femur) - L2 * torch.sin(ALPHA2 + femur + tibia)
    return torch.stack((reach * torch.cos(hip + angles), reach * torch.sin(hip + angles), z), -1)
