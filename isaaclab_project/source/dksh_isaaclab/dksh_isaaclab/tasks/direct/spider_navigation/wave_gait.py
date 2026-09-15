"""Kinematics for the primitive eight-leg URDF; no Isaac Sim dependency.

The CAD variants have different joint origins and must not use this solver.
All dimensions are metres and angles are radians.
"""

import math
import torch

HIP_LENGTH = 0.08617
FEMUR_LENGTH = 0.100
TIBIA_LENGTH = 0.120
STANCE_FEMUR = 1.40
STANCE_TIBIA = 0.20
STANCE_RADIUS = HIP_LENGTH + FEMUR_LENGTH * math.cos(STANCE_FEMUR) + TIBIA_LENGTH * math.cos(
    STANCE_FEMUR + STANCE_TIBIA
)
STANCE_DEPTH = FEMUR_LENGTH * math.sin(STANCE_FEMUR) + TIBIA_LENGTH * math.sin(
    STANCE_FEMUR + STANCE_TIBIA
)


def foot_targets(phase, speed, period=2.0, lift=0.006, duty=0.875):
    """N x 8 x 3 foot targets relative to each hip, in body-aligned axes.

    Seven stance legs and one swing leg. Phase is cycles, not radians.
    The stride follows the commanded body speed during stance.
    """
    angles = torch.arange(8, device=phase.device, dtype=phase.dtype) * (math.pi / 4)
    # Alternate opposite sides instead of lifting neighbouring legs in sequence.
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=phase.device, dtype=phase.dtype)
    leg_phase = (phase[:, None] + order[None, :] / 8).remainder(1.0)
    swing = ((leg_phase - duty) / (1 - duty)).clamp(0, 1)
    stride = (speed * period * duty).clamp(max=0.020)[:, None]
    travel = torch.where(
        leg_phase < duty,
        0.5 - leg_phase / duty,
        -0.5 + swing.square() * (3 - 2 * swing),
    )
    x = STANCE_RADIUS * torch.cos(angles)[None, :] + stride * travel
    y = STANCE_RADIUS * torch.sin(angles)[None, :].expand_as(x)
    z = -STANCE_DEPTH + lift * torch.sin(math.pi * swing).square()
    return torch.stack((x, y, z), dim=-1)


def inverse_kinematics(feet):
    """Return N x 8 x (hip, femur, tibia), matching the primitive URDF axes."""
    angles = torch.arange(8, device=feet.device, dtype=feet.dtype) * (math.pi / 4)
    x, y, z = feet.unbind(-1)
    hip = (torch.atan2(y, x) - angles + math.pi).remainder(2 * math.pi) - math.pi
    reach = torch.sqrt(x.square() + y.square()) - HIP_LENGTH
    depth = -z
    cosine = (reach.square() + depth.square() - FEMUR_LENGTH**2 - TIBIA_LENGTH**2) / (
        2 * FEMUR_LENGTH * TIBIA_LENGTH
    )
    tibia = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
    femur = torch.atan2(depth, reach) - torch.atan2(
        TIBIA_LENGTH * torch.sin(tibia), FEMUR_LENGTH + TIBIA_LENGTH * torch.cos(tibia)
    )
    return torch.stack((hip, femur, tibia), dim=-1)


def forward_kinematics(joints):
    angles = torch.arange(8, device=joints.device, dtype=joints.dtype) * (math.pi / 4)
    hip, femur, tibia = joints.unbind(-1)
    reach = HIP_LENGTH + FEMUR_LENGTH * torch.cos(femur) + TIBIA_LENGTH * torch.cos(femur + tibia)
    depth = FEMUR_LENGTH * torch.sin(femur) + TIBIA_LENGTH * torch.sin(femur + tibia)
    return torch.stack((reach * torch.cos(hip + angles), reach * torch.sin(hip + angles), -depth), dim=-1)
