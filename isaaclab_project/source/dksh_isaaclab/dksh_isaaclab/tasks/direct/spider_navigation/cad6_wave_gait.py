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


def foot_targets(phase, speed, period=2.4, lift=0.004, stride_limit=0.012):
    angles = torch.arange(6, device=phase.device, dtype=phase.dtype) * (math.pi / 3)
    # Offsets distribute six non-overlapping swing windows over a wave cycle.
    offsets = torch.tensor([0, 3, 1, 4, 2, 5], device=phase.device, dtype=phase.dtype) / 6
    p = (phase[:, None] + offsets).remainder(1)
    duty = 5 / 6
    swing = ((p - duty) / (1 - duty)).clamp(0, 1)
    travel = torch.where(p < duty, 0.5 - p / duty, -0.5 + swing.square() * (3 - 2 * swing))
    stride = (speed * period * duty).clamp(0, stride_limit)[:, None]
    x = STANCE_RADIUS * torch.cos(angles)[None, :] + stride * travel
    y = STANCE_RADIUS * torch.sin(angles)[None, :].expand_as(x)
    z = STANCE_Z + lift * torch.sin(math.pi * swing).square()
    return torch.stack((x, y, z), -1)


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
