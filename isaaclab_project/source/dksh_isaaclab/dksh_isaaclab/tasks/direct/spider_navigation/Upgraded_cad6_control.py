"""CAD toe kinematics and continuous foothold trajectories, independent of Isaac Sim."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


UPGRADED_URDF = Path(__file__).resolve().parents[6] / "assets/spiderbot_variants/onshape_20260928/robot.training.urdf"


def Upgraded_rotation(axis, angle):
    """Rodrigues rotations with arbitrary batch dimensions."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + angle.sin()[..., None, None] * skew + (1 - angle.cos())[..., None, None] * (skew @ skew)


class UpgradedCad6Kinematics:
    def __init__(self, toe_offsets, collision_offsets, proximal_offsets, *, device="cpu", dtype=torch.float32):
        joints = {j.get("name"): j for j in ET.parse(UPGRADED_URDF).getroot().findall("joint")}
        origins, rotations, axes, limits = [], [], [], []
        for leg in range(6):
            leg_origins, leg_rotations, leg_axes, leg_limits = [], [], [], []
            for part in ("hip", "femur", "tibia"):
                joint = joints[f"leg_{leg}_{part}_joint"]
                origin = joint.find("origin")
                leg_origins.append([float(v) for v in origin.get("xyz").split()])
                rpy = torch.tensor([float(v) for v in origin.get("rpy", "0 0 0").split()], dtype=dtype)
                basis = torch.eye(3, dtype=dtype)
                roll, pitch, yaw = [Upgraded_rotation(basis[i], rpy[i]) for i in range(3)]
                leg_rotations.append(yaw @ pitch @ roll)
                axis = torch.tensor([float(v) for v in joint.find("axis").get("xyz").split()], dtype=dtype)
                leg_axes.append(axis / axis.norm())
                limit = joint.find("limit")
                leg_limits.append([float(limit.get("lower")), float(limit.get("upper"))])
            origins.append(leg_origins)
            rotations.append(torch.stack(leg_rotations))
            axes.append(torch.stack(leg_axes))
            limits.append(leg_limits)
        self.origins = torch.tensor(origins, dtype=dtype, device=device)
        self.rotations = torch.stack(rotations).to(device)
        self.axes = torch.stack(axes).to(device)
        self.limits = torch.tensor(limits, dtype=dtype, device=device)
        self.toes = torch.tensor(toe_offsets, dtype=dtype, device=device)
        self.collision_points = []
        for part in ("hip", "femur", "tibia"):
            points = [list(collision_offsets[f"leg_{leg}_{part}"]) for leg in range(6)]
            if part == "tibia":
                points = [p + list(proximal_offsets[leg]) for leg, p in enumerate(points)]
            length = max(map(len, points))
            # Repeating a witness preserves the minimum without a padding mask.
            points = [p + [p[0]] * (length - len(p)) for p in points]
            self.collision_points.append(torch.tensor(points, dtype=dtype, device=device))

    def forward(self, angles):
        rotation = torch.eye(3, dtype=angles.dtype, device=angles.device).expand(*angles.shape[:-1], 3, 3)
        position = torch.zeros_like(angles)
        frames, pivots, axes = [], [], []
        for joint in range(3):
            position = position + (rotation @ self.origins[:, joint, :, None]).squeeze(-1)
            rotation = rotation @ self.rotations[:, joint]
            pivots.append(position)
            axes.append((rotation @ self.axes[:, joint, :, None]).squeeze(-1))
            rotation = rotation @ Upgraded_rotation(self.axes[:, joint], angles[..., joint])
            frames.append((rotation, position))
        toe = position + (rotation @ self.toes[..., None]).squeeze(-1)
        jacobian = torch.stack([torch.linalg.cross(axis, toe - pivot) for axis, pivot in zip(axes, pivots)], -1)
        return toe, jacobian, frames

    def clearance(self, angles):
        """Link/proximal-foot clearance above each toe's local horizontal plane."""
        toes, _, frames = self.forward(angles)
        clearances = []
        for points, (rotation, origin) in zip(self.collision_points, frames):
            z = (rotation[..., 2, None, :] * points).sum(-1) + origin[..., 2, None]
            clearances.append(z.amin(-1) - toes[..., 2])
        return torch.stack(clearances, -1).amin(-1)

    def inverse(self, targets, seed, limits=None, *, iterations=6, tolerance=0.001, clearance=0.008):
        """Damped IK with joint bounds and a per-leg accepted-target mask."""
        limits = self.limits if limits is None else limits
        low, high = limits.unbind(-1)
        angles = seed.clone().clamp(low, high)
        eye = torch.eye(3, dtype=angles.dtype, device=angles.device)
        finite = torch.isfinite(targets).all(-1)
        safe_targets = torch.where(finite[..., None], targets, self.forward(seed)[0])
        for _ in range(iterations):
            position, jacobian, _ = self.forward(angles)
            error = safe_targets - position
            system = jacobian @ jacobian.transpose(-1, -2) + 0.002 ** 2 * eye
            delta = (jacobian.transpose(-1, -2) @ torch.linalg.solve(system, error[..., None])).squeeze(-1)
            angles = (angles + delta.clamp(-0.15, 0.15)).clamp(low, high)
        error = (self.forward(angles)[0] - safe_targets).norm(dim=-1)
        valid = finite & (error <= tolerance) & (self.clearance(angles) >= clearance)
        return torch.where(valid[..., None], angles, seed), valid, error


class UpgradedFootTrajectory:
    """Tripod schedule with planted-foot advection and latched swing endpoints.

    Command is body-frame vx, vy, yaw-rate. Actions are bounded per-leg XYZ
    touchdown offsets. Positive Z also raises the swing apex by the same amount.
    State consists only of commanded quantities, so it can run on the real robot.
    """

    feature_count = 109

    def __init__(self, home, *, period=2.0, duty=0.65, lift=0.02, offset_limit=0.015):
        if not all(math.isfinite(v) and v > 0 for v in (period, lift, offset_limit)) or not 0.5 < duty < 0.9:
            raise ValueError("Invalid Upgraded gait parameters")
        self.home = home.clone()
        self.period, self.duty, self.lift, self.offset_limit = period, duty, lift, offset_limit
        self.offsets = home.new_tensor((0.0, 0.5, 0.0, 0.5, 0.0, 0.5))
        self.elapsed = home.new_zeros(home.shape[0])
        self.position = home.clone()
        self.start = home.clone()
        self.end = home.clone()
        self.start_velocity = torch.zeros_like(home)
        self.end_velocity = torch.zeros_like(home)
        self.height = home.new_full(home.shape[:2], lift)

    def reset(self, env_ids):
        self.elapsed[env_ids] = 0
        self.position[env_ids] = self.home[env_ids]
        self.start[env_ids] = self.home[env_ids]
        self.end[env_ids] = self.home[env_ids]
        self.start_velocity[env_ids] = 0
        self.end_velocity[env_ids] = 0
        self.height[env_ids] = self.lift

    def phase(self):
        return (self.elapsed[:, None] / self.period + self.offsets).remainder(1)

    @staticmethod
    def velocity_at(points, command):
        velocity = torch.zeros_like(points)
        velocity[..., 0] = -command[:, 0, None] + command[:, 2, None] * points[..., 1]
        velocity[..., 1] = -command[:, 1, None] - command[:, 2, None] * points[..., 0]
        return velocity

    @staticmethod
    def advect(points, command, dt):
        angle = command[:, 2, None] * dt
        sine, cosine = angle.sin(), angle.cos()
        sinc = torch.sinc(angle / math.pi)
        cosc = angle / 2 * torch.sinc(angle / (2 * math.pi)).square()
        vx, vy = command[:, 0, None], command[:, 1, None]
        moved = points.clone()
        moved[..., 0] = cosine * points[..., 0] + sine * points[..., 1] - dt * (sinc * vx + cosc * vy)
        moved[..., 1] = -sine * points[..., 0] + cosine * points[..., 1] - dt * (-cosc * vx + sinc * vy)
        return moved

    def advance(self, command, actions, dt):
        if not math.isfinite(dt) or not 0 < dt < self.period * (1 - self.duty) / 2:
            raise ValueError("Trajectory dt must resolve the swing phase")
        if not torch.isfinite(command).all() or not torch.isfinite(actions).all():
            raise ValueError("Upgraded controller inputs must be finite")
        previous_phase = self.phase()
        # Fade the velocity command in, while feet initially occupy their stance.
        command = command * (self.elapsed / self.period).clamp(0, 1)[:, None]
        previous_stance = previous_phase < self.duty
        # Advance stance for only its fraction of a boundary-crossing interval.
        until_lift = (self.duty - previous_phase).clamp_min(0) * self.period
        stance_dt = torch.minimum(until_lift, until_lift.new_tensor(dt)) * previous_stance
        self.position.copy_(self.advect(self.position, command, stance_dt))
        self.elapsed += dt
        phase = self.phase()
        swinging = phase >= self.duty
        liftoff = previous_stance & swinging
        offsets = actions.reshape_as(self.home).clamp(-1, 1) * self.offset_limit
        target = self.home - self.velocity_at(self.home, command) * (self.period * self.duty / 2) + offsets
        self.start.copy_(torch.where(liftoff[..., None], self.position, self.start))
        self.end.copy_(torch.where(liftoff[..., None], target, self.end))
        self.start_velocity.copy_(torch.where(liftoff[..., None], self.velocity_at(self.position, command), self.start_velocity))
        self.end_velocity.copy_(torch.where(liftoff[..., None], self.velocity_at(target, command), self.end_velocity))
        activity = ((command[:, :2].norm(dim=-1) + 0.2 * command[:, 2].abs()) / 0.03).clamp(0, 1)
        height = activity[:, None] * (self.lift + offsets[..., 2].clamp_min(0))
        self.height.copy_(torch.where(liftoff, height, self.height))
        landed = ~previous_stance & ~swinging
        u = ((phase - self.duty) / (1 - self.duty)).clamp(0, 1)
        u = torch.where(landed, torch.ones_like(u), u)[..., None]
        duration = self.period * (1 - self.duty)
        # Hermite endpoints match planted-foot velocity at both boundaries.
        swing = ((2*u**3 - 3*u**2 + 1) * self.start + (-2*u**3 + 3*u**2) * self.end
                 + (u**3 - 2*u**2 + u) * duration * self.start_velocity
                 + (u**3 - u**2) * duration * self.end_velocity)
        swing[..., 2] += 16 * u[..., 0]**2 * (1-u[..., 0])**2 * self.height
        self.position.copy_(torch.where((swinging | landed)[..., None], swing, self.position))
        after_landing = phase * self.period * landed
        self.position.copy_(self.advect(self.position, command, after_landing))
        return self.position

    def features(self):
        phase = self.phase() * (2 * math.pi)
        return torch.cat((phase.sin(), phase.cos(),
                          ((self.position - self.home) / 0.1).flatten(1),
                          ((self.start - self.home) / 0.1).flatten(1),
                          ((self.end - self.home) / 0.1).flatten(1),
                          (self.start_velocity / 0.1).flatten(1),
                          (self.end_velocity / 0.1).flatten(1),
                          self.height / 0.05, (self.elapsed / self.period).clamp(0, 1)[:, None]), dim=-1)
