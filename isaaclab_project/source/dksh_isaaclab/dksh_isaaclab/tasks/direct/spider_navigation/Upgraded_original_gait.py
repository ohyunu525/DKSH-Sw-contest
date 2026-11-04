"""Batched, hardware-free port of the supplied ``hexapod_joy_walk.Gait`` core.

Source: HexapodRobot2.zip / hexapod_joy_walk.py, SHA256
948b1a00a98724c1ae61230c54fe5d1bc5fb700f84df90e4521e66c890924c32.
Preserved equations: swing_profile (414-430), Gait._target (829-835),
_update_ref (846-853), _build (963-997), _budget (999-1025), and the
velocity ramp / stopping conditions of tick (1047-1133).

Intentional adaptations: tensors and SI units, CAD home positions and leg order,
physical velocity commands instead of joystick fractions, and pure proposal /
commit transactions. CAD IK, strict target-rate limiting and non-blocking contact
probing belong to the environment adapter. The original firmware angle model,
20-degree tucking, hardware output, blocking IMU routines and permissive speed
governor are NOT executed. This module does not read or import either ZIP.
"""

from __future__ import annotations

import math

import torch


UPGRADED_GAIT_SOURCE_SHA256 = '948b1a00a98724c1ae61230c54fe5d1bc5fb700f84df90e4521e66c890924c32'
UPGRADED_GAIT_PORT = 'Upgraded_original_Gait_CAD_SI_tensor_v1'


def Upgraded_swing_profile(u):
    """Original horizontal cosine and vertical rise/plateau/quadratic descent."""
    horizontal = torch.where(u <= 0.08, 0.0,
                            torch.where(u >= 0.92, 1.0,
                                        0.5 - 0.5 * torch.cos(math.pi * (u - 0.08) / 0.84)))
    height = torch.where(u < 0.40, torch.sin(0.5 * math.pi * u / 0.40),
                         torch.where(u < 0.50, 1.0, ((1.0 - u) / 0.50).square()))
    return horizontal, height


class UpgradedOriginalGait:
    """Continuous alternating-tripod trajectory, expressed in the CAD body frame.

    Commands are [forward m/s, left m/s, CCW rad/s]. A proposed state has no side
    effects, so an IK/rate governor can retry a shorter trajectory interval while
    using the same physical ``control_dt`` for the velocity ramp. ``commit`` can
    accept different proposals for different environments.
    """

    feature_count = 55
    state_names = ('position', 'start', 'phase', 'v', 'w', 'in_swing', 'walking', 'ground', 'ref_off')

    def __init__(self, home, *, period=2.4, duty=0.6, lift=0.02, stride=0.03,
                 speed_max=0.03, accel=0.05, yaw_accel=math.radians(60),
                 turn_speed_max=math.radians(15), max_turn=math.radians(12), land_approach=0.004):
        if home.ndim != 3 or home.shape[1:] != (6, 3) or not torch.isfinite(home).all():
            raise ValueError('Original Gait requires finite CAD feet [environments, 6, 3]')
        self.settings = dict(period=period, duty=duty, lift=lift, stride=stride, speed_max=speed_max,
                             accel=accel, yaw_accel=yaw_accel, turn_speed_max=turn_speed_max,
                             max_turn=max_turn, land_approach=land_approach)
        if any(not math.isfinite(v) or v <= 0 for k, v in self.settings.items() if k != 'land_approach'):
            raise ValueError('Gait settings must be finite and positive')
        if not 0.5 < duty < 1 or not math.isfinite(land_approach) or land_approach < 0:
            raise ValueError('Gait needs duty in (0.5, 1) and a nonnegative landing clearance')
        for key, value in self.settings.items():
            setattr(self, key, float(value))
        self.home = home.clone()
        self.position = home.clone()
        self.start = home.clone()
        self.base_ground = home[..., 2].clone()
        self.ground = self.base_ground.clone()
        self.phase = home.new_zeros(len(home))
        self.v = home.new_zeros((len(home), 2))
        self.w = home.new_zeros(len(home))
        self.ref_off = home.new_zeros(len(home))
        self.in_swing = torch.zeros(home.shape[:2], dtype=torch.bool, device=home.device)
        self.walking = torch.zeros(len(home), dtype=torch.bool, device=home.device)
        # Original A=(0,2,4), B=(1,3,5); archive->CAD mapping [2,1,0,5,4,3]
        # preserves the groups. CAD geometry is used directly, with no guessed
        # rigid rotation of the approximate archive body model.
        self.offset = home.new_tensor([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])[None]

    def reset(self, ids):
        self.position[ids] = self.home[ids]
        self.start[ids] = self.home[ids]
        self.ground[ids] = self.base_ground[ids]
        for value in (self.phase, self.v, self.w, self.ref_off, self.in_swing, self.walking):
            value[ids] = 0

    def _interval(self, dt):
        dt = torch.as_tensor(dt, device=self.home.device, dtype=self.home.dtype)
        if dt.ndim == 0:
            dt = dt.expand(len(self.home))
        if dt.shape != self.phase.shape:
            raise ValueError('Gait interval must be scalar or one value per environment')
        return dt

    def _target(self, v, w):
        """Original touchdown lookahead over half of the stance interval."""
        half_stance = self.period * self.duty / 2
        angle = w[:, None] * half_stance
        c, s = angle.cos(), angle.sin()
        x, y = self.home[..., 0], self.home[..., 1]
        return torch.stack((c*x - s*y + v[:, 0, None]*half_stance,
                            s*x + c*y + v[:, 1, None]*half_stance), -1)

    def _reference_offset(self, in_swing):
        keep = ~in_swing
        values = torch.where(keep, self.position[..., 2] - self.base_ground, torch.inf).sort(-1).values
        count = keep.sum(-1)
        hi = values.gather(1, (count // 2).clamp(0, 5)[:, None])[:, 0]
        lo = values.gather(1, ((count - 1) // 2).clamp(0, 5)[:, None])[:, 0]
        return torch.where(count > 0, (hi + lo) / 2, self.ref_off)

    def _build(self, phase, dt, v, w, v_target=None, w_target=None, lift_scale=1.0,
               ref_off=None, in_swing=None):
        """Original _build: replan touchdown, preserve swing starts, advect stance."""
        dt = self._interval(dt)
        if v_target is None:
            v_target, w_target = v, w
        if in_swing is None:
            in_swing = self.in_swing
        if ref_off is None:
            ref_off = self.ref_off
        p = (phase[:, None] - self.offset).remainder(1.0)
        swing = p < (1.0 - self.duty)
        starts = torch.where((swing & ~in_swing)[..., None], self.position, self.start)
        horizontal, height = Upgraded_swing_profile(p / (1.0 - self.duty))
        target = self._target(v_target, w_target)
        z_to = self.base_ground + ref_off[:, None] + self.land_approach
        swing_xy = starts[..., :2] + (target - starts[..., :2]) * horizontal[..., None]
        scale = self._interval(lift_scale)[:, None]
        swing_z = self.ground + (z_to - self.ground) * horizontal + height * self.lift * scale
        landed = in_swing & ~swing
        stance_xy = torch.where(landed[..., None], target, self.position[..., :2])
        stance_z = torch.where(landed, z_to, self.position[..., 2])
        angle = -w[:, None] * dt[:, None]
        c, s = angle.cos(), angle.sin()
        x, y = stance_xy.unbind(-1)
        stance_xy = torch.stack((c*x - s*y - v[:, 0, None]*dt[:, None],
                                 s*x + c*y - v[:, 1, None]*dt[:, None]), -1)
        feet = torch.cat((torch.where(swing[..., None], swing_xy, stance_xy),
                          torch.where(swing, swing_z, stance_z)[..., None]), -1)
        return feet, swing, starts

    def _budget(self, phase, dt, v, w, *, ref_off=None, in_swing=None):
        """Original eight-step bisection preventing excessive planted-foot travel."""
        radius = self.stride / 2 + 0.001
        old_distance = (self.position[..., :2] - self.home[..., :2]).norm(dim=-1)

        def okay(k):
            feet, swing, _ = self._build(phase, dt, v*k[:, None], w*k,
                                        ref_off=ref_off, in_swing=in_swing)
            distance = (feet[..., :2] - self.home[..., :2]).norm(dim=-1)
            return (~((distance > radius) & (distance > old_distance + 1e-12) & ~swing)).all(-1)

        one = torch.ones_like(self.phase)
        full = okay(one)
        lo, hi = torch.zeros_like(one), one.clone()
        for _ in range(8):
            mid = (lo + hi) / 2
            feasible = okay(mid)
            lo = torch.where(feasible, mid, lo)
            hi = torch.where(feasible, hi, mid)
        return torch.where(full, one, lo)

    def proposal(self, dt, command, *, control_dt=None, lift_scale=1.0, stop_body=False):
        """Return a candidate state; all tensors retained by the controller stay unchanged."""
        if command.shape != (len(self.home), 3):
            raise ValueError('Gait commands must have shape [environments, 3]')
        dt = self._interval(dt)
        physical_dt = dt if control_dt is None else self._interval(control_dt)
        vmax = min(self.stride / (self.duty * self.period), self.speed_max)
        wmax = min(self.max_turn / (self.duty * self.period), self.turn_speed_max)
        desired_v = command[:, :2].clamp(-vmax, vmax)
        desired_w = command[:, 2].clamp(-wmax, wmax)
        v = self.v + (desired_v - self.v).clamp(-self.accel * physical_dt[:, None], self.accel * physical_dt[:, None])
        w = self.w + (desired_w - self.w).clamp(-self.yaw_accel * physical_dt, self.yaw_accel * physical_dt)
        want_move = (command != 0).any(-1)
        active = self.walking | want_move
        starting = ~self.walking & want_move
        old_swing = self.in_swing & ~starting[:, None]
        phase = (torch.where(starting, 0.0, self.phase) + dt / self.period).remainder(1.0)
        ref_off = self._reference_offset(old_swing)
        budget = self._budget(phase, dt, v, w, ref_off=ref_off, in_swing=old_swing)
        effective_v, effective_w = v * budget[:, None], w * budget
        target_v, target_w = v, w
        if stop_body:
            effective_v, effective_w = torch.zeros_like(v), torch.zeros_like(w)
            target_v, target_w = effective_v, effective_w
        feet, swing, starts = self._build(phase, dt, effective_v, effective_w, target_v, target_w,
                                          lift_scale, ref_off, old_swing)
        landed = old_swing & ~swing & active[:, None]
        ground = torch.where(landed, feet[..., 2], self.ground)
        home_aligned = (((feet[..., :2] - self.home[..., :2]).norm(dim=-1) < 0.002)
                        & (feet[..., 2] - ground < 0.001)).all(-1)
        stopped = (~want_move & (v.abs() < 0.0005).all(-1)
                   & (w.abs() < math.radians(0.5)) & home_aligned)
        candidate = dict(position=feet, start=starts, phase=phase, v=v, w=w, in_swing=swing,
                         walking=active & ~stopped, ground=ground, ref_off=ref_off)
        if stop_body:
            candidate['v'], candidate['w'] = effective_v, effective_w
        for name in self.state_names:
            value = candidate[name]
            mask = active.reshape((len(active),) + (1,) * (value.ndim - 1))
            candidate[name] = torch.where(mask, value, getattr(self, name))
        candidate.update(landed=landed, dist=effective_v.norm(dim=-1) * dt * active,
                         body_scale=budget)
        return candidate

    def commit(self, proposal, mask=None):
        if mask is None:
            mask = torch.ones_like(self.walking)
        for name in self.state_names:
            getattr(self, name)[mask] = proposal[name][mask]

    def features(self):
        """State needed to distinguish identical sensors at different gait phases."""
        angle = self.phase * (2 * math.pi)
        return torch.cat((angle.sin()[:, None], angle.cos()[:, None],
                          ((self.position - self.home) / 0.1).flatten(1),
                          ((self.start - self.home) / 0.1).flatten(1),
                          (self.ground - self.base_ground) / 0.05, self.ref_off[:, None] / 0.05,
                          self.v / self.speed_max, self.w[:, None] / self.turn_speed_max,
                          self.in_swing.to(self.home.dtype), self.walking[:, None].to(self.home.dtype)), -1)
