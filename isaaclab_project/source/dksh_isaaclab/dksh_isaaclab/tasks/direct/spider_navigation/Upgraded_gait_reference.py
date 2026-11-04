"""CAD IK and simulation-time contact adapter for the original tripod Gait.

The tensor core preserves the archive's foot trajectory equations. This adapter
replaces firmware angles, blocking serial/IMU probes and the permissive governor.
Labels are the absolute INPUT targets to the existing motor filter, not filtered
targets (which would be filtered twice when replayed by BC).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .joint_angle_control import bounded_angle_targets, normalize_angles
from .Upgraded_original_gait import UpgradedOriginalGait


class UpgradedGaitReference:
    WAIT, WALK, PROBE, RECENTER, FAILED = range(5)
    feature_count = UpgradedOriginalGait.feature_count + 40
    fault_names = {0: 'none', 1: 'support_timeout', 2: 'foot_not_released',
                   3: 'support_lost', 4: 'swing_contact', 5: 'ground_not_found',
                   6: 'tilt_limit', 7: 'support_margin', 8: 'ik_rejected',
                   9: 'target_tracking_timeout', 10: 'recenter_timeout'}

    def __init__(self, home, kinematics, neutral, limits, *, period=2.4, duty=0.6,
                 lift=0.02, stride=0.03, speed_max=0.03, accel=0.05,
                 land_approach=0.004, probe_speed=0.02, probe_depth=0.025,
                 settle_time=0.2, debounce_time=0.06, timeout=6.0,
                 support_margin=0.01, clearance=0.008, rate_limit=1.5,
                 smoothing=0.35, tracking_tolerance=0.02, recenter_speed=0.01,
                 recenter_deadband=0.0003, recenter_outlier=0.008):
        settings = dict(period=period, duty=duty, lift=lift, stride=stride, speed_max=speed_max,
                        accel=accel, land_approach=land_approach, probe_speed=probe_speed,
                        probe_depth=probe_depth, settle_time=settle_time,
                        debounce_time=debounce_time, timeout=timeout, support_margin=support_margin,
                        clearance=clearance, rate_limit=rate_limit, smoothing=smoothing,
                        tracking_tolerance=tracking_tolerance, recenter_speed=recenter_speed,
                        recenter_deadband=recenter_deadband, recenter_outlier=recenter_outlier)
        if any(not math.isfinite(v) or v <= 0 for v in settings.values()) or smoothing > 1:
            raise ValueError('Gait adapter settings must be finite and positive; smoothing <= 1')
        self.settings = settings
        self.core = UpgradedOriginalGait(home, period=period, duty=duty, lift=lift, stride=stride,
                                        speed_max=speed_max, accel=accel, land_approach=land_approach)
        self.kinematics, self.limits = kinematics, limits
        self.neutral, self.angles = neutral.clone(), neutral.clone()
        self.stage = torch.zeros(len(home), dtype=torch.long, device=home.device)
        self.fault = torch.zeros_like(self.stage)
        self.pending = torch.zeros(home.shape[:2], dtype=torch.bool, device=home.device)
        self.released = torch.zeros_like(self.pending)
        self.release_age = home.new_zeros(home.shape[:2])
        self.pressed = torch.zeros_like(self.release_age)
        self.probe_origin = home[..., 2].clone()
        self.elapsed, self.stable, self.unsupported, self.blocked = [home.new_zeros(len(home)) for _ in range(4)]
        self.scale = home.new_ones(len(home))

    @property
    def home(self):
        return self.core.home

    @property
    def position(self):
        return self.core.position

    @property
    def failed(self):
        return self.stage == self.FAILED

    def reset(self, ids):
        self.core.reset(ids)
        self.angles[ids] = self.neutral[ids]
        self.stage[ids] = self.WAIT
        for state in (self.fault, self.pending, self.released, self.release_age, self.pressed,
                      self.elapsed, self.stable, self.unsupported, self.blocked):
            state[ids] = 0
        self.probe_origin[ids] = self.position[ids, :, 2]
        self.scale[ids] = 1

    def fail(self, mask, reason):
        mask = mask & ~self.failed
        self.fault[mask] = reason
        self.stage[mask] = self.FAILED

    def _enter(self, mask, stage):
        self.stage[mask] = stage
        self.elapsed[mask] = 0
        self.blocked[mask] = 0

    def _margin(self, support, position=None):
        """Commanded support polygon about body origin, not measured CoM balance."""
        p = (self.position if position is None else position)[..., :2]
        edge = p[:, None] - p[:, :, None]
        other = p[:, None, None] - p[:, :, None, None]
        cross = edge[..., 0, None] * other[..., 1] - edge[..., 1, None] * other[..., 0]
        hull = ((cross >= -1e-7) | ~support[:, None, None]).all(-1)
        length = edge.norm(dim=-1)
        hull &= support[:, :, None] & support[:, None] & (length > 1e-6)
        distance = (edge[..., 1] * p[:, :, None, 0] - edge[..., 0] * p[:, :, None, 1]) / length.clamp_min(1e-6)
        margin = torch.where(hull, distance, torch.inf).flatten(1).amin(-1)
        return torch.where(support.sum(-1) >= 3, margin, -torch.inf)

    def _plane_shift(self):
        """Original _feet_plane: fit height/tilt, reject isolated height outliers.

        A bounded per-tick correction replaces its blocking 150 ms interpolation.
        Re-evaluating the plane requires no hidden interpolation endpoint state.
        """
        p = self.position
        matrix = torch.cat((torch.ones_like(p[..., :1]), p[..., :2]), -1)
        offset = p[..., 2] - self.core.base_ground

        def fit(keep):
            weighted = matrix * keep[..., None]
            normal = matrix.transpose(-1, -2) @ weighted
            rhs = weighted.transpose(-1, -2) @ offset[..., None]
            # Tiny regularization also makes a degenerate commanded stance safe.
            eye = torch.eye(3, device=p.device, dtype=p.dtype)
            coefficients = torch.linalg.solve(normal + 1e-9 * eye, rhs)
            return (matrix @ coefficients).squeeze(-1)

        first = fit(torch.ones_like(offset))
        keep = (offset - first).abs() < self.settings['recenter_outlier']
        refine = (keep.sum(-1) >= 4) & (keep.sum(-1) < 6)
        second = fit(keep.to(p.dtype))
        return torch.where(refine[:, None], second, first)

    def _solve(self, position, applied, dt):
        s = self.settings
        q, valid, _ = self.kinematics.inverse(position, self.angles, self.limits, clearance=s['clearance'])
        valid = valid.all(-1)
        action = normalize_angles(q.flatten(1), self.limits.flatten(1, 2))
        predicted = bounded_angle_targets(action, applied.flatten(1), self.limits.flatten(1, 2),
                                         dt=dt, rate_limit=s['rate_limit'], smoothing=s['smoothing'])
        gap = (predicted - q.flatten(1)).abs().amax(-1)
        jump = (q - self.angles).abs().flatten(1).amax(-1)
        trackable = (gap <= s['tracking_tolerance']) & (jump <= s['rate_limit'] * dt + 1e-6)
        return q, valid, trackable

    def advance(self, contacts, command, gravity, dt, applied, mask=None):
        if not math.isfinite(dt) or not 0 < dt <= 0.05:
            raise ValueError('Original Gait requires a control interval <= 50 ms')
        n = len(self.stage)
        if (contacts.shape != (n, 6) or command.shape != (n, 3) or gravity.shape != (n, 3)
                or applied.shape != (n, 6, 3) or not torch.isfinite(command).all()
                or not torch.isfinite(gravity).all() or not torch.isfinite(applied).all()):
            raise ValueError('Invalid original Gait feedback')
        mask = torch.ones_like(self.failed) if mask is None else mask.bool()
        active = mask & ~self.failed
        contacts = contacts.bool()
        s, core = self.settings, self.core
        before = self.stage.clone()
        self.elapsed[active] += dt
        self.stable[active] = torch.where(contacts.all(-1)[active], self.stable[active] + dt, 0)
        self.pressed[active] = torch.where(contacts[active], self.pressed[active] + dt, 0)
        self.release_age[active] = torch.where(~contacts[active], self.release_age[active] + dt, 0)
        self.released[active] |= self.release_age[active] >= s['debounce_time']
        self.fail(active & (gravity[:, 2] > -0.90), 6)

        # Hold the old support tripod while the newly landed tripod is probed.
        support = ~(core.in_swing | self.pending)
        supported = (contacts | ~support).all(-1)
        self.unsupported[active] = torch.where(supported[active], 0, self.unsupported[active] + dt)
        self.fail(active & (before != self.WAIT) & (self.unsupported >= 0.12), 3)
        waiting = active & (before == self.WAIT) & ~self.failed
        moving = command.abs().sum(-1) > 1e-6
        ready = waiting & moving & (self.stable >= s['settle_time'])
        self._enter(ready, self.WALK)
        self.fail(waiting & moving & (self.elapsed > s['timeout']), 1)

        walking = active & (self.stage == self.WALK) & ~self.failed
        p = (core.phase[:, None] - core.offset).remainder(1) / (1 - s['duty'])
        stuck = (core.in_swing & (p > 0.5) & ~self.released).any(-1)
        early = (core.in_swing & (p > 0.15) & (p < 0.5) & self.released
                 & (self.pressed >= s['debounce_time'])).any(-1)
        self.fail(walking & stuck, 2)
        self.fail(walking & early, 4)
        walking &= ~self.failed

        unresolved = walking.clone()
        any_ik = torch.zeros_like(walking)
        # Recompute the trajectory at a shorter phase interval, without changing
        # the real control interval. No original "send anyway" governor branch.
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            if not unresolved.any():
                break
            proposal = core.proposal(dt * scale, command, control_dt=dt)
            q, valid, trackable = self._solve(proposal['position'], applied, dt)
            any_ik |= valid & unresolved
            chosen = unresolved & valid & trackable
            lifting = proposal['in_swing'] & ~core.in_swing
            margin = self._margin(~proposal['in_swing'], proposal['position'])
            unsafe = chosen & lifting.any(-1) & (~torch.isfinite(margin) | (margin < s['support_margin']))
            self.fail(unsafe, 7)
            chosen &= ~self.failed
            landed = proposal['landed'] & chosen[:, None]
            self.released[chosen] &= ~lifting[chosen]
            self.release_age[chosen] = torch.where(lifting[chosen], 0, self.release_age[chosen])
            self.pressed[chosen] = torch.where(lifting[chosen], 0, self.pressed[chosen])
            core.commit(proposal, chosen)
            self.angles[chosen] = q[chosen]
            self.scale[chosen] = scale
            self.blocked[chosen] = 0
            newly_probing = landed.any(-1)
            self.pending |= landed
            self.probe_origin[newly_probing] = core.position[newly_probing, :, 2]
            self._enter(newly_probing, self.PROBE)
            unresolved &= ~chosen & ~self.failed
        self.fail(unresolved & ~any_ik, 8)
        self.blocked[unresolved] += dt
        self.scale[unresolved] = 0
        self.fail(unresolved & (self.blocked > s['timeout']), 9)

        # Sim-time replacement for blocking Grounder.probe(). Exactly one foot
        # is lowered at a time; no phase/body translation occurs while probing.
        probing = active & (before == self.PROBE) & ~self.failed
        for _ in range(6):  # drain already stable contacts without lowering them
            first = self.pending.long().argmax(-1)
            selected = F.one_hot(first, 6).bool() & self.pending
            confirmed = probing & ((self.pressed >= s['debounce_time']) & selected).any(-1)
            done = selected & confirmed[:, None]
            core.ground[done] = core.position[..., 2][done]
            self.pending[done] = False
        selected = F.one_hot(self.pending.long().argmax(-1), 6).bool() & self.pending
        lower = selected & ~contacts & probing[:, None]
        position = core.position.clone()
        position[..., 2] -= lower * (s['probe_speed'] * dt)
        too_deep = ((self.probe_origin - position[..., 2] > s['probe_depth']) & selected).any(-1)
        self.fail(probing & (too_deep | (self.elapsed > s['timeout'])), 5)
        lowering = probing & lower.any(-1) & ~self.failed
        if lowering.any():
            q, valid, trackable = self._solve(position, applied, dt)
            self.fail(lowering & ~valid, 8)
            accepted = lowering & valid & trackable
            core.position[accepted] = position[accepted]
            self.angles[accepted] = q[accepted]
            self.blocked[accepted] = 0
            self.blocked[lowering & ~accepted] += dt
            self.fail(lowering & (self.blocked > s['timeout']), 9)
        completed = probing & ~self.pending.any(-1) & ~self.failed
        self._enter(completed, self.RECENTER)

        # Original Gait recenters after probing to avoid cumulative body sag:
        # each nominal landing starts above the last support-height estimate.
        # Remove the fitted plane while all six feet support the body. Only the
        # commanded geometry is used here; no terrain truth enters the actor.
        recentering = active & (before == self.RECENTER) & ~self.failed
        if recentering.any():
            shift = self._plane_shift()
            magnitude = shift.abs().amax(-1)
            settled = recentering & (magnitude <= s['recenter_deadband'])
            self._enter(settled, self.WALK)
            correcting = recentering & ~settled
            step = shift * (s['recenter_speed'] * dt / magnitude.clamp_min(1e-9)).clamp(max=1)[:, None]
            position = core.position.clone()
            position[..., 2] -= step
            q, valid, trackable = self._solve(position, applied, dt)
            self.fail(correcting & ~valid, 8)
            accepted = correcting & valid & trackable & contacts.all(-1)
            core.position[accepted] = position[accepted]
            core.ground[accepted] -= step[accepted]
            core.ref_off[accepted] = core._reference_offset(core.in_swing)[accepted]
            self.angles[accepted] = q[accepted]
            self.fail(correcting & (self.elapsed > s['timeout']), 10)
        return self.angles

    def features(self):
        s = self.settings
        return torch.cat((self.core.features(), F.one_hot(self.stage, 5).to(self.home.dtype),
                          self.pending.to(self.home.dtype), self.released.to(self.home.dtype),
                          (self.release_age / s['debounce_time']).clamp(0, 1),
                          (self.pressed / s['debounce_time']).clamp(0, 1),
                          torch.stack((self.elapsed / s['timeout'], self.stable / s['settle_time'],
                                       self.unsupported / 0.12, self.blocked / s['timeout'], self.scale), -1).clamp(0, 1),
                          (self.probe_origin - self.core.base_ground) / s['probe_depth']), -1)
