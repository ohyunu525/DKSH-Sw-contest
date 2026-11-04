"""Simulation-only CAD adaptation of HexapodRobot.zip's single-foot sequence.

The archive's hardware/serial modules are NEVER imported. Its unavailable body
kinematics and firmware angles are replaced by the project's measured CAD FK/IK.
This is a new conservative reference, not a claim to reproduce the original robot.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class UpgradedCrawlReference:
    WAIT, LIFT, TRANSFER, PROBE, SHIFT, FAILED = range(6)
    # CAD leg order: front-left/right, middle-left/right, rear-left/right.
    order = (0, 5, 1, 4, 2, 3)
    feature_count = 13
    fault_names = {0: "none", 1: "support_timeout", 2: "foot_not_released",
                   3: "support_lost", 4: "swing_contact", 5: "ground_not_found",
                   6: "tilt_limit", 7: "support_margin", 8: "ik_rejected"}

    def __init__(self, home, *, stride=0.03, lift=0.025, probe_speed=0.03,
                 probe_depth=0.02, lift_time=0.5, transfer_time=0.6,
                 shift_time=0.5, settle_time=0.2, debounce_time=0.06,
                 timeout=6.0, support_margin=0.015):
        settings = dict(stride=stride, lift=lift, probe_speed=probe_speed, probe_depth=probe_depth,
                        lift_time=lift_time, transfer_time=transfer_time, shift_time=shift_time,
                        settle_time=settle_time, debounce_time=debounce_time, timeout=timeout,
                        support_margin=support_margin)
        if home.ndim != 3 or home.shape[1:] != (6, 3) or not torch.isfinite(home).all():
            raise ValueError("Reference home must be finite [environments, 6, 3]")
        if any(not math.isfinite(v) or v <= 0 for v in settings.values()):
            raise ValueError("Reference settings must be finite and positive")
        self.settings = settings
        self.home = home.clone()
        self.position, self.start = home.clone(), home.clone()
        self.destination = home.clone()
        self.direction = home.new_zeros((len(home), 2))
        self.stage = torch.zeros(len(home), dtype=torch.long, device=home.device)
        self.cursor = torch.zeros_like(self.stage)
        self.fault = torch.zeros_like(self.stage)
        self.elapsed = home.new_zeros(len(home))
        self.stable, self.released, self.pressed, self.unsupported = [self.elapsed.clone() for _ in range(4)]

    @property
    def failed(self):
        return self.stage == self.FAILED

    def active_leg(self):
        return torch.tensor(self.order, device=self.stage.device)[self.cursor]

    def reset(self, ids):
        for value in (self.position, self.start, self.destination):
            value[ids] = self.home[ids]
        for value in (self.direction, self.stage, self.cursor, self.fault, self.elapsed,
                      self.stable, self.released, self.pressed, self.unsupported):
            value[ids] = 0

    def fail(self, mask, reason):
        mask = mask & ~self.failed
        self.fault[mask] = reason
        self.stage[mask] = self.FAILED

    def _enter(self, mask, stage):
        self.stage[mask] = stage
        self.elapsed[mask] = 0
        self.start[mask] = self.position[mask]

    def support_margin(self):
        """Distance of body origin to the convex hull of the five commanded feet.

        Commanded geometry only: contact and actual body stability are separately
        checked. This is not a dynamic balance or actual center-of-mass guarantee.
        """
        p = self.position[..., :2]
        edge = p[:, None, :, :] - p[:, :, None, :]
        other = p[:, None, None, :, :] - p[:, :, None, None, :]
        cross = edge[..., 0, None] * other[..., 1] - edge[..., 1, None] * other[..., 0]
        keep = torch.arange(6, device=p.device)[None] != self.active_leg()[:, None]
        hull = ((cross >= -1e-7) | ~keep[:, None, None, :]).all(-1)
        length = edge.norm(dim=-1)
        hull &= keep[:, :, None] & keep[:, None, :] & (length > 1e-6)
        distance = (edge[..., 1] * p[:, :, None, 0] - edge[..., 0] * p[:, :, None, 1]) / length.clamp_min(1e-6)
        return torch.where(hull, distance, torch.inf).flatten(1).amin(-1)

    def advance(self, contacts, command, gravity, dt, mask=None):
        if not math.isfinite(dt) or not 0 < dt <= 0.05:
            raise ValueError("Reference requires a positive control interval <= 50 ms")
        if (contacts.shape != self.home.shape[:2] or command.shape != (len(self.home), 3)
                or gravity.shape != command.shape or not torch.isfinite(command).all()
                or not torch.isfinite(gravity).all()):
            raise ValueError("Invalid reference feedback")
        mask = torch.ones_like(self.failed) if mask is None else mask
        active = mask & ~self.failed
        contacts = contacts.bool()
        leg = self.active_leg()
        moving = F.one_hot(leg, 6).bool()
        foot_down = (contacts & moving).any(-1)
        supported = (contacts | moving).all(-1)
        # During body translation ALL six feet must remain planted.
        supported = torch.where(self.stage == self.SHIFT, contacts.all(-1), supported)
        self.elapsed[active] += dt
        self.stable[active] = torch.where(contacts.all(-1)[active], self.stable[active] + dt, 0)
        self.unsupported[active] = torch.where(supported[active], 0, self.unsupported[active] + dt)
        self.released[active] = torch.where(~foot_down[active], self.released[active] + dt, 0)
        self.pressed[active] = torch.where(foot_down[active], self.pressed[active] + dt, 0)
        self.fail(active & (gravity[:, 2] > -0.90), 6)
        moving_stage = (self.stage >= self.LIFT) & (self.stage <= self.PROBE)
        self.fail(active & moving_stage & (self.unsupported >= 0.1), 3)
        before = self.stage.clone()
        s = self.settings
        waiting = active & (before == self.WAIT)
        walking = command[:, :2].norm(dim=-1) > 1e-6
        ready = waiting & walking & (self.stable >= s['settle_time'])
        margin = self.support_margin()
        self.fail(ready & (~torch.isfinite(margin) | (margin < s['support_margin'])), 7)
        ready &= ~self.failed
        self.direction[ready] = F.normalize(command[ready, :2], dim=-1)
        self.destination[ready] = self.position[ready]
        xy = self.home[..., :2] + self.direction[:, None] * s['stride'] / 2
        selected = ready[:, None] & moving
        self.destination[..., :2].copy_(torch.where(selected[..., None], xy, self.destination[..., :2]))
        self._enter(ready, self.LIFT)
        self.released[ready] = self.pressed[ready] = 0
        self.fail(waiting & walking & (self.elapsed > s['timeout']), 1)

        lifting = active & (before == self.LIFT) & ~self.failed
        u = (self.elapsed / s['lift_time']).clamp(0, 1)
        ease = u.square() * (3 - 2*u)
        z = self.start[..., 2] + ease[:, None] * s['lift']
        self.position[..., 2].copy_(torch.where((lifting[:, None] & moving), z, self.position[..., 2]))
        lifted = lifting & (self.elapsed >= s['lift_time'])
        self.fail(lifted & (self.released < s['debounce_time']), 2)
        self._enter(lifted & ~self.failed, self.TRANSFER)

        transferring = active & (before == self.TRANSFER) & ~self.failed
        self.fail(transferring & (self.pressed >= s['debounce_time']), 4)
        transferring &= ~self.failed
        u = (self.elapsed / s['transfer_time']).clamp(0, 1)
        ease = (u.square() * (3 - 2*u))[:, None, None]
        xy = self.start[..., :2] + ease * (self.destination[..., :2] - self.start[..., :2])
        self.position[..., :2].copy_(torch.where((transferring[:, None] & moving)[..., None], xy, self.position[..., :2]))
        self._enter(transferring & (self.elapsed >= s['transfer_time']), self.PROBE)

        probing = active & (before == self.PROBE) & ~self.failed
        landed = probing & (self.pressed >= s['debounce_time'])
        lower = probing & ~foot_down  # stop descending on the first press, then debounce
        self.position[..., 2] -= (lower[:, None] & moving) * (s['probe_speed'] * dt)
        too_low = ((self.position[..., 2] < self.home[..., 2] - s['probe_depth']) & moving).any(-1)
        self.fail(probing & ~landed & (too_low | (self.elapsed > s['timeout'])), 5)
        self._enter(landed & ~self.failed, self.SHIFT)

        shifting = active & (before == self.SHIFT) & ~self.failed
        self.fail(shifting & (self.stable == 0) & (self.unsupported >= 0.1), 3)
        shifting &= ~self.failed
        u = (self.elapsed / s['shift_time']).clamp(0, 1)
        ease = u.square() * (3 - 2*u)
        xy = self.start[..., :2] - ease[:, None, None] * self.direction[:, None] * (s['stride'] / 6)
        self.position[..., :2].copy_(torch.where(shifting[:, None, None], xy, self.position[..., :2]))
        shifted = shifting & (self.elapsed >= s['shift_time'])
        self.cursor[shifted] = (self.cursor[shifted] + 1) % 6
        self._enter(shifted, self.WAIT)
        self.stable[shifted] = 0
        return self.position

    def features(self):
        return torch.cat((F.one_hot(self.stage, 6), F.one_hot(self.active_leg(), 6),
                          (self.elapsed / self.settings['timeout']).clamp(0, 1)[:, None]), dim=-1).to(self.home.dtype)
