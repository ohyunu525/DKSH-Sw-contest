"""Estimate sole support from link forces and terrain height without Isaac Sim.

Isaac Lab 2.1 exposes net link forces, not contact point locations. A tibia's
force alone therefore cannot distinguish its sole from a shaft/wall collision.
This helper requires upward force and a sole close to the supporting surface.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch


def sole_contact_state(
    force_history: torch.Tensor, *, sensor_feet: list[int], sensor_body: list[int],
    sole_positions_w: torch.Tensor, surface_heights: torch.Tensor,
    force_threshold: float, height_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return supported feet ``(N,L)`` and undesired contacts ``(N,)``.

    ``force_history`` is world-frame force history ``(N,H,B,3)``. Foot indices
    and sole positions share leg order; surface heights are sampled separately
    under each sole. A foot needs upward vertical force above the threshold in
    any history sample and current sole height within the inclusive tolerance.
    Tibia force without valid sole support counts as an undesired contact, as
    does sufficiently large force on any configured non-foot body.

    This is a geometric support estimate, not contact-point reconstruction.
    Shape, index and scalar validation avoids device synchronization per step.
    """
    if not math.isfinite(force_threshold) or force_threshold <= 0:
        raise ValueError("force_threshold must be finite and positive")
    if not math.isfinite(height_tolerance) or height_tolerance < 0:
        raise ValueError("height_tolerance must be finite and nonnegative")
    if force_history.ndim != 4 or force_history.shape[-1] != 3 or force_history.shape[1] < 1:
        raise ValueError("force_history must have shape (N,H,B,3) with nonempty history")
    count, _, bodies, _ = force_history.shape
    feet_indices = tuple(sensor_feet)
    body_indices = tuple(sensor_body)
    if not feet_indices:
        raise ValueError("sensor_feet must contain at least one foot index")
    for name, indices in (("sensor_feet", feet_indices), ("sensor_body", body_indices)):
        if any(not isinstance(index, Integral) or isinstance(index, bool) for index in indices):
            raise ValueError(f"{name} must contain integer body indices")
        if any(index < 0 or index >= bodies for index in indices):
            raise ValueError(f"{name} contains an index outside the sensor body range")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must contain unique indices")
    if set(feet_indices).intersection(body_indices):
        raise ValueError("sensor_feet and sensor_body must be disjoint")
    legs = len(feet_indices)
    if sole_positions_w.shape != (count, legs, 3):
        raise ValueError("sole_positions_w must have shape (N,L,3) matching sensor_feet")
    if surface_heights.shape != (count, legs):
        raise ValueError("surface_heights must have shape (N,L) matching sensor_feet")
    if sole_positions_w.device != force_history.device or surface_heights.device != force_history.device:
        raise ValueError("forces, sole positions and surface heights must share a device")
    if not all(value.is_floating_point() for value in (force_history, sole_positions_w, surface_heights)):
        raise ValueError("forces, sole positions and surface heights must be floating-point tensors")

    # Convert tuples to lists so PyTorch indexes the body dimension rather than
    # interpreting tuple entries as indices of separate tensor dimensions.
    feet_indices = list(feet_indices)
    body_indices = list(body_indices)
    force_magnitudes = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1)
    upward = force_history[:, :, feet_indices, 2].amax(dim=1) > force_threshold
    near_surface = (sole_positions_w[..., 2] - surface_heights).abs() <= height_tolerance
    feet = upward & near_surface
    tibia_without_support = (force_magnitudes[:, feet_indices] > force_threshold) & ~feet
    body = tibia_without_support.any(dim=-1)
    if body_indices:
        body |= (force_magnitudes[:, body_indices] > force_threshold).any(dim=-1)
    return feet, body


def _validate_foot_geometry(foot_positions_w: torch.Tensor, surface_heights: torch.Tensor) -> None:
    if (foot_positions_w.ndim != 4 or foot_positions_w.shape[-1] != 3
            or foot_positions_w.shape[1] < 1 or foot_positions_w.shape[2] < 1):
        raise ValueError("foot_positions_w must have shape (N,L,K,3) with nonempty feet and witnesses")
    if surface_heights.shape != foot_positions_w.shape[:-1]:
        raise ValueError("surface_heights must have shape (N,L,K) matching foot_positions_w")
    if surface_heights.device != foot_positions_w.device:
        raise ValueError("foot positions and surface heights must share a device")
    if not foot_positions_w.is_floating_point() or not surface_heights.is_floating_point():
        raise ValueError("foot positions and surface heights must be floating-point tensors")


def closest_foot_support(
    foot_positions_w: torch.Tensor, surface_heights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select each foot's actual geometry witness closest to its own terrain.

    Inputs have shape ``(N,L,K,3)`` and ``(N,L,K)``. Returns world positions
    ``(N,L,3)``, matching terrain heights ``(N,L)`` and witness indices ``(N,L)``.
    Callers can gather matching point velocities with those indices for slip.
    Witnesses must describe foot geometry, not the tibia shaft. All validation
    inspects metadata only, avoiding device synchronization before selection.
    """
    _validate_foot_geometry(foot_positions_w, surface_heights)
    gaps = (foot_positions_w[..., 2] - surface_heights).abs()
    indices = gaps.argmin(dim=-1)
    positions = foot_positions_w.gather(2, indices[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    heights = surface_heights.gather(2, indices[..., None]).squeeze(2)
    return positions, heights, indices


def foot_contact_state(
    force_history: torch.Tensor, *, sensor_feet: list[int], sensor_body: list[int],
    foot_positions_w: torch.Tensor, surface_heights: torch.Tensor,
    force_threshold: float, height_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate support using any actual foot-geometry witness near terrain.

    Foot force history and body-contact classification follow the legacy sole
    helper. The closest witness makes the height condition equivalent to any
    foot witness lying within tolerance, allowing heel support with a raised
    toe. Upward force is still required; horizontal wall forces, downward-only
    forces and unsupported tibia impacts remain undesired contacts.
    """
    positions, heights, _ = closest_foot_support(foot_positions_w, surface_heights)
    return sole_contact_state(
        force_history, sensor_feet=sensor_feet, sensor_body=sensor_body,
        sole_positions_w=positions, surface_heights=heights,
        force_threshold=force_threshold, height_tolerance=height_tolerance,
    )


def distal_foot_contact_state(
    force_history: torch.Tensor, *, sensor_feet: list[int], sensor_body: list[int],
    distal_foot_positions_w: torch.Tensor, distal_surface_heights: torch.Tensor,
    proximal_foot_positions_w: torch.Tensor, proximal_surface_heights: torch.Tensor,
    force_threshold: float, height_tolerance: float, proximal_height_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate distal support while rejecting loaded non-distal grounding.

    Distal and proximal geometry are leg-major ``(N,L,K,3)`` witness patches;
    their witness counts may differ. The proximal argument must contain the
    entire hull outside the distal patch, including the knee and middle shaft.
    Returns support ``(N,L)``, undesired
    contact ``(N,)`` and loaded legs without acceptable distal support ``(N,L)``.
    A loaded proximal patch at/below its clearance tolerance invalidates a leg
    even when its distal tip also reaches terrain, preventing a flat shaft
    from qualifying as a foot. Deep proximal penetration is also rejected.

    Net link forces cannot identify the physical contact point. The proximal
    check is deliberately a conservative geometry test gated by link load;
    it does not infer a collision from a nearby unloaded swing leg.
    A non-distal witness near the patch boundary can therefore reject a loaded
    foot before actual shaft contact; height tolerance controls that margin.
    Historical full-foot and toe-button contracts are unchanged.
    """
    if not math.isfinite(proximal_height_tolerance) or proximal_height_tolerance < 0:
        raise ValueError("proximal_height_tolerance must be finite and nonnegative")
    _validate_foot_geometry(proximal_foot_positions_w, proximal_surface_heights)
    supported, body = foot_contact_state(
        force_history, sensor_feet=sensor_feet, sensor_body=sensor_body,
        foot_positions_w=distal_foot_positions_w, surface_heights=distal_surface_heights,
        force_threshold=force_threshold, height_tolerance=height_tolerance,
    )
    if proximal_foot_positions_w.shape[:2] != supported.shape:
        raise ValueError("proximal geometry must share the distal geometry's environment and leg counts")
    if proximal_foot_positions_w.device != force_history.device:
        raise ValueError("proximal geometry and force history must share a device")
    loaded = (
        torch.linalg.vector_norm(force_history[:, :, list(sensor_feet)], dim=-1).amax(dim=1)
        > force_threshold
    )
    proximal_grounded = (
        proximal_foot_positions_w[..., 2] - proximal_surface_heights
    ).amin(dim=-1) <= proximal_height_tolerance
    supported = supported & ~proximal_grounded
    non_distal_loaded = loaded & ~supported
    return supported, body | non_distal_loaded.any(dim=-1), non_distal_loaded


def strict_leg_contact_state(
    force_history: torch.Tensor, *, sensor_feet: list[int],
    sensor_leg_nonfeet: list[list[int]], sensor_body: list[int],
    distal_supported: torch.Tensor, force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify loads across entire legs, including femur-only support.

    Returns loaded legs ``(N,L)``, invalid loaded legs ``(N,L)`` and nonfoot
    body contacts ``(N,)``. A leg is loaded by any of its tibia, femur or hip
    links. Any hip/femur load invalidates that leg, even if its distal tip is
    supported; tibia load is invalid when ``distal_supported`` is false.
    The caller supplies distal support after its force and geometry checks.

    Sensor indices may have any order. ``sensor_leg_nonfeet`` provides each
    leg's hip/femur indices in the same public leg order as ``sensor_feet``.
    ``sensor_body`` additionally covers base and other nonfoot links; overlap
    with the mapped hip/femur indices is allowed. The final result aggregates
    the union, so a mapped femur cannot disappear from body-contact reporting.
    Force magnitudes use every history sample and all directions. An unloaded
    swing leg is never classified as an invalid load solely from geometry.
    Nonfinite force samples fail closed as invalid loads on their own body;
    a malformed tibia sample invalidates even otherwise valid distal support.
    """
    if not math.isfinite(force_threshold) or force_threshold <= 0:
        raise ValueError("force_threshold must be finite and positive")
    if force_history.ndim != 4 or force_history.shape[-1] != 3 or force_history.shape[1] < 1:
        raise ValueError("force_history must have shape (N,H,B,3) with nonempty history")
    if not force_history.is_floating_point():
        raise ValueError("force_history must be a floating-point tensor")
    count, _, bodies, _ = force_history.shape

    def indices(value, name):
        try:
            result = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain integer body indices") from exc
        if any(not isinstance(index, Integral) or isinstance(index, bool) for index in result):
            raise ValueError(f"{name} must contain integer body indices")
        if any(index < 0 or index >= bodies for index in result):
            raise ValueError(f"{name} contains an index outside the sensor body range")
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must contain unique indices")
        return result

    feet_indices = indices(sensor_feet, "sensor_feet")
    body_indices = indices(sensor_body, "sensor_body")
    if not feet_indices:
        raise ValueError("sensor_feet must contain at least one foot index")
    try:
        leg_groups = tuple(sensor_leg_nonfeet)
    except TypeError as exc:
        raise ValueError("sensor_leg_nonfeet must provide one group per foot") from exc
    if len(leg_groups) != len(feet_indices):
        raise ValueError("sensor_leg_nonfeet must provide one group per foot")
    leg_groups = tuple(indices(group, "sensor_leg_nonfeet") for group in leg_groups)
    leg_indices = tuple(index for group in leg_groups for index in group)
    if len(set(leg_indices)) != len(leg_indices):
        raise ValueError("sensor_leg_nonfeet groups must be disjoint across legs")
    nonfoot_indices = sorted(set(body_indices).union(leg_indices))
    if set(feet_indices).intersection(nonfoot_indices):
        raise ValueError("foot and nonfoot sensor indices must be disjoint")
    if distal_supported.shape != (count, len(feet_indices)) or distal_supported.dtype != torch.bool:
        raise ValueError("distal_supported must be boolean with shape (N,L) matching sensor_feet")
    if distal_supported.device != force_history.device:
        raise ValueError("distal support and force history must share a device")

    invalid_force = ~torch.isfinite(force_history).all(dim=-1).all(dim=1)
    body_loaded = (
        torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1) > force_threshold
    ) | invalid_force
    tibia_loaded = body_loaded[:, list(feet_indices)]
    nonfoot_leg_loaded = torch.stack([
        body_loaded[:, list(group)].any(dim=-1) for group in leg_groups
    ], dim=-1)
    loaded_legs = tibia_loaded | nonfoot_leg_loaded
    invalid_legs = (
        nonfoot_leg_loaded | (tibia_loaded & ~distal_supported)
        | invalid_force[:, list(feet_indices)]
    )
    nonfoot_body_contact = body_loaded[:, nonfoot_indices].any(dim=-1)
    return loaded_legs, invalid_legs, nonfoot_body_contact


def toe_button_contact_state(
    force_history: torch.Tensor, *, sensor_feet: list[int], sensor_body: list[int],
    foot_positions_w: torch.Tensor, foot_surface_heights: torch.Tensor,
    button_positions_w: torch.Tensor, button_surface_heights: torch.Tensor,
    previous_pressed: torch.Tensor, force_threshold: float,
    press_force_threshold: float, release_force_threshold: float,
    foot_height_tolerance: float, button_height_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return six simulated toe-button states and undesired robot contacts.

    Each switch uses the upward tibia force reported by its link ContactSensor
    and one CAD-derived distal toe witness. Force and height hysteresis emulate
    a digital push-button; this is a simulation proxy, not a physical switch
    model. Body-contact classification still uses the full foot collision hull
    and all configured non-foot links.
    """
    if (not math.isfinite(press_force_threshold) or press_force_threshold <= 0
            or not math.isfinite(release_force_threshold) or release_force_threshold < 0
            or release_force_threshold >= press_force_threshold):
        raise ValueError("button release force must be finite, nonnegative and below its press force")
    if not math.isfinite(button_height_tolerance) or button_height_tolerance < 0:
        raise ValueError("button_height_tolerance must be finite and nonnegative")
    support, body = foot_contact_state(
        force_history, sensor_feet=sensor_feet, sensor_body=sensor_body,
        foot_positions_w=foot_positions_w, surface_heights=foot_surface_heights,
        force_threshold=force_threshold, height_tolerance=foot_height_tolerance,
    )
    count, _, _, _ = force_history.shape
    legs = len(sensor_feet)
    if button_positions_w.shape != (count, legs, 3):
        raise ValueError("button_positions_w must have shape (N,L,3) matching sensor_feet")
    if button_surface_heights.shape != (count, legs):
        raise ValueError("button_surface_heights must have shape (N,L) matching sensor_feet")
    if previous_pressed.shape != (count, legs) or previous_pressed.dtype != torch.bool:
        raise ValueError("previous_pressed must be a boolean tensor with shape (N,L)")
    tensors = (button_positions_w, button_surface_heights, previous_pressed)
    if any(value.device != force_history.device for value in tensors):
        raise ValueError("button inputs and force history must share a device")
    if any(not value.is_floating_point() for value in tensors[:2]):
        raise ValueError("button positions and surface heights must be floating-point tensors")

    indices = list(sensor_feet)
    upward_force = force_history[:, :, indices, 2].amax(dim=1)
    near_button_surface = (
        (button_positions_w[..., 2] - button_surface_heights).abs() <= button_height_tolerance
    )
    pressed = (
        (upward_force > press_force_threshold) & near_button_surface
    ) | (
        previous_pressed & (upward_force > release_force_threshold) & near_button_surface
    )
    # Keep the full-hull support calculation explicit: it validates this
    # classifier's geometry and prevents unsupported tibia impacts counting as
    # foot support even though the policy sees only button bits.
    del support
    return pressed, body
