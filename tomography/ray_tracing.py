from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .geometry import AcquisitionGeometry
from .grid import LayeredVelocityModel, VelocityModel

Point = Tuple[float, float]


def straight_ray_path(
    source: Point, receiver: Point, n_samples: int = 200
) -> np.ndarray:
    """
    Sample a straight ray path between source and receiver.
    """

    source = np.asarray(source, dtype=float)
    receiver = np.asarray(receiver, dtype=float)
    t = np.linspace(0.0, 1.0, n_samples)
    path = np.outer(1 - t, source) + np.outer(t, receiver)
    return path


def travel_time_along_path(model: VelocityModel, path: np.ndarray) -> float:
    """
    Integrate travel time along a polyline path through the model.
    """

    segments = path[1:] - path[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    midpoints = 0.5 * (path[1:] + path[:-1])
    velocities = model.bilinear_interpolate(midpoints)
    return float(np.sum(lengths / velocities))


def travel_time_matrix(
    model: VelocityModel,
    geometry: AcquisitionGeometry,
    n_samples: int = 200,
) -> np.ndarray:
    """
    Compute predicted travel times for all source/receiver pairs.
    """

    times = np.zeros((geometry.n_sources, geometry.n_receivers))
    for (i, j), (src, rec) in zip(
        geometry.pair_indices(), geometry.ray_pairs(), strict=True
    ):
        path = straight_ray_path(src, rec, n_samples=n_samples)
        times[i, j] = travel_time_along_path(model, path)
    return times


def head_wave_travel_time(
    layered: LayeredVelocityModel, offset: float, interface: int = 0
) -> float:
    """
    Analytic travel time of a critically refracted wave along the given interface.
    """

    theta_c, intercept, crossover = layered.head_wave_parameters(interface)
    if offset < crossover:
        raise ValueError("offset is shorter than crossover distance; no head wave.")
    v_half = layered.velocities[interface + 1]
    along_interface = (offset - crossover) / v_half
    return intercept + along_interface


def shoot_head_wave(
    layered: LayeredVelocityModel, offset: float, interface: int = 0
) -> np.ndarray:
    """
    Construct the polyline path of a head wave using a simple shooting approach.
    """

    theta_c, _, crossover = layered.head_wave_parameters(interface)
    if offset < crossover:
        raise ValueError("offset is shorter than crossover distance; no head wave.")

    depth = layered.layer_depth(interface)
    horizontal_leg = depth * np.tan(theta_c)
    v_half = layered.velocities[interface + 1]
    interface_offset = offset - 2 * horizontal_leg
    if interface_offset < 0:
        # Numerical guard; should not happen for offset >= crossover
        interface_offset = 0.0

    points: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (horizontal_leg, depth),
        (horizontal_leg + interface_offset, depth),
        (offset, 0.0),
    ]
    return np.array(points)


def hodograph_samples(
    layered: LayeredVelocityModel,
    offsets: Sequence[float],
    interface: int = 0,
) -> dict:
    """
    Compute direct and head-wave travel times for plotting hodographs.
    """

    offsets = np.asarray(offsets, dtype=float)
    if np.any(offsets < 0.0):
        raise ValueError("offsets must be non-negative.")
    direct = offsets / layered.velocities[0]

    _, intercept, crossover = layered.head_wave_parameters(interface)
    head = np.full_like(offsets, np.nan, dtype=float)

    mask = offsets >= crossover
    if np.any(mask):
        v2 = layered.velocities[interface + 1]
        head[mask] = intercept + (offsets[mask] - crossover) / v2

    return {
        "offsets": offsets,
        "direct": direct,
        "head": head,
        "crossover": crossover,
        "intercept": intercept,
    }
