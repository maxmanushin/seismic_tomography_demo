from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


def _ensure_array(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return array


@dataclass
class VelocityModel:
    """
    2D velocity grid used for straight-ray tomography.

    Coordinates follow the geophysical convention: `x` horizontal, `z` positive
    downward.  Velocity values are expressed in km/s by default, although units
    are arbitrary as long as the same scaling is used across the workflow.
    """

    x: np.ndarray
    z: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

        if self.x.ndim != 1 or self.z.ndim != 1:
            raise ValueError("x and z must be 1-D grids.")
        if self.velocity.shape != (self.z.size, self.x.size):
            raise ValueError(
                "velocity array shape must match (len(z), len(x)) grid dimensions."
            )
        if np.any(self.velocity <= 0.0):
            raise ValueError("velocity values must be strictly positive.")

    @property
    def slowness(self) -> np.ndarray:
        return 1.0 / self.velocity

    def copy(self) -> "VelocityModel":
        return VelocityModel(self.x.copy(), self.z.copy(), self.velocity.copy())

    def cell_sizes(self) -> Tuple[np.ndarray, np.ndarray]:
        dx = np.diff(self.x)
        dz = np.diff(self.z)
        return dx, dz

    def bilinear_interpolate(self, points: np.ndarray) -> np.ndarray:
        """
        Bilinear interpolation of velocities at arbitrary points.
        """

        x = self.x
        z = self.z
        vx = np.clip(np.searchsorted(x, points[:, 0]) - 1, 0, x.size - 2)
        vz = np.clip(np.searchsorted(z, points[:, 1]) - 1, 0, z.size - 2)

        x1 = x[vx]
        x2 = x[vx + 1]
        z1 = z[vz]
        z2 = z[vz + 1]

        tx = (points[:, 0] - x1) / (x2 - x1 + 1e-12)
        tz = (points[:, 1] - z1) / (z2 - z1 + 1e-12)

        v = self.velocity
        v11 = v[vz, vx]
        v12 = v[vz, vx + 1]
        v21 = v[vz + 1, vx]
        v22 = v[vz + 1, vx + 1]

        return (
            (1 - tx) * (1 - tz) * v11
            + tx * (1 - tz) * v12
            + (1 - tx) * tz * v21
            + tx * tz * v22
        )

    def clamp_velocities(self, v_min: float, v_max: float) -> None:
        np.clip(self.velocity, v_min, v_max, out=self.velocity)


@dataclass(frozen=True)
class LayeredVelocityModel:
    """
    1D layered model used for analytic travel-time curves.

    Parameters
    ----------
    velocities : sequence of floats
        Velocity of each layer, from top to bottom.
    thicknesses : sequence of floats
        Physical thickness of each layer, positive downward.  The last layer can
        be assigned np.inf to represent a half-space.  Both sequences must have
        the same length.
    """

    velocities: np.ndarray
    thicknesses: np.ndarray

    def __post_init__(self) -> None:
        velocities = _ensure_array("velocities", self.velocities)
        thicknesses = _ensure_array("thicknesses", self.thicknesses)
        if velocities.size != thicknesses.size:
            raise ValueError("velocities and thicknesses must have equal length.")
        if np.any(velocities <= 0.0):
            raise ValueError("velocities must be positive.")
        if np.any(thicknesses <= 0.0) and not np.isinf(thicknesses[-1]):
            raise ValueError("thicknesses must be positive; last layer may be inf.")
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "thicknesses", thicknesses)

    @property
    def n_layers(self) -> int:
        return int(self.velocities.size)

    @property
    def interface_depths(self) -> np.ndarray:
        return np.cumsum(self.thicknesses[:-1])

    def layer_depth(self, index: int) -> float:
        """
        Depth of the bottom interface of the given layer.
        """

        if index < 0 or index >= self.n_layers - 1:
            raise IndexError("layer index out of range for finite-thickness layers.")
        return float(self.interface_depths[index])

    def head_wave_parameters(self, interface: int) -> Tuple[float, float, float]:
        """
        Return (critical_angle, intercept_time, crossover_distance) for the
        interface between layers `interface` and `interface + 1`.
        """

        if interface < 0 or interface >= self.n_layers - 1:
            raise IndexError("interface must reference a finite-thickness layer.")

        v1 = self.velocities[interface]
        v2 = self.velocities[interface + 1]
        if v2 <= v1:
            raise ValueError("Head waves require v2 > v1 at the interface.")

        theta_c = np.arcsin(v1 / v2)
        depth = self.layer_depth(interface)
        intercept_time = 2 * depth * np.cos(theta_c) / v1
        crossover_distance = 2 * depth * np.tan(theta_c)
        return theta_c, intercept_time, crossover_distance

    def half_space_velocity(self) -> float:
        return float(self.velocities[-1])

    def depth_profile(self, sampling: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return sampled depth/velocity profile for plotting.
        """

        total_depth = np.sum(np.where(np.isfinite(self.thicknesses), self.thicknesses, 0))
        z = np.linspace(0.0, max(total_depth, 1.0), sampling)
        v = np.empty_like(z)
        current_depth = 0.0
        layer_idx = 0
        interfaces = list(self.interface_depths) + [np.inf]
        for i, depth in enumerate(z):
            while depth >= interfaces[layer_idx]:
                layer_idx = min(layer_idx + 1, self.n_layers - 1)
            v[i] = self.velocities[layer_idx]
        return z, v
