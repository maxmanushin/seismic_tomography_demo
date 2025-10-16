from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .geometry import AcquisitionGeometry
from .grid import VelocityModel
from .ray_tracing import travel_time_matrix


@dataclass
class SyntheticScenario:
    """
    Collection of helper objects for a toy tomography inversion.
    """

    true_model: VelocityModel
    starting_model: VelocityModel
    geometry: AcquisitionGeometry
    observed_travel_times: np.ndarray
    metadata: Dict[str, object] | None = None


def gaussian_anomaly(
    x: np.ndarray,
    z: np.ndarray,
    center: Tuple[float, float],
    amplitude: float,
    sigma_x: float,
    sigma_z: float,
) -> np.ndarray:
    """
    Generate a 2D Gaussian anomaly on the provided meshgrid.
    """

    gx, gz = np.meshgrid(x, z)
    cx, cz = center
    return amplitude * np.exp(
        -((gx - cx) ** 2) / (2 * sigma_x**2) - ((gz - cz) ** 2) / (2 * sigma_z**2)
    )


def layered_velocity_field(
    x: np.ndarray,
    z: np.ndarray,
    velocities: Tuple[float, ...],
    interfaces: Tuple[np.ndarray, ...],
    roughness: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Construct a geologically plausible layered model with floating interfaces.

    Parameters
    ----------
    velocities:
        Velocity assigned to each layer (top to bottom).
    interfaces:
        Sequence of functions z(x) sampled on the same x-grid and representing
        depth of the interface between layers. The length must be len(velocities) - 1.
    roughness:
        Standard deviation of small-scale random perturbations (adds heterogeneity).
    seed:
        Random seed for reproducible roughness.
    """

    if len(velocities) - 1 != len(interfaces):
        raise ValueError("interfaces must have len(velocities) - 1 entries.")

    nz, nx = len(z), len(x)
    field = np.empty((nz, nx), dtype=float)
    layer_indices = np.zeros((nz, nx), dtype=int)

    z_grid = z[:, np.newaxis]
    for ix in range(nx):
        depth = z_grid[:, 0]
        boundaries = [interfaces[j][ix] for j in range(len(interfaces))]
        boundaries.append(np.inf)
        current_layer = np.zeros(nz, dtype=int)
        start_depth = 0.0
        layer = 0
        for boundary in boundaries:
            mask = (depth >= start_depth) & (depth < boundary)
            current_layer[mask] = layer
            start_depth = boundary
            layer += 1
        layer_indices[:, ix] = current_layer

    for layer, v in enumerate(velocities):
        field[layer_indices == layer] = v

    if roughness > 0.0:
        rng = np.random.default_rng(seed)
        perturbation = rng.normal(scale=roughness, size=field.shape)
        field += perturbation

    return field


def gradient_velocity_field(
    x: np.ndarray,
    z: np.ndarray,
    v0: float,
    gz: float,
    gx: float = 0.0,
) -> np.ndarray:
    """
    Continuous gradient model: v(x, z) = v0 + gz*z + gx*(x - mean(x)).

    Parameters
    ----------
    v0 : float
        Base velocity at surface and central x.
    gz : float
        Vertical gradient (m/s per metre, positive increases with depth).
    gx : float
        Lateral gradient (m/s per metre); defaults to 0.
    """

    X, Z = np.meshgrid(x, z)
    x0 = float(np.mean(x))
    V = v0 + gz * Z + gx * (X - x0)
    return V


def simple_layered_start(
    x: np.ndarray,
    z: np.ndarray,
    velocities: Tuple[float, ...],
    interface_depths: Tuple[float, ...],
) -> np.ndarray:
    """
    Build a horizontally layered starting model with flat interfaces.
    """

    if len(velocities) - 1 != len(interface_depths):
        raise ValueError("interface_depths must have len(velocities) - 1 entries.")

    field = np.empty((len(z), len(x)), dtype=float)
    boundaries = list(interface_depths) + [np.inf]
    layer = 0
    for depth_upper, depth_lower in zip((0.0, *interface_depths), boundaries):
        mask = (z >= depth_upper) & (z < depth_lower)
        field[mask, :] = velocities[layer]
        layer += 1
    return field


def generate_observations(
    model: VelocityModel,
    geometry: AcquisitionGeometry,
    noise_level: float = 0.0,
    random_state: int | None = 42,
) -> np.ndarray:
    """
    Compute synthetic observations (travel times) with optional Gaussian noise.
    """

    predicted = travel_time_matrix(model, geometry)
    if noise_level <= 0:
        return predicted
    rng = np.random.default_rng(random_state)
    noise = rng.normal(scale=noise_level, size=predicted.shape)
    return predicted + noise


def create_geologic_scenario(
    random_state: int | None = 7,
    noise_level: float = 0.003,
    gradient_v0: float = 1800.0,
    gradient_gz: float = 0.5,
    gradient_gx: float = 0.0,
) -> SyntheticScenario:
    """
    Build a scenario with gently undulating layers and localised anomalies.
    """

    rng = np.random.default_rng(random_state)

    nx, nz = 80, 50
    x = np.linspace(0.0, 4000.0, nx)
    z = np.linspace(0.0, 1800.0, nz)

    # Floating interfaces generated as smooth sinusoids + random component
    interfaces = []
    interface_depths = [500.0, 1100.0]
    for depth in interface_depths:
        bumps = 80.0 * np.sin(2 * np.pi * x / 2500.0)
        bumps += 40.0 * np.sin(2 * np.pi * x / 800.0 + rng.uniform(0, np.pi))
        perturb = rng.normal(scale=30.0, size=x.shape)
        interfaces.append(depth + bumps + perturb)

    # Combine lithology (offsets by layer) with continuous gradient background
    # Lithologies (surface velocities): sand, clay, limestone
    litho_surface_velocities = (1800.0, 2400.0, 3400.0)
    gradient_field = gradient_velocity_field(
        x, z, v0=gradient_v0, gz=gradient_gz, gx=gradient_gx
    )
    # Offsets relative to gradient surface velocity v0
    offsets = tuple(v - gradient_v0 for v in litho_surface_velocities)
    offset_field = layered_velocity_field(
        x,
        z,
        velocities=offsets,
        interfaces=tuple(interfaces),
        roughness=15.0,
        seed=random_state,
    )
    true_velocity = gradient_field + offset_field

    # Insert local anomalies to highlight heterogeneity
    anomaly_centers = [(1200.0, 600.0), (2600.0, 1300.0)]
    for (cx, cz), amp in zip(anomaly_centers, (250.0, -180.0)):
        true_velocity += gaussian_anomaly(
            x,
            z,
            center=(cx, cz),
            amplitude=amp,
            sigma_x=450.0,
            sigma_z=250.0,
        )

    # Gradient start (simpler than truth to promote learning)
    start_v0 = gradient_v0 - 50.0
    start_gz = gradient_gz * 0.6
    start_gx = gradient_gx * 0.3
    starting_velocity = gradient_velocity_field(x, z, v0=start_v0, gz=start_gz, gx=start_gx)

    geometry = AcquisitionGeometry.surface_spread(
        n_sources=14, n_receivers=14, spread_length=x.max()
    )

    true_model = VelocityModel(x, z, true_velocity)
    start_model = VelocityModel(x, z, starting_velocity)
    observations = generate_observations(true_model, geometry, noise_level=noise_level)

    return SyntheticScenario(
        true_model=true_model,
        starting_model=start_model,
        geometry=geometry,
        observed_travel_times=observations,
        metadata={
            "description": "Layered lithologies over continuous gradient with gaussian anomalies.",
            "layer_count": len(litho_surface_velocities),
            "start_interface_depths": tuple(interface_depths),
            "lithologies": ("sand", "clay", "limestone"),
            "litho_surface_velocities": litho_surface_velocities,
            "gradient": {"v0": float(gradient_v0), "gz": float(gradient_gz), "gx": float(gradient_gx)},
            "start_model_type": "gradient",
            "start_gradient": {"v0": float(start_v0), "gz": float(start_gz), "gx": float(start_gx)},
            # Provide interface polylines for overlay in the UI
            "interfaces_x": x,
            "interfaces_z": tuple(interfaces),
        },
    )


def create_homogeneous_scenario(
    noise_level: float = 0.005,
    random_state: int | None = 42,
) -> SyntheticScenario:
    """
    Retain the original simple Gaussian anomaly case for comparison.
    """

    rng = np.random.default_rng(random_state)

    nx, nz = 60, 40
    x = np.linspace(0.0, 3000.0, nx)
    z = np.linspace(0.0, 1500.0, nz)

    interface_depths = (400.0, 900.0, 1350.0)
    layer_count = len(interface_depths) + 1
    true_layer_velocities = tuple(
        float(v) for v in np.sort(rng.uniform(1700.0, 3400.0, size=layer_count))
    )
    true_velocity = simple_layered_start(
        x,
        z,
        velocities=true_layer_velocities,
        interface_depths=interface_depths,
    )

    start_layer_velocities = tuple(float(v) for v in np.linspace(1800.0, 3000.0, layer_count))
    starting_velocity = simple_layered_start(
        x,
        z,
        velocities=start_layer_velocities,
        interface_depths=interface_depths,
    )

    geometry = AcquisitionGeometry.surface_spread(
        n_sources=12, n_receivers=12, spread_length=x.max()
    )

    true_model = VelocityModel(x, z, true_velocity)
    start_model = VelocityModel(x, z, starting_velocity)
    observations = generate_observations(true_model, geometry, noise_level=noise_level, random_state=random_state)

    return SyntheticScenario(
        true_model=true_model,
        starting_model=start_model,
        geometry=geometry,
        observed_travel_times=observations,
        metadata={
            "description": "Horizontally layered random velocity model.",
            "layer_count": layer_count,
            "start_interface_depths": interface_depths,
            "start_layer_velocities": start_layer_velocities,
            "true_layer_velocities": true_layer_velocities,
        },
    )


def create_gradient_scenario(
    v0: float = 1800.0,
    gz: float = 0.6,
    gx: float = 0.0,
    noise_level: float = 0.003,
    random_state: int | None = 13,
) -> SyntheticScenario:
    """
    Scenario with a simple continuous velocity gradient to encourage stable updates.

    True model uses the specified gradient; starting model uses a milder gradient
    to guarantee non-zero residuals.
    """

    nx, nz = 80, 50
    x = np.linspace(0.0, 4000.0, nx)
    z = np.linspace(0.0, 1800.0, nz)

    true_v = gradient_velocity_field(x, z, v0=v0, gz=gz, gx=gx)
    # Start with under-estimated gradient
    start_v = gradient_velocity_field(x, z, v0=v0 - 50.0, gz=gz * 0.6, gx=gx * 0.3)

    geometry = AcquisitionGeometry.surface_spread(
        n_sources=14, n_receivers=14, spread_length=x.max()
    )

    true_model = VelocityModel(x, z, true_v)
    start_model = VelocityModel(x, z, start_v)
    observations = generate_observations(true_model, geometry, noise_level=noise_level, random_state=random_state)

    return SyntheticScenario(
        true_model=true_model,
        starting_model=start_model,
        geometry=geometry,
        observed_travel_times=observations,
        metadata={
            "description": "Continuous gradient velocity model.",
            "start_model_type": "gradient",
            "start_gradient": {"v0": float(v0 - 50.0), "gz": float(gz * 0.6), "gx": float(gx * 0.3)},
            "true_gradient": {"v0": float(v0), "gz": float(gz), "gx": float(gx)},
        },
    )


def create_toy_scenario(kind: str = "geologic", **kwargs) -> SyntheticScenario:
    """
    Dispatch factory for different educational scenarios.

    Parameters
    ----------
    kind:
        Either "geologic" (undulating layers with heterogeneity) or "simple"
        (original Gaussian anomaly model).
    """

    kind = kind.lower()
    if kind == "geologic":
        return create_geologic_scenario(**kwargs)
    if kind == "simple":
        return create_homogeneous_scenario(**kwargs)
    if kind == "gradient":
        return create_gradient_scenario(**kwargs)
    raise ValueError("Unknown scenario kind. Use 'geologic' or 'simple'.")
