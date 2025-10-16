"""
Educational seismotomography toolkit.

This package bundles a lightweight forward model for straight–ray tomography,
utilities for layered-media travel-time curves, and helper routines for
gradient-based inversion.  It is intentionally scoped for interactive
visualisation inside the accompanying Streamlit application.
"""

from .geometry import AcquisitionGeometry
from .grid import LayeredVelocityModel, VelocityModel
from .inversion import (
    gradient_descent_update,
    misfit,
    project_to_bounds,
    predict_times_linear,
    sirt_update,
)
from .ray_tracing import (
    head_wave_travel_time,
    hodograph_samples,
    shoot_head_wave,
    straight_ray_path,
    travel_time_matrix,
)
from .synthetic import (
    SyntheticScenario,
    create_toy_scenario,
    create_geologic_scenario,
    create_homogeneous_scenario,
    create_gradient_scenario,
    generate_observations,
    simple_layered_start,
    gradient_velocity_field,
)

__all__ = [
    "AcquisitionGeometry",
    "LayeredVelocityModel",
    "VelocityModel",
    "gradient_descent_update",
    "sirt_update",
    "misfit",
    "project_to_bounds",
    "predict_times_linear",
    "head_wave_travel_time",
    "hodograph_samples",
    "shoot_head_wave",
    "straight_ray_path",
    "travel_time_matrix",
    "SyntheticScenario",
    "create_toy_scenario",
    "create_geologic_scenario",
    "create_homogeneous_scenario",
    "create_gradient_scenario",
    "generate_observations",
    "simple_layered_start",
    "gradient_velocity_field",
]
