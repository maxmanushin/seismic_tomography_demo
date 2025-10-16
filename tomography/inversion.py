from __future__ import annotations

from typing import Tuple

import numpy as np

from .geometry import AcquisitionGeometry
from .grid import VelocityModel
from .ray_tracing import straight_ray_path


def misfit(predicted: np.ndarray, observed: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Return (objective_value, residual) for half-squared L2 misfit.
    """

    residual = predicted - observed
    value = 0.5 * np.mean(residual**2)
    return float(value), residual


def _ray_lengths(
    model: VelocityModel,
    geometry: AcquisitionGeometry,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble ray segment lengths per cell and predicted travel times.
    """

    n_rays = geometry.n_sources * geometry.n_receivers
    nz, nx = model.velocity.shape
    lengths = np.zeros((n_rays, nz, nx), dtype=float)
    predicted = np.zeros(n_rays, dtype=float)

    for idx, (src, rec) in enumerate(geometry.ray_pairs()):
        path = straight_ray_path(src, rec, n_samples=n_samples)
        segments = path[1:] - path[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)
        midpoints = 0.5 * (path[1:] + path[:-1])

        ix = np.clip(np.searchsorted(model.x, midpoints[:, 0]) - 1, 0, nx - 1)
        iz = np.clip(np.searchsorted(model.z, midpoints[:, 1]) - 1, 0, nz - 1)
        for l, i_x, i_z in zip(seg_lengths, ix, iz, strict=False):
            lengths[idx, i_z, i_x] += l

        velocities = model.bilinear_interpolate(midpoints)
        predicted[idx] = np.sum(seg_lengths / velocities)

    return lengths, predicted


def gradient_descent_update(
    model: VelocityModel,
    geometry: AcquisitionGeometry,
    observed: np.ndarray,
    step_size: float,
    n_samples: int = 200,
    tikhonov_lambda: float = 0.0,
    reference_velocity: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a single gradient-descent velocity update with scaled step length.

    Parameters
    ----------
    model:
        Current velocity model.
    geometry:
        Acquisition geometry with sources/receivers.
    observed:
        Observed travel times, shaped (n_sources, n_receivers).
    step_size:
        Gradient-descent step size (scales the velocity update).
    n_samples:
        Number of interpolation samples along each straight ray.
    tikhonov_lambda:
        Weight of zeroth-order Tikhonov regularisation (L2 towards reference).
    reference_velocity:
        Reference field for the Tikhonov penalty. Defaults to the current model
        if not supplied, which keeps the penalty inactive.

    Returns
    -------
    updated_velocity, predicted_times, gradient
        Updated velocity model, predicted travel times reshaped to
        (n_sources, n_receivers), and the gradient in velocity units
        (useful for visualisation). The actual update is scaled so that
        the maximum absolute change equals the provided `step_size`.
    """

    lengths, predicted = _ray_lengths(model, geometry, n_samples=n_samples)
    residual = predicted - observed.ravel()
    nz, nx = model.velocity.shape

    inv_vsq = 1.0 / (model.velocity**2)
    gradient = np.zeros_like(model.velocity)

    for ray_idx in range(lengths.shape[0]):
        gradient -= residual[ray_idx] * lengths[ray_idx] * inv_vsq

    if tikhonov_lambda > 0.0:
        if reference_velocity is None:
            reference_velocity = model.velocity
        gradient += tikhonov_lambda * (model.velocity - reference_velocity)

    max_abs = np.max(np.abs(gradient))
    if max_abs < 1e-12:
        delta = np.zeros_like(gradient)
    else:
        delta = step_size * gradient / max_abs

    updated_velocity = model.velocity - delta
    updated_velocity = np.clip(updated_velocity, 1e-3, None)

    return updated_velocity, predicted.reshape(observed.shape), gradient


def project_to_bounds(
    velocity: np.ndarray, vmin: float, vmax: float
) -> np.ndarray:
    """
    Clip the velocity field to physically plausible bounds.
    """

    if vmin <= 0 or vmax <= 0:
        raise ValueError("Velocity bounds must be positive.")
    if vmin > vmax:
        raise ValueError("vmin must not exceed vmax.")
    return np.clip(velocity, vmin, vmax)


def _predict_linear_from_lengths(lengths: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """
    Predict travel times using linear slowness model: T = G · s, s = 1/v.

    lengths: (n_rays, nz, nx)
    velocity: (nz, nx)
    returns: (n_rays,)
    """

    slowness = 1.0 / velocity
    return lengths.reshape(lengths.shape[0], -1) @ slowness.ravel()


def predict_times_linear(
    model: VelocityModel, geometry: AcquisitionGeometry, n_samples: int = 200
) -> np.ndarray:
    """
    Predict travel times via linear slowness approximation consistent with SIRT/ART.
    Shape: (n_sources, n_receivers)
    """

    lengths, _ = _ray_lengths(model, geometry, n_samples=n_samples)
    t = _predict_linear_from_lengths(lengths, model.velocity)
    return t.reshape(geometry.n_sources, geometry.n_receivers)


def _laplacian(field: np.ndarray) -> np.ndarray:
    """
    5-point discrete Laplacian with Neumann-like boundary handling.
    """

    zpad = np.pad(field, ((1, 1), (1, 1)), mode="edge")
    center = zpad[1:-1, 1:-1]
    lap = (
        zpad[1:-1, 0:-2]
        + zpad[1:-1, 2:]
        + zpad[0:-2, 1:-1]
        + zpad[2:, 1:-1]
        - 4.0 * center
    )
    return lap


def sirt_update(
    model: VelocityModel,
    geometry: AcquisitionGeometry,
    observed: np.ndarray,
    step_size: float,
    n_samples: int = 200,
    tikhonov_lambda: float = 0.0,
    smooth_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One SIRT-like update in slowness with diagonal preconditioning and optional smoothing.

    - Uses linear forward: T = G · s, with s = 1/v.
    - Gradient in slowness: g_s = G^T r, r = (G s - tobs).
    - Precondition by diag(G^T 1) to account for ray coverage.
    - Add zeroth-order Tikhonov (towards current s) and Laplacian smoothing.

    Returns updated velocity field, predicted times and gradient (in velocity units) for display.
    """

    lengths, _pred = _ray_lengths(model, geometry, n_samples=n_samples)
    s = 1.0 / model.velocity
    G = lengths.reshape(lengths.shape[0], -1)
    r = (G @ s.ravel() - observed.ravel())
    gs = (G.T @ r).reshape(model.velocity.shape)

    # Preconditioner diag(G^T 1)
    coverage = (G.T @ np.ones_like(r)).reshape(model.velocity.shape)
    coverage = np.where(coverage > 0, coverage, 1.0)
    gs /= coverage

    if tikhonov_lambda > 0.0:
        gs += tikhonov_lambda * (s - s)  # no external reference; keep form consistency

    if smooth_weight > 0.0:
        gs += smooth_weight * _laplacian(s)

    # Convert slowness-gradient to velocity-gradient for visualisation and bounded step in m/s
    gv = -gs / (model.velocity**2)
    max_abs = np.max(np.abs(gv))
    if max_abs < 1e-12:
        delta_v = np.zeros_like(gv)
    else:
        delta_v = step_size * gv / max_abs

    v_new = model.velocity - delta_v
    v_new = np.clip(v_new, 1e-3, None)

    # Predicted consistent with linear forward
    t_pred = _predict_linear_from_lengths(lengths, v_new).reshape(observed.shape)
    return v_new, t_pred, gv
