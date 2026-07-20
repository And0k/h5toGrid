"""
Fixtures for calibration tests — synthetic ellipsoid data with known geometry.

All fixtures use seeded ``np.random.default_rng`` for determinism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tcm.calibration.moments import fibonacci_sphere


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_sphere_pts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample *n* points uniformly on a unit sphere, shape ``(3, N)``."""
    raw = rng.normal(size=(3, n))
    return raw / np.linalg.norm(raw, axis=0, keepdims=True)


def apply_calibration(
    sphere: np.ndarray,
    gain: np.ndarray,
    bias: np.ndarray,
    *,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Produce raw data from a unit sphere via the inverse calibration.

    Satisfies ``gain @ (raw - bias) ≈ sphere`` when noise is zero.
    """
    inv_gain = np.linalg.inv(gain)
    pts = inv_gain @ sphere + bias
    if noise_sigma > 0:
        assert rng is not None
        pts += rng.normal(scale=noise_sigma, size=pts.shape)
    return pts


def to_dataset(
    pts: np.ndarray, channels: tuple[str, str, str], n: int,
) -> xr.Dataset:
    """Wrap ``(3, N)`` array into ``xr.Dataset`` over ``time``."""
    time = pd.date_range("2024-01-01", periods=n, freq="100ms")
    return xr.Dataset(
        {ch: ("time", pts[i]) for i, ch in enumerate(channels)},
        coords={"time": time},
    )


# --------------------------------------------------------------------------- #
# Known geometry dicts (for parametric tests)
# --------------------------------------------------------------------------- #

@pytest.fixture(
    params=[
        {"gain_diag": [1.2, 0.9, 1.0], "bias": [2.0, -1.5, 0.5]},
        {"gain_diag": [1.0, 1.0, 1.0], "bias": [0.0, 0.0, 0.0]},
        {"gain_diag": [0.8, 1.1, 1.3], "bias": [-3.0, 0.5, 1.0]},
    ],
    ids=["standard", "identity", "asymmetric"],
)
def known_geom(request) -> dict:
    """Parametrised gain/bias geometry dict."""
    p = request.param
    return {
        "gain": np.diag(p["gain_diag"]),
        "bias": np.array(p["bias"]).reshape(3, 1),
    }


# --------------------------------------------------------------------------- #
# Sample datasets
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_magnetometer_ds() -> xr.Dataset:
    """5000-sample noisy magnetometer data on a known ellipsoid."""
    rng = np.random.default_rng(seed=42)
    n = 5000
    gain = np.diag([1.2, 0.9, 1.0])
    bias = np.array([[2.0], [-1.5], [0.5]])
    sphere = make_sphere_pts(n, rng)
    pts = apply_calibration(sphere, gain, bias, noise_sigma=0.02, rng=rng)
    return to_dataset(pts, ("Mx", "My", "Mz"), n)


@pytest.fixture
def sample_accelerometer_ds() -> xr.Dataset:
    """3000-sample noisy accelerometer data — identity gain, zero bias."""
    rng = np.random.default_rng(seed=7)
    n = 3000
    sphere = make_sphere_pts(n, rng)
    pts = apply_calibration(sphere, np.eye(3), np.zeros((3, 1)), noise_sigma=0.01, rng=rng)
    return to_dataset(pts, ("Ax", "Ay", "Az"), n)


@pytest.fixture
def outlier_magnetometer_ds() -> xr.Dataset:
    """Sample magnetometer with ~5 % extreme outliers injected."""
    rng = np.random.default_rng(seed=99)
    n = 5000
    gain = np.diag([1.2, 0.9, 1.0])
    bias = np.array([[2.0], [-1.5], [0.5]])
    sphere = make_sphere_pts(n, rng)
    pts = apply_calibration(sphere, gain, bias, noise_sigma=0.02, rng=rng)
    # Inject outliers
    n_out = n // 20
    idx = rng.choice(n, n_out, replace=False)
    pts[:, idx] = rng.normal(scale=20.0, size=(3, n_out))
    return to_dataset(pts, ("Mx", "My", "Mz"), n)


# --------------------------------------------------------------------------- #
# Geometry generators (from calibration-distribute_files/conftest.py)
# --------------------------------------------------------------------------- #

def dense_sphere_grid(n_inclination: int = 800, n_azimuth: int = 1600) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint-rule lat/lon grid + dOmega weights on S^2, for numerically integrating a test function."""
    inclination = (np.arange(n_inclination) + .5) * np.pi / n_inclination     # cell-centered, avoid pole
    azimuth = (np.arange(n_azimuth) + .5) * 2 * np.pi / n_azimuth
    theta, phi = np.meshgrid(inclination, azimuth, indexing="ij")
    points = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).reshape(3, -1)
    d_omega = (np.sin(theta) * (np.pi / n_inclination) * (2 * np.pi / n_azimuth)).ravel()
    return points, d_omega


def clustered_sphere(n_uniform: int, n_cluster: int, cluster_center: np.ndarray, cluster_spread: float,
                      rng: np.random.Generator) -> np.ndarray:
    """n_uniform points spread over the whole sphere plus n_cluster crowded within `cluster_spread` of
    `cluster_center` — the adversarial case simulating a rotation protocol that spends most of its
    time in one orientation."""
    center = np.asarray(cluster_center) / np.linalg.norm(cluster_center)
    perturbed = center[:, np.newaxis] + cluster_spread * rng.normal(size=(3, n_cluster))
    return np.hstack([fibonacci_sphere(n_uniform), perturbed / np.linalg.norm(perturbed, axis=0)])


def multi_axis_spin_geometry(tilts_deg: np.ndarray, azimuths_deg: np.ndarray, n_per_spin: int = 1000,
                              reference: np.ndarray = np.array([0., 0., 1.])) -> np.ndarray:
    """
    The actual calibration protocol, literally: for each (tilt, azimuth) setting the device's own spin
    axis is held fixed while it completes a full spin — n_per_spin points tracing a circle of angular
    radius = tilt around that axis (since axis . reference = cos(tilt) by construction) — then the
    axis is repositioned to the next (tilt, azimuth) and the spin repeats. Different from
    `belt_geometry`: that traces full latitude rings around one fixed world axis; this traces many
    small circles scattered at len(tilts_deg) * len(azimuths_deg) discrete locations.

    :param tilts_deg: axis inclination(s) from `reference`, degrees, one spin per (tilt, azimuth) pair.
    :param azimuths_deg: axis azimuth(s) (e.g. compass bearing of the horizontal component), degrees.
    :param n_per_spin: samples per full spin.
    :param reference: the world-fixed vector each spin traces around its axis (gravity- or field-like).
    :return: (3, len(tilts_deg)*len(azimuths_deg)*n_per_spin).
    """
    psi = np.linspace(0, 2 * np.pi, n_per_spin, endpoint=False)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    columns = []
    for tilt in np.deg2rad(tilts_deg):
        for azimuth in np.deg2rad(azimuths_deg):
            axis = np.array([np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)])
            # Rodrigues' rotation-by-angle formula (not the single-pair form in orientation.rotate --
            # see calibration_wiki.md, "Test geometry generators"): reference rotated by psi around
            # axis, vectorized over all psi at once
            columns.append(reference[:, np.newaxis]*cos_psi + np.cross(axis, reference)[:, np.newaxis]*sin_psi
                            + axis[:, np.newaxis] * (axis @ reference) * (1 - cos_psi))
    return np.hstack(columns)


def belt_geometry(n_azimuth: int, tilt_deg: np.ndarray) -> np.ndarray:
    """
    Rotate-about-vertical-at-several-tilts protocol (continuous-azimuth variant): latitude "belts",
    not a uniform sphere — dense along each belt, empty between them, and increasingly crowded (in
    absolute terms) toward tilts near 0 deg/180 deg where a belt's true circumference shrinks but
    sample count per belt does not.
    """
    tilt = np.deg2rad(tilt_deg)
    azimuth = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)
    x0, z0 = np.sin(tilt), np.cos(tilt)                          # un-rotated point for each tilt, y0 = 0
    x = np.outer(x0, cos_az).ravel()
    y = np.outer(x0, sin_az).ravel()
    z = np.repeat(z0, n_azimuth)
    return np.stack([x, y, z])
