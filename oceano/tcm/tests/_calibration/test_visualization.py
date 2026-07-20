"""Smoke tests for calibration visualisation — verify no crash on render.

These exercise the exact code path that produces
``AttributeError: 'LineCollection' object has no attribute 'do_3d_projection'``
when ``make_axes_locatable`` is used on a 3-D axes.  Each test calls
``fig.savefig`` which triggers ``fig.draw(renderer)`` — the same path
as the real ``run_calibration`` pipeline.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless — must be before any pyplot import

from io import BytesIO
import numpy as np
import pytest

from tcm.calibration.calibrate import SensorCalibration
from tcm.calibration.moments import fibonacci_sphere
from _calibration.conftest import make_sphere_pts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _save(fig):
    """Render figure to a PNG buffer — triggers the full draw pipeline."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    buf.close()


def _make_raw_and_calibrated(n: int = 2000, seed: int = 42, field_magnitude: float = 1.0):
    """Return (raw3d, gain, bias) for a known ellipsoid with noise."""
    rng = np.random.default_rng(seed)
    gain = np.diag([1.2, 0.9, 1.0])
    bias = np.array([[2.0], [-1.5], [0.5]])
    sphere = make_sphere_pts(n, rng)
    # Convention: gain @ (raw - bias) ≈ field_magnitude * direction
    raw3d = np.linalg.inv(gain) @ (field_magnitude * sphere) + bias + rng.normal(scale=0.02, size=(3, n))
    return raw3d, gain, bias


# --------------------------------------------------------------------------- #
# calibrate_plot
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("projection", ["sphere", "mollweide"])
def test_calibrate_plot_renders(projection):
    """calibrate_plot must render and savefig without error for both projections."""
    from tcm.calibration.visualization import calibrate_plot

    raw3d, gain, bias = _make_raw_and_calibrated()
    fig = calibrate_plot(raw3d, gain, bias, projection=projection)
    assert fig is not None
    _save(fig)


@pytest.mark.parametrize(
    "projection,field_magnitude",
    [("sphere", 52000.0), ("mollweide", 9.81), ("sphere", 1.0)],
    ids=["sphere-mag52k", "mollweide-g9.81", "sphere-unit"],
)
def test_calibrate_plot_field_magnitude(projection, field_magnitude):
    """calibrate_plot must scale ellipsoid + sphere to field_magnitude."""
    from tcm.calibration.visualization import calibrate_plot

    raw3d, gain, bias = _make_raw_and_calibrated(field_magnitude=field_magnitude)
    fig = calibrate_plot(raw3d, gain, bias, projection=projection, field_magnitude=field_magnitude)
    assert fig is not None
    _save(fig)


@pytest.mark.parametrize("projection", ["sphere", "mollweide"])
def test_calibrate_plot_reuse(projection):
    """Reusing a figure across two channels must not crash (the original bug)."""
    from tcm.calibration.visualization import calibrate_plot

    raw3d1, gain, bias = _make_raw_and_calibrated(n=1500, seed=1)
    raw3d2, _, _ = _make_raw_and_calibrated(n=1500, seed=2)

    fig = calibrate_plot(raw3d1, gain, bias, projection=projection)
    assert fig is not None
    _save(fig)

    # Second call reuses the same figure
    fig = calibrate_plot(raw3d2, gain, bias, fig=fig, projection=projection)
    assert fig is not None
    _save(fig)


# --------------------------------------------------------------------------- #
# coverage_heatmap
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("projection", ["mollweide", "sphere"])
def test_coverage_heatmap_renders(projection):
    """coverage_heatmap (coverage + uncertainty two-subplot) must render."""
    from tcm.calibration.visualization import coverage_heatmap
    from tcm.calibration import robust

    rng = np.random.default_rng(seed=99)
    raw3d, gain, bias = _make_raw_and_calibrated(n=1000, seed=99)
    calibration = SensorCalibration(bias, gain)

    n_query = 100
    query_dirs = fibonacci_sphere(n_query)
    _, density = robust.coverage_at(raw3d, calibration, query_directions=query_dirs)

    # Build a minimal uncertainty dict matching robust.uncertainty_at output
    z_score = rng.normal(size=n_query)
    uncertainty = {
        "systematic_z_score": z_score,
        "jackknife_spread_rad": np.abs(rng.normal(scale=0.01, size=n_query)),
        "noise_floor": 0.005,
        "n_regions_used": 10,
    }

    sample_dirs = raw3d / np.linalg.norm(raw3d, axis=0)

    fig = coverage_heatmap(
        query_dirs, density,
        projection=projection,
        sample_directions=sample_dirs,
        uncertainty=uncertainty,
    )
    assert fig is not None
    _save(fig)


def test_coverage_heatmap_no_uncertainty():
    """coverage_heatmap must handle missing uncertainty gracefully."""
    from tcm.calibration.visualization import coverage_heatmap
    from tcm.calibration import robust

    raw3d, gain, bias = _make_raw_and_calibrated(n=500, seed=7)
    calibration = SensorCalibration(bias, gain)

    query_dirs = fibonacci_sphere(50)
    _, density = robust.coverage_at(raw3d, calibration, query_directions=query_dirs)

    fig = coverage_heatmap(query_dirs, density, projection="mollweide")
    assert fig is not None
    _save(fig)
