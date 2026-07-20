"""
Calibration pipeline — the full bin → fit → reject loop.

Replaces the inline loop from ``incl_calibr_hy.main`` (lines 850-883)
and the trivial ``iterative_calibrate`` wrapper that was in
:mod:`tcm.calibration.calibrate`.

Design
------
* **Pure numpy** — no xarray, no dask, no matplotlib.
* One public function: :func:`calibrate_pipeline`.
* Progressive distance-threshold rejection (legacy strategy) with
  configurable parameters.
* Optional callbacks for logging / visualisation so callers can
  plug in whatever they need without this module depending on
  matplotlib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from tcm import utils2init
from tcm.calibration.calibrate import calibrate_channel, coef2str, SensorCalibration
from tcm.calibration import robust  # calibrate,

lf = utils2init.LoggingStyleAdapter(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class PipelineConfig:
    """Parameters for the calibration pipeline.

    This is the **single source of truth** for all pipeline parameters.
    :mod:`tcm.config` re-exports it as ``ConfigProcCalib`` for Hydra's
    ConfigStore — no separate duplicate dataclass.

    Attributes
    ----------
    robust
        Use :func:`robust.autocalibrate` (default) — iterative
        MAD-based outlier rejection tied to the calibration goal.
        ``False`` → legacy progressive distance-threshold rejection.
    field_magnitude
        Known reference magnitude (IGRF total field for M,
        standard gravity for A).  ``1.0`` = unit sphere.
        Per-channel override via freeform ``+proc.field_magnitudes.M=52000``
        (see :func:`run_calibration`).
    weighted
        Use weighted Li-Griffiths fit (corrects uneven angular
        coverage).  Only effective when ``robust=True``.
    mad_threshold
        MAD multiplier for outlier rejection.  4 ≈ 2.7σ for
        Gaussian data.  Only effective when ``robust=True``.
    max_iterations
        Upper bound on fit/reject/refit cycles in autocalibrate.
    calibration_projection
        ``"sphere"`` (default) or ``"mollweide"`` for the
        two-panel calibration ellipsoid / unit-sphere plot.
    coverage_projection
        ``"mollweide"`` or ``"sphere"`` for coverage + uncertainty plot.
    dist_*
        Legacy progressive-rejection schedule (``robust=False``).
    """
    robust: bool = True
    field_magnitude: float = 1.0
    weighted: bool = True
    mad_threshold: float = 4.0
    max_iterations: int = 5
    calibration_projection: str = "sphere"
    coverage_projection: str = "mollweide"
    # Legacy progressive-rejection (robust=False)
    dist_outliers_max_pct: float = 10.0
    dist_check_range: list[float] = field(default_factory=lambda: [0.07, 0.3])
    dist_check_steps: int = 3


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationResult:
    """Output of :func:`calibrate_pipeline`.

    Attributes
    ----------
    gain
        ``(3, 3)`` calibration gain matrix.
    bias
        ``(3, 1)`` calibration bias offset (raw-space center).
    inlier_mask
        Shape ``(N,)`` boolean — ``True`` for points accepted after
        progressive rejection.
    calibration
        :class:`SensorCalibration` NamedTuple ``(bias, a2d)``
        for use with :mod:`tcm.calibration.robust` and
        :mod:`tcm.calibration.orientation`.
    n_inliers
        Number of inlier points after rejection.
    n_outliers
        Number of rejected outlier points.
    residual_median
        Median absolute radial residual on inliers.
    residual_p95
        95th percentile absolute radial residual on inliers.
    """
    gain: np.ndarray
    bias: np.ndarray
    inlier_mask: np.ndarray
    calibration: SensorCalibration = field(repr=False, default=None)
    n_inliers: int = 0
    n_outliers: int = 0
    residual_median: float = 0.0
    residual_p95: float = 0.0


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def calibrate_pipeline(
    data_3d: np.ndarray,
    cfg: PipelineConfig = PipelineConfig(),
    *,
    on_iter: Optional[Callable[[int, float, float, np.ndarray, np.ndarray], None]] = None,
) -> CalibrationResult:
    """Run calibration with robust outlier rejection (default) or legacy progressive schedule.

    Parameters
    ----------
    data_3d : ``(3, N)`` raw sensor data (after channel filtering).
    cfg : Pipeline parameters.
    on_iter : optional callback ``(step_idx, ...)`` for legacy mode only.

    Returns
    -------
    CalibrationResult
    """
    n_pts = data_3d.shape[1]

    if cfg.robust:
        # ── Robust path: autocalibrate with MAD-based outlier rejection ──
        calibration, history = robust.autocalibrate(
            data_3d, cfg.field_magnitude,
            max_iterations=cfg.max_iterations,
            mad_threshold=cfg.mad_threshold,
            weighted=cfg.weighted,
        )
        inlier = robust.reject_outliers(
            data_3d, calibration, cfg.field_magnitude, mad_threshold=cfg.mad_threshold
        )
        gain, bias = calibration.a2d, calibration.bias
    else:
        # ── Legacy path: progressive distance-threshold rejection ────────
        inlier = np.ones(n_pts, dtype=bool)
        dist_check = np.linspace(*np.sqrt(cfg.dist_check_range), cfg.dist_check_steps) ** 2
        gain, bias = np.eye(3), np.zeros((3, 1))
        calibration = SensorCalibration(bias, gain)

        for i_step, dc in enumerate(dist_check):
            gain, bias = calibrate_channel(data_3d[:, inlier])
            calibration = SensorCalibration(bias, gain)
            calibrated = gain @ (data_3d[:, inlier] - bias)
            dist = np.abs(1.0 - np.linalg.norm(calibrated, axis=0))
            new_inlier = dist < dc
            outliers_pct = (inlier.sum() - new_inlier.sum()) * 100.0 / max(inlier.sum(), 1)

            if on_iter is not None:
                on_iter(i_step, dc, outliers_pct, gain, bias)

            if outliers_pct > cfg.dist_outliers_max_pct:
                lf.debug("dist {:.4f}: {:.1f}% outliers — too many, using previous fit", dc, outliers_pct)
                break
            lf.debug("dist {:.4f}: {:.1f}% outliers", dc, outliers_pct)
            inlier[inlier] = new_inlier

    # ── Quality metrics ───────────────────────────────────────────────────
    n_inliers = inlier.sum().item()
    n_outliers = n_pts - n_inliers
    residual = np.abs(robust.radial_residuals(data_3d[:, inlier], calibration, cfg.field_magnitude))
    res_med = np.median(residual).item()
    res_p95 = np.quantile(residual, 0.95).item()

    lf.info(
        "calibrated: {}/{} inliers ({:.1f}% outliers) residual range=[{:.4g}, {:.4g}]",
        n_inliers, n_pts, 100 * n_outliers / max(n_pts, 1), res_med, res_p95,
    )
    log_coefs(gain, bias, msg="Calibration result")

    return CalibrationResult(
        gain=gain,
        bias=bias,
        inlier_mask=inlier,
        calibration=calibration,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        residual_median=res_med,
        residual_p95=res_p95,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def log_coefs(gain: np.ndarray, bias: np.ndarray, *, msg: str = "Calibration coefficients") -> None:
    """Log gain/bias in human-readable form."""
    a_str, b_str = coef2str(gain, bias)
    lf.info("{:s}:\nA = \n{:s}\nb = \n{:s}", msg, a_str, b_str)
