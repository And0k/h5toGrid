"""Shared visualization utilities for calibration plots.

Layout constants, colorbar helpers, and projection utilities used by
:mod:`visualization` and :mod:`vis_coverage`.  Isolated here so that
both modules stay DRY without circular imports.

Adapted from the HydroGenerator ``calibration-distribute_files/visualization.py``
(TwoSlopeNorm pattern, Layout composition, ``_attach_external_colorbar``).

Matplotlib is imported **lazily inside functions** so that importing this
module at the top of :mod:`run` does not trigger the slow Qt5Agg backend
initialisation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# ── Layout constants — single source of truth for calibration plots ─────
CELL_W = 5.0     # single axes width  [inches]
CELL_H = 4.5     # single axes height [inches]
CBAR_W = 0.15    # colorbar width as fraction of axes width
CBAR_PAD = 0.02  # gap between axes edge and colorbar


def _figsize(
    ncols: int, nrows: int, *, extra_w: float = 1.5, extra_h: float = 1.0, cbar_cols: int = 0,
) -> Tuple[float, float]:
    """Compute figure size from grid dimensions.

    Parameters
    ----------
    ncols, nrows
        Data columns / rows in the grid.
    extra_w, extra_h
        Inches reserved for labels and margins.
    cbar_cols
        Colorbar columns already inside the GridSpec (added to width).
    """
    w = ncols * CELL_W + cbar_cols * CBAR_W + extra_w
    h = nrows * CELL_H + extra_h
    return (w, h)


def _attach_external_colorbar(ax, mappable, size: float = CBAR_W, pad: float = CBAR_PAD, **kwargs):
    """Attach a colorbar flush against *ax*.

    Uses :func:`make_axes_locatable` so the colorbar does not steal space
    from neighboring subplots.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    plt.colorbar(mappable, cax=cax, **kwargs)


def to_lonlat(vecs_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project ``(3, N)`` unit vectors to Mollweide-compatible longitude / latitude.

    Returns
    -------
    lon, lat
        Radians, ``lon ∈ (-π, π]``, ``lat ∈ [-π/2, π/2]``.
    """
    lon = np.arctan2(vecs_3d[1], vecs_3d[0])
    lat = np.arcsin(np.clip(vecs_3d[2], -1, 1))
    return lon, lat


def residual_colors(
    raw3d: np.ndarray, calibrated: np.ndarray, *, reference_magnitude: float = 1.0, cmap: str = "RdBu_r",
):
    """Signed residual metrics for raw + calibrated panels with shared diverging norm.

    Raw panel: ``mean(‖r‖) − ‖r‖`` — negative = farther from centre than average.
    Calibrated panel: ``reference_magnitude − ‖calibrated‖`` — negative = outside sphere.

    Both norms are :class:`TwoSlopeNorm` centered at 0 so that the
    diverging colormap is symmetric.  Returns a 5-tuple so callers can
    unpack concisely::

        raw_c, cal_c, raw_n, cal_n, cmap_name = residual_colors(raw, cal)
    """
    from matplotlib.colors import TwoSlopeNorm

    raw_norm = np.linalg.norm(raw3d, axis=0)
    cal_norm = np.linalg.norm(calibrated, axis=0)
    raw_signed = np.mean(raw_norm) - raw_norm
    cal_signed = reference_magnitude - cal_norm

    raw_emax = max(float(np.abs(raw_signed).max()), 1e-10)
    cal_emax = max(float(np.abs(cal_signed).max()), 1e-10)
    return (
        raw_signed,
        cal_signed,
        TwoSlopeNorm(vmin=-raw_emax, vcenter=0, vmax=raw_emax),
        TwoSlopeNorm(vmin=-cal_emax, vcenter=0, vmax=cal_emax),
        cmap,
    )
