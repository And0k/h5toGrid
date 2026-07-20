"""
3-D spatial bin averaging on the unit sphere.

Replaces ``tcm._dask_legacy.incl_calibr_hy.bin_avg_3d_partial`` —
bins by **both** inclination (θ) and azimuth (φ) via
``scipy.stats.binned_statistic_2d``, unlike the azimuth-only
``spatial_bin_avg`` previously in :mod:`tcm.calibration.calibrate`.

.. |legacy| replace:: ``incl_calibr_hy.bin_avg_3d_partial``
"""
from __future__ import annotations


import numpy as np
from scipy import stats

from tcm import utils2init

lf = utils2init.LoggingStyleAdapter(__name__)


# --------------------------------------------------------------------------- #
# Coordinate conversion
# --------------------------------------------------------------------------- #

def xyz2spherical(xyz: np.ndarray) -> np.ndarray:
    """Cartesian → spherical coordinates.

    Parameters
    ----------
    xyz
        Shape ``(3, N)`` — 3-D Cartesian points.

    Returns
    -------
    rtp
        Shape ``(3, N)`` — ``(radius, theta, phi)`` where:

        * *theta* is the polar angle from +Z (``0`` … ``π``)
        * *phi* is the azimuthal angle from +X (``-π`` … ``π``)
    """
    rtp = np.empty_like(xyz)
    xy = xyz[0] ** 2 + xyz[1] ** 2
    rtp[0] = np.sqrt(xy + xyz[2] ** 2)            # radius
    rtp[1] = np.arctan2(np.sqrt(xy), xyz[2])       # inclination from Z
    rtp[2] = np.arctan2(xyz[1], xyz[0])            # azimuth
    return rtp


def spherical2xyz(rtp: np.ndarray) -> np.ndarray:
    """Spherical → Cartesian coordinates.

    Parameters
    ----------
    rtp
        Shape ``(3, N)`` — ``(radius, theta, phi)``.

    Returns
    -------
    xyz
        Shape ``(3, N)``.
    """
    r, t, p = rtp
    st = np.sin(t)
    return np.array([
        r * st * np.cos(p),
        r * st * np.sin(p),
        r * np.cos(t),
    ])


# --------------------------------------------------------------------------- #
# 2-D bin averaging on the sphere
# --------------------------------------------------------------------------- #

def bin_avg_3d(
    data_3d: np.ndarray,
    n_bins: int = 200,
    *,
    range_phi: tuple[float, float] = (0, 2 * np.pi),
    range_cosphi: tuple[float, float] = (-1, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce non-uniform point distribution by averaging in spherical bins.

    Each point is converted to spherical coordinates ``(r, θ, φ)``, then
    binned on ``(φ, cos θ)`` with :func:`scipy.stats.binned_statistic_2d`.
    Where multiple points fall in the same bin they are replaced by the
    bin mean.  This preserves spatial coverage without over-weighting
    over-sampled orientations.

    Parameters
    ----------
    data_3d
        Shape ``(3, N)`` — raw sensor data (3 channels × time).
    n_bins
        Number of bins *per dimension* (total 2-D grid = ``n_bins²``).
        Default 200 matches the legacy ``bin_avg_3d_partial`` default.
    range_phi
        ``(min, max)`` for the azimuthal angle φ.
    range_cosphi
        ``(min, max)`` for ``cos(θ)``; the default ``(-1, 1)`` covers the
        full sphere.  Restricting this range (e.g. ``(-0.8, 0.8)``) can
        help when data never reaches the poles.

    Returns
    -------
    centers
        Shape ``(3, M)`` — averaged bin centers (M ≤ *n_bins*², non-empty
        bins only).  These are the *averages* of all points inside each bin,
        not bin midpoints.
    counts
        Shape ``(M,)`` — number of original points in each bin (≥ 1).
    raw3d_other
        Shape ``(3, K)`` — *all* original points that were replaced by
        averaging (i.e. points in bins with count > 1).  Useful for
        visualisation.
    n_cell
        Shape ``(M,)`` — same as *counts*, but includes all bins (including
        those with count = 1).  Legacy API compatibility.
    """
    if data_3d.shape[1] == 0:
        return np.empty((3, 0)), np.empty(0, dtype=int), np.empty((3, 0)), np.empty(0, dtype=int)

    rtp = xyz2spherical(data_3d)  # (radius, theta, phi)

    # Recommended minimum bins to get ~1 point / 2-D cell in near-uniform data
    bins_min = int(np.sqrt(rtp.shape[1]))
    if n_bins > bins_min:
        lf.info(
            "n_bins=%d > sqrt(N)=%d — many bins will be empty",
            n_bins, bins_min,
        )

    # Bin statistic: φ on x-axis, cos θ on y-axis, radius as the value
    bin_stat = stats.binned_statistic_2d(
        rtp[2],                    # φ
        np.cos(rtp[1]),            # cos θ  → uniform du = sin(θ)dθ
        rtp[0],                    # radius (averand)
        bins=n_bins,
        range=[range_phi, range_cosphi],
    )

    # Sort by bin number to group co-located points
    i_sort = np.argsort(bin_stat.binnumber, kind='stable')
    binnum_sorted = bin_stat.binnumber[i_sort]
    data_sorted = data_3d[:, i_sort]

    # Find run boundaries (where bin number changes)
    edge = np.flatnonzero(np.diff(binnum_sorted)) + 1
    bounds = np.column_stack(
        [np.concatenate([[0], edge]),
         np.concatenate([edge, [len(i_sort)]])]
    )
    n_in_cell = np.diff(bounds, axis=1).ravel()

    # Identify bins with > 1 point
    b_multi = n_in_cell > 1

    # All original points that get averaged away
    multi_idx = np.concatenate(
        [np.arange(st, en) for (st, en), multi in zip(bounds, b_multi) if multi]
    )
    raw3d_other = data_sorted[:, multi_idx]

    # Build output: mean per multi-point bin, lone point otherwise
    centers = np.column_stack([
        data_sorted[:, st:en].mean(axis=1) if multi else data_sorted[:, st]
        for (st, en), multi in zip(bounds, b_multi)
    ])
    # counts = n_in_cell (all bins represented)
    return centers, n_in_cell, raw3d_other, n_in_cell


# --------------------------------------------------------------------------- #
# Simple 2-D bin centres (for quick visualisation, no averaging)
# --------------------------------------------------------------------------- #

def bin_centers_2d(
    data_3d: np.ndarray,
    n_bins: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin-average by azimuth only (1-D binning, legacy compatibility).

    .. deprecated::
        Use :func:`bin_avg_3d` which bins in both θ and φ.

    This is the old ``spatial_bin_avg`` logic — azimuth-only.  Kept for
    backward compatibility but **not** recommended for calibration.
    """
    az = np.arctan2(data_3d[1], data_3d[0])
    bin_idx = np.digitize(az, np.linspace(-np.pi, np.pi, n_bins + 1)) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    centers = np.zeros((3, n_bins))
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = bin_idx == i
        counts[i] = mask.sum()
        if counts[i]:
            centers[:, i] = data_3d[:, mask].mean(axis=1)
    return centers, counts