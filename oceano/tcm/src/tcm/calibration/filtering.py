"""
Channel filtering — per-channel despiking for calibration.

Extracted from ``tcm._dask_legacy.incl_calibr_hy.filter_channes``
(plotting parts moved to :mod:`tcm.calibration.visualization`).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from tcm import filters_scipy, utils2init

lf = utils2init.LoggingStyleAdapter(__name__)


def despike_channels(
    data_3d: np.ndarray,
    *,
    blocks: tuple[int, ...] = (21, 7),
    offsets: tuple[float, ...] = (1.5, 2),
    std_smooth_sigma: float = 4.0,
    x: Optional[Mapping[str, Any]] = None,
    y: Optional[Mapping[str, Any]] = None,
    z: Optional[Mapping[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Despike each channel (x, y, z) by forward+backward ``despike``.

    Parameters
    ----------
    data_3d
        Shape ``(3, N)`` — raw sensor data.
    blocks, offsets, std_smooth_sigma
        Default despike parameters.  Overridden per channel via *x/y/z*.
    x, y, z
        Per-channel overrides (``blocks``, ``offsets``, ``std_smooth_sigma``).

    Returns
    -------
    data_3d
        Shape ``(3, M)`` — input with spike rows removed (``M ≤ N``).
    mask_good
        Shape ``(N,)`` boolean — ``True`` for kept points.
    """
    n = data_3d.shape[1]
    mask_good = np.ones(n, dtype=bool)
    per_ch = [x, y, z]

    for i, (label, ch_cfg) in enumerate(zip('xyz', per_ch)):
        a = data_3d[i]
        mask_good &= ~np.isnan(a)

        if not len(offsets):
            continue

        ch_blocks = np.minimum(
            ch_cfg.get('blocks', blocks) if ch_cfg else blocks,
            mask_good.sum(),
        )
        ch_offsets = ch_cfg.get('offsets', offsets) if ch_cfg else offsets
        ch_sigma = ch_cfg.get('std_smooth_sigma', std_smooth_sigma) if ch_cfg else std_smooth_sigma

        cfg = dict(offsets=ch_offsets, blocks=ch_blocks, std_smooth_sigma=ch_sigma)

        # Forward pass (reversed order → despikes big spikes first)
        a_f = np.float64(a[mask_good][::-1])
        a_f, _ = filters_scipy.despike(a_f, **cfg)
        # Backward pass
        a_f, _ = filters_scipy.despike(a_f[::-1], **cfg)

        b_nan = np.isnan(a)
        b_nan[mask_good] = np.isnan(a_f)
        n_before = mask_good.sum()
        mask_good &= ~b_nan
        lf.info(
            "despike({:s}, offsets={}, blocks={}): deleted={:d}",
            label, ch_offsets, ch_blocks, n_before - mask_good.sum(),
        )

    return data_3d[:, mask_good], mask_good
