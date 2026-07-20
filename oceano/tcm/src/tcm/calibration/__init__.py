"""
xarray-native calibration pipeline.

Replaces ``tcm._dask_legacy.incl_calibr_hy`` with pure-numpy math
and ``xr.Dataset`` I/O.

Submodules
----------
* :mod:`run` — entry point
* :mod:`calibrate` — ellipsoid fitting (pure numpy kernels).
* :mod:`moments` — sample-weighting scheme for Li-Griffiths fitting.
* :mod:`spatial_binning` — 3-D bin averaging on the sphere (θ + φ).
* :mod:`filtering` — per-channel despiking.
* :mod:`pipeline` — full iterative fit → reject loop.
* :mod:`visualization` — 3-D ellipsoid / channel diagnostic plots.
* :mod:`orientation` — zero-tilt zeroing, heading reference, azimuth_shift.
"""
