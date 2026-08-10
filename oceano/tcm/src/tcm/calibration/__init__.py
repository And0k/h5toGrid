"""
xarray-native calibration pipeline: turns raw multi-position sensor recordings into a bias + gain
("ellipsoid") calibration, with outlier rejection, uncertainty diagnostics, and orientation alignment.
For a first read, ``run.run_calibration`` is the entry point that ties every submodule below together;
``calibrate.calibrate`` is the core fit if only that piece is needed.

Replaces ``tcm._dask_legacy.incl_calibr_hy`` with pure-numpy math
and ``xr.Dataset`` I/O.

Submodules
----------
* :mod:`run` — entry point
* :mod:`calibrate` — ellipsoid fitting (pure numpy kernels).
* :mod:`moments` — sample-weighting scheme for Li-Griffiths fitting.
* :mod:`robust` — outlier rejection, iterative refit, uncertainty/coverage diagnostics,
  field-data autocalibration.
* :mod:`spatial_binning` — 3-D bin averaging on the sphere (θ + φ).
* :mod:`filtering` — per-channel despiking.
* :mod:`pipeline` — full iterative fit → reject loop.
* :mod:`visualization` — 3-D ellipsoid / channel diagnostic plots.
* :mod:`vis_common` — shared layout/colorbar/projection helpers for the two plotting modules.
* :mod:`vis_coverage` — sphere coverage maps (Fibonacci/HEALPix, Voronoi/grid rendering).
* :mod:`orientation` — zero-tilt zeroing, heading reference, azimuth_shift.
"""
