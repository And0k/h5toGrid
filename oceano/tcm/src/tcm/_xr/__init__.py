"""xarray-native pipeline for inclinometer data.

Design goals
------------
* ``Dataset`` = 1-D time series backed by ``dask.array``.
* All chunking explicit via ``.chunk({"time": ...})``.
* No ``dask.dataframe`` imports — ever.

Public API (stable names, unstable internals)
-----------------------------------------------
* ``_xr.calc`` – low-level math via ``xr.apply_ufunc``.
* ``_xr.physical`` – velocity, pressure, binning.
* ``_xr.io`` – CSV / netCDF / HDF5 helpers.
* ``_xr.dataset`` – generic ``Dataset`` helpers (``open_csv``, ``merge_probes``).
"""