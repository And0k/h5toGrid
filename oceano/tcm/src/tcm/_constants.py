"""Project-wide constants — zero internal dependencies.

Single source of truth for
- path names
- optional-dependency availability flags
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

VERSION = "2026.07"

# ---------------------------------------------------------------------------
# Optional-dependency availability (resolved once at import time)
# ---------------------------------------------------------------------------

try:
    import h5py as _h5py
except ImportError:
    _h5py = None
H5_AVAILABLE: bool = _h5py is not None
"""Whether the ``h5py`` package is importable (needed for NC4/HDF5 direct I/O)."""

try:
    import tables as _tables
except ImportError:
    _tables = None
TABLES_AVAILABLE: bool = _tables is not None
"""Whether ``pytables`` is importable (needed for ``pd.HDFStore`` and HDF5 dimension scales)."""

try:
    import netCDF4 as _netCDF4
    nc_engine = "h5netcdf"  # "netcdf4"
except ImportError:
    _netCDF4 = None
    nc_engine = "h5netcdf"
NC4_AVAILABLE: bool = _netCDF4 is not None
"""Whether ``netCDF4`` is importable (xarray NC engine)."""

# ---------------------------------------------------------------------------
# Raw-data layout
# ---------------------------------------------------------------------------

# Canonical name for the directory that anchors all relative processing paths.
RAW_DIR_NAME: str = "_raw"

# Project root — parent of ``scripts/`` dir (where pyproject.toml lives).
# Used by :func:`safe_cfg_dir` to guard against polluting the repo.
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CFG_PATH = PROJECT_ROOT / "cfg"

# Module path for @hydra.main
# Requires tcm/cfg/__init__.py and tcm/cfg/cfg_proc/__init__.py for pkg:// resolution.
BUNDLED_CFG_PKG = f"pkg://{PROJECT_ROOT.name}.cfg.cfg_proc"

# Supported extensions grouped by backend
_EXT_CSV = {".txt", ".csv", ".tsv"}
_EXT_HDF5 = {".h5"}  # , ".hdf5" not need
_EXT_NC = {".nc"}  # , ".nc4" not need
