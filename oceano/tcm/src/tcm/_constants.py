"""Project-wide constants — zero internal dependencies.

Single source of truth for
- path names
- optional-dependency availability flags
- resolved ``use_h5`` runtime state (mirror of ``ConfigProgram.use_h5``)
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
except ImportError:
    _netCDF4 = None
NC4_AVAILABLE: bool = _netCDF4 is not None
"""Whether ``netCDF4`` is importable (xarray NC engine)."""

nc_suffixes = (".nc", )  # +".nc4"?
hdf5_suffixes = (".h5", ".hdf5")


# ---------------------------------------------------------------------------
# Resolved ``use_h5`` runtime state
# ---------------------------------------------------------------------------
# Mirrors ``ConfigProgram.use_h5`` after startup resolution.
# Written once by :func:`use_h5_set` from ``processing.run()``.
# Read by ``storage.py``, ``coefs.py``, and other modules that don't
# receive the config dict directly.

_use_h5: Optional[bool] = None
"""Resolved ``use_h5`` state — set once at pipeline startup.

- ``True``  — binary (NC/HDF5) I/O enabled.  Proceed without extra logging.
- ``False`` — binary I/O disabled (user override *or* forced by missing libs).
              Log a warning at each skip point.
- ``None``  — binary I/O unavailable and user didn't request it.
              Skip silently.
"""


def use_h5_set(value: Optional[bool]) -> None:
    """Store the resolved ``use_h5`` value for module-level access.

    Called once from :func:`processing.run` after resolving the user's
    ``ConfigProgram.use_h5`` setting against actual library availability.
    """
    global _use_h5
    _use_h5 = value


def use_h5_get() -> Optional[bool]:
    """Return the current ``use_h5`` state for guard checks.

    Returns
    -------
    ``True``
        Binary I/O is enabled — proceed without extra logging.
    ``False``
        Binary I/O is disabled (user override *or* forced by missing libs).
        The **caller** should log a warning at the skip point.
    ``None``
        Binary I/O is unavailable and user didn't request it.
        Skip **silently** (no log message).
    """
    return _use_h5


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
_EXT_HDF5 = {".h5", ".hdf5"}
_EXT_NC = {".nc", ".nc4"}
