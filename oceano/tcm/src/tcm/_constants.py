"""Project-wide constants — zero internal dependencies.

Single source of truth for
- path names
- optional-dependency availability flags
- build metadata (version, URLs) via :func:`version_meta`
"""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

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
    import h5netcdf as _h5netcdf  # h5py-backed NC engine — the only one used
except ImportError:
    _h5netcdf = None
nc_engine = "h5netcdf"
NC4_AVAILABLE: bool = _h5netcdf is not None
"""Whether the NC4 xarray engine (``h5netcdf``) is importable — gates probe
combine (:data:`nc_engine` is the single engine for all NC I/O; the ``netCDF4``
package is never used — it would drag the netCDF-C stack: netcdf.dll →
libxml2/libcurl → ICU ≈ 39 MB into frozen builds)."""

# Supported extensions grouped by backend — single source of truth, mirrors
# ``meta_finder.config.extensions_text|extensions_hdf5|extensions_archive`` so
# discovery, table enumeration and binary dispatch see the same sets.
EXT_CSV = {".txt", ".csv", ".tsv"}
EXT_HDF5 = {".h5", ".hdf5", ".mat"}
EXT_NC = {".nc"}
ARCHIVE_EXTS = {".zip", ".7z"}
"""Archive extensions — zip + 7z, mirrors ``meta_finder.config.extensions_archive``."""

# Available extensions given current installation — HDF5/NC require h5py
# (NC depends on HDF5 stack, so both are hidden when H5 is unavailable).
EXT_HDF5_AVAILABLE = EXT_HDF5 if H5_AVAILABLE else set()
EXT_NC_AVAILABLE = EXT_NC if H5_AVAILABLE else set()
EXT_DATA_AVAILABLE = EXT_CSV | EXT_HDF5_AVAILABLE | EXT_NC_AVAILABLE
"""All data extensions that can actually be processed in this environment."""

# ---------------------------------------------------------------------------
# Project root paths
# ---------------------------------------------------------------------------

# Canonical name for the directory that anchors all relative processing paths.
RAW_DIR_NAME: str = "_raw"

# Package directory (``…/tcm/src/tcm`` in the src layout) — home of the
# bundled ``cfg/`` tree; NOT the repo root (see :data:`REPO_ROOT`).
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CFG_PATH = PROJECT_ROOT / "cfg"


def _repo_root() -> Path:
    """Protected code-project root — processing must never create files in it.

    * **Frozen distributive** — the executable's own directory: ``__file__``
      lives in the transient ``_MEIPASS`` extraction, and the app folder is
      the only "project dir" a double-clicked exe can pollute (an empty
      ``input.path`` means ``./`` = the launch cwd).
    * **Dev / installed** — outermost ancestor of this file still carrying a
      ``pyproject.toml`` (``oceano/`` in the src layout; also found from envs
      nested inside the repo).  Falls back to :data:`PROJECT_ROOT` for bare
      installs, so at least the package tree itself stays protected.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    root = PROJECT_ROOT
    for anc in PROJECT_ROOT.parents:
        if (anc / "pyproject.toml").is_file():
            root = anc
    return root


REPO_ROOT: Path = _repo_root()

# Module path for @hydra.main
# Requires tcm/cfg/__init__.py and tcm/cfg/cfg_proc/__init__.py for pkg:// resolution.
BUNDLED_CFG_PKG = f"pkg://{PROJECT_ROOT.name}.cfg.cfg_proc"


# In frozen app: PyInstaller data root.
def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return PROJECT_ROOT.parent.parent


# Documentation sits next to the project dir in both layouts: dev repo has
# ``oceano/tcm/docs``; the frozen tree mirrors the repo (``_MEIPASS`` ≙ repo
# root, package at ``oceano/tcm/src/tcm`` — see spec_common.collect_docs), so
# the same two-parents-up formula lands on ``oceano/tcm/docs`` in both.  Readme
# links via ``DOC_DIR.parent`` (readme*.md) resolve the same way.
DOC_DIR = PROJECT_ROOT.parents[1] / "docs"


@cache
def version_meta() -> dict:
    """Build metadata: version, product, repo/docs URLs.

    Frozen exe → ``_MEIPASS/version_meta.json``.
    Dev → ``scripts/build/version_meta.json`` relative to ``resource_root()``
    (``oceano/tcm`` in src-layout).
    Missing → empty dict (graceful degradation).
    """
    candidates = [
        resource_root() / "version_meta.json",
        resource_root() / "scripts" / "build" / "version_meta.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}
