# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for tcm_gui — Tkinter GUI for TCM data processing.

Build: pixi run -e bin-optim-tcm build-tcm-gui

Environment ``bin-optim-tcm`` includes h5py, scipy, matplotlib, numba
(via ``bin-optim`` feature) — all kept for the GUI.  Only pyarrow,
Jupyter, and dev-only packages are excluded.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.building.build_main import COLLECT, EXE, PYZ, Analysis

if "SPECPATH" not in globals():
    SPEC_DIR = Path(os.path.abspath(__file__)).parent
else:
    SPEC_DIR = Path(SPECPATH)

# Shared utilities
sys.path.insert(0, str(SPEC_DIR))
from spec_common import (
    EXCLUDE_BINARIES,
    RUNTIME_DLLs,
    collect_docs,
    collect_first_party_pkgs,
    load_meta,
    should_keep_binary,
    should_keep_data,
    DATA_PKG_PREFIXES,
    PROJECT_ROOT,
)

# Build-env requirement: tksheet ships with the `tk-gui` feature; without it the
# exe would build "successfully" but die with ModuleNotFoundError at first import
try:
    import tksheet  # noqa: F401
except ImportError as e:
    raise SystemExit(
        "tcm_gui build env lacks 'tksheet' — use an env with the 'tk-gui' feature "
        "(bin-optim-tcm, noh5-tcm-gui), not e.g. noh5-tcm"
    ) from e

print(f"{SPEC_DIR=}")
META = load_meta(SPEC_DIR)
VERSION = META["version"]

block_cipher = None

TCM_SRC = "src/tcm"
TCM_PROJ = "oceano/tcm"  # project dir in the repo-mirrored frozen tree
TCM_REL = f"{TCM_PROJ}/src/tcm"
GUI_SRC = "src/tcm_gui"
GUI_REL = "oceano/tcm_gui/src/tcm_gui"

_ENV_PREFIX = os.path.dirname(sys.executable)
_ENV_LIB_BIN = os.path.join(_ENV_PREFIX, "Library", "bin")
_site_pkgs = os.path.join(_ENV_PREFIX, "Lib", "site-packages")
SITE_PKGS = Path(_site_pkgs)

_ALL_DIST_DLLS = RUNTIME_DLLs

print(
    f"[spec] DLLs to bundle: {len(_ALL_DIST_DLLS)} "
    f"({sum(os.path.getsize(os.path.join(_ENV_LIB_BIN, d)) for d in _ALL_DIST_DLLS if os.path.isfile(os.path.join(_ENV_LIB_BIN, d))) / 1024**2:.1f} MB)"
)


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

_DOC_EXCLUDE = {"todo.md", "potential_functionality_and_improvement.md"}
# Generated third-party browser runtime — resolves via resource_root()/_build
# at runtime (tcm_gui.browser.server._VEND_DIR); first-party web/ ships inside
# src/tcm_gui and arrives with the GUI_SRC data below.
added_files = [
    (str(PROJECT_ROOT / TCM_SRC), TCM_REL),
    (str(PROJECT_ROOT / GUI_SRC), GUI_REL),
    *collect_first_party_pkgs(),
    (str(SPEC_DIR / "version_meta.json"), "."),
    (str(PROJECT_ROOT / "_build" / "browser-runtime"), "_build/browser-runtime"),
    *collect_docs(_DOC_EXCLUDE),
    # Entry-point readmes → oceano/tcm/readme*.md next to docs/ (the About
    # header "local" docs link opens readme.md; served like any other markdown).
    (str(PROJECT_ROOT / "readme.md"), TCM_PROJ),
    (str(PROJECT_ROOT / "readme_Ru.md"), TCM_PROJ),
    *(collect_data_files("hydra", subdir="conf") + collect_data_files("hydra_plugins.hydra_colorlog")),
    *collect_data_files("pygeomag"),
    *[
        (str(SITE_PKGS / dir_to / file), dir_to.replace("*", "0"))
        for file, dir_to in [
            ("METADATA", "pandas-*.dist-info"),
            ("METADATA", "numpy-*.dist-info"),
            ("__init__.py", "hydra/conf"),
            ("__init__.py", "hydra_plugins/hydra_colorlog/conf"),
        ]
    ],
]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
print(f"{PROJECT_ROOT=}")
a = Analysis(
    [str(PROJECT_ROOT / "scripts" / "tcm_gui.py")],
    pathex=[str(PROJECT_ROOT), str(PROJECT_ROOT / "src")],
    binaries=[
        (os.path.join(_ENV_LIB_BIN, dll), ".")
        for dll in _ALL_DIST_DLLS
        if os.path.isfile(os.path.join(_ENV_LIB_BIN, dll))
    ],
    datas=added_files,
    hiddenimports=[
        # All first-party packages (tcm, tcm_gui, utils, veusz_helpers,
        # meta_finder) ship as source datas — no hiddenimports for them: the
        # entry script's visible imports (tcm_gui.py → tcm_gui.app → tcm) still
        # lead Analysis through the dev graph, pulling every transitive dep.
        # The rows below are reachable only via datas-shipped tcm_gui code
        # (browser server.py, coefficient sheet) → invisible to Analysis
        "colorlog",
        "colorlog.formatter",
        "omegaconf",
        "ruamel.yaml",
        "dask",
        "dask.base",
        "dask.diagnostics",
        "numba",
        "numba.core",
        "pygeomag",
        "pandas",
        "pandas._libs",
        "xarray",
        "h5netcdf",  # NC engine — imported dynamically by xarray (engine=nc_engine)
        # to Analysis; the documentation browser needs these stdlib modules
        "http.server",  # tcm_gui/browser/server.py
        "webbrowser",  # tcm_gui/browser/browser.py (also NOT in excludes)
        "tksheet",
    ]
    + collect_submodules("hydra")
    + collect_submodules("hydra_plugins"),
    hookspath=[str(PROJECT_ROOT / "scripts" / "build" / "hooks")],
    hooksconfig={},
    runtime_hooks=[
        str(PROJECT_ROOT / "scripts" / "build" / "rthook_repo_layout.py"),
        str(PROJECT_ROOT / "scripts" / "build" / "rthook_hydra_pkg.py"),
        # rthook_noh5_bins.py NOT used — GUI shows full config defaults
    ],
    excludes=[
        # MKL — OpenBLAS env (from noh5 feature)
        "mkl",
        "mkl_rt",
        "mkl_core",
        "mkl_intel_thread",
        "mkl_sequential",
        "mkl_tbb_thread",
        "mkl_def",
        "mkl_avx",
        "mkl_avx2",
        "mkl_avx512",
        "mkl_mc",
        "mkl_mc3",
        "mkl_vml_def",
        "mkl_vml_avx",
        "mkl_vml_avx2",
        "mkl_vml_avx512",
        "mkl_vml_cmpt",
        "mkl_vml_mc",
        "mkl_vml_mc3",
        "mkl_blacs_ilp64",
        "mkl_blacs_lp64",
        "mkl_scalapack_ilp64",
        "mkl_scalapack_lp64",
        "mkl_cdft_core",
        "mkl_pgi_thread",
        "mkl_msg",
        # HDF5 modules not used by GUI code path
        "tcm.h5inclinometer_coef",
        "tcm.h5",
        "tcm.h5_dask_pandas",
        "tcm.incl_h5_utils",
        "tcm.incl_h5spectrum",
        "tcm.incl_calibr_hy",
        # netCDF4 package unused — engine is h5netcdf (h5py); excluding it drops
        # the whole netCDF-C binary chain (netcdf.dll → libxml2/libcurl → ICU)
        "netCDF4",
        # PyArrow (not needed)
        "pyarrow",
        "pyarrow.libs",
        "pyarrow.flight",
        "pyarrow._flight",
        "pyarrow.gandiva",
        "botocore",
        "botocore.session",
        "botocore.utils",
        "botocore.awsrequest",
        "botocore.client",
        "botocore.endpoint",
        "botocore.httpsession",
        "botocore.parsers",
        "botocore.serialize",
        "botocore.validate",
        "botocore.credentials",
        "botocore.auth",
        "botocore.eventstream",
        "botocore.handlers",
        "botocore.loaders",
        "botocore.model",
        "botocore.paginate",
        "botocore.retries",
        "botocore.waiter",
        "botocore.compat",
        "botocore.exceptions",
        "botocore.stub",
        "botocore.translate",
        "botocore.vendored",
        "certifi",
        "charset_normalizer",
        "charset_normalizer.utf8",
        "charset_normalizer.md",
        "google_crc32c",
        "numcodecs",
        "numcodecs.registry",
        "numcodecs.compat",
        "numcodecs.blosc",
        "numcodecs.zstd",
        "numcodecs.lz4",
        "zstandard",
        "zstd",
        # Heavy libs not used by GUI
        "numexpr",
        "IPython",
        "jupyter",
        "notebook",
        "jupyterlab",
        "jupyter_server",
        "jupyter_client",
        "jupyter_core",
        "nbformat",
        "nbconvert",
        "bokeh",
        "dtale",
        "selenium",
        "sklearn",
        "polars",
        "geopandas",
        "pyproj",
        "seaborn",
        "gsw",
        "statsmodels",
        "sympy",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "qtpy",
        "qtconsole",
        "lxml",
        "openpyxl",
        "cryptography",
        "sphinx",
        "docutils",
        "pytest",
        "_pytest",
        "py",
        "pluggy",
        "setuptools",
        "pkg_resources",
        "wheel",
        "pip",
        # note: "webbrowser" must stay AVAILABLE (not excluded) — the
        # documentation browser (tcm_gui/browser) imports it to open the
        # system default browser; it was once excluded here by mistake.
        "tornado",
        "msgpack",
        "lz4",
        "numba.core.tbbpool",
        "unittest",
        "test",
        "tests",
        "curses",
        "readline",
        "lib2to3",
        "py_compile",
        "compileall",
        "distributed",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


# ---------------------------------------------------------------------------
# Post-analysis filters (shared via spec_common)
# ---------------------------------------------------------------------------

a.binaries = [b for b in a.binaries if should_keep_binary(b)]
a.datas = [d for d in a.datas if should_keep_data(d)]

# Exclude both tcm.* and tcm_gui.* from pure — collected as data instead
a.pure = [m for m in a.pure if not m[0].startswith(DATA_PKG_PREFIXES)]
# Exclude distributed scheduler (not available in bin-optim-tcm)
_dist_prefix = "distributed."
a.pure = [m for m in a.pure if not (m[0] == "distributed" or m[0].startswith(_dist_prefix))]

# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="tcm_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app — no console window
    icon=SPEC_DIR / "tcm.ico",
    version=SPEC_DIR / "version_info.txt",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="tcm_gui",
)
