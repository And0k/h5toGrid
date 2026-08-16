# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for tcm_proc — HDF5-free inclinometer data processor.

Build (noh5-tcm): pixi run -e noh5-tcm pyinstaller tcm/scripts/build/tcm_proc.spec
Build (bin-optim-tcm): pixi run -e bin-optim-tcm pyinstaller tcm/scripts/build/tcm_proc.spec

Uses the 'noh5' pixi environment as primary target (OpenBLAS, no MKL).
The spec is also valid in bin-optim-tcm (xarray/dask/scipy/numba included
but Parquet write and HDF5 modules remain excluded).
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

if "SPECPATH" not in globals():
    from PyInstaller.building.build_main import COLLECT, EXE, PYZ, Analysis

    SPEC_DIR = Path(os.path.abspath(__file__)).parent
    print("SPECPATH =", SPEC_DIR)
    print("CWD =", os.getcwd())
    print("FILE =", __file__)
else:
    SPEC_DIR = Path(SPECPATH)

# Shared utilities
sys.path.insert(0, str(SPEC_DIR))
from spec_common import (
    EXCLUDE_BINARIES,
    RUNTIME_DLLs,
    collect_docs,
    load_meta,
    should_keep_binary,
    should_keep_data,
)

META = load_meta()
VERSION = META["version"]

block_cipher = None

PROJECT_ROOT = SPEC_DIR.parent.parent
TCM_SRC = "src/tcm"  # source layout under PROJECT_ROOT
TCM_REL = "tcm"  # destination name in the bundled app

_ENV_PREFIX = os.path.dirname(sys.executable)
_ENV_LIB_BIN = os.path.join(_ENV_PREFIX, "Library", "bin")
_site_pkgs = os.path.join(_ENV_PREFIX, "Lib", "site-packages")
SITE_PKGS = Path(_site_pkgs)

# Runtime DLLs — from spec_common (OpenBLAS, stdlib)
_ALL_DIST_DLLS = RUNTIME_DLLs

print(
    f"[spec] DLLs to bundle: {len(_ALL_DIST_DLLS)} "
    f"({sum(os.path.getsize(os.path.join(_ENV_LIB_BIN, d)) for d in _ALL_DIST_DLLS if os.path.isfile(os.path.join(_ENV_LIB_BIN, d))) / 1024**2:.1f} MB)"
)


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

_DOC_EXCLUDE = {"todo.md", "potential_functionality_and_improvement.md"}
added_files = [
    (str(PROJECT_ROOT / TCM_SRC), TCM_REL),
    (str(SPEC_DIR / "version_meta.json"), "."),
    *collect_docs(_DOC_EXCLUDE),
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

a = Analysis(
    [str(PROJECT_ROOT / "scripts" / "tcm_proc.py")],
    pathex=[str(PROJECT_ROOT), str(PROJECT_ROOT / "src")],
    binaries=[
        (os.path.join(_ENV_LIB_BIN, dll), ".")
        for dll in _ALL_DIST_DLLS
        if os.path.isfile(os.path.join(_ENV_LIB_BIN, dll))
    ],
    datas=added_files,
    hiddenimports=[
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
        "tcm._constants",
    ]
    + collect_submodules("hydra")
    + collect_submodules("hydra_plugins"),
    hookspath=[str(PROJECT_ROOT / "scripts" / "build" / "hooks")],
    hooksconfig={},
    runtime_hooks=[
        str(PROJECT_ROOT / "scripts" / "build" / "rthook_hydra_pkg.py"),
        str(PROJECT_ROOT / "scripts" / "build" / "rthook_noh5_bins.py"),
    ],
    excludes=[
        "h5py",
        "tables",
        "pytables",
        "hdf5",
        "tcm.h5inclinometer_coef",
        "tcm.h5",
        "tcm.h5_dask_pandas",
        "tcm.incl_h5_utils",
        "tcm.incl_h5spectrum",
        "tcm.incl_calibr_hy",
        "tcm.veuszPropagate",
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
        "matplotlib",
        "bokeh",
        "dtale",
        "selenium",
        "sklearn",
        "scipy",
        "scipy.linalg",
        "scipy.sparse",
        "scipy.fft",
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
        "PIL",
        "Pillow",
        "lxml",
        "openpyxl",
        "cryptography",
        "tkinter",
        "_tkinter",
        "tcl",
        "tk",
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
        "webbrowser",
        "pyarrow.flight",
        "pyarrow._flight",
        "pyarrow.gandiva",
        # PyArrow core + all transitive deps — NOT needed for text pipeline,
        # but automatically collected by PyInstaller through xarray/pandas
        # optional imports.  Excluding the top-level package prevents the
        # entire pyarrow subtree from being bundled.
        "pyarrow",
        "pyarrow.libs",
        # pyarrow transitive deps (S3/Azure SDK, compression, etc.)
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
        # pyarrow's compression C extensions — .pyd files handled by
        # _exclude_binaries below
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
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


# ---------------------------------------------------------------------------
# Post-analysis filters (shared via spec_common)
# ---------------------------------------------------------------------------

# CLI-only: also exclude .h5/.hdf5 binaries (noh5 env has no h5py)
_cli_extra = [".h5", ".hdf5"]

a.binaries = [b for b in a.binaries if should_keep_binary(b, _cli_extra)]
a.datas = [d for d in a.datas if should_keep_data(d)]

a.pure = [m for m in a.pure if not m[0].startswith(TCM_REL)]
# Exclude distributed scheduler modules (not available in noh5 environment)
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
    name="tcm_proc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
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
    name="tcm_proc",
)
