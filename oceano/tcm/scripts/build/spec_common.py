# -*- mode: python ; coding: utf-8 -*-  # noqa: UP009 — spec files expect this header
"""Shared PyInstaller spec utilities for tcm builds.

Imported by ``tcm_proc.spec`` and ``tcm_gui.spec`` to avoid
duplicating binary filters, data collectors, and version loading.
"""

import os
from pathlib import Path

from tcm import _constants

PROJECT_ROOT = _constants.resource_root()
SPEC_DIR = Path(os.path.abspath(__file__)).parent

# pyarrow internal .pyd extensions that depend on excluded native DLLs.
_EXCLUDE_PYARROW_PYD: set[str] = {
    "_parquet",
    "_orc",
    "_dataset",
    "_dataset_orc",
    "_dataset_parquet",
    "_fs",
    "_gcsfs",
    "_s3fs",
    "_hdfs",
    "_flight",
    "_gandiva",
    "_acero",
    "_json",
    "_csv",
    "_compute",
    "_exec_plan",
    "_substrait",
    "_cdata",
    "_dataset_schema",
    "_helpers",
    "_azurefs",
}

# Binary basename substrings to exclude (case-insensitive).
# NOTE: "zstd"/"lz4" bare names excluded — only Python extensions ("_zstd", "_lz4")
# and pyarrow's "libzstd.dll".  The native zstd.dll / lz4.dll are kept because
# llvmlite (numba) depends on zstd.dll.
EXCLUDE_BINARIES: list[str] = [
    "mkl_",
    "pyarrow",
    "parquet.dll",
    "libzstd.dll",
    "_zstd",
    "_lz4",
    "botocore",
    "certifi",
    "charset_normalizer",
    "google_crc32c",
    "numcodecs",
]

# Runtime DLLs for OpenBLAS + Python stdlib — shared by all tcm builds.
RUNTIME_DLLs: list[str] = [
    "libmpdec-4.dll",
    "liblzma.dll",
    "libexpat.dll",
    "ffi-8.dll",
    "yaml.dll",
    "sqlite3.dll",
    "libzmq-mt-4_3_5.dll",
    "tbb12.dll",
    "tbbmalloc.dll",
    "tbbmalloc_proxy.dll",
    "openblas.dll",
    "libcblas.dll",
    "libblas.dll",
    "liblapack.dll",
]


def should_keep_binary(entry: tuple, extra_exclude: list[str] | None = None) -> bool:
    """Filter ``Analysis.binaries`` by name pattern and pyarrow .pyd names.

    PyInstaller TOC tuple: ``(dest_name, src_path, typecode)``.
    """
    patterns = EXCLUDE_BINARIES + (extra_exclude or [])
    dest_name, _src, _type = entry
    name = os.path.basename(dest_name).lower()
    if any(pat in name for pat in patterns):
        return False
    pyd_mod = os.path.splitext(name)[0].split(".")[0]
    return not (
        dest_name.startswith("pyarrow" + os.sep)
        and (pyd_mod in _EXCLUDE_PYARROW_PYD or name.endswith(".dll"))
    )


def should_keep_data(entry: tuple) -> bool:
    """Exclude todo/ & done/ internal dirs and pattern-matched files from bundled data."""
    dest_name = entry[0]
    name = os.path.basename(dest_name).lower()
    if any(pat in name for pat in EXCLUDE_BINARIES):
        return False
    parts = dest_name.replace("\\", os.sep).replace("/", os.sep).split(os.sep)
    return not {"todo", "done"}.intersection(parts)


def load_meta(spec_dir: Path | None = None) -> dict:
    """Read ``version_meta.json`` from *spec_dir* (default: next to this script)."""
    import json

    path = (spec_dir or SPEC_DIR) / "version_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def collect_docs(exclude: set[str] | None = None) -> list[tuple[str, str]]:
    """Collect ``docs/`` files as ``(src, dest_dir)`` tuples, excluding named files."""
    exclude = exclude or set()
    return [
        (str(p), str(p.parent.relative_to(PROJECT_ROOT)))
        for p in (PROJECT_ROOT / "docs").rglob("*")
        if p.is_file() and p.name not in exclude
    ]
