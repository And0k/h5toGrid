# -*- mode: python ; coding: utf-8 -*-  # noqa: UP009 — spec files expect this header
"""Shared PyInstaller spec utilities for tcm builds.

Imported by ``tcm_proc.spec`` and ``tcm_gui.spec`` to avoid
duplicating binary filters, data collectors, and version loading.
"""

import os
import sys
from pathlib import Path

from tcm import _constants

PROJECT_ROOT = _constants.resource_root()
SPEC_DIR = Path(os.path.abspath(__file__)).parent

# Pixi env name (e.g. noh5-tcm) — suffixes build dirs so envs never collide
ENV_NAME = Path(sys.executable).resolve().parent.name


def build_layout(app: str, root: Path | None) -> tuple[Path, Path]:
    """Resolve ``(--workpath, --distpath)`` for a PyInstaller build of *app*.

    *root* (e.g. ``B:\\Temp`` via ``--build-root``) gets an env-suffixed
    ``<app>-env=<ENV_NAME>`` subfolder — builds from different pixi envs never
    collide and the project drive stays light; ``None`` keeps in-repo default.
    """
    base = root / f"{app}-env={ENV_NAME}" if root else PROJECT_ROOT
    return base / "build", base / "dist"


PROJECTS_ROOT = PROJECT_ROOT.parent  # oceano/ — sibling projects (collect_docs)
REPO_ROOT = PROJECTS_ROOT.parent  # repository root — shared/* packages live here

# First-party editable packages imported top-level by tcm/tcm_gui — shipped as
# source datas.  Keys are repo-relative src paths, used verbatim as frozen dests
# (`_MEIPASS` mirrors the dev repo root exactly; see rthook_repo_layout): the
# statically visible import chain breaks at the datas-shipped tcm code, so
# Analysis can't see `from utils import …` / `import meta_finder` inside it —
# only datas guarantee every submodule
FIRST_PARTY_PKGS: dict[str, Path] = {
    rel: REPO_ROOT / rel
    for rel in (
        "shared/utils/src/utils",
        "shared/veusz_helpers/src/veusz_helpers",
        "oceano/meta_finder/src/meta_finder",
    )
}

# Pure-module prefixes shipped as datas instead (avoids duplication at freeze);
# "tcm" covers tcm_gui too
DATA_PKG_PREFIXES = ("tcm", "utils", "veusz_helpers", "meta_finder")


def collect_first_party_pkgs() -> list[tuple[str, str]]:
    """``(src_file, dest_dir)`` datas for :data:`FIRST_PARTY_PKGS` (subdirs
    preserved), skipping dev junk (`copy/` backups, tests, veusz-bound
    ``veuszPropagate``, bytecode caches)."""
    return [
        (str(p), str(Path(pkg) / p.relative_to(d).parent))
        for pkg, d in FIRST_PARTY_PKGS.items()
        for p in d.rglob("*")
        if p.is_file()
        and "copy" not in p.parts
        and "__pycache__" not in p.parts
        and not p.name.startswith(("test_", "veuszPropagate"))
        and not p.name.endswith((".pyc", ".pyo"))
    ]


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
# "icu*" — ICU (International Components for Unicode) DLLs: icudt78 33 MB +
# icuuc/icuin/icuio/icutu/icutest — Qt internationalization data pulled in
# transitively (PySide6 is excluded as a module, but PyInstaller's binary
# dependency walk still collects these from the conda env).  The Tkinter GUI
# never loads them.  Component prefixes, not bare "icu", so data-file names
# like "circulation…" (contains "icu") are not accidentally dropped.
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
    # netCDF-C binary stack — dead weight once netCDF4 package is excluded
    # (engine is h5netcdf/h5py); icu* alone is ~36 MB
    "netcdf",
    "libxml2",
    "libcurl",
    "psl-",
    "icudt",
    "icuuc",
    "icuin",
    "icuio",
    "icutu",
    "icutest",
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

# ── libarchive native closure ────────────────────────────────────────────────
# libarchive-c loads archive.dll dynamically (ctypes, bare name) — invisible to
# PyInstaller's dependency walk — and meta_finder.utils_sys imports it at module
# level (archive time/burst extraction), so a frozen scan dies with
# PyInstallerImportError: Failed to load dynlib 'archive.dll'.  Compute the PE
# import closure of archive.dll against the build env's Library/bin at build
# time: hand-maintained lists rot with every conda solve (libxml2 → iconv → …).

LIB_BIN = Path(sys.executable).resolve().parent / "Library" / "bin"


def _pe_imports(dll: Path) -> set[str]:
    """Bare names of DLLs in the PE import table of *dll* (PE32+/PE32)."""
    import struct

    b = dll.read_bytes()
    e_lfanew = struct.unpack_from("<I", b, 0x3C)[0]
    coff = e_lfanew + 4
    nsec = struct.unpack_from("<H", b, coff + 2)[0]
    opt_size = struct.unpack_from("<H", b, coff + 16)[0]
    opt = coff + 20
    magic = struct.unpack_from("<H", b, opt)[0]
    ddir = opt + (112 if magic == 0x20B else 96)
    rva = struct.unpack_from("<I", b, ddir + 8)[0]
    if not rva:
        return set()

    def off(r: int) -> int:  # RVA → file offset via section table
        for vsize, va, rsize, praw in (
            struct.unpack_from("<IIII", b, opt + opt_size + 40 * i + 8) for i in range(nsec)
        ):
            if va <= r < va + max(vsize, rsize):
                return r - va + praw
        raise ValueError(f"RVA {r:#x} unmapped in {dll.name}")

    names = set()
    for i in range(64):  # import descriptors, 0-terminated entry ends the walk
        name_rva = struct.unpack_from("<I", b, off(rva) + 20 * i + 12)[0]
        if not name_rva:
            break
        pos = off(name_rva)
        names.add(b[pos : b.index(b"\0", pos)].decode())
    return names


def _dll_closure(entry: str = "archive.dll", root: Path = LIB_BIN) -> list[str]:
    """*entry* + transitive deps found in *root* (BFS order, deps of deps last)."""
    if not (root / entry).is_file():
        return []
    seen: dict[str, None] = {}
    todo = [entry]
    while todo:
        name = todo.pop(0)
        if name.lower() in seen or not (root / name).is_file():
            continue  # system DLL (kernel32, api-ms-*, …) or absent → loader handles
        seen[name.lower()] = None
        todo += sorted(_pe_imports(root / name))
    return list(seen)


# ICU data DLL is LoadLibrary'd by icuuc at runtime (not in any import table) —
# pull it in explicitly whenever the icuuc closure member is present.
LIBARCHIVE_DLLs: list[str] = [n for n in _dll_closure() if not n.lower().startswith(("api-ms-", "vcruntime"))]
if any(n.startswith("icuuc") for n in LIBARCHIVE_DLLs) and not any(
    n.startswith("icudt") for n in LIBARCHIVE_DLLs
):
    LIBARCHIVE_DLLs += sorted(  # noqa: runtime-only load — not visible to the import walk
        p.name for p in LIB_BIN.glob("icudt*.dll") if p.name.lower() not in map(str.lower, LIBARCHIVE_DLLs)
    )
# Exclusion filter must never drop closure members (e.g. "libxml2" in EXCLUDE_BINARIES).
KEEP_BINARIES: set[str] = {n.lower() for n in LIBARCHIVE_DLLs}


def should_keep_binary(entry: tuple, extra_exclude: list[str] | None = None) -> bool:
    """Filter ``Analysis.binaries`` by name pattern and pyarrow .pyd names.

    PyInstaller TOC tuple: ``(dest_name, src_path, typecode)``.
    """
    patterns = EXCLUDE_BINARIES + (extra_exclude or [])
    dest_name, _src, _type = entry
    name = os.path.basename(dest_name).lower()
    if name in KEEP_BINARIES:
        return True  # libarchive closure — explicitly bundled, exempt from excludes
    if any(pat in name for pat in patterns):
        return False
    pyd_mod = os.path.splitext(name)[0].split(".")[0]
    return not (
        dest_name.startswith("pyarrow" + os.sep)
        and (pyd_mod in _EXCLUDE_PYARROW_PYD or name.endswith(".dll"))
    )


def should_keep_data(entry: tuple) -> bool:
    """Exclude todo/ & done/ dirs, dev junk (`AGENTS.md`, `descript.ion`,
    ``*-`` backup scripts) and pattern-matched files from bundled data."""
    dest_name = entry[0]
    name = os.path.basename(dest_name).lower()
    if name in ("descript.ion", "agents.md") or name.endswith(("-.py", "~.py", "-.md", "~.md")):
        return False
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
    """Collect ``docs/`` files as ``(src, dest_dir)`` tuples, excluding named files.

    The frozen tree mirrors the dev repo — ``_MEIPASS`` ≙ repo root, project docs
    land at ``oceano/<proj>/docs`` — so the docs' repo-relative crosslinks
    (``../../…`` → project dir, ``../../../…`` → ``oceano/``) resolve identically
    in dev and frozen.  Includes ``PROJECT_ROOT/docs`` (the ``tcm`` package) and
    sibling-project docs (e.g. ``meta_finder/docs``).
    """
    exclude = exclude or set()
    proj_rel = f"{PROJECTS_ROOT.name}/{PROJECT_ROOT.name}"
    docs = [
        (str(p), f"{proj_rel}/{p.parent.relative_to(PROJECT_ROOT).as_posix()}")
        for p in (PROJECT_ROOT / "docs").rglob("*")
        if p.is_file() and p.name not in exclude
    ]
    # Sibling-project docs (e.g. meta_finder) — under oceano/ like their sources
    for sibling in sorted(PROJECTS_ROOT.iterdir()):
        if not sibling.is_dir() or sibling.name.startswith((".", "_")) or sibling.name == PROJECT_ROOT.name:
            continue
        for p in (sibling / "docs").rglob("*"):
            if p.is_file() and p.name not in exclude:
                docs.append(
                    (str(p), f"{PROJECTS_ROOT.name}/{p.parent.relative_to(PROJECTS_ROOT).as_posix()}")
                )
    return docs
