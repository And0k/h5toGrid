from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Union

import pandas as pd
import xarray as xr

from tcm import format, csv_load, utils2init

lf = utils2init.LoggingStyleAdapter(__name__)


# --------------------------------------------------------------------------- #
# DataFrame ↔ Dataset conversion
# --------------------------------------------------------------------------- #

def dataframe_to_ds(df: pd.DataFrame, chunk_time: Optional[int] = None) -> xr.Dataset:
    """Convert a time-indexed DataFrame to an (optionally chunked) xarray Dataset."""
    if df.index.name != "time":
        df = df.rename_axis("time")
    if df.index.has_duplicates:
        n_dup = df.index.size - df.index.unique().size
        first_dup = df.index[df.index.duplicated()][0]
        raise ValueError(
            f"dataframe_to_ds: {n_dup} duplicate time index value(s). "
            f"First duplicate: {first_dup}"
        )
    ds = xr.Dataset.from_dataframe(df)
    if chunk_time is not None:
        ds = ds.chunk({"time": chunk_time})
    return ds


# --------------------------------------------------------------------------- #
# CSV file resolution — shared by open_csv / open_csv_chunks
# --------------------------------------------------------------------------- #

def _resolve_csv_files_dict(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    text_type: str = "i",
) -> tuple[Dict, str]:
    """Build ``csv_files_dict`` and ``text_type`` from *path* argument.

    :return: ``(csv_files_dict, text_type)`` ready for :func:`csv_load.load_from_csv_gen`.
    """
    if isinstance(path, (list, tuple)):
        csv_files_dict_raw: Dict = {}
        for p in path:
            p = Path(p)
            if (identity := format.probe_from_name(p.stem.lower())) is None:
                identity = (text_type, 0)
            csv_files_dict_raw.setdefault(identity, []).append(p)
        return csv_files_dict_raw, text_type

    # Directory or glob/regex pattern → delegate to search_csv_files which handles all
    if (path := Path(path)).is_dir() or any(c in path.name for c in '*?'):
        return csv_load.search_csv_files(path), text_type
    # Single file
    if (identity := format.probe_from_name(Path(path).stem.lower())) is None:
        identity = (text_type, 0)
    return {identity: [Path(path)]}, text_type


# --------------------------------------------------------------------------- #
# CSV → Dataset (single-shot)
# --------------------------------------------------------------------------- #

def open_csv(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    text_type: str = "i",
    cfg_in: Optional[Mapping[str, Any]] = None,
    chunk_time: Optional[int] = None,
) -> xr.Dataset:
    """Load raw inclinometer CSV file(s) into a single :class:`xarray.Dataset`.

    Wraps :func:`tcm.csv_load.load_from_csv_gen` — handles file discovery,
    raw correction, headerless parsing, and time correction.

    Parameters
    ----------
    path
        Glob pattern (e.g. ``_raw/*i*.txt``) or explicit file path(s).
    text_type
        Probe model letter (``'i'``, ``'p'``, ``'b'``, ``'d'``, ``'w'``).
    cfg_in
        Optional overrides merged into ``csv_load.cfg_default["in"]``.
    chunk_time
        If given, chunk the ``time`` dimension into blocks of this size.

    Returns
    -------
    xr.Dataset
    """
    from tcm import csv_load

    csv_files_dict, text_type = _resolve_csv_files_dict(path, text_type)
    cfg_merged = {**csv_load.cfg_default["in"], **(cfg_in or {})}
    cfg_merged["text_type"] = text_type

    # Collect all chunks into one DataFrame, then convert
    frames: list[pd.DataFrame] = []
    for df, (_i1_pid, _pid, _path_csv) in csv_load.load_from_csv_gen(
        csv_files_dict=csv_files_dict, cfg_in=cfg_merged,
    ):
        if df is not None and len(df):
            frames.append(df)

    if not frames:
        lf.warning("No data loaded from {:s}", str(path))
        return xr.Dataset()

    df_all = pd.concat(frames)
    if df_all.index.name == "Time":
        df_all = df_all.rename_axis("time")
    return dataframe_to_ds(df_all, chunk_time=chunk_time)


# --------------------------------------------------------------------------- #
# CSV → Dataset (chunked generator)
# --------------------------------------------------------------------------- #

def open_csv_chunks(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    text_type: str = "i",
    cfg_in: Optional[Mapping[str, Any]] = None,
    chunk_time: Optional[int] = None,
) -> Iterator[tuple[xr.Dataset, tuple[int, str, Path]]]:
    """Yield ``(xr.Dataset, (i1_pid, pcid, path_csv))`` per CSV chunk.

    Wraps :func:`tcm.csv_load.load_from_csv_gen` but yields one
    :class:`xr.Dataset` per chunk instead of accumulating into a single
    Dataset.  Suitable for streaming processing of large files.

    Parameters
    ----------
    path
        Glob pattern or explicit file path(s).
    text_type
        Probe model letter.
    cfg_in
        Optional overrides merged into ``csv_load.cfg_default["in"]``.
    chunk_time
        If given, chunk each yielded Dataset's ``time`` dimension.

    Yields
    ------
    tuple[xr.Dataset, tuple[int, str, Path]]
        ``(ds, (i1_pid, pcid, path_csv))`` — one per CSV chunk.
    """
    # CSV: merge csv_load defaults
    cfg_merged = {**csv_load.cfg_default["in"], **(cfg_in or {})}
    cfg_merged.setdefault("text_type", text_type)
    csv_files_dict, cfg_merged["text_type"] = _resolve_csv_files_dict(path, text_type)

    for df, meta in csv_load.load_from_csv_gen(
        csv_files_dict=csv_files_dict, cfg_in=cfg_merged,
    ):
        if df is None or not len(df):
            continue
        if df.index.name == "Time":
            df = df.rename_axis("time")
        ds = dataframe_to_ds(df, chunk_time=chunk_time)
        yield ds, meta


# --------------------------------------------------------------------------- #
# NC file → Dataset + coefs
# --------------------------------------------------------------------------- #

def open_nc(
    path: Union[str, Path],
    *,
    tbl: str = "",
    chunk_time: Optional[int] = None,
    engine: str = "netcdf4",
) -> tuple[xr.Dataset, Optional[Dict[str, Any]]]:
    """Load data and coefs from a NetCDF4 file.

    Reads data variables from the ``/{tbl}/`` group (or root if *tbl* is
    empty) and attempts to extract coefs from ``/{tbl}/coef/``.

    :param path: Path to ``.nc`` file.
    :param tbl: Table group name (e.g. ``"incl_01"``).  If empty, reads root.
    :param chunk_time: If given, chunk the ``time`` dimension.
    :param engine: netCDF backend engine.
    :return: ``(ds, coefs_dict)`` — coefs_dict is ``None`` when no coef group.
    """
    from tcm._xr.coefs import load_coefs_from_nc

    if not (path := Path(path)).is_file():
        lf.warning("{} does not exist", path)
        return None, None

    chunks = {"time": chunk_time} if chunk_time else None

    # xr.open_dataset with group= reads the NC4 group as a Dataset
    ds = xr.open_dataset(path, engine=engine, chunks=chunks, group=tbl or None)

    # Ensure time is a coordinate (may be stored as a variable)
    if "time" in ds and "time" not in ds.coords:
        ds = ds.set_coords("time")

    # Extract coefs from /{tbl}/coef/ if present
    coefs = load_coefs_from_nc(path, tbl) if tbl else None

    lf.debug(
        "Loaded NC: {} group={}, {:d} vars, {:d} time steps, coefs={}",
        path, tbl or "(root)", len(ds.data_vars), ds.sizes.get("time", 0),
        "yes" if coefs else "no",
    )
    return ds, coefs


# --------------------------------------------------------------------------- #
# Merge probes
# --------------------------------------------------------------------------- #

def merge_probes(ds_dict: Dict[str, xr.Dataset]) -> xr.Dataset:
    """Concatenate per-probe Datasets along a new ``probe`` dimension."""
    if not ds_dict:
        raise ValueError("merge_probes requires at least one probe Dataset")
    probes = list(ds_dict.keys())
    datasets = list(ds_dict.values())
    return xr.concat(datasets, dim="probe", join="outer").assign_coords(probe=probes)
