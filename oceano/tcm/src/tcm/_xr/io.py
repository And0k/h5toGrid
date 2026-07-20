"""
I/O helpers for the xarray-native pipeline.
"""
from __future__ import annotations
import fnmatch
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import pandas as pd
import xarray as xr

from tcm import _constants, csv_load, utils2init
from tcm._xr import storage, filters, dataset

lf = utils2init.LoggingStyleAdapter(__name__)

def load_raw(
    path: Optional[Union[str, Path]] = None,
    tbl: str = "",
    text_type: str = "i",
    cfg_in: Optional[Mapping[str, Any]] = None,
    chunk_time: Optional[int] = None,
) -> tuple[Optional[xr.Dataset], Optional[Dict[str, Any]]]:
    """Load raw inclinometer data from any supported format.

    Auto-detects format by file extension and dispatches to the
    appropriate backend.  Returns ``(ds, coefs)`` where *coefs* is
    ``None`` for CSV input (coefs come from a separate file) or a dict
    for HDF5/NC input (extracted from ``/{tbl}/coef/`` group).

    This is the **single public entry point** used by both the
    processing pipeline (:func:`tcm.processing.run`) and the
    calibration pipeline (:mod:`tcm.calibration.run`).

    Parameters
    ----------
    path
        Source file path. If none, cfg_in["path"] is used. Extension determines backend:

        * ``.nc`` / ``.nc4`` → :func:`_xr.dataset.open_nc`
        * ``.h5`` / ``.hdf5`` → :func:`open_hdf5`
        * ``.txt`` / ``.csv`` / ``.tsv`` → :func:`_xr.dataset.open_csv`

    tbl
        Table / group name inside the file (e.g. ``"incl63"``).
        For NC this is the HDF5-style group; for HDF5 it is the
        ``pandas.HDFStore`` key pattern; for CSV it is ignored.

    text_type
        Probe model letter for CSV parsing (``'i'``, ``'p'``, ``'b'``,
        ``'d'``, ``'w'``).  Ignored for NC/HDF5.

    cfg_in
        Resolved input config dict (``cfg.input`` as plain dict).  Used for
        **all** formats: ``time_ranges`` (window drop — NC/HDF5 only; CSV
        applies it via ``time_corr``), ``min``/``max`` (raw-column DROP),
        ``dt_hole_warning`` (gap check).  Keys like ``corr_time_mode``,
        ``dt_interp_between`` are wired automatically for CSV via
        ``csv_load.cfg_default["in"]`` merge.

    chunk_time
        If given, chunk the ``time`` dimension into blocks of this size.

    Returns
    -------
    ds
        :class:`xr.Dataset` or ``None`` if the file does not exist or
        yields no data.

    coefs
        Coefficient dict (for HDF5/NC) or ``None`` (for CSV).

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If *path* does not exist.

    Examples
    --------
    >>> from tcm._xr.io import load_raw
    >>> ds, coefs = load_raw("experiment.raw.nc", tbl="incl63")
    >>> ds, coefs = load_raw("experiment.raw.h5", tbl="incl63")
    >>> ds, coefs = load_raw("_raw/i63_01.txt", text_type="i")
    """
    if path is None:
        path = cfg_in["path"]
    path = Path(path)
    if (suffix := path.suffix.lower()) in _constants._EXT_NC:
        # ── NC ───────────────────────────────────────────────────────────────
        from tcm._xr.dataset import open_nc
        ds, coefs = open_nc(path, tbl=tbl, chunk_time=chunk_time)
        if cfg_in:  # NC has no time_corr — apply load-stage windowing + drop + hole check
            ds = filters.apply_load_time_ranges(ds, cfg_in.get("time_ranges"))
    elif suffix in _constants._EXT_HDF5:
        # ── HDF5 ─────────────────────────────────────────────────────────────
        ds, coefs = open_hdf5(path, table=tbl, chunk_time=chunk_time)
        if cfg_in:  # HDF5 has no time_corr — apply load-stage windowing + drop + hole check
            ds = filters.apply_load_time_ranges(ds, cfg_in.get("time_ranges"))
    elif suffix in _constants._EXT_CSV:
        # ── CSV / TXT ────────────────────────────────────────────────────────
        frames = []
        for ds_chunk, _meta in dataset.open_csv_chunks(
            path, text_type=text_type, cfg_in=cfg_in, chunk_time=chunk_time,
        ):
            frames.append(ds_chunk)
        if not frames:
            lf.warning("No data loaded from {}", path)
            return None, None

        ds = frames[0] if len(frames) == 1 else xr.concat(frames, dim="time")
        coefs = None  # CSV has no embedded coefs
        # CSV already applies time_ranges via time_corr during parsing — no re-apply here.
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}' for load_raw(). "
            f"Supported: {sorted(_constants._EXT_NC | _constants._EXT_HDF5 | _constants._EXT_CSV)}"
        )

    if ds is None:
        return None, coefs

    # Strip tz-aware datetime64 → naive before any downstream computation.
    # numpy cannot handle datetime64[ns, UTC] (pandas extension dtype);
    # .values on tz-aware coords returns an object array of datetime objects,
    # which crashes .astype("datetime64[ns]") in filters/binning/storage.
    ds = storage._strip_tz_datetime(ds)

    if cfg_in:  # apply raw-col DROP + hole check
        ds = filters.filter_global_minmax(ds, cfg_in)
        filters.warn_on_holes(ds, cfg_in.get("dt_hole_warning"))
    lf.info(
        "Loaded {}: {} ({:d} vars, {:d} rows)",
        suffix, path.name,
        len(ds.data_vars),
        ds.sizes.get("time", 0),
    )
    return ds, coefs



# --------------------------------------------------------------------------- #
# CSV → Dataset
# --------------------------------------------------------------------------- #

def load_csv_as_ds(
    path: Union[str, Path],
    *,
    index_col: str = "Time",
    parse_dates: bool = True,
    chunk_time: Optional[int] = None,
) -> xr.Dataset:
    """
    Load a CSV/TSV file into an :class:`xarray.Dataset`.

    Parameters
    ----------
    path
        CSV/TSV file path.
    index_col
        Column to use as the time index (default ``"Time"`` — matches TSV output convention).
    parse_dates
        Passed to :func:`pandas.read_csv`.
    chunk_time
        If given, chunk the ``time`` dimension into blocks of this size.

    Returns
    -------
    xr.Dataset
    """
    df = pd.read_csv(path, sep="\t", index_col=0, parse_dates=[0] if parse_dates else None)
    df.index.name = "time"  # normalise to xr-standard dimension name
    ds = xr.Dataset.from_dataframe(df)
    if chunk_time is not None:
        ds = ds.chunk({"time": chunk_time})
    return ds


# --------------------------------------------------------------------------- #
# Dataset → CSV
# --------------------------------------------------------------------------- #

def ds_to_csv(
    ds: xr.Dataset,
    path: Union[str, Path],
    *,
    split_period: Optional[str] = None,
    text_date_format: Optional[str] = None,
    text_columns: Optional[list] = None,
    float_format: str = "%.5g",
    sep: str = "\t",
) -> list[Path]:
    """Write Dataset to TSV/CSV file(s).

    Matches legacy ``dd_to_csv`` format: tab separator, ``%.5g`` floats,
    Vdir/inclination rounded to 4 decimals, Pressure to 3.
    Index column is always written as ``Time`` (legacy convention).

    Parameters
    ----------
    ds
        Dataset to write.
    path
        Output file path.
    split_period
        If given (e.g. ``"D"`` for daily), split output into separate files
        per period.  File names get a date suffix before the extension.
    text_date_format
        strftime format for the index.  If ``None``, index is written as-is.
    text_columns
        If set, only these columns appear in output.
    float_format
        Printf-style float format (default ``"%.5g"``).
    sep
        Column separator (default ``"\\t"`` = TSV).

    Returns
    -------
    list[Path]
        Paths of written files.
    """
    path = Path(path)
    df = ds.to_dataframe()

    # Unstack non-time index levels into columns — avoids MultiIndex on strftime/format
    extra_levels = [n for n in df.index.names if n != "time"]
    if extra_levels:
        df = df.unstack(extra_levels)
        # Flatten MultiIndex columns: ("Ax", "p1") → "Ax_p1"
        df.columns = df.columns.map(lambda t: "_".join(str(x) for x in t if str(x)))

    # Legacy convention: TSV index column header is "Time"
    if df.index.name == "time":
        df.index.name = "Time"

    # Rounding (match legacy: Vdir=4, inclination=4, Pressure=3)
    for col, decimals in [("Vdir", 4), ("inclination", 4), ("Pressure", 3)]:
        if col in df.columns:
            df[col] = df[col].round(decimals)

    # Column filtering
    if text_columns:
        cols = [c for c in text_columns if c in df.columns]
        df = df[cols]

    # Date formatting
    if text_date_format:
        df.index = df.index.strftime(text_date_format)

    csv_kwargs = dict(sep=sep, float_format=float_format, encoding="ascii")

    if split_period is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(df) > 1_000_000:
            _write_large_csv(df, path, csv_kwargs)
        else:
            df.to_csv(path, **csv_kwargs)
        return [path]

    # Split by time period
    written: list[Path] = []
    for period_label, group in df.groupby(pd.Grouper(freq=split_period)):
        if group.empty:
            continue
        date_str = period_label.strftime("%Y%m%d") if hasattr(period_label, "strftime") else str(period_label)
        stem = path.stem
        suffix = path.suffix
        out = path.parent / f"{stem}_{date_str}{suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(group) > 1_000_000:
            _write_large_csv(group, out, csv_kwargs)
        else:
            group.to_csv(out, **csv_kwargs)
        written.append(out)
    return written


def _write_large_csv(df: "pd.DataFrame", path: Path, csv_kwargs: dict, chunk_size: int = 100_000) -> None:
    """Write large DataFrame to CSV with tqdm progress bar."""
    from tqdm import tqdm

    n_rows = len(df)
    lf.info("Writing {} rows to {}", n_rows, path.name)
    for i in tqdm(range(0, n_rows, chunk_size), desc=f"Writing {path.name}", unit="chunk"):
        df.iloc[i : i + chunk_size].to_csv(
            path, mode="w" if i == 0 else "a", header=i == 0, **csv_kwargs,
        )


# --------------------------------------------------------------------------- #
# Dataset → netCDF
# --------------------------------------------------------------------------- #

def save_netcdf(
    ds: xr.Dataset,
    path: Union[str, Path],
    *,
    engine: str = "netcdf4",
) -> Path:
    """
    Persist a Dataset to netCDF.

    Parameters
    ----------
    ds
        Dataset to write.
    path
        Output path (``.nc``).
    engine
        Backend engine for :meth:`xarray.Dataset.to_netcdf`.

    Returns
    -------
    Path
        Written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = storage._drop_battery(ds)
    ds = storage._downcast_float32(ds)
    enc = {**storage._force_epoch(ds), **storage._compression_encoding(ds)}
    ds.to_netcdf(path, engine=engine, encoding=enc)
    return path


def open_netcdf(
    path: Union[str, Path],
    *,
    chunk_time: Optional[int] = None,
    engine: str = "netcdf4",
) -> xr.Dataset:
    """
    Open a netCDF file as a (possibly chunked) Dataset.

    Parameters
    ----------
    path
        Path to the ``.nc`` file.
    chunk_time
        If given, chunk the ``time`` dimension.
    engine
        Backend engine.

    Returns
    -------
    xr.Dataset
    """
    chunks = {"time": chunk_time} if chunk_time else None
    return xr.open_dataset(path, engine=engine, chunks=chunks)


# --------------------------------------------------------------------------- #
# Legacy HDF5 bridge
# --------------------------------------------------------------------------- #

def open_hdf5(
    path: Union[str, Path],
    table: str = "incl*",
    chunk_time: Optional[int] = None,
) -> tuple[xr.Dataset, Optional[Dict[str, Any]]]:
    """Load HDF5 table and coefs into an xarray Dataset via pandas.

    Reads data from matching tables and attempts to extract coefs from
    ``/{tbl}/coef/`` groups (same structure as :func:`load_coefs`).

    :param path: HDF5 file path.
    :param table: Table name or pattern for :func:`pandas.read_hdf`.
    :param chunk_time: If given, chunk the ``time`` dimension.
    :return: ``(ds, coefs_dict)`` — coefs_dict is ``None`` when no coef group.
    """
    from tcm._constants import TABLES_AVAILABLE
    if not TABLES_AVAILABLE:
        raise ImportError(
            "pytables (tables) required to open HDF5 files — "
            "install pytables or use NC/CSV input"
        )
    from tcm.incl_calc.coefs import load_coefs

    if not (path := Path(path)).is_file():
        lf.warning("{} does not exist", path)
        return None, None
    store = pd.HDFStore(path, mode="r")
    try:
        matching = [k for k in store.keys() if fnmatch.fnmatch(k, table)]
        if not matching:
            raise KeyError(f"No table matching '{table}' in {path}")
        frames = [store.select(k) for k in matching]
    finally:
        store.close()

    df = frames[0] if len(frames) == 1 else pd.concat(frames)

    if df.index.name != "time":
        df.index.name = "time"

    ds = xr.Dataset.from_dataframe(df)
    if chunk_time is not None:
        ds = ds.chunk({"time": chunk_time})

    # Extract coefs from first matching table that has them
    coefs = None
    for key in matching:
        tbl_name = key.lstrip("/")
        coefs = load_coefs(Path(path), tbl_name)
        if coefs is not None:
            lf.debug("Extracted coefs from {} (table={})", path, tbl_name)
            break

    return ds, coefs
