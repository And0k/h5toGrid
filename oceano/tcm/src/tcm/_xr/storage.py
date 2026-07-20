"""Persistent storage helpers for the xarray-native pipeline.

Provides raw/processed netCDF persistence with incremental-update support
and NC log table I/O (replacing HDF5 log tables).
Replaces HDF5-based storage from ``_dask_legacy`` with netCDF4.
"""
from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from tcm import  _constants, utils2init

_h5py = _constants._h5py
lf = utils2init.LoggingStyleAdapter(__name__)


def _nc_guard(operation: str) -> bool:
    """Check whether binary (NC/HDF5) I/O is enabled.

    Returns ``True`` when I/O should proceed (``use_h5 is True``).
    Returns ``False`` and logs appropriately when disabled:

    * ``use_h5 is False`` → **warning** (user override or forced by missing libs).
    * ``use_h5 is None``  → **silent** skip (auto-detected unavailable).
    """
    bio = _constants.use_h5_get()
    if bio is True:
        return True
    if bio is False:
        lf.warning("{} — skipped (use_h5=False)", operation)
    return False

# CF-standard time encoding — single source of truth for all NC I/O.
# Both data groups and log tables use this encoding so that the same
# datetime value has identical on-disk representation everywhere.
# Change the epoch to switch all on-disk time encoding at once.
_EPOCH_NS: int = np.datetime64((_ := "1970-01-01"), "ns").astype(np.int64)
_CF_TIME_UNITS = f"seconds since {_}"  # str(_EPOCH_NS.astype("datetime64[ns]").astype("datetime64[D]"))
_CF_CALENDAR = "proleptic_gregorian"

# Compression defaults for all data variables with meaningful sizes.
# Applied automatically — no scale_factor/add_offset (avoid precision loss).
_ZLIB_CFG: dict[str, Any] = {
    "zlib": True,
    "complevel": 9,
    "shuffle": True,
    "fletcher32": True,
}


def _force_epoch(ds: xr.Dataset) -> dict[str, Any]:
    """Return xarray encoding dict forcing CF-standard epoch on ``time`` coord."""
    if "time" not in ds.coords:
        return {}
    return {"time": {"units": _CF_TIME_UNITS, "calendar": _CF_CALENDAR, "dtype": "f8"}}


def _compression_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Return xarray encoding dict with zlib compression for each data variable.

    Only numeric variables (float/int, not string/datetime) get compression.
    The ``time`` coordinate is handled separately by :func:`_force_epoch`.
    No ``scale_factor`` / ``add_offset`` — avoid implicit dtype change.
    """
    enc: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        if da.dtype.kind in ("f", "i", "u"):
            enc[name] = {**_ZLIB_CFG, "dtype": "float32"}
    return enc


def _downcast_float32(ds: xr.Dataset) -> xr.Dataset:
    """Convert all float64 data variables to float32 in-place.

    Coordinate variables (e.g. ``time``) are left unchanged.
    Float32 is sufficient for all measured geophysical data and halves
    on-disk size.  Applied before every NC write.
    """
    for name, da in ds.data_vars.items():
        if da.dtype == np.float64:
            ds[name] = da.astype(np.float32)
    return ds


def _drop_battery(ds: xr.Dataset) -> xr.Dataset:
    """Drop ``Battery`` variable from Dataset (non-raw outputs only).

    Battery is retained in ``*.raw.nc`` for completeness but excluded
    from all processed/binned outputs and TSV exports.
    """
    if "Battery" in ds.data_vars:
        ds = ds.drop_vars("Battery")
    return ds


def _dt_ns_to_cf(arr: np.ndarray) -> np.ndarray:
    """datetime64[ns] → float64 seconds since ``_CF_TIME_UNITS`` epoch (for h5py write)."""
    return (arr.astype("datetime64[ns]").astype(np.int64) - _EPOCH_NS) / 1e9


def _cf_to_dt_ns(raw: np.ndarray, units: Union[str, bytes] = "") -> np.ndarray:
    """float64 seconds OR legacy int64 ns → datetime64[ns] (for h5py read).

    Backward-compatible: inspects dtype to handle both CF-float64 and
    legacy-int64 formats.  For float64, parses the epoch date from *units*
    (e.g. ``"seconds since 1970-01-01"``).  When *units* is empty the
    legacy-1970 path is used.  Accepts ``bytes`` (from h5py attrs).
    """
    if isinstance(units, bytes):
        units = units.decode()
    if raw.dtype.kind == "f":
        epoch_ns = (
            np.datetime64(units.removeprefix("seconds since "), "ns").astype(np.int64)
            if units else 0
        )
        return ((raw * 1e9).astype(np.int64) + epoch_ns).astype("datetime64[ns]")
    # Legacy int64 path — nanoseconds since 1970 (old h5py format)
    return raw.astype(np.int64).astype("datetime64[ns]")


def _write_time_ds(grp: "_h5py.Group", name: str, data: np.ndarray) -> None:
    """Create a CF-standard time dataset (float64, ``"seconds since ..."``)."""
    ds = grp.create_dataset(name, data=data, dtype="f8")
    ds.attrs["units"] = _CF_TIME_UNITS
    ds.attrs["calendar"] = _CF_CALENDAR


# --------------------------------------------------------------------------- #
# NC log table I/O
# --------------------------------------------------------------------------- #

def read_nc_log(nc_path: Union[str, Path], tbl: str, tables_log: str = "{}/logFiles") -> xr.Dataset:
    """Read log from ``/{tbl}/{log_group}`` group in a NC4 file.

    Returns an :class:`xr.Dataset` with dimension ``Date0`` (``datetime64[ns]``)
    and variables: ``fileName`` (str), ``fileChangeTime``, ``DateEnd``,
    ``DateProc`` (all ``datetime64[ns]``).
    Returns empty Dataset (zero-length ``Date0``) when group is missing or
    h5py unavailable.

    Parameters
    ----------
    nc_path
        NC4 file path.
    tbl
        Table name (e.g. ``"incl63"``).
    tables_log
        Log group name template.  Default ``"{}/logFiles"`` — ``{}`` is
        replaced by *tbl*.  Can be overridden by ``cfg.out.tables_log``.
    """
    nc_path = Path(nc_path)
    empty = xr.Dataset(coords={"Date0": np.array([], dtype="datetime64[ns]")})
    if not _nc_guard("NC log read"):
        return empty
    if not nc_path.exists():
        return empty

    log_grp_path = tables_log.format(tbl)
    with _h5py.File(nc_path, "r") as f:
        if log_grp_path not in f:
            return empty
        grp = f[log_grp_path]
        n = grp["fileName"].shape[0] if "fileName" in grp else 0
        if n == 0:
            return empty

        # Read variables — _cf_to_dt_ns handles both CF float64 and legacy int64
        file_names = [v.decode() if isinstance(v, bytes) else str(v) for v in grp["fileName"][:]]
        date0 = _cf_to_dt_ns(grp["Date0"][:], grp["Date0"].attrs.get("units", ""))

        data_vars: dict[str, Any] = {"fileName": ("Date0", file_names)}
        for col in ("fileChangeTime", "DateEnd", "DateProc"):
            if col in grp:
                data_vars[col] = ("Date0", _cf_to_dt_ns(grp[col][:], grp[col].attrs.get("units", "")))

    lf.debug("Read {} log entries from {}//{}", n, nc_path, log_grp_path)
    return xr.Dataset(data_vars, coords={"Date0": date0})


def write_nc_log(
    nc_path: Union[str, Path],
    tbl: str,
    log: xr.Dataset,
    tables_log: str = "{}/logFiles",
) -> None:
    """Write or overwrite log Dataset to ``/{tbl}/{log_group}`` in a NC4 file.

    Inverse of :func:`read_nc_log`.  Datetime variables stored as CF-standard
    float64 seconds (``"seconds since 1970-01-01"``) — same encoding as data
    groups in :func:`_write_dataset_to_nc_group`.  No-op when h5py unavailable.

    Parameters
    ----------
    nc_path
        NC4 file path.
    tbl
        Table name (e.g. ``"incl63"``).
    log
        Log Dataset to write.
    tables_log
        Log group name template.  Default ``"{}/logFiles"`` — ``{}`` is
        replaced by *tbl*.  Can be overridden by ``cfg.out.tables_log``.
    """
    if not _nc_guard("NC log write"):
        return
    nc_path = Path(nc_path)
    log_grp_path = tables_log.format(tbl)

    with _h5py.File(nc_path, "a") as f:
        # Remove old log group if present (full overwrite)
        if log_grp_path in f:
            del f[log_grp_path]
        grp = f.create_group(log_grp_path)

        n = log.sizes.get("Date0", 0)
        # Variable-length UTF-8 string type — supports Cyrillic, CJK, etc.
        str_dt = _h5py.string_dtype(encoding="utf-8")

        if n == 0:
            grp.create_dataset("fileName", data=np.array([], dtype=object), dtype=str_dt)
            _write_time_ds(grp, "Date0", np.array([], dtype=np.float64))
            f.flush()
            return

        # Date0 — CF-standard float64 seconds
        _write_time_ds(grp, "Date0", _dt_ns_to_cf(log["Date0"].values))
        # fileName as variable-length UTF-8 strings
        grp.create_dataset(
            "fileName", data=[str(v) for v in log["fileName"].values], dtype=str_dt,
        )

        # Datetime columns — same CF encoding
        for col in ("fileChangeTime", "DateEnd", "DateProc"):
            if col in log:
                _write_time_ds(grp, col, _dt_ns_to_cf(log[col].values))

        f.flush()

    lf.debug("Wrote {} log entries to {}//{}", n, nc_path, log_grp_path)


# --------------------------------------------------------------------------- #
# Timezone-aware datetime → naive conversion for netCDF compat
# --------------------------------------------------------------------------- #

def _strip_tz_datetime(ds: xr.Dataset) -> xr.Dataset:
    """Convert tz-aware datetime64 coordinates/vars to naive.

    ``datetime64[ns, UTC]`` is a pandas extension dtype that numpy cannot
    interpret (TypeError in ``np.issubdtype``).  xarray's netCDF writer
    hits this during CF encoding, so we strip timezone before persist.

    Uses ``pd.DatetimeIndex.tz_localize(None)`` to drop timezone info,
    then converts to ``datetime64[ns]`` for netCDF compatibility.
    """
    for coord_name in list(ds.coords):
        coord = ds.coords[coord_name]
        try:
            if coord.dtype.tz is None:
                continue
        except (TypeError, AttributeError):
            continue
        naive = pd.DatetimeIndex(coord.values).tz_localize(None).to_numpy(dtype="datetime64[ns]")
        ds = ds.assign_coords({coord_name: naive})

    # Fix tz-aware data variables (uncommon but defensive)
    for var_name in list(ds.data_vars):
        var = ds[var_name]
        try:
            if var.dtype.tz is None:
                continue
        except (TypeError, AttributeError):
            continue
        naive = pd.DatetimeIndex(var.values).tz_localize(None).to_numpy(dtype="datetime64[ns]")
        ds[var_name] = naive

    return ds


def store_raw(
    ds: xr.Dataset,
    path: Union[str, Path],
    attrs: Optional[Dict[str, Any]] = None,
    engine: str = "netcdf4",
) -> Path:
    """Write a raw Dataset to netCDF with global attributes.

    Parameters
    ----------
    ds
        Dataset to persist.
    path
        Output ``.nc`` path.
    attrs
        Global attributes merged into ``ds.attrs`` before writing.
    engine
        netCDF backend.

    Returns
    -------
    Path
        Written file path.
    """
    path = Path(path)
    if not _nc_guard("NC raw write"):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if attrs:
        ds = ds.assign_attrs(attrs)
    ds = _strip_tz_datetime(ds)
    ds = _downcast_float32(ds)
    enc = {**_force_epoch(ds), **_compression_encoding(ds)}
    try:
        ds.to_netcdf(path, engine=engine, encoding=enc)
    except (ImportError, ValueError) as e:
        lf.warning("NC write failed (engine={}): {} — skipped {}", engine, e, path)
        return path
    lf.info(
        "Stored raw: {:s} ({:d} vars, {:d} time steps)", str(path), len(ds.data_vars), ds.sizes.get("time", 0)
    )
    return path


def store_processed(
    ds: xr.Dataset,
    path: Union[str, Path],
    *,
    group: Optional[str] = None,
    mode: str = "w",
    engine: str = "netcdf4",
) -> Path:
    """Write (or append) a processed Dataset to netCDF.

    Parameters
    ----------
    ds
        Dataset to persist.
    path
        Output ``.nc`` path.
    group
        NetCDF4 group name (e.g. ``"i_01"``).  When set, writes into
        ``/{group}/`` within the file — enabling per-probe groups in a
        shared ``*.proc.nc``.
    mode
        ``"w"`` (default) overwrites; ``"a"`` appends variables to an
        existing file (requires ``scipy`` or ``netcdf4`` engine).
    engine
        netCDF backend.

    Returns
    -------
    Path
        Written file path.
    """
    path = Path(path)
    if not _nc_guard("NC processed write"):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = _drop_battery(ds)
    ds = _strip_tz_datetime(ds)
    ds = _downcast_float32(ds)
    enc = {**_force_epoch(ds), **_compression_encoding(ds)}
    try:
        ds.to_netcdf(path, group=group, mode=mode, engine=engine, encoding=enc)
    except (ImportError, ValueError, OSError) as e:
        # ValueError: dimension size mismatch on mode='a'
        # OSError: netCDF4/h5py file-handle conflict (e.g. after h5py rebuild)
        if mode == "a" and group and _constants.use_h5_get() is True:
            try:
                _append_to_nc_group(ds, path, group)
                lf.info(
                    "Appended processed to {:s}//{} ({:d} rows via h5py resize)",
                    str(path), group, ds.sizes.get("time", 0),
                )
                return path
            except Exception:
                lf.debug("h5py append also failed for {}//{}", path, group, exc_info=True)
        lf.warning("NC write failed (engine={}): {} — skipped {}", engine, e, path)
        return path
    lf.info(
        "Stored processed in {:s} (group={}, mode={:s}, {:d} vars)",
        str(path), group or "(root)", mode, len(ds.data_vars),
    )
    return path


def store_processed_incremental(
    ds: xr.Dataset,
    path: Union[str, Path],
    *,
    group: str,
    mode: str = "a",
    engine: str = "netcdf4",
    filter_params: Optional[str] = None,
    force_reprocess: bool = False,
) -> Path:
    """Write processed data only if group doesn't already cover this time range.

    Checks existing group's time range before writing.  Skips if the new data
    is fully contained within the existing range (avoids duplicates on re-run).

    **Re-run warning**: when *filter_params* is given, stores it as the
    ``_run_params`` attribute on the group (sorted text of filter, window,
    and coefficient params). On skip, if the stored text differs from
    *filter_params*, prints a unified diff warning so the user knows the
    run params changed since the last write. Set *force_reprocess* to
    ``True`` to bypass the containment skip + force overwrite of overlapping
    data.

    Uses h5py for the time-range check to avoid netCDF4/h5py file-handle
    conflicts when the same file was previously modified by h5py (e.g. after
    a resize-fallback rebuild).

    Parameters
    ----------
    ds
        Dataset to persist.
    path
        Output ``.nc`` path.
    group
        NetCDF4 group name (required — incremental is per-group).
    mode
        Write mode (default ``"a"`` for append).
    engine
        netCDF backend.
    filter_params
        Full sorted text representation of the resolved filtering parameters
        (filter + window).  Stored as ``_filter_params`` attr; on skip,
        compared to the stored value — diff-printed if they differ.
    force_reprocess
        When ``True``, bypass the containment skip and force overwrite of
        the overlapping group (uses ``mode="w"`` for the group).

    Returns
    -------
    Path
        Written file path (or existing path if skipped).
    """
    path = Path(path)
    if not _nc_guard("NC incremental write"):
        return path

    # Strip tz early — .values on tz-aware coords returns object arrays that
    # crash .astype("datetime64[ns]") / .view(np.int64) below.
    ds = _strip_tz_datetime(ds)

    if path.exists():
        try:
            with _h5py.File(str(path), "r") as f:
                if group in f and "time" in f[group]:
                    time_dset = f[group]["time"]
                    if time_dset.shape[0] > 0:
                        ex_ns = _cf_to_dt_ns(
                            time_dset[:], time_dset.attrs.get("units", ""),
                        ).astype(np.int64)
                        new_ns = ds["time"].values.astype("datetime64[ns]").astype(np.int64)
                        contained = new_ns.min() >= ex_ns.min() and new_ns.max() <= ex_ns.max()

                        # Re-run warning: compare stored run params (filter + coefs text) to current
                        stored_params = f[group].attrs.get("_run_params", "")
                        if isinstance(stored_params, bytes):
                            stored_params = stored_params.decode()
                        if contained and not force_reprocess and filter_params and stored_params != filter_params:
                            _warn_run_params_diff(group, path, stored_params, filter_params)

                        if contained and not force_reprocess:
                            lf.debug(
                                "Skipping {} — data already covered ({} to {})",
                                group,
                                ex_ns.min().astype("datetime64[ns]"),
                                ex_ns.max().astype("datetime64[ns]"),
                            )
                            return path
        except (AttributeError, KeyError, OSError):
            pass  # group doesn't exist yet or is corrupted — fall through to write

    # force_reprocess: overwrite the group entirely (mode="w" on the group)
    write_mode = "w" if force_reprocess else mode
    ds_out = ds
    if ds_out.sizes.get("time", 0):
        # Deduplicate at float64-seconds precision (CF encoding resolution).
        # If timestamps differing by < utils_time_corr.DT_CF_NS which are unique
        # at datetime64[ns] precision but collapse to the same float64-second value
        # when written to netCDF.  Monotonicity check is O(n) vs O(n·log n) for np.unique.
        # Defensive: utils_time_corr.snap_to_grid already aware of DT_CF_NS
        t_cf = (ds_out["time"].values.view(np.int64) - _EPOCH_NS).astype(np.float64) / 1e9
        mono = np.ones(t_cf.size, dtype=bool)
        mono[1:] = np.diff(t_cf) > 0
        if not mono.all():
            n_near_dup = int((~mono).sum())
            lf.warning(
                "{}//{}: removing {} sub-μs near-duplicate time(s) (CF float64 resolution)",
                path.name, group, n_near_dup,
            )
            ds_out = ds_out.isel(time=mono)
    if filter_params:
        ds_out = ds_out.assign_attrs(_run_params=filter_params)
    return store_processed(ds_out, path, group=group, mode=write_mode, engine=engine)


def _warn_run_params_diff(group: str, path: Path, stored: str, current: str) -> None:
    """Print unified diff between stored and current run params, then warn.

    Run params include filter, window, and coefficient text built by
    :func:`tcm.processing._build_filter_params_text`. Stored as ``_run_params``
    attribute on the NC group; on skip, current text is diff-compared to stored
    to surface changes in filter, input window, or coefficients.
    """
    import difflib

    diff_lines = list(difflib.unified_diff(
        stored.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile="stored (last write)",
        tofile="current (this run)",
        n=3,
    ))
    diff_text = "".join(diff_lines) if diff_lines else ""
    lf.warning(
        "Run params changed since last write — re-run with +force_reprocess=True "
        "to overwrite (or Delete {}//{}). Diff:\n{}",
        path.name, group, diff_text,
    )


def open_processed(
    path: Union[str, Path],
    *,
    chunks: Optional[int] = None,
    engine: str = "netcdf4",
) -> xr.Dataset:
    """Open a processed netCDF file, optionally with dask chunking.

    Parameters
    ----------
    path
        Path to ``.nc`` file.
    chunks
        If given, chunk the ``time`` dimension by this many steps.
    engine
        netCDF backend.

    Returns
    -------
    xr.Dataset
    """
    chunk_spec = {"time": chunks} if chunks else None
    return xr.open_dataset(path, engine=engine, chunks=chunk_spec)


def open_processed_grouped(
    path: Union[str, Path],
    *,
    chunks: Optional[int] = None,
    engine: str = "netcdf4",
) -> dict[str, xr.Dataset]:
    """Open per-probe groups from a shared ``*.proc.nc`` file.

    Returns ``{group_name: Dataset}`` for each top-level group that
    contains a ``time`` dimension.  Use with :func:`merge_probes` to
    combine into a single Dataset with a ``probe`` dimension.
    """
    if _constants.use_h5_get() is not True:
        raise ImportError("cannot list groups in NC4 files (use_h5 wasn't set True)")

    path = Path(path)
    chunk_spec = {"time": chunks} if chunks else None
    result: dict[str, xr.Dataset] = {}

    with _h5py.File(path, "r") as f:
        groups = [k for k in f.keys() if isinstance(f[k], _h5py.Group) and "time" in f[k]]

    for grp in groups:
        result[grp] = xr.open_dataset(path, group=grp, engine=engine, chunks=chunk_spec)

    lf.debug("Opened {} groups from {}", len(result), path)
    return result


def incremental_skip(
    path: Union[str, Path],
    input_mtime: float,
) -> bool:
    """Check whether output is already up-to-date relative to input.

    Returns ``True`` when the output file exists **and** is newer than
    *input_mtime* (i.e. processing can be skipped).

    Parameters
    ----------
    path
        Output file path to check.
    input_mtime
        ``os.stat().st_mtime`` of the input file.

    Returns
    -------
    bool
    """
    path = Path(path)
    if not path.exists():
        return False
    output_mtime = path.stat().st_mtime
    is_up_to_date = output_mtime > input_mtime
    if is_up_to_date:
        lf.debug("Skipping {:s}: output is up-to-date", str(path))
    return is_up_to_date


# --------------------------------------------------------------------------- #
# NC incremental append — position-aware, never re-sort
# --------------------------------------------------------------------------- #

class _Overlap(NamedTuple):
    """`new_t` split against existing `[ex_min, ex_max]` into the only two slices worth keeping.

    *existing* is authoritative and never re-sorted or overwritten, so any of `new_t` already
    inside its span carries no new information — only `.head` (strictly < ex_min) and `.tail`
    (strictly > ex_max) can ever be written.  Which of the two is non-empty *is* the relation
    (:attr:`rel`), derived once here instead of re-classified by every caller: this is the single
    source of truth for both the write action and the log label, not a second switch downstream.
    """
    head_end: int    # new_t[:head_end]   — PREPEND candidate, strictly < ex_min
    tail_start: int  # new_t[tail_start:] — APPEND candidate,  strictly > ex_max
    n: int           # len(new_t)

    @classmethod
    def of(cls, new_t: np.ndarray, ex_min: np.int64, ex_max: np.int64) -> "_Overlap":
        """Classify sorted *new_t* against existing range — O(log n) via searchsorted, no copy."""
        return cls(new_t.searchsorted(ex_min, side="left"), new_t.searchsorted(ex_max, side="right"), new_t.size)

    @property
    def head(self) -> slice:
        return slice(0, self.head_end)

    @property
    def tail(self) -> slice:
        return slice(self.tail_start, None)

    @property
    def has_head(self) -> bool:
        return bool(self.head_end > 0)  # np.searchsorted → np.intp; cast needed since `match`

    @property
    def has_tail(self) -> bool:
        return bool(self.tail_start < self.n)  # below tests True/False by identity (`is`), not `==`

    @property
    def rel(self) -> str:
        """Allen-relation name for logging — en.wikipedia.org/wiki/Allen's_interval_algebra.

        One of BEFORE/OVERLAP_HEAD/CONTAINED/WRAPS/OVERLAP_TAIL/AFTER.  Plain `str`, not an enum:
        nothing branches on this beyond a log line and the warn/debug check in `append_to_nc` —
        an enum class would be six named integers doing a string's job.
        """
        match self.has_head, self.has_tail:
            case False, False: return "CONTAINED"    # new ⊆ existing (equality included)
            case True, True: return "WRAPS"          # new ⊃ existing, extends past both sides
            case True, False: return "BEFORE" if self.head_end == self.n else "OVERLAP_HEAD"
            case False, True: return "AFTER" if self.tail_start == 0 else "OVERLAP_TAIL"

    def write(self, ds_new: xr.Dataset, nc_path: Path, tbl: str) -> bool:
        """Execute the write this relation *is* — the only place condition and action now meet.

        `has_head`/`has_tail` drive this directly; `.rel` plays no role here, it only labels
        the outcome for :func:`append_to_nc`'s log line.
        """
        if self.has_head:
            _prepend_nc_group(_strip_tz_datetime(ds_new.isel(time=self.head)), nc_path, tbl)
        if self.has_tail:
            _append_to_nc_group(_strip_tz_datetime(ds_new.isel(time=self.tail)), nc_path, tbl)
        return self.has_head or self.has_tail


def _fresh_write(ds_new: xr.Dataset, nc_path: Path, tbl: str, reason: str) -> bool:
    """Write *ds_new* as a brand-new group — shared by every "nothing to compare against" guard."""
    ds_new = _strip_tz_datetime(ds_new)
    _write_dataset_to_nc_group(ds_new, nc_path, tbl)
    lf.debug("{} {}//{}: fresh write {:d} time steps", reason, nc_path, tbl, ds_new.sizes.get("time", 0))
    return True


def append_to_nc(
    ds_new: xr.Dataset,
    nc_path: Union[str, Path],
    tbl: str,
) -> bool:
    """Append *ds_new* to ``/{tbl}/`` group in an existing NC4 file.  Returns whether data was written.

    Position-aware: classifies *ds_new* against the existing time range via :class:`_Overlap` and
    acts on its two derived slices directly — **never re-sorts**, and nothing here re-derives the
    relation to decide what to do:

    - no existing data → fresh group write.
    - ``.head`` non-empty (new < existing start) → streaming h5py prepend.
    - ``.tail`` non-empty (new > existing end) → h5py resize+append, no re-read of existing data.
    - both non-empty (:attr:`_Overlap.rel` is ``WRAPS`` — *existing* is straddled) → both, in order.
    - neither (*ds_new* ⊆ *existing*) → skip, nothing new.

    All file I/O uses h5py exclusively to avoid HDF5 file-handle conflicts
    between h5py and netCDF4 backends within a single process.
    """
    if not _nc_guard("NC append"):
        return False
    nc_path = Path(nc_path)

    # Strip tz early — .values on tz-aware coords returns object arrays that
    # crash .astype("datetime64[ns]") / .view(np.int64) below.
    ds_new = _strip_tz_datetime(ds_new)

    # Read existing time range using h5py (avoids netCDF4 cache conflicts). FileNotFoundError/KeyError
    # are routine — first write for this file/table; a bare OSError means something's actually wrong.
    needs_fresh_write = None  # reason string, or None
    try:
        with _h5py.File(nc_path, "r") as f:
            if tbl not in f or "time" not in f[tbl]:
                needs_fresh_write = "No group —"
            elif (time_dset := f[tbl]["time"]).shape[0] == 0:
                needs_fresh_write = "Empty existing group —"
            else:
                ex_ns = _cf_to_dt_ns(time_dset[:], time_dset.attrs.get("units", "")).astype(np.int64)
                # Diagnostic: log shapes of all datasets in group (detects transposition/corruption)
                shapes = {n: f[tbl][n].shape for n in f[tbl] if isinstance(f[tbl][n], _h5py.Dataset)}
                lf.debug("Existing {}//{} shapes: {}", nc_path, tbl, shapes)
    except FileNotFoundError:
        needs_fresh_write = "No file —"
    except KeyError:
        needs_fresh_write = "No existing group/time —"
    except OSError:
        lf.exception("{}//{}: existing file unreadable — falling back to fresh write", nc_path, tbl)
        needs_fresh_write = "Unreadable —"

    # Call _fresh_write OUTSIDE the h5py read block to avoid Windows HDF5 mandatory locking
    if needs_fresh_write is not None:
        return _fresh_write(ds_new, nc_path, tbl, needs_fresh_write)

    new_ns = ds_new["time"].values.astype("datetime64[ns]").astype(np.int64)
    ov = _Overlap.of(new_ns, ex_ns[0], ex_ns[-1])
    (lf.warning if (rel := ov.rel) not in ("BEFORE", "AFTER", "CONTAINED") else lf.debug)(
        "{}//{}: {} — new [{}, {}] vs existing [{}, {}], keeping head={:d}/tail={:d} of {:d}",
        nc_path, tbl, rel,
        new_ns[0].astype("datetime64[ns]"), new_ns[-1].astype("datetime64[ns]"),
        ex_ns[0].astype("datetime64[ns]"), ex_ns[-1].astype("datetime64[ns]"),
        ov.head_end, ov.n - ov.tail_start, ov.n,
    )
    return ov.write(ds_new, nc_path, tbl)  # no-op, returns False when CONTAINED (both slices empty)


def _append_to_nc_group(
    ds_new: xr.Dataset,
    nc_path: Path,
    tbl: str,
) -> None:
    """Extend existing NC4 group datasets with *ds_new* data.

    Reads existing group shape, resizes each h5py dataset, and writes
    new rows at the end — no re-read, no re-write of existing data.
    Requires datasets to be created with ``maxshape=(None,)`` (new
    files satisfy this automatically; pre-existing files are rebuilt).
    """
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    ds_new = _downcast_float32(ds_new)

    # Prepare numpy arrays for writing
    time_cf = _dt_ns_to_cf(ds_new["time"].values)
    var_arrays: dict[str, np.ndarray] = {"time": time_cf}
    for name, da in ds_new.data_vars.items():
        vals = da.values
        if vals.dtype.kind in ("U", "S"):
            vals = vals.astype("S")
        var_arrays[name] = vals
    n_new = len(time_cf)

    try:
        _h5py_extend_group(nc_path, tbl, n_new, var_arrays)
    except (TypeError, KeyError) as exc:
        # Fallback: datasets lack maxshape → rebuild group without full concat
        lf.debug("Resize failed for {}//{} ({}): rebuilding group", nc_path, tbl, exc)
        _rebuild_and_append(nc_path, tbl, var_arrays, n_new)
        lf.info("Rebuilt {}//{} (+{:d} rows, resize fallback)", nc_path, tbl, n_new)


def _h5py_extend_group(
    nc_path: Path, tbl: str, n_new: int, var_arrays: dict[str, np.ndarray],
) -> None:
    """Low-level h5py resize+write for extendable datasets.

    Extends ALL time-indexed datasets in the group — not just those in
    *var_arrays*.  Variables absent from *var_arrays* are padded with their
    fill value (or NaN/0) to keep the time dimension consistent.
    """
    with _h5py.File(str(nc_path), "a") as f:
        grp = f[tbl]
        # Extend time dataset
        time_dset = grp["time"]
        n_old = time_dset.shape[0]
        n_total = n_old + n_new
        time_dset.resize(n_total, axis=0)
        time_dset[n_old:] = var_arrays["time"]

        # Extend ALL time-indexed 1-D datasets (not just those in var_arrays)
        for name in grp:
            if name == "time":
                continue
            dset = grp[name]
            if not isinstance(dset, _h5py.Dataset) or dset.ndim != 1:
                continue
            n_d = dset.shape[0]
            if n_d == n_total:
                continue  # already extended (e.g. by a previous call)
            if n_d != n_old:
                lf.warning(
                    "Dataset {}//{} size {} <> time size {} — skipping extend",
                    tbl, name, n_d, n_old,
                )
                continue
            dset.resize(n_total, axis=0)
            if name in var_arrays:
                dset[n_old:] = var_arrays[name]
            else:
                # Variable absent from new data — fill with dtype-appropriate value
                fill = np.nan if dset.dtype.kind == "f" else np.zeros((), dtype=dset.dtype)
                dset[n_old:] = np.full(n_new, fill, dtype=dset.dtype)

        # Re-attach HDF5 dimension scales — h5py resize can strip the
        # DIMENSION_LIST attribute that netCDF4 requires to map variables
        # to their coordinate dimensions.  Without this, xr.open_dataset
        # with engine="netcdf4" raises AttributeError on NoneType.dimensions.
        time_dset.make_scale("time")
        for name in grp:
            if name == "time" or not isinstance(grp[name], _h5py.Dataset):
                continue
            dset = grp[name]
            if dset.ndim >= 1:
                dset.dims[0].attach_scale(time_dset)

        f.flush()
    lf.debug("Extended {}//{} by {:d} rows (total {:d})", nc_path, tbl, n_new, n_total)


def _rebuild_and_append(
    nc_path: Path, tbl: str, var_arrays: dict[str, np.ndarray],
    n_new: int, chunk: int = 50_000,
) -> None:
    """Rebuild an NC4 group with extendable datasets and append new rows.

    Used when h5py ``resize()`` fails (non-extendable datasets).
    Reads existing rows in chunks via h5py, creates a new group with
    ``maxshape=(None,)``, copies old data chunk-wise, then appends new data.
    Never loads the full dataset into RAM.
    """
    tmp_grp = f"_{tbl}_rebuild"
    with _h5py.File(str(nc_path), "a") as f:
        old_grp = f[tbl]
        n_old = old_grp["time"].shape[0]
        n_total = n_old + n_new

        # Create replacement group with extendable datasets
        if tmp_grp in f:
            del f[tmp_grp]
        new_grp = f.create_group(tmp_grp)

        # Time coordinate
        time_dset = new_grp.create_dataset("time", shape=(n_total,), dtype="f8", maxshape=(None,))
        time_dset.attrs["units"] = _CF_TIME_UNITS
        time_dset.attrs["calendar"] = _CF_CALENDAR
        time_dset.make_scale("time")

        # Data variables — copy shape/dtype/attrs, create extendable + compressed
        var_names = [n for n in old_grp if isinstance(old_grp[n], _h5py.Dataset) and n != "time"]
        new_dsets = {}
        for name in var_names:
            old_ds = old_grp[name]
            kw: dict[str, Any] = dict(
                shape=(n_total,) + old_ds.shape[1:],
                dtype=old_ds.dtype,
                maxshape=(None,) + old_ds.shape[1:],
            )
            if old_ds.dtype.kind in ("f", "i", "u"):
                kw.update(compression="gzip", compression_opts=9, shuffle=True, fletcher32=True)
            new_ds = new_grp.create_dataset(name, **kw)
            for k, v in old_ds.attrs.items():
                new_ds.attrs[k] = v
            new_ds.dims[0].attach_scale(new_grp["time"])
            new_dsets[name] = new_ds

        # Copy existing data chunk-wise
        for start in range(0, n_old, chunk):
            end = min(start + chunk, n_old)
            new_grp["time"][start:end] = old_grp["time"][start:end]
            for name in var_names:
                new_dsets[name][start:end] = old_grp[name][start:end]

        # Append new data
        new_grp["time"][n_old:] = var_arrays["time"]
        for name, arr in var_arrays.items():
            if name == "time" or name not in new_dsets:
                continue
            new_dsets[name][n_old:] = arr

        # Swap groups
        del f[tbl]
        f.move(tmp_grp, tbl)
        f.flush()
    lf.debug("Rebuilt {}//{}: {:d} old + {:d} new = {:d} total", nc_path, tbl, n_old, n_new, n_total)


def _prepend_nc_group(
    ds_new: xr.Dataset, nc_path: Path, tbl: str, chunk: int = 50_000,
) -> None:
    """Prepend *ds_new* before existing data — streaming, O(chunk) memory.

    Shifts existing rows right in chunks, then writes new rows at index 0.
    This avoids loading the entire existing dataset into RAM.
    """
    ds_new = _strip_tz_datetime(ds_new)
    ds_new = _downcast_float32(ds_new)
    n_new = ds_new.sizes.get("time", 0)
    time_cf = _dt_ns_to_cf(ds_new["time"].values)
    var_arrays: dict[str, np.ndarray] = {"time": time_cf}
    for name, da in ds_new.data_vars.items():
        vals = da.values
        if vals.dtype.kind in ("U", "S"):
            vals = vals.astype("S")
        var_arrays[name] = vals

    with _h5py.File(str(nc_path), "a") as f:
        grp = f[tbl]
        n_old = grp["time"].shape[0]
        n_total = n_old + n_new

        # Collect extendable 1-D datasets (time-indexed)
        dsets = {name: grp[name] for name in grp
                 if isinstance(grp[name], _h5py.Dataset) and grp[name].ndim == 1
                 and grp[name].shape[0] == n_old}

        # 1. Resize all datasets to new total length
        for dset in dsets.values():
            dset.resize(n_total, axis=0)

        # 2. Shift existing data right in chunks (back→front to avoid overwrite)
        #    Iterate from end of old data backwards: copy [start:end] → [start+n_new:end+n_new]
        for end in range(n_old, 0, -chunk):
            start = max(0, end - chunk)
            for dset in dsets.values():
                dset[start + n_new: end + n_new] = dset[start:end]

        # 3. Write new data at index 0
        dsets["time"][:n_new] = var_arrays["time"]
        for name, arr in var_arrays.items():
            if name == "time" or name not in dsets:
                continue
            dsets[name][:n_new] = arr

        # Re-attach HDF5 dimension scales after resize+shift (same reason as _h5py_extend_group)
        grp["time"].make_scale("time")
        for name in dsets:
            if name != "time":
                dsets[name].dims[0].attach_scale(grp["time"])

        f.flush()
    lf.debug("Prepended {}//{}: +{:d} rows (total {:d})", nc_path, tbl, n_new, n_total)


def _read_nc_group_h5py(nc_path: Path, tbl: str) -> xr.Dataset:
    """Read all variables from ``/{tbl}/`` group into an xr.Dataset via h5py."""
    with _h5py.File(str(nc_path), "r") as f:
        return _read_nc_group_as_dataset(f, tbl)


def ensure_dim_scales(nc_path: Path) -> None:
    """Re-attach HDF5 dimension scales for every group in a netCDF4 file.

    After h5py modifications (resize, rebuild, prepend), the
    ``make_scale``/``attach_scale`` metadata that netCDF4 requires may be
    missing or broken. Calling this once before reading the file with
    ``engine="netcdf4"`` restores compatibility. Idempotent — safe to
    call on files whose scales are already correct.

    Windows file-handle quirk: netCDF4/xarray may hold a cached handle
    past the close call; a brief ``gc.collect()`` before opening h5py
    lets the OS release the lock so the "a" mode open below succeeds.
    ``OSError`` (file lock) is treated the same way as ``RuntimeError``:
    logged as a warning and the process continues without dim-scale metadata.
    """
    if not nc_path.exists() or _constants.use_h5_get() is not True:
        return
    import gc
    gc.collect()
    try:
        with _h5py.File(str(nc_path), "a") as f:
            for grp_name in list(f.keys()):
                grp = f[grp_name]
                if not isinstance(grp, _h5py.Group) or "time" not in grp:
                    continue
                time_dset = grp["time"]
                if not isinstance(time_dset, _h5py.Dataset):
                    continue
                time_dset.make_scale("time")
                for name in grp:
                    if name == "time" or not isinstance(grp[name], _h5py.Dataset):
                        continue
                    dset = grp[name]
                    if dset.ndim >= 1:
                        dset.dims[0].attach_scale(time_dset)
    except (RuntimeError, OSError) as exc:
        lf.warning(
            "ensure_dim_scales failed for {} ({}): continuing without dim-scale metadata",
            nc_path.name,
            exc,
        )
        return
    lf.debug("Ensured dimension scales in {}", nc_path.name)


def _read_nc_group_as_dataset(f: "_h5py.File", tbl: str) -> xr.Dataset:
    """Read all variables from a h5py group into an xr.Dataset."""
    grp = f[tbl]
    data_vars = {}
    coords = {}
    for name in grp:
        if not isinstance(grp[name], _h5py.Dataset):
            continue
        arr = grp[name][:]
        # Decode dimension names from netCDF dimension scales
        dim_names = (
            tuple(d.label if hasattr(d, "label") else d.name for d in grp[name].dims)
            if grp[name].dims else (name,)
        )
        if arr.ndim != len(dim_names):
            lf.warning("{}//{}: arr.shape={} but dim_names={} — using generated names", tbl, name, arr.shape, dim_names)
            dim_names = tuple(f"d{i}" for i in range(arr.ndim))
        da = xr.DataArray(arr, dims=dim_names)
        if name == "time":
            coords["time"] = _cf_to_dt_ns(arr, grp[name].attrs.get("units", ""))
        else:
            data_vars[name] = da
    return xr.Dataset(data_vars, coords=coords)


def _write_dataset_to_nc_group(
    ds: xr.Dataset,
    nc_path: Path,
    tbl: str,
) -> None:
    """Write an xr.Dataset into ``/{tbl}/`` group using h5py exclusively.

    Bypasses both xarray's ``to_netcdf`` and ``netCDF4.Dataset`` to avoid
    HDF5 file-handle conflicts when the same file is opened by multiple
    backends within a single process.  Uses h5py dimension scales so
    xr.open_dataset can read the data back with proper dimension names.

    All datasets are created with ``maxshape=(None,)`` along the time
    axis so :func:`_append_to_nc_group` can ``resize()`` them for
    O(1) tail-appends without re-reading existing data.
    """
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    ds = _downcast_float32(ds)
    with _h5py.File(str(nc_path), "a") as f:
        # Delete existing group if present
        if tbl in f:
            del f[tbl]
        grp = f.create_group(tbl)

        # Write time coordinate as a dimension scale (same CF encoding as log table)
        time_dset = grp.create_dataset(
            "time", data=_dt_ns_to_cf(ds["time"].values), dtype="f8", maxshape=(None,),
        )
        time_dset.attrs["units"] = _CF_TIME_UNITS
        time_dset.attrs["calendar"] = _CF_CALENDAR
        time_dset.make_scale("time")

        # Write data variables with dimension scale references + gzip compression
        for name, da in ds.data_vars.items():
            vals = da.values
            if vals.dtype.kind in ("U", "S"):
                vals = vals.astype("S")
            ms = (None,) * vals.ndim if "time" in da.dims else None
            # Numeric vars: gzip+shuffle compression (mirrors _ZLIB_CFG for h5py)
            kw = dict(maxshape=ms)
            if vals.dtype.kind in ("f", "i", "u"):
                kw.update(
                    compression="gzip", compression_opts=9,
                    shuffle=True, fletcher32=True,
                )
            dset = grp.create_dataset(name, data=vals, **kw)
            # Attach time dimension scale to first axis (all vars are time-indexed)
            if "time" in da.dims:
                dset.dims[0].attach_scale(grp["time"])

        # Write global attributes
        for k, v in ds.attrs.items():
            grp.attrs[k] = v


# --------------------------------------------------------------------------- #
# NC log dedup + same-file-newer detection — no-re-sort incremental
# --------------------------------------------------------------------------- #

class _LogDecision(IntEnum):
    """Result of checking a file against existing NC log records."""
    SKIP = 0       # same fileName, same/older fileChangeTime
    RESUME = 1     # same fileName, newer fileChangeTime
    NEW_FILE = 2   # different fileName (or no log records)


def check_file_vs_log(
    cur: Mapping[str, Any],
    existing: xr.Dataset,
) -> _LogDecision:
    """Compare current file metadata against existing NC log records.

    Replaces :func:`keep_recorded_nc` with richer semantics:

    - **SKIP**: same ``fileName``, existing ``fileChangeTime >= cur``.
    - **RESUME**: same ``fileName``, but ``cur.fileChangeTime`` is newer
      → only the tail portion (after existing data end) needs appending.
    - **NEW_FILE**: no matching ``fileName`` in log → full processing.

    :param cur: dict with ``fileName`` and ``fileChangeTime`` keys.
    :param existing: log Dataset from :func:`read_nc_log`.
    :return: decision enum guiding downstream handling.
    """
    if existing.sizes.get("Date0", 0) == 0:
        return _LogDecision.NEW_FILE

    fn = cur["fileName"]
    fct = np.datetime64(cur["fileChangeTime"], "ns")
    fn_match = fn == existing["fileName"].values
    if not fn_match.any():
        return _LogDecision.NEW_FILE

    # Found matching fileName — compare fileChangeTime
    existing_fcts = existing["fileChangeTime"].values[fn_match]
    if fct <= existing_fcts.max():
        return _LogDecision.SKIP  # same or older file
    return _LogDecision.RESUME  # newer version of same file


def keep_recorded_nc(
    cur: Mapping[str, Any],
    existing: xr.Dataset,
    keep_newer: bool = True,
) -> bool:
    """Check whether *cur* file is already recorded in NC log.

    Backward-compatible wrapper around :func:`check_file_vs_log`.
    Returns ``True`` when the file should be skipped.
    """
    decision = check_file_vs_log(cur, existing)
    if decision == _LogDecision.SKIP:
        return True
    if decision == _LogDecision.RESUME:
        return not keep_newer  # keep_newer=False → skip even newer files
    return False  # NEW_FILE → don't skip


# --------------------------------------------------------------------------- #
# NC incremental update — position-aware, resume-aware, no re-sort
# --------------------------------------------------------------------------- #

def nc_incremental_update(
    ds_new: xr.Dataset,
    nc_path: Union[str, Path],
    tbl: str,
    file_meta: dict,
) -> bool:
    """Check log, trim overlap if needed, then append/prepend to NC.

    Replaces the old read-merge-sort-rewrite cycle with position-aware
    logic that **never re-sorts** combined data:

    1. **Log check**: compare *file_meta* against existing ``/{tbl}/logFiles``:
       - Same ``fileName`` + same/older ``fileChangeTime`` → **skip entirely**.
       - Same ``fileName`` + newer ``fileChangeTime`` → **resume mode**:
         trim *ds_new* to only the portion after existing data's last time,
         then append. Replace the old log row and add one for the new end.
       - Different ``fileName`` → **full compare** against existing time range.

    2. **Position compare + write**: :func:`append_to_nc` classifies *ds_new* against the existing
       NC group via :class:`_Overlap` and writes directly off that classification (see its
       docstring) — head → prepend, tail → append, both → both (``WRAPS``, *existing* straddled),
       neither → skip (*ds_new* ⊆ existing). Overlapping (non-``BEFORE``/``AFTER``/``CONTAINED``)
       cases log at **warning** level; existing data is always preserved.

    :param ds_new: New data to append.
    :param nc_path: Path to ``.raw.nc`` file.
    :param tbl: Table group name (e.g. ``"incl_p5"``).
    :param file_meta: Dict with ``fileName`` and ``fileChangeTime`` keys.
    :return: ``True`` if data was appended, ``False`` if skipped.
    """
    nc_path = Path(nc_path)
    if not _nc_guard("NC incremental update"):
        return False

    # Strip tz early — .values on tz-aware coords returns object arrays that
    # crash np.datetime64(obj, "ns") and .astype("datetime64[ns]") downstream.
    ds_new = _strip_tz_datetime(ds_new)

    # Step 1: log check
    log = read_nc_log(nc_path, tbl)
    fct_ns = np.datetime64(file_meta["fileChangeTime"], "ns")
    cur = {**file_meta, "fileChangeTime": fct_ns}
    decision = check_file_vs_log(cur, log)

    if decision == _LogDecision.SKIP:
        lf.debug("Skipping {} — already recorded in log", file_meta.get("fileName"))
        return False

    if decision == _LogDecision.RESUME:
        return _resume_append(ds_new, nc_path, tbl, file_meta, log, fct_ns)

    # Step 2: NEW_FILE — full compare against existing time range
    if not append_to_nc(ds_new, nc_path, tbl):
        return False

    # Step 3: update log — append one row for this file
    times = ds_new["time"]
    new_rec = xr.Dataset(
        {
            "fileName": ("Date0", [file_meta.get("fileName", "")]),
            "fileChangeTime": ("Date0", [fct_ns]),
            "DateEnd": ("Date0", [np.datetime64(times.values[-1], "ns")]),
            "DateProc": ("Date0", [np.datetime64("now", "ns")]),
        },
        coords={"Date0": [np.datetime64(times.values[0], "ns")]},
    )
    updated = xr.concat([log, new_rec], dim="Date0") if log.sizes["Date0"] > 0 else new_rec
    write_nc_log(nc_path, tbl, updated)
    return True


def _resume_append(
    ds_new: xr.Dataset,
    nc_path: Path,
    tbl: str,
    file_meta: dict,
    log: xr.Dataset,
    fct_ns: np.datetime64,
) -> bool:
    """Resume-mode append: same file was updated, append only new tail.

    Reads existing data's last time, trims *ds_new* to only the portion
    after that time, appends (position must be AFTER since each input
    file is internally sorted). Replaces old log row + adds new end row.
    """
    if not nc_path.exists():
        return append_to_nc(ds_new, nc_path, tbl)

    # Strip tz early — .values on tz-aware coords returns object arrays that
    # crash .astype("datetime64[ns]") / .view(np.int64) below.
    ds_new = _strip_tz_datetime(ds_new)

    try:
        with _h5py.File(str(nc_path), "r") as f:
            if tbl not in f or "time" not in f[tbl]:
                return append_to_nc(ds_new, nc_path, tbl)
            time_dset = f[tbl]["time"]
            if time_dset.shape[0] == 0:
                return append_to_nc(ds_new, nc_path, tbl)
            # Use last existing time as the cut point
            ex_ns = _cf_to_dt_ns(
                time_dset[:], time_dset.attrs.get("units", ""),
            ).astype(np.int64)
    except (AttributeError, KeyError, OSError):
        return append_to_nc(ds_new, nc_path, tbl)

    ex_max_ns = ex_ns[-1]
    new_ns = ds_new["time"].values.astype("datetime64[ns]").astype(np.int64)

    # Trim: keep only rows where new_ns > ex_max_ns
    idx_start = np.searchsorted(new_ns, ex_max_ns, side="right")
    n_new_total = ds_new.sizes["time"]

    if idx_start >= n_new_total:
        lf.info("Resume {}: no new data after existing end — skipping", tbl)
        # Still update log to reflect we checked this newer version
        return False

    ds_tail = ds_new.isel(time=slice(idx_start, None))
    lf.info(
        "Resume {}: appending {:d}/{:d} time steps (existing end={})",
        tbl, ds_tail.sizes["time"], n_new_total,
        ex_max_ns.astype("datetime64[ns]"),
    )

    # Append tail (must be AFTER existing data → safe to _append_to_nc_group)
    ds_tail = _strip_tz_datetime(ds_tail)
    _append_to_nc_group(ds_tail, nc_path, tbl)

    # Update log: replace old row for this fileName + add new end-part row
    fn = file_meta.get("fileName", "")
    fn_match = log["fileName"].values == fn
    old_start = log["Date0"].values[fn_match][0] if fn_match.any() else None
    log_updated = log

    # Replace existing row (same Date0=start, updated fileChangeTime)
    if fn_match.any():
        # Remove old rows for this fileName
        keep_mask = ~fn_match
        log_updated = log.isel(Date0=keep_mask)

    # Add two rows: (1) original start with new fct, (2) tail end
    tail_start = np.datetime64(ds_tail["time"].values[0], "ns")  # end of original portion
    tail_end = np.datetime64(ds_tail["time"].values[-1], "ns")
    start_end_rec = [
        xr.Dataset(
            {
                "fileName": ("Date0", [fn]),
                "fileChangeTime": ("Date0", [fct_ns]),
                "DateEnd": ("Date0", [tail_edge]),
                "DateProc": ("Date0", [np.datetime64("now", "ns")]),
            },
            coords={"Date0": [coord_start]},
        )
        for tail_edge, coord_start in [(tail_start, old_start or tail_start), (tail_end, tail_start)]
    ]
    log_updated = xr.concat(
        [log_updated, *start_end_rec] if log_updated.sizes["Date0"] > 0 else start_end_rec, dim="Date0"
    )
    write_nc_log(nc_path, tbl, log_updated)
    return True
