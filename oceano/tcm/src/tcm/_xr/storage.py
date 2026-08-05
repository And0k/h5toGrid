"""Persistent storage helpers for the xarray-native pipeline.

Provides raw/processed netCDF persistence with incremental-update support
and NC log table I/O (replacing HDF5 log tables).
Replaces HDF5-based storage from ``_dask_legacy`` with netCDF4.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import xarray as xr

from tcm import policy, utils2init
from tcm._constants import _h5py, nc_engine
from tcm._xr import store_params
from tcm._xr.nc_utils import (
    CF_CALENDAR,
    CF_TIME_UNITS,
    EPOCH_NS,
    cf_to_dt_ns,
    compression_encoding,
    downcast_float32,
    dt_ns_to_cf,
    force_epoch,
    strip_tz_datetime,
    write_time_ds,
)

lf = utils2init.LoggingStyleAdapter(__name__)


# --------------------------------------------------------------------------- #
# NC log table I/O
# --------------------------------------------------------------------------- #


def read_nc_log(nc_path: str | Path, tbl: str, tables_log: str = "{}/logFiles") -> xr.Dataset:
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
    if not policy.io().allow_nc("NC log read"):
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
        date0 = cf_to_dt_ns(grp["Date0"][:], grp["Date0"].attrs.get("units", ""))

        data_vars: dict[str, Any] = {"fileName": ("Date0", file_names)}
        for col in ("fileChangeTime", "DateEnd", "DateProc"):
            if col in grp:
                data_vars[col] = ("Date0", cf_to_dt_ns(grp[col][:], grp[col].attrs.get("units", "")))

    lf.debug("Read {} log entries from {}//{}", n, nc_path, log_grp_path)
    return xr.Dataset(data_vars, coords={"Date0": date0})


def write_nc_log(
    nc_path: str | Path,
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
    if not policy.io().allow_nc("NC log write"):
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
            write_time_ds(grp, "Date0", np.array([], dtype=np.float64))
            f.flush()
            return

        # Date0 — CF-standard float64 seconds
        write_time_ds(grp, "Date0", dt_ns_to_cf(log["Date0"].values))
        # fileName as variable-length UTF-8 strings
        grp.create_dataset(
            "fileName",
            data=[str(v) for v in log["fileName"].values],
            dtype=str_dt,
        )

        # Datetime columns — same CF encoding
        for col in ("fileChangeTime", "DateEnd", "DateProc"):
            if col in log:
                write_time_ds(grp, col, dt_ns_to_cf(log[col].values))

        f.flush()

    lf.debug("Wrote {} log entries to {}//{}", n, nc_path, log_grp_path)


def store_raw(
    ds: xr.Dataset,
    path: str | Path,
    attrs: dict[str, Any] | None = None,
    engine: str = nc_engine,
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
    if not policy.io().allow_nc("NC raw write"):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if attrs:
        ds = ds.assign_attrs(attrs)
    ds = strip_tz_datetime(ds)
    ds = downcast_float32(ds)
    enc = {**force_epoch(ds), **compression_encoding(ds)}
    try:
        ds.to_netcdf(path, engine=engine, encoding=enc)
    except (ImportError, ValueError) as e:
        lf.warning("NC write failed (engine={}): {} — skipped {}", engine, e, path)
        return path
    lf.info(
        "Stored raw: {:s} ({:d} vars, {:d} time steps)", str(path), len(ds.data_vars), ds.sizes.get("time", 0)
    )
    return path


def delete_h5py_group(nc_path: str | Path, tbl: str) -> bool:
    """Delete group *tbl* from NC file via h5py.  Returns ``True`` on success."""
    nc_path = Path(nc_path)
    if not nc_path.exists():
        return False
    try:
        with _h5py.File(str(nc_path), "a") as f:
            if tbl in f:
                del f[tbl]
                f.flush()
                return True
        return False
    except (OSError, KeyError):
        lf.debug("Failed to delete group {} from {}", tbl, nc_path, exc_info=True)
        return False


def _drop_battery(ds: xr.Dataset) -> xr.Dataset:
    """Drop ``Battery`` variable from Dataset (non-raw outputs only).

    Battery is retained in ``*.raw.nc`` for completeness but excluded
    from all processed/binned outputs and TSV exports.
    """
    if "Battery" in ds.data_vars:
        ds = ds.drop_vars("Battery")
    return ds


def store_processed(
    ds: xr.Dataset,
    path: str | Path,
    *,
    group: str | None = None,
    mode: str = "w",
    engine: str = nc_engine,
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
    if not policy.io().allow_nc("NC processed write"):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = _drop_battery(ds)
    ds = strip_tz_datetime(ds)
    ds = downcast_float32(ds)
    enc = {**force_epoch(ds), **compression_encoding(ds)}
    # For mode='a' with a group, delegate to ``append_to_nc`` which:
    # - creates extendable ``maxshape=(None,)`` datasets on fresh write
    #   (via ``_write_dataset_to_nc_group``), so subsequent resizes never
    #   need the expensive ``_rebuild_and_append`` path;
    # - classifies time overlap via ``_Overlap`` and only writes the
    #   non-overlapping head/tail, preventing duplicate timestamps;
    # - returns False when data is fully contained (idempotent re-run).
    if mode == "a" and group and policy.io():
        written = append_to_nc(ds, path, group)
        (lf.info if written else lf.debug)(
            "Appended processed to {:s}//{} ({:d} rows, written={})",
            str(path),
            group,
            ds.sizes.get("time", 0),
            written,
        )
        return path
    try:
        ds.to_netcdf(path, group=group, mode=mode, engine=engine, encoding=enc)
    except (ImportError, ValueError, OSError) as e:
        lf.warning("NC write failed (engine={}): {} — skipped {}", engine, e, path)
        return path
    lf.info(
        "Stored processed in {:s} (group={}, mode={:s}, {:d} vars)",
        str(path),
        group or "(root)",
        mode,
        len(ds.data_vars),
    )
    return path


def store_processed_incremental(
    ds: xr.Dataset,
    path: str | Path,
    *,
    group: str,
    mode: str = "a",
    engine: str = nc_engine,
    filter_params: str | None = None,
    force_reprocess: bool = False,
) -> Path:
    """Write processed data only if group doesn't already cover this time range.

    Checks existing group's time range before writing.  Skips if the new data
    is fully contained within the existing range (avoids duplicates on re-run).

    When *filter_params* is given, appends it to the ``/param_spans/{group}``
    interval table.  On skip, the latest stored entry is compared to
    *filter_params* (ignoring ``input.time_ranges`` lines) — a **ValueError**
    is raised if they differ.  Set *force_reprocess* to ``True``
    (``out.overwrite_db="splice"``) to bypass the containment skip.

    When data extends beyond existing and *force_reprocess* is ``False``,
    a warning is emitted if stored params differ — existing data is kept
    and only the extending part is appended.

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
        (filter + window + coefs).  Stored in ``/param_spans/{group}``.
    force_reprocess
        When ``True``, bypass the containment skip and splice overlapping
        data via :func:`splice_group`.  Set via ``out.overwrite_db="splice"``.

    Returns
    -------
    Path
        Written file path (or existing path if skipped).
    """
    path = Path(path)
    if not policy.io().allow_nc("NC incremental write"):
        return path

    # Strip tz early — .values on tz-aware coords returns object arrays that
    # crash .astype("datetime64[ns]") / .view(np.int64) below.
    ds = strip_tz_datetime(ds)

    # Read parameter spans from /param_spans/{group}.
    stored_params = store_params.read_param_spans(path, group)
    stored_latest = store_params.get_latest_params(stored_params)
    if path.exists():
        try:
            with _h5py.File(str(path), "r") as f:
                if group in f and "time" in f[group]:
                    time_dset = f[group]["time"]
                    if time_dset.shape[0] > 0:
                        ex_ns = cf_to_dt_ns(
                            time_dset[:],
                            time_dset.attrs.get("units", ""),
                        ).astype(np.int64)
                        new_ns = ds["time"].values.astype("datetime64[ns]").astype(np.int64)
                        contained = new_ns.min() >= ex_ns.min() and new_ns.max() <= ex_ns.max()

                        if contained and not force_reprocess:
                            if (
                                filter_params
                                and stored_latest
                                and strip_time_ranges(stored_latest) != strip_time_ranges(filter_params)
                            ):
                                raise ValueError(
                                    f"Coefficients/params changed for {path.name}//{group} since "
                                    f"last write — re-run with out.overwrite_db=splice to overwrite "
                                    f"(or delete the group). Diff ( --- stored, +++ current):\n"
                                    f"{warn_run_params_diff(stored_latest, filter_params)}"
                                )

                            lf.debug(
                                "Skipping {} — data already covered ({} to {})",
                                group,
                                ex_ns.min().astype("datetime64[ns]"),
                                ex_ns.max().astype("datetime64[ns]"),
                            )
                            return path

                        if (
                            not contained
                            and not force_reprocess
                            and filter_params
                            and stored_latest
                            and strip_time_ranges(stored_latest) != strip_time_ranges(filter_params)
                        ):
                            lf.warning(
                                "{}//{}: existing data has different params — "
                                "keeping existing, appending new only.\n{}",
                                path.name,
                                group,
                                warn_run_params_diff(stored_latest, filter_params),
                            )

                        if not force_reprocess:
                            ex_cf = (ex_ns - EPOCH_NS).astype(np.float64) / 1e9
                            new_cf = (new_ns - EPOCH_NS).astype(np.float64) / 1e9
                            if new_cf.min() >= ex_cf[0] and new_cf.max() <= ex_cf[-1]:
                                lf.debug(
                                    "Skipping {} — data contained at CF float64 precision (Δmax={:.1f}ns)",
                                    group,
                                    (new_ns.max() - ex_ns.max()),
                                )
                                return path
        except (AttributeError, KeyError, OSError):
            pass  # group doesn't exist yet or is corrupted — fall through to write

    ds_out = ds
    if ds_out.sizes.get("time", 0):
        # Deduplicate at float64-seconds precision (CF encoding resolution).
        # Timestamps differing by < DT_CF_NS are unique at ns precision but
        # collapse to the same float64-second value on disk.  Monotonicity
        # check is O(n) vs O(n·log n) for np.unique.
        t_cf = (ds_out["time"].values.view(np.int64) - EPOCH_NS).astype(np.float64) / 1e9
        mono = np.ones(t_cf.size, dtype=bool)
        mono[1:] = np.diff(t_cf) > 0
        if not mono.all():
            n_near_dup = int((~mono).sum())
            lf.warning(
                "{}//{}: removing {} sub-μs near-duplicate time(s) (CF float64 resolution)",
                path.name,
                group,
                n_near_dup,
            )
            ds_out = ds_out.isel(time=mono)

    # Force-reprocess path: splice replaces the old group with new data,
    # preserving head/tail outside the new time range.
    if force_reprocess:
        splice_group(path, group, ds_out)
        lf.info("Spliced {:s}//{} ({:d} rows)", str(path), group, ds_out.sizes.get("time", 0))
    else:
        store_processed(ds_out, path, group=group, mode=mode, engine=engine)

    # Write param_spans after data write (NC file now exists).
    if filter_params:
        new_params = store_params.append_param(filter_params, stored_params)
        store_params.write_param_spans(path, group, new_params)
        lf.info(
            "Param spans written for {:s}//{} ({:d} interval(s))",
            str(path),
            group,
            new_params.sizes.get("interval", 0),
        )

    return path


def splice_group(nc_path: str | Path, tbl: str, ds_new: xr.Dataset) -> bool:
    """Replace the overlapping portion of *tbl* in *nc_path* with *ds_new*.

    Keeps existing data outside ``[ds_new.time.min(), ds_new.time.max()]``
    intact (head + tail), concatenates with *ds_new*, and rewrites the group.
    Uses boolean masking for robust boundary handling — no exact-match
    requirement at splice points.

    Parameters
    ----------
    nc_path
        NetCDF file path.
    tbl
        Group name (e.g. ``"i_p05"`` or ``"i_p05bin600s"``).
    ds_new
        Replacement dataset (must have ``time`` coordinate).

    Returns
    -------
    bool
        ``True`` if the group was written, ``False`` if nothing to write.
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        _write_dataset_to_nc_group(ds_new, nc_path, tbl)
        return True
    try:
        with _h5py.File(str(nc_path), "r") as f:
            if tbl not in f or "time" not in f[tbl]:
                _write_dataset_to_nc_group(ds_new, nc_path, tbl)
                return True
            existing = _read_nc_group_as_dataset(f, tbl)
    except (OSError, KeyError, AttributeError):
        _write_dataset_to_nc_group(ds_new, nc_path, tbl)
        return True

    new_min, new_max = ds_new["time"].values.min(), ds_new["time"].values.max()
    # Boolean masking — robust against non-exact boundary matches
    head = existing.where(existing.time < new_min, drop=True)
    tail = existing.where(existing.time > new_max, drop=True)

    parts = [p for p in (head, ds_new, tail) if p.sizes.get("time", 0) > 0]
    if not parts:
        return False
    result = xr.concat(parts, dim="time")
    delete_h5py_group(nc_path, tbl)
    _write_dataset_to_nc_group(result, nc_path, tbl)
    return True


def trim_group_to_range(
    nc_path: str | Path,
    tbl: str,
    time_start: np.datetime64 | None = None,
    time_end: np.datetime64 | None = None,
) -> bool:
    """Trim *tbl* in *nc_path* to ``[time_start, time_end]`` and rewrite.

    Removes data outside the specified time window.  If the group is
    entirely outside the window, deletes it.  If nothing changes, returns
    ``False`` without touching the file.

    Parameters
    ----------
    nc_path
        NetCDF file path.
    tbl
        Group name.
    time_start
        Inclusive lower bound (``None`` = keep all earlier data).
    time_end
        Inclusive upper bound (``None`` = keep all later data).

    Returns
    -------
    bool
        ``True`` if the file was modified, ``False`` otherwise.
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        return False
    try:
        with _h5py.File(str(nc_path), "r") as f:
            if tbl not in f or "time" not in f[tbl]:
                return False
            ds = _read_nc_group_as_dataset(f, tbl)
    except (OSError, KeyError, AttributeError):
        return False

    n_before = ds.sizes.get("time", 0)
    if n_before == 0:
        return False
    ds_trimmed = ds.sel(time=slice(time_start, time_end))
    if ds_trimmed.sizes["time"] == n_before:
        return False  # nothing to trim
    if ds_trimmed.sizes["time"] == 0:
        delete_h5py_group(nc_path, tbl)
        store_params.delete_param_spans(nc_path, tbl)
        return True
    delete_h5py_group(nc_path, tbl)
    _write_dataset_to_nc_group(ds_trimmed, nc_path, tbl)
    trimmed = store_params.trim_param_spans(store_params.read_param_spans(nc_path, tbl), time_start, time_end)
    store_params.write_param_spans(nc_path, tbl, trimmed)
    return True


def open_processed(
    path: str | Path,
    *,
    chunks: int | None = None,
    engine: str = nc_engine,
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
    path: str | Path,
    *,
    chunks: int | None = None,
    engine: str = nc_engine,
) -> dict[str, xr.Dataset]:
    """Open per-probe groups from a shared ``*.proc.nc`` file.

    Returns ``{group_name: Dataset}`` for each top-level group that
    contains a ``time`` dimension.  Use with :func:`merge_probes` to
    combine into a single Dataset with a ``probe`` dimension.
    """
    policy.io().require_nc("listing groups in NC4 files")

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
    path: str | Path,
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

    head_end: int  # new_t[:head_end]   — PREPEND candidate, strictly < ex_min
    tail_start: int  # new_t[tail_start:] — APPEND candidate,  strictly > ex_max
    n: int  # len(new_t)

    @classmethod
    def of(cls, new_t: np.ndarray, ex_min: np.int64, ex_max: np.int64) -> _Overlap:
        """Classify sorted *new_t* against existing range — O(log n) via searchsorted, no copy."""
        return cls(
            new_t.searchsorted(ex_min, side="left"), new_t.searchsorted(ex_max, side="right"), new_t.size
        )

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
            case False, False:
                return "CONTAINED"  # new ⊆ existing (equality included)
            case True, True:
                return "WRAPS"  # new ⊃ existing, extends past both sides
            case True, False:
                return "BEFORE" if self.head_end == self.n else "OVERLAP_HEAD"
            case False, True:
                return "AFTER" if self.tail_start == 0 else "OVERLAP_TAIL"

    def write(self, ds_new: xr.Dataset, nc_path: Path, tbl: str) -> bool:
        """Execute the write this relation *is* — the only place condition and action now meet.

        `has_head`/`has_tail` drive this directly; `.rel` plays no role here, it only labels
        the outcome for :func:`append_to_nc`'s log line.
        """
        if self.has_head:
            _prepend_nc_group(strip_tz_datetime(ds_new.isel(time=self.head)), nc_path, tbl)
        if self.has_tail:
            _append_to_nc_group(strip_tz_datetime(ds_new.isel(time=self.tail)), nc_path, tbl)
        return self.has_head or self.has_tail


def _fresh_write(ds_new: xr.Dataset, nc_path: Path, tbl: str, reason: str) -> bool:
    """Write *ds_new* as a brand-new group — shared by every "nothing to compare against" guard."""
    ds_new = strip_tz_datetime(ds_new)
    _write_dataset_to_nc_group(ds_new, nc_path, tbl)
    lf.debug("{} {}//{}: fresh write {:d} time steps", reason, nc_path, tbl, ds_new.sizes.get("time", 0))
    return True


def append_to_nc(
    ds_new: xr.Dataset,
    nc_path: str | Path,
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
    if not policy.io().allow_nc("NC append"):
        return False
    nc_path = Path(nc_path)

    # Strip tz early — .values on tz-aware coords returns object arrays that
    # crash .astype("datetime64[ns]") / .view(np.int64) below.
    ds_new = strip_tz_datetime(ds_new)

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
                ex_ns = cf_to_dt_ns(time_dset[:], time_dset.attrs.get("units", "")).astype(np.int64)
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
        nc_path,
        tbl,
        rel,
        new_ns[0].astype("datetime64[ns]"),
        new_ns[-1].astype("datetime64[ns]"),
        ex_ns[0].astype("datetime64[ns]"),
        ex_ns[-1].astype("datetime64[ns]"),
        ov.head_end,
        ov.n - ov.tail_start,
        ov.n,
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
    ds_new = downcast_float32(ds_new)

    # Prepare numpy arrays for writing
    time_cf = dt_ns_to_cf(ds_new["time"].values)
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
    nc_path: Path,
    tbl: str,
    n_new: int,
    var_arrays: dict[str, np.ndarray],
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
                    tbl,
                    name,
                    n_d,
                    n_old,
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
                dset.dims[0].label = "time"

        f.flush()
    lf.debug("Extended {}//{} by {:d} rows (total {:d})", nc_path, tbl, n_new, n_total)


def _rebuild_and_append(
    nc_path: Path,
    tbl: str,
    var_arrays: dict[str, np.ndarray],
    n_new: int,
    chunk: int = 50_000,
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
        time_dset.attrs["units"] = CF_TIME_UNITS
        time_dset.attrs["calendar"] = CF_CALENDAR
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
            new_ds.dims[0].label = "time"
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
    ds_new: xr.Dataset,
    nc_path: Path,
    tbl: str,
    chunk: int = 50_000,
) -> None:
    """Prepend *ds_new* before existing data — streaming, O(chunk) memory.

    Shifts existing rows right in chunks, then writes new rows at index 0.
    This avoids loading the entire existing dataset into RAM.
    """
    ds_new = strip_tz_datetime(ds_new)
    ds_new = downcast_float32(ds_new)
    n_new = ds_new.sizes.get("time", 0)
    time_cf = dt_ns_to_cf(ds_new["time"].values)
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
        dsets = {
            name: grp[name]
            for name in grp
            if isinstance(grp[name], _h5py.Dataset) and grp[name].ndim == 1 and grp[name].shape[0] == n_old
        }

        # 1. Resize all datasets to new total length
        for dset in dsets.values():
            dset.resize(n_total, axis=0)

        # 2. Shift existing data right in chunks (back→front to avoid overwrite)
        #    Iterate from end of old data backwards: copy [start:end] → [start+n_new:end+n_new]
        for end in range(n_old, 0, -chunk):
            start = max(0, end - chunk)
            for dset in dsets.values():
                dset[start + n_new : end + n_new] = dset[start:end]

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
                dsets[name].dims[0].label = "time"

        f.flush()
    lf.debug("Prepended {}//{}: +{:d} rows (total {:d})", nc_path, tbl, n_new, n_total)


def ensure_dim_scales(nc_path: Path) -> None:
    """Re-attach HDF5 dimension scales for every group in a netCDF4 file.

    After h5py modifications (resize, rebuild, prepend), the
    ``make_scale``/``attach_scale`` metadata that xarray expects may be
    missing or broken. Calling this once before ``xr.open_dataset``
    restores compatibility. Idempotent — skips the write-mode open when
    all scales are already attached (avoids HDF5 metadata bloat from
    redundant ``attach_scale`` calls).

    Windows file-handle quirk: netCDF4/xarray may hold a cached handle
    past the close call; a brief ``gc.collect()`` before opening h5py
    lets the OS release the lock so the "a" mode open below succeeds.
    ``OSError`` (file lock) is treated the same way as ``RuntimeError``:
    logged as a warning and the process continues without dim-scale metadata.
    """
    if not nc_path.exists() or not policy.io():
        return
    import gc

    gc.collect()

    # Read-only pre-check: skip write-mode open when all scales are already
    # correct.  h5py's ``attach_scale`` reallocates HDF5 metadata even when
    # the scale is already attached (+736 bytes per file on repeated calls).
    if _h5py is None:
        return
    try:
        needs_fix = False
        with _h5py.File(str(nc_path), "r") as f:
            for grp_name in list(f.keys()):
                grp = f[grp_name]
                if not isinstance(grp, _h5py.Group) or "time" not in grp:
                    continue
                time_dset = grp["time"]
                if not isinstance(time_dset, _h5py.Dataset):
                    continue
                if not time_dset.is_scale:
                    needs_fix = True
                    break
                n_time = time_dset.shape[0]
                for name in grp:
                    if name == "time" or not isinstance(grp[name], _h5py.Dataset):
                        continue
                    dset = grp[name]
                    if (
                        dset.ndim >= 1
                        and not dset.is_scale
                        and dset.shape[0] == n_time
                        and "time" not in list(dset.dims[0].keys())
                    ):
                        needs_fix = True
                        break
                if needs_fix:
                    break
    except (RuntimeError, OSError):
        needs_fix = True

    if not needs_fix:
        lf.debug("Dim scales intact in {} — skipped write-mode open", nc_path.name)
        return

    n_fixed = 0
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
                n_time = time_dset.shape[0]
                for name in grp:
                    if name == "time" or not isinstance(grp[name], _h5py.Dataset):
                        continue
                    dset = grp[name]
                    # Only re-attach time to datasets whose first axis length
                    # matches time — guards against attaching time to e.g. a
                    # "probe" coordinate (2 elements) whose make_scale failed.
                    if dset.ndim >= 1 and not dset.is_scale and dset.shape[0] == n_time:
                        dset.dims[0].attach_scale(time_dset)
                        dset.dims[0].label = "time"
                        n_fixed += 1
    except (RuntimeError, OSError) as exc:
        lf.warning(
            "ensure_dim_scales failed for {} ({}): continuing without dim-scale metadata",
            nc_path.name,
            exc,
        )
        return
    lf.debug("Re-attached dim scales for {} datasets in {}", n_fixed, nc_path.name)


def _read_nc_group_as_dataset(f: _h5py.File, tbl: str) -> xr.Dataset:
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
            if grp[name].dims
            else (name,)
        )
        if arr.ndim != len(dim_names):
            lf.warning(
                "{}//{}: arr.shape={} but dim_names={} — using generated names",
                tbl,
                name,
                arr.shape,
                dim_names,
            )
            dim_names = tuple(f"d{i}" for i in range(arr.ndim))
        da = xr.DataArray(arr, dims=dim_names)
        if name == "time":
            coords["time"] = cf_to_dt_ns(arr, grp[name].attrs.get("units", ""))
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
    ds = downcast_float32(ds)
    with _h5py.File(str(nc_path), "a") as f:
        # Delete existing group if present
        if tbl in f:
            del f[tbl]
        grp = f.create_group(tbl)

        # Write ALL coordinates as dimension scales — h5netcdf requires every
        # axis of every variable to have a labeled dimension (no mixing of
        # labeled/unlabeled dims).  Time gets CF encoding; others are stored raw.
        for coord_name, coord in ds.coords.items():
            if coord_name == "time":
                coord_dset = grp.create_dataset(
                    "time",
                    data=dt_ns_to_cf(coord.values),
                    dtype="f8",
                    maxshape=(None,),
                )
                coord_dset.attrs["units"] = CF_TIME_UNITS
                coord_dset.attrs["calendar"] = CF_CALENDAR
            else:
                vals = coord.values
                if vals.dtype.kind in ("U", "S"):
                    vals = vals.astype("S")
                coord_dset = grp.create_dataset(coord_name, data=vals)
            coord_dset.make_scale(coord_name)

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
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                    fletcher32=True,
                )
            dset = grp.create_dataset(name, data=vals, **kw)
            # Attach dimension scales to every axis and set the label
            # so _read_nc_group_as_dataset can recover dim names via d.label
            for i, dim in enumerate(da.dims):
                if dim in grp and isinstance(grp[dim], _h5py.Dataset) and grp[dim].is_scale:
                    dset.dims[i].attach_scale(grp[dim])
                    dset.dims[i].label = dim

        # Write global attributes
        for k, v in ds.attrs.items():
            grp.attrs[k] = v


# --------------------------------------------------------------------------- #
# NC log dedup + same-file-newer detection — no-re-sort incremental
# --------------------------------------------------------------------------- #


class _LogDecision(IntEnum):
    """Result of checking a file against existing NC log records."""

    SKIP = 0  # same fileName, same/older fileChangeTime
    RESUME = 1  # same fileName, newer fileChangeTime
    NEW_FILE = 2  # different fileName (or no log records)


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
    nc_path: str | Path,
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
    if not policy.io().allow_nc("NC incremental update"):
        return False

    # Strip tz early — .values on tz-aware coords returns object arrays that
    # crash np.datetime64(obj, "ns") and .astype("datetime64[ns]") downstream.
    ds_new = strip_tz_datetime(ds_new)

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
    ds_new = strip_tz_datetime(ds_new)

    try:
        with _h5py.File(str(nc_path), "r") as f:
            if tbl not in f or "time" not in f[tbl]:
                return append_to_nc(ds_new, nc_path, tbl)
            time_dset = f[tbl]["time"]
            if time_dset.shape[0] == 0:
                return append_to_nc(ds_new, nc_path, tbl)
            # Read raw float64 seconds (the on-disk representation) for
            # precision-faithful comparison — avoids float64→int64→float64
            # round-trip that can shift the last ~100 ns.
            ex_max_cf = float(time_dset[-1])
    except (AttributeError, KeyError, OSError):
        return append_to_nc(ds_new, nc_path, tbl)

    new_ns = ds_new["time"].values.astype("datetime64[ns]").astype(np.int64)

    # Trim: keep only rows where new > existing end at CF float64 precision.
    new_cf = (new_ns - EPOCH_NS).astype(np.float64) / 1e9
    idx_start = int(np.searchsorted(new_cf, ex_max_cf, side="right"))
    n_new_total = ds_new.sizes["time"]

    if idx_start >= n_new_total:
        lf.info("Resume {}: no new data after existing end — skipping", tbl)
        # Still update log to reflect we checked this newer version
        return False

    ds_tail = ds_new.isel(time=slice(idx_start, None))
    lf.info(
        "Resume {}: appending {:d}/{:d} time steps (existing end={:.9f}s)",
        tbl,
        ds_tail.sizes["time"],
        n_new_total,
        ex_max_cf,
    )

    # Append tail (must be AFTER existing data → safe to _append_to_nc_group)
    ds_tail = strip_tz_datetime(ds_tail)
    _append_to_nc_group(ds_tail, nc_path, tbl)

    # Update log: replace old row for this fileName with single row spanning
    # the full range (original start → new end).  One row per file keeps the
    # log idempotent and avoids precision drift from CF float64 round-trip
    # when multiple rows for the same fileName accumulate.
    fn = file_meta.get("fileName", "")
    fn_match = log["fileName"].values == fn
    old_start = log["Date0"].values[fn_match][0] if fn_match.any() else None
    if fn_match.any():
        log_updated = log.isel(Date0=~fn_match)
    else:
        log_updated = log

    tail_end = np.datetime64(ds_tail["time"].values[-1], "ns")
    updated_rec = xr.Dataset(
        {
            "fileName": ("Date0", [fn]),
            "fileChangeTime": ("Date0", [fct_ns]),
            "DateEnd": ("Date0", [tail_end]),
            "DateProc": ("Date0", [np.datetime64("now", "ns")]),
        },
        coords={"Date0": [old_start or np.datetime64(ds_tail["time"].values[0], "ns")]},
    )
    log_updated = xr.concat(
        [log_updated, updated_rec] if log_updated.sizes["Date0"] > 0 else [updated_rec], dim="Date0"
    )
    write_nc_log(nc_path, tbl, log_updated)
    return True


# ---------------------------------------------------------------------------
# Run-params text helpers — used by storage and store_params modules.
# ---------------------------------------------------------------------------
def strip_time_ranges(params: str) -> str:
    """Remove ``input.time_ranges=...`` lines from *params* text.

    Used to compare stored vs current run parameters while ignoring the
    time window — coefficient/filter changes are detected even when only
    ``input.time_ranges`` differs.
    """
    return "\n".join(line for line in params.splitlines() if not line.startswith("input.time_ranges"))


def warn_run_params_diff(stored: str, current: str) -> str:
    """Construct readable diff between stored and current run parameters."""
    import difflib

    def parse(text: str) -> dict[str, str]:
        params: dict[str, str] = {}
        key = None
        value_lines: list[str] = []
        for line in text.splitlines():
            if "=" in line and not line.startswith((" ", "\t")):
                if key is not None:
                    params[key] = value_lines
                key, first = line.split("=", 1)
                value_lines = [first]
            else:
                value_lines.append(line)
        if key is not None:
            params[key] = value_lines
        return params

    def marker(old: str, new: str) -> str:
        """One marker line: ^ replace, - delete, + insert."""
        sm = difflib.SequenceMatcher(None, old, new)
        out = [" "] * max(len(old), len(new))
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            ch = {"replace": "^", "delete": "-", "insert": "+"}[tag]
            end = max(i2, i1 + (j2 - j1)) if tag == "replace" else i2
            for i in range(i1, min(end, len(out))):
                out[i] = ch
        return "".join(out).rstrip()

    out = []
    try:
        stored, current = parse(stored), parse(current)
        for key in sorted(stored.keys() | current.keys()):
            old, new = stored.get(key), current.get(key)
            if old is None:
                out.extend(f"+ {key}={line}" if i == 0 else f"+   {line}" for i, line in enumerate(new))
                continue
            if new is None:
                out.extend(f"- {key}={line}" if i == 0 else f"-   {line}" for i, line in enumerate(old))
                continue
            if old == new:
                continue
            for i in range(max(len(old), len(new))):
                o = old[i] if i < len(old) else ""
                c = new[i] if i < len(new) else ""
                old_text, new_text = (f"{key}={o}", f"{key}={c}") if i == 0 else (f"  {o}", f"  {c}")
                if i < len(old):
                    out.append(f"- {old_text}")
                if i < len(new):
                    out.append(f"+ {new_text}")
                if i < len(old) and i < len(new) and (m := marker(old_text, new_text)).strip():
                    out.append(f"  {m}")
        return "\n".join(out)
    except Exception as e:
        return f"(Error comparing parameters: {e})"
