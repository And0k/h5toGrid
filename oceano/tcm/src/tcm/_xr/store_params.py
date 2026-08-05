"""Parameter spans I/O — interval-based storage replacing ``_run_params`` JSON attr.

Stores per-table processing parameters as an HDF5 sibling group
``/param_spans/{tbl}`` with interval semantics:

- coord ``start`` (``datetime64[ns]``) — interval boundaries
- var ``params`` (str) — sorted key=value text per interval
- var ``meta`` (str JSON) — metadata per interval

Interval *i* covers ``[start[i], start[i+1])`` or ``[start[i], ∞)`` if last.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from tcm import policy
from tcm._xr.nc_utils import cf_to_dt_ns, dt_ns_to_cf, _h5py, write_time_ds

_PARAM_SPANS_GROUP = "param_spans"
_SENTINEL_END = np.datetime64("9999-12-31", "ns")

EMPTY_PARAM_SPANS = xr.Dataset(
    coords={"start": ("interval", np.array([], dtype="datetime64[ns]"))},
    data_vars={
        "params": ("interval", np.array([], dtype=object)),
        "meta": ("interval", np.array([], dtype=object)),
    },
)


def param_spans_grp(tbl: str) -> str:
    """H5py group path: ``param_spans/{tbl}``."""
    return f"{_PARAM_SPANS_GROUP}/{tbl}"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_param_spans(nc_path: str | Path, tbl: str) -> xr.Dataset:
    """Read parameter spans from ``/param_spans/{tbl}`` subgroup.

    Returns :class:`xr.Dataset` with dimension ``interval`` and variables:
    ``start`` (``datetime64[ns]``), ``params`` (str), ``meta`` (str JSON).
    Returns :data:`EMPTY_PARAM_SPANS` when the group is missing or h5py unavailable.
    """
    nc_path = Path(nc_path)
    if not policy.io().allow_nc("NC param_spans read") or not nc_path.exists():
        return EMPTY_PARAM_SPANS
    grp_path = param_spans_grp(tbl)
    with _h5py.File(nc_path, "r") as f:
        if grp_path not in f:
            return EMPTY_PARAM_SPANS
        grp = f[grp_path]
        if "start" not in grp or grp["start"].shape[0] == 0:
            return EMPTY_PARAM_SPANS
        n = grp["start"].shape[0]
        start = cf_to_dt_ns(grp["start"][:], grp["start"].attrs.get("units", ""))
        _decode = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        params = (
            np.array([_decode(v) for v in grp["params"][:]], dtype=object)
            if "params" in grp
            else np.full(n, "", dtype=object)
        )
        meta = (
            np.array([_decode(v) for v in grp["meta"][:]], dtype=object)
            if "meta" in grp
            else np.full(n, "{}", dtype=object)
        )
    return xr.Dataset(
        coords={"start": ("interval", start)},
        data_vars={"params": ("interval", params), "meta": ("interval", meta)},
    )


def write_param_spans(nc_path: str | Path, tbl: str, params: xr.Dataset) -> None:
    """Write parameter spans Dataset to ``/param_spans/{tbl}`` (full overwrite).

    ``start`` stored as CF-standard float64 seconds.  ``params``/``meta``
    stored as variable-length UTF-8 strings.
    """
    if not policy.io().allow_nc("NC param_spans write"):
        return
    nc_path = Path(nc_path)
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    grp_path = param_spans_grp(tbl)
    str_dt = _h5py.string_dtype(encoding="utf-8")
    with _h5py.File(nc_path, "a") as f:
        if grp_path in f:
            del f[grp_path]
        grp = f.create_group(grp_path)
        n = params.sizes.get("interval", 0)
        if n == 0:
            write_time_ds(grp, "start", np.array([], dtype=np.float64))
            grp.create_dataset("params", data=np.array([], dtype=object), dtype=str_dt)
            grp.create_dataset("meta", data=np.array([], dtype=object), dtype=str_dt)
        else:
            write_time_ds(grp, "start", dt_ns_to_cf(params["start"].values))
            grp.create_dataset("params", data=list(params["params"].values), dtype=str_dt)
            grp.create_dataset("meta", data=list(params["meta"].values), dtype=str_dt)
        f.flush()


# ---------------------------------------------------------------------------
# Query / mutation
# ---------------------------------------------------------------------------


def get_latest_params(params: xr.Dataset) -> str:
    """Extract the latest (last) ``params`` entry.  ``""`` when empty."""
    n = params.sizes.get("interval", 0)
    return str(params["params"].values[-1]) if n else ""


def append_param(text: str, params: xr.Dataset) -> xr.Dataset:
    """Append *text* as a new interval.  Dedup: skip if latest already matches."""
    if params.sizes.get("interval", 0) > 0 and get_latest_params(params) == text:
        return params
    new_entry = xr.Dataset(
        coords={"start": ("interval", np.array([np.datetime64("now", "ns")], dtype="datetime64[ns]"))},
        data_vars={
            "params": ("interval", np.array([text], dtype=object)),
            "meta": ("interval", np.array(["{}"], dtype=object)),
        },
    )
    return new_entry if params.sizes.get("interval", 0) == 0 else xr.concat([params, new_entry], dim="interval")


# ---------------------------------------------------------------------------
# Splice / trim / delete
# ---------------------------------------------------------------------------


def _interval_ends(starts: np.ndarray) -> np.ndarray:
    """Each interval's effective end: next start or sentinel."""
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = _SENTINEL_END
    return ends


def splice_param_spans(
    params: xr.Dataset,
    new_min: np.datetime64,
    new_max: np.datetime64,
    text: str,
) -> xr.Dataset:
    """Replace intervals overlapping ``[new_min, new_max]`` with a single new entry.

    Keeps intervals entirely outside the window.  Adjacent intervals with
    identical params are merged.
    """
    if params.sizes.get("interval", 0) == 0:
        return append_param(text, EMPTY_PARAM_SPANS)

    starts, params_arr = params["start"].values, params["params"].values
    ends = _interval_ends(starts)
    keep = (ends <= new_min) | (starts >= new_max)

    all_starts = np.concatenate([starts[keep], [new_min]])
    all_params = np.concatenate([params_arr[keep], np.array([text], dtype=object)])
    order = np.argsort(all_starts)
    all_starts, all_params = all_starts[order], all_params[order]

    mask = np.empty(len(all_starts), dtype=bool)
    mask[0] = True
    mask[1:] = all_params[1:] != all_params[:-1]

    return xr.Dataset(
        coords={"start": ("interval", all_starts[mask])},
        data_vars={
            "params": ("interval", all_params[mask]),
            "meta": ("interval", np.full(mask.sum(), "{}", dtype=object)),
        },
    )


def trim_param_spans(
    params: xr.Dataset,
    time_start: np.datetime64 | None,
    time_end: np.datetime64 | None,
) -> xr.Dataset:
    """Keep only intervals that overlap ``[time_start, time_end]``."""
    n = params.sizes.get("interval", 0)
    if n == 0:
        return params

    starts = params["start"].values
    ends = _interval_ends(starts)
    keep = np.ones(n, dtype=bool)
    if time_start is not None:
        keep &= ends > time_start
    if time_end is not None:
        keep &= starts <= time_end

    return EMPTY_PARAM_SPANS if not keep.any() else xr.Dataset(
        coords={"start": ("interval", starts[keep])},
        data_vars={
            "params": ("interval", params["params"].values[keep]),
            "meta": ("interval", params["meta"].values[keep]),
        },
    )


def delete_param_spans(nc_path: str | Path, tbl: str) -> None:
    """Delete ``/param_spans/{tbl}`` subgroup if it exists."""
    nc_path = Path(nc_path)
    if not nc_path.exists():
        return
    grp_path = param_spans_grp(tbl)
    with _h5py.File(nc_path, "a") as f:
        if grp_path in f:
            del f[grp_path]
