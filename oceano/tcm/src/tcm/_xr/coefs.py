"""Coefficient preparation and NC coefs I/O.

Provides:
- :func:`prep_cfg_for_probe` — per-probe config builder.
- :func:`save_coefs_to_nc` — write coefs dict into a NetCDF4 raw file's
  ``/{tbl}/coef/`` group, using h5py (NC4 files are HDF5).  Delegates the
  actual HDF5 write to :func:`h5inclinometer_coef.h5copy_coef`.
- :func:`load_coefs_from_nc` — read coefs back from NC4 file.
- :func:`prepare_coefs` — zeroing (vertical ``Rz`` + azimuth ``azimuth_shift_deg``)
  from ``time_ranges_zeroing``, plus magnetic declination correction.
- :func:`coef_zeroing_rotation_from_data` — vertical tilt zeroing from accel data.
- :func:`coef_azimuth_from_data` — magnetic North azimuth from mag+accel data.

All functions avoid importing ``dask.dataframe``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from utils import log_init

from tcm import policy
from tcm._constants import _h5py
from tcm._xr import filters as filters_xr
from tcm.calibration import calibrate, orientation

lf = log_init.LoggingStyleAdapter(__name__)

# Keys that _coefs_to_h5_dict renames or skips when building the flat dict.
# "date" and "path" are excluded from datasets — stored as HDF5 attributes
# on the ``/{tbl}/coef`` group (write-only for ``path``).
_RENAMED_OR_SKIP = frozenset(
    {
        "Ag",
        "Cg",
        "Ah",
        "Ch",
        "azimuth_shift_deg",
        "kVabs",
        "date",
        "dates",
        "i",
        "path",
    }
)

# Reverse of _coefs_to_h5_dict's rename: short coef name → h5copy_coef rel_path.
# Used to translate prepare_coefs's ``dates`` dict keys so h5copy_coef can
# match them by rel_path suffix.  Non-renamed keys (Rz, P_t, …) use ``//coef//{k}``.
_COEF_SHORT_TO_H5 = {
    **{f"{m}{ch}": f"//coef//{ch_u}//{m}" for ch, ch_u in (("h", "H"), ("g", "G")) for m in ("A", "C")},
    "azimuth_shift_deg": "//coef//H//azimuth_shift_deg",
    "kVabs": "//coef//Vabs0",
}


def _coefs_to_h5_dict(coef: Mapping[str, Any], pcid: str | None = None, date: str | None = None) -> dict:
    """Convert raw coefs dict to flat ``{h5_path: value}`` using ``//coef//`` separator.

    Mirrors :func:`tcm.incl_calc.coefs.coefs_format_for_h5` but lives here
    to avoid a circular import at the xr layer boundary.  Structure::

        //coef//G//A, //coef//G//C, //coef//H//A, //coef//H//C,
        //coef//H//azimuth_shift_deg, //coef//Vabs0, //coef//Rz, //coef//P_t, …

    String/Path values (e.g. ``path``) are intentionally excluded — they are
    persisted as HDF5 attributes of the parent group, not datasets.
    ``pid``/``date`` are also attributes in the new layout.
    """
    if coef is None:
        coef = {}
    elif "Rz" not in coef and ("Ag" in coef or "Ah" in coef):
        coef = {**coef, "Rz": np.eye(3)}

    return {
        **{
            f"//coef//{ch_u}//{m}": coef[f"{m}{ch}"]
            for ch, ch_u in (("h", "H"), ("g", "G"))
            for m in ("A", "C")
            if f"{m}{ch}" in coef
        },
        **(
            {"//coef//H//azimuth_shift_deg": coef["azimuth_shift_deg"]} if "azimuth_shift_deg" in coef else {}
        ),
        **({"//coef//Vabs0": coef["kVabs"]} if "kVabs" in coef else {}),
        **{
            f"//coef//{k}": p
            for k, p in coef.items()
            if k not in _RENAMED_OR_SKIP
            and isinstance(p, np.ndarray)
            and not isinstance(p, (str, Path))
        },
    }


def save_coefs_to_nc(
    nc_path: Path,
    tbl: str,
    coefs: Mapping[str, Any],
    pcid: str | None = None,
    dates: Any = None,
) -> None:
    """Write *coefs* dict into ``/{tbl}/coef/`` group of a NetCDF4 file.

    Builds the flat ``//coef//`` dict via :func:`_coefs_to_h5_dict`, then
    delegates the HDF5 write to :func:`h5inclinometer_coef.h5copy_coef`
    (handles shape mismatch→delete+recreate, NaN masking,
    ``timestamp`` attributes, and ``True``→ISO-date in *dates*).

    String/Path coefs (e.g. ``path``) and ``date``/``pid`` are persisted as
    HDF5 attributes of the ``/{tbl}/coef`` group, not datasets.  ``path``
    is write-only for external inspection and never read back.

    :param nc_path: Path to ``.raw.nc`` file (created if missing).
    :param tbl: Table group name (e.g. ``"incl_01"``).
    :param coefs: Raw coefs dict (output of :func:`get_coefs`).
    :param pcid: Probe Column ID (written as ``//coef//pid`` attribute).
    :param dates: If truthy, numeric datasets get ``timestamp`` attr.
    """
    policy.io().require_nc("saving coefs to NC/HDF5")
    from tcm import h5inclinometer_coef as _h5coef

    h5_dict = _coefs_to_h5_dict(coefs, pcid=pcid, date=None)
    lf.debug("Saving coefs to {}: tbl={}, keys={}", nc_path, tbl, list(h5_dict))

    # Translate dates dict keys from short names ("Ag") to h5copy_coef's
    # rel_path ("//coef//G//A") so its suffix lookup matches.  Coefs without
    # an explicit date default to True (→ current ISO date via h5copy_coef).
    # Exclude path (string attribute) from change tracking.
    if isinstance(dates, dict):
        dates = {
            _COEF_SHORT_TO_H5.get(k, f"//coef//{k}"): v for k, v in dates.items() if k != "path"
        }
        dates |= {p: True for p in h5_dict if p not in dates}

    with _h5py.File(nc_path, "a") as h5f:
        _h5coef.h5copy_coef(None, h5f, tbl, dict_matrices=h5_dict, dates=dates)
        coef_grp = h5f.require_group(f"{tbl}/coef")
        # Clean old string datasets (now attributes)
        for legacy in ("date", "pid", "path"):
            if legacy in coef_grp and isinstance(coef_grp[legacy], _h5py.Dataset):
                del coef_grp[legacy]
        # Persist string coefs as attributes of the coef group (generalized)
        for k, v in (coefs or {}).items():
            if isinstance(v, (str, Path)) and k != "dates":
                coef_grp.attrs[k] = str(v)
        if pcid:
            coef_grp.attrs["pid"] = str(pcid)
        # Date attribute: prefer coefs["date"], else now
        if date_str := coefs.get("date"):
            coef_grp.attrs["date"] = str(date_str)
        elif "date" not in coef_grp.attrs:
            coef_grp.attrs["date"] = datetime.now().replace(microsecond=0).isoformat()

    lf.info("Coefs saved to {}//{}: {} datasets", nc_path, tbl, len(h5_dict))


def _read_coefs_from_coef_group(coef_grp: _h5py.Group) -> dict[str, Any]:
    """Walk ``/{tbl}/coef/`` HDF5 group and return coefs dict (without ``date``).

    Shared traversal for :func:`load_coefs_from_nc` (NC4 files) and
    :func:`tcm.incl_calc.coefs.load_coefs` (``.h5`` files).
    Both use identical h5py group structure written by
    :func:`tcm.h5inclinometer_coef.h5copy_coef`.

    Name mapping:
    - ``G/A`` → ``Ag``, ``G/C`` → ``Cg``, ``H/A`` → ``Ah``, ``H/C`` → ``Ch``
    - ``H/azimuth_shift_deg`` → ``azimuth_shift_deg``
    - ``Vabs0`` → ``kVabs``

    :param coef_grp: h5py Group at ``/{tbl}/coef/``.
    :return: coefs dict with ``dates`` sub-dict but **no** ``date`` key
        (date resolution differs between NC and HDF5 callers).
    """
    coefs_dict: dict[str, Any] = {"dates": {}}
    for name_l1, item_l1 in coef_grp.items():
        if isinstance(item_l1, _h5py.Group):
            for name_l2, item_l2 in item_l1.items():
                if not isinstance(item_l2, _h5py.Dataset):
                    continue
                coef_key = f"{name_l2}{name_l1.lower()}" if name_l2[-1:].isupper() else name_l2
                coefs_dict[coef_key] = item_l2[()]
                if "timestamp" in item_l2.attrs:
                    coefs_dict["dates"][coef_key] = str(item_l2.attrs["timestamp"])
        elif isinstance(item_l1, _h5py.Dataset):
            if name_l1 == "pid":
                continue
            coef_key = "kVabs" if name_l1 == "Vabs0" else name_l1
            coefs_dict[coef_key] = item_l1[()]
            if "timestamp" in item_l1.attrs:
                coefs_dict["dates"][coef_key] = str(item_l1.attrs["timestamp"])
    return coefs_dict


def load_coefs_from_nc(nc_path: Path, tbl: str) -> dict[str, Any] | None:
    """Load coefs from ``/{tbl}/coef/`` group of a NetCDF4 file.

    Reverse of :func:`save_coefs_to_nc`.  Reads the HDF5 group structure
    written by the writer and returns a dict compatible with
    :func:`tcm.incl_calc.coefs.get_coefs` output format::

        {"Ag": ..., "Cg": ..., "Ah": ..., "Ch": ..., "kVabs": ...,
         "azimuth_shift_deg": ..., "dates": {...}, "date": ...}

    Delegates group traversal to :func:`_read_coefs_from_coef_group`.
    String attributes (e.g. ``path``) are intentionally **not** loaded
    (write-only for external inspection).

    :param nc_path: Path to ``.raw.nc`` file.
    :param tbl: Table group name (e.g. ``"incl_01"``).
    :return: Coefs dict or ``None`` if file/table missing.
    """
    policy.io().require_nc("loading coefs from NC/HDF5")
    nc_path = Path(nc_path)
    if not nc_path.exists():
        lf.debug("NC file not found: {}", nc_path)
        return None

    with _h5py.File(nc_path, "r") as h5f:
        coef_path = f"{tbl}/coef"
        if coef_path not in h5f:
            lf.debug("Coef group not found: {} in {}", coef_path, nc_path)
            return None

        coefs_dict = _read_coefs_from_coef_group(h5f[coef_path])

        # Date from coef group attribute (set by save_coefs_to_nc); fallback to old dataset
        if "date" in h5f[coef_path].attrs:
            coefs_dict["date"] = str(h5f[coef_path].attrs["date"])
        elif "date" in h5f[coef_path] and isinstance(h5f[coef_path]["date"], _h5py.Dataset):
            raw = h5f[coef_path]["date"][()]
            coefs_dict["date"] = raw.decode() if isinstance(raw, bytes) else str(raw)
        # path attribute is write-only, do not load

    lf.debug("Loaded coefs from {}: keys={}", nc_path, list(coefs_dict))
    return coefs_dict


# ---------------------------------------------------------------------------
# Coefs preparation
# ---------------------------------------------------------------------------


def coef_zeroing_rotation_from_data(
    ds_raw: xr.Dataset,
    time_ranges: list | None = None,
    Ag: np.ndarray | None = None,
    Cg: np.ndarray | None = None,
) -> np.ndarray | None:
    """Compute zeroing rotation matrix from raw data within *time_ranges*.

    xr-native replacement.

    :return: 3×3 rotation matrix or ``None`` when time range has no data.
    """
    if not time_ranges:
        return None

    ds_sel = filters_xr.apply_load_time_ranges(ds_raw, time_ranges)
    if ds_sel.sizes.get("time", 0) == 0:
        lf.warning(
            "Zeroing data -> no-op: time_ranges_zeroing {} – {} not in current data range",
            *pd.to_datetime(time_ranges, utc=True)[[0, -1]],
        )
        return None

    a_raw = np.stack([ds_sel[v].values for v in ("Ax", "Ay", "Az")])
    R, incl, spread = orientation.zeroing_rotation(a_raw, orientation.SensorCalibration(Cg, Ag))
    lf.info(
        "Zeroing tilt in interval {} – {} ({:d} points, mean tilt={:.3g}°, angular spread σ={:.3g}°): R={}",
        *ds_sel["time"].values[[0, -1]],
        ds_sel.sizes["time"],
        incl,
        spread,
        calibrate.coef2str(R)[0],
    )
    return R


def coef_azimuth_from_data(
    ds_raw: xr.Dataset,
    time_ranges: list | None = None,
    Ah: np.ndarray | None = None,
    Ch: np.ndarray | None = None,
    Ag: np.ndarray | None = None,
    Cg: np.ndarray | None = None,
) -> float | None:
    """Compute magnetic North azimuth shift (degrees) from raw data within *time_ranges*.

    Uses :func:`orientation.azimuth_shift` — calibrated unit vectors only,
    no velocity/magnitude (``kVabs``) dependency.

    :param Ah, Ch: magnetometer calibration (A matrix, C bias vector).
    :param Ag, Cg: accelerometer calibration (for the horizontal plane reference).
    :return: degrees or ``None`` when time range has no data.
    """
    if not time_ranges:
        return None

    ds_sel = filters_xr.apply_load_time_ranges(ds_raw, time_ranges)
    if ds_sel.sizes.get("time", 0) == 0:
        lf.warning(
            "Zeroing data -> no-op: time_ranges_zeroing {} – {} not in current data range",
            *pd.to_datetime(time_ranges, utc=True)[[0, -1]],
        )
        return None

    a_raw = np.stack([ds_sel[v].values for v in ("Ax", "Ay", "Az")])
    m_raw = np.stack([ds_sel[v].values for v in ("Mx", "My", "Mz")])
    shift = orientation.azimuth_shift(
        m_raw,
        orientation.SensorCalibration(Ch, Ah),
        a_raw,
        orientation.SensorCalibration(Cg, Ag),
    )
    lf.info(
        "Zeroing azimuth in interval {} – {} ({:d} points): azimuth shift={:.3g}°",
        *ds_sel["time"].values[[0, -1]],
        ds_sel.sizes["time"],
        shift,
    )
    return shift


def get_coef_zeroing_matrix(Rz=None, g0xyz=None, Ag=None, Cg=None, **kwargs):
    """
    Returns rotation matrix based on g0xyz (not uses input Rz) and corresponding message if g0xyz is not None,
    Else returns Rz (if Rz != np.eye(3) else None) and empty msg

    :param Rz: rotation
    :param g0xyz: (mean) accelerometer raw data vector at zero tilt
    :param Ag: _description_
    :param Cg: _description_
    :return: R, msg: rotation matrix and message
    """
    if g0xyz is not None:
        zenith = orientation.to_unit_vector(
            np.asarray(g0xyz, dtype=np.float64).reshape(3, 1),
            calibration=orientation.SensorCalibration(Cg, Ag),
        )
        Rz = orientation.rotate(zenith, np.array([0.0, 0.0, 1.0]))
        # old: Rz = coef_zeroing_rotation(g0xyz[:, None], np.float64(Ag), Cg)
        msg_rotated = "with new rotation to user defined zero point (g0xyz) "
    elif Rz is not None and (Rz != np.eye(3)).any():
        msg_rotated = ""
    else:
        Rz, msg_rotated = None, ""
    return Rz, msg_rotated


def prepare_coefs(
    coefs: dict,
    ds_raw: xr.Dataset,
    *,
    g0xyz: list | None = None,
    time_ranges_zeroing: list | None = None,
    time_ranges_azimuth: list | None = None,
    azimuth_add: float | None = None,
    coordinates: tuple | None = None,
    data_date: datetime | None = None,
) -> tuple[dict, np.ndarray | None, dict, str]:
    """Prepare coefficients: apply zeroing rotation and azimuth correction.

    xr-native replacement.

    :param coefs: Raw coefficients dict from :func:`get_coefs`.
    :param ds_raw: Raw inclinometer Dataset (needs Ax, Ay, Az, Mx, My, Mz columns).
    :param g0xyz: Raw accelerometer vector at zero tilt (``input.calib.g0xyz``) —
        overrides ``Rz`` with a computed rotation; takes precedence over a
        g0xyz found inside *coefs* (file-sourced).
    :param time_ranges_zeroing: Time ranges for **tilt** zeroing (``Rz`` rotation).
    :param time_ranges_azimuth: Time ranges for **azimuth** zeroing
        (``azimuth_shift_deg`` from mag+accel unit vectors via
        :func:`orientation.azimuth_shift`).  Independent of *time_ranges_zeroing*.
    :param azimuth_add: Additional manual azimuth offset (degrees).
    :param coordinates: ``(lat, lon)`` for magnetic declination correction.
    :param data_date: Data timestamp for declination lookup.
    :return: ``(coefs_merged, coef_zeroing_matrix, dates, msg)``.
    """
    from tcm.incl_calc.coefs import get_coef_azimuth_shift

    coefs_new: dict = {}
    if "azimuth_shift_deg" in coefs:
        coefs_new["azimuth_shift_deg"] = get_coef_azimuth_shift(
            azimuth_add,
            coordinates,
            coefs.get("azimuth_shift_deg", 0),
            data_date,
        )

    msg_zeroed = ""
    if time_ranges_zeroing:
        rotation_coef = coef_zeroing_rotation_from_data(
            ds_raw,
            time_ranges=time_ranges_zeroing,
            Ag=coefs["Ag"],
            Cg=coefs["Cg"],
        )
        if rotation_coef is None:
            lf.debug("time_ranges_zeroing not in current data range")
        else:
            coefs_new["Rz"] = rotation_coef
            msg_zeroed += "with new tilt rotation from time_ranges_zeroing "

    if time_ranges_azimuth:
        azimuth = coef_azimuth_from_data(
            ds_raw,
            time_ranges=time_ranges_azimuth,
            Ah=coefs["Ah"],
            Ch=coefs["Ch"],
            Ag=coefs["Ag"],
            Cg=coefs["Cg"],
        )
        if azimuth is not None:
            coefs_new["azimuth_shift_deg"] = get_coef_azimuth_shift(
                azimuth_add,
                coordinates,
                azimuth,
                data_date,
            )
            msg_zeroed += "with new azimuth from time_ranges_azimuth "

    if g0xyz is not None:  # config-sourced g0xyz wins over file-sourced
        coefs = {**coefs, "g0xyz": g0xyz}
    coef_zeroing_matrix, msg_rotated = get_coef_zeroing_matrix(**coefs)

    dates = coefs.get("dates", {})
    for k, v in coefs_new.items():
        try:
            cur_prev = coefs.get(k)
            if cur_prev is not None and (
                (cur_prev == v).all() if isinstance(cur_prev, np.ndarray) else cur_prev == v
            ):
                continue
        except (KeyError, TypeError):
            pass
        dates[k] = True

    return {**coefs, **coefs_new}, coef_zeroing_matrix, dates, msg_zeroed + msg_rotated
