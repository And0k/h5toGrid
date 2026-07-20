"""Coefficient preparation and NC coefs I/O.

Provides:
- :func:`prep_cfg_for_probe` — per-probe config builder (replaces legacy version).
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

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from tcm import _constants
from tcm._constants import _h5py

import numpy as np
import pandas as pd
import xarray as xr

from tcm import format, incl_calc
from tcm._xr import filters as filters_xr
from tcm.utils2init import LoggingStyleAdapter
from tcm.calibration import orientation, calibrate
lf = LoggingStyleAdapter(__name__)

# Keys that _coefs_to_h5_dict renames or skips when building the flat dict.
# "date" (singular) is excluded from the catch-all numpy-array comprehension
# because it is always written as a fixed-length string dataset at line 66.
_RENAMED_OR_SKIP = frozenset({
    "Ag", "Cg", "Ah", "Ch", "azimuth_shift_deg", "kVabs", "date", "dates", "i",
})

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
    to avoid importing Layer 0 at the xr layer boundary.  Structure::

        //coef//G//A, //coef//G//C, //coef//H//A, //coef//H//C,
        //coef//H//azimuth_shift_deg, //coef//Vabs0, //coef//pid, //coef//date
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
            {"//coef//H//azimuth_shift_deg": coef["azimuth_shift_deg"]}
            if "azimuth_shift_deg" in coef
            else {}
        ),
        **({"//coef//Vabs0": coef["kVabs"]} if "kVabs" in coef else {}),
        **{
            f"//coef//{k}": p
            for k, p in coef.items()
            if k not in _RENAMED_OR_SKIP and isinstance(p, np.ndarray)
        },
        "//coef//pid": pcid,
        "//coef//date": str(date) if date else datetime.now().replace(microsecond=0).isoformat(),
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
    (handles str/bool→dtype, shape mismatch→delete+recreate, NaN masking,
    ``timestamp`` attributes, and ``True``→ISO-date in *dates*).

    :param nc_path: Path to ``.raw.nc`` file (created if missing).
    :param tbl: Table group name (e.g. ``"incl_01"``).
    :param coefs: Raw coefs dict (output of :func:`get_coefs`).
    :param pcid: Probe Column ID (written as ``//coef//pid``).
    :param dates: If truthy, numeric datasets get ``timestamp`` attr.
    """
    if _constants.use_h5_get() is not True:
        raise ImportError("cannot save coefs to NC/HDF5 (use_h5 wasn't set True)")
    from tcm import h5inclinometer_coef as _h5coef

    h5_dict = _coefs_to_h5_dict(coefs, pcid=pcid, date=coefs.get("date"))
    lf.debug("Saving coefs to {}: tbl={}, keys={}", nc_path, tbl, list(h5_dict))

    # Translate dates dict keys from short names ("Ag") to h5copy_coef's
    # rel_path ("//coef//G//A") so its suffix lookup matches.  Coefs without
    # an explicit date default to True (→ current ISO date via h5copy_coef).
    if isinstance(dates, dict):
        dates = {_COEF_SHORT_TO_H5.get(k, f"//coef//{k}"): v for k, v in dates.items()}
        dates |= {p: True for p in h5_dict if p not in dates}

    with _h5py.File(nc_path, "a") as h5f:
        _h5coef.h5copy_coef(None, h5f, tbl, dict_matrices=h5_dict, dates=dates)
        # Mirror legacy: date attribute on the coef group itself
        coef_grp = h5f.require_group(f"{tbl}/coef")
        if date_str := h5_dict.get("//coef//date"):
            coef_grp.attrs["date"] = str(date_str)

    lf.info("Coefs saved to {}//{}: {} datasets", nc_path, tbl, len(h5_dict))


def load_coefs_from_nc(nc_path: Path, tbl: str) -> dict[str, Any] | None:
    """Load coefs from ``/{tbl}/coef/`` group of a NetCDF4 file.

    Reverse of :func:`save_coefs_to_nc`.  Reads the HDF5 group structure
    written by the writer and returns a dict compatible with
    :func:`tcm.incl_calc.coefs.get_coefs` output format::

        {"Ag": ..., "Cg": ..., "Ah": ..., "Ch": ..., "kVabs": ...,
         "azimuth_shift_deg": ..., "dates": {...}, "date": ...}

    Name mapping mirrors :func:`tcm.incl_calc.coefs.load_coefs` HDF5 branch:
    - ``G/A`` → ``Ag``, ``G/C`` → ``Cg``, ``H/A`` → ``Ah``, ``H/C`` → ``Ch``
    - ``H/azimuth_shift_deg`` → ``azimuth_shift_deg``
    - ``Vabs0`` → ``kVabs``

    :param nc_path: Path to ``.raw.nc`` file.
    :param tbl: Table group name (e.g. ``"incl_01"``).
    :return: Coefs dict or ``None`` if file/table missing.
    """
    if _constants.use_h5_get() is not True:
        raise ImportError("cannot load coefs from NC/HDF5 (use_h5 wasn't set True)")
    nc_path = Path(nc_path)
    if not nc_path.exists():
        lf.debug("NC file not found: {}", nc_path)
        return None

    with _h5py.File(nc_path, "r") as h5f:
        coef_path = f"{tbl}/coef"
        if coef_path not in h5f:
            lf.debug("Coef group not found: {} in {}", coef_path, nc_path)
            return None

        coef_grp = h5f[coef_path]
        coefs_dict: dict[str, Any] = {"dates": {}}

        # Walk two-level structure: /{tbl}/coef/{G,H}/{A,C,...} and /{tbl}/coef/{Vabs0,pid,...}
        for name_l1, item_l1 in coef_grp.items():
            if isinstance(item_l1, _h5py.Group):
                # Two-level: G→{A,C}, H→{A,C,azimuth_shift_deg}
                for name_l2, item_l2 in item_l1.items():
                    if not isinstance(item_l2, _h5py.Dataset):
                        continue
                    # Build coef key: last-char upper → append channel letter lowered
                    coef_key = (
                        f"{name_l2}{name_l1.lower()}" if name_l2[-1:].isupper() else name_l2
                    )
                    coefs_dict[coef_key] = item_l2[()]
                    if "timestamp" in item_l2.attrs:
                        coefs_dict["dates"][coef_key] = str(item_l2.attrs["timestamp"])
            elif isinstance(item_l1, _h5py.Dataset):
                # Top-level datasets: Vabs0→kVabs, pid→skip, date→date
                if name_l1 == "pid":
                    continue
                coef_key = "kVabs" if name_l1 == "Vabs0" else name_l1
                coefs_dict[coef_key] = item_l1[()]
                if "timestamp" in item_l1.attrs:
                    coefs_dict["dates"][coef_key] = str(item_l1.attrs["timestamp"])

        # Date from coef group attribute (set by save_coefs_to_nc)
        if "date" in coef_grp.attrs:
            coefs_dict["date"] = str(coef_grp.attrs["date"])

    lf.debug("Loaded coefs from {}: keys={}", nc_path, list(coefs_dict))
    return coefs_dict


def prep_cfg_for_probe(
    pcid: str,
    cfg_in_for_probes: Mapping[str, Any],
    cfg_in_common: Mapping[str, Any],
    cfg: Mapping[str, Any],
    path_csv: Optional[Path] = None,
) -> MutableMapping[str, Any]:
    """Build probe-specific config with coefficients.

    Replaces ``legacy.incl_calc.coefs.prep_cfg_for_probe`` for the
    xr-native pipeline.  Does **not** import ``dask.dataframe``.

    Differences from legacy ``cur_cfg``:
    - No HDF5 raw-DB-as-coefs-source logic (handled separately).
    - coefs_paths chain: explicit ``coefs_path`` → class default only.

    :param pcid: Probe output Column ID (e.g. ``"i_01"``).
    :param cfg_in_for_probes: per-probe overrides keyed by pcid.
    :param cfg_in_common: input config common to all probes.
    :param cfg: top-level config dict (``cfg["input"]``, ``cfg["out"]``, ``cfg["filter"]``).
    :param path_csv: if set, overrides ``cfg1["input"]["path"]`` with the corrected CSV path.
    :return: ``cfg1`` dict with keys ``input``, ``out``, ``filter``, and loaded coefs.
    """
    from tcm.config import ConfigIn_InclProc

    cfg1: MutableMapping[str, Any] = {
        "input": {**cfg_in_common.copy(), **cfg_in_for_probes.get(pcid, {})},
        "out": dict(cfg["out"]),
        "filter": dict(cfg["filter"]),
    }

    # Build coefs_paths: explicit coefs_path → class default → yaml_export dir.
    # The yaml_export fallback lets ``dist/tcm_clc_txt`` packaging (without the
    # bundled ``calibration.h5`` file) load coefs silently from exported YAMLs.
    coefs_paths: list = []
    if cp := cfg1["input"].get("coefs_path"):
        coefs_paths.append(cp)
    if (default_cp := ConfigIn_InclProc.coefs_path) and default_cp not in coefs_paths:
        coefs_paths.append(default_cp)
    if default_cp is not None:
        yaml_dir = Path(default_cp).parent / "yaml_export"
        if yaml_dir not in coefs_paths:
            coefs_paths.append(yaml_dir)

    tbl = format.pcid_to_raw_name(pcid)
    cfg1["input"]["coefs"] = incl_calc.coefs.get_coefs(
        coefs_paths, tbl, coefs_ovr=cfg1["input"].get("coefs") or None,
    )
    lf.info("Coefs for {}: paths={}, date={}", pcid, coefs_paths, cfg1["input"]["coefs"].get("date", "N/A"))

    # Override path with corrected CSV path if provided
    if path_csv is not None:
        cfg1["input"]["path"] = path_csv

    # Expand glob "incl*" to the concrete raw table name for this probe
    if cfg1["input"].get("tables") and cfg1["input"]["tables"][0] == "incl*":
        cfg1["input"]["tables"] = [format.pcid_to_raw_name(pcid)]

    return cfg1


# ---------------------------------------------------------------------------
# Coefs preparation (xr-native) — replaces legacy coef_prepare
# ---------------------------------------------------------------------------


def coef_zeroing_rotation_from_data(
    ds_raw: xr.Dataset,
    time_ranges: list | None = None,
    Ag: np.ndarray | None = None,
    Cg: np.ndarray | None = None
) -> np.ndarray | None:
    """Compute zeroing rotation matrix from raw data within *time_ranges*.

    xr-native replacement for
    :func:`tcm._dask_legacy.incl_calc.coefs.coef_zeroing_rotation_from_data`.
    Uses ``ds.sel(time=slice(...))`` instead of dask.dataframe indexing and
    ``norm_field`` from :mod:`tcm.incl_calc.calc` for calibration.

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
        m_raw, orientation.SensorCalibration(Ch, Ah),
        a_raw, orientation.SensorCalibration(Cg, Ag),
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
        # legacy: Rz = coef_zeroing_rotation(g0xyz[:, None], np.float64(Ag), Cg)
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
    time_ranges_zeroing: list | None = None,
    time_ranges_azimuth: list | None = None,
    azimuth_add: float | None = None,
    coordinates: tuple | None = None,
    data_date: "datetime | None" = None,
) -> tuple[dict, np.ndarray | None, dict, str]:
    """Prepare coefficients: apply zeroing rotation and azimuth correction.

    xr-native replacement for
    :func:`tcm._dask_legacy.incl_calc.coefs.coef_prepare`.
    Does **not** require ``dask.dataframe``.

    :param coefs: Raw coefficients dict from :func:`get_coefs`.
    :param ds_raw: Raw inclinometer Dataset (needs Ax, Ay, Az, Mx, My, Mz columns).
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
            azimuth_add, coordinates, coefs.get("azimuth_shift_deg", 0), data_date,
        )

    msg_zeroed = ""
    if time_ranges_zeroing:
        rotation_coef = coef_zeroing_rotation_from_data(
            ds_raw, time_ranges=time_ranges_zeroing, Ag=coefs["Ag"], Cg=coefs["Cg"],
        )
        if rotation_coef is None:
            lf.debug("time_ranges_zeroing not in current data range")
        else:
            coefs_new["Rz"] = rotation_coef
            msg_zeroed += "with new tilt rotation from time_ranges_zeroing "

    if time_ranges_azimuth:
        azimuth = coef_azimuth_from_data(
            ds_raw, time_ranges=time_ranges_azimuth,
            Ah=coefs["Ah"], Ch=coefs["Ch"], Ag=coefs["Ag"], Cg=coefs["Cg"],
        )
        if azimuth is not None:
            coefs_new["azimuth_shift_deg"] = get_coef_azimuth_shift(
                azimuth_add, coordinates, azimuth, data_date,
            )
            msg_zeroed += "with new azimuth from time_ranges_azimuth "

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
