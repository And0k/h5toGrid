"""Coefficient loading, preparation and zeroing — HDF5 or YAML source."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date as datetime_date
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
)

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from utils import log_init

from tcm import _constants, format, policy, schema, to_omegaconf
from tcm._xr import coefs as _xr_coefs

lf = log_init.LoggingStyleAdapter(__name__)


def coef_rotate(*A, Z):
    """Note: Keyword-Only Z invocation allowed e.g. `coef_rotate(Ag, Ah, Z=Z)`"""
    return [Z @ a for a in A]


def get_coef_azimuth_shift(
    azimuth_add: float | None,
    coordinates: tuple[float, float] | None,
    azimuth_shift_deg: float | np.ndarray = 0,
    data_date: datetime = datetime.now(),
    **kwargs,
) -> np.ndarray:
    if azimuth_add or coordinates:
        msgs = [
            f"(coef. {azimuth_shift_deg.item() if isinstance(azimuth_shift_deg, np.ndarray) else azimuth_shift_deg:g})"
        ]
        if azimuth_add:
            msgs.append(f"(azimuth_shift_deg {azimuth_add:g})")
            azimuth_shift_deg += azimuth_add
        if coordinates:
            mag_decl = mag_dec(*coordinates, data_date, depth=-1)
            msgs.append(f"(magnetic declination {mag_decl:g})")
            azimuth_shift_deg += mag_decl
        lf.warning(
            "Azimuth correction updated to {:g} = {}°",
            azimuth_shift_deg.item() if isinstance(azimuth_shift_deg, np.ndarray) else azimuth_shift_deg,
            " + ".join(msgs),
        )
    return azimuth_shift_deg


def _year_fraction(date: datetime) -> float:
    start = datetime_date(date.year, 1, 1).toordinal()
    year_length = datetime_date(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


def mag_dec(lat, lon, time: datetime, depth: float = 0):
    """Magnetic declination at (lat, lon) for *time* using WMM-2025.

    :param lat: geodetic latitude (degrees, +N)
    :param lon: geodetic longitude (degrees, +E)
    :param time: observation datetime (UTC)
    :param depth: depth below surface in **metres** (negative → above sea level)
    :return: declination in degrees (positive = east of true north)
    """
    from pygeomag import GeoMag

    yeardec = _year_fraction(time)
    return GeoMag().calculate(glat=lat, glon=lon, alt=depth / 1000.0, time=yeardec).d


def _load_coefs_from_yaml(yaml_path: Path) -> dict[str, Any] | None:
    """Load coefficients from a YAML file exported by export_coefs_to_yaml.py.

    Expected structure: ``input.coefs: {Ag: [...], Cg: [...], ...}``
    Returns dict in the same format as HDF5 load_coefs (with 'dates' key).
    """
    cfg = OmegaConf.load(yaml_path)
    coefs_node = cfg.get("input", {}).get("coefs", None)
    if coefs_node is None:
        return None
    coefs_dict = OmegaConf.to_container(coefs_node, resolve=True)
    coefs_dict.setdefault("dates", {})

    lf.debug("Loaded coefficients from {}", yaml_path)
    return coefs_dict


def _resolve_coef_date(coefs_dict: dict[str, Any], coef_grp) -> None:
    """Resolve ``coefs_dict["date"]`` from HDF5 coef group.

    Legacy ``.h5`` files store the date in three different on-disk formats:
    - **Group attribute** (preferred, set by :func:`save_coefs_to_nc`)
    - **Scalar bytes/str dataset** (``np.bytes_`` — ``b'2023-08-12T16:21:30'``)
    - **Float64 array** (legacy μs-since-epoch, e.g. ``incl_p01``)

    NC files only use the group attribute (handled by :func:`load_coefs_from_nc`).
    """
    if "date" in coef_grp.attrs:
        coefs_dict["date"] = str(coef_grp.attrs["date"])
    elif "date" in coefs_dict:
        date_val = coefs_dict["date"]
        if hasattr(date_val, "dtype") and date_val.dtype.kind in ("S", "U"):
            raw = date_val.item() if hasattr(date_val, "item") else date_val
            coefs_dict["date"] = raw.decode() if isinstance(raw, bytes) else str(raw)
        elif isinstance(date_val, np.ndarray) and date_val.dtype.kind == "f":
            with np.errstate(invalid="ignore"):
                finite_max = np.nanmax(date_val)
            if np.isfinite(finite_max):
                coefs_dict["date"] = np.datetime64(int(finite_max), "us").item().isoformat()
            else:
                del coefs_dict["date"]
        elif not isinstance(date_val, (np.ndarray, np.generic)):
            coefs_dict["date"] = str(date_val)
    # Node timestamps may override string date with max
    if "date" in coefs_dict and isinstance(coefs_dict["date"], str):
        if coefs_dict["dates"]:
            date_vals = [np.datetime64(d) for d in coefs_dict["dates"].values()]
            date_vals.append(np.datetime64(coefs_dict["date"]))
            coefs_dict["date"] = str(max(date_vals))


def load_coefs(store, tbl: str):
    """Load coefs from HDF5 store, NC4 file, or YAML file.

    Dispatch order:
    1. Directory → resolve ``{store}/{tbl}.yaml``.
    2. ``.yaml``/``.yml`` suffix → load directly.
    3. ``.nc`` suffix → delegate to :func:`_xr.coefs.load_coefs_from_nc`.
    4. Otherwise → open as HDF5 via ``h5py`` (legacy ``pd.HDFStore`` also
       supported when *store* is an already-open PyTables store).
    """
    store_path = Path(store) if not isinstance(store, Path) else store

    # YAML path: directory → resolve {tbl}.yaml, or direct .yaml file
    if yaml_path := (
        store_path
        if store_path.suffix in (".yaml", ".yml")
        else store_path / f"{tbl}.yaml"
        if store_path.is_dir()
        else None
    ):
        if not yaml_path.exists():
            return None
        return _load_coefs_from_yaml(yaml_path)

    # NC4 path — delegate to xr-native reader
    if store_path.suffix == ".nc":
        return _xr_coefs.load_coefs_from_nc(store_path, tbl)

    # HDF5 path — skip if binary I/O disabled (noh5 mode)
    if not isinstance(store, pd.HDFStore) and store_path.suffix in _constants.EXT_HDF5:
        if not policy.io():
            return None  # silent skip — caller falls through to YAML coefs
        if not store_path.exists():
            lf.debug("Coefficients file {} not found", store_path)
            return None

    # HDF5 load via h5py — coefs stored as plain HDF5 groups/datasets
    # (written by h5inclinometer_coef.h5copy_coef; pytables never needed).
    h5py = _constants._h5py
    if isinstance(store, pd.HDFStore):
        # Legacy: caller passed an open pd.HDFStore — use its underlying filename
        h5_path = store.filename
    else:
        h5_path = store

    with h5py.File(h5_path, mode="r") as f:
        coef_grp = f.get(f"{tbl}/coef")
        if coef_grp is None:
            return None

        # Shared traversal — same h5py structure as NC files
        coefs_dict = _xr_coefs._read_coefs_from_coef_group(coef_grp)

        # Date resolution — HDF5 files have richer on-disk variants than NC:
        # group attr, scalar bytes dataset, numeric array (legacy μs-since-epoch)
        _resolve_coef_date(coefs_dict, coef_grp)

    return coefs_dict


def get_coefs(coefs_paths: Sequence, tbl: str, coefs_ovr: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load and merge coefficients from file(s) + config overrides.

    Converts YAML list values to numpy arrays.  When no coefs are found in
    any *coefs_paths* and no overrides are provided, logs a warning and
    returns an empty dict (processing continues with dataclass defaults).

    :param coefs_paths: ordered list of search paths (H5, YAML dir, NC).
    :param tbl: coefficient table name (from :func:`format.pcid_to_raw_name`).
    :param coefs_ovr: config overrides (``input.coefs``), defaults to None.
    :return: merged coefficients dict with array values as numpy ndarrays.
    """

    # Normalize dataclass → dict (coefs_ovr may come from config.ConfigInCoefs_InclProc)
    if is_dataclass(coefs_ovr) and not isinstance(coefs_ovr, type):
        coefs_ovr = asdict(coefs_ovr)
    if OmegaConf.is_config(coefs_ovr):
        coefs_ovr = OmegaConf.to_container(coefs_ovr, resolve=True)

    defaults = {
        k: v_def
        for k, v in schema.ConfigInCoefs_InclProc.__dataclass_fields__.items()
        if k != "path"
        and (v_def := to_omegaconf.get_field_default(v)) is not None
        and not (isinstance(v_def, (list, dict)) and ((not v_def) or not any(lst != [] for lst in v_def)))
    }

    coefs_ovr_dates: dict[str, Any] = {}
    not_ovr: list = list(defaults)  # assume all are defaults until override proves otherwise
    if coefs_ovr:
        not_ovr = [
            k
            for k, v_def in defaults.items()
            if (
                (v_ovr := coefs_ovr.get(k)) == v_def
                or (isinstance(v_ovr, list) and ((not v_ovr) or not any(lst != [] for lst in v_ovr)))
            )
        ]
        # ``P_t`` (2-D pressure-T polynomial) supersedes the legacy scalar
        # triples ``P``, ``PBattery``, ``PTemp`` — check **early** so they
        # don't keep coefs_paths alive when the user already has ``P_t``.
        if coefs_ovr.get("P_t") is not None:
            not_ovr = [k for k in not_ovr if k not in ("P", "PBattery", "PTemp")]
        if len(not_ovr) < len(defaults):
            if not not_ovr:
                coefs_paths = []
            if coefs_ovr_dates := coefs_ovr.get("dates", {}):
                coefs_ovr_dates = {
                    k: d for k, d in coefs_ovr_dates.items() if k in defaults and k not in not_ovr
                }
        else:
            not_ovr = list(defaults)
            coefs_ovr = {}

    _META_KEYS = frozenset(("dates", "date", "pid"))
    # Source attribution per coef — populated below, returned via module state
    # so callers (get_coefs_from_cfg) can log a precise breakdown.
    from_file: set[str] = set()
    from_ovr: set[str] = set()
    if coefs_paths:
        coefs_load_src: Path | None = None
        for coefs_path in coefs_paths:
            coefs_load = load_coefs(coefs_path, tbl)
            if coefs_load is not None:
                coefs_load_src = coefs_path
                break
        # Also check P_t from loaded coefs (may come from yaml_export, not from coefs_ovr).
        if coefs_load and "P_t" in coefs_load:
            not_ovr = [k for k in not_ovr if k not in ("P", "PBattery", "PTemp")]
        if coefs_load is None:
            lf.warning(
                'Not found coefs "{:s}" in {}, {:s}redefined from run config — using defaults',
                tbl,
                coefs_paths,
                "" if not not_ovr else f"{not_ovr} not " if len(not_ovr) < len(defaults) else "none ",
            )
            coefs_load = {**(coefs_ovr or {}), "dates": coefs_ovr_dates}
        else:
            # Coefs present in the file but absent from overrides → file wins.
            from_file = {
                k
                for k, v in coefs_load.items()
                if k not in _META_KEYS
                and not isinstance(v, (str, bytes))
                and (not coefs_ovr or k not in coefs_ovr)
            }
            # Coefs supplied by overrides merge on top of file values below.
            from_ovr = (
                {
                    k
                    for k in (coefs_ovr or {})
                    if k in defaults and k not in not_ovr and coefs_ovr[k] is not None
                }
                if coefs_ovr
                else set()
            )
            from_file -= from_ovr  # override wins → attribute to override, not file
            lf.debug(
                "Coef sources for {}: from file={} ({}){}, override ({}){}, default ({} items)",
                tbl,
                coefs_load_src,
                len(from_file),
                f": {sorted(from_file)}" if len(from_file) else "",
                len(from_ovr),
                f": {sorted(from_ovr)}" if len(from_ovr) else "",
                len(defaults) - len(from_file) - len(from_ovr),
            )
            coefs_load_dates = coefs_load.get("dates", coefs_ovr_dates)
            if coefs_ovr:
                for k, v in coefs_ovr.items():
                    if v is not None and k in defaults and k not in not_ovr:
                        coefs_load[k] = v
                        try:
                            coefs_load_dates[k] = coefs_ovr_dates[k]
                        except KeyError:
                            continue
            coefs_load["dates"] = coefs_load_dates
    else:
        coefs_load = {**(coefs_ovr or {}), "dates": coefs_ovr_dates}
        from_ovr = (
            {k for k in (coefs_ovr or {}) if k in defaults and k not in not_ovr and coefs_ovr[k] is not None}
            if coefs_ovr
            else set()
        )

    if coefs_load:
        coefs_load = {
            k: np.asarray(v, dtype=np.float64)
            if (isinstance(v, list) and v is not None and k != "dates")
            else v
            for k, v in coefs_load.items()
        }

    out_dates = [
        datetime.fromisoformat(d) if isinstance(d, str) else d
        for d in [coefs_load.get("date")] + list(coefs_load["dates"].values())
        if d
    ]
    if out_dates:
        coefs_load["date"] = max(out_dates)

    # Stash attribution on the function for the caller (get_coefs_from_cfg).
    # Avoids polluting the coefs dict (direct callers like tests/tcm_gui expect a clean dict).
    get_coefs._src = {
        "file": coefs_load_src if coefs_paths else None,
        "from_file": from_file,
        "from_ovr": from_ovr,
        "n_default": max(0, len(defaults) - len(from_file) - len(from_ovr)),
        "paths": list(coefs_paths),
    }
    return coefs_load


def coefs_format_for_h5(coef: Mapping[str, Any], pcid: str = None, date: str | None = None):
    if coef is None:
        coef = schema.ConfigInCoefs_InclProc().__dict__
        if not pcid.split("_")[-1].startswith("p"):
            del coef["P_t"]
        if not pcid.startswith("w"):
            del coef["P"]
            del coef["PBattery"]
            del coef["PTemp"]
    elif "Rz" not in coef and ("Ag" in coef or "Ah" in coef):
        coef["Rz"] = np.eye(3)

    coef_renamed_or_skip = {"Ag", "Cg", "Ah", "Ch", "azimuth_shift_deg", "kVabs", "dates", "i", "path"}
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
            if k not in coef_renamed_or_skip and isinstance(p, np.ndarray)
        },
        "//coef//pid": pcid,
        "//coef//date": date or datetime.now().replace(microsecond=0).isoformat(),
    }


def get_coefs_from_cfg(cfg_in: dict, pcid: str) -> dict:
    """Resolve coefficients: ``input.coefs.path`` file → ``input.coefs`` override.
    1. Build a ``coefs_paths`` fallback chain:
    explicit ``input.coefs.path`` from YAML → class-default HDF5 path → sibling "yaml_export/" dir.
    2. merge logic + converts YAML list values to numpy arrays

    The "yaml_export/" dir should have same coefficients as in default HDF5 path to be used silently in the
    environments without hdf5 support

    :param cfg_in: ``cfg.input`` as a plain dict.
    :param pcid: probe column ID (e.g. ``"i_01"``).
    :return: merged coefficients dict with array values as numpy ndarrays.
    """
    # Hard break: old key input.coefs_path is removed
    if "coefs_path" in cfg_in:
        lf.error(
            "Unknown key 'input.coefs_path' — renamed to 'input.coefs.path'. "
            "Move the value into input.coefs.path (Path | None)."
        )
        raise KeyError("input.coefs_path is removed, use input.coefs.path")
    coefs_cfg = cfg_in.get("coefs") or {}
    # OmegaConf / dataclass → plain dict safe
    if OmegaConf.is_config(coefs_cfg):
        coefs_cfg = OmegaConf.to_container(coefs_cfg, resolve=True)
    coefs_paths: list = []
    if cp := coefs_cfg.get("path"):
        coefs_paths.append(Path(cp) if not isinstance(cp, Path) else cp)
    cp_default = schema.ConfigInCoefs_InclProc.__dataclass_fields__["path"].default
    if cp_default and cp_default not in coefs_paths:
        # Skip H5 path when binary I/O is unavailable/disabled
        if Path(cp_default).suffix not in _constants.EXT_HDF5 or policy.io():
            coefs_paths.append(Path(cp_default))
    # Always add yaml_export dir as fallback (may be the only working source
    # when io().h5 is False or the H5 file is missing in dist builds).
    if cp_default:
        yaml_dir = Path(cp_default).parent / "yaml_export"
        if yaml_dir not in coefs_paths:
            coefs_paths.append(yaml_dir)
    coefs_ovr = {k: v for k, v in coefs_cfg.items() if k != "path"} or None
    cfg_in_coefs = get_coefs(
        coefs_paths,
        tbl=format.pcid_to_raw_name(pcid),
        coefs_ovr=coefs_ovr,
    )
    # Log source attribution: which coefs came from file, override, or default.
    src = getattr(get_coefs, "_src", None)
    get_coefs._src = None  # consume once
    if src:
        from_f, from_o = sorted(src["from_file"]), sorted(src["from_ovr"])
        file_tag = Path(src["file"]).name if src["file"] else "(no file)"
        parts: list[str] = []
        if from_f:
            parts.append(f"{len(from_f)} from {file_tag}: {from_f}")
        if from_o:
            parts.append(f"{len(from_o)} from config override: {from_o}")
        if n_def := src["n_default"]:
            parts.append(f"{n_def} default")
        lf.info(
            "Coefs for {}: {} | date={}",
            pcid,
            "; ".join(parts) if parts else "no coefs (all defaults)",
            cfg_in_coefs.get("date", "N/A"),
        )
    else:
        lf.info(
            "Coefs for {}: paths={}, date={}, {} override keys",
            pcid,
            coefs_paths,
            (coefs_ovr or {}).get("date", "N/A"),
            len(coefs_ovr) if coefs_ovr else 0,
        )
    return cfg_in_coefs
