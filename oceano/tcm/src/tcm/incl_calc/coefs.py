"""Coefficient loading, preparation and zeroing — HDF5 or YAML source."""

from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from datetime import date as datetime_date
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from tcm import _constants, config, to_omegaconf, utils2init
from tcm._xr import coefs as _xr_coefs

lf = utils2init.LoggingStyleAdapter(__name__)


def coef_rotate(*A, Z):
    """Note: Keyword-Only Z invocation allowed e.g. `coef_rotate(Ag, Ah, Z=Z)`"""
    return [Z @ a for a in A]


def get_coef_azimuth_shift(
    azimuth_add: Optional[float],
    coordinates: Optional[Tuple[float, float]],
    azimuth_shift_deg: float | np.ndarray = 0,
    data_date: datetime = datetime.now(),
    **kwargs,
) -> np.ndarray:
    if azimuth_add or coordinates:
        msgs = ["(coef. {:g})".format(
            azimuth_shift_deg.item() if isinstance(azimuth_shift_deg, np.ndarray) else azimuth_shift_deg
        )]
        if azimuth_add:
            msgs.append("(azimuth_shift_deg {:g})".format(azimuth_add))
            azimuth_shift_deg += azimuth_add
        if coordinates:
            mag_decl = mag_dec(*coordinates, data_date, depth=-1)
            msgs.append("(magnetic declination {:g})".format(mag_decl))
            azimuth_shift_deg += mag_decl
        lf.warning(
            "Azimuth correction updated to {:g} = {}°",
            azimuth_shift_deg.item() if isinstance(azimuth_shift_deg, np.ndarray) else azimuth_shift_deg,
            " + ".join(msgs)
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


def _load_coefs_from_yaml(yaml_path: Path) -> Optional[Dict[str, Any]]:
    """Load coefficients from a YAML file exported by export_coefs_to_yaml.py.

    Expected structure: ``input.coefs: {Ag: [...], Cg: [...], ...}``
    Returns dict in the same format as HDF5 load_coefs (with 'dates' key).
    """
    cfg = OmegaConf.load(yaml_path)
    coefs_node = cfg.get("input", {}).get("coefs", None)
    if coefs_node is None:
        return None
    coefs_dict = OmegaConf.to_container(coefs_node, resolve=True)
    coefs_dict.setdefault('dates', {})

    lf.debug("Loaded coefficients from {}", yaml_path)
    return coefs_dict


def load_coefs(store, tbl: str):
    """Load coefs from HDF5 store, NC4 file, or YAML file.

    Dispatch order:
    1. Directory → resolve ``{store}/{tbl}.yaml``.
    2. ``.yaml``/``.yml`` suffix → load directly.
    3. ``.nc`` suffix → delegate to :func:`_xr.coefs.load_coefs_from_nc`.
    4. Otherwise → open as HDF5 via ``pd.HDFStore`` (pytables).
    """
    store_path = Path(store) if not isinstance(store, Path) else store

    # YAML path: directory → resolve {tbl}.yaml, or direct .yaml file
    if (yaml_path := (
        store_path
        if store_path.suffix in (".yaml", ".yml")
        else store_path / f"{tbl}.yaml"
        if store_path.is_dir()
        else None
    )):
        if not yaml_path.exists():
            return None
        return _load_coefs_from_yaml(yaml_path)

    # NC4 path — delegate to xr-native reader
    if store_path.suffix == ".nc":
        return _xr_coefs.load_coefs_from_nc(store_path, tbl)

    # HDF5 path — skip entirely if binary I/O not enabled (expected in noh5 mode)
    if not isinstance(store, pd.HDFStore) and store_path.suffix in _constants.hdf5_suffixes:
        if _constants.use_h5_get() is not True:
            return None
        if not store_path.exists():
            lf.debug("Coefficients file {} not found", store_path)
            return None
    # Guard: pytables required for HDF5 store access
    if not _constants.TABLES_AVAILABLE:
        lf.warning("pytables not available — skipping HDF5 coefs load from {}", store_path)
        return None

    # HDF5 load coefs collecting 'dates' from nodes date attributes and putting max of them in the "date"
    with nullcontext(store) if isinstance(store, pd.HDFStore) else pd.HDFStore(store, mode="r") as s:
        node_coef = s.get_node(f'{tbl}/coef')
        if node_coef is None:
            return
        coefs_dict = {'dates': {}}
        for node_name in node_coef.__members__:
            node_coef_l2 = node_coef[node_name]
            if getattr(node_coef_l2, '__members__', False):
                for node_name_l2 in node_coef_l2.__members__:
                    name = f'{node_name_l2}{node_name.lower() if node_name_l2[-1].isupper() else ""}'
                    coefs_dict[name] = node_coef_l2[node_name_l2].read()
                    try:
                        coefs_dict['dates'][name] = node_coef_l2[node_name_l2].attrs['timestamp']
                    except KeyError:
                        pass
            else:
                if node_name == 'pid':
                    continue  # metadata, not a coef (mirrors load_coefs_from_nc)
                name = node_name if node_name != 'Vabs0' else 'kVabs'
                coefs_dict[name] = node_coef_l2.read()
                try:
                    coefs_dict['dates'][name] = node_coef_l2.attrs['timestamp']
                except KeyError:
                    pass
        try:
            if isinstance(coefs_dict["date"].item(-1), (bytes, str)):
                coefs_dict["date"] = np.nanmax(np.array(coefs_dict["date"], 'M8[s]'))
            else:
                coefs_dict["date"] = np.datetime64(int(np.nanmax(coefs_dict["date"])), "us")
        except (KeyError, TypeError):
            pass
        if coefs_dict["dates"]:
            coefs_dict["date"] = max(
                [coefs_dict["date"]] + [np.datetime64(d) for d in coefs_dict["dates"].values()]
            )
        return coefs_dict


def get_coefs(
    coefs_paths: Sequence, tbl: str, coefs_ovr: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:

    # Normalize dataclass → dict (coefs_ovr may come from config.ConfigInCoefs_InclProc)
    if is_dataclass(coefs_ovr) and not isinstance(coefs_ovr, type):
        coefs_ovr = asdict(coefs_ovr)
    if OmegaConf.is_config(coefs_ovr):
        coefs_ovr = OmegaConf.to_container(coefs_ovr, resolve=True)

    defaults = {
        k: v_def
        for k, v in config.ConfigInCoefs_InclProc.__dataclass_fields__.items()
        if (v_def := to_omegaconf.get_field_default(v)) is not None
        and not (isinstance(v_def, (list, dict)) and ((not v_def) or not any(lst != [] for lst in v_def)))
    }

    coefs_ovr_dates: Dict[str, Any] = {}
    not_ovr: list = list(defaults)  # assume all are defaults until override proves otherwise
    if coefs_ovr:
        not_ovr = [
            k for k, v_def in defaults.items() if (
                (v_ovr := coefs_ovr.get(k)) == v_def or
                (isinstance(v_ovr, list) and ((not v_ovr) or not any(lst != [] for lst in v_ovr)))
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
            if (coefs_ovr_dates := coefs_ovr.get("dates", {})):
                coefs_ovr_dates = {
                    k: d for k, d in coefs_ovr_dates.items() if k in defaults and k not in not_ovr
                }
        else:
            not_ovr = list(defaults)
            coefs_ovr = {}

    _META_KEYS = frozenset(("dates", "date", "pid"))
    if coefs_paths:
        coefs_load_src: Optional[Path] = None
        for coefs_path in coefs_paths:
            coefs_load = load_coefs(coefs_path, tbl)
            if coefs_load is not None:
                coefs_load_src = coefs_path
                break
        # Also check P_t from loaded coefs (may come from yaml_export, not from coefs_ovr).
        if coefs_load and "P_t" in coefs_load:
            not_ovr = [k for k in not_ovr if k not in ("P", "PBattery", "PTemp")]
        if coefs_load is None:
            lf.error(
                'Not found coefficients table "{:s}" in {}, {:s} redefined from current run config!',
                tbl, coefs_paths,
                "but all are" if not not_ovr else
                f"and {not_ovr} are not" if len(not_ovr) < len(defaults) else "and no one are"
            )
            if len(not_ovr) == len(defaults):
                raise ValueError(f'No coefficients provided / found for device "{tbl}"!')
            coefs_load = {**(coefs_ovr or {}), "dates": coefs_ovr_dates}
        else:
            # Log only when loaded coefs provide keys not already in coefs_ovr.
            _new_from_file = (
                {
                    k
                    for k, v in coefs_load.items()
                    if k not in _META_KEYS and k not in coefs_ovr and not isinstance(v, (str, bytes))
                }
                if coefs_ovr
                else {
                    k
                    for k, v in coefs_load.items()
                    if k not in _META_KEYS and not isinstance(v, (str, bytes))
                }
            )
            if _new_from_file:
                lf.debug("Loaded {} new coefs from {}: {}", len(_new_from_file), coefs_load_src, sorted(_new_from_file))
            else:
                lf.debug("All coefs from {} already in overrides", coefs_load_src)
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

    if coefs_load:
        coefs_load = {
            k: np.asarray(v, dtype=np.float64) if (isinstance(v, list) and v is not None and k != "dates") else v
            for k, v in coefs_load.items()
        }

    out_dates = [
        datetime.fromisoformat(d) if isinstance(d, str) else d
        for d in [coefs_load.get("date")] + list(coefs_load["dates"].values())
        if d
    ]
    if out_dates:
        coefs_load["date"] = max(out_dates)
    return coefs_load


def get_coefs_from_cfg(cfg_in, pcid: str) -> Dict[str, Any]:
    """
    Simplified wrapper: load coefs from config only (no HDF5 paths).
    All coefficients must be defined in cfg_in['coefs'] (YAML config).
    """
    coefs_ovr = cfg_in.get("coefs", {})
    return get_coefs(
        coefs_paths=[],
        tbl=format.pcid_to_raw_name(pcid),
        coefs_ovr=coefs_ovr,
    )


def coefs_format_for_h5(coef: Mapping[str, Any], pcid: str = None, date: Optional[str] = None):
    if coef is None:
        coef = config.ConfigInCoefs_InclProc().__dict__
        del coef['g0xyz']
        if not pcid.split("_")[-1].startswith('p'):
            del coef['P_t']
        if not pcid.startswith('w'):
            del coef["P"]
            del coef["PBattery"]
            del coef["PTemp"]
    elif 'Rz' not in coef and ("Ag" in coef or "Ah" in coef):
        coef['Rz'] = np.eye(3)

    coef_renamed_or_skip = {"Ag", "Cg", "Ah", "Ch", "azimuth_shift_deg", "kVabs", "dates", "i"}
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


# ---------------------------------------------------------------------------
# Probe-level config builders (moved from _dask_legacy/processing.py)
# ---------------------------------------------------------------------------
