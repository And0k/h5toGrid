import dataclasses
from typing import Any

from omegaconf import OmegaConf
from tcm import config
from tcm.config import ConfigFilter_InclProc, ConfigIn_InclProc, ConfigOut_InclProc, ConfigProgram
from tcm.to_omegaconf import get_field_default


def default_cfg() -> dict:
    """Return a plain dict with all ``config.ConfigIn_InclProc`` defaults wrapped as ``input`` section."""
    return {"input": OmegaConf.to_container(OmegaConf.structured(config.ConfigIn_InclProc()), resolve=True)}


def build_defaults(schema: type) -> dict[str, Any]:
    """Build `{field_name: default}` from a dataclass using :func:`get_field_default`.

    Nested dataclasses are recursively expanded into sub-dicts.
    """
    result: dict[str, Any] = {}
    for fld in dataclasses.fields(schema):
        d = get_field_default(fld)
        result[fld.name] = build_defaults(type(d)) if dataclasses.is_dataclass(type(d)) else d
    return result


# Nested defaults for every config section — single source of truth for gray-out.
CFG_DEFAULTS: dict[str, dict[str, Any]] = {
    section: build_defaults(cls)
    for section, cls in [
        ("input", ConfigIn_InclProc),
        ("out", ConfigOut_InclProc),
        ("filter", ConfigFilter_InclProc),
        ("program", ConfigProgram),
    ]
}

COEF_SHAPES: dict[str, tuple[int, ...]] = {
    "Ag": (3, 3),
    "Cg": (3,),
    "Ah": (3, 3),
    "Ch": (3,),
    "Rz": (3, 3),
    "kVabs": (6,),
    "P_t": (3, 3),
    "P": (2,),
    "PBattery": (2,),
    "PTemp": (2,),
    "azimuth_shift_deg": (),
    "g0xyz": (3,),
}