# !/usr/bin/env python3
# coding:utf-8
"""
Base dataclasses and OmegaConf helpers shared by the xr-native pipeline.

Legacy Hydra machinery (ConfigInCsv, ConfigOut, main_call, etc.) has been
moved to ``_dask_legacy/cfg_compat.py``.
"""
import sys

from dataclasses import Field, fields, is_dataclass, MISSING
from datetime import date as datetime_date
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Mapping, Optional, Union, get_args, get_origin, get_type_hints
from collections import abc

from omegaconf import DictConfig, ListConfig

from tcm import config
from numpy import ndarray


# ---------------------------------------------------------------------------
# OmegaConf conversion utilities
# ---------------------------------------------------------------------------

def timedelta_to_iso8601(timedelta_obj):
    """Convert a :class:`timedelta` to an ISO-8601 duration string."""
    total_seconds = int(timedelta_obj.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"PT{hours}H{minutes}M{seconds}S"


def to_omegaconf_compatible_types(value):
    """Recursively convert value to OmegaConf-compatible primitives."""
    # numpy scalars (np.float64, np.int64, etc.) pass isinstance(x, float/int)
    # but are not representable by ruamel.yaml or OmegaConf — convert to native.
    if hasattr(value, 'dtype') and value.ndim == 0:
        return value.item()
    match value:
        case None | int() | float() | str() | bool():
            return value
        case DictConfig() | abc.Mapping():
            return {k: to_omegaconf_compatible_types(v) for k, v in value.items()}
        case ListConfig() | abc.Sequence():
            return [to_omegaconf_compatible_types(item) for item in value]
        case datetime_date() | datetime():
            return value.isoformat()
        case timedelta():
            return timedelta_to_iso8601(value)
        case ndarray():
        # case _ if type(value).__name__ == "ndarray" and type(value).__module__.startswith("numpy"):
            return value.tolist() if value.size > 1 else value.item()
        case _:
            return str(value)


def is_optional_type(field_type) -> bool:
    """Check if a field is of type Optional (Union[X, None])."""
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def get_generic_and_actual_type(type_hint):
    """Unwrap Annotated and Optional types until a generic type or bare type is found.

    :return: the first generic type found (else None) along with its immediate type argument.
    """
    generic_type = None
    while True:
        generic_type = get_origin(type_hint)
        if generic_type is None:
            break

        type_args = get_args(type_hint)
        if generic_type == Union:
            type_hint = next((arg for arg in type_args if arg is not type(None)), None)
            continue
        if generic_type == Annotated:
            type_hint = type_args[0]
            continue
        type_hint = get_args(type_hint)
        break
    return generic_type, type_hint


def get_field_default(fld: Field):
    """Return the default value for a dataclass field (or ``None`` if MISSING)."""
    match fld:
        case Field(default=v) if v is not MISSING:
            return v
        case Field(default_factory=f) if f is not MISSING:
            return f()
        case _:
            return None


def to_omegaconf_compatible_type(value, field_type, default_value=None):
    """Convert *value* to an OmegaConf-compatible type matching *field_type*.

    Handles Optional, dataclass, Annotated, collection, numpy, datetime, and
    timedelta types.  Returns ``None`` for values equal to the *default_value*.
    """
    if value is None:
        return None

    origin_type, field_type = get_generic_and_actual_type(field_type)

    if origin_type:
        if origin_type in (list, tuple, dict):
            if isinstance(value, (dict, DictConfig)):
                converted_dict = {}
                _, value_type = field_type
                for k, v in value.items():
                    v = to_omegaconf_compatible_type(
                        v, value_type,
                        default_value.get(k) if default_value not in (MISSING, None) else None,
                    )
                    if v is None:
                        continue
                    converted_dict[k] = v
                return converted_dict
            else:
                if hasattr(value, 'dtype'):
                    value = value.tolist()
                return [to_omegaconf_compatible_type(item, field_type[0]) for item in value]

    if is_dataclass(field_type):
        return to_omegaconf_merge_compatible(value, field_type)[0]

    if field_type in (int, float, str, bool):
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        elif isinstance(value, timedelta):
            return int(value.total_seconds())

    # int → float coercion (Python int is not a subclass of float)
    if field_type is float and isinstance(value, int):
        return float(value)

    if hasattr(value, "dtype"):
        value = value.item()

    if field_type is not Any:
        try:
            if isinstance(value, field_type):
                return value
        except TypeError:
            pass  # field_type is not a concrete type (e.g. Optional[float])

    if field_type not in (Any, str):
        raise TypeError(f"Can not convert {value!r} to {field_type}")

    if isinstance(value, (datetime_date, datetime)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return timedelta_to_iso8601(value)
    return str(value)


def to_omegaconf_merge_compatible(unstructured: Mapping[str, Any], schema) -> Dict[str, Any]:
    """Convert *unstructured* dict to OmegaConf-merge-compatible form against *schema*.

    Excludes ``None`` fields (keeps ``None`` elements in lists) and fields
    equal to default values.  Handles nested structured configs and numpy types.

    :return: ``(converted_dict, ignored_keys)``.
    """
    if not is_dataclass(schema):
        raise TypeError("Provided schema is not a dataclass.")
    # Use get_type_hints() to resolve string annotations (from __future__ import annotations)
    resolved_types = get_type_hints(schema)
    schema_fields = {fld.name: (resolved_types[fld.name], get_field_default(fld)) for fld in fields(schema)}
    converted: Dict[str, Any] = {}
    ignored_keys: list = []
    for key, value in unstructured.items():
        if key in schema_fields:
            if value is not None:
                if (
                    converted_value := to_omegaconf_compatible_type(value, *schema_fields[key])
                ) != schema_fields[key][1]:
                    converted[key] = converted_value
        else:
            ignored_keys.append(key)
    return converted, ignored_keys
