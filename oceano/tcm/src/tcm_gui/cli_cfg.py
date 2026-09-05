import copy
import dataclasses
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf

from tcm import schema, to_omegaconf


def default_cfg() -> dict:
    """Return a plain dict with all ``Config.input`` defaults wrapped as ``input`` section."""
    return {"input": _structured_section("input")}


def build_defaults(cls: type) -> dict[str, Any]:
    """Build `{field_name: default}` from a dataclass using :func:`get_field_default`.

    Nested dataclasses are recursively expanded into sub-dicts.
    """
    result: dict[str, Any] = {}
    for fld in dataclasses.fields(cls):
        d = to_omegaconf.get_field_default(fld)
        result[fld.name] = build_defaults(type(d)) if dataclasses.is_dataclass(type(d)) else d
    return result


# Derive section types from Config — single import replaces five.
_SECTION_TYPES: dict[str, type] = {
    name: tp
    for name, tp in get_type_hints(schema.Config).items()
    if name != "defaults" and dataclasses.is_dataclass(tp)
}

# Nested defaults for every config section — single source of truth for gray-out.
CFG_DEFAULTS: dict[str, dict[str, Any]] = {
    section: build_defaults(cls) for section, cls in _SECTION_TYPES.items()
}


def _structured_section(section: str) -> dict[str, Any]:
    """Plain-dict defaults for one ``Config`` section via its structured type."""
    return OmegaConf.to_container(OmegaConf.structured(_SECTION_TYPES[section]()), resolve=True)


def full_default_cfg() -> dict:
    """Plain dict with defaults for every ``Config`` section (full-mode placeholder)."""
    return {section: _structured_section(section) for section in _SECTION_TYPES}


def ensure_full_cfg(cfg: dict) -> dict:
    """Backfill missing ``Config`` sections/leaves in *cfg* with structured defaults.

    Full-mode tree (`ConfigSheet._build_full`) renders whatever keys *cfg*
    carries — a placeholder or thin run YAML holding only ``input`` would show
    just that section. Missing sections are added whole; present sections are
    deep-filled leaf-wise so every option is visible (defaults render dim gray
    via ``CFG_DEFAULTS``). Mutates and returns *cfg*.
    """
    for section in _SECTION_TYPES:
        if not isinstance(cfg.get(section), dict):
            cfg[section] = _structured_section(section)
        else:
            _deep_fill(cfg[section], _structured_section(section))
    return cfg


def _deep_fill(target: dict, defaults: dict) -> None:
    """Insert missing leaves from *defaults* into *target* recursively (in place)."""
    for k, v in defaults.items():
        if k not in target:
            target[k] = copy.deepcopy(v)
        elif isinstance(target[k], dict) and isinstance(v, dict):
            _deep_fill(target[k], v)


# ── Coefficient shape inference ──────────────────────────────────────────────

_COEF_META_SKIP = frozenset(("dates", "date", "path"))  # tree-level metadata, not row items

# Derive ConfigInCoefs_InclProc from Config.input.coefs — no direct import.
COEFS_TYPE: type = type(
    to_omegaconf.get_field_default(
        next(f for f in dataclasses.fields(_SECTION_TYPES["input"]) if f.name == "coefs")
    )
)


def _shape_of(value: Any) -> tuple[int, ...]:
    """Concrete default → shape tuple.  Scalar=(), 1D=(n,), 2D=(n, m)."""
    if isinstance(value, list):
        return (len(value), len(value[0])) if value and isinstance(value[0], list) else (len(value),)
    return ()


def _shape_from_annotation(tp: Any) -> tuple[int, ...]:
    """Extract shape from ``Annotated[T, shape]`` metadata; ``()`` if absent."""
    # Unwrap Optional (Union[X, None])
    if get_origin(tp) is Union:
        tp = next((a for a in get_args(tp) if a is not type(None)), tp)
    if get_origin(tp) is Annotated:
        meta = get_args(tp)[1:]
        if meta:
            m = meta[0]
            return m if isinstance(m, tuple) else (m,)
    return ()


def infer_coef_shapes(coefs_type: type = COEFS_TYPE) -> dict[str, tuple[int, ...]]:
    """Derive coefficient array shapes from a structured config's defaults.

    Shape sources (priority):
      1. Default value structure (when not ``None``)
      2. ``Annotated`` metadata (e.g. ``Annotated[list[float], 3]``)

    Restrictions:
      - Fields must follow the convention: ``List[List[float]]`` for 2D,
        ``List[float]`` for 1D, ``float`` for scalar.
      - ``None``-default fields **must** carry ``Annotated`` metadata.
      - ``dates`` / ``date`` fields are skipped (tree-level metadata).
    """
    hints = get_type_hints(coefs_type, include_extras=True)
    shapes: dict[str, tuple[int, ...]] = {}
    for fld in dataclasses.fields(coefs_type):
        if fld.name in _COEF_META_SKIP:
            continue
        default = to_omegaconf.get_field_default(fld)
        shapes[fld.name] = (
            _shape_of(default) if default is not None else _shape_from_annotation(hints[fld.name])
        )
    return shapes


COEF_SHAPES: dict[str, tuple[int, ...]] = infer_coef_shapes()

NO_DEFAULT = object()


def default_for_path(path: str) -> Any:
    """Walk dotted config path through :data:`CFG_DEFAULTS`, return default or :data:`NO_DEFAULT`.

    Handles array indices (e.g. ``Ag[0]``) and ``None`` → ``""``.
    """
    parts = path.split(".")
    if not parts or parts[0] not in CFG_DEFAULTS:
        return NO_DEFAULT
    current: Any = CFG_DEFAULTS[parts[0]]
    for part in parts[1:]:
        if current is None:
            return ""
        # Handle array indices like "Ag[0]"
        if "[" in part:
            name, idx_str = part.split("[", 1)
            idx = int(idx_str.rstrip("]"))
        else:
            name, idx = part, None
        if isinstance(current, dict) and name in current:
            current = current[name]
        else:
            return NO_DEFAULT
        if idx is not None:
            if current is None:
                return ""
            if isinstance(current, (list, tuple)) and idx < len(current):
                current = current[idx]
            else:
                return NO_DEFAULT
    return current
