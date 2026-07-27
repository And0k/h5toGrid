import dataclasses
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf

from tcm.config import Config
from tcm.to_omegaconf import get_field_default


def default_cfg() -> dict:
    """Return a plain dict with all ``Config.input`` defaults wrapped as ``input`` section."""
    input_type = _SECTION_TYPES["input"]
    return {"input": OmegaConf.to_container(OmegaConf.structured(input_type()), resolve=True)}


def build_defaults(schema: type) -> dict[str, Any]:
    """Build `{field_name: default}` from a dataclass using :func:`get_field_default`.

    Nested dataclasses are recursively expanded into sub-dicts.
    """
    result: dict[str, Any] = {}
    for fld in dataclasses.fields(schema):
        d = get_field_default(fld)
        result[fld.name] = build_defaults(type(d)) if dataclasses.is_dataclass(type(d)) else d
    return result


# Derive section types from Config — single import replaces five.
_SECTION_TYPES: dict[str, type] = {
    name: tp
    for name, tp in get_type_hints(Config).items()
    if name != "defaults" and dataclasses.is_dataclass(tp)
}

# Nested defaults for every config section — single source of truth for gray-out.
CFG_DEFAULTS: dict[str, dict[str, Any]] = {
    section: build_defaults(cls) for section, cls in _SECTION_TYPES.items()
}

# ── Coefficient shape inference ──────────────────────────────────────────────

_COEF_META_SKIP = frozenset(("dates", "date"))  # tree-level metadata, not row items

# Derive ConfigInCoefs_InclProc from Config.input.coefs — no direct import.
COEFS_TYPE: type = type(
    get_field_default(next(f for f in dataclasses.fields(_SECTION_TYPES["input"]) if f.name == "coefs"))
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
        default = get_field_default(fld)
        shapes[fld.name] = (
            _shape_of(default) if default is not None
            else _shape_from_annotation(hints[fld.name])
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