"""Unit tests for _cell_spec — Hydra type classification for tksheet rendering."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

import pytest

from tcm_gui._cell_spec import (
    BOOL_SPEC,
    DATE_SPEC,
    NUMBER_SPEC,
    TEXT_SPEC,
    CellSpec,
    _unwrap,
    as_bool,
    classify_type,
    enum_values,
    iso_secs,
    resolve_dataclass_field,
)


# ── fixtures: enums + dataclasses for resolution tests ──────────────────────


class Color(Enum):
    RED = "red"
    GREEN = "green"


class Return(StrEnum):
    END = "<end>"
    CFG = "<cfg_from_args>"


@dataclass
class Inner:
    flag: bool = False
    value: float = 1.0


@dataclass
class Outer:
    name: str = "test"
    inner: Inner = field(default_factory=Inner)
    color: Color = Color.RED
    count: Optional[int] = None


@dataclass
class Root:
    program: Outer = field(default_factory=Outer)
    labels: List[str] = field(default_factory=list)


# ── _unwrap ─────────────────────────────────────────────────────────────────


class TestUnwrap:
    def test_plain_type_passthrough(self):
        """Plain type returns itself."""
        assert _unwrap(int) is int
        assert _unwrap(str) is str

    def test_optional_unwrapped(self):
        """Optional[X] → X."""
        assert _unwrap(Optional[int]) is int

    def test_annotated_unwrapped(self):
        """Annotated[X, ...] → X."""
        assert _unwrap(Annotated[int, "meta"]) is int

    def test_nested_annotated_optional(self):
        """Annotated[Optional[X], ...] → X."""
        assert _unwrap(Annotated[Optional[float], "hint"]) is float

    def test_union_single_non_none(self):
        """Union[X, None] → X."""
        assert _unwrap(Union[str, None]) is str

    def test_union_multi_returns_first(self):
        """Union[X, Y] → first non-None type."""
        result = _unwrap(Union[int, float])
        assert result is int

    def test_plain_class_passthrough(self):
        """A plain class is returned as-is."""
        assert _unwrap(Path) is Path


# ── classify_type ───────────────────────────────────────────────────────────


class TestClassifyType:
    @pytest.mark.parametrize(
        ("tp", "expected"),
        [
            pytest.param(bool, BOOL_SPEC, id="bool"),
            pytest.param(int, NUMBER_SPEC, id="int"),
            pytest.param(float, NUMBER_SPEC, id="float"),
            pytest.param(str, TEXT_SPEC, id="str"),
            pytest.param(Path, TEXT_SPEC, id="Path"),
            pytest.param(date, DATE_SPEC, id="date"),
            pytest.param(datetime, DATE_SPEC, id="datetime"),
            pytest.param(Color, CellSpec("enum", Color), id="enum"),
            pytest.param(Return, CellSpec("enum", Return), id="StrEnum"),
            pytest.param(dict, TEXT_SPEC, id="dict_fallback"),
            pytest.param(list, TEXT_SPEC, id="list_fallback"),
            pytest.param(Optional[int], NUMBER_SPEC, id="Optional_int"),
            pytest.param(Annotated[float, "x"], NUMBER_SPEC, id="Annotated_float"),
        ],
    )
    def test_classify_type(self, tp, expected):
        """Type → correct CellSpec kind (and enum for Enum types)."""
        result = classify_type(tp)
        assert result.kind == expected.kind, f"{tp}: kind {result.kind} != {expected.kind}"
        assert result.enum is expected.enum, f"{tp}: enum {result.enum} != {expected.enum}"


# ── resolve_dataclass_field ────────────────────────────────────────────────


class TestResolveDataclassField:
    def test_simple_field(self):
        """Direct field resolves to its type."""
        assert resolve_dataclass_field(Outer, "name") is str

    def test_nested_field(self):
        """Dotted path walks into nested dataclasses."""
        tp = resolve_dataclass_field(Root, "program.inner.flag")
        assert tp is bool

    def test_optional_field(self):
        """Optional[X] unwraps to X."""
        assert resolve_dataclass_field(Outer, "count") is int

    def test_enum_field(self):
        """Enum field returns the Enum class."""
        assert resolve_dataclass_field(Outer, "color") is Color

    def test_unknown_field_returns_any(self):
        """Non-existent field → Any."""
        assert resolve_dataclass_field(Outer, "nonexistent") is Any

    def test_non_dataclass_path_returns_any(self):
        """Walking past a non-dataclass → Any."""
        assert resolve_dataclass_field(Outer, "name.length") is Any

    def test_empty_path(self):
        """Empty path returns the root type."""
        assert resolve_dataclass_field(Outer, "") is Outer

    def test_list_field_returns_list_origin(self):
        """List[str] → type with list origin."""
        from typing import get_origin

        tp = resolve_dataclass_field(Root, "labels")
        assert get_origin(tp) is list, f"expected list origin, got {tp}"


# ── as_bool ─────────────────────────────────────────────────────────────────


class TestAsBool:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(True, True, id="py_true"),
            pytest.param(False, False, id="py_false"),
            pytest.param(1, True, id="int_1"),
            pytest.param(0, False, id="int_0"),
            pytest.param("true", True, id="str_true"),
            pytest.param("True", True, id="str_True"),
            pytest.param("TRUE", True, id="str_TRUE"),
            pytest.param("yes", True, id="str_yes"),
            pytest.param("on", True, id="str_on"),
            pytest.param("1", True, id="str_1"),
            pytest.param("false", False, id="str_false"),
            pytest.param("0", False, id="str_0"),
            pytest.param("", False, id="str_empty"),
            pytest.param("no", False, id="str_no"),
        ],
    )
    def test_as_bool(self, value, expected):
        assert as_bool(value) is expected, f"as_bool({value!r}): {as_bool(value)} != {expected}"


# ── enum_values ─────────────────────────────────────────────────────────────


class TestEnumValues:
    def test_str_enum_values(self):
        """StrEnum uses .value (which equals the string)."""
        assert enum_values(Return) == ["<end>", "<cfg_from_args>"]

    def test_plain_enum_names(self):
        """Regular Enum also uses .value."""
        assert enum_values(Color) == ["red", "green"]


# ── iso_secs ────────────────────────────────────────────────────────────────


class TestIsoSecs:
    def test_canonical_iso_t(self):
        """Every accepted spelling → ISO ``T``-separated seconds."""
        assert iso_secs("2024-01-15 10:30:00") == "2024-01-15T10:30:00"
        assert iso_secs("2024-01-15T10:30:00") == "2024-01-15T10:30:00"
        assert iso_secs("15.01.2024") == "2024-01-15T00:00:00"
        assert iso_secs("2024-01-15") == "2024-01-15T00:00:00"

    def test_unparseable_and_empty(self):
        """Unparseable → None; empties fall back via ``iso_secs(s) or s``."""
        assert iso_secs("not a date") is None
        assert iso_secs("") is None
        assert (iso_secs("") or "") == ""
