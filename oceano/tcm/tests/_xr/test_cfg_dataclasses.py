"""TDD tests for to_omegaconf — OmegaConf type conversion and merge helpers.

These define expected behavior; code must satisfy them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List, Optional

import numpy as np
import pytest

from tcm.to_omegaconf import to_omegaconf_compatible_type, to_omegaconf_merge_compatible


# Module-level schema (not nested) so @dataclass resolves types correctly
# even with `from __future__ import annotations`.
@dataclass
class _SampleSchema:
    name: str = "default"
    value: float = 0.0
    count: int = 0
    tags: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# to_omegaconf_compatible_type
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestToOmegaconfCompatibleType:
    """Primitive type coercion for OmegaConf structured configs."""

    @pytest.mark.parametrize(
        ("value", "field_type", "expected"),
        [
            # int → float (the crash case)
            pytest.param(0, float, 0.0, id="int-to-float"),
            pytest.param(180, float, 180.0, id="int-to-float-nonzero"),
            # float stays float
            pytest.param(3.14, float, 3.14, id="float-stays"),
            # int stays int
            pytest.param(42, int, 42, id="int-stays"),
            # str stays str
            pytest.param("hello", str, "hello", id="str-stays"),
            # bool stays bool
            pytest.param(True, bool, True, id="bool-stays"),
            # timedelta → int seconds
            pytest.param(timedelta(seconds=300), int, 300, id="timedelta-to-int"),
            # list[1] unwrapped for scalar types
            pytest.param([42], int, 42, id="list1-unwrapped-int"),
            pytest.param([3.14], float, 3.14, id="list1-unwrapped-float"),
            # numpy scalar → python scalar
            pytest.param(np.float64(2.5), float, 2.5, id="numpy-float64-to-float"),
            pytest.param(np.int64(7), int, 7, id="numpy-int64-to-int"),
        ],
    )
    def test_scalar_conversion(self, value, field_type, expected):
        result = to_omegaconf_compatible_type(value, field_type)
        assert result == expected
        assert type(result) is type(expected)

    def test_list_of_floats(self):
        """List[int] with field_type=List[float] → List[float]."""
        result = to_omegaconf_compatible_type([10, -10, 3, 70], List[float])
        assert result == [10.0, -10.0, 3.0, 70.0]
        assert all(isinstance(v, float) for v in result)

    def test_list_of_lists(self):
        """Nested list conversion."""
        result = to_omegaconf_compatible_type(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            List[List[float]],
        )
        assert result == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_none_passes_through(self):
        result = to_omegaconf_compatible_type(None, Optional[float])
        assert result is None

    def test_any_type_passes_through(self):
        result = to_omegaconf_compatible_type("anything", Any)
        assert result == "anything"


# ---------------------------------------------------------------------------
# to_omegaconf_merge_compatible
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestToOmegaconfMergeCompatible:
    """Full dict → structured config conversion."""

    @pytest.mark.parametrize(
        ("data", "check"),
        [
            pytest.param(
                {"name": "test", "value": 180, "count": 5},
                lambda r: r["value"] == 180.0 and type(r["value"]) is float,
                id="int-to-float",
            ),
            pytest.param(
                {"name": "test", "value": 1.0, "count": 1, "tags": None},
                lambda r: "tags" not in r,
                id="skips-none",
            ),
            pytest.param(
                {"name": "test", "value": 1.0, "count": 1, "unknown_key": "x"},
                lambda r: "unknown_key" not in r,
                id="ignores-unknown",
            ),
            pytest.param(
                {"name": "test", "value": 0.0, "count": 0},
                lambda r: "name" in r and "value" not in r and "count" not in r,
                id="skips-defaults",
            ),
        ],
    )
    def test_merge(self, data, check):
        result, _ = to_omegaconf_merge_compatible(data, _SampleSchema)
        assert check(result)
