"""Tests for path resolution — find_dir_raw, find_dir_raw_absolute.

Expected project layout::

    data_dir/            ← find_dir_raw_absolute returns THIS
        _raw/            ← find_dir_raw returns THIS
            i_01.txt
        cfg_proc/

TDD: these define expected behavior; code must satisfy them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tcm._constants import RAW_DIR_NAME
from tcm.paths import find_dir_raw_absolute
from tcm.paths import find_dir_raw


def _touch_inside(tmp_path: Path, rel: str) -> Path:
    """Create a file at tmp_path / rel, making parents as needed."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


@pytest.mark.xr
class TestFindDirRaw:
    """find_dir_raw returns the _raw/ directory itself."""

    @pytest.mark.parametrize(
        ("rel_path", "expected_raw_offset"),
        [
            pytest.param(f"{RAW_DIR_NAME}/i_01.txt", RAW_DIR_NAME, id="file-inside-raw"),
            pytest.param(f"{RAW_DIR_NAME}/sub/i_01.txt", RAW_DIR_NAME, id="nested-inside-raw"),
        ],
    )
    def test_returns_raw_dir(self, tmp_path, rel_path, expected_raw_offset):
        f = _touch_inside(tmp_path, rel_path)
        assert find_dir_raw(f) == tmp_path / expected_raw_offset

    def test_returns_none_when_no_raw_ancestor(self, tmp_path):
        f = _touch_inside(tmp_path, "other/i_01.txt")
        assert find_dir_raw(f) is None


@pytest.mark.xr
class TestFindDirRawAbsolute:
    """find_dir_raw_absolute returns the _raw/ directory.

    The ``_raw`` directory is the anchor for all relative paths:
    ``cfg_proc/run/`` (configs), ``cfg_proc/log/`` (Hydra logs).
    """

    @pytest.mark.parametrize(
        "rel_path",
        [
            pytest.param(f"{RAW_DIR_NAME}/i_01.txt", id="file-inside-raw"),
            pytest.param(f"{RAW_DIR_NAME}/*i*.txt", id="glob-inside-raw"),
            pytest.param(f"{RAW_DIR_NAME}/sub/i_01.txt", id="nested-file"),
        ],
    )
    def test_returns_raw_dir(self, tmp_path, rel_path):
        """All paths under _raw/ resolve to _raw/ itself."""
        p = tmp_path / rel_path
        if "*" not in rel_path:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch() if not p.exists() else None
        assert find_dir_raw_absolute(p) == tmp_path / RAW_DIR_NAME

    def test_handles_relative_path_with_raw(self):
        """Relative paths containing _raw/ are resolved via parent walk."""
        result = find_dir_raw_absolute(Path(f"{RAW_DIR_NAME}/i_01.txt"))
        assert result == Path(RAW_DIR_NAME), (
            f"Expected {RAW_DIR_NAME} dir for relative raw path, got {result}"
        )

    def test_fallback_when_no_raw_ancestor(self, tmp_path):
        f = _touch_inside(tmp_path, "experiment/i_01.txt")
        assert find_dir_raw_absolute(f) == tmp_path / "experiment"
