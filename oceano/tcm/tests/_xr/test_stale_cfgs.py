"""Tests for find_stale_cfgs — stale detection without ProbeFiles."""
from __future__ import annotations


import pytest

from tcm.config_yaml import find_stale_cfgs


@pytest.mark.xr
class TestFindStaleCfgs:
    """find_stale_cfgs detects stale configs by checking YAML input.path existence."""

    def test_all_valid_returns_empty(self, tmp_path):
        """When all YAML input.paths exist, returns empty dict."""
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        real_file = tmp_path / "data.txt"
        real_file.write_text("data\n")
        (run_dir / "@i_01.yaml").write_text(f"input:\n path: {real_file}\n")
        assert find_stale_cfgs({"i01": ["@i_01"]}, run_dir) == {}

    def test_missing_path_detected(self, tmp_path):
        """YAML referencing non-existent file → pcid in stale dict with its stem."""
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "@i_01.yaml").write_text("input:\n path: /nonexistent/file.txt\n")
        result = find_stale_cfgs({"i01": ["@i_01"]}, run_dir)
        assert "i01" in result, f"{result=!r}"
        assert result["i01"] == ["@i_01"], f"{result=!r}"

    def test_missing_yaml_file_detected(self, tmp_path):
        """When YAML file itself doesn't exist, pcid is stale with its stem."""
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        result = find_stale_cfgs({"i01": ["@i_01"]}, run_dir)
        assert "i01" in result, f"{result=!r}"
        assert result["i01"] == ["@i_01"], f"{result=!r}"

    def test_no_probe_files_needed(self, tmp_path):
        """Signature has no ProbeFiles parameter — works without discover_probes."""
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        real_file = tmp_path / "data.txt"
        real_file.write_text("data\n")
        (run_dir / "@i_01.yaml").write_text(f"input:\n path: {real_file}\n")
        (run_dir / "@i_02.yaml").write_text("input:\n path: /nonexistent/other.txt\n")
        result = find_stale_cfgs({"i01": ["@i_01"], "i02": ["@i_02"]}, run_dir)
        assert result == {"i02": ["@i_02"]}, f"{result=!r}"

    def test_multiple_stems_per_pcid(self, tmp_path):
        """When a pcid has multiple stems, all stale ones are listed."""
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        real_file = tmp_path / "data.txt"
        real_file.write_text("data\n")
        # i67 has two YAMLs — one stale, one valid
        (run_dir / "251204_1823@i_067.yaml").write_text("input:\n path: /nonexistent/old.txt\n")
        (run_dir / "251204_1823@i_67.yaml").write_text(f"input:\n path: {real_file}\n")
        result = find_stale_cfgs({"i67": ["251204_1823@i_067", "251204_1823@i_67"]}, run_dir)
        assert result == {"i67": ["251204_1823@i_067"]}, f"{result=!r}"
