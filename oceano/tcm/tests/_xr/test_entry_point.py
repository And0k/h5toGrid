"""Tests for tcm.cli, tcm.config ConfigStore registration, and bundled config accessibility."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from tcm import _constants
from tcm.cli import DEFAULT_GLOB, parse_data_path
from tcm.config_yaml import has_run_yamls


@pytest.mark.xr
class TestParseDataPath:
    """parse_data_path extracts positional arg and returns remaining argv."""

    @pytest.mark.parametrize(
        ("argv", "expected_posix", "expected_remaining_len"),
        [
            pytest.param(["prog", "_raw/*i*.txt", "--multirun", "run=glob(*)"], "_raw/*i*.txt", 3, id="positional-first"),
            pytest.param(["prog", "--multirun", "run=glob(*)"], "_raw/*I*.txt", 3, id="default-glob"),
            pytest.param(["prog", "some/path", "program.return_=generate"], "some/path", 2, id="with-override"),
        ],
    )
    def test_parse_data_path(self, argv, expected_posix, expected_remaining_len):
        path_in, remaining = parse_data_path(argv)
        assert PurePosixPath(expected_posix) == PurePosixPath(path_in.as_posix())
        assert remaining[0] == argv[0]
        assert len(remaining) == expected_remaining_len

    def test_no_positional_uses_default(self):
        path_in, remaining = parse_data_path(["prog", "key=val", "--flag"])
        assert PurePosixPath(DEFAULT_GLOB) == PurePosixPath(path_in.as_posix())
        assert len(remaining) == 3

    @pytest.mark.parametrize(
        ("argv", "expected_last_segment", "expected_remaining_len"),
        [
            pytest.param(
                ["prog", "B:/Cruises/BalticSea/251201_ABP64@i", "t-chain/inclinometer/_raw/file.txt"],
                "t-chain/inclinometer/_raw/file.txt",
                1,
                id="comma-split-@i,t-chain",
            ),
            pytest.param(
                ["prog", "B:/data/250101@w", "t-chain@i_p5/inclinometer/file.txt"],
                "t-chain@i_p5/inclinometer/file.txt",
                1,
                id="multiple-commas",
            ),
        ],
    )
    def test_comma_split_path_rejoined(self, argv, expected_last_segment, expected_remaining_len):
        """Path fragments split by comma are rejoined with commas."""
        path_in, remaining = parse_data_path(argv)
        assert "," in str(path_in), (
            f"Comma not present in reconstructed path: {path_in}"
        )
        assert path_in.as_posix().endswith(expected_last_segment), (
            f"Expected path ending with {expected_last_segment}, got {path_in.as_posix()}"
        )
        assert len(remaining) == expected_remaining_len, (
            f"Expected {expected_remaining_len} remaining args, got {len(remaining)}: {remaining}"
        )

    def test_comma_split_with_trailing_flags(self):
        """Flags after comma-split path are preserved in remaining."""
        argv = ["prog", "B:/data/251201_ABP64@i", "t-chain/inclinometer/_raw/file.txt", "filter.corr_time_mode=false"]
        path_in, remaining = parse_data_path(argv)
        assert "," in str(path_in), (
            f"Comma missing in reconstructed path: {path_in}"
        )
        assert "filter.corr_time_mode=false" in remaining, (
            f"Flag lost from remaining: {remaining}"
        )


@pytest.mark.xr
class TestHydraArgvQuoting:
    """input.path is injected directly into DictConfig, bypassing Hydra's override parser."""

    def test_input_path_not_in_hydra_argv(self, tmp_path):
        """input.path must NOT appear in Hydra argv — injected via OmegaConf instead."""
        from tcm.cli import _build_hydra_argv

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        argv = _build_hydra_argv(data_dir)

        input_path_arg = next((a for a in argv if "input.path" in a), None)
        assert input_path_arg is None, (
            f"input.path must not appear in Hydra argv (bypasses ANTLR parser), got: {input_path_arg}"
        )

    def test_searchpath_in_hydra_argv(self, tmp_path):
        """cfg_proc exists → --config-dir added to argv (argparse layer, not ANTLR)."""
        from tcm.cli import _build_hydra_argv

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "cfg_proc").mkdir()

        argv = _build_hydra_argv(data_dir)

        assert "--config-dir" in argv, f"--config-dir missing from argv: {argv}"
        idx = argv.index("--config-dir")
        assert argv[idx + 1] == str(data_dir / "cfg_proc"), (
            f"--config-dir value wrong: {argv[idx + 1]}"
        )

    def test_searchpath_absent_without_cfg_proc(self, tmp_path):
        """No --config-dir when cfg_proc dir doesn't exist."""
        from tcm.cli import _build_hydra_argv

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        argv = _build_hydra_argv(data_dir)

        assert "--config-dir" not in argv, f"--config-dir should be absent: {argv}"

    @pytest.mark.parametrize(
        "data_dir_name",
        [
            pytest.param("251201_ABP64@i,t-chain", id="comma"),
            pytest.param("path(with)parens", id="parens"),
            pytest.param("path[with]brackets", id="brackets"),
            pytest.param("path{with}braces", id="braces"),
            pytest.param("path=with=equals", id="equals"),
        ],
    )
    def test_searchpath_no_escape_needed(self, tmp_path, data_dir_name):
        """--config-dir passes raw native path — argparse handles ALL special chars."""
        from tcm.cli import _build_hydra_argv

        data_dir = tmp_path / data_dir_name / "_raw"
        (data_dir / "cfg_proc").mkdir(parents=True)

        argv = _build_hydra_argv(data_dir)

        assert "--config-dir" in argv, f"--config-dir missing from argv: {argv}"
        idx = argv.index("--config-dir")
        # No escaping — argparse passes the path through as-is
        assert argv[idx + 1] == str(data_dir / "cfg_proc"), (
            f"--config-dir value should be raw native path, got: {argv[idx + 1]}"
        )
        # No backslash-escaping, no quoting
        assert "\\" not in argv[idx + 1] or "\\" in str(data_dir / "cfg_proc"), (
            f"Unexpected escaping in --config-dir value: {argv[idx + 1]}"
        )

    @pytest.mark.parametrize(
        "path_str",
        [
            pytest.param("251201_ABP64@i,t-chain/file.txt", id="comma"),
            pytest.param("251201_ABP64@i,t-chain/O'Brien/file.txt", id="comma+single-quote"),
            pytest.param('251201_ABP64@i,t-chain/O"Brien/file.txt', id="comma+double-quote"),
            pytest.param("_raw/file.txt", id="no-special-chars"),
        ],
    )
    def test_input_path_injected_via_overrides(self, tmp_path, path_str):
        """input.path injected into overrides dict for any path — OmegaConf merge bypasses ANTLR."""
        from tcm.cli import _prepare_overrides

        path_in = Path(path_str)
        overrides: dict = {}

        result = _prepare_overrides(path_in, overrides)

        assert result["input"]["path"] == Path(path_str).as_posix(), (
            f"input.path not in overrides: {result}"
        )

    def test_input_path_merges_with_existing_overrides(self, tmp_path):
        """Existing overrides are preserved when input.path is injected."""
        from tcm.cli import _prepare_overrides

        path_in = tmp_path / "data" / "file.txt"
        overrides = {"filter": {"min": 0.1}}

        result = _prepare_overrides(path_in, overrides)

        assert result["input"]["path"] == path_in.as_posix(), (
            f"input.path not in overrides: {result}"
        )
        assert result["filter"]["min"] == 0.1, (
            f"Existing override lost: {result}"
        )


@pytest.mark.xr
class TestBundledConfigAccessible:
    """tcm.cfg.cfg_proc must be a valid Python package with __init__.py."""

    _CFG_PROC = _constants.CFG_PATH / "cfg_proc"

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(_constants.CFG_PATH / "__init__.py", id="cfg/__init__.py"),
            pytest.param(_CFG_PROC / "__init__.py", id="cfg_proc/__init__.py"),
            pytest.param(_CFG_PROC / "config.yaml", id="config.yaml"),
        ],
    )
    def test_required_file_exists(self, path):
        assert path.is_file(), (
            f"Missing {path} — Hydra module resolution requires __init__.py in every package directory"
        )


@pytest.mark.xr
class TestConfigStoreRegistration:
    """tcm.config must register structured config groups with Hydra ConfigStore."""

    @staticmethod
    def _cs_repo():
        from hydra.core.config_store import ConfigStore
        return ConfigStore.instance().repo

    @pytest.mark.parametrize("group", [
        pytest.param("input", id="input/base"),
        pytest.param("out", id="out/base"),
        pytest.param("filter", id="filter/base"),
        pytest.param("program", id="program/base"),
    ])
    def test_group_registered(self, group):
        repo = self._cs_repo()
        assert group in repo and "base.yaml" in repo[group], (
            f"ConfigStore missing '{group}/base'"
        )


@pytest.mark.xr
class TestHasRunYamls:
    """has_run_yamls correctly detects YAML files in cfg_proc/run/."""

    def test_empty_dir_returns_false(self, tmp_path):
        assert has_run_yamls(tmp_path / "nonexistent") is False

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            pytest.param("@i_01.yaml", True, id="yaml-present"),
            pytest.param("readme.txt", False, id="no-yaml"),
        ],
    )
    def test_dir_with_files(self, tmp_path, filename, expected):
        run_dir = tmp_path / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / filename).write_text("content\n")
        assert has_run_yamls(run_dir) is expected
