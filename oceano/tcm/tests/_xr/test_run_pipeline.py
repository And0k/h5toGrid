"""Tests for run() orchestrator pipeline.

Covers: discover → generate configs → process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig

from tcm import _constants, processing, cli
from tcm.config import Return

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def project_dir(tmp_path):
    """Project directory with _raw/ containing test CSV files."""
    raw = tmp_path / _constants.RAW_DIR_NAME
    raw.mkdir()
    for name in ("i_01.txt", "i_02.txt"):
        (raw / name).write_text(
            "2024,06,13,12,00,00,100,200,300\n"
            "2024,06,13,12,00,01,101,201,301\n",
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture()
def make_cfg():
    """Factory fixture: build a DictConfig for run()."""

    def _make(project_dir: Path, ids=None):
        raw = project_dir / _constants.RAW_DIR_NAME
        return DictConfig({
            "input": {"path": str(raw / "*i*.txt"), "ids": ids},
            "out": {"dt_bins": [0], "dir": str(project_dir / "out")},
            "filter": {},
            "program": {"return_": Return.END, "verbose": "INFO"},        })

    return _make


def _mock_config_yaml(mocker, *, existed=None, stale=None, save_ret=None):
    """Patch all config_yaml functions run() calls."""
    mocker.patch("tcm.config_yaml.get_existed_cfgs", return_value=existed or {})
    mocker.patch("tcm.config_yaml.find_stale_cfgs", return_value=set())
    mocker.patch("tcm.config_yaml.save_config_to_yaml", return_value=save_ret or {})
    mocker.patch("tcm.config_yaml.sync_yamls_devmeta_and_hydra")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.xr
@pytest.mark.parametrize(
    ("path_val", "ids_val", "existed", "exc_type", "match"),
    [
        pytest.param(None, None, {}, ValueError, "cfg.input.path", id="no-path"),
        pytest.param("/raw/i.txt", ["i99"], {"i01": ["@i_01"]}, ValueError, "no configs", id="unknown-id"),
    ],
)
def test_run_errors(path_val, ids_val, existed, exc_type, match, tmp_path, mocker):
    """run() raises the expected error for invalid inputs."""
    if path_val and not path_val.startswith("/no"):
        p = tmp_path / _constants.RAW_DIR_NAME / "i.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        path_val = str(p)

    cfg = DictConfig({
        "input": {"path": path_val, "ids": ids_val},
        "out": {},
        "filter": {},
        "program": {"return_": Return.END, "verbose": "INFO"},
    })
    _mock_config_yaml(mocker, existed=existed)
    with pytest.raises(exc_type, match=match):
        processing.run(cfg)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestRunPipeline:
    """run(cfg) happy-path scenarios."""

    def test_generates_and_processes(self, project_dir, make_cfg, mocker):
        """Empty config dir → save_config_to_yaml → process each."""
        cfg = make_cfg(project_dir)
        raw_dir = project_dir / _constants.RAW_DIR_NAME
        run_dir = raw_dir / "cfg_proc" / "run"

        mocker.patch(
            "tcm.config_yaml.get_existed_cfgs",
            side_effect=[{}, {"i01": ["@i_01"], "i02": ["@i_02"]}],
        )
        mocker.patch("tcm.config_yaml.find_stale_cfgs", return_value=set())
        mocker.patch("tcm.config_yaml.save_config_to_yaml", return_value={})
        mock_sync = mocker.patch("tcm.config_yaml.sync_yamls_devmeta_and_hydra")

        def _load_matching_stem(yaml_path):
            """Return config with input.path stem matching the YAML stem."""
            yaml_stem = Path(yaml_path).stem.rsplit("@", 1)[-1]
            return DictConfig({
                "input": {"path": str(raw_dir / f"{yaml_stem}.txt")},
                "out": {"dt_bins": [0]},
                "filter": {},
            })
        mocker.patch("tcm.processing.OmegaConf.load", side_effect=_load_matching_stem)
        mock_proc = mocker.patch("tcm.processing.run_processing")

        run_dir.mkdir(parents=True)
        for stem in ("@i_01", "@i_02"):
            (run_dir / f"{stem}.yaml").write_text("input:\n  path: dummy\n")

        processing.run(cfg)
        assert mock_proc.call_count == 2
        mock_sync.assert_called_once()  # regression: time_ranges sync invoked

    @pytest.mark.parametrize(
        ("ids", "expected_count"),
        [pytest.param(None, 2, id="all-configs"), pytest.param(["i01"], 1, id="filter-by-ids")],
    )
    def test_processes_with_or_without_filter(self, project_dir, make_cfg, mocker, ids, expected_count):
        """Empty id → all configs; explicit ids → filtered."""
        cfg = make_cfg(project_dir, ids=ids)
        raw_dir = project_dir / _constants.RAW_DIR_NAME
        run_dir = raw_dir / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "@i_01.yaml").write_text("input:\n  path: dummy\n")
        (run_dir / "@i_02.yaml").write_text("input:\n  path: dummy\n")
        _mock_config_yaml(mocker, existed={"i01": ["@i_01"], "i02": ["@i_02"]})

        def _load_matching_stem(yaml_path):
            yaml_stem = Path(yaml_path).stem.rsplit("@", 1)[-1]
            return DictConfig({
                "input": {"path": str(raw_dir / f"{yaml_stem}.txt")},
                "out": {"dt_bins": [0]},
                "filter": {},
            })
        mocker.patch("tcm.processing.OmegaConf.load", side_effect=_load_matching_stem)
        mock_proc = mocker.patch("tcm.processing.run_processing")

        processing.run(cfg)
        assert mock_proc.call_count == expected_count

    def test_no_configs_exits_cleanly(self, project_dir, make_cfg, mocker):
        """No configs + no files → no processing, no crash."""
        cfg = make_cfg(project_dir)
        _mock_config_yaml(mocker)
        mock_proc = mocker.patch("tcm.processing.run_processing")

        processing.run(cfg)
        mock_proc.assert_not_called()

    def test_sync_devmeta_called_after_generation(self, project_dir, make_cfg, mocker):
        """Regression: run() must invoke sync_yamls_devmeta_and_hydra after generation.

        Was lost in xr-native replacement (legacy 24xx_clc_hy.py called it;
        processing.run did not), so time_ranges from info_devices.yaml were
        never recorded in run YAMLs.
        """
        cfg = make_cfg(project_dir)
        raw_dir = project_dir / _constants.RAW_DIR_NAME
        run_dir = raw_dir / "cfg_proc" / "run"

        mocker.patch("tcm.config_yaml.get_existed_cfgs", side_effect=[{}, {"i01": ["@i_01"]}])
        mocker.patch("tcm.config_yaml.find_stale_cfgs", return_value=set())
        mocker.patch("tcm.config_yaml.save_config_to_yaml", return_value={})
        mock_sync = mocker.patch("tcm.config_yaml.sync_yamls_devmeta_and_hydra")

        def _load_matching_stem(yaml_path):
            yaml_stem = Path(yaml_path).stem.rsplit("@", 1)[-1]
            return DictConfig({
                "input": {"path": str(raw_dir / f"{yaml_stem}.txt")},
                "out": {"dt_bins": [0]},
                "filter": {},
            })
        mocker.patch("tcm.processing.OmegaConf.load", side_effect=_load_matching_stem)
        mocker.patch("tcm.processing.run_processing")

        run_dir.mkdir(parents=True)
        (run_dir / "@i_01.yaml").write_text("input:\n  path: dummy\n")

        processing.run(cfg)
        mock_sync.assert_called_once()
        # dev_dir = dir_raw.parent (cruise folder, where info_devices.yaml lives)
        assert mock_sync.call_args.args[0] == raw_dir.parent
        assert mock_sync.call_args.args[1] == run_dir

    def test_ghost_config_skipped_when_stem_mismatches_input_path(
        self, project_dir, make_cfg, mocker, caplog,
    ):
        """YAML whose stem ≠ input.path stem is skipped (manual copy/backup)."""
        cfg = make_cfg(project_dir)
        raw_dir = project_dir / _constants.RAW_DIR_NAME
        run_dir = raw_dir / "cfg_proc" / "run"
        run_dir.mkdir(parents=True)

        real_input = raw_dir / "@i_01.txt"
        real_input.write_text("data\n")  # create actual file so stale check passes
        # Real config: stem matches input.path stem
        (run_dir / "@i_01.yaml").write_text(f"input:\n  path: {real_input}\n")
        # Ghost config: manually copied YAML — same input.path as the real one
        ghost_name = "@i_p1 \u2014 \u043a\u043e\u043f\u0438\u044f.yaml"
        (run_dir / ghost_name).write_text(f"input:\n  path: {real_input}\n")

        _mock_config_yaml(mocker, existed={
            "i01": ["@i_01", "@i_p1 \u2014 \u043a\u043e\u043f\u0438\u044f"],
        })

        # Both YAMLs point to the same real_input file (simulates copy scenario)
        cfg_for_real = DictConfig({
            "input": {"path": str(real_input)}, "out": {"dt_bins": [0]}, "filter": {},
        })
        cfg_for_ghost = DictConfig({
            "input": {"path": str(real_input)}, "out": {"dt_bins": [0]}, "filter": {},
        })
        mocker.patch(
            "tcm.processing.OmegaConf.load",
            side_effect=[cfg_for_real, cfg_for_ghost],
        )
        mock_proc = mocker.patch("tcm.processing.run_processing")

        with caplog.at_level("WARNING", logger="tcm.processing"):
            processing.run(cfg)

        # Only the real config is processed; ghost is skipped
        assert mock_proc.call_count == 1
        assert "skipping" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Binary inputs (NC/HDF5) — skip text-file probe discovery
# ---------------------------------------------------------------------------

@pytest.mark.xr
class TestBinaryInputSkipsDiscovery:
    """run(cfg) bypasses ``cfg_proc/run/*.yaml`` discovery for NC/HDF5.

    Binary files carry their own coefs (``/{tbl}/coef/``) and are not part of
    the per-text-file discovery pipeline.  ``run`` must call ``run_processing``
    directly per ``input.tables`` entry and never touch ``config_yaml.*``.
    """

    def _cfg(self, project_dir, tables):
        raw = project_dir / _constants.RAW_DIR_NAME
        raw.mkdir(exist_ok=True)
        return DictConfig({
            "input": {
                "path": str(raw / "260624.raw.nc"),
                "tables": list(tables),
            },
            "out": {"dt_bins": [0], "dir": str(project_dir / "out")},
            "filter": {},
            "program": {"return_": Return.END, "verbose": "INFO"},        })

    def test_skip_discovery_for_nc(self, tmp_path, mocker):
        """NC path → no ``config_yaml`` calls; one ``run_processing`` per table."""
        cfg = self._cfg(tmp_path, ["incl_p05"])
        mock_get = mocker.patch("tcm.config_yaml.get_existed_cfgs")
        mock_save = mocker.patch("tcm.config_yaml.save_config_to_yaml")
        mock_sync = mocker.patch("tcm.config_yaml.sync_yamls_devmeta_and_hydra")
        mock_proc = mocker.patch("tcm.processing.run_processing")

        processing.run(cfg)

        mock_get.assert_not_called()
        mock_save.assert_not_called()
        mock_sync.assert_not_called()
        mock_proc.assert_called_once()
        assert mock_proc.call_args[0][0].input.tables == ["incl_p05"]

    @pytest.mark.parametrize("ext", [".nc", ".nc4", ".h5", ".hdf5"])
    def test_callable_for_all_binary_exts(self, tmp_path, mocker, ext):
        """Every binary extension triggers the direct path."""
        cfg = self._cfg(tmp_path, ["incl63"])
        cfg.input.path = cfg.input.path.replace(".nc", ext)
        mocker.patch("tcm.processing.run_processing")
        for name in ("get_existed_cfgs", "save_config_to_yaml",
                    "sync_yamls_devmeta_and_hydra"):
            mocker.patch(f"tcm.config_yaml.{name}")
        processing.run(cfg)  # must not raise / not fall through to discovery

    def test_one_run_processing_per_table(self, tmp_path, mocker):
        """Multiple tables in one binary file → one ``run_processing`` each."""
        cfg = self._cfg(tmp_path, ["incl63", "incl64"])
        mocker.patch("tcm.config_yaml.get_existed_cfgs")
        mocker.patch("tcm.config_yaml.save_config_to_yaml")
        mocker.patch("tcm.config_yaml.sync_yamls_devmeta_and_hydra")
        mock_proc = mocker.patch("tcm.processing.run_processing")

        processing.run(cfg)

        assert mock_proc.call_count == 2
        pinned = [list(c.args[0].input.tables) for c in mock_proc.call_args_list]
        assert pinned == [["incl63"], ["incl64"]]


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_hydra():
    """Reset GlobalHydra state between tests."""
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()
    sys.modules.pop("scripts.tcm_clc", None)


@pytest.mark.xr
class TestMainEndToEnd:
    """main() → @hydra.main → run(cfg)."""

    def test_composes_and_calls_run(self, tmp_path, monkeypatch, mocker):
        """Reproduces: python -m scripts.tcm_clc D:/data/_raw/*i*.txt"""
        raw_dir = tmp_path / _constants.RAW_DIR_NAME
        raw_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", str(raw_dir / "*I*.txt")])

        mock_run = mocker.patch("tcm.processing.run")
        cli.call_in_raw_dir(processing.run)

        mock_run.assert_called_once()
        cfg = mock_run.call_args[0][0]
        assert cfg.input.path is not None
        assert cfg.input is not None
        assert cfg.out is not None
        assert cfg.filter is not None
        assert cfg.program is not None

    @pytest.mark.parametrize(
        "extra_args,check",
        [
            pytest.param(
                ["out.text_path=./custom"],
                lambda cfg: "custom" in str(cfg.out.text_path),
                id="override-out-path",
            ),
            pytest.param(
                ["input.ids=[i01]"],
                lambda cfg: list(cfg.input.ids) == ["i01"],
                id="override-ids",
            ),
        ],
    )
    def test_main_passes_overrides(self, tmp_path, monkeypatch, mocker, extra_args, check):
        """Hydra CLI overrides flow through to run(cfg)."""
        raw_dir = tmp_path / _constants.RAW_DIR_NAME
        raw_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", str(raw_dir / "*I*.txt")] + extra_args)

        mock_run = mocker.patch("tcm.processing.run")
        cli.call_in_raw_dir(processing.run)

        cfg = mock_run.call_args[0][0]
        assert check(cfg), f"Override not applied: {extra_args}"

    def test_main_rejects_missing_raw(self, tmp_path, monkeypatch, mocker):
        """main() exits or errors when input has no _raw/ ancestor."""
        f = tmp_path / "no_raw_here.txt"
        f.touch()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", "no_raw_here.txt"])

        mock_run = mocker.patch("tcm.processing.run")
        cli.call_in_raw_dir(processing.run)
        cfg = mock_run.call_args[0][0]
        # Path resolved without _raw/ — find_dir_raw_absolute falls back to parent
        assert cfg.input.path is not None

    def test_main_autoassigns_raw_dir(self, tmp_path, monkeypatch, mocker, caplog):
        """main() proceeds when input lacks `_raw/` ancestor — with a warning."""
        no_raw = tmp_path / "incoming"
        no_raw.mkdir()
        (no_raw / "i_01.txt").write_text("2024,06,13,12,00,00,100,200,300\n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", str(no_raw / "i_01.txt")])

        mock_run = mocker.patch("tcm.processing.run")
        with caplog.at_level("WARNING", logger="tcm.processing"):
            cli.call_in_raw_dir(processing.run)

        mock_run.assert_called_once()
        assert "Using deepest dir as anchor" in caplog.text, (
            f"Expected 'Using deepest dir as anchor' warning in log, got: {caplog.text}"
        )

    def test_main_rejects_data_inside_project(self, tmp_path, monkeypatch, mocker):
        """main() rejects data directory inside the code project."""

        raw_dir = tmp_path / _constants.RAW_DIR_NAME
        raw_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", str(raw_dir / "*I*.txt")])
        monkeypatch.setattr(_constants, "PROJECT_ROOT", tmp_path)



        with pytest.raises(SystemExit):
            cli.call_in_raw_dir(processing.run)
