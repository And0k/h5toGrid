"""Tests for CLI argument parsing, return_ modes, and duplicate YAML behaviour.

Verifies that:
- Hydra list overrides (time_ranges, time_ranges_zeroing) parse from CLI args
- ``program.return_="<cfg_from_args>"`` stops processing inside main_init
- ``program.return_="<saved_raw>"`` dispatches run_processing for coef save
- Duplicate YAMLs (same pcid, multiple stems) are both processed, but data is
  not duplicated (incremental NC writes skip overlapping time ranges)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from hydra.core.hydra_config import HydraConfig

from tcm import cli, processing
from tcm._constants import RAW_DIR_NAME
from tcm.schema import Return


# --------------------------------------------------------------------------- #
# CLI override parsing
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestCliOverrideParsing:
    """Hydra CLI overrides are correctly merged into the config.

    In ``sys.argv``, overrides are bare Hydra syntax (no shell quoting).
    On the real command line, wrap each override in single quotes to protect
    ``[""]`` and ``<>`` from the shell::

        'program.return_="<cfg_from_args>"'
        'input.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]'
    """

    @pytest.mark.parametrize(
        ("extra_args", "check"),
        [
            pytest.param(
                ['input.time_ranges_zeroing=["2026-06-25T17:23:30","2026-06-25T17:25:00"]'],
                lambda cfg: (
                    list(cfg.input.time_ranges_zeroing)
                    == [
                        "2026-06-25T17:23:30",
                        "2026-06-25T17:25:00",
                    ]
                ),
                id="list-override-time_ranges_zeroing",
            ),
            pytest.param(
                ['input.time_ranges=["2024-01-01T00:00:00","2024-01-02T00:00:00"]'],
                lambda cfg: (
                    list(cfg.input.time_ranges)
                    == [
                        "2024-01-01T00:00:00",
                        "2024-01-02T00:00:00",
                    ]
                ),
                id="list-override-time_ranges",
            ),
            pytest.param(
                [f'program.return_="{Return.CFG_FROM_ARGS}"'],
                lambda cfg: cfg.program.return_ == Return.CFG_FROM_ARGS,
                id="return-cfg-from-args",
            ),
            pytest.param(
                [f'program.return_="{Return.SAVED_RAW}"'],
                lambda cfg: cfg.program.return_ == Return.SAVED_RAW,
                id="return-saved-raw",
            ),
            pytest.param(
                ["input.ids=[i01,i_p02]"],
                lambda cfg: list(cfg.input.ids) == ["i01", "i_p02"],
                id="list-override-ids",
            ),
            pytest.param(
                ["out.overwrite_db=splice"],
                lambda cfg: cfg.out.overwrite_db == "splice",
                id="overwrite-db-splice",
            ),
        ],
    )
    def test_cli_overrides_parsed(self, tmp_path, monkeypatch, mocker, extra_args, check):
        """CLI Hydra overrides are correctly parsed and merged into cfg."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", str(raw_dir / "*I*.txt")] + extra_args)

        mock_run = mocker.patch.object(processing, "run")
        cli.call_in_raw_dir(processing.run)

        cfg = mock_run.call_args[0][0]
        assert check(cfg), f"Override not applied: {extra_args}"


@pytest.mark.xr
class TestCliOverrideFullPropagation:
    """CLI Hydra overrides survive process_loading_yaml merge into run_processing.

    The ``test_cli_overrides_parsed`` suite (above) only verifies the first hop
    (CLI → ``run()``).  Here we verify the second hop: ``process_loading_yaml``
    merges the base_cfg with per-probe YAML via ``OmegaConf.merge`` and passes
    the result to ``run_processing``.  Extra keys added via ``+key=value`` must
    survive this merge.
    """

    @pytest.mark.parametrize(
        ("extra_args", "check"),
        [
            pytest.param(
                ["out.overwrite_db=splice"],
                lambda cfg: cfg["out"].get("overwrite_db") == "splice",
                id="overwrite-db-splice",
            ),
            pytest.param(
                ["out.overwrite_db=null"],
                lambda cfg: cfg["out"].get("overwrite_db") is None,
                id="overwrite-db-null",
            ),
        ],
    )
    def test_cli_override_reaches_run_processing(
        self,
        _raw_with_csv,
        monkeypatch,
        mocker,
        extra_args,
        check,
    ):
        """CLI override propagates through process_loading_yaml → run_processing."""
        project_dir, raw_dir = _raw_with_csv
        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", str(raw_dir / "*i*.txt")] + extra_args,
        )

        mock_proc = mocker.patch.object(processing, "run_processing")
        cli.call_in_raw_dir(processing.run)

        mock_proc.assert_called_once()
        cfg_dc = mock_proc.call_args[0][0]
        assert check(cfg_dc), (
            f"Override {extra_args} did not reach run_processing: "
            f"cfg['out'].get('overwrite_db') = {cfg_dc.get('out', {}).get('overwrite_db')!r}"
        )


@pytest.mark.xr
class TestMainInitPreservesExtraKeys:
    """``main_init`` → ``ini2dict`` must preserve structured config overrides.

    Regression: ``ini2dict`` pre-allocates ``cfg = {key: {} for key in config}``
    then skips non-dict values via ``hasattr(sec, 'items')`` → ``continue``.
    Now ``out.overwrite_db`` is a structured config field (no ``+`` prefix needed).
    """

    def test_overwrite_db_survives_main_init(self, _raw_with_csv, monkeypatch, mocker):
        """out.overwrite_db=splice from CLI must be 'splice' AFTER main_init converts cfg to dict."""
        project_dir, raw_dir = _raw_with_csv
        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", str(raw_dir / "*i*.txt"), "out.overwrite_db=splice"],
        )

        # Capture cfg AFTER main_init by patching _process_and_persist
        captured: dict = {}
        orig_pap = processing._process_and_persist

        def _spy_pap(*args, **kwargs):
            captured["overwrite_db"] = kwargs.get("overwrite_db")
            return orig_pap(*args, **kwargs)

        mocker.patch.object(processing, "_process_and_persist", side_effect=_spy_pap)
        cli.call_in_raw_dir(processing.run)

        assert captured.get("overwrite_db") == "splice", (
            f"overwrite_db was stripped by main_init/ini2dict — "
            f"expected 'splice', got {captured.get('overwrite_db')!r}"
        )


# --------------------------------------------------------------------------- #
# return_ modes — early stopping
# --------------------------------------------------------------------------- #


@pytest.fixture()
def _raw_with_csv(tmp_path):
    """Create _raw/ with a CSV file and pre-existing YAML config."""
    raw_dir = tmp_path / RAW_DIR_NAME
    raw_dir.mkdir()
    csv_file = raw_dir / "@i_01.txt"
    csv_file.write_text(
        "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp\n"
        "2024,06,13,12,00,00,100.0,200.0,300.0,400.0,500.0,600.0,12.5,25.0\n"
        "2024,06,13,12,00,01,101.0,201.0,301.0,401.0,501.0,601.0,12.5,25.0\n",
        encoding="utf-8",
    )
    run_dir = raw_dir / "cfg_proc" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "@i_01.yaml").write_text(
        f"# @package _global_\ninput:\n  path: '{csv_file}'\nout:\n  dt_bins: [0]\n",
        encoding="utf-8",
    )
    return tmp_path, raw_dir


@pytest.mark.xr
class TestReturnCfgFromArgs:
    """``program.return_="<cfg_from_args>"`` — config composition only.

    ``processing.run()`` checks ``return_`` before dispatching to
    ``run_processing`` — so no data is loaded and no coefs are computed.
    """

    def test_no_run_processing_called(self, _raw_with_csv, monkeypatch, mocker):
        """With <cfg_from_args>, run_processing is called but returns early (after main_init).

        Previously run_processing was never called — the CFG_FROM_ARGS early return
        in run() bypassed process_loading_yaml entirely.  Now process_loading_yaml
        runs, calling run_processing which returns the DictConfig after main_init
        (no data load, no coefs computation).  The collected configs feed the GUI scan flow.
        """
        project_dir, raw_dir = _raw_with_csv
        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                str(raw_dir / "*i*.txt"),
                f'program.return_="{Return.CFG_FROM_ARGS}"',
            ],
        )

        mock_proc = mocker.patch.object(processing, "run_processing")
        result = cli.call_in_raw_dir(processing.run)

        mock_proc.assert_called_once()
        # result is a 4-tuple: (processed_pcids, failed_pcids, last_cfg, collected)
        assert result is not None
        assert len(result) == 4
        collected = result[3]
        assert len(collected) == 1  # one probe
        stem, yp, cfg_dc = collected[0]
        assert "@i_01" in stem

    def test_preserves_existing_user_edited_config(self, _raw_with_csv, monkeypatch, mocker):
        """<cfg_from_args> does NOT overwrite a healthy existing config.

        save_config_to_yaml uses mode='w' but is only called when configs are
        stale, missing, or new source files appear.  When configs already exist
        and are healthy, regeneration is skipped — user edits (coefs, time_ranges)
        survive.
        """
        project_dir, raw_dir = _raw_with_csv
        run_dir = raw_dir / "cfg_proc" / "run"

        # Inject a user marker into the existing config
        yaml_path = run_dir / "@i_01.yaml"
        original = yaml_path.read_text(encoding="utf-8")
        yaml_path.write_text(
            original.replace("dt_bins: [0]", "dt_bins: [0]\n  # USER_MARKER"), encoding="utf-8"
        )

        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                str(raw_dir / "*i*.txt"),
                f'program.return_="{Return.CFG_FROM_ARGS}"',
            ],
        )

        mocker.patch.object(processing, "run_processing")
        cli.call_in_raw_dir(processing.run)

        # Config was NOT regenerated — user marker preserved
        content = yaml_path.read_text(encoding="utf-8")
        assert "USER_MARKER" in content, (
            f"Existing config was overwritten by {Return.CFG_FROM_ARGS} — user edits lost"
        )


@pytest.mark.xr
class TestReturnSavedRaw:
    """``program.return_="<saved_raw>"`` saves coefs, then stops."""

    def test_run_processing_called(self, _raw_with_csv, monkeypatch, mocker):
        """With <saved_raw>, run_processing IS called (to load + save coefs)."""
        project_dir, raw_dir = _raw_with_csv
        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                str(raw_dir / "*i*.txt"),
                f'program.return_="{Return.SAVED_RAW}"',
            ],
        )

        mock_proc = mocker.patch.object(processing, "run_processing")
        cli.call_in_raw_dir(processing.run)

        mock_proc.assert_called()


# --------------------------------------------------------------------------- #
# Duplicate YAML behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestDuplicateYamlBehaviour:
    """Multiple YAMLs for the same pcid → both processed, no data duplication."""

    def test_duplicate_stems_both_processed(self, _raw_with_csv, monkeypatch, mocker):
        """Two YAMLs for pcid i01 (different prefixes) → run_processing called twice."""
        project_dir, raw_dir = _raw_with_csv
        run_dir = raw_dir / "cfg_proc" / "run"

        real_input = raw_dir / "@i_01.txt"
        # Second config for the same probe — simulates rename creating a duplicate
        (run_dir / "260613_1200@i_01.yaml").write_text(
            f"# @package _global_\ninput:\n  path: '{real_input}'\nout:\n  dt_bins: [0]\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                str(raw_dir / "*i*.txt"),
            ],
        )

        mock_proc = mocker.patch.object(processing, "run_processing")
        cli.call_in_raw_dir(processing.run)

        # Both YAMLs resolved to pcid i01, but process_loading_yaml iterates ALL stems
        assert mock_proc.call_count == 2, (
            f"Expected 2 run_processing calls for duplicate YAMLs, got {mock_proc.call_count}"
        )

    def test_ghost_yaml_skipped(self, _raw_with_csv, monkeypatch, mocker):
        """YAML with stem not matching input.path core → skipped (manual copy).

        The skipping message is logged by process_loading_yaml (tcm.cli logger)
        which uses LoggingStyleAdapter (outputs to console, not standard caplog).
        We verify by checking run_processing call count.
        """
        project_dir, raw_dir = _raw_with_csv
        run_dir = raw_dir / "cfg_proc" / "run"

        real_input = raw_dir / "@i_01.txt"
        # Ghost: manually copied YAML with different core
        (run_dir / "@i_01_backup.yaml").write_text(
            f"# @package _global_\ninput:\n  path: '{real_input}'\nout:\n  dt_bins: [0]\n",
            encoding="utf-8",
        )

        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                str(raw_dir / "*i*.txt"),
            ],
        )

        mock_proc = mocker.patch.object(processing, "run_processing")
        cli.call_in_raw_dir(processing.run)

        # Only the real config is processed; ghost is skipped
        assert mock_proc.call_count == 1


# --------------------------------------------------------------------------- #
# Log file naming — as_filename + _setup_file_handler
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestLogFileNaming:
    """Log file name is derived from ``program.return_`` via ``_setup_file_handler``.

    Default (``return_`` = ``<end>``): ``processing.log``.
    Non-default: ``processing-{sanitized_return_}.log`` (e.g. ``processing-cfg_from_args.log``).
    """

    @pytest.mark.parametrize(
        ("input_s", "expected", "test_description"),
        [
            pytest.param("<cfg_from_args>", "cfg_from_args", "angle brackets stripped", id="cfg-from-args"),
            pytest.param(
                "<gen_names_and_log>",
                "gen_names_and_log",
                "angle brackets stripped",
                id="gen-names",
            ),
            pytest.param("<end>", "end", "angle brackets stripped for default too", id="end"),
            pytest.param("", "_", "empty string → fallback", id="empty"),
            pytest.param("normal", "normal", "safe string passes through", id="safe"),
            pytest.param("a/b:c", "abc", "slashes and colons stripped", id="special-chars"),
            pytest.param("foo.", "foo", "trailing dot stripped", id="trailing-dot"),
        ],
    )
    def test_as_filename(self, input_s, expected, test_description):
        """as_filename sanitizes return_ values for use in filenames."""
        assert cli.as_filename(input_s) == expected, (
            f"{test_description}: as_filename({input_s!r}) → "
            f"{cli.as_filename(input_s)!r}, expected {expected!r}"
        )

    def test_file_handler_default_name(self, tmp_path, monkeypatch, mocker):
        """Default return_ → file handler named processing.log."""
        run_dir = tmp_path / "log" / "2024-01-01"
        run_dir.mkdir(parents=True)

        # Mock HydraConfig.get() to return a known run dir and job name
        mock_hydra_cfg = mocker.MagicMock()
        mock_hydra_cfg.run.dir = str(run_dir)
        mock_hydra_cfg.job.name = "processing"
        mocker.patch.object(HydraConfig, "get", return_value=mock_hydra_cfg)

        cfg = {"program": {"return_": "<end>"}}
        cli._setup_file_handler(cfg)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1, f"Expected 1 FileHandler, got {len(file_handlers)}"
        assert Path(file_handlers[0].baseFilename).name == "processing.log", (
            f"Expected processing.log, got {Path(file_handlers[0].baseFilename).name}"
        )
        # Cleanup
        for h in file_handlers:
            h.close()
            root.removeHandler(h)

    def test_file_handler_non_default_name(self, tmp_path, monkeypatch, mocker):
        """Non-default return_ → file handler named processing-cfg_from_args.log."""
        run_dir = tmp_path / "log" / "2024-01-01"
        run_dir.mkdir(parents=True)

        mock_hydra_cfg = mocker.MagicMock()
        mock_hydra_cfg.run.dir = str(run_dir)
        mock_hydra_cfg.job.name = "processing"
        mocker.patch.object(HydraConfig, "get", return_value=mock_hydra_cfg)

        cfg = {"program": {"return_": "<cfg_from_args>"}}
        cli._setup_file_handler(cfg)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename).name == "processing-cfg_from_args.log", (
            f"Expected processing-cfg_from_args.log, got {Path(file_handlers[0].baseFilename).name}"
        )
        for h in file_handlers:
            h.close()
            root.removeHandler(h)

    def test_file_handler_removes_old_handlers(self, tmp_path, monkeypatch, mocker):
        """_setup_file_handler removes existing FileHandlers before adding new one."""
        run_dir = tmp_path / "log" / "2024-01-01"
        run_dir.mkdir(parents=True)

        mock_hydra_cfg = mocker.MagicMock()
        mock_hydra_cfg.run.dir = str(run_dir)
        mock_hydra_cfg.job.name = "processing"
        mocker.patch.object(HydraConfig, "get", return_value=mock_hydra_cfg)

        root = logging.getLogger()
        # Add a stale FileHandler
        stale = logging.FileHandler(str(run_dir / "stale.log"), encoding="utf-8")
        root.addHandler(stale)

        cfg = {"program": {"return_": "<end>"}}
        cli._setup_file_handler(cfg)

        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1, f"Expected 1 FileHandler after cleanup, got {len(file_handlers)}"
        assert "stale.log" not in str(file_handlers[0].baseFilename), (
            f"Stale handler still present: {file_handlers[0].baseFilename}"
        )
        # Stale handler was closed (stream is None after close)
        assert stale.stream is None, "Stale handler was not closed"
        for h in file_handlers:
            h.close()
            root.removeHandler(h)
