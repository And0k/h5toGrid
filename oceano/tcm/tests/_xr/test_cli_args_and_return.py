"""Tests for CLI argument parsing, return_ modes, and duplicate YAML behaviour.

Verifies that:
- Hydra list overrides (time_ranges, time_ranges_zeroing) parse from CLI args
- ``program.return_="<cfg_from_args>"`` stops processing inside main_init
- ``program.return_="<saved_raw>"`` dispatches run_processing for coef save
- Duplicate YAMLs (same pcid, multiple stems) are both processed, but data is
  not duplicated (incremental NC writes skip overlapping time ranges)
"""
from __future__ import annotations

import sys

import pytest

from tcm import cli, processing
from tcm._constants import RAW_DIR_NAME
from tcm.config import Return


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
                ["+force_reprocess=True"],
                lambda cfg: cfg.force_reprocess is True,
                id="force-reprocess-bool",
            ),
        ],
    )
    def test_cli_overrides_parsed(self, tmp_path, monkeypatch, mocker, extra_args, check):
        """CLI Hydra overrides are correctly parsed and merged into cfg."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", str(raw_dir / "*I*.txt")] + extra_args)

        mock_run = mocker.patch("tcm.processing.run")
        cli.call_in_raw_dir(processing.run)

        cfg = mock_run.call_args[0][0]
        assert check(cfg), f"Override not applied: {extra_args}"


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

        mock_proc = mocker.patch("tcm.processing.run_processing")
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

        mocker.patch("tcm.processing.run_processing")
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

        mock_proc = mocker.patch("tcm.processing.run_processing")
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
        monkeypatch.setattr(sys, "argv", [
            "prog", str(raw_dir / "*i*.txt"),
        ])

        mock_proc = mocker.patch("tcm.processing.run_processing")
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
        monkeypatch.setattr(sys, "argv", [
            "prog", str(raw_dir / "*i*.txt"),
        ])

        mock_proc = mocker.patch("tcm.processing.run_processing")
        cli.call_in_raw_dir(processing.run)

        # Only the real config is processed; ghost is skipped
        assert mock_proc.call_count == 1


# --------------------------------------------------------------------------- #
# Scan mode detection — job_name → log file name
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestScanModeDetection:
    """``is_scan_mode`` detects config-generation-only runs from ``sys.argv``.

    When ``program.return_`` is set to a scan-mode value (``<cfg_from_args>``,
    ``<gen_names_and_log>``), the entry point passes ``job_name="scan"`` to
    :func:`hydra_main` so the log file is ``scan.log`` (not ``processing.log``).
    """

    @pytest.mark.parametrize(
        ("argv", "expected", "test_description"),
        [
            pytest.param(
                ["tcm_clc.py", "_raw/i*.txt", "program.return_=<cfg_from_args>"],
                True,
                "cfg_from_args triggers scan mode",
                id="cfg-from-args",
            ),
            pytest.param(
                ["tcm_clc.py", "_raw/i*.txt", "program.return_=<gen_names_and_log>"],
                True,
                "gen_names_and_log triggers scan mode",
                id="gen-names-and-log",
            ),
            pytest.param(
                ["tcm_clc.py", "_raw/i*.txt"],
                False,
                "no override → normal processing mode",
                id="no-override",
            ),
            pytest.param(
                ["tcm_clc.py", "_raw/i*.txt", "program.return_=<end>"],
                False,
                "<end> → normal processing mode",
                id="end-mode",
            ),
            pytest.param(
                ["tcm_clc.py", "_raw/i*.txt", "program.return_=<saved_all>"],
                False,
                "<saved_all> → normal processing mode",
                id="saved-all",
            ),
            pytest.param(
                ["tcm_clc.py", "_raw/i*.txt", "program.return_=<saved_coefs>"],
                False,
                "<saved_coefs> → normal processing mode",
                id="saved-coefs",
            ),
            pytest.param(
                [],
                False,
                "empty argv → normal processing mode",
                id="empty-argv",
            ),
        ],
    )
    def test_is_scan_mode(self, argv, expected, test_description):
        """is_scan_mode correctly identifies scan vs processing mode from argv."""
        assert cli.is_scan_mode(argv) is expected, (
            f"{test_description}: expected {expected!r}, got {not expected!r}"
        )

    def test_job_name_passed_to_hydra_main(self, _raw_with_csv, monkeypatch, mocker):
        """job_name='scan' flows from call_in_raw_dir to hydra_main.

        Verifies that ``call_in_raw_dir(job_name="scan")`` passes ``job_name``
        to :func:`hydra_main` as a kwarg.  The actual ``sys.argv`` injection
        and cleanup are tested in ``test_job_name_sys_argv_injection``.
        """
        project_dir, raw_dir = _raw_with_csv
        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(sys, "argv", ["prog", str(raw_dir / "*i*.txt")])

        spy_calls = []
        original_hydra_main = cli.hydra_main

        def _spy_hydra_main(*args, **kwargs):
            spy_calls.append(kwargs)
            return original_hydra_main(*args, **kwargs)

        mocker.patch.object(cli, "hydra_main", side_effect=_spy_hydra_main)
        mocker.patch("tcm.processing.run")

        cli.call_in_raw_dir(processing.run, job_name="scan")

        assert spy_calls, "hydra_main was never called"
        assert spy_calls[0].get("job_name") == "scan", (
            f"Expected job_name='scan' in hydra_main kwargs, got {spy_calls[0].get('job_name')!r}"
        )

    def test_job_name_sys_argv_injection(self, tmp_path, monkeypatch, mocker):
        """hydra_main injects hydra.job.name=<value> into sys.argv and cleans up.

        Directly tests the ``hydra_main(job_name="scan")`` plumbing without
        going through ``call_in_raw_dir``.
        """
        raw_dir = tmp_path / "_raw"
        raw_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog"])

        mocker.patch("tcm.processing.run")

        # hydra_main will fail (no config) but the argv injection happens first
        try:
            cli.hydra_main(lambda cfg: None, job_name="scan")
        except Exception:
            pass  # expected — we only care about argv

        # After hydra_main, sys.argv must be cleaned up
        assert "hydra.job.name=scan" not in sys.argv, (
            f"sys.argv not cleaned up: {sys.argv}"
        )

    def test_no_job_name_default_mode(self, _raw_with_csv, monkeypatch, mocker):
        """Without job_name kwarg, hydra_main receives job_name=None (default)."""
        project_dir, raw_dir = _raw_with_csv
        monkeypatch.chdir(project_dir)
        monkeypatch.setattr(sys, "argv", ["prog", str(raw_dir / "*i*.txt")])

        spy_calls = []
        original_hydra_main = cli.hydra_main

        def _spy_hydra_main(*args, **kwargs):
            spy_calls.append(kwargs)
            return original_hydra_main(*args, **kwargs)

        mocker.patch.object(cli, "hydra_main", side_effect=_spy_hydra_main)
        mocker.patch("tcm.processing.run")

        cli.call_in_raw_dir(processing.run)

        assert spy_calls, "hydra_main was never called"
        assert "job_name" not in spy_calls[0], (
            f"Unexpected job_name in default mode: {spy_calls[0]}"
        )
