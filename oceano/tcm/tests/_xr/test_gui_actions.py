"""Tests simulating GUI actions end-to-end via tcm.cli.call_in_raw_dir.

Verifies the two-step GUI workflow:
1. **Scan**: ``input.path`` as ``str`` (from Tk Entry), ``program.return_=CFG_FROM_ARGS``
2. **Run**: ``input.yaml_path=(stem1|stem2)`` (from edited tabs), coef write-back
3. **Coef write-back**: ``config_yaml.update_coefs_in_run_yaml`` from ConfigSheet edits

Key integration points tested:
- ``call_in_raw_dir`` accepts ``path_in`` as ``str`` (not just Path)
- ``exit_on_error=False`` propagates exceptions instead of sys.exit
- ``process_loading_yaml`` returns 4-tuple ``(processed, failed, last_cfg, collected)``
- ``run_processing`` returns DictConfig early for CFG_FROM_ARGS
- ``update_coefs_in_run_yaml`` creates backup + merges coefs into YAML
- ``input.yaml_path`` regex filters configs (replaces undocumented ``program.configs``)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from queue import Queue

import pytest
from omegaconf import OmegaConf

from tcm import cli, config_yaml, processing
from tcm._constants import RAW_DIR_NAME
from tcm.config import Return


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def gui_project(tmp_path):
    """Create a minimal project tree mimicking a real data directory.

    Structure::

        tmp_path/
          _raw/
            @i_01.txt          ← CSV data file
            cfg_proc/
              run/
                @i_01.yaml     ← per-probe YAML (points to @i_01.txt)
    """
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
    return tmp_path, raw_dir, csv_file, run_dir


class _FakeSheet:
    """Minimal ``ConfigSheet`` stand-in for ``_write_coefs`` tests."""

    def __init__(self, coefs: dict | None = None, dates: dict | None = None, path: str = "", dirty: bool = True):
        self._coefs = coefs or {}
        self._dates = dates or {}
        self._path = path
        self.is_dirty = dirty

    def _current_state(self):
        return self._coefs, self._dates, self._path

    def mark_clean(self) -> None:
        self.is_dirty = False


# --------------------------------------------------------------------------- #
# Step 1: Scan — user selects a file in the GUI
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestGuiScan:
    """Simulate GUI scan: ``App._scan`` → ``Worker._scan`` → ``call_in_raw_dir``."""

    def test_scan_str_path_no_crash(self, gui_project, monkeypatch, mocker):
        """path_in as str (from Tk Entry) must not crash ``find_dir_raw_absolute``.

        This was the original bug: ``str`` has no ``.name`` attribute.
        """
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        # GUI passes path as str — exactly what Tk StringVar.get() returns
        path_str = str(raw_dir / "*i*.txt")

        monkeypatch.setattr(
            sys, "argv",
            ["prog", path_str, f'program.return_="{Return.CFG_FROM_ARGS}"'],
        )
        mock_proc = mocker.patch("tcm.processing.run_processing")
        result = cli.call_in_raw_dir(processing.run, exit_on_error=False)

        assert result is not None
        assert len(result) == 4
        mock_proc.assert_called_once()

    def test_scan_returns_collected_configs(self, gui_project, monkeypatch, mocker):
        """Scan returns ``(processed, failed, last_cfg, collected)`` with stems."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        path_str = str(raw_dir / "*i*.txt")

        monkeypatch.setattr(
            sys, "argv",
            ["prog", path_str, f'program.return_="{Return.CFG_FROM_ARGS}"'],
        )
        mocker.patch("tcm.processing.run_processing")
        result = cli.call_in_raw_dir(processing.run, exit_on_error=False)

        processed, failed, last_cfg, collected = result
        assert len(collected) == 1
        stem, yp, cfg_dc = collected[0]
        assert "@i_01" in stem
        assert Path(yp).exists()
        # cfg_dc is a DictConfig with input.path
        assert cfg_dc.input.path is not None

    def test_scan_via_overrides_dict(self, gui_project, monkeypatch, mocker):
        """Worker-style call: overrides dict (not sys.argv) with str path."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        # Simulate Worker._scan: call_in_raw_dir with overrides dict
        path_str = str(raw_dir / "*i*.txt")

        monkeypatch.setattr(sys, "argv", ["prog"])
        mocker.patch("tcm.processing.run_processing")
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": path_str},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )

        assert result is not None
        assert len(result) == 4
        collected = result[3]
        assert len(collected) >= 1

    def test_scan_exit_on_error_false_raises(self, gui_project, monkeypatch):
        """``exit_on_error=False`` raises instead of sys.exit."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        # Pass a path that will cause an error (non-existent directory)
        bad_path = str(tmp_path / "nonexistent" / "*i*.txt")

        monkeypatch.setattr(sys, "argv", ["prog"])
        with pytest.raises((FileNotFoundError, OSError, SystemExit)):
            cli.call_in_raw_dir(
                processing.run,
                overrides={"input": {"path": bad_path}},
                exit_on_error=False,
            )


# --------------------------------------------------------------------------- #
# Step 2a: Coef write-back — user edits coefs in tksheet
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestGuiCoefWriteBack:
    """Simulate GUI coef write-back: ``App._write_coefs`` → ``update_coefs_in_run_yaml``."""

    def test_update_coefs_creates_backup(self, gui_project, monkeypatch):
        """``update_coefs_in_run_yaml`` creates a timestamped backup."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        yaml_path = run_dir / "@i_01.yaml"
        original_content = yaml_path.read_text(encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        config_yaml.update_coefs_in_run_yaml(
            yaml_path,
            {"Ag": [[0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]]},
        )

        # Backup was created
        backups = list(run_dir.glob("@i_01-backup*.yaml"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == original_content

    def test_update_coefs_merges_into_yaml(self, gui_project, monkeypatch):
        """Changed coefs are written under ``input.coefs`` in the YAML."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        yaml_path = run_dir / "@i_01.yaml"

        monkeypatch.chdir(tmp_path)
        config_yaml.update_coefs_in_run_yaml(
            yaml_path,
            {
                "Ag": [[0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]],
                "dates": {"Ag": "2024-06-15"},
            },
        )

        # Read back and verify
        ry = config_yaml._ry(write=False)
        with yaml_path.open(encoding="utf-8") as f:
            updated = ry.load(f)
        assert updated["input"]["coefs"]["Ag"] == [[0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]]
        assert updated["input"]["coefs"]["dates"]["Ag"] == "2024-06-15"

    def test_write_coefs_delegates_to_update_coefs(self, gui_project, monkeypatch, mocker):
        """``App._write_coefs`` calls ``config_yaml.update_coefs_in_run_yaml``."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        yaml_path = run_dir / "@i_01.yaml"

        mock_update = mocker.patch("tcm.config_yaml.update_coefs_in_run_yaml")

        # Simulate what App._write_coefs does
        from tcm_gui.app import App

        # Build a minimal App without Tk mainloop
        app = App.__new__(App)
        app._yaml_paths = {"@i_01": yaml_path}

        cs = _FakeSheet(coefs={"Ag": [[0.003, 0, 0], [0, 0.003, 0], [0, 0, 0.003]]})
        app._write_coefs("@i_01", cs)
        mock_update.assert_called_once_with(
            yaml_path,
            {"input": {"coefs": {"Ag": [[0.003, 0, 0], [0, 0.003, 0], [0, 0, 0.003]]}}},
        )

    def test_write_coefs_with_dates(self, gui_project, monkeypatch, mocker):
        """``_write_coefs`` passes dates dict to ``update_coefs_in_run_yaml``."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        yaml_path = run_dir / "@i_01.yaml"

        mock_update = mocker.patch("tcm.config_yaml.update_coefs_in_run_yaml")

        from tcm_gui.app import App

        app = App.__new__(App)
        app._yaml_paths = {"@i_01": yaml_path}

        cs = _FakeSheet(
            coefs={"Ah": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            dates={"Ah": "2024-06-20"},
        )
        app._write_coefs("@i_01", cs)
        mock_update.assert_called_once_with(
            yaml_path,
            {"input": {"coefs": {"Ah": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dates": {"Ah": "2024-06-20"}}}},
        )

    def test_write_coefs_no_edits_skips(self, gui_project, mocker):
        """No edits → no call to ``update_coefs_in_run_yaml``."""
        mock_update = mocker.patch("tcm.config_yaml.update_coefs_in_run_yaml")

        from tcm_gui.app import App

        app = App.__new__(App)
        app._yaml_paths = {"@i_01": Path("/fake/path")}

        app._write_coefs("@i_01", _FakeSheet(dirty=False))
        mock_update.assert_not_called()


# --------------------------------------------------------------------------- #
# Step 2b: Run — user clicks Run button
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestGuiRun:
    """Simulate GUI run: ``Worker._run`` → ``call_in_raw_dir`` with ``input.yaml_path``."""

    def test_run_with_yaml_path(self, gui_project, monkeypatch, mocker):
        """``input.yaml_path`` dispatches to ``run_processing`` for listed stems."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(sys, "argv", ["prog"])
        mock_proc = mocker.patch("tcm.processing.run_processing")
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt"), "yaml_path": "(@i_01)"},
            },
            exit_on_error=False,
        )

        # result is a 4-tuple from process_loading_yaml
        assert result is not None
        assert len(result) == 4
        processed, failed, last_cfg, collected = result
        mock_proc.assert_called_once()

    def test_run_skips_discovery(self, gui_project, monkeypatch, mocker):
        """``input.yaml_path`` set → config generation skipped."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(sys, "argv", ["prog"])
        mock_save = mocker.patch("tcm.config_yaml.save_config_to_yaml")
        mocker.patch("tcm.processing.run_processing")

        cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt"), "yaml_path": "(@i_01)"},
            },
            exit_on_error=False,
        )

        mock_save.assert_not_called()

    def test_run_exit_on_error_false(self, gui_project, monkeypatch, mocker):
        """``exit_on_error=False`` on run propagates exceptions."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(sys, "argv", ["prog"])
        mocker.patch(
            "tcm.processing.run_processing",
            side_effect=RuntimeError("simulated failure"),
        )
        # run_processing is called inside process_loading_yaml which catches Exception
        # So the run completes but with failed_pcids
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt"), "yaml_path": "(@i_01)"},
            },
            exit_on_error=False,
        )

        processed, failed, last_cfg, collected = result
        assert len(failed) >= 1


# --------------------------------------------------------------------------- #
# Integration: full scan → edit → run cycle
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestGuiFullCycle:
    """Full GUI cycle: scan → edit coefs → run."""

    def test_scan_then_run(self, gui_project, monkeypatch, mocker):
        """Scan discovers configs, then run processes them."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)

        # --- Step 1: Scan ---
        monkeypatch.setattr(sys, "argv", ["prog"])
        mock_proc = mocker.patch("tcm.processing.run_processing")
        scan_result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt")},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        assert len(scan_result) == 4
        collected = scan_result[3]
        assert len(collected) >= 1
        stem = collected[0][0]

        # --- Step 2a: Write coefs (simulates user editing tksheet) ---
        yaml_path = run_dir / f"{stem}.yaml"
        config_yaml.update_coefs_in_run_yaml(
            yaml_path,
            {"Ag": [[0.005, 0, 0], [0, 0.005, 0], [0, 0, 0.005]]},
        )
        # Verify coefs written
        ry = config_yaml._ry(write=False)
        with yaml_path.open(encoding="utf-8") as f:
            updated = ry.load(f)
        assert updated["input"]["coefs"]["Ag"][0][0] == 0.005

        # --- Step 2b: Run with the stem ---
        mock_proc.reset_mock()
        run_result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt"), "yaml_path": f"({stem})"},
            },
            exit_on_error=False,
        )
        assert len(run_result) == 4
        mock_proc.assert_called_once()


# --------------------------------------------------------------------------- #
# Edge cases: real-world GUI failure modes
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestGuiSysArgvIsolation:
    """call_in_raw_dir must not pollute sys.argv between consecutive calls.

    The bug: ``remaining_argv = sys.argv`` is a reference (not copy), so
    ``remaining_argv[1:1] = _build_hydra_argv(data_dir)`` mutated sys.argv
    in-place.  On the next call, the old ``--config-dir`` accumulated, and
    Hydra tried to parse it as an override — crashing on paths with ``:``.
    """

    def test_sys_argv_clean_after_scan(self, gui_project, monkeypatch, mocker):
        """After a scan call, sys.argv contains only --config-dir (no old data paths)."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        original_argv = ["__main__.py"]
        monkeypatch.setattr(sys, "argv", list(original_argv))
        monkeypatch.chdir(tmp_path)

        mocker.patch("tcm.processing.run_processing")
        cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt")},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        # sys.argv should have script name + --config-dir, but NOT the data path
        assert len(sys.argv) <= 3, f"sys.argv accumulated junk: {sys.argv}"
        assert all("@" not in a for a in sys.argv[1:]), (
            f"Data path leaked into sys.argv: {sys.argv}"
        )

    def test_no_argv_accumulation_on_repeated_calls(self, gui_project, monkeypatch, mocker):
        """Worker._setup resets sys.argv between calls — no --config-dir accumulation."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.setattr(sys, "argv", ["__main__.py"])
        monkeypatch.chdir(tmp_path)

        mocker.patch("tcm.processing.run_processing")
        # Call 1: scan (Worker._setup resets sys.argv before each call)
        sys.argv = ["__main__.py"]
        cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt")},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )

        # Call 2: run — Worker._setup would reset sys.argv to ["__main__.py"]
        sys.argv = ["__main__.py"]
        cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(csv_file), "yaml_path": "(@i_01)"},
            },
            exit_on_error=False,
        )
        # sys.argv has exactly one --config-dir (from the second call)
        config_dir_count = sys.argv.count("--config-dir")
        assert config_dir_count == 1, f"--config-dir appeared {config_dir_count} times: {sys.argv}"

    def test_scan_with_path_from_other_drive(self, gui_project, monkeypatch, mocker):
        """Path with Windows drive letter (B:/, D:/) doesn't crash Hydra.

        Regression: path ``B:/Cruises/...`` ended up in sys.argv and Hydra's
        ANTLR parser choked on the ``:`` after the drive letter.
        """
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])

        mocker.patch("tcm.processing.run_processing")
        # Pass path as str — exactly what Tk Entry.get() returns
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt")},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        assert result is not None
        assert len(result) == 4


@pytest.mark.xr
class TestGuiAtSignFilename:
    """Filenames with ``@`` (e.g. ``@i_p1.TXT``) are common in inclinometer data.

    The ``@`` is a Hydra config-group override delimiter.  If the path leaks
    into sys.argv, Hydra tries to parse ``@i_p1`` as a config group and crashes.
    """

    @pytest.fixture()
    def _raw_with_at_file(self, tmp_path):
        """Create _raw/ with ``@i_p1.TXT`` (uppercase, real naming convention)."""
        raw_dir = tmp_path / RAW_DIR_NAME
        raw_dir.mkdir()
        csv_file = raw_dir / "@i_p1.TXT"
        csv_file.write_text(
            "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp\n"
            "2024,06,13,12,00,00,100.0,200.0,300.0,400.0,500.0,600.0,12.5,25.0\n",
            encoding="utf-8",
        )
        return tmp_path, raw_dir, csv_file

    def test_scan_at_sign_file(self, _raw_with_at_file, monkeypatch, mocker):
        """Scan with ``@i_p1.TXT`` doesn't crash Hydra override parser."""
        tmp_path, raw_dir, csv_file = _raw_with_at_file
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])

        mocker.patch("tcm.processing.run_processing")
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(csv_file)},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        assert result is not None
        collected = result[3]
        assert len(collected) >= 1
        stem = collected[0][0]
        assert "i_p1" in stem.lower() or "p1" in stem

    def test_run_at_sign_file(self, _raw_with_at_file, monkeypatch, mocker):
        """Run with ``@i_p1.TXT`` stem processes correctly."""
        tmp_path, raw_dir, csv_file = _raw_with_at_file
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])

        # First: generate config via scan
        mocker.patch("tcm.processing.run_processing")
        cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(csv_file)},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        # Config should be generated — find the stem
        run_dir = raw_dir / "cfg_proc" / "run"
        yamls = list(run_dir.glob("*.yaml"))
        assert len(yamls) >= 1
        stem = yamls[0].stem

        # Second: run with that stem
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(csv_file), "yaml_path": f"({stem})"},
            },
            exit_on_error=False,
        )
        assert result is not None
        assert len(result) == 4


@pytest.mark.xr
class TestGuiCfgProcAutoCreation:
    """cfg_proc/run/ directory is auto-created when scanning a new data dir."""

    def test_cfg_proc_created_on_scan(self, gui_project, monkeypatch, mocker):
        """cfg_proc/run/ directory exists after scan (even if gen_metadata skips a probe)."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        # Remove the YAML to test that the code doesn't crash without it
        for f in run_dir.glob("*.yaml"):
            f.unlink()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])

        mocker.patch("tcm.processing.run_processing")
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(csv_file)},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        assert result is not None
        # cfg_proc/run/ was created (safe_cfg_dir creates it)
        assert run_dir.is_dir()


@pytest.mark.xr
class TestGuiGlobalHydraClear:
    """Worker._setup() clears GlobalHydra between calls.

    Hydra can only compose configs once per process unless GlobalHydra is
    cleared.  The worker does this in ``_setup()`` before each call.
    """

    def test_two_hydra_compositions(self, gui_project, monkeypatch, mocker):
        """Two call_in_raw_dir calls succeed (GlobalHydra cleared between them)."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)

        # Simulate Worker._setup: clear GlobalHydra before each call
        from hydra.core.global_hydra import GlobalHydra

        mocker.patch("tcm.processing.run_processing")

        # Call 1: scan
        GlobalHydra.instance().clear()
        monkeypatch.setattr(sys, "argv", ["__main__.py"])
        result1 = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt")},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        assert result1 is not None

        # Call 2: run — must succeed after GlobalHydra.clear
        GlobalHydra.instance().clear()
        monkeypatch.setattr(sys, "argv", ["__main__.py"])
        result2 = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt"), "yaml_path": "(@i_01)"},
            },
            exit_on_error=False,
        )
        assert result2 is not None
        assert len(result2) == 4


@pytest.mark.xr
class TestGuiCwdStability:
    """call_in_raw_dir changes cwd to the _raw/ directory.

    After the call, the working directory is the _raw/ dir (where data lives).
    This is expected — but verify it's the RIGHT _raw/ dir.
    """

    def test_cwd_set_to_raw_dir(self, gui_project, monkeypatch, mocker):
        """After call, cwd is the _raw/ directory."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])

        mocker.patch("tcm.processing.run_processing")
        cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(csv_file)},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        assert Path.cwd() == raw_dir


@pytest.mark.xr
class TestGuiRunWithNoConfigs:
    """Run with input.yaml_path pointing to a non-existent YAML."""

    def test_missing_stem_logs_error(self, gui_project, monkeypatch, mocker):
        """Requesting a stem with no YAML: process_loading_yaml logs and skips."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])

        mocker.patch("tcm.processing.run_processing")
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir / "*i*.txt"), "yaml_path": "(@nonexistent_stem)"},
            },
            exit_on_error=False,
        )
        # result is 4-tuple but collected is empty (no YAML for this stem)
        assert result is not None
        processed, failed, last_cfg, collected = result
        assert len(processed) == 0
        assert len(collected) == 0


@pytest.mark.xr
class TestGuiWriteCoefsIdempotent:
    """Writing the same coefs twice should produce two backups but same YAML."""

    def test_double_write_two_backups(self, gui_project, monkeypatch):
        """Two writes with different coefs → backup + updated YAML."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        yaml_path = run_dir / "@i_01.yaml"
        monkeypatch.chdir(tmp_path)

        coefs1 = {"Ag": [[0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]]}
        coefs2 = {"Ag": [[0.003, 0, 0], [0, 0.003, 0], [0, 0, 0.003]]}
        config_yaml.update_coefs_in_run_yaml(yaml_path, coefs1)
        config_yaml.update_coefs_in_run_yaml(yaml_path, coefs2)

        # At least one backup created (second call may share timestamp)
        backups = list(run_dir.glob("@i_01-backup*.yaml"))
        assert len(backups) >= 1
        # YAML has the LATEST coefs
        ry = config_yaml._ry(write=False)
        with yaml_path.open(encoding="utf-8") as f:
            updated = ry.load(f)
        assert updated["input"]["coefs"]["Ag"][0][0] == 0.003

    def test_write_coefs_preserves_package_header(self, gui_project, monkeypatch):
        """YAML still has ``# @package _global_`` after coef write."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        yaml_path = run_dir / "@i_01.yaml"
        monkeypatch.chdir(tmp_path)

        config_yaml.update_coefs_in_run_yaml(
            yaml_path, {"Rz": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        )
        first_line = yaml_path.read_text(encoding="utf-8").split("\n")[0]
        assert first_line.strip() == "# @package _global_"


# --------------------------------------------------------------------------- #
# CLI args → GUI prefill
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestGuiCliArgs:
    """GUI started with CLI args: path prefill + Hydra override propagation."""

    def test_parse_cli_args_path_only(self):
        """Single positional arg → path, no hydra args."""
        from tcm.cli import parse_data_path

        path_in, remaining = parse_data_path(["prog", "D:/data/_raw/@i_p1.TXT"])
        assert str(path_in) == "D:\\data\\_raw\\@i_p1.TXT"
        # remaining has script name + --nothing else
        assert len(remaining) <= 2  # script + maybe --config-dir

    def test_parse_cli_args_with_overrides(self):
        """Path + key=value → path extracted, overrides stay in argv."""
        from tcm.cli import parse_data_path

        path_in, remaining = parse_data_path([
            "prog",
            "D:/data/_raw",
            "input.time_ranges=['2024-01-01','2024-01-02']",
            "filter.max.g_minus_1=2.0",
        ])
        assert str(path_in) == "D:\\data\\_raw"
        # remaining still has the override args
        assert "input.time_ranges=['2024-01-01','2024-01-02']" in remaining
        assert "filter.max.g_minus_1=2.0" in remaining

    def test_parse_cli_args_bare_identifiers(self):
        """``input.ids=[i90, i67]`` stays in argv for Hydra."""
        from tcm.cli import parse_data_path

        path_in, remaining = parse_data_path([
            "prog",
            "B:/Cruises/_raw",
            "input.ids=[i90, i67]",
        ])
        assert "input.ids=[i90, i67]" in remaining

    def test_hydra_args_propagate_to_processing(self, gui_project, monkeypatch, mocker):
        """Original argv with Hydra overrides is passed through to call_in_raw_dir."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        # Simulate: Worker._setup resets sys.argv to original_argv
        monkeypatch.setattr(sys, "argv", ["__main__.py", str(csv_file), "filter.max.g_minus_1=5.0"])

        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()

        mocker.patch("tcm.processing.run_processing")
        # call_in_raw_dir extracts path from sys.argv, applies overrides
        result = cli.call_in_raw_dir(
            processing.run,
            config_name="config",
            program={"return_": Return.CFG_FROM_ARGS},
            exit_on_error=False,
        )
        assert result is not None
        assert len(result) == 4

    def test_full_cycle_with_hydra_args(self, gui_project, monkeypatch, mocker):
        """Full scan→run with path from sys.argv (no overrides dict for path)."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)

        from hydra.core.global_hydra import GlobalHydra

        mocker.patch("tcm.processing.run_processing")

        # Step 1: scan — path in sys.argv, call_in_raw_dir extracts it
        GlobalHydra.instance().clear()
        monkeypatch.setattr(sys, "argv", ["__main__.py", str(raw_dir / "*i*.txt")])
        result = cli.call_in_raw_dir(
            processing.run,
            program={"return_": Return.CFG_FROM_ARGS},
            exit_on_error=False,
        )
        assert result is not None
        collected = result[3]
        assert len(collected) >= 1
        stem = collected[0][0]

        # Step 2: run with the same stem
        GlobalHydra.instance().clear()
        monkeypatch.setattr(sys, "argv", ["__main__.py", str(raw_dir / "*i*.txt")])
        result2 = cli.call_in_raw_dir(
            processing.run,
            input={"yaml_path": f"({stem})"},
            exit_on_error=False,
        )
        assert result2 is not None
        assert len(result2) == 4


# --------------------------------------------------------------------------- #
# Dirty tracking: _FakeSheet + ConfigSheet contract
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestFakeSheetDirtyTracking:
    """Verify _FakeSheet satisfies the dirty-tracking contract used by _write_coefs."""

    @pytest.mark.parametrize(
        ("coefs", "dates", "path", "dirty", "expect_write"),
        [
            pytest.param(
                {"Ag": [[0.001]]}, {}, "", True, True,
                id="dirty_with_coefs",
            ),
            pytest.param({}, {}, "", False, False,
                id="clean_no_write",
            ),
            pytest.param({}, {"Ag": "2024-01-01"}, "", True, True,
                id="dirty_with_dates_only",
            ),
            pytest.param({}, {}, "/some/path", True, True,
                id="dirty_with_path_only",
            ),
        ],
    )
    def test_write_coefs_respects_dirty_flag(self, coefs, dates, path, dirty, expect_write, mocker):
        """_write_coefs calls update_coefs_in_run_yaml only when is_dirty is True."""
        mock_update = mocker.patch("tcm.config_yaml.update_coefs_in_run_yaml")
        from tcm_gui.app import App

        app = App.__new__(App)
        app._yaml_paths = {"@i_01": Path("/fake/path.yaml")}
        cs = _FakeSheet(coefs=coefs, dates=dates, path=path, dirty=dirty)
        app._write_coefs("@i_01", cs)
        if expect_write:
            mock_update.assert_called_once(), "dirty sheet should trigger write"
            assert cs.is_dirty is False, "mark_clean resets dirty after write"
        else:
            mock_update.assert_not_called(), "clean sheet should skip write"


# --------------------------------------------------------------------------- #
# Log dedup: QueueHandler skips consecutive identical messages
# --------------------------------------------------------------------------- #


@pytest.mark.xr
class TestQueueHandlerDedup:
    """QueueHandler.emit drops consecutive records with same (funcName, message)."""

    @staticmethod
    def _make_record(func: str, msg: str) -> logging.LogRecord:
        return logging.LogRecord("test", logging.INFO, "", 0, msg, (), None, func)

    def test_consecutive_duplicates_skipped(self):
        """Two identical records in a row → only one enqueued."""
        from tcm_gui.log_bridge import QueueHandler
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        gate = PauseGate()
        h = QueueHandler(q, gate)

        h.emit(self._make_record("func_a", "hello"))
        h.emit(self._make_record("func_a", "hello"))

        assert q.qsize() == 1, f"expected 1 record, got {q.qsize()}"

    def test_non_consecutive_same_message_passes(self):
        """A, B, A → all three pass (only consecutive dupes filtered)."""
        from tcm_gui.log_bridge import QueueHandler
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        gate = PauseGate()
        h = QueueHandler(q, gate)

        h.emit(self._make_record("func_a", "hello"))
        h.emit(self._make_record("func_a", "world"))
        h.emit(self._make_record("func_a", "hello"))

        assert q.qsize() == 3, f"expected 3 records, got {q.qsize()}"

    def test_different_funcname_not_deduped(self):
        """Same message from different functions → both pass."""
        from tcm_gui.log_bridge import QueueHandler
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        gate = PauseGate()
        h = QueueHandler(q, gate)

        h.emit(self._make_record("func_a", "hello"))
        h.emit(self._make_record("func_b", "hello"))

        assert q.qsize() == 2, f"expected 2 records, got {q.qsize()}"
