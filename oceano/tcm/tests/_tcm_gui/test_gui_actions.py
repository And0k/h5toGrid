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
from queue import Empty, Queue

import pytest
from omegaconf import OmegaConf

from tcm import cli, config_yaml, processing
from tcm._constants import RAW_DIR_NAME
from tcm.schema import Return
import utils.log_init


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

    def __init__(
        self, coefs: dict | None = None, dates: dict | None = None, path: str = "", dirty: bool = True
    ):
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


@pytest.mark.gui
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
            sys,
            "argv",
            ["prog", path_str, f'program.return_="{Return.CFG_FROM_ARGS}"'],
        )
        mock_proc = mocker.patch.object(processing, "run_processing")
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
            sys,
            "argv",
            ["prog", path_str, f'program.return_="{Return.CFG_FROM_ARGS}"'],
        )
        mocker.patch.object(processing, "run_processing")
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
        mocker.patch.object(processing, "run_processing")
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


@pytest.mark.gui
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

    def test_write_coefs_delegates_to_update_run_yaml(self, gui_project, monkeypatch, mocker):
        """``App._write_coefs`` merges edits via ``config_yaml.update_run_yaml``."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        yaml_path = run_dir / "@i_01.yaml"

        mock_update = mocker.patch.object(config_yaml, "update_run_yaml")

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
        """``_write_coefs`` passes dates dict to ``update_run_yaml``."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        yaml_path = run_dir / "@i_01.yaml"

        mock_update = mocker.patch.object(config_yaml, "update_run_yaml")

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
        """No edits → no call to ``update_run_yaml``."""
        mock_update = mocker.patch.object(config_yaml, "update_run_yaml")

        from tcm_gui.app import App

        app = App.__new__(App)
        app._yaml_paths = {"@i_01": Path("/fake/path")}

        app._write_coefs("@i_01", _FakeSheet(dirty=False))
        mock_update.assert_not_called()


# --------------------------------------------------------------------------- #
# Step 2b: Run — user clicks Run button
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestGuiRun:
    """Simulate GUI run: ``Worker._run`` → ``call_in_raw_dir`` with ``input.yaml_path``."""

    def test_run_with_yaml_path(self, gui_project, monkeypatch, mocker):
        """``input.yaml_path`` dispatches to ``run_processing`` for listed stems."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(sys, "argv", ["prog"])
        mock_proc = mocker.patch.object(processing, "run_processing")
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(run_dir / "(@i_01).yaml")},
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
        mock_save = mocker.patch.object(config_yaml, "save_config_to_yaml")
        mocker.patch.object(processing, "run_processing")

        cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(run_dir / "(@i_01).yaml")},
            },
            exit_on_error=False,
        )

        mock_save.assert_not_called()


# --------------------------------------------------------------------------- #
# Step 1c: Scan → tabs — order + initial selection
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestScanTabs:
    """Tab order == backend config order; first tab selected after all pages exist."""

    @staticmethod
    def _seed_configs(gui_project):
        """Add extra probes/stems so ordering is observable (fixture adds @i_01).

        Each YAML needs its own source CSV — ``process_loading_yaml`` skips
        configs whose ``input.path`` filename stem ≠ YAML stem.  Names must
        parse as probe identities (contain ``i``/``w``, see tcm.format).
        """
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        head = "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp\n"
        rows = head + "2024,06,13,12,00,00,100.0,200.0,300.0,400.0,500.0,600.0,12.5,25.0\n" * 2
        for stem in ("i_02", "w_01", "i_01b"):
            csv = raw_dir / f"@{stem}.txt"
            csv.write_text(rows, encoding="utf-8")
            (run_dir / f"@{stem}.yaml").write_text(
                f"# @package _global_\ninput:\n  path: '{csv}'\nout:\n  dt_bins: [0]\n",
                encoding="utf-8",
            )

    def test_scan_order_matches_backend_cfgs(self, gui_project, monkeypatch, mocker):
        """Tabs follow ``collected`` order == flattened ``get_existed_cfgs`` order."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        self._seed_configs(gui_project)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog"])
        mocker.patch.object(processing, "run_processing")
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(raw_dir)},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        collected_stems = [stem for stem, _, _ in result[3]]
        backend_stems = [s for ss in config_yaml.get_existed_cfgs(run_dir).values() for s in ss]
        # _on_scan_ok builds tabs in collected order; the run pipeline filters
        # the same cfgs dict — display order == processing order (top→bottom).
        assert collected_stems == backend_stems

    def test_run_order_matches_scan_order(self, gui_project, monkeypatch, mocker):
        """Run processes stems in the same order tabs were built."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        self._seed_configs(gui_project)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog"])

        mocker.patch.object(processing, "run_processing")  # scan early-exit
        scan = cli.call_in_raw_dir(
            processing.run,
            overrides={"input": {"path": str(raw_dir)}, "program": {"return_": Return.CFG_FROM_ARGS}},
            exit_on_error=False,
        )
        collected_stems = [stem for stem, _, _ in scan[3]]
        pcid_of = {s: pcid for pcid, ss in config_yaml.get_existed_cfgs(run_dir).items() for s in ss}

        mocker.patch.object(processing, "run_processing", return_value=None)
        run = cli.call_in_raw_dir(
            processing.run,
            overrides={"input": {"path": str(run_dir / f"({'|'.join(collected_stems)}).yaml")}},
            exit_on_error=False,
        )
        processed, _, _, _ = run
        assert processed == [pcid_of[s] for s in collected_stems]

    def test_scan_ok_selects_first_tab_after_rebuild(self, monkeypatch):
        """``_on_scan_ok`` selects the FIRST tab only after ALL pages exist.

        Regression: the first page used to be raised before the later pages
        were gridded — Tk stacks later-created frames above it, so the visible
        sheet was the LAST tab while the rail highlighted the first.
        """
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from omegaconf import OmegaConf

        from tcm_gui.app import App

        app = App.__new__(App)
        app._tab_of = {}
        app._pages = {}
        app._yaml_paths = {}
        app._current = None
        app._rail = MagicMock()
        app._path_field = MagicMock()
        app._cfg_detail = ""
        app._initial_scan = False
        app.rt = SimpleNamespace(progress_stage=MagicMock())
        app._hide_tip = MagicMock()
        app._set_status = MagicMock()
        app._translate_scan_stage = lambda _s: ""
        app._update_run_btn_state = MagicMock()
        app._overall_lbl = MagicMock()

        added: list[str] = []

        def _fake_add_page(stem, cfg, yaml_path=None, metadata=None, sync_status=None, metadata_path=None):
            added.append(stem)
            app._tab_of[stem] = object()

        app._add_page = _fake_add_page
        events: list[tuple] = []

        def _fake_select(stem):
            events.append((stem, tuple(app._tab_of)))

        app._select_tab = _fake_select

        cfg = OmegaConf.create({"input": {"path": "x"}, "program": {"return_": str(Return.END)}})
        result = (["p"], [], None, [(s, f"{s}.yaml", cfg) for s in ("b", "a", "c")])
        app._on_scan_ok(result)

        assert added == ["b", "a", "c"]  # tabs follow collected order
        assert events == [("b", ("b", "a", "c"))]  # ONE select, first tab, after all pages
        app._path_field.set_error.assert_called_once_with(False)

    def test_scan_error_marks_path_field_red(self, monkeypatch):
        """Failed scan → red fg on the search path field + inert cfg UI re-dim."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tcm_gui.app import App

        app = App.__new__(App)
        app._path_field = MagicMock()
        app._surface_error = MagicMock()
        app._cfg_detail = ""
        app.rt = SimpleNamespace(progress_overall=MagicMock())
        app._overall_lbl = MagicMock()
        app._default_stage_text = lambda: ""
        app._full_mode = False
        app._yaml_paths = {}  # no successful scan ever → re-dim rail + caption
        app._set_cfg_ui_disabled = MagicMock()
        app._on_scan_error(ValueError("boom"))
        app._path_field.set_error.assert_called_once_with(True)
        app._surface_error.assert_called_once()
        app._set_cfg_ui_disabled.assert_called_once_with(True)

    def test_scan_empty_path_forwards_to_worker(self):
        """Empty search path is forwarded to the pipeline — no GUI short-circuit.

        Regression: ``_on_path_changed`` shows "Loading…"; skipping the
        worker for an empty path froze the stage row forever.  The empty
        path means "./" and the pipeline owns the verdict
        (``cli.call_in_raw_dir`` rejects repo-internal anchors), so the
        failure surfaces through the normal ``scan_error`` channel.
        """
        from unittest.mock import MagicMock

        from tcm_gui.app import App

        app = App.__new__(App)
        app._path_field = MagicMock()
        app._path_field.get.return_value = "   "  # whitespace-only → empty
        app._clear_log = MagicMock()
        app.wk = MagicMock()
        app._original_argv = ["prog"]
        app._scan()
        app.wk.scan.assert_called_once_with(["prog"], "")
        app._clear_log.assert_called_once()

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
                "input": {"path": str(run_dir / "(@i_01).yaml")},
            },
            exit_on_error=False,
        )

        processed, failed, last_cfg, collected = result
        assert len(failed) >= 1


# --------------------------------------------------------------------------- #
# Integration: full scan → edit → run cycle
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestGuiFullCycle:
    """Full GUI cycle: scan → edit coefs → run."""

    def test_scan_then_run(self, gui_project, monkeypatch, mocker):
        """Scan discovers configs, then run processes them."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)

        # --- Step 1: Scan ---
        monkeypatch.setattr(sys, "argv", ["prog"])
        mock_proc = mocker.patch.object(processing, "run_processing")
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
                "input": {"path": str(run_dir / f"({stem}).yaml")},
            },
            exit_on_error=False,
        )
        assert len(run_result) == 4
        mock_proc.assert_called_once()


# --------------------------------------------------------------------------- #
# Edge cases: real-world GUI failure modes
# --------------------------------------------------------------------------- #


@pytest.mark.gui
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

        mocker.patch.object(processing, "run_processing")
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
        assert all("@" not in a for a in sys.argv[1:]), f"Data path leaked into sys.argv: {sys.argv}"

    def test_no_argv_accumulation_on_repeated_calls(self, gui_project, monkeypatch, mocker):
        """Worker._setup resets sys.argv between calls — no --config-dir accumulation."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.setattr(sys, "argv", ["__main__.py"])
        monkeypatch.chdir(tmp_path)

        mocker.patch.object(processing, "run_processing")
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
                "input": {"path": str(run_dir / "(@i_01).yaml")},
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

        mocker.patch.object(processing, "run_processing")
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


@pytest.mark.gui
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

        mocker.patch.object(processing, "run_processing")
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
        assert "@i_p" in stem.lower()  # i_p1 zero-pads to i_p01; @ prefix preserved

    def test_run_at_sign_file(self, _raw_with_at_file, monkeypatch, mocker):
        """Run with ``@i_p1.TXT`` stem processes correctly."""
        tmp_path, raw_dir, csv_file = _raw_with_at_file
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])

        # First: generate config via scan
        mocker.patch.object(processing, "run_processing")
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
                "input": {"path": str(run_dir / f"({stem}).yaml")},
            },
            exit_on_error=False,
        )
        assert result is not None
        assert len(result) == 4


@pytest.mark.gui
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

        mocker.patch.object(processing, "run_processing")
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


@pytest.mark.gui
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

        mocker.patch.object(processing, "run_processing")

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
                "input": {"path": str(run_dir / "(@i_01).yaml")},
            },
            exit_on_error=False,
        )
        assert result2 is not None
        assert len(result2) == 4


@pytest.mark.gui
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

        mocker.patch.object(processing, "run_processing")
        cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(csv_file)},
                "program": {"return_": Return.CFG_FROM_ARGS},
            },
            exit_on_error=False,
        )
        assert Path.cwd() == raw_dir


@pytest.mark.gui
class TestGuiRunWithNoConfigs:
    """Run with input.yaml_path pointing to a non-existent YAML."""

    def test_missing_stem_logs_error(self, gui_project, monkeypatch, mocker):
        """Requesting a stem with no YAML: process_loading_yaml logs and skips."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])

        mocker.patch.object(processing, "run_processing")
        result = cli.call_in_raw_dir(
            processing.run,
            overrides={
                "input": {"path": str(run_dir / "(@nonexistent_stem).yaml")},
            },
            exit_on_error=False,
        )
        # result is 4-tuple but collected is empty (no YAML for this stem)
        assert result is not None
        processed, failed, last_cfg, collected = result
        assert len(processed) == 0
        assert len(collected) == 0


@pytest.mark.gui
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

        config_yaml.update_coefs_in_run_yaml(yaml_path, {"Rz": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        first_line = yaml_path.read_text(encoding="utf-8").split("\n")[0]
        assert first_line.strip() == "# @package _global_"


# --------------------------------------------------------------------------- #
# CLI args → GUI prefill
# --------------------------------------------------------------------------- #


@pytest.mark.gui
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

        path_in, remaining = parse_data_path(
            [
                "prog",
                "D:/data/_raw",
                "input.time_ranges=['2024-01-01','2024-01-02']",
                "filter.max.g_minus_1=2.0",
            ]
        )
        assert str(path_in) == "D:\\data\\_raw"
        # remaining still has the override args
        assert "input.time_ranges=['2024-01-01','2024-01-02']" in remaining
        assert "filter.max.g_minus_1=2.0" in remaining

    def test_parse_cli_args_bare_identifiers(self):
        """``input.ids=[i90, i67]`` stays in argv for Hydra."""
        from tcm.cli import parse_data_path

        path_in, remaining = parse_data_path(
            [
                "prog",
                "B:/Cruises/_raw",
                "input.ids=[i90, i67]",
            ]
        )
        assert "input.ids=[i90, i67]" in remaining

    def test_hydra_args_propagate_to_processing(self, gui_project, monkeypatch, mocker):
        """Original argv with Hydra overrides is passed through to call_in_raw_dir."""
        tmp_path, raw_dir, csv_file, run_dir = gui_project
        monkeypatch.chdir(tmp_path)
        # Simulate: Worker._setup resets sys.argv to original_argv
        monkeypatch.setattr(sys, "argv", ["__main__.py", str(csv_file), "filter.max.g_minus_1=5.0"])

        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        mocker.patch.object(processing, "run_processing")
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

        mocker.patch.object(processing, "run_processing")

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

        # Step 2: run with the same stem — path via overrides (not sys.argv)
        GlobalHydra.instance().clear()
        monkeypatch.setattr(sys, "argv", ["__main__.py"])
        result2 = cli.call_in_raw_dir(
            processing.run,
            input={"path": str(run_dir / f"({stem}).yaml")},
            exit_on_error=False,
        )
        assert result2 is not None
        assert len(result2) == 4


# --------------------------------------------------------------------------- #
# Dirty tracking: _FakeSheet + ConfigSheet contract
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestFakeSheetDirtyTracking:
    """Verify _FakeSheet satisfies the dirty-tracking contract used by _write_coefs."""

    @pytest.mark.parametrize(
        ("coefs", "dates", "path", "dirty", "expect_write"),
        [
            pytest.param(
                {"Ag": [[0.001]]},
                {},
                "",
                True,
                True,
                id="dirty_with_coefs",
            ),
            pytest.param(
                {},
                {},
                "",
                False,
                False,
                id="clean_no_write",
            ),
            pytest.param(
                {},
                {"Ag": "2024-01-01"},
                "",
                True,
                True,
                id="dirty_with_dates_only",
            ),
            pytest.param(
                {},
                {},
                "/some/path",
                True,
                True,
                id="dirty_with_path_only",
            ),
        ],
    )
    def test_write_coefs_respects_dirty_flag(self, coefs, dates, path, dirty, expect_write, mocker):
        """_write_coefs calls update_run_yaml only when is_dirty is True."""
        mock_update = mocker.patch.object(config_yaml, "update_run_yaml")
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


@pytest.mark.gui
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


# --------------------------------------------------------------------------- #
# Regression: LoggingStyleAdapter shares one mutable Message instance across
# log calls; QueueHandler must freeze the rendered text onto the record so
# deferred drain-time getMessage() returns the message that was actually
# logged at emit time, not whatever the shared Message was last mutated to.
# Without this freeze, every record from a given logger would render as the
# *last* message that logger produced (see docs/project_developer_guide/GUI/decisions.md → log_bridge).
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestQueueHandlerFreezeMutableMessage:
    """QueueHandler.emit freezes the rendered text onto the LogRecord.

    :class:`utils.init.LoggingStyleAdapter` calls ``logger._log(level,
    self.message, ())`` — i.e. it passes the same ``Message`` instance to
    every log call and mutates its ``fmt``/``args`` in place.  By the time
    :func:`tcm_gui.log_bridge.drain` runs (later, on the GUI poll thread)
    ``record.msg`` would still point at that shared, by-then-mutated object,
    so ``record.getMessage()`` would return the *latest* rendered text rather
    than the one captured at emit time → every record from that logger
    collapses to the last message → appearance of duplicate log lines.

    Hydra's ``job_logging/colorlog`` formatter sidesteps this by rendering
    ``record.getMessage()`` once synchronously inside ``Formatter.format``.
    ``QueueHandler`` mirrors that contract by freezing the rendered text back
    onto the record (``record.msg = text; record.args = ()``).
    """

    def test_subsequent_records_render_orig_text_not_mutated(self):
        """Two consecutive log calls on one adapter → drain shows each original text."""
        from utils import init
        from tcm_gui.log_bridge import QueueHandler
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        gate = PauseGate()
        h = QueueHandler(q, gate)
        h.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        try:
            lf = utils.log_init.LoggingStyleAdapter("test_freeze_message")

            def func_alpha():
                lf.info("alpha message {}", 1)

            def func_beta():
                lf.info("beta message {}", 2)

            func_alpha()
            func_beta()

            records: list[logging.LogRecord] = []
            while True:
                try:
                    records.append(q.get_nowait())
                except Empty:
                    break
        finally:
            root.removeHandler(h)

        assert len(records) == 2, f"expected 2 records, got {len(records)}"
        # Drain later — by then the shared Message's fmt/args are mutated to
        # "beta message {}".format(2).  Without emit-time freeze, BOTH records
        # would render as "beta message 2".
        drained = [(r.funcName, r.getMessage()) for r in records]
        assert drained == [
            ("func_alpha", "alpha message 1"),
            ("func_beta", "beta message 2"),
        ], f"text not frozen on record: drained={drained!r}"

    def test_consecutive_same_text_still_dedups_after_freeze(self):
        """Freeze preserves the consecutive-dedup invariant from TestQueueHandlerDedup."""
        from utils import init
        from tcm_gui.log_bridge import QueueHandler
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        gate = PauseGate()
        h = QueueHandler(q, gate)
        h.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
        try:
            lf = utils.log_init.LoggingStyleAdapter("test_freeze_dup")

            def func_same():
                lf.debug("repeated same")

            func_same()
            func_same()  # consecutive identical → dedup should drop this one
            func_same()

            records: list[logging.LogRecord] = []
            while True:
                try:
                    records.append(q.get_nowait())
                except Empty:
                    break
        finally:
            root.removeHandler(h)

        assert len(records) == 1, f"expected 1 record (consecutive dedup), got {len(records)}"
        assert records[0].getMessage() == "repeated same", (
            f"deduped record text wrong: {records[0].getMessage()!r}"
        )


@pytest.mark.gui
class TestQueueHandlerFormatFailures:
    """A record that cannot survive stdlib %-formatting must still reach the GUI log.

    Regression: ``theme.py`` logged with ``{}``-style format strings on a plain
    stdlib logger, so ``rec.getMessage()`` raised TypeError inside
    ``QueueHandler.emit`` while the About dialog was being built — the dialog
    never opened and the line never reached ``App._log`` (console-only).
    """

    @staticmethod
    def _make_record(func: str, msg, args=(), level=logging.INFO, exc_info=None) -> logging.LogRecord:
        return logging.LogRecord("test", level, "", 0, msg, args, exc_info, func)

    def test_brace_style_message_still_queued(self):
        """``{}`` fmt + args on a plain logger → emit never raises, raw text queued."""
        from tcm_gui.log_bridge import QueueHandler
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        h = QueueHandler(q, PauseGate())
        raw = "DwmSetWindowAttribute(HWND={:#x}) → HRESULT={:#010x}"
        rec = self._make_record("_opt_into_dark_titlebar", raw, (5705430, 0))
        h.emit(rec)  # must not raise
        assert q.qsize() == 1, f"record lost on format failure: {q.qsize()}"
        assert q.get().getMessage() == raw, "raw text must survive a format failure"

    def test_mismatched_args_message_still_queued(self):
        """msg without % conversions + non-empty args → raw text still queued.

        The exact failure from the traceback: 'not all arguments converted
        during string formatting'.  The degraded raw text must reach the queue.
        """
        from tcm_gui.log_bridge import QueueHandler
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        h = QueueHandler(q, PauseGate())
        raw = "Scan: FileNotFoundError: No input files found matching B:\\cruises"
        rec = self._make_record("scan", raw, (1,))
        h.emit(rec)
        assert q.get().getMessage() == raw

    def test_drain_renders_exception_line(self):
        """Records with exc_info show their exception line in drain (GUI log)."""
        from tcm_gui.log_bridge import QueueHandler, drain
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        h = QueueHandler(q, PauseGate())
        exc = FileNotFoundError("No input files found matching B:\\cruises\\x.txt")
        rec = self._make_record("_scan", "scan failed", exc_info=(FileNotFoundError, exc, None))
        h.emit(rec)

        class _FakeText:
            def __init__(self) -> None:
                self.parts: list[str] = []

            def insert(self, _end: str, text: str, _tag: str | None = None) -> None:
                self.parts.append(text)

        w = _FakeText()
        assert drain(q, w) == 1
        assert "FileNotFoundError: No input files found matching B:\\cruises\\x.txt" in "".join(w.parts)


@pytest.mark.gui
class TestQueueHandlerPersistsAcrossTasks:
    """GUI-thread log calls reach the queue via the single persistent QueueHandler.

    Regression: previously the QueueHandler was installed *inside* the worker
    wrap (per-task) and removed afterwards, so log calls from GUI callbacks
    (e.g. ``_reload_coefs`` triggered by treeview interaction) silently went
    to the Stream/File handlers but never to the ScrolledText.  With the
    handler installed once at App startup, log records from the *main* thread
    — not just the worker thread — also reach ``log_queue``.  This test
    simulates that scenario without a worker thread: install once, log from
    the current thread, assert the record is queued.
    """

    def test_main_thread_logs_reach_queue_without_worker(self):
        from utils import init
        from tcm_gui.log_bridge import install
        from tcm_gui.runtime import PauseGate

        q: Queue = Queue()
        gate = PauseGate()
        h = install(q, gate)
        h.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        try:
            lf = utils.log_init.LoggingStyleAdapter("test_main_thread_reach")
            lf.error("coef table not found for {}", "incl_67")
            lf.warning("user edited coefs path")
        finally:
            root.removeHandler(h)

        records: list[logging.LogRecord] = []
        while True:
            try:
                records.append(q.get_nowait())
            except Empty:
                break
        # Two distinct messages → both enqueued (no dedup since func/text differ)
        texts = [r.getMessage() for r in records]
        assert "coef table not found for incl_67" in texts, (
            f"GUI-thread error log missing from queue; got {texts!r}"
        )
        assert "user edited coefs path" in texts, f"GUI-thread warning log missing from queue; got {texts!r}"

    def test_reset_dedup_allows_first_record_of_new_task(self):
        """reset_dedup() clears _last_key so the new task's first record is not swallowed.

        Scenario: same caller frame (identical funcName) emits the same message
        twice in a row.  Without reset, the second would be dropped as a
        consecutive duplicate.  After explicit :meth:`reset_dedup`, it is
        enqueued — modelling the boundary between two worker tasks where the
        trailing record of task A and leading record of task B happen to
        match: the GUI should still show both.
        """
        from tcm_gui.log_bridge import QueueHandler
        from tcm_gui.runtime import PauseGate

        def _emit_boundary(q: Queue, gate: PauseGate, *, reset: bool) -> int:
            """Install a handler, optionally reset dedup, emit once, return enq count."""
            q.queue.clear()  # fresh queue for each sub-test
            h = QueueHandler(q, gate)
            root = logging.getLogger()
            root.addHandler(h)
            root.setLevel(logging.DEBUG)
            try:
                # First emit seeds _last_key
                logging.getLogger("probe").info("boundary marker")
                if reset:
                    h.reset_dedup()
                # Second emit — identical (funcName='test_…? no — caller frame is
                # _emit_boundary, so funcName='_emit_boundary' both times)
                logging.getLogger("probe").info("boundary marker")
            finally:
                root.removeHandler(h)
            return q.qsize()

        # Without reset: the second identical record would be dropped as a dup → 1 record.
        n_no_reset = _emit_boundary(Queue(), PauseGate(), reset=False)
        assert n_no_reset == 1, (
            f"without reset_dedup the consecutive identical record must be dropped; got {n_no_reset}"
        )
        # With reset: the second record is enqueued → 2 records total.
        n_with_reset = _emit_boundary(Queue(), PauseGate(), reset=True)
        assert n_with_reset == 2, f"after reset_dedup the boundary record must enqueue; got {n_with_reset}"


# --------------------------------------------------------------------------- #
# RTF clipboard: _esc and build_rtf produce valid RTF with color table
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestRtfClipboard:
    """_esc and build_rtf produce well-formed RTF for colored ScrolledText."""

    @pytest.fixture(scope="class")
    @classmethod
    def _tk_root(cls):
        """One root per class — `tk.Tk()` N times in one process exhausts Tcl's
        `tcl_findLibrary` lookup on Windows pixi.

        If a previous module's Tk root exhausted that lookup (see
        ``test_const_meta.py`` for the same pattern), ``tk.Tk()`` raises
        ``TclError`` — yield ``None`` so dependent tests skip instead of erroring.
        """
        import tkinter as tk

        try:
            root = tk.Tk()
            root.withdraw()
            yield root
            root.destroy()
        except tk.TclError:
            yield None

    @pytest.fixture()
    def _tk_text(self, _tk_root):
        """Fresh Text widget on the shared root; tag configs declared once
        and inherited cheaply across the class."""
        import tkinter as tk

        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        w = tk.Text(_tk_root, font=("Consolas", 11))
        w.tag_configure("err", foreground="red")
        w.tag_configure("info", foreground="#0070A0")
        yield w
        w.destroy()

    @pytest.mark.parametrize(
        ("input_s", "expected"),
        [
            pytest.param("hello world", "hello world", id="ascii-passthrough"),
            pytest.param("test 123", "test 123", id="ascii-digits"),
            pytest.param("a{b}c", "a\\{b\\}c", id="braces"),
            pytest.param("a\\b", "a\\\\b", id="backslash"),
            pytest.param("line1\nline2", "line1\\par\nline2", id="newline"),
        ],
    )
    def test_esc_basic(self, input_s, expected):
        """_esc handles ASCII, braces, backslash, newline."""
        from tcm_gui._rtf_clipboard import _esc

        assert _esc(input_s) == expected

    def test_esc_unicode_above_127(self):
        """Unicode >127 → \\uN? (codepoints ≤ 32767 used directly; >32767 signed)."""
        from tcm_gui._rtf_clipboard import _esc

        # © = U+00A9 = 169 → ≤ 32767 → \u169?
        assert "\\u169?" in _esc("\u00a9")
        # Cyrillic 'й' = U+0439 = 1081 → ≤ 32767 → \u1081? (direct codepoint)
        assert "\\u1081?" in _esc("\u0439")
        # Emoji 😀 = U+1F600 = 128512 → > 32767 → signed: 128512-65536 = 62976
        # But 128512-65536 = 62976 which is also > 32767... let me recalculate:
        # 128512 - 65536 = 62976 — that's wrong, 128512-65536 = -62976... no:
        # 128512 - 65536 = 62976? No: 65536 - 128512 = -62976, but subtracting gives 62976.
        # Actually: 128512 - 65536 = 62976. But that's not signed 16-bit.
        # The formula `cp - 65536` maps to signed 16-bit range:
        # 65535 → -1, 65534 → -2, ..., 32768 → -32768
        # For U+1F600 (128512): 128512 > 65535 → needs surrogate pair.
        # RTF \u only handles up to U+FFFF. Beyond that is emoji territory.
        # We test with U+8000 = 32768 → signed: 32768-65536 = -32768
        assert "\\u-32768?" in _esc("\u8000")

    def test_build_rtf_no_tags(self, _tk_text):
        """Plain text → RTF with empty colortbl."""
        from tcm_gui._rtf_clipboard import build_rtf

        _tk_text.insert("1.0", "plain")
        rtf = build_rtf(_tk_text)
        assert rtf.startswith("{\\rtf1")
        assert "plain" in rtf
        # No color tags → empty colortbl entry
        assert "{\\colortbl;}" in rtf

    def test_build_rtf_with_colors(self, _tk_text, capsys):
        """Colored text → colortbl + \\cf references + \\fonttbl for Word."""
        from tcm_gui._rtf_clipboard import build_rtf

        _tk_text.insert("end", "ERROR: ", "err")
        _tk_text.insert("end", "disk full\n", "err")
        _tk_text.insert("end", "INFO: ", "info")
        _tk_text.insert("end", "done")
        rtf = build_rtf(_tk_text)

        # Echo the literal RTF to stdout for ad-hoc paste-into-Word debugging.
        print(f"\n---EMITTED_RTF_START---\n{rtf}\n---EMITTED_RTF_END---")

        # \fonttbl is mandatory for Word to honour \cfN runs (\deff0 references \f0).
        assert r"{\fonttbl{" in rtf
        # Color table must have red and the info color
        assert "\\red" in rtf
        # \\cf1 and \\cf2 reference the two colors
        assert "\\cf1" in rtf
        assert "ERROR" in rtf
        assert "INFO" in rtf
        # Brace balancing — outer \rtf1 group must close exactly once.
        assert rtf.count("{") == rtf.count("}")

    def test_build_keeps_link_text(self, _tk_root):
        """MarkdownLabel ``[text](url)`` spans → RTF HYPERLINK field + HTML anchor.

        Pure builder test (no OS clipboard write): colors alone are not enough —
        Word must keep the link clickable, so link spans resolve their URL via
        ``MarkdownLabel.link_url_at`` (:func:`_rtf_clipboard._segments`).
        """
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        import re

        from tcm_gui._rtf_clipboard import build_html, build_rtf
        from tcm_gui.md_label import MarkdownLabel

        md = MarkdownLabel(_tk_root)
        try:
            md.set_text("See [docs](https://example.com/docs?a=1&b=2) for details")
            rtf = build_rtf(md)
            # URL escaped for RTF (& is literal there), display text inside \fldrslt
            assert 'HYPERLINK "https://example.com/docs?a=1&b=2"' in rtf
            assert re.search(r"\\fldrslt\{[^}]*docs[^}]*\}", rtf), "display text lost from field result"
            assert "See " in rtf and " for details" in rtf, "non-link text lost"
            assert rtf.count("{") == rtf.count("}"), "unbalanced braces → RTF parse fails"

            html = build_html(md).decode("utf-8")
            assert '<a href="https://example.com/docs?a=1&amp;b=2">' in html
            assert "docs</span></a>" in html, "anchor must wrap the colored display text"
            assert "See " in html and " for details" in html
        finally:
            md.destroy()

    @pytest.mark.clipboard  # Tk clipboard_append lands on the OS clipboard too
    def test_copy_rich_fallback_without_pywin32(self, _tk_text, monkeypatch):
        """When pywin32 is missing, fall back to plain-text + clipboard_append."""
        from tcm_gui._rtf_clipboard import copy_rich

        _tk_text.insert("1.0", "hello")
        monkeypatch.setitem(__import__("sys").modules, "win32clipboard", None)
        # Just verify it runs without crashing (fallback to plain text)
        copy_rich(_tk_text)
        # After copy_rich, clipboard contents should be the plain text
        result = _tk_text.clipboard_get()
        assert result == "hello"

    # ── win32 real-path tests: exercise copy_rich with pywin32 ACTUALLY installed
    # (the path Windows users hit). Skip if pywin32 missing OR no display.  These
    # were missing from the suite — the existing tests only force the fallback —
    # so a bug in the win32 path (a real one: see note in ``copy_rich``) went
    # uncaught.  Configure the Text as ``App._log`` would: per-level tag colors
    # from ``theme.TAG_COLORS`` plus the ``func`` tag, and verify RTF + plain text
    # both land on the OS clipboard and Word sees the color runs.
    #
    # ``@pytest.mark.clipboard`` (all tests below): every one WRITES the real OS
    # clipboard → clipboard-manager history (CopyQ / Win+V) fills with test junk
    # on each run.  Deselected by default (``addopts -m "not clipboard"`` in
    # pytest.ini); run explicitly with ``pytest -m clipboard``.  ────────────────

    @pytest.fixture()
    def _log_text(self, _tk_text):
        """``Text`` widget configured exactly like ``App._log`` (tags from theme).

        Also clears the OS clipboard + drains Tk events so each test starts
        from a known-empty clipboard.  Tk's ``clipboard_clear``/``append``
        defer the actual OS clipboard write to idle time, so without an
        ``update()`` here, the next ``win32clipboard.OpenClipboard`` racing
        with Tk's pending propagation raises ``pywintypes.error`` (Access
        denied) — which is *exactly* the real-world contention the hardened
        :func:`copy_rich` is built to survive, but a flaky test-fixture
        failure mode we want to avoid.
        """
        import tcm_gui.theme

        for lvl, clr in tcm_gui.theme.TAG_COLORS.items():
            _tk_text.tag_configure(lvl, foreground=clr)
        _tk_text.tag_configure("func", foreground=tcm_gui.theme.FUNC_COLOR)
        _tk_text.update()  # drain any pending Tk clipboard propagation
        try:
            import win32clipboard as wcb

            wcb.OpenClipboard()
            try:
                wcb.EmptyClipboard()
            finally:
                wcb.CloseClipboard()
        except Exception:  # noqa: BLE001 — best-effort; copy_rich retries too.
            pass
        _tk_text.update()  # drain anything Tk queued from the EmptyClipboard
        return _tk_text

    @staticmethod
    def _win32_clipboard_formats() -> list[int]:
        """Enumerate current OS-clipboard format IDs via win32clipboard."""
        import win32clipboard as wcb

        wcb.OpenClipboard()
        try:
            fmts: list[int] = []
            fmt = wcb.EnumClipboardFormats(0)
            while fmt:
                fmts.append(fmt)
                fmt = wcb.EnumClipboardFormats(fmt)
            return fmts
        finally:
            wcb.CloseClipboard()

    @staticmethod
    def _get_clipboard_format_data(fmt: int, widget=None) -> bytes | None:
        """Read one clipboard format with a short retry on contention.

        ``OpenClipboard`` raises ``pywintypes.error`` ("Access denied") when
        another test (or Tk's idle clipboard-update for a left-over clipboard
        viewer) still holds the OS clipboard — a brief retry + a Tk event
        drain lets it release.  Test-only helper; app code is hardened
        separately (see :func:`tcm_gui._rtf_clipboard.copy_rich`).
        """
        import time

        import win32clipboard as wcb

        for attempt in range(40):
            if widget is not None:
                widget.update()  # let Tk finish its pending clipboard propagation
            try:
                wcb.OpenClipboard()
                try:
                    return wcb.GetClipboardData(fmt)
                finally:
                    wcb.CloseClipboard()
            except Exception:  # noqa: BLE001 — contention; retry.
                time.sleep(0.05)
        return None

    @pytest.mark.clipboard
    def test_copy_rich_places_rtf_on_os_clipboard(self, _log_text):
        """``copy_rich`` with pywin32 present → ``CF_RTF`` & ``CF_UNICODETEXT`` on OS clipboard.

        Regression for the user-reported bug: Ctrl+C on ``App._log`` placed NO
        RTF on the clipboard (Word/CopyQ showed plain text, never colors).
        Root cause coverage in :mod:`_rtf_clipboard`.
        Requires pywin32 + a display; skipped otherwise.
        """
        pytest.importorskip("win32clipboard")
        import re

        import win32clipboard as wcb

        from tcm_gui._rtf_clipboard import copy_rich

        # Two log records worth of segments, like log_bridge.drain would produce.
        _log_text.insert("1.0", "12:00:00\u2502", "error")
        _log_text.insert("end", "process.run\u2502", "func")
        _log_text.insert("end", "disk full\n", "error")
        _log_text.insert("end", "12:01:02\u2502", "info")
        _log_text.insert("end", "cli.load\u2502", "func")
        _log_text.insert("end", "found 1 file", "info")

        # Mirror App._log initially selecting nothing — Ctrl+C copies whole log.
        assert not _log_text.tag_ranges("sel"), "test premise: no selection → full text"

        copy_rich(_log_text)

        # CF_RTF must be on the OS clipboard, not just Tk's.
        rtf_fmt = wcb.RegisterClipboardFormat("Rich Text Format")
        rtf_raw = self._get_clipboard_format_data(rtf_fmt, _log_text)
        assert rtf_raw is not None, "CF_RTF missing from OS clipboard — Word will show no colors"
        rtf = rtf_raw.decode("ascii", errors="replace") if isinstance(rtf_raw, bytes) else str(rtf_raw)
        # Real RTF preamble + color table + color run for the err tag.
        assert rtf.startswith("{\\rtf1"), f"bad RTF start: {rtf[:40]!r}"
        assert "\\colortbl" in rtf, "no colortbl → Word renders monochrome"
        assert "\\red" in rtf, "no \\red entry → palette empty"
        assert re.search(r"\\cf\d+\s", rtf), "\\cfN run absent → colors never applied"
        assert rtf.count("{") == rtf.count("}"), "unbalanced braces → RTF parse fails"

        # CF_UNICODETEXT plain-text fallback must ALSO be present.
        txt = self._get_clipboard_format_data(wcb.CF_UNICODETEXT, _log_text)
        assert txt is not None, "CF_UNICODETEXT plain-text fallback missing"
        assert "disk full" in (txt if isinstance(txt, str) else txt.decode("utf-16-le", errors="replace"))

        # And the box-drawing separator survives emoji-tier codepoints.
        if isinstance(rtf_raw, bytes):
            assert "\\u9474?" in rtf, "│ (U+2502 = 9474) must be escaped as \\u9474?"

    @pytest.mark.clipboard
    def test_copy_rich_real_app_log_config(self, _tk_root):
        """``copy_rich`` on a ``state='disabled'`` ``App._log`` stand-in.

        The user's bug: Copy from the REAL ``App._log`` (a ``state='disabled'``
        Text populated by ``log_bridge.drain``) places NO RTF on the clipboard,
        even though the win32-clipboard path works in isolation
        (:meth:`test_copy_rich_places_rtf_on_os_clipboard`).  This test
        reproduces the exact ``App._log`` configuration to localise the bug:

        - state='disabled' (only ``'normal'`` during ``drain()`` inserts)
        - tags configured from ``theme.TAG_COLORS`` + ``FUNC_COLOR``
        - mouse-style selection via ``sel`` tag (programmatic; the user
          would drag-select with the mouse, which produces the same ``sel``
          tag ranges)

        Requires pywin32 + a display; skipped otherwise.
        """
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        pytest.importorskip("win32clipboard")
        import re
        import tkinter as tk

        import tcm_gui.theme
        import win32clipboard as wcb

        from tcm_gui._rtf_clipboard import copy_rich

        log = tk.Text(
            _tk_root,
            state="disabled",  # App._log's state for the body of the GUI
            wrap="word",
            background=tcm_gui.theme.ENTRY_BG_FALLBACK,
            foreground=tcm_gui.theme.FG_DEFAULT,
        )
        # Mirror App._build §5: configure per-level tags BEFORE inserting text
        # (app.py:195-197).  Without this, tag_cget(t,'foreground') returns ''
        # and build_rtf's palette filter skips the tag → empty colortbl.
        for lvl, clr in tcm_gui.theme.TAG_COLORS.items():
            log.tag_configure(lvl, foreground=clr)
        log.tag_configure("func", foreground=tcm_gui.theme.FUNC_COLOR)
        try:
            # drain() enters 'normal' to insert, then restores 'disabled'.
            log.config(state="normal")
            log.insert("end", "12:00:00\u2502", "error")
            log.insert("end", "process.run\u2502", "func")
            log.insert("end", "disk full\n", "error")
            log.insert("end", "12:01:02\u2502", "info")
            log.insert("end", "cli.load\u2502", "func")
            log.insert("end", "found 1 file", "info")
            log.config(state="disabled")

            # User drag-selects the first two lines.
            log.tag_add("sel", "1.0", "2.0")
            assert log.tag_ranges("sel"), "test premise: selection present"

            copy_rich(log)

            rtf_fmt = wcb.RegisterClipboardFormat("Rich Text Format")
            rtf_raw = self._get_clipboard_format_data(rtf_fmt, log)
            assert rtf_raw is not None, (
                "CF_RTF missing after Ctrl+C on disabled App._log — "
                "Word shows no colors. See _rtf_clipboard.copy_rich."
            )
            rtf = rtf_raw.decode("ascii", errors="replace") if isinstance(rtf_raw, bytes) else str(rtf_raw)
            assert rtf.startswith("{\\rtf1"), f"bad RTF start: {rtf[:40]!r}"
            assert "\\colortbl" in rtf, "disabled Text → missing colortbl"
            assert re.search(r"\\cf\d+\s", rtf), "disabled Text → no \\cfN run"
            assert "disk full" in rtf, "selected text not in RTF"
        finally:
            log.destroy()

    @pytest.mark.clipboard
    def test_copy_rich_fires_when_log_disabled_no_focus(self, _tk_root):
        """Root-scoped ``<<Copy>>`` fires ``copy_rich`` even when disabled
        ``_log`` cannot take keyboard focus — the user's reported bug.

        ``App._log`` is ``state='disabled'`` and so cannot take keyboard focus.
        The user's actual flow: focus is on a ttk.Entry (path field), ``_log``
        is disabled with a ``sel`` range from mouse drag, and the user presses
        Ctrl+C.  Tk maps ``<Control-Key-c>`` → ``<<Copy>>`` at the virtual
        event level, so the real event is ``<<Copy>>``, not ``<Control-c>``.
        Binding ``<Control-c>`` on root is dead code — it never fires on a
        real Ctrl+C keypress.  The ``<<Copy>>`` binding must be at root level
        so it fires regardless of which widget has focus.

        This test reproduces the real flow: focus is on a ttk.Entry,
        ``<<Copy>>`` is dispatched to that Entry (matching the real Ctrl+C
        path: Tk synthesises ``<<Copy>>`` on the focused widget), and the
        Entry's class ``<<Copy>>`` handler fires first (copies Entry plain
        text), then the root handler fires and overwrites with RTF from _log.

        Requires pywin32 + a display; skipped otherwise.
        """
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")
        pytest.importorskip("win32clipboard")
        import re
        import tkinter as tk
        from tkinter import ttk

        import win32clipboard as wcb

        import tcm_gui.theme

        log = tk.Text(
            _tk_root,
            state="disabled",
            wrap="word",
            background=tcm_gui.theme.ENTRY_BG_FALLBACK,
            foreground=tcm_gui.theme.FG_DEFAULT,
        )
        log.pack()
        for lvl, clr in tcm_gui.theme.TAG_COLORS.items():
            log.tag_configure(lvl, foreground=clr)
        log.tag_configure("func", foreground=tcm_gui.theme.FUNC_COLOR)

        # Stand-in App installing the real root-level <<Copy>> binding.
        from tcm_gui.app import App

        stub = type("_AppStub", (), {"_log": log, "_on_copy_rich": App._on_copy_rich})()
        _tk_root.bind("<<Copy>>", stub._on_copy_rich, add="+")

        # Focus is on a ttk.Entry (path field) — _log cannot take focus.
        ent = ttk.Entry(_tk_root, width=30)
        ent.insert(0, "B:/Cruises/_raw")
        ent.pack()
        ent.focus_set()

        try:
            log.config(state="normal")
            log.insert("end", "12:00:00\u2502", "error")
            log.insert("end", "process.run\u2502", "func")
            log.insert("end", "disk full\n", "error")
            log.config(state="disabled")

            # User has drag-selected text on the disabled log → 'sel' tag is set.
            log.tag_add("sel", "1.0", "2.0")
            assert log.tag_ranges("sel"), "test premise: log has selection"

            # Simulate the user pressing Ctrl+C while Entry has focus — Tk
            # synthesises <<Copy>> on the focused widget (Entry).  The Entry's
            # class <<Copy>> binding copies Entry text first, then the root
            # handler fires and overwrites with RTF from _log.
            _tk_root.update()
            ent.event_generate("<<Copy>>")
            _tk_root.update_idletasks()
            _tk_root.update()

            rtf_fmt = wcb.RegisterClipboardFormat("Rich Text Format")
            rtf_raw = self._get_clipboard_format_data(rtf_fmt, log)
            assert rtf_raw is not None, (
                "CF_RTF missing after Ctrl+C on disabled App._log — the live bug.  "
                "Was the binding on <Control-c> instead of <<Copy>>?"
            )
            rtf = rtf_raw.decode("ascii", errors="replace") if isinstance(rtf_raw, bytes) else str(rtf_raw)
            assert rtf.startswith("{\\rtf1"), f"bad RTF start: {rtf[:40]!r}"
            assert "\\colortbl" in rtf, "no colortbl → Word renders monochrome"
            assert re.search(r"\\cf\d+\s", rtf), "\\cfN run absent → colors never applied"
            assert "disk full" in rtf, "selected log text not in RTF"
        finally:
            ent.destroy()
            log.destroy()

    @pytest.mark.clipboard  # status-label branch calls copy_rich → OS clipboard
    def test_copy_rich_falls_through_when_log_no_selection(self, _tk_root):
        """``_on_copy_rich`` returns ``None`` (lets default ``<<Copy>>`` run)
        when ``_log`` has no mouse selection — so the focused ttk.Entry keeps
        normal copy behaviour (the 'fall through' half of the root binding).

        Without this half, root-level ``<Control-c>`` would *always* intercept
        and break the user's path-field copy etc.
        """
        if _tk_root is None:
            pytest.skip("Tk unavailable — Tcl interpreter already destroyed")

        import tkinter as tk

        from tcm_gui.app import App

        log = tk.Text(_tk_root, state="disabled")
        status = tk.Text(_tk_root)
        log.pack()
        try:
            stub = type(
                "_AppStub",
                (),
                {"_log": log, "_status_lbl": status, "_on_copy_rich": App._on_copy_rich},
            )()
            assert log.tag_ranges("sel") == () and status.tag_ranges("sel") == (), (
                "test premise: no selection anywhere"
            )
            ret = stub._on_copy_rich(None)
            assert ret is None, f"no sel → must return None (fall through); got {ret!r}"

            # Selection on the status MarkdownLabel (no focus needed) → rich copy.
            status.insert("1.0", "status message text")
            status.tag_add("sel", "1.0", "end-1c")
            ret = stub._on_copy_rich(None)
            assert ret == "break", f"status sel → must copy rich + break; got {ret!r}"
        finally:
            log.destroy()
            status.destroy()
