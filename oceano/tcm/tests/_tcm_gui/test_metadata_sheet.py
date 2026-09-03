"""Style: test_coef_sheet.py, test_gui_actions.py (gui_project fixture).

Reference: docs/reference/config_reference.md (metadata section)
Guides: docs/project_developer_guide/GUI/architecture.md, docs/project_developer_guide/GUI/decisions.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omegaconf import OmegaConf

from tcm import cli, format
from tcm._constants import RAW_DIR_NAME
from tcm.schema import Config, Return


def _write_info_devices(cruise: Path):
    (cruise / "info_devices.yaml").write_text(
        "# Instrument_ID: [Point, Sea_depth, H_above_bot, Symbol, Lat, Lon, Time_st, Time_en, Burst_dt, Bursts_t, Comment]\n"
        '"i67": ["", ~, 0, "↟", ~, ~, "2026-07-11T12:24:52", "2026-07-20T13:27:27", 60, 600]\n'
        '"i90": ["", ~, 0, "↟", ~, ~, "2026-07-11T12:20:12", "2026-07-20T10:34:06", 60, 600]\n',
        encoding="utf-8",
    )


def test_run_does_not_crash_on_metadata_time_range(tmp_path, monkeypatch, mocker):
    """Metadata time_range has has_date+is_metadata — must not KeyError on Run.

    Regression: get_edited_dates iterated is_metadata rows which lack ``key``,
    so ``out[m['key']]`` raised KeyError and _on_run crashed before the worker
    thread started — GUI log stayed empty while vscode console showed the traceback.
    Covers both the crash fix and the no-silent-failure fix (App._on_run try/except).
    """
    import tcm_gui.coef_sheet as coef_sheet

    mock_sh = MagicMock()
    mock_sh.total_columns.return_value = 6
    mock_sh.total_rows.return_value = 0
    mock_sh.get_children.return_value = []
    mock_sh.get_cell_data.return_value = ""
    mock_sh.tag_names.return_value = []
    mock_sh.winfo_rgb.return_value = (0, 0, 0)
    _items2: dict = {}
    _kids2: dict = {}

    def _insert2(**kw):
        iid = f"iid_{kw.get('text', 'x')}_{len(_items2)}"
        parent = kw.get("parent") or ""
        _kids2.setdefault(parent, []).append(iid)
        _items2[iid] = {"values": kw.get("values", ()), "text": kw.get("text", "")}
        return iid

    mock_sh.insert.side_effect = _insert2

    def _item2(iid=None, **kw):
        if iid is None:
            return {}
        if kw:
            if iid in _items2 and "text" in kw:
                _items2[iid]["text"] = kw["text"]
            return {}
        return _items2.get(iid, {})

    mock_sh.item.side_effect = _item2
    mock_sh.get_children.side_effect = lambda parent="": list(_kids2.get(parent or "", ()))
    # Minimal cfg with time_ranges so _build_metadata would also create time_range row
    cfg = {
        "input": {
            "path": str(tmp_path / "dummy.txt"),
            "time_ranges": ["2026-07-11T12:00:00", "2026-07-20T12:00:00"],
            "coefs": {},
        }
    }
    # Provide a fake date cell so _ph.get returns something (to trigger the has_date branch)
    with patch.object(coef_sheet, "Sheet", return_value=mock_sh):
        cs = coef_sheet.ConfigSheet.__new__(coef_sheet.ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg
        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ()
        cs._snap_meta = ()
        cs._fg_default = "#000000"
        cs._int_row_of = {}
        cs._vis = ()
        cs._col_resize = MagicMock()
        cs.on_edit_begin = None
        cs._readonly = False
        cs._field_iid = cs._field_pending = cs._field_show_job = cs._field_hide_job = None
        cs._hover_field = cs._hover_btn = None
        cs.on_hover_status = None
        # (hover_status removed — live detail via _time_ranges_detail)
        cs._status_iid = None
        cs._status_hint = ""
        cs._metadata = None
        cs._sync_status = None
        cs._rebuild_row_caches = MagicMock()
        cs._apply_open = MagicMock()
        cs._apply_styles = MagicMock()
        cs._apply_default_fg = MagicMock()
        cs._apply_placeholders = MagicMock()
        cs._apply_validations = MagicMock()
        cs._stretch_last_col = MagicMock()
        cs._hide_hover_field = MagicMock()
        cs._clear_status = MagicMock()
        cs._take_snapshot = MagicMock()
        # Build metadata node — creates has_date+is_metadata time_range row
        cs._build_metadata()
        # Force a date placeholder to be "present" so get_edited_dates would return it if buggy
        # Mock _ph.get to return a date for that row
        cs._ph = MagicMock()
        # Find the time_range iid's row
        tr_iid = next((i for i, m in cs._meta.items() if m.get("label") == "time_range"), None)
        assert tr_iid is not None
        # Map it to a fake row 99
        cs._row_map = lambda: {tr_iid: 99}
        # Make _ph.get return a date for that row/col
        cs._ph.get.return_value = "2026-07-11T12:20:12"
        # This used to raise KeyError: 'key' because is_metadata rows have no 'key'
        dates = cs.get_edited_dates()
        # Metadata time_range must NOT leak into coef dates
        assert dates == {}, f"is_metadata rows must be skipped, got {dates}"
        # _current_state (used by _on_run -> _write_coefs) must not raise
        coefs, got_dates, path = cs._current_state()
        assert got_dates == {}


def test_build_metadata_empty_path_does_not_probe_cwd(tmp_path):
    """No-args GUI launch must NOT anchor-probe the cwd for the device file.

    Regression: the metadata-path default did ``Path(path or "").absolute()``
    BEFORE the empty check — ``Path("").absolute()`` is the cwd, which
    differs from ``Path(".")``, so the guard never matched and
    ``find_dir_raw_absolute`` logged "Not standard input path <project
    root>" on every argless startup.  Repo-internal paths are skipped too.
    """
    import tcm_gui.coef_sheet as coef_sheet

    # (path, probe expected?) — empty/"." and repo-internal paths skip the
    # probe; a normal outside path still derives the device-file default.
    cases = [("", False), (".", False), (str(tmp_path), True), (str(Path(__file__).resolve()), False)]
    for path_val, expect_probe in cases:
        mock_sh = MagicMock()
        _items: dict = {}

        def _insert(_items=_items, **kw):
            iid = f"iid_{len(_items)}"
            _items[iid] = {"values": kw.get("values", ()), "text": kw.get("text", "")}
            return iid

        mock_sh.insert.side_effect = _insert
        cs = coef_sheet.ConfigSheet.__new__(coef_sheet.ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = {"input": {"path": path_val}}
        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ()
        cs._snap_meta = ()
        cs._fg_default = "#000000"
        cs._int_row_of = {}
        cs._vis = ()
        cs._col_resize = MagicMock()
        cs.on_edit_begin = None
        cs._readonly = False
        cs._status_iid = None
        cs._status_hint = ""
        cs._metadata = None
        cs._sync_status = None
        cs._rebuild_row_caches = MagicMock()
        cs._apply_open = MagicMock()
        cs._apply_styles = MagicMock()
        cs._apply_default_fg = MagicMock()
        cs._apply_placeholders = MagicMock()
        cs._apply_validations = MagicMock()
        cs._stretch_last_col = MagicMock()
        cs._hide_hover_field = MagicMock()
        cs._clear_status = MagicMock()
        cs._take_snapshot = MagicMock()
        with patch("tcm.paths.find_dir_raw_absolute") as spy:
            cs._build_metadata()
        if expect_probe:
            spy.assert_called_once()
        else:
            spy.assert_not_called()  # no probe → no "Not standard input path" warning
        meta_iid = next(i for i, m in cs._meta.items() if m.get("is_metadata_root"))
        if not expect_probe:
            assert _items[meta_iid]["values"][0] == "", (
                f"device-file default must stay empty for path={path_val!r}"
            )


def test_metadata_sheet_shows_device_not_fallback(tmp_path, monkeypatch, mocker):
    """With cruise/info_devices.yaml, burst_dt/t must be 60/600, not '?'.

    Regression: _build_metadata fallback to input.time_ranges[0,-1] was overwriting
    device values when _meta_for_stem returned None (e.g. wrong device-dir lookup).
    """
    cruise = tmp_path / "260711_Pionerskiy@i"
    raw = cruise / RAW_DIR_NAME
    raw.mkdir(parents=True)
    _write_info_devices(cruise)
    for stem in ("@i_90", "@i_67"):
        (raw / f"{stem}.TXT").write_text(
            "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp\n"
            "2026,07,11,13,10,34,100,200,300,400,500,600,12.5,25.0\n"
            "2026,07,20,09,55,05,101,201,301,401,501,601,12.5,25.0\n",
            encoding="utf-8",
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog"])
    # Return a real DictConfig so collected can be OmegaConf.to_container'd
    mocker.patch(
        "tcm.processing.run_processing",
        return_value=OmegaConf.create({"input": {"path": "x"}, "program": {"return_": str(Return.END)}}),
    )
    res = cli.call_in_raw_dir(
        __import__("tcm.processing", fromlist=["run"]).run,
        input={"path": str(raw)},
        program={"return_": Return.CFG_FROM_ARGS},
        exit_on_error=False,
    )
    assert len(res) >= 4
    collected = res[3]
    assert len(collected) == 2, collected

    from meta_finder import io_info_files

    device_meta = io_info_files.read_metadata_file(cruise / "info_devices.yaml")
    stems = [s for s, _, _ in collected]
    stems_by_pcid: dict[str, list[str]] = {}
    for s in stems:
        pc = format.to_pcid_from_name(format.stem_to_pcid(s))
        stems_by_pcid.setdefault(pc, []).append(s)
    for v in stems_by_pcid.values():
        v.sort()

    def _meta_for_stem(stem: str):
        pcid = format.to_pcid_from_name(format.stem_to_pcid(stem))
        for cand in (pcid, pcid.replace("_", "")):
            if cand in device_meta:
                ent = device_meta[cand]
                if isinstance(ent, dict):
                    sid = (
                        str(stems_by_pcid.get(pcid, [stem]).index(stem))
                        if stem in stems_by_pcid.get(pcid, [])
                        else "0"
                    )
                    if sid in ent:
                        return list(ent[sid])
                    if "0" in ent:
                        return list(ent["0"])
                elif isinstance(ent, (list, tuple)):
                    return list(ent)
        return None

    stem0, _, cfg_dc0 = collected[0]
    cfg0 = OmegaConf.to_container(cfg_dc0, resolve=True)
    if cfg0.get("program", {}).get("return_") == str(Return.CFG_FROM_ARGS):
        cfg0["program"]["return_"] = str(Return.END)
    md0 = _meta_for_stem(stem0)
    assert md0 is not None
    assert md0[8] == 60 and md0[9] == 600

    import tcm_gui.coef_sheet as coef_sheet

    mock_sh = MagicMock()
    mock_sh.total_columns.return_value = 6
    mock_sh.total_rows.return_value = 0
    mock_sh.get_children.return_value = []
    mock_sh.get_cell_data.return_value = ""
    mock_sh.tag_names.return_value = []
    mock_sh.winfo_rgb.return_value = (0, 0, 0)
    _items: dict = {}
    _kids: dict = {}

    def _insert(**kw):
        iid = f"iid_{kw.get('text', 'x')}_{len(_items)}"
        parent = kw.get("parent") or ""
        _kids.setdefault(parent, []).append(iid)
        _items[iid] = {"values": kw.get("values", ()), "text": kw.get("text", "")}
        return iid

    mock_sh.insert.side_effect = _insert

    def _item(iid=None, **kw):
        if iid is None:
            return {}
        if kw:
            if iid in _items and "text" in kw:
                _items[iid]["text"] = kw["text"]
            return {}
        return _items.get(iid, {})

    mock_sh.item.side_effect = _item
    mock_sh.get_children.side_effect = lambda parent="": list(_kids.get(parent or "", ()))

    with patch.object(coef_sheet, "Sheet", return_value=mock_sh):
        cs = coef_sheet.ConfigSheet.__new__(coef_sheet.ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg0
        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ()
        cs._snap_meta = ()
        cs._fg_default = "#000000"
        cs._int_row_of = {}
        cs._vis = ()
        cs._col_resize = MagicMock()
        cs.on_edit_begin = None
        cs._readonly = False
        cs._field_iid = cs._field_pending = cs._field_show_job = cs._field_hide_job = None
        cs._hover_field = cs._hover_btn = None
        cs.on_hover_status = None
        # (hover_status removed — live detail via _time_ranges_detail)
        cs._status_iid = None
        cs._status_hint = ""
        cs._metadata = md0
        cs._sync_status = None
        cs._rebuild_row_caches = MagicMock()
        cs._apply_open = MagicMock()
        cs._apply_styles = MagicMock()
        cs._apply_default_fg = MagicMock()
        cs._apply_placeholders = MagicMock()
        cs._apply_validations = MagicMock()
        cs._stretch_last_col = MagicMock()
        cs._hide_hover_field = MagicMock()
        cs._clear_status = MagicMock()
        cs._take_snapshot = MagicMock()
        cs._cfg = cfg0
        cs._build_metadata()
        burst_iid = next((i for i, m in cs._meta.items() if m.get("label") == "burst_dt/t"), None)
        assert burst_iid is not None
        vals = _items[burst_iid]["values"]
        assert vals[0] == "60", f"burst_dt should be 60, got {vals[0]!r}"
        assert vals[1] == "600", f"bursts_t should be 600, got {vals[1]!r}"
        tr_iid = next((i for i, m in cs._meta.items() if m.get("label") == "time_range"), None)
        assert md0[6] in _items[tr_iid]["values"], f"time_range missing {md0[6]!r}"


# ── metadata node (re)insertion must not flag the tab dirty ──────────────────


def test_metadata_rebuild_keeps_tab_clean(_session_tk_root, tmp_path):
    """Regression: rebuilding the ``metadata`` subtree left ``is_dirty`` True.

    ``_rebuild_metadata_rows`` deletes old rows and inserts new ones with
    fresh iids but retook only ``_snap_meta`` — the iid-keyed coefs snapshot
    ``_snap`` kept stale/deleted iids, so ``is_dirty`` compared unequal
    forever: rail showed ``*`` without any user edit and Run rewrote YAML.
    Fix under test: rebuild retakes both snapshots.

    Uses a real ``info_devices.yaml`` so the metadata is loaded, not
    autofilled (which now correctly stays dirty until Run).
    """
    import tkinter as tk

    from tcm_gui.cli_cfg import default_cfg
    from tcm_gui.coef_sheet import ConfigSheet

    if _session_tk_root is None:
        pytest.skip("Tk not available")
        return
    root = _session_tk_root
    try:
        root.geometry("600x300+40+40")
        root.deiconify()
    except tk.TclError:
        pytest.skip("Tk not available")
        return

    info = tmp_path / "info_devices.yaml"
    _write_info_devices(tmp_path)

    # Read a real entry from the file so the sheet loads (not autofills)
    # the metadata — autofilled metadata is now correctly dirty until Run.
    from meta_finder import io_info_files as _io

    _loaded_md = next(iter(_io.read_metadata_file(info).values()))
    if isinstance(_loaded_md, dict):
        _loaded_md = next(iter(_loaded_md.values()))
    _loaded_md = list(_loaded_md)

    sheet = ConfigSheet(root)
    sheet.sh.pack(fill="both", expand=True)
    try:
        sheet.load(
            default_cfg(),
            full=False,
            config_root=Config,
            return_enum=Return,
            metadata=_loaded_md,
            metadata_path=str(info),
        )
        root.update_idletasks()
        root.update()
        assert not sheet.is_dirty, "tab dirty right after clean load"
        assert not sheet.is_metadata_dirty(), "loaded (non-autofilled) metadata must be clean"

        # Hover-commit on metadata_path → subtree rebuilt with fresh iids
        sheet._reload_metadata_from(str(info))
        root.update_idletasks()
        root.update()
        assert not sheet.is_dirty, "metadata node insertion flagged tab dirty"
        assert not sheet.is_metadata_dirty(), "reload-from-disk must not leave metadata dirty"

        # Direct rebuild entry (browse select) — same invariant
        sheet._rebuild_metadata_rows()
        root.update_idletasks()
        root.update()
        assert not sheet.is_dirty
        assert not sheet.is_metadata_dirty()
    finally:
        sheet.sh.destroy()


def test_poll_dirty_tabs_propagates_real_bool(_session_tk_root, tmp_path):
    """Regression: ``_poll_dirty_tabs`` stored a bound method in ``TabRail``
    state because ``is_metadata_dirty`` was referenced without ``()`` — a bound
    method is always truthy, so ``_full_label`` appended ``*`` to EVERY tab
    forever (clean sheet included).  A clean sheet must propagate
    ``set_dirty(stem, False)`` (a real bool).

    Uses a real ``info_devices.yaml`` so loaded metadata is clean baseline.
    """
    import tkinter as tk

    from unittest.mock import MagicMock

    from tcm_gui.app import App
    from tcm_gui.cli_cfg import default_cfg
    from tcm_gui.coef_sheet import ConfigSheet

    if _session_tk_root is None:
        pytest.skip("Tk not available")
        return
    root = _session_tk_root
    try:
        root.geometry("600x300+40+40")
        root.deiconify()
    except tk.TclError:
        pytest.skip("Tk not available")
        return

    info = tmp_path / "info_devices.yaml"
    _write_info_devices(tmp_path)

    from meta_finder import io_info_files as _io2

    _loaded_md = next(iter(_io2.read_metadata_file(info).values()))
    if isinstance(_loaded_md, dict):
        _loaded_md = next(iter(_loaded_md.values()))
    _loaded_md = list(_loaded_md)

    sheet = ConfigSheet(root)
    sheet.sh.pack(fill="both", expand=True)
    try:
        sheet.load(
            default_cfg(),
            full=False,
            config_root=Config,
            return_enum=Return,
            metadata=_loaded_md,
            metadata_path=str(info),
        )
        root.update_idletasks()
        assert not sheet.is_dirty
        assert not sheet.is_metadata_dirty(), "loaded metadata must be clean baseline"

        app = App.__new__(App)
        app._pages = {"default": sheet}
        app._rail = MagicMock()
        app._poll_dirty_tabs()
        app._rail.set_dirty.assert_called_once_with("default", False, False)
        # the stored values must be real bools, never a truthy object
        assert app._rail.set_dirty.call_args.args[1] is False
        assert app._rail.set_dirty.call_args.args[2] is False
    finally:
        sheet.sh.destroy()


def test_edit_coef_date_marks_tab_dirty(_session_tk_root):
    """Regression: editing a coef date cell (tksheet col 1 on a date-only row
    with ``max_col=0``) left ``is_dirty`` False — ``_data_snapshot`` skipped
    the whole row, so the date edit was invisible to dirty tracking.
    """
    import tkinter as tk

    from tcm_gui._sheet_tint import _DATE_PH_COL
    from tcm_gui.cli_cfg import COEF_SHAPES
    from tcm_gui.coef_sheet import ConfigSheet

    if _session_tk_root is None:
        pytest.skip("Tk not available")
        return
    root = _session_tk_root
    try:
        root.geometry("700x400+40+40")
        root.deiconify()
    except tk.TclError:
        pytest.skip("Tk not available")
        return

    coefs: dict = {}
    for name, shape in COEF_SHAPES.items():
        if not shape:
            coefs[name] = 0.0
        elif len(shape) == 2:
            coefs[name] = [[0.0] * shape[1] for _ in range(shape[0])]
        else:
            coefs[name] = [0.0] * shape[0]
    cfg = {"input": {"path": "D:/x/dummy.txt", "coefs": coefs}}

    sheet = ConfigSheet(root)
    sheet.sh.pack(fill="both", expand=True)
    try:
        sheet.load(cfg, full=False, config_root=Config, return_enum=Return)
        root.update_idletasks()
        root.update()
        assert not sheet.is_dirty, "tab dirty right after clean load"

        date_rows = [
            (iid, m) for iid, m in sheet._meta.items() if m.get("has_date") and m.get("max_col", 0) == 0
        ]
        assert date_rows, "expected at least one date-only coef row"
        iid, _ = date_rows[0]
        r = sheet._internal_row(iid)
        assert r is not None

        # Simulate a real date edit: begin-edit clears the ghost, then commit
        sheet._ph.clear(sheet.sh, r, _DATE_PH_COL)
        sheet.sh.set_cell_data(r, _DATE_PH_COL, "2026-08-02T03:04:05", redraw=True)
        root.update()
        assert sheet.is_dirty, "editing a coef date did not mark the tab dirty"

        # Revert to the ghost-empty original → clean again
        sheet.mark_clean()
        assert not sheet.is_dirty
    finally:
        sheet.sh.destroy()


def test_autofilled_metadata_dirty_but_not_coefs(_session_tk_root, tmp_path, mocker):
    """New/autofilled metadata must be treated as dirty — but it must
    NOT mark the coefs dirty (separate flags), so ``_write_coefs`` won't rewrite
    an unchanged run YAML.  ``_write_metadata`` persists it as a new device file.

    Covers both autofill shapes — the regression case is the *empty* branch
    (no ``time_ranges`` to seed ``time_st``/``time_en``); the old check
    ``autofilled and any(not is_placeholder(v) for v in md_list)`` was False
    for it, hiding the bug.
    """
    import tkinter as tk

    from meta_finder import io_info_files

    from tcm_gui.app import App
    from tcm_gui.coef_sheet import ConfigSheet

    if _session_tk_root is None:
        pytest.skip("Tk not available")
        return
    root = _session_tk_root
    try:
        root.geometry("700x400+40+40")
        root.deiconify()
    except tk.TclError:
        pytest.skip("Tk not available")
        return

    counter = {"n": 0}

    def _make_sheet(cfg: dict) -> tuple[ConfigSheet, Path]:
        counter["n"] += 1
        md_p = tmp_path / f"info_{counter['n']}.yaml"
        sh = ConfigSheet(root)
        sh.sh.pack(fill="both", expand=True)
        sh.load(cfg, full=False, config_root=Config, return_enum=Return, metadata_path=str(md_p))
        root.update_idletasks()
        root.update()
        return sh, md_p

    # Case 1: time_ranges present → autofill seeds time_st/time_en (was passing)
    sheet, md_path = _make_sheet({
        "input": {
            "path": "D:/x/_raw/dummy.txt",
            "time_ranges": ["2026-07-11T12:20:12", "2026-07-11T12:20:13"],
            "coefs": {},
        },
    })
    try:
        # separate dirty flags: new metadata dirty, coefs still clean
        assert sheet.is_metadata_dirty() is True, "autofilled metadata must be dirty (time_ranges branch)"
        assert not sheet.is_dirty, "autofilled metadata must not mark coefs dirty"

        # Run writes the absent info_devices.yaml from the autofilled values
        app = App.__new__(App)
        app._pages = {"240613_1200@i_01": sheet}
        mocker.patch.object(app, "_load_device_meta", return_value=(None, tmp_path, md_path))
        app._write_metadata()

        assert md_path.is_file(), "Run must create absent info_devices.yaml"
        data = io_info_files.read_metadata_file(md_path)
        entry = data.get("i01", {}).get("0")
        assert entry is not None, f"expected i01.0 entry, got {data!r}"
        assert str(entry[6]) == "2026-07-11T12:20:12", f"time_st wrong: {entry[6]!r}"
        assert str(entry[7]) == "2026-07-11T12:20:13", f"time_en wrong: {entry[7]!r}"

        # persisted → no longer dirty
        assert not sheet.is_metadata_dirty(), "after write metadata must be clean"
    finally:
        sheet.sh.destroy()

    # Case 2: NO time_ranges — every autofilled value is a placeholder.
    # Regression: old rule ``any(not is_placeholder(v) for v in md_list)``
    # was False here, so the metadata was reported clean even though the file
    # is absent and must be created on Run.  User must see ``*`` so any
    # prompt-before-close / branch-on-dirty works.
    sheet2, md_path2 = _make_sheet({
        "input": {
            "path": "D:/x/_raw/dummy.txt",
            "coefs": {},
        },
    })
    try:
        assert sheet2.is_metadata_dirty() is True, (
            "autofilled metadata must be dirty even with no time_ranges to seed"
        )
        assert not sheet2.is_dirty, "autofilled metadata must not mark coefs dirty (empty branch)"

        # Run still persists the (all-placeholder) stub — ``_write_metadata``
        # falls back to ``file_absent`` for not-dirty pages, but we now also
        # satisfy the dirty-flag branch.
        app2 = App.__new__(App)
        app2._pages = {"240613_1200@i_01": sheet2}
        mocker.patch.object(app2, "_load_device_meta", return_value=(None, tmp_path, md_path2))
        app2._write_metadata()

        assert md_path2.is_file(), "Run must create info_devices.yaml from empty stub too"
        assert not sheet2.is_metadata_dirty(), "after write metadata must be clean (empty branch)"
    finally:
        sheet2.sh.destroy()


def test_comma_cruise_autofilled_shows_star(_session_tk_root, tmp_path):
    """Cruise dir with comma (``@i,t-chain``) must not break autofill dirty.

    Regression for B:/Cruises/BalticSea/251201_ABP64@i,t-chain/... — the
    ``_scan`` comma-split ``re.split(r\",(?=[A-Za-z]:[\\\\/])\")`` left a
    single comma-path intact (``Path.exists()`` guard), but we also must
    verify the sheet itself marks a new ``info_devices.yaml`` dirty when the
    file is absent, even though the cruise name contains a comma.
    The GUI showed no ``*`` (neither rail nor ``metadata*`` label) because
    the old ``any(not is_placeholder…)`` left the empty stub clean and
    ``_apply_metadata_dirty_label`` was never called on load.
    """
    import tkinter as tk

    from tcm_gui.coef_sheet import ConfigSheet

    if _session_tk_root is None:
        pytest.skip("Tk not available")
        return
    root = _session_tk_root
    try:
        root.geometry("700x400+40+40")
        root.deiconify()
    except tk.TclError:
        pytest.skip("Tk not available")
        return

    cruise = tmp_path / "251201_ABP64@i,t-chain" / "inclinometer" / "_raw" / "251205_0426_st_with_t-chain"
    cruise.mkdir(parents=True)
    # No info_devices.yaml — autofilled
    raw_file = cruise / "@i_90.TXT"
    raw_file.write_text(
        "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp\n"
        "2026,07,11,13,10,34,100,200,300,400,500,600,12.5,25.0\n",
        encoding="utf-8",
    )
    cfg = {
        "input": {
            "path": str(raw_file),
            "coefs": {},
        }
    }
    # metadata_path not passed — derived from input.path via _build_metadata's
    # ``find_dir_raw_absolute`` → ``inclinometer/info_devices.yaml`` (contains comma parent)
    sheet = ConfigSheet(root)
    sheet.sh.pack(fill="both", expand=True)
    try:
        sheet.load(cfg, full=False, config_root=Config, return_enum=Return)
        root.update_idletasks()
        root.update()
        assert sheet.is_metadata_dirty() is True, "comma cruise with no info_devices must be dirty"
        assert not sheet.is_dirty, "autofilled must not mark coefs dirty"
        # Label must already be ``metadata*`` on load, not only after an edit
        meta_iid = next(i for i, m in sheet._meta.items() if m.get("is_metadata_root"))
        label = sheet.sh.item(meta_iid).get("text", "")
        assert label == "metadata*", f"expected 'metadata*' label, got {label!r}"
        # Rail side as well — App helper
        from unittest.mock import MagicMock

        from tcm_gui.app import App

        app = App.__new__(App)
        app._pages = {"251205_0426_st_with_t-chain": sheet}
        app._rail = MagicMock()
        app._poll_dirty_tabs()
        app._rail.set_dirty.assert_called_once()
        assert app._rail.set_dirty.call_args.args[1] is False, "config must not be dirty"
        assert app._rail.set_dirty.call_args.args[2] is True, "metadata must be dirty"
        # Comma-split helper must leave a single existing comma-path intact
        from tcm_gui.app import App as _App

        # Simulate _scan's comma guard — single comma-path that exists
        p = str(raw_file)
        assert "," in p and Path(p).exists()
        # The regex split would produce 1 part, not 2, so it stays single
        import re

        parts = tuple(part.strip() for part in re.split(r",(?=[A-Za-z]:[\\/])", p) if part.strip())
        # Single path → not treated as multi
        assert not (len(parts) > 1 and all(Path(x).is_absolute() for x in parts)), (
            f"single comma-path must not be split as multi, got {parts!r}"
        )
    finally:
        sheet.sh.destroy()
