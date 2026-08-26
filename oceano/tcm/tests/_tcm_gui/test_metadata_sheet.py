"""Style: test_coef_sheet.py, test_gui_actions.py (gui_project fixture).

Reference: docs/reference/config_reference.md (metadata section)
Guides: docs/project_developer_guide/GUI.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from tcm import cli, format
from tcm._constants import RAW_DIR_NAME
from tcm.schema import Config, Return


def _write_info_devices(cruise: Path):
    (cruise / "info_devices.yaml").write_text(
        '# Instrument_ID: [Point, Sea_depth, H_above_bot, Symbol, Lat, Lon, Time_st, Time_en, Burst_dt, Bursts_t, Comment]\n'
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
        iid = f"iid_{kw.get('text','x')}_{len(_items2)}"
        parent = kw.get("parent") or ""
        _kids2.setdefault(parent, []).append(iid)
        _items2[iid] = {"values": kw.get("values", ()), "text": kw.get("text","")}
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
    cfg = {"input": {"path": str(tmp_path / "dummy.txt"), "time_ranges": ["2026-07-11T12:00:00", "2026-07-20T12:00:00"], "coefs": {}}}
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
                    sid = str(stems_by_pcid.get(pcid, [stem]).index(stem)) if stem in stems_by_pcid.get(pcid, []) else "0"
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
        iid = f"iid_{kw.get('text','x')}_{len(_items)}"
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
