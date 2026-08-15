"""Unit tests for coef_sheet — ConfigSheet with mocked tksheet.Sheet.

Tests path tracking, _cell_spec_for, _apply_styles (checkbox/dropdown/align),
and const.tk_color_to_hex without requiring a real Tk event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import tcm_gui.coef_sheet as coef_sheet
import tcm_gui.theme as theme
from tkinter import ttk

from tcm_gui._cell_spec import BOOL_SPEC, NUMBER_SPEC, TEXT_SPEC

# ── lightweight fixtures ────────────────────────────────────────────────────


class _FakeReturn(Enum):
    END = "<end>"
    CFG = "<cfg_from_args>"


@dataclass
class _FakeProgram:
    return_: str = "<end>"
    b_interact: bool = False
    verbose: str = "INFO"


@dataclass
class _FakeInput:
    path: str | None = None
    dt_from_utc: int | None = 0


@dataclass
class _FakeConfig:
    input: _FakeInput = field(default_factory=_FakeInput)
    program: _FakeProgram = field(default_factory=_FakeProgram)


# ── const.tk_color_to_hex ────────────────────────────────────────────────────


class TestResolveBg:
    def test_hex_passthrough(self):
        """Already-hex color passes through unchanged."""
        from tcm_gui.theme import tk_color_to_hex

        mock_widget = MagicMock()
        mock_widget.winfo_rgb.return_value = (0xF0 * 257, 0xF0 * 257, 0xF0 * 257)
        assert tk_color_to_hex(mock_widget, "#F0F0F0") == "#f0f0f0"

    def test_system_color_converted(self):
        """'SystemButtonFace' → hex via winfo_rgb."""
        from tcm_gui.theme import tk_color_to_hex

        mock_widget = MagicMock()
        # Windows SystemButtonFace ≈ #F0F0F0 → (0xF0*257, 0xF0*257, 0xF0*257)
        mock_widget.winfo_rgb.return_value = (61680, 61680, 61680)
        result = tk_color_to_hex(mock_widget, "SystemButtonFace")
        assert result.startswith("#"), f"expected hex, got {result}"
        assert len(result) == 7, f"expected #rrggbb, got {result}"

    def test_invalid_color_fallback(self):
        """TclError → return original string."""
        from tkinter import TclError

        from tcm_gui.theme import tk_color_to_hex

        mock_widget = MagicMock()
        mock_widget.winfo_rgb.side_effect = TclError("bad color")
        assert tk_color_to_hex(mock_widget, "invalid") == "invalid"


# ── _cell_spec_for ──────────────────────────────────────────────────────────


class TestCellSpecFor:
    @staticmethod
    def _make_sheet():
        """Create a ConfigSheet with a mocked Sheet — no Tk needed."""
        from tcm_gui.coef_sheet import ConfigSheet

        with patch.object(coef_sheet, "Sheet"):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = MagicMock()
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = {}
        cs._config_root = _FakeConfig
        cs._return_enum = _FakeReturn
        cs._snap = ({}, {}, "")
        return cs

    def test_coef_type_returns_number(self):
        """Rows with coef meta types (2d, 1d, 1d_flat, scalar) → NUMBER_SPEC."""
        cs = self._make_sheet()
        for coef_type in ("2d", "1d", "1d_flat", "scalar"):
            m = {"type": coef_type, "path": "input.coefs.Ag"}
            result = cs._cell_spec_for("iid", m, 1)
            assert result is NUMBER_SPEC, f"type={coef_type}: expected NUMBER_SPEC, got {result}"

    def test_bool_path_returns_bool(self):
        """Path resolving to bool → BOOL_SPEC."""
        cs = self._make_sheet()
        m = {"path": "program.b_interact"}
        result = cs._cell_spec_for("iid", m, 1)
        assert result is BOOL_SPEC, f"expected BOOL_SPEC, got {result}"

    def test_str_path_returns_text(self):
        """Path resolving to str → TEXT_SPEC."""
        cs = self._make_sheet()
        m = {"path": "program.verbose"}
        result = cs._cell_spec_for("iid", m, 1)
        assert result is TEXT_SPEC, f"expected TEXT_SPEC, got {result}"

    def test_int_path_returns_number(self):
        """Path resolving to int → NUMBER_SPEC."""
        cs = self._make_sheet()
        m = {"path": "input.dt_from_utc"}
        result = cs._cell_spec_for("iid", m, 1)
        assert result is NUMBER_SPEC, f"expected NUMBER_SPEC, got {result}"

    def test_enum_override_for_return(self):
        """program.return_ → CellSpec("enum", _FakeReturn) via override."""
        cs = self._make_sheet()
        m = {"path": "program.return_"}
        result = cs._cell_spec_for("iid", m, 1)
        assert result.kind == "enum", f"expected enum, got {result.kind}"
        assert result.enum is _FakeReturn

    def test_unknown_path_returns_text(self):
        """Path not in dataclass → TEXT_SPEC (safe fallback)."""
        cs = self._make_sheet()
        m = {"path": "nonexistent.field"}
        result = cs._cell_spec_for("iid", m, 1)
        assert result is TEXT_SPEC, f"expected TEXT_SPEC, got {result}"


# ── path tracking in _ins ───────────────────────────────────────────────────


class TestInsPathTracking:
    @staticmethod
    def _make_sheet():
        from tcm_gui.coef_sheet import ConfigSheet

        with patch.object(coef_sheet, "Sheet"):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = MagicMock()
        # Make insert return incrementing iids
        cs.sh.insert.side_effect = lambda **kw: f"iid_{kw.get('text', 'x')}"
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = {}
        cs._config_root = None
        cs._return_enum = None
        cs._snap = ({}, {}, "")
        return cs

    def test_root_node_path(self):
        """Root node (no parent) → path = node text."""
        cs = self._make_sheet()
        cs._ins("", "input", ["v1", "v2", "v3", "v4", "v5", "v6"], "")
        iid = "iid_input"
        assert cs._meta[iid]["path"] == "input"

    def test_child_node_path(self):
        """Child node → path = parent.path + '.' + text."""
        cs = self._make_sheet()
        parent = cs._ins("", "program", [""] * 6, "")
        child = cs._ins(parent, "verbose", ["INFO"] + [""] * 5, "", meta={"max_col": 1})
        assert cs._meta[child]["path"] == "program.verbose"

    def test_nested_path(self):
        """Three levels deep → correct dotted path."""
        cs = self._make_sheet()
        root = cs._ins("", "input", [""] * 6, "")
        mid = cs._ins(root, "coefs", [""] * 6, "")
        leaf = cs._ins(mid, "Ag", ["0.001"] + [""] * 5, "", meta={"max_col": 1})
        assert cs._meta[leaf]["path"] == "input.coefs.Ag"

    def test_array_child_path(self):
        """Array child like Ag[0] → path includes index."""
        cs = self._make_sheet()
        root = cs._ins("", "input", [""] * 6, "")
        mid = cs._ins(root, "coefs", [""] * 6, "")
        child = cs._ins(mid, "Ag[0]", ["0.001", "0", "0"] + [""] * 3, "", meta={"max_col": 3})
        assert cs._meta[child]["path"] == "input.coefs.Ag[0]"


# ── _clear_cell_widgets ─────────────────────────────────────────────────────


class TestClearCellWidgets:
    def test_deletes_dropdown_and_checkbox(self):
        """Both delete_dropdown and delete_checkbox are called."""
        from tcm_gui.coef_sheet import ConfigSheet

        mock_sh = MagicMock()
        ConfigSheet._clear_cell_widgets(mock_sh, 0, 1)
        mock_sh.delete_dropdown.assert_called_once_with(0, 1)
        mock_sh.delete_checkbox.assert_called_once_with(0, 1)

    def test_tolerates_missing_methods(self):
        """No crash if Sheet doesn't have delete methods."""
        from tcm_gui.coef_sheet import ConfigSheet

        mock_sh = MagicMock()
        mock_sh.delete_dropdown.side_effect = AttributeError
        mock_sh.delete_checkbox.side_effect = AttributeError
        # Should not raise
        ConfigSheet._clear_cell_widgets(mock_sh, 0, 1)


# ── _apply_styles integration (mocked Sheet) ───────────────────────────────


class TestApplyStyles:
    @staticmethod
    def _make_loaded_sheet(cfg: dict | None = None):
        """Load a minimal config into ConfigSheet with fully mocked Sheet.

        Wires ``sh.get_children`` from recorded inserts and opens every node so
        ``_apply_styles`` styles all rows (the implementation skips rows whose
        ``_row_map`` lookup returns None — i.e. collapsed or unreachable).
        """
        from tcm_gui.coef_sheet import ConfigSheet

        if cfg is None:
            cfg = {"input": {"path": "/data", "coefs": {"Ag": [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]}}}

        mock_sh = MagicMock()
        mock_sh.total_columns.return_value = 6
        mock_sh.total_rows.return_value = 0
        _kids: dict[Any, list[str]] = {}
        _items: dict[str, dict] = {}

        def _insert(**kw):
            iid = f"iid_{kw.get('text', 'x')}"
            _kids.setdefault(kw.get("parent") or "", []).append(iid)
            _items[iid] = {"values": kw.get("values", ())}
            return iid

        mock_sh.insert.side_effect = _insert
        mock_sh.item.side_effect = lambda iid, **_kw: _items.get(iid, {})
        mock_sh.get_children.side_effect = lambda parent="": list(_kids.get(parent or "", ()))
        mock_sh.get_cell_data.return_value = ""
        mock_sh.tag_names.return_value = []
        mock_sh.winfo_rgb.return_value = (61680, 61680, 61680)  # #F0F0F0

        with patch.object(coef_sheet, "Sheet", return_value=mock_sh):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg
        from tcm.schema import Config, Return

        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ({}, {}, "")
        cs._fg_default = "#000000"
        # __new__ bypasses __init__ — set row-cache fields manually
        cs._int_row_of = {}
        cs._vis = ()
        cs._col_resize = MagicMock()
        cs.on_edit_begin = None

        cs._build_coefs(cfg)
        # Open every constructed node so all rows receive styling.
        for m in cs._meta.values():
            m["open"] = True
        return cs, mock_sh

    @patch.object(theme, "tk_color_to_hex", return_value="#F0F0F0")
    def test_node_column_gets_bg(self, mock_resolve):
        """Index canvas (tree column) gets bg: global option + per-cell highlight."""
        cs, mock_sh = self._make_loaded_sheet()
        cs._apply_styles()

        # Global fallback (newer builds / later-expanded rows)
        mock_sh.set_options.assert_any_call(index_background="#F0F0F0")
        # Per-cell highlight on the index canvas (7.6.x draw path)
        index_calls = [c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("canvas") == "index"]
        assert len(index_calls) > 0, (
            f"expected highlight_cells(canvas='index') calls, got: {mock_sh.highlight_cells.call_args_list}"
        )

    @patch.object(theme, "tk_color_to_hex", return_value="#F0F0F0")
    def test_coef_cells_right_aligned(self, mock_resolve):
        """Coef data cells are right-aligned (number type)."""
        cs, mock_sh = self._make_loaded_sheet()
        cs._apply_styles()

        # align_cells is called positionally: align_cells(r, c, align="e", redraw=False)
        # OR align_cells(r, c, "e", redraw=False) — check both patterns
        right_align_calls = [
            c
            for c in mock_sh.align_cells.call_args_list
            if c.kwargs.get("align") == "e" or (len(c.args) >= 3 and c.args[2] == "e")
        ]
        assert len(right_align_calls) > 0, (
            f"expected right-aligned coef cells, got calls: {mock_sh.align_cells.call_args_list}"
        )

    @patch.object(theme, "tk_color_to_hex", return_value="#F0F0F0")
    def test_date_cells_right_aligned(self, mock_resolve):
        """Date cells (coefs parent with has_date) are right-aligned (``e``).

        See :func:`_apply_styles` section 3 and the CellSpec.kind table in
        how_gui_works.md — ``"date"`` is right-aligned, not left.
        """
        cs, mock_sh = self._make_loaded_sheet()
        cs._apply_styles()

        right_align_calls = [
            c
            for c in mock_sh.align_cells.call_args_list
            if c.kwargs.get("align") == "e" or (len(c.args) >= 3 and c.args[2] == "e")
        ]
        # coefs node has has_date=True → date column should be right-aligned
        assert len(right_align_calls) > 0, "expected right-aligned date cells"

    @patch.object(theme, "tk_color_to_hex", return_value="#F0F0F0")
    def test_header_highlighted(self, mock_resolve):
        """Node labels styled with BLUE_FG via highlight_cells on index canvas."""
        from tcm_gui.cli_cfg import default_for_path

        ag_default = default_for_path("input.coefs.Ag")
        cfg = {"input": {"path": "/data", "coefs": {"Ag": ag_default}}}
        cs, mock_sh = self._make_loaded_sheet(cfg)
        cs._apply_styles()

        # Node labels use BLUE_FG on index canvas for at-default nodes
        index_calls = [c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("canvas") == "index"]
        blue_calls = [c for c in index_calls if c.kwargs.get("fg") == "#0055CC"]
        assert len(blue_calls) > 0, f"expected BLUE_FG on index canvas, got: {index_calls}"


# ── coefs_path as child row of input ────────────────────────────────────────


class TestCoefsPathChildRow:
    """``coefs_path`` must appear as a separate child row under ``input``,
    not as a second column in the ``input`` node row.

    Regression: putting it in ``values[1]`` of the ``input`` node made the
    value invisible to the user (the ``input`` row shows only ``path`` in
    column 0; the second column is not visually prominent).
    """

    @staticmethod
    def _make_loaded_sheet(cfg: dict | None = None):
        """Load a config with ``coefs_path`` into ConfigSheet (non-full mode)."""
        from tcm_gui.coef_sheet import ConfigSheet

        if cfg is None:
            cfg = {
                "input": {
                    "path": "/data",
                    "coefs_path": "/coefs/calibration.h5",
                    "coefs": {"Ag": [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]},
                }
            }

        mock_sh = MagicMock()
        mock_sh.total_columns.return_value = 6
        mock_sh.total_rows.return_value = 0
        _kids: dict[Any, list[str]] = {}

        def _insert(**kw):
            iid = f"iid_{kw.get('text', 'x')}"
            _kids.setdefault(kw.get("parent") or "", []).append(iid)
            return iid

        mock_sh.insert.side_effect = _insert
        mock_sh.get_children.side_effect = lambda parent="": list(_kids.get(parent or "", ()))
        mock_sh.get_cell_data.return_value = ""
        mock_sh.tag_names.return_value = []
        mock_sh.winfo_rgb.return_value = (61680, 61680, 61680)

        with patch.object(coef_sheet, "Sheet", return_value=mock_sh):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg
        from tcm.schema import Config, Return

        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ({}, {}, "")
        cs._fg_default = "#000000"
        # __new__ bypasses __init__ — set row-cache + hover fields manually
        cs._int_row_of = {}
        cs._vis = ()
        cs._col_resize = MagicMock()
        cs._field_iid = None
        cs._field_pending = None
        cs._field_show_job = None
        cs._field_hide_job = None
        cs._hover_field = None
        cs._hover_btn = None
        cs._iid_of_row = {}
        cs.on_hover_status = None
        cs.on_edit_begin = None
        cs.hover_status = {}
        cs._status_iid = None
        cs._status_hint = ""

        cs._build_coefs(cfg)
        for m in cs._meta.values():
            m["open"] = True
        return cs, mock_sh

    def test_coefs_path_is_meta_entry(self):
        """``coefs_path`` exists as a row in _meta with its own iid."""
        cs, _ = self._make_loaded_sheet()
        coefs_path_iids = [iid for iid, m in cs._meta.items() if m.get("key") == "coefs_path"]
        assert len(coefs_path_iids) == 1, (
            f"expected exactly one 'coefs_path' entry in _meta, found {len(coefs_path_iids)}: "
            f"{[cs._meta[i].get('key') for i in coefs_path_iids]}"
        )

    def test_coefs_path_is_child_of_input(self):
        """``coefs_path`` iid's parent is the ``input`` node iid."""
        cs, _ = self._make_loaded_sheet()
        input_iid = next(iid for iid, m in cs._meta.items() if m.get("type") == "input")
        coefs_path_iid = next(iid for iid, m in cs._meta.items() if m.get("key") == "coefs_path")
        assert cs._meta[coefs_path_iid].get("parent") == input_iid, (
            f"coefs_path parent should be input iid={input_iid!r}, "
            f"got {cs._meta[coefs_path_iid].get('parent')!r}"
        )

    def test_coefs_path_value_displayed(self):
        """``coefs_path`` row shows the config value in column 0."""
        _, mock_sh = self._make_loaded_sheet()
        insert_calls = [c for c in mock_sh.insert.call_args_list if c.kwargs.get("text") == "coefs_path"]
        assert len(insert_calls) == 1, f"expected 1 insert for coefs_path, got {len(insert_calls)}"
        values = insert_calls[0].kwargs.get("values")
        assert values and values[0] == "/coefs/calibration.h5", (
            f"coefs_path values[0] should be '/coefs/calibration.h5', got {values!r}"
        )

    def test_input_node_max_col_unchanged(self):
        """``input`` node keeps ``max_col=1`` (only path in column 0)."""
        cs, _ = self._make_loaded_sheet()
        input_iid = next(iid for iid, m in cs._meta.items() if m.get("type") == "input")
        assert cs._meta[input_iid].get("max_col") == 1, (
            f"input node max_col should be 1 (not 2), got {cs._meta[input_iid].get('max_col')}"
        )

    def test_input_values_unchanged(self):
        """``input`` node row shows only ``path`` in column 0."""
        _, mock_sh = self._make_loaded_sheet()
        input_insert = [c for c in mock_sh.insert.call_args_list if c.kwargs.get("text") == "input"]
        assert len(input_insert) == 1
        values = input_insert[0].kwargs.get("values")
        assert values[0] == "/data", f"input values[0] should be '/data', got {values[0]!r}"
        assert all(v == "" for v in values[1:]), f"input values[1:] should all be empty, got {values[1:]!r}"

    def test_input_node_has_browse_flag(self):
        """``input`` node meta carries ``browse: True`` for the browse button."""
        cs, _ = self._make_loaded_sheet()
        input_iid = next(iid for iid, m in cs._meta.items() if m.get("type") == "input")
        assert cs._meta[input_iid].get("browse") is True, (
            f"input node should have browse=True, got {cs._meta[input_iid].get('browse')!r}"
        )

    def test_coefs_path_has_browse_flag(self):
        """``coefs_path`` child meta carries ``browse: True`` for the browse button."""
        cs, _ = self._make_loaded_sheet()
        coefs_path_iid = next(iid for iid, m in cs._meta.items() if m.get("key") == "coefs_path")
        assert cs._meta[coefs_path_iid].get("browse") is True, (
            f"coefs_path should have browse=True, got {cs._meta[coefs_path_iid].get('browse')!r}"
        )

    def test_non_path_rows_lack_browse_flag(self):
        """Rows like ``Ag``, ``coefs`` must NOT have ``browse: True``."""
        cs, _ = self._make_loaded_sheet()
        browse_keys = [m.get("key") or m.get("path") for m in cs._meta.values() if m.get("browse")]
        assert sorted(browse_keys) == sorted(["input", "coefs_path"]), (
            f"only input and coefs_path should have browse flag, got {browse_keys!r}"
        )


# ── browse button lifecycle ─────────────────────────────────────────────────


class TestBrowseButtonLifecycle:
    """Browse button lifecycle via BrowseOverlay delegation.

    ``BrowseButtonManager`` owns a ``BrowseOverlay`` (``self._ov``).
    The overlay owns the button widget (``self._ov._button``).
    """

    @staticmethod
    def _make_manager():
        """Create a BrowseButtonManager with a mock sheet."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        return BrowseButtonManager(mock_sh, MagicMock()), mock_sh

    def test_detach_destroys_button(self):
        """``detach()`` must destroy the button via ``_ov.hide()``."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_editor = MagicMock()
        mock_editor.winfo_height.return_value = 20
        mock_sh.get_text_editor_widget.return_value = None

        mock_btn = MagicMock()
        with patch.object(ttk, "Button", return_value=mock_btn):
            mgr = BrowseButtonManager(mock_sh, MagicMock())
            mgr.attach(0, 0)
            mock_sh.get_text_editor_widget.return_value = mock_editor
            mgr._acquire_and_place()

        assert mgr._ov.visible, "overlay button should be visible after acquire"

        mgr.detach()

        mock_btn.destroy.assert_called_once(), "detach() must destroy the button widget"
        assert not mgr._ov.visible, "overlay must be hidden after detach"

    def test_detach_no_button_is_safe(self):
        """Calling ``detach()`` when nothing is active must not raise."""
        mgr, _ = self._make_manager()
        mgr.detach()
        assert not mgr._ov.visible

    def test_attach_cancels_stale_retry(self):
        """A second ``attach()`` must cancel the pending retry from the first."""
        mgr, mock_sh = self._make_manager()
        mgr.attach(0, 0)
        first_job = mgr._retry_job
        assert first_job is not None

        mgr.attach(0, 0)
        mock_sh.after_cancel.assert_any_call(first_job)
        assert mgr._retry_job != first_job or mgr._retry_job is not None

    def test_button_not_created_without_attach(self):
        """No button widget must exist before ``attach()`` + retry fires."""
        mgr, _ = self._make_manager()
        assert not mgr._ov.visible, "button must not exist before attach"

    def test_end_edit_detaches_unconditionally(self):
        """``_on_end_edit_cell`` must call ``mgr.detach()`` for ANY row,
        not just coefs_path — prevents ghost buttons on non-path rows."""

        cs, mock_sh = TestCoefsPathChildRow._make_loaded_sheet()
        cs._mgr = MagicMock()
        ag_iid = next(iid for iid, m in cs._meta.items() if m.get("type") == "2d")
        cs._iid_at_row = MagicMock(return_value=ag_iid)
        mock_sh.get_cell_data.return_value = ""

        event = MagicMock()
        event.row = 5
        event.column = 0
        event.value = "1.0"
        cs._on_end_edit_cell(event)

        (
            cs._mgr.detach.assert_called_once(),
            ("detach() must be called for non-path rows too (prevents ghost button)"),
        )

    def test_begin_edit_attaches_only_for_browse_rows(self):
        """``_on_begin_edit_cell`` must call ``attach()`` only for rows
        with ``browse: True`` in meta — never for numeric/coef rows.

        It must still call ``detach()`` unconditionally (prevents ghost
        buttons from a previous browse-row edit).
        """

        cs, mock_sh = TestCoefsPathChildRow._make_loaded_sheet()
        cs._mgr = MagicMock()

        ag_iid = next(iid for iid, m in cs._meta.items() if m.get("type") == "2d")
        cs._iid_at_row = MagicMock(return_value=ag_iid)
        mock_sh.get_cell_data.return_value = ""

        event = MagicMock()
        event.row = 10
        event.column = 0
        cs._on_begin_edit_cell(event)

        cs._mgr.attach.assert_not_called(), ("attach() must NOT be called for non-browse rows like Ag")
        (
            cs._mgr.detach.assert_called_once(),
            ("detach() must be called for ALL rows (cleans up previous button)"),
        )

    def test_begin_edit_hides_hover_field(self):
        """``_on_begin_edit_cell`` must hide the hovered PathField (edit takes over)."""
        cs, mock_sh = TestCoefsPathChildRow._make_loaded_sheet()
        cs._mgr = MagicMock()

        input_iid = next(iid for iid, m in cs._meta.items() if m.get("type") == "input")
        cs._iid_at_row = MagicMock(return_value=input_iid)
        mock_sh.get_cell_data.return_value = ""

        cs._hide_hover_field = MagicMock()
        event = MagicMock()
        event.row = 0
        event.column = 0
        cs._on_begin_edit_cell(event)

        cs._hide_hover_field.assert_called_once(), ("hover field must be hidden when edit begins (handoff)")

    def test_f3_stale_retry_cancelled(self):
        """F3 — ``attach()`` cancels any pending retry from a previous cycle.

        Regression: L1 — without cancellation, a stale retry would land on
        whatever editor is current (possibly a non-path row).
        """
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_sh.get_text_editor_widget.return_value = None

        with patch.object(ttk, "Button"):
            mgr = BrowseButtonManager(mock_sh, MagicMock())
            mgr.attach(0, 0)
            first_job = mgr._retry_job
            assert first_job is not None, "precondition: retry scheduled"

            mgr.attach(1, 0)

        (
            mock_sh.after_cancel.assert_any_call(first_job),
            ("attach() must cancel the previous retry (F3 — prevents L1)"),
        )

    def test_f3_attach_resets_existing_overlay(self):
        """F3 — ``attach()`` must hide a live overlay from a previous cycle
        before starting a new retry.  Without this, buttons accumulate.

        Regression: L3 — ``attach`` ≠ reset.
        """
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_sh.get_text_editor_widget.return_value = None

        mock_btn = MagicMock()
        with patch.object(ttk, "Button", return_value=mock_btn):
            mgr = BrowseButtonManager(mock_sh, MagicMock())

            mock_editor_1 = MagicMock()
            mock_editor_1.winfo_height.return_value = 20
            mgr.attach(0, 0)
            mock_sh.get_text_editor_widget.return_value = mock_editor_1
            mgr._acquire_and_place()
            assert mgr._ov.visible, "precondition: first overlay visible"

            mock_sh.get_text_editor_widget.return_value = None
            mgr.attach(1, 0)

        (
            mock_btn.destroy.assert_called_once(),
            ("attach() must destroy the previous button (F3 — idempotent reset)"),
        )

    def test_write_cell_targets_col0(self):
        """``_write_cell`` must write to column 0 and call the restyler."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_sh.get_text_editor_widget.return_value = None

        restyler = MagicMock()
        mgr = BrowseButtonManager(mock_sh, MagicMock(), on_edit_restyler=restyler)
        mgr.attach(5, 2, iid="test_iid")

        # Simulate write_cell directly
        mgr._write_cell("/selected/path")

        (
            mock_sh.set_cell_data.assert_called_once_with(5, 0, "/selected/path"),
            ("write_cell must write to column 0 regardless of edit column"),
        )
        (
            restyler.assert_called_once_with("test_iid", 0, "/selected/path"),
            ("restyler must be called with (iid, 0, text)"),
        )
        (
            mock_sh.after_idle.assert_called_once_with(mock_sh.close_text_editor),
            ("editor must be closed after write"),
        )

    def test_read_cell_reads_col0(self):
        """``_read_cell`` must read column 0 of the target row."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_sh.get_cell_data.return_value = "/data/coefs"

        mgr = BrowseButtonManager(mock_sh, MagicMock())
        mgr.attach(3, 1)  # editing col 1

        result = mgr._read_cell()

        (
            mock_sh.get_cell_data.assert_called_with(3, 0),
            ("read_cell must read column 0 regardless of edit column"),
        )
        assert result == "/data/coefs"


# ── hover browse button status hints ────────────────────────────────────────


class TestHoverBtnStatusHints:
    """The hover browse button shows button-specific hints, distinct from
    the row hover text — mirroring the top PathField's button scheme.

    Regression: ``_show_hover_field`` used to give the coefs button the
    row's own ``_coefs_status_hint`` callable → hovering the button was
    visually a no-op ("like no button at all").
    """

    @staticmethod
    def _sheet_with_hover():
        """Loaded sheet with mocked floated field + hover button."""
        cs, _ = TestCoefsPathChildRow._make_loaded_sheet()
        cs._hover_field = MagicMock()
        cs._hover_field._editing = False
        cs._hover_btn = MagicMock()
        cs.on_hover_status = MagicMock()
        return cs

    def test_coefs_btn_gets_button_hints(self):
        """coefs_path row → dir/file button hints from str.yaml (not row text)."""
        from tcm_gui._i18n import STRINGS

        cs = self._sheet_with_hover()
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs_path")
        cs._show_hover_field(iid, 1, 10)
        assert cs._hover_btn._status_hint == STRINGS["browse_btn.status"]
        assert cs._hover_btn._status_hint_files == STRINGS["browse_btn.status_files"]

    def test_input_btn_resets_files_hint(self):
        """input row → static hint; stale coefs ``_status_hint_files`` reset
        (singleton button — state leaks across rows otherwise)."""
        cs = self._sheet_with_hover()
        cs._hover_btn._status_hint_files = "stale"
        iid = next(i for i, m in cs._meta.items() if m.get("type") == "input")
        cs._show_hover_field(iid, 0, 10)
        assert cs._hover_btn._status_hint == cs._status_hint
        assert cs._hover_btn._status_hint_files == ""

    def test_shift_toggle_yields_to_hovered_button(self):
        """Shift pressed while pointer is ON the button → ``_on_shift_toggle``
        must NOT publish the row text: the button's poll re-publishes its own
        hint on the transition; the row text would overwrite it.

        Regression: button hint appeared for <80 ms, then the row message
        returned although the pointer never moved.
        """
        cs = self._sheet_with_hover()
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs_path")
        cs._status_iid = iid
        cs._hover_btn._hovered = True
        cs._on_shift_toggle(None)
        cs.on_hover_status.assert_not_called()

    def test_shift_toggle_publishes_when_btn_not_hovered(self):
        """Shift pressed with pointer on the row (button not hovered) →
        row text is re-published as before."""
        cs = self._sheet_with_hover()
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs_path")
        cs._status_iid = iid
        cs._hover_btn._hovered = False
        with patch.object(cs, "_pointer_in_field", return_value=True):
            cs._on_shift_toggle(None)
        assert cs.on_hover_status.call_count == 1
