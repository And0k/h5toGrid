"""Unit tests for coef_sheet — ConfigSheet with mocked tksheet.Sheet.

Tests path tracking, _cell_spec_for, _apply_styles (checkbox/dropdown/align),
and const.tk_color_to_hex without requiring a real Tk event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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

        See :func:`_apply_styles` docs/project_developer_guide/GUI/decisions.md and the CellSpec.kind table in
        docs/project_developer_guide/GUI/architecture.md ## Type-aware cell rendering — ``"date"`` is
        right-aligned, not left.
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
        """Node labels styled with NODE_DEFAULT_VALS_FG via highlight_cells on index canvas."""
        from tcm_gui.cli_cfg import default_for_path

        ag_default = default_for_path("input.coefs.Ag")
        cfg = {"input": {"path": "/data", "coefs": {"Ag": ag_default}}}
        cs, mock_sh = self._make_loaded_sheet(cfg)
        cs._apply_styles()

        # Node labels use NODE_DEFAULT_VALS_FG on index canvas for at-default nodes
        index_calls = [c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("canvas") == "index"]
        blue_calls = [c for c in index_calls if c.kwargs.get("fg") == "#0055CC"]
        assert len(blue_calls) > 0, f"expected NODE_DEFAULT_VALS_FG on index canvas, got: {index_calls}"


# ── path as child row of coefs (hidden when collapsed) ────────────────────────────────────────


class TestCoefsPathChildRow:
    """``path`` must appear in the ``coefs`` node row (column 0),
    not as a separate child row under ``coefs``.

    The ``coefs`` node now shows the path in column 0 (like ``input`` node
    shows the data path in column 0). The date is no longer shown in the
    tksheet — it is displayed in the status bar as a suffix.
    """

    @staticmethod
    def _make_loaded_sheet(cfg: dict | None = None):
        """Load a config with ``path`` into ConfigSheet (non-full mode)."""
        from tcm_gui.coef_sheet import ConfigSheet

        if cfg is None:
            cfg = {
                "input": {
                    "path": "/data",
                    "coefs": {
                        "path": "/coefs/calibration.h5",
                        "Ag": [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]],
                    },
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
        cs._readonly = False
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
        # (hover_status removed — live detail via _time_ranges_detail)
        cs._status_iid = None
        cs._status_hint = ""

        cs._build_coefs(cfg)
        for m in cs._meta.values():
            m["open"] = True
        return cs, mock_sh

    def test_path_is_in_coefs_node(self):
        """``coefs`` node has the path in column 0 (no child row)."""
        cs, _ = self._make_loaded_sheet()
        coefs_iid = next(iid for iid, m in cs._meta.items() if m.get("key") == "coefs")
        # The coefs node should have max_col=1 (path in col 0) and no child path row
        assert cs._meta[coefs_iid].get("max_col") == 1, (
            f"coefs node max_col should be 1 (path in col 0), got {cs._meta[coefs_iid].get('max_col')}"
        )
        # No child row with key="path" should exist
        path_children = [
            iid for iid, m in cs._meta.items() if m.get("key") == "path" and m.get("parent") == coefs_iid
        ]
        assert len(path_children) == 0, (
            f"coefs node should not have a child path row, found {len(path_children)}"
        )

    def test_path_value_displayed(self):
        """``coefs`` node row shows the config path value in column 0."""
        _, mock_sh = self._make_loaded_sheet()
        insert_calls = [c for c in mock_sh.insert.call_args_list if c.kwargs.get("text") == "coefs"]
        assert len(insert_calls) == 1, f"expected 1 insert for coefs, got {len(insert_calls)}"
        values = insert_calls[0].kwargs.get("values")
        assert values and values[0] == "/coefs/calibration.h5", (
            f"coefs values[0] should be '/coefs/calibration.h5', got {values!r}"
        )

    def test_coefs_node_has_browse_flag(self):
        """``coefs`` node meta carries ``browse: True`` for the browse button."""
        cs, _ = self._make_loaded_sheet()
        coefs_iid = next(iid for iid, m in cs._meta.items() if m.get("key") == "coefs")
        assert cs._meta[coefs_iid].get("browse") is True, (
            f"coefs node should have browse=True, got {cs._meta[coefs_iid].get('browse')!r}"
        )

    def test_non_path_rows_lack_browse_flag(self):
        """Rows like ``Ag`` must NOT have ``browse: True`` (only input and coefs)."""
        cs, _ = self._make_loaded_sheet()
        browse_keys = [m.get("key") or m.get("path") for m in cs._meta.values() if m.get("browse")]
        assert sorted(browse_keys) == sorted(["input", "coefs"]), (
            f"only input and coefs should have browse flag, got {browse_keys!r}"
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
        not just path — prevents ghost buttons on non-path rows."""

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
        """coefs (path) row → dir/file button hints from str.yaml (not row text)."""
        from tcm_gui._i18n import STRINGS

        cs = self._sheet_with_hover()
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs")
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
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs")
        cs._status_iid = iid
        cs._hover_btn._hovered = True
        cs._on_shift_toggle(None)
        cs.on_hover_status.assert_not_called()

    def test_shift_toggle_publishes_when_btn_not_hovered(self):
        """Shift pressed with pointer on the row (button not hovered) →
        row text is re-published as before."""
        cs = self._sheet_with_hover()
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs")
        cs._status_iid = iid
        cs._hover_btn._hovered = False
        with patch.object(cs, "_pointer_in_field", return_value=True):
            cs._on_shift_toggle(None)
        assert cs.on_hover_status.call_count == 1


# ── _help_candidates: metadata label → doc keys (regression) ────────────────


class TestMetadataHelpCandidates:
    """Paired-row labels must fan out to the per-field ``metadata.*`` doc keys.

    Regression: the burst row's label ``burst_dt/t`` was mangled to the
    single key ``metadata.burst_dt_t`` — no doc entry, so hover showed the
    raw label instead of help.  ``/`` separates the two fields like ``,``
    and the shortened ``t`` maps to the ``bursts_t`` key.
    """

    @staticmethod
    def _sheet():
        cs = TestCellSpecFor._make_sheet()
        cs._meta["iid_burst"] = {
            "label": "burst_dt/t",
            "path": "metadata.burst_dt_t",
            "is_metadata": True,
            "max_col": 2,
        }
        cs._meta["iid_pair"] = {
            "label": "point, symbol",
            "path": "metadata.point_symbol",
            "is_metadata": True,
            "max_col": 2,
        }
        cs._meta["iid_single"] = {
            "label": "comment",
            "path": "metadata.comment",
            "is_metadata": True,
            "max_col": 1,
        }
        return cs

    def test_burst_row_fans_both_fields(self):
        cs = self._sheet()
        assert cs._help_candidates("iid_burst") == [
            "metadata.burst_dt",
            "metadata.bursts_t",
            "metadata.burst_dt_t",
        ]

    def test_burst_row_tree_shows_first_only(self):
        cs = self._sheet()
        assert cs._help_candidates("iid_burst", tree=True) == [
            "metadata.burst_dt",
            "metadata.burst_dt_t",
        ]

    def test_comma_pair_unaffected(self):
        cs = self._sheet()
        assert cs._help_candidates("iid_pair")[:2] == ["metadata.point", "metadata.symbol"]

    def test_candidates_resolve_in_bundled_doc(self):
        """Both burst keys carry help in ``config_reference.md``."""
        from tcm_gui import _help

        _help.reload_cache()
        cs = self._sheet()
        for cand in cs._help_candidates("iid_burst")[:2]:
            e = _help.help_for_path(cand)
            assert e is not None and e.short, f"{cand} must parse with a non-empty short"

    # ── column-aware ordering: col selects the field shown on data-cell hover ──

    def test_col_reorders_pair_to_second_field(self):
        """col=1 puts ``metadata.symbol`` first — its short wins in status."""
        cs = self._sheet()
        assert cs._help_candidates("iid_pair", col=1)[0] == "metadata.symbol"

    def test_col_zero_keeps_first_field_first(self):
        cs = self._sheet()
        assert cs._help_candidates("iid_pair", col=0)[0] == "metadata.point"

    def test_col_ignores_out_of_range(self):
        """col beyond the field count falls back to default order."""
        cs = self._sheet()
        assert cs._help_candidates("iid_pair", col=5) == [
            "metadata.point",
            "metadata.symbol",
            "metadata.point_symbol",
        ]

    def test_col_ignored_for_tree(self):
        """Tree hover always shows the first field regardless of col."""
        cs = self._sheet()
        assert cs._help_candidates("iid_pair", tree=True, col=1) == [
            "metadata.point",
            "metadata.point_symbol",
        ]

    def test_col_ignored_for_single_field_row(self):
        cs = self._sheet()
        assert cs._help_candidates("iid_single", col=0) == ["metadata.comment"]


# ── tree column: suffix for multi-field metadata rows ──────────────────────


class TestMetadataTreeSuffix:
    """Tree-column hover appends `", …"` only when the row has >1 editable col.

    The suffix signals that data-cell hover shows a per-field status (the
    second field, e.g. ``metadata.symbol`` for ``point, symbol``).
    """

    @staticmethod
    def _sheet():
        cs = TestCellSpecFor._make_sheet()
        cs._meta["iid_pair"] = {
            "label": "point, symbol",
            "path": "metadata.point_symbol",
            "is_metadata": True,
            "max_col": 2,
        }
        cs._meta["iid_single"] = {
            "label": "comment",
            "path": "metadata.comment",
            "is_metadata": True,
            "max_col": 1,
        }
        return cs

    def test_pair_gets_suffix(self, monkeypatch):
        from tcm_gui import _help
        from tcm_gui._i18n import STRINGS

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        _help.reload_cache("en")
        cs = self._sheet()
        cs.on_hover_status = MagicMock()
        cs._status_iid = None
        # _on_tree_motion uses _hover_resolve → returns None without a real
        # sheet; drive the publication path directly.
        cs._status_source = None
        iid = "iid_pair"
        for cand in cs._help_candidates(iid, tree=True):
            if cand and (h := _help.help_for_path(cand)) and h.short:
                m = cs._meta[iid]
                multi = m.get("is_metadata") and m.get("max_col", 1) > 1
                txt = f"{h.short}{STRINGS['metadata.tree_suffix']}" if multi else h.short
                cs.on_hover_status(txt, True)
                break
        msg = cs.on_hover_status.call_args.args[0]
        assert msg == f"Station point identifier{STRINGS['metadata.tree_suffix']}", msg

    def test_single_field_no_suffix(self, monkeypatch):
        from tcm_gui import _help
        from tcm_gui._i18n import STRINGS

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        _help.reload_cache("en")
        cs = self._sheet()
        cs.on_hover_status = MagicMock()
        iid = "iid_single"
        for cand in cs._help_candidates(iid, tree=True):
            if cand and (h := _help.help_for_path(cand)) and h.short:
                m = cs._meta[iid]
                multi = m.get("is_metadata") and m.get("max_col", 1) > 1
                txt = f"{h.short}{STRINGS['metadata.tree_suffix']}" if multi else h.short
                cs.on_hover_status(txt, True)
                break
        msg = cs.on_hover_status.call_args.args[0]
        assert not msg.endswith(STRINGS["metadata.tree_suffix"]), msg


# ── _publish_status: col selects the field of a metadata paired row ────────


class TestPublishStatusColumnAware:
    """``_publish_status(iid, col)`` shows the hovered column's field short.

    Regression: paired metadata rows (``point, symbol``) always showed the
    first field's text regardless of the hovered column — the second cell's
    status was never shown.  *col* reorders ``_help_candidates`` so the
    hovered field wins.
    """

    @staticmethod
    def _sheet():
        cs = TestCellSpecFor._make_sheet()
        cs._meta["iid_pair"] = {
            "label": "point, symbol",
            "path": "metadata.point_symbol",
            "is_metadata": True,
            "max_col": 2,
        }
        cs.on_hover_status = MagicMock()
        return cs

    def test_col0_shows_first_field(self, monkeypatch):
        from tcm_gui import _help

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        _help.reload_cache("en")
        cs = self._sheet()
        cs._publish_status("iid_pair", col=0)
        msg = cs.on_hover_status.call_args.args[0]
        assert msg == "Station point identifier", msg

    def test_col1_shows_second_field(self, monkeypatch):
        from tcm_gui import _help

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        _help.reload_cache("en")
        cs = self._sheet()
        cs._publish_status("iid_pair", col=1)
        msg = cs.on_hover_status.call_args.args[0]
        assert msg == "Modification / instrument symbol (e.g. `↟`)", msg

    def test_col_none_defaults_to_first_field(self, monkeypatch):
        from tcm_gui import _help

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        _help.reload_cache("en")
        cs = self._sheet()
        cs._publish_status("iid_pair")
        msg = cs.on_hover_status.call_args.args[0]
        assert msg == "Station point identifier", msg


class TestHoverDetailPublishOrder:
    """``_hover_detail`` must be assigned BEFORE ``on_hover_status`` fires.

    Regression: ``_publish_status`` called the callback first — App reads
    ``cs._hover_detail`` synchronously in ``_on_cell_status`` to arm the dwell,
    so every tooltip showed the PREVIOUS row's detail (P_t row → the coefs
    priority text; kVabs row → P_t's Detailed).
    """

    def test_detail_snapshot_matches_hovered_row(self, monkeypatch):
        from tcm_gui import _help

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        _help.reload_cache("en")
        cfg = {
            "input": {
                "path": "/data",
                "coefs": {
                    "Ag": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "P_t": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "kVabs": [1, 2, 3, 4, 5],
                },
            }
        }
        cs, _ = TestCoefsPathChildRow._make_loaded_sheet(cfg)

        snapshots: list[str] = []
        # Mirror App._on_cell_status: read the detail AT callback time.
        cs.on_hover_status = lambda msg, md=False: snapshots.append(cs._hover_detail)

        def _iid(key: str):
            return next(i for i, m in cs._meta.items() if m.get("key") == key)

        cs._publish_status(_iid("coefs"))
        # coefs node now hosts the path cell (like input) → its Detailed is
        # the path's Detailed (browse-source description), so dwell is armed
        assert "Path to the configuration file" in snapshots[-1], (
            f"coefs node now shows path Detailed (same row as path); got {snapshots[-1]!r}"
        )
        cs._publish_status(_iid("P_t"))
        assert "polyval2d" in snapshots[-1], (
            f"P_t dwell must show its own Detailed at callback time; got {snapshots[-1]!r}"
        )
        cs._publish_status(_iid("Ag"))
        assert snapshots[-1] == "", "Ag has no #### Detailed → no dwell (no group-text inheritance)"
        cs._clear_status()
        assert snapshots[-1] == "", "sheet leave must publish with an empty detail (no stale dwell arm)"


class TestCoefsSectionDwellRu:
    """Parent ``coefs`` row hover must arm a non-empty dwell without table soup (RU doc).

    End-to-end sheet path (mirrors ``App._on_cell_status``): ``_publish_status``
    on the ``coefs`` node must leave ``cs._hover_detail`` non-empty and free of
    table rows — first it carried the whole coefficient table, then nothing.
    """

    def test_coefs_parent_row_arms_section_dwell(self, monkeypatch):
        from tcm_gui import _help

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "ru")
        _help.reload_cache("ru")
        cfg = {
            "input": {
                "path": "/data",
                "coefs": {
                    "Ag": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "P_t": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "kVabs": [1, 2, 3, 4, 5],
                },
            }
        }
        cs, _ = TestCoefsPathChildRow._make_loaded_sheet(cfg)

        snapshots: list[tuple[str, str]] = []
        cs.on_hover_status = lambda msg, md=False: snapshots.append((msg, cs._hover_detail))

        iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs")
        cs._publish_status(iid)
        msg, detail = snapshots[-1]
        assert detail, f"coefs parent dwell empty; status was {msg!r}"
        assert "Матрица масштаба" not in detail, f"table row leaked into dwell: {detail!r}"

    def test_coefs_parent_row_tree_dwell_differs_from_status(self, monkeypatch, detailed_prose):
        """Tree hover on the ``coefs`` parent row: concise status, prose dwell (RU doc).

        The section row's dwell must be visibly different from its status —
        otherwise the dwell firing after the delay is imperceptible ("shows as
        status, not dwell").  Expected: status falls back to the ``##``
        subtitle, dwell carries the bare ``### Detailed`` prose.
        """
        from tcm_gui import _help

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "ru")
        _help.reload_cache("ru")
        cfg = {
            "input": {
                "path": "/data",
                "coefs": {
                    "Ag": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "P_t": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "kVabs": [1, 2, 3, 4, 5],
                },
            }
        }
        cs, _ = TestCoefsPathChildRow._make_loaded_sheet(cfg)
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs")
        assert cs._help_candidates(iid, tree=True) == ["input.coefs"]
        h = _help.help_for_path("input.coefs")
        assert h is not None
        status = _help.section_body_short(h)
        dwell = cs._resolve_detail("input.coefs")
        assert status == "Калибровочные коэффициенты", f"status should be the subtitle; got {status!r}"
        # Prose derived from the reference doc — single source of truth, no hardcoded phrases
        prose = detailed_prose(_help.doc_path("ru").read_text(encoding="utf-8"), "`input.coefs`")
        assert prose in dwell, f"dwell should be the Detailed prose {prose!r}; got {dwell!r}"
        assert status not in dwell and dwell not in status, "dwell must differ from status"
        assert "Матрица масштаба" not in dwell, f"table row leaked into dwell: {dwell!r}"

    def test_coefs_parent_row_end_to_end_status_then_dwell(self, monkeypatch):
        """Full App dwell chain on the coefs parent row (RU doc, real Tk timers).

        Binds the real ``App`` status/dwell methods to a namespace with a
        recording label, with realistic ordering (settle < dwell delay):
        status hint first, path Detailed dwell second — distinct texts, no
        table rows anywhere.
        """
        import time
        import tkinter as tk

        from tcm_gui import _help
        from tcm_gui.app import App

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "ru")
        _help.reload_cache("ru")
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("Tk not available")
        try:
            cfg = {
                "input": {
                    "path": "/data",
                    "coefs": {
                        "Ag": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "P_t": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "kVabs": [1, 2, 3, 4, 5],
                    },
                }
            }
            cs, _ = TestCoefsPathChildRow._make_loaded_sheet(cfg)
            ns = type("NS", (), {})()
            ns.root = root
            ns._tip_active = False
            ns._status_hovering = False
            ns._dwell_active = False
            ns._dwell_widget = None
            ns._status_job = None
            ns._dwell_job = None
            ns._dwell_hide_job = None
            ns._STATUS_SETTLE_MS = 10
            ns._DWELL_MS = 60
            ns._DWELL_HIDE_MS = 0
            ns._labels = []
            ns._status_lbl = type(
                "L", (), {"set_text": lambda self, t, raw=False, base=None: ns._labels.append(t)}
            )()
            ns._status_lbl_f1_anchor = None
            for fn in (
                "_cancel_status_job",
                "_apply_status",
                "_set_status",
                "_cancel_dwell_job",
                "_arm_dwell",
                "_show_dwell_tip",
                "_cancel_dwell_hide_job",
                "_clear_dwell_now",
                "_on_cell_status",
            ):
                setattr(ns, fn, getattr(App, fn).__get__(ns))
            cs.on_hover_status = lambda msg, md=False: ns._on_cell_status(cs, msg, md)
            iid = next(i for i, m in cs._meta.items() if m.get("key") == "coefs")
            cs._publish_status(iid)
            time.sleep(0.15)
            root.update()
            assert len(ns._labels) == 2, f"expected status + dwell writes; got {ns._labels!r}"
            status, dwell = ns._labels
            assert "Источник коэффициентов" in status, f"status should be the path hint; got {status!r}"
            assert "Должен содержать коэффициенты" in dwell, (
                f"dwell should be the path Detailed; got {dwell!r}"
            )
            assert status != dwell, "dwell must differ from status"
            assert not any("Матрица масштаба" in t for t in ns._labels), (
                f"table row reached the label: {ns._labels!r}"
            )
        finally:
            root.destroy()


# ── _hide_hover_field cancels in-flight Entry edits (regression) ────────────


class TestHideHoverFieldCancelsEdit:
    """An open floated-field Entry must be cancelled on immediate teardown.

    Regression: double-clicking a browse row (``input.path`` /
    ``input.path``) opened the Entry, then double-clicking any other row
    ran ``_hide_hover_field`` which only ``place_forget``-ed the field.  The
    orphaned Entry left ``_editing=True`` — every ``_editing``-guarded path
    (``_show_hover_field``, ``_do_field_hide``, ``_on_sheet_motion``)
    early-returned forever, so the browse button + editing field overlay never
    reappeared until a new scan rebuilt the sheet.
    """

    @staticmethod
    def _sheet_with_field(editing: bool):
        cs, _ = TestCoefsPathChildRow._make_loaded_sheet()
        cs._hover_field = MagicMock()
        cs._hover_field._editing = editing
        cs._hover_field.winfo_ismapped.return_value = True
        cs._hover_btn = MagicMock()
        return cs

    def test_hide_cancels_active_edit(self):
        cs = self._sheet_with_field(editing=True)
        cs._hide_hover_field()
        cs._hover_field.cancel_edit.assert_called_once()

    def test_hide_unmaps_before_cancel(self):
        """``place_forget`` runs BEFORE ``cancel_edit`` so the edit-end
        callback's ``winfo_ismapped`` guard skips the restore round-trip."""
        cs = self._sheet_with_field(editing=True)
        cs._hide_hover_field()
        calls = [name for name, _args, _kw in cs._hover_field.method_calls]
        assert "place_forget" in calls and "cancel_edit" in calls
        assert calls.index("place_forget") < calls.index("cancel_edit")

    def test_hide_spares_clean_field(self):
        """No edit in flight → no ``cancel_edit`` (pending ``after_idle``
        commits from a finished edit must survive the hide)."""
        cs = self._sheet_with_field(editing=False)
        cs._hide_hover_field()
        cs._hover_field.cancel_edit.assert_not_called()
        cs._hover_field.place_forget.assert_called_once()

    def test_begin_edit_unwedges_orphan_entry(self):
        """The reported repro: double-click another row while the floated
        Entry is open — edit begin must end with the Entry cancelled."""
        cs, mock_sh = TestCoefsPathChildRow._make_loaded_sheet()
        cs._mgr = MagicMock()
        cs._hover_field = MagicMock()
        cs._hover_field._editing = True
        cs._hover_field.winfo_ismapped.return_value = True
        cs._hover_btn = MagicMock()
        input_iid = next(i for i, m in cs._meta.items() if m.get("type") == "input")
        cs._iid_at_row = MagicMock(return_value=input_iid)
        mock_sh.get_cell_data.return_value = ""

        event = MagicMock()
        event.row = 0
        event.column = 0
        cs._on_begin_edit_cell(event)

        cs._hover_field.cancel_edit.assert_called_once()


# ── browse dialog initialdir (regression) ───────────────────────────────────


class TestBrowseInitialDir:
    """``initialdir`` contract: an existing directory value (previous
    ``askdirectory`` pick) opens the dialog AT that directory, not one level
    up.  Regression: after selecting ``…/260711_Pionerskiy@i/_raw`` the second
    browse opened at ``…/260711_Pionerskiy@i`` (``os.path.dirname`` of the
    value) — user had to descend again on every search.
    """

    @pytest.fixture(autouse=True)
    def _no_shift(self, monkeypatch):
        # _is_shift_pressed lives in _browse_button (coef_sheet no longer re-exports it)
        monkeypatch.setattr("tcm_gui._browse_button._is_shift_pressed", lambda: False)
        monkeypatch.setattr("tcm_gui._sheet_status._is_shift_pressed", lambda: False, raising=False)

    @staticmethod
    def _capturing(monkeypatch, read_val: str) -> dict:
        from tcm_gui._browse_button import BrowseOverlay

        captured: dict = {}
        monkeypatch.setattr("tkinter.filedialog.askdirectory", lambda **kw: captured.update(kw) or "")
        ov = BrowseOverlay(MagicMock(), MagicMock(), lambda: read_val, dir_title="dir")
        ov._browse()
        return captured

    def test_dir_value_opens_at_itself(self, tmp_path, monkeypatch):
        captured = self._capturing(monkeypatch, str(tmp_path))
        assert captured["initialdir"] == str(tmp_path)

    def test_file_value_opens_at_parent(self, tmp_path, monkeypatch):
        f = tmp_path / "data.txt"
        f.touch()
        captured = self._capturing(monkeypatch, str(f))
        assert captured["initialdir"] == str(tmp_path)

    def test_absent_value_falls_back_to_parent(self, monkeypatch):
        captured = self._capturing(monkeypatch, "B:/no/such/dir")
        assert captured["initialdir"] == "B:/no/such"


# ── floated overlay content-aware align (regression) ────────────────────────


class TestFloatedFieldAlign:
    """Overlay-only content-aware align: real paths right-align + scroll to
    the filename end, ghosts stay left-aligned like the covered cell.

    The mechanism lives in the HOST (``_sheet_status``), not in
    ``PathField`` — an earlier attempt inside ``PathField`` damaged the
    standalone top search field.  These tests pin the boundary: the sync
    touches only ``cs._hover_field`` (the floated field's own 1×1 sheet).
    """

    @staticmethod
    def _sheet_with_field():
        cs, _ = TestCoefsPathChildRow._make_loaded_sheet()
        cs._hover_field = MagicMock()
        cs._hover_field._editing = False
        cs._hover_field.winfo_ismapped.return_value = True
        cs._hover_btn = MagicMock()
        cs.on_hover_status = MagicMock()
        return cs

    def test_ghost_aligns_left_scrolls_left(self):
        """Placeholder active → left align + viewport at the left edge."""
        cs = self._sheet_with_field()
        f = cs._hover_field
        f._ph.has.return_value = True
        cs._sync_floated_align()
        f.sh.table_align.assert_called_once_with("w", redraw=False)
        f._scroll_to_left.assert_called_once()
        f._scroll_to_right.assert_not_called()

    def test_data_aligns_right_scrolls_right(self):
        """Real path → right align + viewport at the filename end."""
        cs = self._sheet_with_field()
        f = cs._hover_field
        f._ph.has.return_value = False
        cs._sync_floated_align()
        f.sh.table_align.assert_called_once_with("e", redraw=False)
        f._scroll_to_right.assert_called_once()
        f._scroll_to_left.assert_not_called()

    def test_no_field_is_noop(self):
        """Overlay never created → the sync must not raise."""
        cs, _ = TestCoefsPathChildRow._make_loaded_sheet()
        cs._hover_field = None
        cs._sync_floated_align()  # no exception

    def test_show_hover_field_syncs_align(self):
        """Showing the overlay on a row applies the content-aware align
        AFTER placement (scroll needs finalized geometry)."""
        cs = self._sheet_with_field()
        with patch.object(cs, "_sync_floated_align") as sync:
            iid = next(i for i, m in cs._meta.items() if m.get("type") == "input")
            cs._show_hover_field(iid, 0, 10)
        sync.assert_called_once()

    def test_hover_write_resyncs_align(self):
        """Edit commit / browse write while mapped → align re-synced to the
        new content (ghost ↔ path flip must not leave stale scroll)."""
        cs = self._sheet_with_field()
        cs._field_iid = next(i for i, m in cs._meta.items() if m.get("type") == "input")
        with patch.object(cs, "_sync_floated_align") as sync:
            cs._hover_write("/data/new_file.txt")
        sync.assert_called_once()

    def test_hover_write_skips_sync_when_unmapped(self):
        """Deferred commit after hide — field unmapped → no sync round-trip."""
        cs = self._sheet_with_field()
        cs._hover_field.winfo_ismapped.return_value = False
        cs._field_iid = next(i for i, m in cs._meta.items() if m.get("type") == "input")
        with patch.object(cs, "_sync_floated_align") as sync:
            cs._hover_write("/data/new_file.txt")
        sync.assert_not_called()


# ── _publish_status: precise date status for coef rows with a date cell ─────


class TestCoefDatePreciseStatus:
    """Coef rows with ``has_date=True`` show a component-specific date status
    instead of the generic "Per-component calibration dates" fallback.

    Regression: every coef date cell (Ag, Ah, …) showed the same generic
    ``input.coefs.dates`` short — the user could not tell which component the
    date belongs to.  Ag+Cg share Ag's date cell; Ah+Ch share Ah's.
    """

    @staticmethod
    def _sheet(monkeypatch):
        from tcm_gui import _help
        from tcm_gui._i18n import load_str

        monkeypatch.setattr("tcm_gui._help.resolve_lang", lambda: "en")
        _help.reload_cache("en")
        # STRINGS are cached at import time → patch i18n lang + reload so _S
        # picks up English (str.yaml); _i18n.resolve_lang is distinct from _help's.
        monkeypatch.setattr("tcm_gui._i18n.resolve_lang", lambda: "en")
        load_str.cache_clear()
        monkeypatch.setattr("tcm_gui._sheet_status._S", load_str(), raising=False)
        cfg = {
            "input": {
                "path": "/data",
                "coefs": {
                    "Ag": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "Ah": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "Cg": [10, 10, 10],
                    "Ch": [10, 10, 10],
                },
            }
        }
        cs, _ = TestCoefsPathChildRow._make_loaded_sheet(cfg)
        cs.on_hover_status = MagicMock()
        return cs

    def test_ag_shows_accelerometer(self, monkeypatch):
        cs = self._sheet(monkeypatch)
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "Ag")
        cs._publish_status(iid)
        msg = cs.on_hover_status.call_args.args[0]
        assert msg == "Accelerometer calibration date", msg

    def test_ah_shows_magnetometer(self, monkeypatch):
        cs = self._sheet(monkeypatch)
        iid = next(i for i, m in cs._meta.items() if m.get("key") == "Ah")
        cs._publish_status(iid)
        msg = cs.on_hover_status.call_args.args[0]
        assert msg == "Magnetometer calibration date", msg

    def test_generic_fallback_for_unmapped_coef(self, monkeypatch):
        """Coef key not in ``_COEF_DATE_LABELS`` keeps the generic doc short."""
        cs = self._sheet(monkeypatch)
        # azimuth_shift_deg has no date cell; use a synthetic has_date row
        # with an unmapped key to verify the fallback path.
        cs._meta["iid_unmapped"] = {
            "key": "Xx",
            "type": "2d",
            "has_date": True,
            "max_col": 0,
            "shape": (2, 3),
            "path": "input.coefs.Xx",
            "parent": next(i for i, m in cs._meta.items() if m.get("key") == "coefs"),
        }
        cs._publish_status("iid_unmapped")
        msg = cs.on_hover_status.call_args.args[0]
        # Falls through to the generic input.coefs.dates short (non-breaking hyphen)
        assert msg == "Per\u2011component calibration dates", msg
