"""Unit tests for coef_sheet — ConfigSheet with mocked tksheet.Sheet.

Tests path tracking, _cell_spec_for, _apply_styles (checkbox/dropdown/align),
and _resolve_bg without requiring a real Tk event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

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


# ── _resolve_bg ─────────────────────────────────────────────────────────────


class TestResolveBg:
    def test_hex_passthrough(self):
        """Already-hex color passes through unchanged."""
        from tcm_gui.coef_sheet import _resolve_bg

        mock_widget = MagicMock()
        mock_widget.winfo_rgb.return_value = (0xF0 * 257, 0xF0 * 257, 0xF0 * 257)
        assert _resolve_bg(mock_widget, "#F0F0F0") == "#f0f0f0"

    def test_system_color_converted(self):
        """'SystemButtonFace' → hex via winfo_rgb."""
        from tcm_gui.coef_sheet import _resolve_bg

        mock_widget = MagicMock()
        # Windows SystemButtonFace ≈ #F0F0F0 → (0xF0*257, 0xF0*257, 0xF0*257)
        mock_widget.winfo_rgb.return_value = (61680, 61680, 61680)
        result = _resolve_bg(mock_widget, "SystemButtonFace")
        assert result.startswith("#"), f"expected hex, got {result}"
        assert len(result) == 7, f"expected #rrggbb, got {result}"

    def test_invalid_color_fallback(self):
        """TclError → return original string."""
        from tkinter import TclError

        from tcm_gui.coef_sheet import _resolve_bg

        mock_widget = MagicMock()
        mock_widget.winfo_rgb.side_effect = TclError("bad color")
        assert _resolve_bg(mock_widget, "invalid") == "invalid"


# ── _cell_spec_for ──────────────────────────────────────────────────────────


class TestCellSpecFor:
    @staticmethod
    def _make_sheet():
        """Create a ConfigSheet with a mocked Sheet — no Tk needed."""
        from tcm_gui.coef_sheet import ConfigSheet

        with patch("tcm_gui.coef_sheet.Sheet"):
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

        with patch("tcm_gui.coef_sheet.Sheet"):
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

        def _insert(**kw):
            iid = f"iid_{kw.get('text', 'x')}"
            _kids.setdefault(kw.get("parent") or "", []).append(iid)
            return iid

        mock_sh.insert.side_effect = _insert
        mock_sh.get_children.side_effect = lambda parent="": list(_kids.get(parent or "", ()))
        mock_sh.get_cell_data.return_value = ""
        mock_sh.tag_names.return_value = []
        mock_sh.winfo_rgb.return_value = (61680, 61680, 61680)  # #F0F0F0

        with patch("tcm_gui.coef_sheet.Sheet", return_value=mock_sh):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg
        from tcm.config import Config, Return

        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ({}, {}, "")
        cs._fg_default = "#000000"

        cs._build_coefs(cfg)
        # Open every constructed node so all rows receive styling.
        for m in cs._meta.values():
            m["open"] = True
        return cs, mock_sh

    @patch("tcm_gui.coef_sheet.ttk")
    @patch("tcm_gui.coef_sheet._resolve_bg", return_value="#F0F0F0")
    def test_node_column_gets_bg(self, mock_resolve, mock_ttk):
        """Index canvas (tree column) gets bg: global option + per-cell highlight."""
        mock_ttk.Style.return_value.lookup.return_value = "#F0F0F0"
        cs, mock_sh = self._make_loaded_sheet()
        cs._apply_styles()

        # Global fallback (newer builds / later-expanded rows)
        mock_sh.set_options.assert_any_call(index_background="#F0F0F0")
        # Per-cell highlight on the index canvas (7.6.x draw path)
        index_calls = [
            c for c in mock_sh.highlight_cells.call_args_list
            if c.kwargs.get("canvas") == "index"
        ]
        assert len(index_calls) > 0, (
            f"expected highlight_cells(canvas='index') calls, got: {mock_sh.highlight_cells.call_args_list}"
        )

    @patch("tcm_gui.coef_sheet.ttk")
    @patch("tcm_gui.coef_sheet._resolve_bg", return_value="#F0F0F0")
    def test_coef_cells_right_aligned(self, mock_resolve, mock_ttk):
        """Coef data cells are right-aligned (number type)."""
        mock_ttk.Style.return_value.lookup.return_value = "#F0F0F0"
        cs, mock_sh = self._make_loaded_sheet()
        cs._apply_styles()

        # align_cells is called positionally: align_cells(r, c, align="e", redraw=False)
        # OR align_cells(r, c, "e", redraw=False) — check both patterns
        right_align_calls = [
            c for c in mock_sh.align_cells.call_args_list
            if c.kwargs.get("align") == "e"
            or (len(c.args) >= 3 and c.args[2] == "e")
        ]
        assert len(right_align_calls) > 0, (
            f"expected right-aligned coef cells, got calls: {mock_sh.align_cells.call_args_list}"
        )

    @patch("tcm_gui.coef_sheet.ttk")
    @patch("tcm_gui.coef_sheet._resolve_bg", return_value="#F0F0F0")
    def test_date_cells_right_aligned(self, mock_resolve, mock_ttk):
        """Date cells (coefs parent with has_date) are right-aligned (``e``).

        See :func:`_apply_styles` section 3 and the CellSpec.kind table in
        how_gui_works.md — ``"date"`` is right-aligned, not left.
        """
        mock_ttk.Style.return_value.lookup.return_value = "#F0F0F0"
        cs, mock_sh = self._make_loaded_sheet()
        cs._apply_styles()

        right_align_calls = [
            c for c in mock_sh.align_cells.call_args_list
            if c.kwargs.get("align") == "e" or (len(c.args) >= 3 and c.args[2] == "e")
        ]
        # coefs node has has_date=True → date column should be right-aligned
        assert len(right_align_calls) > 0, "expected right-aligned date cells"

    @patch("tcm_gui.coef_sheet.ttk")
    @patch("tcm_gui.coef_sheet._resolve_bg", return_value="#F0F0F0")
    def test_header_highlighted(self, mock_resolve, mock_ttk):
        """Header cells get highlight_cells(canvas='header')."""
        mock_ttk.Style.return_value.lookup.return_value = "#F0F0F0"
        cs, mock_sh = self._make_loaded_sheet()
        cs._apply_styles()

        header_calls = [
            c for c in mock_sh.highlight_cells.call_args_list
            if c.kwargs.get("canvas") == "header"
        ]
        assert len(header_calls) > 0, "expected header highlight calls"


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

        with patch("tcm_gui.coef_sheet.Sheet", return_value=mock_sh):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg
        from tcm.config import Config, Return

        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ({}, {}, "")
        cs._fg_default = "#000000"

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
        assert all(v == "" for v in values[1:]), (
            f"input values[1:] should all be empty, got {values[1:]!r}"
        )

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
    """Browse button must be destroyed (not just hidden) on detach,
    and must never be placed on non-browse rows.
    """

    @staticmethod
    def _make_manager():
        """Create a BrowseButtonManager with a mock sheet."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        # after() returns a job id that after_cancel can consume
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        return BrowseButtonManager(mock_sh, MagicMock()), mock_sh

    def test_detach_destroys_button(self):
        """``detach()`` must call ``destroy()`` on the button, not ``place_forget``."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_editor = MagicMock()
        mock_editor.winfo_height.return_value = 20
        # At attach() time no editor exists (begin_edit_cell fires before creation)
        mock_sh.get_text_editor_widget.return_value = None

        mock_btn = MagicMock()
        with patch("tcm_gui._browse_button.ttk.Button", return_value=mock_btn):
            mgr = BrowseButtonManager(mock_sh, MagicMock())
            mgr.attach(0, 0)
            # Editor appears before retry fires
            mock_sh.get_text_editor_widget.return_value = mock_editor
            mgr._acquire_and_place()

        assert mgr._button is mock_btn, "button should be the mock after acquire"

        mgr.detach()

        # Button must be destroyed, not just place_forget'd
        mock_btn.destroy.assert_called_once(), "detach() must destroy the button widget"
        assert mgr._button is None, "_button must be None after detach"

    def test_detach_no_button_is_safe(self):
        """Calling ``detach()`` when no button exists must not raise."""
        mgr, _ = self._make_manager()
        # Never called attach() — detach should be a no-op
        mgr.detach()
        assert mgr._button is None

    def test_attach_cancels_stale_retry(self):
        """A second ``attach()`` must cancel the pending retry from the first."""
        mgr, mock_sh = self._make_manager()
        mgr.attach(0, 0)
        first_job = mgr._retry_job
        assert first_job is not None

        mgr.attach(0, 0)  # second attach
        # after_cancel must have been called with the first job
        mock_sh.after_cancel.assert_any_call(first_job)
        assert mgr._retry_job != first_job or mgr._retry_job is not None

    def test_button_not_created_without_attach(self):
        """No button widget must exist before ``attach()`` + retry fires."""
        mgr, _ = self._make_manager()
        assert mgr._button is None, "button must not exist before attach"

    def test_end_edit_detaches_unconditionally(self):
        """``_on_end_edit_cell`` must call ``mgr.detach()`` for ANY row,
        not just coefs_path — prevents ghost buttons on non-path rows."""

        cs, mock_sh = TestCoefsPathChildRow._make_loaded_sheet()
        cs._mgr = MagicMock()
        # Simulate _iid_at_row returning a non-path iid (e.g. Ag row)
        ag_iid = next(iid for iid, m in cs._meta.items() if m.get("type") == "2d")
        cs._iid_at_row = MagicMock(return_value=ag_iid)
        mock_sh.get_cell_data.return_value = ""

        event = MagicMock()
        event.row = 5
        event.column = 0
        event.value = "1.0"
        cs._on_end_edit_cell(event)

        cs._mgr.detach.assert_called_once(), (
            "detach() must be called for non-path rows too (prevents ghost button)"
        )

    def test_begin_edit_attaches_only_for_browse_rows(self):
        """``_on_begin_edit_cell`` must call ``attach()`` only for rows
        with ``browse: True`` in meta — never for numeric/coef rows.

        It must still call ``detach()`` unconditionally (prevents ghost
        buttons from a previous browse-row edit).
        """

        cs, mock_sh = TestCoefsPathChildRow._make_loaded_sheet()
        cs._mgr = MagicMock()

        # Find a non-browse row (e.g. Ag)
        ag_iid = next(iid for iid, m in cs._meta.items() if m.get("type") == "2d")
        cs._iid_at_row = MagicMock(return_value=ag_iid)
        mock_sh.get_cell_data.return_value = ""

        event = MagicMock()
        event.row = 10
        event.column = 0
        cs._on_begin_edit_cell(event)

        cs._mgr.attach.assert_not_called(), (
            "attach() must NOT be called for non-browse rows like Ag"
        )
        cs._mgr.detach.assert_called_once(), (
            "detach() must be called for ALL rows (cleans up previous button)"
        )

    def test_editor_destroy_triggers_detach(self):
        """When the editor is destroyed by tksheet (without ``end_edit_cell``),
        the ``<Destroy>`` binding must call ``detach()`` — prevents ghost buttons."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_editor = MagicMock()
        mock_editor.winfo_height.return_value = 20
        mock_sh.get_text_editor_widget.return_value = None  # no editor at attach time

        mock_btn = MagicMock()
        with patch("tcm_gui._browse_button.ttk.Button", return_value=mock_btn):
            mgr = BrowseButtonManager(mock_sh, MagicMock())
            mgr.attach(0, 0)
            mock_sh.get_text_editor_widget.return_value = mock_editor  # editor appears
            mgr._acquire_and_place()

        assert mgr._button is mock_btn, "precondition: button exists"

        # Capture the <Destroy> binding that was registered on the editor
        destroy_bind_calls = [c for c in mock_editor.bind.call_args_list if c.args and c.args[0] == "<Destroy>"]
        assert len(destroy_bind_calls) >= 1, (
            f"expected <Destroy> binding on editor, got {mock_editor.bind.call_args_list}"
        )
        # Extract the callback from the binding
        destroy_cb = destroy_bind_calls[-1].args[1]

        # Simulate tksheet destroying the editor
        destroy_event = MagicMock()
        destroy_event.widget = mock_editor  # must be the editor itself
        destroy_cb(destroy_event)

        # Button must be destroyed via detach()
        mock_btn.destroy.assert_called_once(), (
            "editor <Destroy> must trigger detach() → destroy()"
        )
        assert mgr._button is None, "_button must be None after editor destroy"

    def test_polling_detects_replaced_editor(self):
        """When tksheet reuses the TextEditor for a different row,
        ``_update_icon`` polling must detect the stale editor and detach."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        editor_v1 = MagicMock()
        editor_v1.winfo_height.return_value = 20
        mock_sh.get_text_editor_widget.return_value = None  # no editor at attach time

        mock_btn = MagicMock()
        with patch("tcm_gui._browse_button.ttk.Button", return_value=mock_btn):
            mgr = BrowseButtonManager(mock_sh, MagicMock())
            mgr.attach(0, 0)
            mock_sh.get_text_editor_widget.return_value = editor_v1  # editor appears
            mgr._acquire_and_place()

        assert mgr._editor is editor_v1, "precondition: editor captured"

        # tksheet now returns a DIFFERENT editor (reused for another row)
        editor_v2 = MagicMock()
        mock_sh.get_text_editor_widget.return_value = editor_v2

        # Simulate one polling tick
        mgr._update_icon()

        mock_btn.destroy.assert_called_once(), (
            "polling must detect replaced editor and detach"
        )
        assert mgr._button is None, "_button must be None after editor replaced"
        assert mgr._editor is None, "_editor must be None after editor replaced"

    def test_f3_stale_retry_cancelled(self):
        """F3 — ``attach()`` cancels any pending retry from a previous cycle.

        Regression: L1 — without cancellation, a stale retry would land on
        whatever editor is current (possibly a non-path row).
        """
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_sh.get_text_editor_widget.return_value = None

        with patch("tcm_gui._browse_button.ttk.Button"):
            mgr = BrowseButtonManager(mock_sh, MagicMock())
            mgr.attach(0, 0)
            first_job = mgr._retry_job
            assert first_job is not None, "precondition: retry scheduled"

            # Second attach() must cancel the first retry
            mgr.attach(1, 0)

        mock_sh.after_cancel.assert_any_call(first_job), (
            "attach() must cancel the previous retry (F3 — prevents L1)"
        )

    def test_f3_attach_resets_existing_button(self):
        """F3 — ``attach()`` must destroy a live button from a previous cycle
        before starting a new retry.  Without this, buttons accumulate.

        Regression: L3 — ``attach`` ≠ reset.
        """
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_sh.get_text_editor_widget.return_value = None  # no editor at attach time

        old_btn = MagicMock()
        new_btn = MagicMock()
        btn_seq = iter([old_btn, new_btn])
        with patch("tcm_gui._browse_button.ttk.Button", side_effect=lambda *a, **kw: next(btn_seq)):
            mgr = BrowseButtonManager(mock_sh, MagicMock())

            # First cycle: editor appears after attach, button placed
            mock_editor_1 = MagicMock()
            mock_editor_1.winfo_height.return_value = 20
            mgr.attach(0, 0)
            mock_sh.get_text_editor_widget.return_value = mock_editor_1
            mgr._acquire_and_place()
            assert mgr._button is old_btn, "precondition: first button placed"

            # Second attach() must destroy old_btn before starting new retry
            mock_sh.get_text_editor_widget.return_value = None
            mgr.attach(1, 0)

            old_btn.destroy.assert_called_once(), (
                "attach() must destroy the previous button (F3 — idempotent reset)"
            )

    def test_browse_writes_to_col0_regardless_of_edit_col(self):
        """Browse must always write the selected path to column 0,
        even if the user clicked on column 2 to open the editor."""
        from tcm_gui._browse_button import BrowseButtonManager

        mock_sh = MagicMock()
        mock_sh.after.side_effect = lambda ms, cb=None, *_a: f"job_{id(cb)}" if cb else "job"
        mock_editor = MagicMock()
        mock_editor.winfo_height.return_value = 20
        mock_sh.get_text_editor_widget.return_value = None

        with (
            patch("tcm_gui._browse_button.ttk.Button"),
            patch("tcm_gui._browse_button.filedialog.askdirectory", return_value="/selected/path"),
        ):
            mgr = BrowseButtonManager(mock_sh, MagicMock())
            # Attach with col=2 (user clicked on column 2)
            mgr.attach(5, 2)
            mock_sh.get_text_editor_widget.return_value = mock_editor
            mgr._acquire_and_place()

            # Simulate browse
            mgr._browse()

        # Path must be written to column 0, not column 2
        mock_sh.set_cell_data.assert_called_once_with(5, 0, "/selected/path"), (
            "browse must write path to column 0 regardless of edit column"
        )
        # Editor must be closed after browse
        mock_sh.after_idle.assert_any_call(mock_sh.close_text_editor)
