"""Tests for default-value gray foreground and end-edit toggle in ConfigSheet.

Covers:
- _default_for_path: dotted path resolution through _CFG_DEFAULTS
- _default_for_cell: per-cell default lookup (scalar, 1D, 2D, None-default)
- _apply_end_edit_style: gray toggle using event.value (not stale sh.item)
- Array child paths: no doubling (parent.name.name[i] -> parent.name[i])
- _iid_at_row: uses tksheet API, not depth-first walk
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import tcm_gui.coef_sheet as coef_sheet
import pytest

from tcm_gui.cli_cfg import NO_DEFAULT, default_for_path
from tcm_gui.coef_sheet import ConfigSheet
from tcm_gui.theme import CELL_DEFAULT_VAL_FG


# -- _default_for_path -----------------------------------------------------------


class TestDefaultForPath:
    """Walk dotted paths through _CFG_DEFAULTS."""

    def test_scalar_default(self):
        """input.coefs.azimuth_shift_deg -> 180."""
        assert default_for_path("input.coefs.azimuth_shift_deg") == 180

    def test_2d_array_element(self):
        """input.coefs.Ag[0] -> first row of Ag default."""
        result = default_for_path("input.coefs.Ag[0]")
        assert isinstance(result, list), f"expected list, got {type(result)}"
        assert len(result) == 3, f"expected 3 elements, got {len(result)}"

    def test_2d_array_scalar_element(self):
        """Nested indexing [i][j] is not in path -- cell col_idx handles 2nd dim.

        Path input.coefs.Ag[0] returns the first row (a list).
        _default_for_cell with col_idx=0 picks the first element.
        """
        row = default_for_path("input.coefs.Ag[0]")
        assert isinstance(row, list), f"expected list for Ag[0], got {type(row)}"
        assert abs(row[0] - 0.00173) < 1e-6, f"expected ~0.00173, got {row[0]}"

    def test_none_default_field(self):
        """P_t is Optional[...] = None -> '' (treated as no-user-default)."""
        result = default_for_path("input.coefs.P_t")
        assert result == "" or result is None, f"P_t default: {result!r}"

    def test_none_default_indexed(self):
        """input.coefs.P_t[0] -> '' or _NO_DEFAULT (P_t default is None)."""
        result = default_for_path("input.coefs.P_t[0]")
        assert result is NO_DEFAULT or result == "", f"P_t[0] default: {result!r}"

    def test_missing_section(self):
        """Unknown section -> _NO_DEFAULT."""
        assert default_for_path("nonexistent.field") is NO_DEFAULT

    def test_missing_field(self):
        """Known section, unknown field -> _NO_DEFAULT."""
        assert default_for_path("input.nonexistent") is NO_DEFAULT

    def test_1d_flat_array(self):
        """input.coefs.Cg -> default list."""
        result = default_for_path("input.coefs.Cg")
        assert isinstance(result, list), f"expected list, got {type(result)}"

    def test_out_section(self):
        """out.dt_bins -> default from ConfigOut_InclProc."""
        result = default_for_path("out.dt_bins")
        assert result is not NO_DEFAULT, "out.dt_bins should have a default"

    def test_program_return(self):
        """program.return_ -> Return.END value."""
        result = default_for_path("program.return_")
        assert result is not NO_DEFAULT, "program.return_ should have a default"

    def test_doubled_path_fails(self):
        """Bug regression: input.coefs.Ag.Ag[0] should NOT resolve."""
        result = default_for_path("input.coefs.Ag.Ag[0]")
        assert result is NO_DEFAULT, f"doubled path should fail, got {result!r}"


# -- _default_for_cell -----------------------------------------------------------


class TestDefaultForCell:
    """Per-cell default lookup using meta path + col_idx."""

    @staticmethod
    def _make_sheet():
        with patch.object(coef_sheet, "Sheet"):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = MagicMock()
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = {}
        cs._config_root = None
        cs._return_enum = None
        cs._snap = ()
        return cs

    def test_scalar_col0(self):
        """azimuth_shift_deg scalar at col 0 -> 180."""
        cs = self._make_sheet()
        m = {"path": "input.coefs.azimuth_shift_deg", "type": "scalar"}
        assert cs._default_for_cell("iid", m, 0) == 180

    def test_scalar_col1_no_default(self):
        """Scalar at col 1 -> _NO_DEFAULT (scalar has only col 0)."""
        cs = self._make_sheet()
        m = {"path": "input.coefs.azimuth_shift_deg", "type": "scalar"}
        assert cs._default_for_cell("iid", m, 1) is NO_DEFAULT

    def test_2d_child_col0(self):
        """Ag[0] row, col 0 -> first element of first row."""
        cs = self._make_sheet()
        m = {"path": "input.coefs.Ag[0]", "type": "_coef_child"}
        result = cs._default_for_cell("iid", m, 0)
        assert abs(float(result) - 0.00173) < 1e-6, f"expected ~0.00173, got {result}"

    def test_2d_child_col1(self):
        """Ag[0] row, col 1 -> second element of first row."""
        cs = self._make_sheet()
        m = {"path": "input.coefs.Ag[0]", "type": "_coef_child"}
        result = cs._default_for_cell("iid", m, 1)
        assert float(result) == 0, f"expected 0, got {result}"

    def test_none_default_returns_empty(self):
        """Field with None default -> '' (empty string)."""
        cs = self._make_sheet()
        m = {"path": "input.coefs.P_t[0]", "type": "_coef_child"}
        result = cs._default_for_cell("iid", m, 0)
        assert result == "" or result is NO_DEFAULT

    def test_empty_path_returns_no_default(self):
        """Empty path -> _NO_DEFAULT."""
        cs = self._make_sheet()
        m = {"path": ""}
        assert cs._default_for_cell("iid", m, 0) is NO_DEFAULT

    def test_dict_default_rejected(self):
        """Non-leaf path resolving to dict -> _NO_DEFAULT (not a cell value)."""
        cs = self._make_sheet()
        m = {"path": "input"}
        assert cs._default_for_cell("iid", m, 0) is NO_DEFAULT


# -- Array child path construction -----------------------------------------------


class TestArrayChildPaths:
    """Verify _ins_2d / _ins_1d / _ins_generic produce correct paths (no doubling)."""

    @staticmethod
    def _make_sheet():
        with patch.object(coef_sheet, "Sheet"):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = MagicMock()
        cs.sh.insert.side_effect = lambda **kw: f"iid_{kw.get('text', 'x')}"
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = {}
        cs._config_root = None
        cs._return_enum = None
        cs._snap = ()
        return cs

    def test_2d_child_path_not_doubled(self):
        """_ins_2d: child path = parent.path[i], not parent.path.name[i]."""
        cs = self._make_sheet()
        root = cs._ins("", "input", [""] * 6, "", open_=True)
        mid = cs._ins(root, "coefs", [""] * 6, "", open_=True)
        cs._ins_2d(mid, "Test", [[1.0, 2.0], [3.0, 4.0]], (2, 2), "")
        parent_path = cs._meta["iid_Test"]["path"]
        assert parent_path == "input.coefs.Test"
        children = cs._meta["iid_Test"]["children"]
        assert len(children) == 2
        for i, child_iid in enumerate(children):
            child_path = cs._meta[child_iid]["path"]
            expected = f"input.coefs.Test[{i}]"
            assert child_path == expected, f"child {i}: expected path={expected!r}, got {child_path!r}"

    def test_1d_child_path_not_doubled(self):
        """_ins_1d: child path = parent.path (not parent.path.name)."""
        cs = self._make_sheet()
        root = cs._ins("", "input", [""] * 6, "", open_=True)
        mid = cs._ins(root, "coefs", [""] * 6, "", open_=True)
        cs._ins_1d(mid, "kVabs", [1.0, 2.0, 3.0], 3, "")
        parent_path = cs._meta["iid_kVabs"]["path"]
        child_iid = cs._meta["iid_kVabs"]["child"]
        child_path = cs._meta[child_iid]["path"]
        assert parent_path == "input.coefs.kVabs"
        assert child_path == "input.coefs.kVabs", f"1D child should share parent path, got {child_path!r}"

    def test_generic_2d_child_path_not_doubled(self):
        """_ins_generic: nested 2D array child paths are not doubled."""
        cs = self._make_sheet()
        root = cs._ins("", "out", [""] * 6, "", open_=True)
        cs._ins_generic(root, "dt_bins", [[0, 2], [600, 3600]])
        parent_path = cs._meta["iid_dt_bins"]["path"]
        assert parent_path == "out.dt_bins"
        child0_iid = "iid_dt_bins[0]"
        if child0_iid in cs._meta:
            child0_path = cs._meta[child0_iid]["path"]
            assert child0_path == "out.dt_bins[0]", (
                f"generic 2D child: expected 'out.dt_bins[0]', got {child0_path!r}"
            )


# -- _apply_end_edit_style gray toggle ----------------------------------------------------


class TestOnEndEditGrayToggle:
    """Simulate end_edit_cell events and verify gray foreground toggling."""

    @staticmethod
    def _make_loaded_sheet():
        """Create a ConfigSheet with coefs loaded, metadata populated.

        Wires ``sh.get_children`` from the recorded ``insert(parent=…)`` calls
        so :meth:`_walk_visible` (used by :meth:`_iid_at_row`) yields the same
        tree :meth:`_build_coefs` constructed.  Sets ``_fg_default`` since the
        stub bypasses :meth:`_apply_styles` (where it is normally resolved).
        """
        from tcm.schema import Config, Return

        cfg = {
            "input": {
                "path": "/data",
                "coefs": {
                    "Ag": [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]],
                    "azimuth_shift_deg": 180.0,
                },
            }
        }
        mock_sh = MagicMock()
        mock_sh.total_columns.return_value = 6
        mock_sh.total_rows.return_value = 0
        _kids: dict[Any, list[str]] = {}
        _iid_counter = 0

        def _insert(**kw):
            nonlocal _iid_counter
            _iid_counter += 1
            iid = f"iid_{kw.get('text', _iid_counter)}"
            _kids.setdefault(kw.get("parent") or "", []).append(iid)
            return iid

        mock_sh.insert.side_effect = _insert
        mock_sh.get_children.side_effect = lambda parent="": list(_kids.get(parent or "", ()))
        mock_sh.winfo_rgb.return_value = (61680, 61680, 61680)

        with patch.object(coef_sheet, "Sheet", return_value=mock_sh):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg
        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ()
        cs._fg_default = "#000000"  # normally resolved in _apply_styles
        # __new__ bypasses __init__ — set row-cache fields manually
        cs._int_row_of = {}
        cs._vis = ()
        cs._col_resize = MagicMock()

        cs._build_coefs(cfg)

        # Open the coefs subtree and its 2D child containers (Ag, …) so every
        # editable cell has a display row — matches the visible state when a
        # user clicks a cell.  _is_open consults _meta[iid]["open"] directly
        # (rather than via the _item_hook wrapper, which the stub bypasses),
        # so flip it inline on every node along the open path.
        for iid, m in cs._meta.items():
            if m.get("key") == "coefs" or m.get("type") == "2d":
                m["open"] = True

        # Display-row → iid map — _walk(visible=True) yields only nodes
        # whose ancestors are all open; _meta filter mirrors _rebuild_row_caches.
        display_iids = list(iid for iid in cs._walk(visible=True) if iid in cs._meta)
        _display_row = {iid: idx for idx, iid in enumerate(display_iids)}
        # ``get_row_from_iid`` is still used by tests that verify dual-row
        # semantics; expose the visible-order index through it.
        mock_sh.get_row_from_iid = lambda iid, rm=_display_row: rm.get(iid)
        # Populate _vis so _iid_at_row resolves display rows correctly.
        cs._vis = tuple(display_iids)

        return cs, mock_sh

    @staticmethod
    def _make_event(row: int, column: int, value: Any) -> SimpleNamespace:
        """Create a minimal EventDataDict-like event."""
        return SimpleNamespace(row=row, column=column, value=value)

    @staticmethod
    def _find_iid(cs, path):
        """Find iid by meta path."""
        for iid, m in cs._meta.items():
            if m.get("path") == path:
                return iid
        return None

    def test_edit_non_default_removes_gray(self):
        """Editing a default-valued cell to a non-default value -> gray cleared.

        Clear side uses :attr:`_fg_default` explicitly (never ``fg=None`` —
        that's a per-key merge no-op in tksheet 7.x: docs/project_developer_guide/GUI/decisions.md).
        """
        cs, mock_sh = self._make_loaded_sheet()
        azimuth_iid = self._find_iid(cs, "input.coefs.azimuth_shift_deg")
        assert azimuth_iid is not None, "azimuth_shift_deg not found in _meta"
        azimuth_row = cs.sh.get_row_from_iid(azimuth_iid)

        event = self._make_event(row=azimuth_row, column=0, value=181)
        cs._apply_end_edit_style(event)

        clear_calls = [
            c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("fg") == cs._fg_default
        ]
        assert len(clear_calls) > 0, (
            f"expected highlight_cells(fg=_fg_default) to clear gray, "
            f"got calls: {mock_sh.highlight_cells.call_args_list}"
        )

    def test_edit_to_default_restores_gray(self):
        """Editing a non-default cell back to default -> gray restored."""
        cs, mock_sh = self._make_loaded_sheet()
        azimuth_iid = self._find_iid(cs, "input.coefs.azimuth_shift_deg")
        assert azimuth_iid is not None
        azimuth_row = cs.sh.get_row_from_iid(azimuth_iid)

        event = self._make_event(row=azimuth_row, column=0, value=180)
        cs._apply_end_edit_style(event)

        gray_calls = [c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("fg") == CELL_DEFAULT_VAL_FG]
        assert len(gray_calls) > 0, (
            f"expected highlight_cells(fg=_DEFAULT_FG) to restore gray, "
            f"got calls: {mock_sh.highlight_cells.call_args_list}"
        )

    def test_uses_event_value_not_stale_sheet(self):
        """_apply_end_edit_style must use event.value, not sh.item(iid).get('values').

        Bug: set_cell_data_undo(redraw=False) does not sync treeview item
        values before end_edit_cell fires, so sh.item() returns stale data.
        """
        cs, mock_sh = self._make_loaded_sheet()
        azimuth_iid = self._find_iid(cs, "input.coefs.azimuth_shift_deg")
        assert azimuth_iid is not None
        azimuth_row = cs.sh.get_row_from_iid(azimuth_iid)

        # Make sh.item return STALE values (the old default value 180)
        mock_sh.item.return_value = {"values": ("180", "", "", "", "", "")}

        # Event says new value is 181 (non-default)
        event = self._make_event(row=azimuth_row, column=0, value=181)
        cs._apply_end_edit_style(event)

        clear_calls = [
            c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("fg") == cs._fg_default
        ]
        assert len(clear_calls) > 0, (
            f"gray should be cleared (event.value=181 != default=180), "
            f"but got: {mock_sh.highlight_cells.call_args_list}"
        )

    def test_no_default_field_ignored(self):
        """Container nodes (input.coefs) resolve to a dict default -> handler
        returns early (no highlight_cells calls).  The ``input`` row itself is
        *not* in this category: ``_default_for_cell`` appends ``.path`` to
        resolve it as a leaf cell (see docs/project_developer_guide/GUI/decisions.md, two-row system).
        """
        cs, mock_sh = self._make_loaded_sheet()
        coefs_iid = self._find_iid(cs, "input.coefs")
        if coefs_iid is None:
            pytest.skip("input.coefs container not found in _meta")
        coefs_row = cs.sh.get_row_from_iid(coefs_iid)

        mock_sh.reset_mock()
        event = self._make_event(row=coefs_row, column=0, value="anything")
        cs._apply_end_edit_style(event)

        # Container cell has NO_DEFAULT → no cell-gray toggle, but node-label
        # walk still recolors ancestors (blue/black via _node_at_default) —
        # those use canvas="index" and are not cell-gray toggles.
        cell_calls = [c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("canvas") != "index"]
        assert len(cell_calls) == 0, (
            f"container row (dict default) should not toggle cell gray, "
            f"got: {cell_calls}"
        )

    def test_2d_child_gray_toggle(self):
        """2D array child (Ag[0]) gray toggle with correct path resolution."""
        cs, mock_sh = self._make_loaded_sheet()
        ag_child_iid = self._find_iid(cs, "input.coefs.Ag[0]")
        if ag_child_iid is None:
            for iid, m in cs._meta.items():
                if "Ag.Ag[0]" in str(m.get("path", "")):
                    pytest.fail(
                        f"Doubled path found: {m['path']!r}. Array child path construction is broken."
                    )
            pytest.skip("Ag[0] not found in _meta -- tree structure may differ")
        ag_row = cs.sh.get_row_from_iid(ag_child_iid)

        mock_sh.reset_mock()
        event = self._make_event(row=ag_row, column=0, value=0.002)
        cs._apply_end_edit_style(event)

        clear_calls = [
            c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("fg") == cs._fg_default
        ]
        assert len(clear_calls) > 0, (
            f"2D child gray should clear for non-default value, got: {mock_sh.highlight_cells.call_args_list}"
        )

    def test_iid_at_row_uses_tksheet_api(self):
        """_iid_at_row resolves display rows to iids via _walk(visible=True).

        The walk consults ``sh.get_children`` + the open-state bookkeeping in
        ``_meta[iid]["open"]`` (the ``item()`` wrapper); ``get_row_from_iid``
        is NOT consulted here (see docs/project_developer_guide/GUI/decisions.md, two-row system).
        """
        cs, mock_sh = self._make_loaded_sheet()
        visible = list(cs._vis)

        for r, iid in enumerate(visible):
            assert cs._iid_at_row(r) == iid, (
                f"_iid_at_row({r}) = {cs._iid_at_row(r)!r}, expected {iid!r} "
                f"(path={cs._meta[iid].get('path')!r})"
            )

    def test_iid_at_row_returns_none_for_unknown(self):
        """_iid_at_row returns None for row with no matching iid."""
        cs, _ = self._make_loaded_sheet()
        assert cs._iid_at_row(9999) is None


# -- Parent-node blue propagation -------------------------------------------------


class TestNodeAtDefault:
    """Verify :meth:`_node_at_default` propagates the at-default state up the tree.

    A parent node (e.g. ``input.coefs``, a 2D coef container, or a 1D-with-dates
    container) must read as ``True`` ONLY when EVERY leaf in its subtree matches
    its config dataclass default.  The 1D-with-dates case is especially subtle:
    the parent carries ``len`` for array-shape metadata but holds no own value
    cells — the child node does — so the parent must defer to its child.
    """

    @staticmethod
    def _make_loaded_sheet():
        """Build a sheet with a 2D coef (Ag), a 1D-with-dates coef (kVabs), a
        1D-flat coef (Cg) and a scalar (azimuth_shift_deg), all seeded from the
        real config defaults so every editable cell starts at its default.

        Returns ``(cs, values)`` where ``values`` is the mutable {iid: list}
        backing ``sh.item(iid)['values']`` — tests edit it in place to simulate
        user edits without re-stubbing ``sh.item``.
        """
        from tcm.schema import Config, Return
        from tcm_gui.cli_cfg import default_for_path

        kv = default_for_path("input.coefs.kVabs")
        cg = default_for_path("input.coefs.Cg")
        ag = default_for_path("input.coefs.Ag")
        cfg = {
            "input": {
                "path": "/data",
                "coefs": {
                    "Ag": ag,
                    "kVabs": kv,
                    "Cg": cg,
                    "azimuth_shift_deg": 180.0,
                },
            }
        }
        mock_sh = MagicMock()
        mock_sh.total_columns.return_value = 6
        mock_sh.total_rows.return_value = 0
        _kids: dict[Any, list[str]] = {}
        _iid_counter = 0
        values: dict[str, list] = {}

        def _insert(**kw):
            nonlocal _iid_counter
            _iid_counter += 1
            # tksheet generates unique iids; the mock must too, since _ins_1d
            # inserts both a parent and a child with the SAME text.  Decorate
            # with the counter so siblings never collide.
            iid = f"iid_{kw.get('text', _iid_counter)}_{_iid_counter}"
            _kids.setdefault(kw.get("parent") or "", []).append(iid)
            values[iid] = list(kw.get("values") or [])
            return iid

        mock_sh.insert.side_effect = _insert
        mock_sh.get_children.side_effect = lambda parent="": list(_kids.get(parent or "", ()))
        mock_sh.item.side_effect = lambda iid, **kw: {"values": tuple(values.get(iid, []))}
        mock_sh.winfo_rgb.return_value = (61680, 61680, 61680)

        with patch.object(coef_sheet, "Sheet", return_value=mock_sh):
            cs = ConfigSheet.__new__(ConfigSheet)
        cs.sh = mock_sh
        cs._meta = {}
        cs._nv = 6
        cs._full = False
        cs._cfg = cfg
        cs._config_root = Config
        cs._return_enum = Return
        cs._snap = ()
        cs._fg_default = "#000000"
        cs._int_row_of = {}
        cs._vis = ()
        cs._col_resize = MagicMock()

        cs._build_coefs(cfg)
        return cs, values

    @staticmethod
    def _find_iid(cs, path):
        for iid, m in cs._meta.items():
            if m.get("path") == path:
                return iid
        return None

    @staticmethod
    def _find_iid_by(cs, **criteria):
        """Find the first iid whose meta matches every ``key=value`` criterion."""
        for iid, m in cs._meta.items():
            if all(m.get(k) == v for k, v in criteria.items()):
                return iid
        return None

    def test_coefs_parent_at_default_when_all_children_default(self):
        """``input.coefs`` container reads at-default when every coef matches."""
        cs, _ = self._make_loaded_sheet()
        coefs_iid = self._find_iid(cs, "input.coefs")
        assert coefs_iid is not None, "input.coefs container missing from _meta"
        assert cs._node_at_default(coefs_iid) is True, (
            "input.coefs parent must be at-default when Ag, kVabs, Cg, "
            "azimuth_shift_deg all hold their dataclass default values"
        )

    def test_coefs_parent_not_at_default_when_one_child_differs(self):
        """``input.coefs`` flips to non-default when a single coef changes."""
        cs, values = self._make_loaded_sheet()
        azimuth_iid = self._find_iid(cs, "input.coefs.azimuth_shift_deg")
        assert azimuth_iid is not None, "azimuth_shift_deg leaf missing from _meta"
        # Simulate a user edit: cell 0 departs from its default (181 != 180).
        values[azimuth_iid][0] = 181
        coefs_iid = self._find_iid(cs, "input.coefs")
        assert cs._node_at_default(coefs_iid) is False, (
            "input.coefs parent must NOT be at-default after one leaf scalar departs from its default"
        )

    def test_1d_with_dates_parent_at_default_via_child(self):
        """1D-with-dates parent (kVabs, max_col=0 + len=6) defers to its child row.

        The parent holds date metadata, not array values; the child row holds the
        6 array cells.  If the child cells all match the dataclass default,
        the parent label must read at-default regardless of ``len``.

        Note: ``_ins_1d`` gives the child the SAME path as the parent
        (``input.coefs.kVabs``), so we locate the parent by ``type=='1d'`` +
        ``key=='kVabs'`` rather than by path.
        """
        cs, _ = self._make_loaded_sheet()
        kv_parent = self._find_iid_by(cs, type="1d", key="kVabs")
        assert kv_parent is not None, "kVabs '1d' parent node missing from _meta"
        assert cs._meta[kv_parent].get("max_col") == 0, (
            "kVabs parent must carry max_col=0 (its child row holds the values)"
        )
        assert cs._meta[kv_parent].get("len") == 6, "kVabs parent carries len=6 as array-shape metadata"
        assert cs._node_at_default(kv_parent) is True, (
            "kVabs 1D-with-dates parent must be at-default when its child row "
            "holds the dataclass default array"
        )
