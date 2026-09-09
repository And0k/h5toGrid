"""Tests for generalized red-fg cell validation (``_apply_validations``).

Covers ``check: "exists"`` rows (``input.path``, ``input.coefs.path``): any
cell whose committed path doesn't exist on disk gets red foreground, while
the Run button stays gated on ``input.path`` only (``coefs_path`` is
optional — coefficients may be entered manually).  ``check: "sorted"``
rows (``input.time_ranges``, ``metadata.time_range``) get red foreground on
any date cell breaking the ascending order of the sequence.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import tcm_gui.coef_sheet as coef_sheet
import tcm_gui.theme as theme


def _make_sheet(path: str, coefs_path: str, *, time_ranges: list[str] | None = None):
    """ConfigSheet with the given input/coefs_path values (mocked tksheet).

    ``insert`` records per-row cell values so ``sh.item(iid).get("values")``
    (the read used by ``_apply_validations`` / ``is_path_valid``) resolves.
    """
    from tcm.schema import Config, Return
    from tcm_gui.coef_sheet import ConfigSheet

    cfg = {
        "input": {
            "path": path,
            "time_ranges": time_ranges,
            "coefs": {"path": coefs_path, "Ag": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
        }
    }
    mock_sh = MagicMock()
    mock_sh.total_columns.return_value = 6
    mock_sh.total_rows.return_value = 0
    _kids: dict[Any, list[str]] = {}
    _vals: dict[Any, tuple] = {}
    _n = 0  # monotonic iid counter — len(_kids) would collide (key ≠ row)

    def _insert(**kw):
        nonlocal _n
        _n += 1
        iid = f"iid_{_n}"
        _kids.setdefault(kw.get("parent") or "", []).append(iid)
        _vals[iid] = tuple(kw.get("values") or ())
        return iid

    mock_sh.insert.side_effect = _insert
    mock_sh.get_children.side_effect = lambda parent="": list(_kids.get(parent or "", ()))
    mock_sh.item.side_effect = lambda iid: {"values": _vals.get(iid, ("",))}
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
    cs._snap = ({}, {}, "")
    cs._fg_default = "#000000"
    cs._readonly = False
    cs._int_row_of = {}
    cs._vis = ()
    cs._col_resize = MagicMock()
    cs.on_validity_change = None  # __new__ bypasses __init__ defaults
    cs._build_coefs(cfg)
    for m in cs._meta.values():
        m["open"] = True
    return cs, mock_sh


def _red_calls(mock_sh) -> list:
    """``highlight_cells`` calls painted with the error foreground."""
    return [c for c in mock_sh.highlight_cells.call_args_list if c.kwargs.get("fg") == theme.INVALID_FG]


def _iid(cs, **meta) -> Any:
    return next(iid for iid, m in cs._meta.items() if all(m.get(k) == v for k, v in meta.items()))


class TestCheckMarker:
    def test_checked_rows_only(self, tmp_path):
        """``check: "exists"`` lands on exactly input.path and coefs rows."""
        cs, _ = _make_sheet(str(tmp_path), str(tmp_path))
        checked = [iid for iid, m in cs._meta.items() if m.get("check") == "exists"]
        assert len(checked) == 2
        assert {cs._meta[i]["key"] for i in checked} == {"input", "coefs"}


class TestApplyValidations:
    def test_missing_coefs_path_colored_red(self, tmp_path):
        """Invalid coefs_path → red fg on its row (valid input.path stays normal)."""
        cs, mock_sh = _make_sheet(str(tmp_path), str(tmp_path / "nope"))
        cs._apply_validations()
        red = _red_calls(mock_sh)
        cp_iid = _iid(cs, key="coefs")
        assert len(red) == 1
        assert red[0].kwargs["row"] == cs._row_map()[cp_iid]
        assert red[0].kwargs["column"] == 0

    def test_both_missing_both_red(self, tmp_path):
        missing = str(tmp_path / "nope")
        cs, mock_sh = _make_sheet(missing, missing)
        cs._apply_validations()
        assert len(_red_calls(mock_sh)) == 2

    def test_valid_paths_never_red(self, tmp_path):
        cs, mock_sh = _make_sheet(str(tmp_path), str(tmp_path))
        cs._apply_validations()
        assert _red_calls(mock_sh) == []

    def test_input_path_matching_config_stem_tinted_gray(self, tmp_path):
        """input.path whose stem matches the config stem → gray (belongs to config)."""
        # Create a file whose stem matches the config stem "i_p05"
        data_file = tmp_path / "i_p05.TXT"
        data_file.write_text("dummy")
        cs, mock_sh = _make_sheet(str(data_file), str(tmp_path))
        cs._page_stem = "i_p05"  # config stem matches the input file stem
        in_iid = _iid(cs, type="input")
        cs._apply_validations(in_iid)
        gray = [
            c
            for c in mock_sh.highlight_cells.call_args_list
            if c.kwargs.get("fg") == theme.CELL_DEFAULT_VAL_FG
            and c.kwargs.get("row") == cs._row_map()[in_iid]
        ]
        assert len(gray) == 1, (
            f"matching input.path should be gray, got: {mock_sh.highlight_cells.call_args_list}"
        )

    def test_input_path_same_identity_different_file_still_gray(self, tmp_path):
        """A different file with the same canonical pcid (``i_p05`` → ``i_p5``) reads as default.

        ``pcid_key`` normalizes ``i_p05`` ≡ ``i_p5`` (same pcid + no comment) — a
        renamed/format-different raw file of the SAME probe keeps the gray tint.
        """
        cs, mock_sh = _make_sheet(str(tmp_path / "i_p05.TXT"), str(tmp_path))
        cs._page_stem = "i_p05"
        in_iid = _iid(cs, type="input")
        in_row = cs._row_map()[in_iid]
        data_file = tmp_path / "i_p5.TXT"
        data_file.write_text("dummy")
        mock_sh.item.side_effect = lambda iid, **kw: {"values": (str(data_file), "")}
        cs._apply_validations(in_iid)
        gray = [
            c
            for c in mock_sh.highlight_cells.call_args_list
            if c.kwargs.get("fg") == theme.CELL_DEFAULT_VAL_FG and c.kwargs.get("row") == in_row
        ]
        assert len(gray) == 1, (
            f"same-identity input.path should still be gray, got: {mock_sh.highlight_cells.call_args_list}"
        )

    def test_input_path_other_probe_not_gray(self, tmp_path):
        """A different-probe file (stem does NOT match) → default fg (not gray)."""
        cs, _ = _make_sheet(str(tmp_path / "i_p05.txt"), str(tmp_path))
        cs._page_stem = "i_p05"
        in_iid = _iid(cs, type="input")
        in_row = cs._row_map()[in_iid]
        data_file = tmp_path / "i_p07.TXT"
        data_file.write_text("dummy")
        cs.sh.item.side_effect = lambda iid, **kw: {"values": (str(data_file), "")}
        cs._apply_validations(in_iid)
        gray = [
            c
            for c in cs.sh.highlight_cells.call_args_list
            if c.kwargs.get("fg") == theme.CELL_DEFAULT_VAL_FG and c.kwargs.get("row") == in_row
        ]
        assert len(gray) == 0, f"other-probe input.path should NOT be gray, got: {gray}"

    def test_target_iid_limits_pass(self, tmp_path):
        """A targeted pass re-validates only the edited row."""
        missing = str(tmp_path / "nope")
        cs, mock_sh = _make_sheet(missing, missing)
        in_iid = _iid(cs, type="input")
        cs._apply_validations(in_iid)
        red = _red_calls(mock_sh)
        assert len(red) == 1
        assert red[0].kwargs["row"] == cs._row_map()[in_iid]

    def test_input_path_empty_restores_loaded_default(self, tmp_path):
        """Entering empty on input.path restores the loaded run-YAML path.

        input.path has no schema default (None); its "default" is the value the
        config was loaded with.  Clearing the cell must restore that value, not
        leave it empty.
        """
        data_file = tmp_path / "i_p05.TXT"
        data_file.write_text("dummy")
        cs, mock_sh = _make_sheet(str(data_file), str(tmp_path))
        in_iid = _iid(cs, type="input")
        in_row = cs._row_map()[in_iid]

        mock_sh.reset_mock()
        cs._apply_edit_value(in_iid, 0, "")

        set_calls = [
            c for c in mock_sh.set_cell_data.call_args_list if c.args[0] == in_row and c.args[1] == 0
        ]
        assert len(set_calls) == 1, f"empty input.path should restore its loaded value, got: {set_calls}"
        assert set_calls[0].args[2] == str(data_file), (
            f"expected loaded path restored, got: {set_calls[0].args[2]}"
        )


class TestIsPathValidGating:
    def test_bad_coefs_path_does_not_gate_run(self, tmp_path):
        """Run stays enabled when only coefs_path is missing (manual coef entry)."""
        cs, _ = _make_sheet(str(tmp_path), str(tmp_path / "nope"))
        assert cs.is_path_valid() is True

    def test_missing_input_path_gates_run(self, tmp_path):
        cs, _ = _make_sheet(str(tmp_path / "nope"), str(tmp_path))
        assert cs.is_path_valid() is False

    def test_empty_input_path_gates_run(self, tmp_path):
        cs, _ = _make_sheet("", str(tmp_path))
        assert cs.is_path_valid() is False


class TestSortedCheck:
    def test_sorted_marker_on_time_ranges_row(self, tmp_path):
        """``check: "sorted"`` lands on the input.time_ranges row."""
        cs, _ = _make_sheet(str(tmp_path), str(tmp_path))
        assert cs._meta[_iid(cs, label="time_ranges")].get("check") == "sorted"

    def test_unsorted_pair_red(self, tmp_path):
        """start > end → both date cells of the pair painted with error fg."""
        cs, mock_sh = _make_sheet(
            str(tmp_path), str(tmp_path), time_ranges=["2024-01-02T00:00:00", "2024-01-01T00:00:00"]
        )
        tr = _iid(cs, label="time_ranges")
        cs._apply_validations(tr)
        cells = {(c.kwargs["row"], c.kwargs["column"]) for c in _red_calls(mock_sh)}
        row = cs._row_map()[tr]
        assert cells == {(row, 0), (row, 1)}, f"expected both cells red, got: {cells}"

    def test_sorted_pair_restored(self, tmp_path):
        """Ascending dates → no red; both cells restored to normal fg."""
        cs, mock_sh = _make_sheet(
            str(tmp_path), str(tmp_path), time_ranges=["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
        )
        tr = _iid(cs, label="time_ranges")
        cs._apply_validations(tr)
        assert _red_calls(mock_sh) == []
        row = cs._row_map()[tr]
        restored = [
            c
            for c in mock_sh.highlight_cells.call_args_list
            if c.kwargs.get("row") == row and c.kwargs.get("fg") == cs._fg_default
        ]
        assert len(restored) == 2, f"expected 2 restored cells, got: {restored}"

    def test_single_and_empty_never_flagged(self, tmp_path):
        """One date (or none) — nothing to compare → no red either way."""
        cs, mock_sh = _make_sheet(str(tmp_path), str(tmp_path), time_ranges=["2024-01-01T00:00:00"])
        cs._apply_validations(_iid(cs, label="time_ranges"))
        assert _red_calls(mock_sh) == []

    def test_metadata_time_range_swapped_red(self, tmp_path):
        """metadata.time_range pair with start > end → both cells red."""
        cs, mock_sh = _make_sheet(str(tmp_path), str(tmp_path))
        tr = cs.sh.insert(values=("2024-01-02T00:00:00", "2024-01-01T00:00:00"))
        cs._meta[tr] = {"label": "time_range", "is_metadata": True, "max_col": 2, "check": "sorted"}
        cs._apply_validations(tr)
        cells = {(c.kwargs["row"], c.kwargs["column"]) for c in _red_calls(mock_sh)}
        row = cs._row_map()[tr]
        assert cells == {(row, 0), (row, 1)}, f"expected both cells red, got: {cells}"

    def test_target_iid_limits_sorted_pass(self, tmp_path):
        """A targeted pass re-validates only the edited (time_ranges) row."""
        cs, mock_sh = _make_sheet(
            str(tmp_path), str(tmp_path), time_ranges=["2024-01-02T00:00:00", "2024-01-01T00:00:00"]
        )
        other = _iid(cs, label="time_ranges")
        # A second sorted row — must stay untouched by the targeted pass
        tr2 = cs.sh.insert(values=("2024-01-03T00:00:00", "2024-01-01T00:00:00"))
        cs._meta[tr2] = {"label": "time_range", "is_metadata": True, "max_col": 2, "check": "sorted"}
        mock_sh.reset_mock()
        cs._apply_validations(other)
        rows = {c.kwargs["row"] for c in _red_calls(mock_sh)}
        assert rows == {cs._row_map()[other]}, f"targeted pass leaked to other rows: {rows}"
