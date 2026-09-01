"""Tests for generalized red-fg cell validation (``_apply_validations``).

Covers ``check: "exists"`` rows (``input.path``, ``input.coefs.path``): any
cell whose committed path doesn't exist on disk gets red foreground, while
the Run button stays gated on ``input.path`` only (``coefs_path`` is
optional — coefficients may be entered manually).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import tcm_gui.coef_sheet as coef_sheet
import tcm_gui.theme as theme


def _make_sheet(path: str, coefs_path: str):
    """ConfigSheet with the given input/coefs_path values (mocked tksheet).

    ``insert`` records per-row cell values so ``sh.item(iid).get("values")``
    (the read used by ``_apply_validations`` / ``is_path_valid``) resolves.
    """
    from tcm.schema import Config, Return
    from tcm_gui.coef_sheet import ConfigSheet

    cfg = {
        "input": {
            "path": path,
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

    def test_target_iid_limits_pass(self, tmp_path):
        """A targeted pass re-validates only the edited row."""
        missing = str(tmp_path / "nope")
        cs, mock_sh = _make_sheet(missing, missing)
        in_iid = _iid(cs, type="input")
        cs._apply_validations(in_iid)
        red = _red_calls(mock_sh)
        assert len(red) == 1
        assert red[0].kwargs["row"] == cs._row_map()[in_iid]


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
