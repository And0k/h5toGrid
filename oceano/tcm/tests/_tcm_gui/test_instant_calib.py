"""Headless regression for :mod:`tcm_gui._instant_calib` — no Tk.

Pins parity with the pipeline (same wrapped entry points) plus the
pending/clear contract the sheet checkboxes rely on: each trigger
(``g0xyz`` / ``coordinates`` / ``azimuth_add``) enables its own box only
when complete, strict parses.
"""

from __future__ import annotations

import numpy as np
import pytest

from tcm_gui import _instant_calib as ic

_AG = np.eye(3) * 0.00173
_CG = np.array([10.0, 10.0, 10.0])


def test_state_g0xyz():
    assert ic.g0xyz_state(["", "", ""]) == "empty"
    assert ic.g0xyz_state(["  ", "", ""]) == "empty"
    assert ic.g0xyz_state(["1", "2", "3"]) == "ready"
    assert ic.g0xyz_state(["0", "0", "0"]) == "ready"  # numeric zeros are complete
    assert ic.g0xyz_state(["", "", "0"]) == "incomplete", "blanks make it partial"
    assert ic.g0xyz_state(["1", "", ""]) == "incomplete", "partial must not enable"
    assert ic.g0xyz_state(["1", "abc", "3"]) == "incomplete", "non-numeric must not enable"
    assert not ic.is_g0xyz_pending(["1", "", ""])
    assert ic.is_g0xyz_pending(["1", "2", "3"])


def test_state_coords():
    assert ic.coords_state(["", ""]) == "empty"
    assert ic.coords_state(["  ", ""]) == "empty"
    assert ic.coords_state(["54.7", "20.5"]) == "ready", "pair applies on its own"
    assert ic.coords_state(["", "1"]) == "incomplete", "partial pair must not enable"
    assert ic.coords_state(["54.7", "abc"]) == "incomplete", "non-numeric must not enable"
    assert not ic.is_coords_pending(["", "1"])
    assert ic.is_coords_pending(["54.7", "20.5"])


def test_state_add():
    assert ic.add_state("") == "empty"
    assert ic.add_state("0") == "empty", "schema default counts as empty"
    assert ic.add_state("0.0") == "empty"
    assert ic.add_state("2.5") == "ready", "offset applies on its own"
    assert ic.add_state("abc") == "incomplete", "bad add must not enable"
    assert not ic.is_add_pending("")
    assert ic.is_add_pending("2.5")


def test_parse_strict():
    assert ic.parse_g0xyz(["1", "2", "3"]) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="input.calib.g0xyz"):
        ic.parse_g0xyz(["1", "", "3"])
    assert ic.parse_coords(["", ""]) is None
    assert ic.parse_coords(["54.7", "20.5"]) == [54.7, 20.5]
    with pytest.raises(ValueError, match="lat='' lon='1'"):
        ic.parse_coords(["", "1"])
    assert ic.parse_add("") is None
    assert ic.parse_add("0") is None
    assert ic.parse_add("2.5") == 2.5
    with pytest.raises(ValueError, match="input.calib.azimuth_add"):
        ic.parse_add("abc")


def test_rz_matches_pipeline():
    """Same entry point as prepare_coefs g0xyz path → 3×3 rotation."""
    from tcm._xr.coefs import get_coef_zeroing_matrix

    expect, _ = get_coef_zeroing_matrix(Rz=None, g0xyz=np.array([0.1, 0.2, 9.8]), Ag=_AG, Cg=_CG)
    got = ic.rz_from_g0xyz([0.1, 0.2, 9.8], _AG, _CG)
    assert got.shape == (3, 3)
    np.testing.assert_allclose(got, np.asarray(expect))


def test_shift_adds_once():
    assert ic.shift_with_tuning(180.0, 2.5, None) == pytest.approx(182.5)
    assert ic.shift_with_tuning(180.0, None, None) == pytest.approx(180.0)
