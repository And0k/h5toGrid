"""Headless regression for :mod:`tcm_gui._instant_calib` — no Tk.

Pins parity with the pipeline (same wrapped entry points) plus the
pending/clear contract the sheet checkboxes rely on: ``g0xyz`` pending on
any value, azimuth pending on coords or non-zero add, strict parses.
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


def test_state_azimuth():
    assert ic.azimuth_state(["", ""], "") == "empty"
    assert ic.azimuth_state(["", ""], "0") == "empty"
    assert ic.azimuth_state(["", ""], "0.0") == "empty"
    assert ic.azimuth_state(["", ""], "2.5") == "ready", "lone add applies"
    assert ic.azimuth_state(["54.7", "20.5"], "") == "ready", "lone coords apply"
    assert ic.azimuth_state(["54.7", "20.5"], "1") == "ready"
    assert ic.azimuth_state(["", "1"], "") == "incomplete", "partial coords must not enable"
    assert ic.azimuth_state(["", "1"], "2.5") == "incomplete", "partial coords poison group"
    assert ic.azimuth_state(["", ""], "abc") == "incomplete", "bad add must not enable"
    assert ic.azimuth_state(["54.7", "abc"], "") == "incomplete"
    assert not ic.is_azimuth_pending(["", "1"], "")
    assert ic.is_azimuth_pending(["", ""], "2.5")


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
