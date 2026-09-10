"""Instant in-sheet calibration — data-independent triggers, no Tk.

Pure-logic sidekick for the ``input.calib`` apply checkboxes in
:class:`tcm_gui.coef_sheet.ConfigSheet`. Wraps the exact pipeline entry
points used by :func:`tcm._xr.coefs.prepare_coefs` so GUI instant-apply
and Run-time calibration cannot diverge:

* ``g0xyz → Rz`` via :func:`tcm._xr.coefs.get_coef_zeroing_matrix`
  (needs ``Ag``/``Cg`` only — no dataset read).
* ``coordinates``/``azimuth_add → azimuth_shift_deg`` via
  :func:`tcm.incl_calc.coefs.get_coef_azimuth_shift` (no dataset read).

``time_ranges_zeroing`` / ``time_ranges_azimuth`` are deliberately absent —
they need ``ds_raw`` windows and stay Run-time only.

Pending semantics mirror the sheet. Three states per trigger:

* ``empty`` — no data (box disabled ☑, synced);
* ``ready`` — complete + numeric (box enabled ☐, click applies);
* ``incomplete`` — partial or non-numeric (box disabled ☐, no misleading
  error — fill or clear to proceed).

``g0xyz`` is ready with 3 numerics, ``coordinates`` with 2 numerics,
``azimuth_add`` with a numeric non-zero value (schema default 0 counts as
empty). Each trigger applies and clears fully on its own: the pipeline
layering is additive, so split application commutes with the combined one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Literal

import numpy as np
from utils import log_init

from tcm_gui._cell_spec import parse_float

lf = log_init.LoggingStyleAdapter(__name__)

State = Literal["empty", "incomplete", "ready"]

G0XYZ_N: Final = 3
COORDS_N: Final = 2


def g0xyz_state(cells: Sequence[str]) -> State:
    """Completeness of the g0xyz cells."""
    strs = [(c or "").strip() for c in list(cells)[:G0XYZ_N]]
    if not any(strs):
        return "empty"
    vals = [parse_float(c) for c in strs]
    if len(vals) < G0XYZ_N or any(v is None for v in vals):
        return "incomplete"
    return "ready"


def coords_state(cells: Sequence[str]) -> State:
    """Completeness of the coordinates pair."""
    strs = [(c or "").strip() for c in list(cells)[:COORDS_N]]
    if not any(strs):
        return "empty"
    vals = [parse_float(c) for c in strs]
    if len(vals) < COORDS_N or any(v is None for v in vals):
        return "incomplete"  # partial/non-numeric pair never applies
    return "ready"


def add_state(s: str | None) -> State:
    """Completeness of the azimuth_add scalar (default 0 counts as empty)."""
    t = (s or "").strip()
    if not t:
        return "empty"
    v = parse_float(t)
    if v is None:
        return "incomplete"  # non-numeric add never applies
    return "empty" if v == 0 else "ready"


def is_g0xyz_pending(cells: Sequence[str]) -> bool:
    """True when g0xyz is complete and awaits apply."""
    return g0xyz_state(cells) == "ready"


def is_coords_pending(cells: Sequence[str]) -> bool:
    """True when coordinates are complete and await apply."""
    return coords_state(cells) == "ready"


def is_add_pending(s: str | None) -> bool:
    """True when azimuth_add is set and awaits apply."""
    return add_state(s) == "ready"


def parse_g0xyz(cells: Sequence[str]) -> list[float]:
    """Strict g0xyz parse of ``G0XYZ_N`` floats; raises :exc:`ValueError`."""
    vals = [parse_float(c) for c in list(cells)[:G0XYZ_N]]
    if len(vals) < G0XYZ_N or any(v is None for v in vals):
        raise ValueError(
            f"input.calib.g0xyz needs [Ax, Ay, Az] floats (e.g. [100.5, 50.2, 980.1]); got {list(cells)!r}"
        )
    return [float(v) for v in vals]  # type: ignore[misc]


def parse_coords(cells: Sequence[str]) -> list[float] | None:
    """Strict [lat, lon] parse; None when both blank; raises on partial."""
    strs = [(c or "").strip() for c in list(cells)[:COORDS_N]]
    if not any(strs):
        return None
    vals = [parse_float(c) for c in strs]
    if len(vals) < COORDS_N or any(v is None for v in vals):
        lat, lon = (strs + ["", ""])[:COORDS_N]
        raise ValueError(
            "input.calib.coordinates needs [lat, lon] decimal degrees (e.g. [54.70, 20.51]); "
            f"got lat={lat!r} lon={lon!r}"
        )
    return [float(v) for v in vals]  # type: ignore[misc]


def parse_add(s: str | None) -> float | None:
    """Parse azimuth_add; None when blank/default 0; raises on garbage."""
    t = (s or "").strip()
    if not t:
        return None
    v = parse_float(t)
    if v is None:
        raise ValueError(f"input.calib.azimuth_add must be numeric degrees; got {s!r}")
    return None if v == 0 else float(v)


def rz_from_g0xyz(g0xyz: Sequence[float], Ag: Any, Cg: Any) -> np.ndarray:
    """Rotation matrix from zero-tilt vector — same call as the pipeline."""
    from tcm._xr.coefs import get_coef_zeroing_matrix

    Rz, _ = get_coef_zeroing_matrix(
        Rz=None,
        g0xyz=np.asarray(list(g0xyz), dtype=np.float64),
        Ag=np.asarray(Ag, dtype=np.float64),
        Cg=np.asarray(Cg, dtype=np.float64),
    )
    if Rz is None:
        raise ValueError("zeroing matrix not produced (check Ag/Cg/g0xyz)")
    return np.asarray(Rz, dtype=np.float64)


def shift_with_tuning(base_shift: float, azimuth_add: float | None, coords: Sequence[float] | None) -> float:
    """Azimuth shift with manual tuning — same call as the pipeline."""
    from tcm.incl_calc.coefs import get_coef_azimuth_shift

    out = get_coef_azimuth_shift(azimuth_add, tuple(coords) if coords else None, float(base_shift))
    return float(np.asarray(out).item() if isinstance(out, np.ndarray) else out)
