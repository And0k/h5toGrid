"""Mapping between ``info_devices.yaml`` 11-array and GUI paired ``metadata`` rows."""

from __future__ import annotations

from typing import Any

# Display pairing: label → tuple of 11-array indices (single index → comment).
PAIRS: list[tuple[str, tuple[int, ...]]] = [
    ("point, symbol", (0, 3)),
    ("sea depth, h_above", (1, 2)),
    ("lat, lon", (4, 5)),
    ("time_range", (6, 7)),
    ("burst_dt/t", (8, 9)),
    ("comment", (10,)),
]

REQUIRED_LAST = 7  # time_en — required slice 0..7 writes ~ even when placeholder
PLACEHOLDER_DISPLAY = "?"

# 11-array indices holding numeric fields (sea_depth, h_above, lat, lon,
# burst_dt, bursts_t) — per meta_finder/io_info_files.info_devices_field_names_extended.
NUMERIC_IDXS = frozenset({1, 2, 4, 5, 8, 9})

# Example values for gray placeholder hints (realistic, from user's info_devices.yaml).
EXAMPLES: dict[str, list[str]] = {
    "point, symbol": ["P3", "↟"],
    "sea depth, h_above": ["7.5", "0"],
    "lat, lon": ["54.62", "19.84"],
    "time_range": ["YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DDTHH:MM:SS"],
    "burst_dt/t": ["60", "600"],
    "comment": ["deployment note"],
}


def is_placeholder(v: Any) -> bool:
    return v in ("?", "-", "", None)


def to_display_val(v: Any) -> str:
    return PLACEHOLDER_DISPLAY if is_placeholder(v) else str(v)


def to_display(list11: list[Any] | tuple[Any, ...]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for label, idxs in PAIRS:
        out[label] = [to_display_val(list11[i] if i < len(list11) else None) for i in idxs]
    return out


def display_to_val(s: str) -> Any:
    t = s.strip() if isinstance(s, str) else s
    if t in (PLACEHOLDER_DISPLAY, "-", ""):
        return None
    try:
        fv = float(t)
        return int(fv) if "." not in t and "e" not in t.lower() and fv.is_integer() else fv
    except (TypeError, ValueError):
        return t


def to_storage(paired: dict[str, list[str]], base: list[Any] | None = None) -> list[Any]:
    arr: list[Any] = list(base) if base is not None else [None] * 11
    if len(arr) < 11:
        arr.extend([None] * (11 - len(arr)))
    for label, idxs in PAIRS:
        vals = paired.get(label, [])
        for j, idx in enumerate(idxs):
            if j < len(vals):
                arr[idx] = display_to_val(vals[j])
    return arr[:11]


def station_id(stem: str, stems_sorted: list[str]) -> str:
    try:
        return str(stems_sorted.index(stem))
    except ValueError:
        return "0"
