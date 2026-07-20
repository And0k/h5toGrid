"""Device metadata helpers extracted from veusz_helpers.common.metadata.

Provides :func:`get_path_in_parents`, :func:`load_file_meta`, and
:func:`extract_devices_info` — the only three functions from the external
``veusz_helpers`` package used by TCM.

The original module depended on ``func_vsz.DictKeyIfNoVal`` (a Veusz-specific
translation dict with a module-level Windows-registry read).  For the TCM
distribution we replace it with :class:`_KeyAsDefault`, which returns the key
itself when missing — identical to the English-locale behaviour of the original.
"""

from pathlib import Path
from typing import Dict, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _KeyAsDefault(dict):
    """Return *key* itself when key is missing — replaces veusz_helpers.DictKeyIfNoVal.

    Used as the *mapping* argument to :meth:`str.format_map` so that any
    ``{name}`` placeholders in the pressure string are replaced with ``name``
    (braces stripped), matching the English-locale behaviour of the original.
    """

    def __missing__(self, key: str) -> str:
        return key


_I = _KeyAsDefault()


def _meta_array_to_dict(
    p, b, bd, s, lat=None, lon=None, time_st="", time_en="", burst_dt=None, bursts_t=None
) -> dict:
    """Convert a flat metadata array into a labelled dict.

    Keys follow the Veusz ``vsz_drawer`` convention:
    ``p`` (pressure), ``b`` (magnetic field), ``bd`` (declination),
    ``s`` (sound speed), ``c`` (coordinates), ``r`` (time range),
    ``t`` (burst_dt), ``T`` (bursts_t).
    """
    return dict(
        zip(
            "pbdscrtT",
            [
                p.format_map(_I),
                b,
                None if None in (b, bd) else round(b - bd, 1),
                s,
            ]
            + ([(lat, lon)] if lat else [])
            + ([(time_st, time_en)] if time_st else [])
            + ([burst_dt, bursts_t] if bursts_t else []),
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_path_in_parents(dir: Path, *file_names, target_is_dir=False) -> Path:
    """Locate *file_names* by ascending through parent directories.

    :param dir: starting child directory path.
    :param file_names: candidate file (or directory) names — first match wins.
    :param target_is_dir: when *True*, match directories instead of files.
    :returns: resolved :class:`Path` of the first match.
    :raises FileNotFoundError: when no match is found up to the filesystem root.
    """
    while True:
        for file_name in file_names:
            file = dir / file_name
            if file.is_dir() if target_is_dir else file.is_file():
                return file
        dir_parent = dir.parent
        if dir != dir_parent:
            dir = dir_parent
        else:
            raise FileNotFoundError(str(file_names))


def load_file_meta(path_in: Path) -> dict:
    """Load device metadata from a YAML or JSON file.

    YAML files may use nested dicts per station-id — these are collapsed
    into a single flat list per device (first/last/min/max aggregation).
    """
    with path_in.open(encoding="utf8") as f:
        if path_in.suffix == ".yaml":
            from yaml import safe_load

            content = safe_load(f.read())
            return {
                device_id: [
                    seq[0]
                    if all(s == seq[0] for s in seq)
                    else min(seq)
                    if col == "time_st"
                    else max(seq)
                    if col == "time_en"
                    else np.mean(seq)
                    if col in ["lat", "lon"]
                    else ",".join(str(s) for s in seq)
                    for col, seq in zip(
                        ["p", "b", "bd", "s", "lat", "lon", "time_st", "time_en", "burst_dt", "bursts_t"],
                        zip(*meta.values()),
                    )
                ]
                if isinstance(meta, dict)
                else meta
                for device_id, meta in content.items()
            }
        else:
            import json

            return json.load(f)


def extract_devices_info(meta: dict, devices: Sequence[str]) -> dict:
    """Map probe IDs to their metadata entries.

    :param meta: ``{device_id: [p, b, bd, s, …]}`` from :func:`load_file_meta`.
    :param devices: probe IDs to look up (e.g. ``["i01", "i02"]``).
    :returns: ``{pid: {p: …, b: …, …}}`` for each matching *devices* entry.
    """
    device_info: Dict[str, dict] = {}
    for pid_cur in devices:
        try:
            pid_array = meta[pid_cur]
        except KeyError:
            if not pid_cur or pid_cur[0] == "i":
                continue
            try:
                pid_array = meta[f"i{pid_cur}"]
            except KeyError:
                continue
        device_info[pid_cur] = _meta_array_to_dict(*pid_array)
    return device_info
