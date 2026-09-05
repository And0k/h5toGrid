"""Anchor discovery for trigger dirs holding ``_raw`` descendants — delegates to ``meta_finder`` blocks.

Reuses ``meta_finder.file_finder.find_device_dirs`` / ``is_valid_device_dir``
directly (no pattern copy) so ``CTD_SAIV…`` and ``DOC/GRIDDING`` are
excluded by the same block ``meta_finder`` uses. No unfiltered ``rglob``.
"""

from __future__ import annotations

from pathlib import Path

from utils import log_init

from tcm import _constants, config_yaml, paths

lf = log_init.LoggingStyleAdapter(__name__)


def _anchors_via_meta_finder(root: Path) -> list[Path] | None:
    """Filtered anchor discovery via ``meta_finder`` — returns ``[_raw]`` anchors or ``None``.

    Uses ``find_device_dirs`` (1-level ``iterdir`` filtered by
    ``ptn_dir_exclude`` + ``ptn_device_dir_search`` + ``is_valid_device_dir``).
    No file reads. Returns ``None`` when no device dirs found.
    """
    from meta_finder.file_finder import find_device_dirs, is_valid_device_dir

    device_dirs = find_device_dirs(Path(root))
    if not device_dirs:
        # Root itself may be a device dir (e.g. ``251201_ABP64@i,t-chain``)
        if is_valid_device_dir(Path(root)) and (Path(root) / _constants.RAW_DIR_NAME).is_dir():
            return [(Path(root) / _constants.RAW_DIR_NAME).resolve()]
        return None
    anchors: list[Path] = []
    for dd in device_dirs:
        raw = Path(dd) / _constants.RAW_DIR_NAME
        if raw.is_dir():
            anchors.append(raw.resolve())
        elif dd.name.lower() == _constants.RAW_DIR_NAME.lower() and dd.is_dir():
            anchors.append(Path(dd).resolve())
    return sorted(set(anchors)) if anchors else None


def collect_anchors(path_in: Path, dir_raw: Path) -> list[Path]:
    """Return ``_raw`` anchors for *path_in*: one element normally, several when trigger dir holds N>1 ``_raw`` descendants."""
    path_in = Path(path_in)
    dir_raw = Path(dir_raw)
    lf.info("collect_anchors trigger={} dir_raw={}", path_in, dir_raw)
    # Single _raw anchor — no meta_finder, just return it (tab-fill must not trigger device discovery)
    try:
        resolved_in = path_in.expanduser().resolve()
    except Exception:
        resolved_in = path_in
    if path_in.is_dir() and resolved_in.name.lower() == _constants.RAW_DIR_NAME.lower():
        lf.debug("collect_anchors: single _raw trigger={} → [dir_raw]", path_in)
        return [dir_raw]
    if resolved_in != dir_raw.resolve() or not path_in.is_dir():
        lf.debug("collect_anchors: single-source trigger={} → [dir_raw]", path_in)
        return [dir_raw]

    # Trigger dir without _raw ancestor (equals fallback dir_raw) → may hold multiple _raw descendants.
    # Filtered discovery via meta_finder only — no rglob fallback.
    try:
        mf_anchors = _anchors_via_meta_finder(path_in)
    except Exception as exc:
        # meta_finder unavailable — fail fast, do not fabricate anchors.
        raise FileNotFoundError(f"Anchor discovery requires meta_finder (trigger dir={path_in}): {exc}") from exc
    if mf_anchors is not None:
        lf.info(
            "Trigger dir contains {} _raw anchors via meta_finder: {}",
            len(mf_anchors),
            ", ".join(str(a) for a in mf_anchors),
        )
        return mf_anchors
    # No device dirs found — treat as single anchor (caller will enumerate and raise if empty).
    return [dir_raw]


def merge_existed_cfgs(anchors: list[Path]) -> tuple[dict[str, list[str]], Path]:
    """Merge ``{pcid: [stems]}`` from all *anchors*' ``cfg_proc/run`` + primary ``dir_cfgs``."""
    if not anchors:
        raise ValueError("anchors must not be empty")
    merged: dict[str, list[str]] = {}
    for a in anchors:
        if (d := a / "cfg_proc" / "run").is_dir():
            for k, v in config_yaml.get_existed_cfgs(d).items():
                merged.setdefault(k, []).extend(v)
    for k in merged:
        merged[k] = sorted(set(merged[k]))
    return merged, anchors[0] / "cfg_proc" / "run"
