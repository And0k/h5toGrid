"""Burst metadata helpers — GET (scan) vs WRITE (run).

Bursts (``burst_dt`` idx 8, ``bursts_t`` idx 9) are deployment metadata
(``info_devices.yaml``), not ``input``.  Scan is GET-only (log + return for
GUI autofill); actual run is WRITE (merge into ``info_devices.yaml``).

Both paths reuse :func:`meta_finder.data_proc_funcs.extract_time_info_from_text_file`
gap ``> max(10, 2·avg)`` with ``avg=2`` fallback — no reimplementation.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import OmegaConf
from utils import log_init

from tcm import _constants, format

lf = log_init.LoggingStyleAdapter(__name__)

_PLACEHOLDERS = frozenset({"?", "-", "", None})


def _is_placeholder(v: Any) -> bool:
    return v in _PLACEHOLDERS


def _extract_burst(dir_archive: Path, rel: PurePosixPath, avg: Any) -> tuple[Any, Any] | None:
    """Call :func:`extract_time_info_from_text_file` and return ``(burst_dt, bursts_t)``."""
    try:
        from meta_finder.data_proc_funcs import extract_time_info_from_text_file as _eti
    except ImportError:
        return None
    try:
        info = _eti(dir_archive, rel, averaging_interval=avg or 2)
        if info:
            _, _, burst_dt, bursts_t = info
            return burst_dt, bursts_t
    except Exception:
        lf.debug("Burst extract failed for {}/{}", dir_archive, rel, exc_info=True)
    return None


def _resolve_burst_path(
    path_str: str, dir_raw: Path
) -> tuple[Path, PurePosixPath] | None:
    """Resolve ``input.path`` (loose or archive composite) to ``(dir_archive, rel)``."""
    from tcm.search import is_archive_composite, split_archive_path

    p = Path(str(path_str))
    if is_archive_composite(p):
        sp = split_archive_path(p)
        if sp is not None:
            return sp
        # Fallback: posix string "…/arch.zip/inner.txt" written via as_posix()
        s = str(path_str)
        for ext in (".zip", ".7z"):
            idx = s.lower().find(ext + "/")
            if idx != -1:
                return Path(s[: idx + len(ext)]), PurePosixPath(s[idx + len(ext) + 1 :])
        return None
    # Loose — parent may be posix on Windows, resolve exists else fallback to dir_raw
    fp = Path(str(path_str))
    dir_arc = fp.parent if fp.parent.is_dir() else dir_raw
    # When fp is archive composite written as posix but is_archive_composite missed (case)
    # dir_arc.is_dir() check above already handles missing dirs
    return dir_arc, PurePosixPath(fp.name)


def collect_bursts(dir_raw: Path, collected: list[tuple[str, str, Any]]) -> dict[str, tuple[Any, Any]]:
    """GET missing ``burst_dt/bursts_t`` for display (scan) — no file write."""
    out: dict[str, tuple[Any, Any]] = {}
    if not collected:
        return out
    lf.debug("Burst GET: scanning {} stems", len(collected))
    for stem, _, cfg_dc in collected:
        try:
            cfg = OmegaConf.to_container(cfg_dc, resolve=True) if OmegaConf.is_config(cfg_dc) else dict(cfg_dc or {})
        except Exception:
            continue
        path_str = (cfg.get("input", {}) or {}).get("path", "")
        if not path_str:
            continue
        avg = (cfg.get("input", {}) or {}).get("averaging_interval") or 2
        resolved = _resolve_burst_path(path_str, dir_raw)
        if resolved is None:
            continue
        dir_arc, rel = resolved
        burst = _extract_burst(dir_arc, rel, avg)
        if burst is None:
            continue
        burst_dt, bursts_t = burst
        out[stem] = (burst_dt, bursts_t)
        if burst_dt != "-" or bursts_t != "-":
            lf.info("Burst GET for {} ({}): burst_dt={} bursts_t={}", stem, Path(str(path_str)).name, burst_dt, bursts_t)
        else:
            lf.debug("Burst GET for {}: continuous (-/-) in {}", stem, Path(str(path_str)).name)
    return out


def fill_missing_bursts(dir_raw: Path, collected: list[tuple[str, str, Any]]) -> None:
    """WRITE missing ``burst_dt/bursts_t`` to ``info_devices.yaml`` (actual run)."""
    if not collected:
        return
    try:
        from meta_finder import io_info_files as _io
    except ImportError:
        lf.debug("meta_finder not available — skip burst autofill")
        return

    device_dir = dir_raw.parent if dir_raw.name == _constants.RAW_DIR_NAME else dir_raw
    info_file: Path | None = None
    existing: dict = {}
    for nm in ("info_devices.yaml", "info_devices.json"):
        cand = device_dir / nm
        if cand.is_file():
            info_file = cand
            try:
                existing = _io.read_metadata_file(cand)
            except Exception:
                lf.warning("Failed to read {} — will overwrite", cand, exc_info=True)
                existing = {}
            break
    if info_file is None:
        info_file = device_dir / "info_devices.yaml"

    stems = [s for s, _, _ in collected]
    stems_by_pcid: dict[str, list[str]] = {}
    for s in stems:
        try:
            pc = format.to_pcid_from_name(format.stem_to_pcid(s))
        except Exception:
            pc = s
        stems_by_pcid.setdefault(pc, []).append(s)
    for v in stems_by_pcid.values():
        v.sort()

    updated = False
    new_content: dict[str, dict[str, list[Any]]] = {}
    for stem, yaml_path_str, cfg_dc in collected:
        try:
            pcid = format.to_pcid_from_name(format.stem_to_pcid(stem))
        except Exception:
            continue
        pc_list = stems_by_pcid.get(pcid, [stem])
        sid = str(pc_list.index(stem)) if stem in pc_list else "0"

        ent = existing.get(pcid) or existing.get(pcid.replace("_", ""))
        base_arr: list[Any] | None = None
        if isinstance(ent, dict):
            if sid in ent and isinstance(ent[sid], (list, tuple)):
                base_arr = list(ent[sid])
            elif "0" in ent and isinstance(ent["0"], (list, tuple)):
                base_arr = list(ent["0"])
            else:
                for _k, _v in ent.items():
                    if isinstance(_v, (list, tuple)):
                        base_arr = list(_v)
                        break
        elif isinstance(ent, (list, tuple)):
            base_arr = list(ent)

        if base_arr is not None and len(base_arr) > 9:
            cur_bdt = base_arr[8] if len(base_arr) > 8 else None
            cur_bst = base_arr[9] if len(base_arr) > 9 else None
            if not _is_placeholder(cur_bdt) and not _is_placeholder(cur_bst):
                continue

        cfg: dict[str, Any] = {}
        if cfg_dc is not None:
            try:
                cfg = OmegaConf.to_container(cfg_dc, resolve=True) if OmegaConf.is_config(cfg_dc) else dict(cfg_dc or {})
            except Exception:
                cfg = {}
        if not cfg and yaml_path_str:
            try:
                from ruamel.yaml import YAML as _Y

                _ry = _Y(typ="safe", pure=True)
                with Path(yaml_path_str).open(encoding="utf-8") as _fp:
                    _y = _ry.load(_fp) or {}
                cfg = {"input": _y.get("input", {})}
            except Exception:
                cfg = {}
        path_str = (cfg.get("input", {}) or {}).get("path", "")
        if not path_str:
            continue
        avg = (cfg.get("input", {}) or {}).get("averaging_interval") or 2
        resolved = _resolve_burst_path(path_str, dir_raw)
        if resolved is None:
            continue
        dir_arc, rel = resolved
        burst = _extract_burst(dir_arc, rel, avg)
        if burst is None:
            continue
        burst_dt, bursts_t = burst

        need = base_arr is None or _is_placeholder(base_arr[8] if len(base_arr) > 8 else None) or _is_placeholder(
            base_arr[9] if len(base_arr) > 9 else None
        )
        if base_arr is not None and not need:
            # Check string difference to avoid no-op write of "-"/"-"
            cur_bdt = base_arr[8] if len(base_arr) > 8 else None
            cur_bst = base_arr[9] if len(base_arr) > 9 else None
            if str(cur_bdt) == str(burst_dt) and str(cur_bst) == str(bursts_t):
                lf.debug("Burst for {} already set ({} / {}) — keep", pcid, cur_bdt, cur_bst)
                continue
            need = True
        if not need:
            continue

        lf.info("Burst for {} ({}): burst_dt={} bursts_t={} from {}", pcid, stem, burst_dt, bursts_t, Path(str(path_str)).name)
        if base_arr is None:
            base_arr = [None] * 11
            tr = (cfg.get("input", {}) or {}).get("time_ranges") or []
            if len(tr) >= 2:
                base_arr[6], base_arr[7] = tr[0], tr[1]
        elif len(base_arr) < 11:
            base_arr = list(base_arr) + [None] * (11 - len(base_arr))
        base_arr[8] = burst_dt
        base_arr[9] = bursts_t
        new_content.setdefault(pcid, {})[sid] = base_arr
        updated = True

    if not updated or not new_content:
        lf.debug("No missing bursts to fill in {}", device_dir)
        return

    try:
        from meta_finder.create_info_files import _merge_device_metadata
    except ImportError:
        _merge_device_metadata = None  # type: ignore[assignment]

    merged = dict(existing) if existing else {}
    if existing and _merge_device_metadata is not None:
        try:
            merged = _merge_device_metadata(existing, new_content)
            for pcid, sids in new_content.items():
                for sid, arr in sids.items():
                    if pcid in merged and isinstance(merged[pcid], dict) and sid in merged[pcid]:
                        merged[pcid][sid] = arr
                    elif pcid in merged:
                        if isinstance(merged[pcid], dict):
                            merged[pcid][sid] = arr
                        else:
                            merged[pcid] = {sid: arr}
                    else:
                        merged[pcid] = {sid: arr}
        except Exception:
            lf.warning("Merge failed — overwriting with new_content", exc_info=True)
            merged = {**existing, **new_content}
    else:
        for pcid, sids in new_content.items():
            if pcid not in merged:
                merged[pcid] = sids
            elif isinstance(merged[pcid], dict):
                merged[pcid].update(sids)
            else:
                merged[pcid] = sids

    try:
        _io.write_metadata_file(device_dir, info_file, merged)
        lf.info("Wrote burst metadata for {} to {}", ", ".join(new_content), info_file.name)
    except Exception:
        lf.exception("Failed to write burst metadata to {}", info_file)
