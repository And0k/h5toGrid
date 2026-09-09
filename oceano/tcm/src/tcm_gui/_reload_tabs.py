"""Reload processed GUI tabs from their updated run YAMLs after Run."""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any

import omegaconf

from tcm import config_yaml, format, schema
from tcm_gui import cli_cfg

lf = logging.getLogger(__name__)


def meta_for_stem(stem: str, device_meta: dict | None) -> list | None:
    """Return device metadata groups for *stem* from parsed device metadata."""
    if device_meta is None:
        return None
    try:
        pcid = format.to_pcid_from_name(format.stem_to_pcid(stem))
    except Exception:
        return None
    # Try normalized keys (meta_finder stores normalized ids)
    for cand in (pcid, pcid.replace("_", "")):
        if cand in device_meta:
            ent = device_meta[cand]
            if isinstance(ent, dict):
                groups = [[k, list(v)] for k, v in ent.items() if isinstance(v, (list, tuple))]
                return groups or None
            if isinstance(ent, (list, tuple)):
                return [[0, list(ent)]]
    return None


def compose_reload_cfg(prev_cfg: dict, disk: dict) -> dict:
    """Compose a post-Run sheet config from scan-time state and updated YAML.

    Deep-merge *disk* over *prev_cfg* (disk lists/scalars win, mirroring
    ``OmegaConf.merge`` semantics used for run composition). When the on-disk
    YAML no longer contains ``input.calib``, remove it from the merged config:
    a successful pipeline write consumes those one-shot triggers.
    """
    merged = copy.deepcopy(prev_cfg)
    config_yaml._deep_merge(merged, copy.deepcopy(disk))
    disk_input = disk.get("input") if isinstance(disk, dict) else None
    if not (isinstance(disk_input, dict) and "calib" in disk_input) and isinstance(merged.get("input"), dict):
        merged["input"].pop("calib", None)
    return merged


def reload_tab_after_run(app: Any, stem: str, sheet: Any, processed: set[str]) -> bool:
    """Reload one processed tab from its updated run YAML.

    Rebuild only tabs whose probe completed successfully and that have no
    unsaved config/metadata edits. Post-run sync status is unknown, so reload
    intentionally omits ``sync_status``; the next scan refreshes it.
    """
    try:
        if not (yp := app._yaml_paths.get(stem)) or not yp.is_file():
            return False
        try:
            pcid = format.to_pcid_from_name(format.stem_to_pcid(stem))
        except Exception:
            return False
        if pcid not in processed:
            return False
        is_metadata_dirty = getattr(sheet, "is_metadata_dirty", False)
        if callable(is_metadata_dirty):
            is_metadata_dirty = is_metadata_dirty()
        if sheet.is_dirty or is_metadata_dirty:
            lf.info("Skipping post-run reload for tab %s with unsaved edits", stem)
            return False
        if not isinstance(prev_cfg := getattr(sheet, "_cfg", None), dict):
            return False
        disk = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(yp), resolve=True)
        merged = compose_reload_cfg(prev_cfg, disk)
        if getattr(app, "_full_mode", False):
            cli_cfg.ensure_full_cfg(merged)
        device_meta, _, metadata_path = app._load_device_meta()
        sheet.load(
            merged,
            full=getattr(app, "_full_mode", False),
            config_root=schema.Config,
            return_enum=schema.Return,
            metadata=meta_for_stem(stem, device_meta),
            metadata_path=str(metadata_path) if metadata_path else None,
        )
        return True
    except Exception:
        lf.warning("Post-run reload failed for tab %s", stem, exc_info=True)
        return False


def reload_tabs_after_run(app: Any, processed: Iterable[str]) -> int:
    """Reload every eligible processed tab; return the number of reloads."""
    processed_pcids = set(processed or [])
    return sum(reload_tab_after_run(app, stem, sheet, processed_pcids) for stem, sheet in app._pages.items())
