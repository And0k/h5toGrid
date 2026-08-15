"""States registry — StrEnum labels where value = transport key.

Single source of truth for all state labels used in progress bars,
status lines, and log context.  ``Stage`` values double as display
text (backend-owned); ``ScanStage`` values are ``str.yaml`` i18n keys
translated by the GUI (:meth:`App._translate_scan_stage`).

Usage::

    from tcm.states import Stage, ScanStage
    stage_ctx.set_stage(1, Stage.LOAD, "Loading %s", path.name)
    rt.progress_overall.set(0, 1, ScanStage.SCAN)
"""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    """Per-probe processing phase labels (value = upper-bar description text)."""

    LOAD = "load"  # xr_io.load_raw / _load_batch
    COEFS = "coefs"  # prepare_coefs + save
    PROC = "proc"  # physical.process (calc + binning)
    NC = "NC"  # store_processed_incremental (per bin), use_h5 only
    TSV = "TSV"  # xr_io.ds_to_csv (per bin), text_path only
    COMBINE = "combine"  # _combine_probes (post-loop, not per-probe)


class ScanStage(StrEnum):
    """Scan lifecycle labels — value = ``scan_stage.*`` i18n key in
    ``tcm_gui/str.yaml`` (display text lives there only; the GUI
    translates via ``_translate_scan_stage`` / ``_translate_desc``).

    Driven by ``progress_overall.set()`` at scan boundaries in
    :func:`processing.run` — same mechanism as ``progress_stage`` updates.
    The GUI label shows ``progress_overall.desc`` when active (``tot > 0``),
    or the stored :attr:`App._cfg_state` enum when idle.
    """

    DEFAULT = "scan_stage.default"
    SCAN = "scan_stage.scan"
    DONE = "scan_stage.done"
