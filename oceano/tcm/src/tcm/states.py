"""States registry — StrEnum labels where value = display text.

Single source of truth for all state labels used in progress bars,
status lines, and log context.  Follows the pattern: enum value IS
the human-readable text shown to the user.

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
    """Scan lifecycle labels (value = overall status label text above notebook).

    Driven by ``progress_overall.set()`` at scan boundaries in
    :func:`processing.run` — same mechanism as ``progress_stage`` updates.
    The GUI label shows ``progress_overall.desc`` when active (``tot > 0``),
    or the stored :attr:`App._cfg_state` enum when idle.
    """

    DEFAULT = "Default configuration"
    SCAN = "Processing configurations"
    DONE = "Generated configurations for processing found data"
