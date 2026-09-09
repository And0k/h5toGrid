"""Per-configuration progress: fixed stage weights → overall fraction.

The pipeline always walks the same top-level stages; :data:`WEIGHTS`
defines each stage's contribution to the overall fraction, so a tab's fill
advances proportionally to the work actually done (Processing dominates).
Thread-safe: workers mutate under lock, the GUI polls
:meth:`ProgressBank.snapshot_all` — same contract as ``ProgressState``.
"""

from __future__ import annotations

import threading
from typing import Final

STAGES: Final[tuple[str, ...]] = ("Scan", "Load", "Prepare", "Processing", "Save", "Cleanup", "Finished")
WEIGHTS: Final[dict[str, float]] = dict(
    Scan=5, Load=10, Prepare=5, Processing=60, Save=10, Cleanup=5, Finished=5
)
_ORDER: Final[dict[str, int]] = {s: i for i, s in enumerate(STAGES)}
# Pipeline Stage enum values → canonical bank stages.
# 4-letter prefix covers Load ("load"), Processing ("proc"), Scan, Cleanup,
# Finished; explicit map for the rest.
_ALIASES: Final[dict[str, str]] = {
    "coefs": "Prepare",
    "nc": "Save",
    "tsv": "Save",
    "combine": "Save",
}


def canon_stage(text: str) -> str:
    """Free-form phase description → canonical stage.

    ``"Saving results"`` → ``"Save"``; ``"coefs"`` → ``"Prepare"``;
    unknown → ``""`` (ignored by the bank).
    """
    t = text.strip().lower()
    if mapped := _ALIASES.get(t):
        return mapped
    return next((s for s in STAGES if t.startswith(s.lower()[:4])), "")


class ProgressBank:
    """cfg → {state, stage, inner, lvl}; states: pending/running/done/error."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._st: dict[str, dict] = {}

    def run_start(self, cfgs) -> None:
        with self._lock:
            self._st = {c: {"state": "pending", "stage": "", "inner": 0.0, "lvl": 20} for c in cfgs}

    def clear(self) -> None:
        """Drop all per-config state — a new scan invalidates the last Run's bank.

        Without this, re-selecting the same input path after a successful Run
        repaints every rail fill to 100% from the still-done bank cells before
        the user clicks Run again (the scan worker's ``finish`` calls are
        no-ops on already-done cells, and ``_poll_progress`` reads the stale
        ``done``/1.0 snapshots while ``wk.busy`` is True).
        """
        with self._lock:
            self._st.clear()

    def stage_start(self, cfg: str | None, stage: str) -> None:
        if not cfg or stage not in _ORDER:
            return
        with self._lock:
            if (st := self._st.get(cfg)) and st["state"] in ("pending", "running"):
                # Terminal states are final: post-loop phases (combine) still
                # carry the last config's attribution — re-running it would
                # drop a done fill back to mid-Save (h5 last-config stall).
                st.update(state="running", stage=stage, inner=0.0)

    def inner(self, cfg: str | None, cur: float, tot: float) -> None:
        """Intra-stage progress from GuiTqdm ticks."""
        if not cfg:
            return
        with self._lock:
            if (st := self._st.get(cfg)) and st["state"] == "running":
                st["inner"] = cur / tot if tot else 0.0

    def finish(self, cfg: str, ok: bool = True) -> None:
        with self._lock:
            if st := self._st.get(cfg):
                st["state"] = "done" if ok else "error"
                if ok:
                    st.update(stage="Finished", inner=1.0)

    def raise_lvl(self, cfg: str | None, lvl: int) -> None:
        """Sticky max level — future status dot on the tab cell."""
        if not cfg:
            return
        with self._lock:
            if (st := self._st.get(cfg)) and lvl > st["lvl"]:
                st["lvl"] = lvl

    def snapshot_all(self) -> dict[str, tuple[str, float, str, int]]:
        """cfg → (state, frac, stage, lvl) — one lock acquisition per poll."""
        with self._lock:
            return {c: (st["state"], self._frac(st), st["stage"], st["lvl"]) for c, st in self._st.items()}

    @staticmethod
    def _frac(st: dict) -> float:
        if st["state"] == "pending":
            return 0.0
        if st["state"] == "done":
            return 1.0
        done = sum(w for s, w in WEIGHTS.items() if _ORDER[s] < _ORDER.get(st["stage"], 0))
        cur = WEIGHTS.get(st["stage"], 0.0) * st["inner"]
        return (done + cur) / sum(WEIGHTS.values())
