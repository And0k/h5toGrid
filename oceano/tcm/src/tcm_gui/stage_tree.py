"""Per-page stage tree: live and restored view of one probe's processing.

Top level — stage nodes (1–3 pre-created pending-gray; 4+ appear on
arrival); children — sublevel rows: identity from [### …] boundary
records, live numbers from progress_stage snapshots (active page only).
Fed two ways, one parser: queue records carrying stage_fresh/stage_prefix
(live) and journal.ProbeState (restore).  Stage click → on_pick(pcid,
num, None); sublevel click → on_pick(pcid, num, sub) — App shows the
segment in LogDock.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from tkinter import ttk
from typing import Final

from tcm.journal import ProbeState, parse_prefix

from .const import DEFAULT_FG, FG_DEFAULT, TAG_COLORS, set_widget_meta

_FIXED: Final = ((1, "load"), (2, "coefs"), (3, "proc"))

# Sticky priority: level escalation outranks state, interrupted outranks all
_PRIOR: Final = {
    "pending": 0,
    "done": 1,
    "running": 2,
    "warning": 3,
    "error": 4,
    "critical": 5,
    "interrupted": 6,
}


class StageTree(ttk.Treeview):
    def __init__(self, parent, pcid: str, on_pick) -> None:
        super().__init__(parent, columns=("status",), show="tree headings", selectmode="browse")
        self.pcid, self._on_pick = pcid, on_pick
        self.heading("#0", text="Stage")
        self.heading("status", text="Status")
        self.column("status", width=90, anchor="e")
        self.tag_configure("pending", foreground=DEFAULT_FG)
        self.tag_configure("done", foreground=FG_DEFAULT)
        self.tag_configure("running", foreground=FG_DEFAULT)
        self.tag_configure("interrupted", foreground=TAG_COLORS["error"])
        for lvl, clr in TAG_COLORS.items():
            self.tag_configure(lvl, foreground=clr)
        self._tag: dict[str, str] = {}  # iid → current tag (single)
        self._state: dict[str, str] = {}  # iid → pending|running|done
        self._items: set[str] = set()
        self._cur: str | None = None  # running stage iid
        self._live: str | None = None  # live sublevel row iid
        self.user_owned: set[str] = set()  # nodes the user opened — never auto-close
        self._guard = 0
        for num, name in _FIXED:
            self._ensure(num, name)
        self.bind("<<TreeviewOpen>>", lambda _: self._own(True))
        self.bind("<<TreeviewClose>>", lambda _: self._own(False))
        self.bind("<<TreeviewSelect>>", self._select)
        set_widget_meta(self, status=f"Stage journal: {pcid}")

    # ── nodes ──────────────────────────────────────────────────────
    def _ensure(self, num: int, name: str) -> str:
        iid = f"s{num}"
        if iid not in self._tag:
            self.insert("", "end", iid=iid, text=f"{num} {name}", values=("—",), tags=("pending",))
            self._tag[iid] = self._state[iid] = "pending"
        return iid

    def enter_stage(self, num: int, name: str = "") -> None:
        """[## …] boundary: previous stage → done (children ✔), this one opens."""
        iid = self._ensure(num, name or str(num))
        if iid == self._cur:
            return
        if self._cur:
            for child in self.get_children(self._cur):
                self.item(child, values=("✔",))
            self._set_state(self._cur, "done", "✔")
        self._set_state(iid, "running", "…")
        self._cur, self._live = iid, None
        with self._guarded():
            for other in self._state:
                if other != iid and other not in self.user_owned:
                    self.item(other, open=False)
            self.item(iid, open=True)
        self.see(iid)

    def _set_state(self, iid: str, state: str, status: str) -> None:
        self._state[iid] = state
        self.item(iid, values=(status,))
        if _PRIOR[state] > _PRIOR[self._tag[iid]]:
            self._tag[iid] = state
            self.item(iid, tags=(state,))

    def escalate(self, num: int, lvl: str) -> None:
        self._escalate_iid(f"s{num}", lvl)

    def _escalate_iid(self, iid: str, lvl: str) -> None:
        """WARNING+ paints the node; sticks (level tags never fade)."""
        if iid in self._tag and _PRIOR[lvl] > _PRIOR[self._tag[iid]]:
            self._tag[iid] = lvl
            self.item(iid, tags=(lvl,))

    def enter_sublevel(self, name: str, status: str) -> None:
        """[### …]: row per sublevel name; repeats (chunks) update in place."""
        if not self._cur:
            return
        iid = f"{self._cur}:{name}"
        if iid not in self._items:
            self._items.add(iid)
            self.insert(self._cur, "end", iid=iid, text=name, values=(status or "…",))
        else:
            self.set(iid, "status", status or "…")
        self._live = iid

    def set_live_progress(self, desc: str, n: int, total: int) -> None:
        """progress_stage poll → status of the live row (active page only)."""
        if self._live:
            self.set(self._live, "status", f"{desc} {n}/{total}" if desc else f"{n}/{total}")

    # ── feed: live records / restore ───────────────────────────────
    def observe(self, rec: logging.LogRecord) -> None:
        pf = parse_prefix(getattr(rec, "stage_prefix", ""))
        if not pf or pf.get("pcid") != self.pcid:
            return
        match getattr(rec, "stage_fresh", 0):
            case 2:
                if "num" in pf:
                    self.enter_stage(int(pf["num"]), pf.get("name", ""))
            case 3:
                if "sub" in pf:
                    self.enter_sublevel(pf["sub"], getattr(rec, "boundary_msg", ""))
        if rec.levelno >= logging.WARNING and "num" in pf:
            lvl = rec.levelname.lower()
            self.escalate(int(pf["num"]), lvl)
            if "sub" in pf:
                self._escalate_iid(f"s{pf['num']}:{pf['sub']}", lvl)

    def apply_snap(self, ps: ProbeState, ended: bool) -> None:
        """Restore from Reader: seen stages done; last interrupted when !ended."""
        for num, seg in ps.stages.items():
            iid = self._ensure(num, seg.name)
            if not ended and num == ps.last:
                self._set_state(iid, "interrupted", "прервано")
            else:
                self._set_state(iid, "done", "✔")
        for num, ssegs in ps.subs.items():
            for name in dict.fromkeys(s.name for s in ssegs):
                self._items.add(iid := f"s{num}:{name}")
                self.insert(f"s{num}", "end", iid=iid, text=name, values=("✔",))
        if ps.ok is False and ps.last:
            self.escalate(ps.last, "error")

    def reset(self) -> None:
        self.delete(*self.get_children())
        self._tag.clear()
        self._state.clear()
        self._items.clear()
        self.user_owned.clear()
        self._cur = self._live = None
        for num, name in _FIXED:
            self._ensure(num, name)

    # ── user ownership of open/close ───────────────────────────────
    def _own(self, opened: bool) -> None:
        if self._guard:
            return
        (self.user_owned.add if opened else self.user_owned.discard)(self.focus())

    @contextmanager
    def _guarded(self):
        self._guard += 1
        try:
            yield
        finally:
            self._guard -= 1

    def _select(self, _e) -> None:
        iid = self.focus()
        if ":" in iid:
            stage_iid, name = iid.split(":", 1)
            self._on_pick(self.pcid, int(stage_iid[1:]), name)
        elif iid.startswith("s"):
            self._on_pick(self.pcid, int(iid[1:]), None)
