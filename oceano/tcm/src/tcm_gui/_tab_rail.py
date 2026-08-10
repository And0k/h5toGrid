"""Vertical tab rail: progress column + tab column, configs stacked top→down.

Replaces the native ttk.Notebook tab row (hidden via the ``Bare.TNotebook``
layout in :mod:`app`) — ttk tabs can neither carry per-tab progress fills
nor be placed vertically in this Tk (no ``-tabposition`` before TIP 751).

Geometry, left→right:
  progress column (PROG_W) — vertical fills growing top→down per config;
  tab column (TAB_W)       — rotated labels (``create_text(angle=90)``,
                             true Tk text, reads bottom→up), buttons,
                             selection accent on the notebook-facing edge.

Config *i*'s bar and tab share exactly one vertical extent ``[y0, y1]`` —
progress is aligned to its tab by construction (single layout pass).

Sizing policy (vertical analog of notebook tabs):
  ideal height = label length + padding;
  surplus      → capped grow, remainder stays empty below;
  shortage     → selected keeps ideal, inactive waterfill-compress to MIN_H.

Requires canvas text ``-angle`` (present on Tk 8.6.13+); canvas
coordinates ignore ``tk scaling`` — all pixels via :func:`const.scaled`.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
import tkinter.font as tkfont

from .const import scaled, set_widget_meta, strip_palette


class TabRail(tk.Canvas):
    PROG_W = 12  # logical px — progress column width
    PAD_X = 6  # logical px — text padding inside tab column
    PAD_Y = 10  # logical px — vertical label padding per side
    MIN_H = 32  # logical px floor for compressed inactive tabs
    GROW_CAP = 48  # logical px max grow per tab
    TICK_MS = 60  # animation period
    LERP = 0.25  # fill easing factor per tick
    ANGLE = 90  # CCW → reads bottom→up; flip to 270 if upside down

    def __init__(self, parent: tk.Widget, on_select) -> None:
        super().__init__(parent, highlightthickness=0, takefocus=False)
        self._on_select = on_select
        self._pal = strip_palette(self)
        self.configure(bg=self._pal["base"])
        self._font = tkfont.nametofont("TkDefaultFont")
        self._ls = self._font.metrics("linespace")
        self._rail_w = scaled(self.PROG_W) + self._ls + 2 * scaled(self.PAD_X)
        self.configure(width=self._rail_w)
        self._names: list[str] = []
        self._cells: dict[str, dict] = {}  # name → geometry + item ids
        self._st: dict[str, dict] = {}  # name → visual state
        self._selected: str | None = None
        self._hovered: str | None = None
        self._after_id: str | None = None  # animation loop handle
        self.bind("<Configure>", lambda _e: self._layout())
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", self._leave)
        set_widget_meta(
            self,
            status="Configuration rail — click a tab to switch; "
            "vertical fill = processing progress (top→down)",
        )

    # ── membership ────────────────────────────────────────────────────
    @staticmethod
    def _new_st() -> dict:
        return {"state": "pending", "frac": 0.0, "shown": 0.0, "dirty": False}

    def add_tab(self, name: str) -> None:
        self._names.append(name)
        self._st[name] = self._new_st()
        self._layout()
        if self._selected is None:
            self.set_selected(name)

    def clear(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.delete("all")
        self._cells.clear()
        self._st.clear()
        self._names.clear()
        self._selected = self._hovered = None

    # ── external state feeds (from App polling) ──────────────────────
    def set_state(self, name: str, state: str, frac: float = 0.0) -> None:
        st = self._st.get(name)
        if st is None or (st["state"] == state and st["frac"] == frac):
            return
        old = self._full_label(name)
        st["state"], st["frac"] = state, min(max(frac, 0.0), 1.0)
        if name in self._cells:
            self._refresh(name)  # paint: colors, accent, text
            if self._full_label(name) != old:  # '✔' appeared — geometry
                self._layout()
        if state == "running" and self._after_id is None:
            self._after_id = self.after(self.TICK_MS, self._tick)

    def set_selected(self, name: str | None) -> None:
        if name == self._selected:
            return
        self._selected = name
        self._layout()  # selection affects heights (compression protects it)

    def set_dirty(self, name: str, dirty: bool) -> None:
        st = self._st.get(name)
        if st is None or st["dirty"] == dirty:
            return
        st["dirty"] = dirty
        self._layout()  # '*' changes ideal height

    # ── vertical sizing policy ────────────────────────────────────────
    def _heights(self, H: int) -> dict[str, int]:
        """Content-based heights: capped grow on surplus, waterfill on shortage.

        Ideal = rotated label length + padding; surplus grows tabs capped at
        ``min(GROW_CAP, 35%)`` with the remainder left empty below; shortage
        protects the selected tab and compresses the rest toward MIN_H.
        """
        ideal = {n: self._font.measure(self._full_label(n)) + 2 * scaled(self.PAD_Y) for n in self._names}
        total = sum(ideal.values())
        if total < H:  # grow, capped
            surplus = H - total
            return {
                n: h + min(min(scaled(self.GROW_CAP), h * 35 // 100), h * surplus // max(total, 1))
                for n, h in ideal.items()
            }
        if self._selected in ideal and len(self._names) > 1:  # protect selected
            sel = self._selected
            heights = {sel: ideal[sel]}
            heights.update(self._compress({n: h for n, h in ideal.items() if n != sel}, H - ideal[sel]))
            return heights
        return self._compress(ideal, H)

    @staticmethod
    def _compress(ideal: dict[str, int], budget: int) -> dict[str, int]:
        """Waterfill shrink: smaller tabs freeze at ideal first; floor MIN_H.

        May slightly overflow when even floors don't fit — canvas clips.
        """
        pool = sorted(ideal, key=ideal.__getitem__)
        heights, remaining = {}, budget
        for i, n in enumerate(pool):
            share = remaining // max(len(pool) - i, 1)
            h = min(ideal[n], max(share, scaled(TabRail.MIN_H)))
            heights[n] = h
            remaining -= h
        return heights

    # ── geometry ──────────────────────────────────────────────────────
    def _layout(self) -> None:
        """(Re)place cells top→down; bar and tab of each config share [y0, y1]."""
        H = self.winfo_height()
        if H <= 1 or not self._names:
            return  # not mapped yet; <Configure> will re-run
        heights = self._heights(H)
        self.delete("all")
        self._cells.clear()
        pal, PW = self._pal, scaled(self.PROG_W)
        tx0, tx1 = PW, self._rail_w
        y = 0
        for i, name in enumerate(self._names):
            y0, y1 = y, y + heights[name]
            y = y1
            c = {
                "y0": y0,
                "y1": y1,
                "h": y1 - y0,
                "tx": (tx0 + tx1) // 2,
                # progress column
                "track": self.create_rectangle(0, y0, PW, y1, fill=pal["track"], width=0),
                "fill": self.create_rectangle(0, y0, PW, y0, fill=pal["run"], width=0),
                "edge": self.create_rectangle(0, y0, PW, y0, fill=pal["edge"], width=0),
                # tab column
                "face": self.create_rectangle(tx0, y0, tx1, y1, fill=pal["track"], width=0),
                "acc": self.create_rectangle(
                    tx1 - scaled(4), y0 + scaled(2), tx1 - scaled(1), y1 - scaled(2), fill=pal["sel"], width=0
                ),
            }
            if i:  # hairline separator — exact shared boundary of two configs
                self.create_line(0, y0, self._rail_w, y0, fill=pal["base"])
            c["text"] = self.create_text(
                c["tx"], (y0 + y1) // 2, text=name, angle=self.ANGLE, font=self._font, fill=pal["dim"]
            )
            self._cells[name] = c
            self._refresh(name)

    def _geom_fill(self, name: str) -> None:
        c = self._cells.get(name)
        if c is None:
            return
        front = c["y0"] + int(c["h"] * self._st[name]["shown"])  # grows top→down
        self.coords(c["fill"], 0, c["y0"], scaled(self.PROG_W), front)

    def _geom_edge(self, name: str, t: float) -> None:
        """Glimmer band riding the fill front while running."""
        c, st = self._cells[name], self._st[name]
        if st["shown"] < 0.02:  # nothing to ride on yet
            self.itemconfigure(c["edge"], state="hidden")
            return
        ew = scaled(6)
        front = c["y0"] + int(c["h"] * st["shown"])
        dy = math.sin(t * 3.0) * scaled(3)
        self.itemconfigure(c["edge"], state="normal")
        self.coords(c["edge"], 0, front - ew + dy, scaled(self.PROG_W), front + dy)

    # ── rendering ─────────────────────────────────────────────────────
    def _refresh(self, name: str) -> None:
        c = self._cells.get(name)
        if c is None:
            return
        st, pal = self._st[name], self._pal
        self.itemconfigure(c["face"], fill=pal["hover"] if self._hovered == name else pal["track"])
        if fill_clr := {"running": pal["run"], "done": pal["done"], "error": pal["error"]}.get(st["state"]):
            self.itemconfigure(c["fill"], fill=fill_clr)
        sel = self._selected == name
        dim = st["state"] == "pending" and not sel
        self.itemconfigure(c["acc"], state="normal" if sel else "hidden")
        self.itemconfigure(c["text"], fill=pal["dim"] if dim else pal["text"], text=self._label(name))
        self.itemconfigure(c["edge"], state="normal" if st["state"] == "running" else "hidden")
        self._geom_fill(name)

    def _full_label(self, name: str) -> str:
        """Untruncated label — length measurement source for _heights."""
        st = self._st[name]
        return ("✔ " if st["state"] == "done" else "") + name + ("*" if st["dirty"] else "")

    def _label(self, name: str) -> str:
        """Fit rotated label into cell height (text length = vertical extent)."""
        return self._fit(self._full_label(name), self._cells[name]["h"] - scaled(8))

    def _fit(self, s: str, max_len: int) -> str:
        """Ellipsize to available length — font metrics, not character guesses."""
        if self._font.measure(s) <= max_len:
            return s
        while s and self._font.measure(s + "…") > max_len:
            s = s[:-1]
        return s + "…" if s else ""

    # ── animation (lerp fill + glimmer; idles itself to sleep) ───────
    def _tick(self) -> None:
        self._after_id = None
        busy = False
        t = time.monotonic()
        for name in self._names:
            st = self._st[name]
            if abs(st["frac"] - st["shown"]) > 0.002:
                st["shown"] += (st["frac"] - st["shown"]) * self.LERP
                busy = True
            elif st["shown"] != st["frac"]:
                st["shown"] = st["frac"]
            self._geom_fill(name)
            if st["state"] == "running":
                busy = True
                self._geom_edge(name, t)
        if busy:
            self._after_id = self.after(self.TICK_MS, self._tick)

    # ── pointer ───────────────────────────────────────────────────────
    def _at(self, y: int) -> str | None:
        return next((n for n, c in self._cells.items() if c["y0"] <= y < c["y1"]), None)

    def _click(self, e: tk.Event) -> None:
        # Tab column is the button; progress column stays display-only.
        if e.x >= scaled(self.PROG_W) and (name := self._at(e.y)) is not None:
            self._on_select(name)

    def _motion(self, e: tk.Event) -> None:
        name = self._at(e.y) if e.x >= scaled(self.PROG_W) else None
        if name == self._hovered:
            return
        old, self._hovered = self._hovered, name
        for n in (old, name):
            if n in self._cells:
                self._refresh(n)
        self.configure(cursor="hand2" if name else "")

    def _leave(self, _e: tk.Event) -> None:
        old, self._hovered = self._hovered, None
        if old in self._cells:
            self._refresh(old)
        self.configure(cursor="")
