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

import tkinter as tk
import tkinter.font as tkfont

from ._i18n import STRINGS as _S
from .const import set_widget_meta
from .theme import scaled, strip_palette
import tcm_gui.theme as theme


class TabRail(tk.Canvas):
    PROG_W = 12  # logical px — progress column width
    PAD_X = 6  # logical px — text padding inside tab column
    PAD_Y = 10  # logical px — vertical label padding per side
    MIN_H = 32  # logical px floor for compressed inactive tabs
    GROW_CAP = 48  # logical px max grow per tab
    TICK_MS = 60  # animation period
    LERP = 0.25  # fill easing factor per tick
    ANGLE = 90  # CCW → reads bottom→up; flip to 270 if upside down

    def __init__(self, parent: tk.Widget, on_select, on_hover=None) -> None:
        super().__init__(parent, highlightthickness=0, takefocus=False)
        self._on_select = on_select
        self._on_hover = on_hover  # (name | None) → status bar update
        self._pal = strip_palette(self)
        self.configure(bg=self._pal["base"])
        _base = tkfont.nametofont("TkDefaultFont")
        _actual = _base.actual()
        # Bold variant — independent Font so TkDefaultFont stays untouched elsewhere
        self._font = tkfont.Font(
            root=self,
            family=_actual["family"],
            size=_actual["size"],
            weight="bold",
        )
        _sz = int(_actual["size"])
        self._base_pt = _sz if _sz > 0 else 10  # fallback when actual reports 0 in headless
        self._font_cache: dict[int, tkfont.Font] = {self._base_pt: self._font}
        self._ls = self._font.metrics("linespace")
        self._rail_w = scaled(self.PROG_W) + self._ls + 2 * scaled(self.PAD_X)
        self.configure(width=self._rail_w)
        self._names: list[str] = []
        self._cells: dict[str, dict] = {}  # name → geometry + item ids
        self._st: dict[str, dict] = {}  # name → visual state
        self._selected: str | None = None
        self._hovered: str | None = None
        self._disabled = False  # inert look + click veto — no configs to switch to yet
        self._after_id: str | None = None  # animation loop handle
        self.bind("<Configure>", lambda _e: self._layout())
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", self._leave)
        set_widget_meta(
            self,
            status=_S["rail.status"],
        )

    # ── membership ────────────────────────────────────────────────────
    @staticmethod
    def _new_st() -> dict:
        return {"state": "pending", "frac": 0.0, "shown": 0.0, "dirty_config": False, "dirty_meta": False}

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
        if st["frac"] != st["shown"] and self._after_id is None:
            self._after_id = self.after(self.TICK_MS, self._tick)

    def set_selected(self, name: str | None) -> None:
        if name == self._selected:
            return
        self._selected = name
        self._layout()  # selection affects heights (compression protects it)

    def set_dirty(self, name: str, dirty_config: bool, dirty_meta: bool | None = None) -> None:
        """Mark tab dirty — separate config vs metadata flags.

        ``dirty_meta is None`` keeps backward compat for single-arg callers
        (tests): they pass combined dirty as ``dirty_config`` and we treat it
        as both flags when true, else clean.
        """
        st = self._st.get(name)
        if st is None:
            return
        # Backward compat: single bool means combined dirty
        if dirty_meta is None:
            # If caller passed a single bool, interpret True as both dirty
            # (old behavior: is_dirty or is_metadata_dirty)
            if dirty_config and not (st["dirty_config"] or st["dirty_meta"]):
                st["dirty_config"] = st["dirty_meta"] = True
                self._layout()
            elif not dirty_config and (st["dirty_config"] or st["dirty_meta"]):
                st["dirty_config"] = st["dirty_meta"] = False
                self._layout()
            return
        if st["dirty_config"] == dirty_config and st["dirty_meta"] == dirty_meta:
            return
        st["dirty_config"] = dirty_config
        st["dirty_meta"] = dirty_meta
        # No layout needed — marker is separate, not part of label length
        if name in self._cells:
            self._refresh(name)

    def set_disabled(self, disabled: bool) -> None:
        """Inert look for the whole rail: dim text, no accent, click veto.

        Simple mode before the first successful scan — mirrors the readonly
        tksheet so the placeholder tab + caption read as awaiting a data path.
        """
        if self._disabled == disabled:
            return
        self._disabled = disabled
        for name in self._names:
            self._refresh(name)

    # ── vertical sizing policy ────────────────────────────────────────
    def _min_selected_h(self) -> int:
        """Minimum height for selected/active tab — ~4 chars at 8 pt + padding.

        Guarantees the active tab stays readable (>~4 chars) even under
        extreme compression; tiny tabs lose their margins first (see _refresh).
        """
        return self._font_at(8).measure("n" * 4) + 2 * scaled(self.PAD_Y)

    def _needed_selected_h(self) -> int:
        """Height for selected tab to show its FULL label at 8 pt, no truncation.

        ``_refresh`` budgets ``h - scaled(8)`` for text when ``h >= _min_selected_h``,
        so the cell needs ``full_8 + scaled(8)``; never below ``_min_selected_h``.
        Activating a tab re-layouts (``set_selected → _layout``), so the newly
        active tab steals room from inactive ones and its stripped prefix
        becomes visible.
        """
        if self._selected is None or self._selected not in self._st:
            return self._min_selected_h()
        _full = self._full_label(self._selected)
        return max(
            self._min_selected_h(),
            self._font_at(8).measure(_full) + scaled(8),
        )

    def _heights(self, H: int) -> dict[str, int]:
        """Content-based heights: capped grow on surplus, waterfill on shortage.

        Ideal = rotated label length + padding; surplus grows tabs capped at
        ``min(GROW_CAP, 35%)`` with the remainder left empty below; shortage
        protects the selected tab only when there's room for the rest at MIN_H;
        otherwise compresses ALL tabs together (selected gets a remainder pixel).
        Selected is always forced to at least ~4 chars (see _min_selected_h).
        """
        ideal = {n: self._font.measure(self._full_label(n)) + 2 * scaled(self.PAD_Y) for n in self._names}
        # Enforce selected >= full label (no truncation) before any budget math
        _sel_need = self._needed_selected_h()
        if self._selected in ideal:
            ideal[self._selected] = max(ideal[self._selected], _sel_need)
        total = sum(ideal.values())
        n = len(self._names)
        if total < H:  # grow, capped
            surplus = H - total
            return {
                n_: h + min(min(scaled(self.GROW_CAP), h * 35 // 100), h * surplus // max(total, 1))
                for n_, h in ideal.items()
            }
        sel = self._selected if self._selected in ideal and n > 1 else None
        floor = scaled(self.MIN_H)
        if sel and ideal[sel] + (n - 1) * floor <= H:
            # Enough room: protect selected at ideal, compress the rest.
            sel_h = min(ideal[sel], H)
            heights = {sel: sel_h}
            heights.update(self._compress({k: v for k, v in ideal.items() if k != sel}, H - sel_h, sel))
            return self._ensure_selected_full(heights, sel, _sel_need, H)
        # Shortage: compress ALL tabs together — every tab stays visible,
        # but re-raise selected to its full-label need if squeezed below.
        heights = self._compress(ideal, H, sel)
        return self._ensure_selected_full(heights, sel, _sel_need, H)

    def _ensure_selected_full(
        self, heights: dict[str, int], sel: str | None, need: int, H: int
    ) -> dict[str, int]:
        """Steal room from inactive tabs so selected reaches *need* (full label).

        Others keep >=1 px; tallest shrink first (fair). When even that cannot
        fit (``H < need + n - 1``), selected takes all but 1 px per inactive tab
        and still truncates (middle-ellipsis fallback in ``_resolve_label_font``).
        """
        if not sel or heights.get(sel, 0) >= need:
            return heights
        n = len(heights)
        if H >= need + (n - 1):
            _deficit = need - heights[sel]
            heights[sel] = need
            # Steal evenly (round-robin 1 px) so inactive tabs shrink together —
            # tallest-first would squeeze one tab to 1 px while siblings keep 44.
            _others = [k for k in heights if k != sel]
            while _deficit > 0 and any(heights[k] > 1 for k in _others):
                for k in _others:
                    if _deficit <= 0:
                        break
                    if heights[k] > 1:
                        heights[k] -= 1
                        _deficit -= 1
            return heights
        # Window too short for full selected — best effort: others at 1 px
        for k in heights:
            if k != sel:
                heights[k] = 1
        heights[sel] = max(H - (n - 1), 1)
        return heights

    @staticmethod
    def _compress(ideal: dict[str, int], budget: int, selected: str | None = None) -> dict[str, int]:
        """Distribute *budget* across tabs — waterfill above floor, even split below.

        When ``budget >= n * MIN_H``: smaller tabs freeze at ideal first,
        rest share the remainder with a ``MIN_H`` floor (waterfill).
        When ``budget < n * MIN_H``: every tab gets a share proportional to
        its ideal height (so larger labels get more space), with the selected
        tab receiving a 20% bonus (floored to 1 extra pixel minimum).
        No tab is invisible.
        """
        n = len(ideal)
        if n == 0:
            return {}
        floor = scaled(TabRail.MIN_H)
        if budget < n * floor:
            # Extreme shortage — equal base, selected gets ~20% bonus.
            # Every tab visible; selected is noticeably taller.
            sel_count = 1 if selected and selected in ideal else 0
            n_others = n - sel_count
            # bonus = 20% of per-tab share, min 1 px, capped so others stay >= 1
            per_tab = budget // (n_others + sel_count * 6 // 5)  # 6/5 = 1.2x weight for selected
            bonus = max(per_tab // 5, 1)
            sel_h = min(per_tab + bonus, budget - n_others) if sel_count else 0
            remaining = budget - sel_h
            base_others = remaining // max(n_others, 1)
            rem = remaining - base_others * n_others
            result: dict[str, int] = {}
            if selected and selected in ideal:
                result[selected] = max(sel_h, 1)
            for name in ideal:
                if name == selected:
                    continue
                result[name] = base_others + (1 if rem > 0 else 0)
                rem -= 1
            return result
        # Normal shortage — waterfill with MIN_H floor.
        pool = sorted(ideal, key=ideal.__getitem__)
        heights: dict[str, int] = {}
        remaining = budget
        for i, name in enumerate(pool):
            share = remaining // max(n - i, 1)
            h = min(ideal[name], max(share, floor))
            heights[name] = h
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
        tx_center = (tx0 + tx1) // 2
        tx_right = tx1 - scaled(self.PAD_X)  # right edge padded — for overflow case
        y = 0
        for i, name in enumerate(self._names):
            y0, y1 = y, y + heights[name]
            y = y1
            c = {
                "y0": y0,
                "y1": y1,
                "h": y1 - y0,
                "tx": tx_center,
                "tx_right": tx_right,
                "ty": (y0 + y1) // 2,
                # progress column
                "track": self.create_rectangle(0, y0, PW, y1, fill=pal["track"], width=0),
                "fill": self.create_rectangle(0, y0, PW, y0, fill=pal["run"], width=0),
                # "edge": self.create_rectangle(0, y0, PW, y0, fill=pal["edge"], width=0),
                # tab column
                "face": self.create_rectangle(tx0, y0, tx1, y1, fill=pal["track"], width=0),
                "acc": self.create_rectangle(
                    tx1 - scaled(4), y0 + scaled(2), tx1 - scaled(1), y1 - scaled(2), fill=pal["sel"], width=0
                ),
            }
            if i:  # hairline separator — exact shared boundary of two configs
                self.create_line(0, y0, self._rail_w, y0, fill=pal["base"])
            c["text"] = self.create_text(
                c["tx"],
                c["ty"],
                text=name,
                angle=self.ANGLE,
                anchor="center",
                font=self._font,
                fill=pal["dim"],
            )
            # Dirty marker — top left corner of tab, separate from label
            # Horizontal, small, color indicates which is dirty
            c["dirty"] = self.create_text(
                tx0 + scaled(3),
                y0 + scaled(3),
                text="",
                anchor="nw",
                font=self._font_at(8),
                fill=pal["text"],
            )
            self._cells[name] = c
            self._refresh(name)

    def _geom_fill(self, name: str) -> None:
        c = self._cells.get(name)
        if c is None:
            return
        front = c["y0"] + int(c["h"] * self._st[name]["shown"])  # grows top→down
        self.coords(c["fill"], 0, c["y0"], scaled(self.PROG_W), front)

    # ── rendering ─────────────────────────────────────────────────────
    def _refresh(self, name: str) -> None:
        c = self._cells.get(name)
        if c is None:
            return
        st, pal = self._st[name], self._pal
        self.itemconfigure(c["face"], fill=pal["hover"] if self._hovered == name else pal["track"])
        if fill_clr := {"running": pal["run"], "done": pal["done"], "error": pal["error"]}.get(st["state"]):
            self.itemconfigure(c["fill"], fill=fill_clr)
        sel = self._selected == name and not self._disabled
        dim = not sel  # unselected (or whole rail disabled) → slightly dimmer, no blue
        self.itemconfigure(c["acc"], state="normal" if sel else "hidden")
        # Available vertical length — remove 8px margins when tab <~4 chars (more room for text)
        _min_h = self._min_selected_h()
        _max_vert = c["h"] if c["h"] < _min_h else c["h"] - scaled(8)
        # Bold + shrink-to-fit (min 8 pt); right-align only on overflow, with left ellipsis;
        # rotate to 0° when vertical room <~2 chars (tiny tab) — cheap branch, not hard
        _font, _txt, _anchor, _angle = self._resolve_label_font(name, _max_vert)
        # Anchor / x / angle: center when fits at base, right (s) when shrunk/truncated;
        # for horizontal (angle 0) keep centered in tab column
        if _angle == 0:
            _x, _anchor = c["tx"], "center"
        else:
            _x = c["tx_right"] if _anchor == "s" else c["tx"]
        self.coords(c["text"], _x, c["ty"])
        self.itemconfigure(
            c["text"],
            fill=pal["dim"] if dim else pal["text"],
            text=_txt,
            font=_font,
            anchor=_anchor,
            angle=_angle,
        )
        # Dirty marker — top left corner, separate from label
        # Handle old single-dirty key for backward compat
        if "dirty" in st:
            dc = dm = bool(st["dirty"])
        else:
            dc = bool(st.get("dirty_config", False))
            dm = bool(st.get("dirty_meta", False))
        if not (dc or dm):
            self.itemconfigure(c["dirty"], text="")
        elif dc and dm:
            self.itemconfigure(c["dirty"], text="*", fill=pal["dim"] if dim else pal["text"])
        elif dc:
            self.itemconfigure(c["dirty"], text="*", fill=theme.CONFIG_TREE_BG)
        else:  # only meta
            self.itemconfigure(c["dirty"], text="*", fill=theme.META_TREE_BG)
        self._geom_fill(name)

    def _full_label(self, name: str) -> str:
        """Untruncated label — length measurement source for _heights."""
        st = self._st[name]
        pref = {"running": "▸ ", "done": "✔ "}.get(st["state"], "")
        return pref + name

    def _font_at(self, pt: int) -> tkfont.Font:
        """Bold font at *pt* — cached per size (shared across tabs)."""
        if (f := self._font_cache.get(pt)) is not None:
            return f
        f = tkfont.Font(family=self._font.cget("family"), size=pt, weight="bold")
        self._font_cache[pt] = f
        return f

    def _resolve_label_font(self, name: str, max_len: int) -> tuple[tkfont.Font, str, str, int]:
        """Return (font, text, anchor, angle) that fits *max_len* vertically.

        - Selected tab → FULL label, never truncated: shrink base→8 pt; if even
          8 pt overflows (window too short for ``_needed_selected_h``), fall back
          to middle-ellipsis so date prefix + device tail stay partly visible.
        - Inactive: fits at base pt → center anchor, 90°, full text; shrink
          base→8 pt when slightly over → right anchor (s); at 8 pt still over →
          middle-ellipsis (head…tail) keeping the date prefix + device suffix.
        - When vertical room <~2 chars, rotate to 0° (horizontal) — cheap:
          same selected-full / inactive-middle logic against column width.
        anchor "center" = centered; "s" = right-aligned (east after 90° rotation).
        """
        full = self._full_label(name)
        _base_f = self._font_at(self._base_pt)
        _is_sel = self._selected == name and not self._disabled
        # Tiny vertical room <~2 chars → rotate to horizontal (0°) — cheap, no extra layout
        _tiny_vert = self._font_at(8).measure("n" * 2)
        if max_len < _tiny_vert:
            return self._resolve_horizontal(full, _base_f, _is_sel)
        if _base_f.measure(full) <= max_len:
            return _base_f, full, "center", self.ANGLE
        # Shrink slightly when not fitting — not less than 8 pt
        for pt in range(self._base_pt - 1, 7, -1):  # 8 pt inclusive, below base
            f = self._font_at(pt)
            if f.measure(full) <= max_len:
                return f, full, "s", self.ANGLE
        # Even at 8 pt does not fit
        f8 = self._font_at(8)
        if _is_sel:
            # Layout should have reserved full room; window too short → middle fallback
            if f8.measure(full) <= max_len:
                return f8, full, "s", self.ANGLE
            return f8, self._fit_middle_at(full, max_len, f8), "s", self.ANGLE
        return f8, self._fit_middle_at(full, max_len, f8), "s", self.ANGLE

    def _resolve_horizontal(
        self, full: str, base_f: tkfont.Font, is_sel: bool = False
    ) -> tuple[tkfont.Font, str, str, int]:
        """Fit *full* horizontally inside the tab column (angle 0).

        Called when vertical room <~2 chars; uses the column width as budget.
        Selected → full label; inactive → middle-ellipsis (head…tail).
        """
        # Tab-column inner width (rail width minus progress column minus small pads)
        _avail_w = self._rail_w - scaled(self.PROG_W) - scaled(4)
        if base_f.measure(full) <= _avail_w:
            return base_f, full, "center", 0
        for pt in range(self._base_pt - 1, 7, -1):
            f = self._font_at(pt)
            if f.measure(full) <= _avail_w:
                return f, full, "center", 0
        f8 = self._font_at(8)
        if is_sel and f8.measure(full) <= _avail_w:
            return f8, full, "center", 0
        # Horizontal truncation — middle ellipsis keeps date head + device tail
        return f8, self._fit_middle_at(full, _avail_w, f8), "center", 0

    def _fit_at(self, s: str, max_len: int, font: tkfont.Font) -> str:
        """Suffix-ellipsize *s* to *max_len* using *font* metrics (head…).

        When *max_len* cannot accommodate ~6 characters (ellipsis + 5),
        the ellipsis is suppressed — tiny tabs show the fitting prefix raw
        rather than a cramped "…x" that carries no information.
        """
        if font.measure(s) <= max_len:
            return s
        # Suppress ellipsis when not enough room for ~6 characters
        if font.measure("…" + "n" * 5) > max_len:
            t = s
            while t and font.measure(t) > max_len:
                t = t[:-1]
            return t
        while s and font.measure(s + "…") > max_len:
            s = s[:-1]
        return s + "…" if s else ""

    def _fit_prefix_at(self, s: str, max_len: int, font: tkfont.Font) -> str:
        """Prefix-ellipsize *s* to *max_len* using *font* metrics (…tail).

        Keeps the suffix, ellipsis on the left/overflow edge — used for
        right-aligned truncated tabs so the visible part stays near the right.
        When *max_len* cannot accommodate ~6 characters, the ellipsis is
        suppressed and the fitting suffix is returned raw.
        """
        if font.measure(s) <= max_len:
            return s
        # Suppress ellipsis when not enough room for ~6 characters
        if font.measure("…" + "n" * 5) > max_len:
            t = s
            while t and font.measure(t) > max_len:
                t = t[1:]
            return t
        ell = "…"
        # Drop from the front until ellipsis+suffix fits
        while s and font.measure(ell + s) > max_len:
            s = s[1:]
        return ell + s if s else ""

    def _fit_middle_at(self, s: str, max_len: int, font: tkfont.Font) -> str:
        """Middle-ellipsize *s* to *max_len* using *font* metrics (head…tail).

        Keeps the date prefix (head) + device suffix (tail): config stems share
        long prefixes (``230615_…``), so head-only truncation hides the device
        and tail-only hides the date — middle keeps both partly visible.
        Head gets the extra char on odd splits (date prefix priority).
        When *max_len* cannot accommodate ~6 characters, the ellipsis is
        suppressed and the fitting suffix is returned raw (device id wins —
        all heads look identical there; the full date shows on selection).
        """
        if font.measure(s) <= max_len:
            return s
        # Suppress ellipsis when not enough room for ~6 characters
        if font.measure("…" + "n" * 5) > max_len:
            t = s
            while t and font.measure(t) > max_len:
                t = t[1:]
            return t
        ell = "…"
        # Binary search: max kept chars k with head + ell + tail fitting.
        # head = s[: (k+1)//2] (date side, gets odd extra), tail = s[-(k//2):].
        lo, best = 0, 0
        hi = len(s)
        while lo <= hi:
            mid = (lo + hi) // 2
            _hl, _tl = (mid + 1) // 2, mid // 2
            _cand = s[:_hl] + ell + (s[len(s) - _tl :] if _tl else "")
            if font.measure(_cand) <= max_len:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best <= 0:
            return ""
        _hl, _tl = (best + 1) // 2, best // 2
        return s[:_hl] + ell + (s[len(s) - _tl :] if _tl else "")

    # Backward-compat shims — keep old names delegating to new helpers
    def _label(self, name: str) -> str:  # pragma: no cover
        """Fit rotated label into cell height (text length = vertical extent)."""
        c = self._cells.get(name)
        if c is None:
            return name
        _min_h = self._min_selected_h()
        _max = c["h"] if c["h"] < _min_h else c["h"] - scaled(8)
        return self._resolve_label_font(name, _max)[1]

    def _fit(self, s: str, max_len: int) -> str:  # pragma: no cover
        """Ellipsize to available length — font metrics, not character guesses."""
        return self._fit_at(s, max_len, self._font)

    # ── animation (lerp fill + glimmer; idles itself to sleep) ───────
    def _tick(self) -> None:
        """Easing of fills toward targets; stops once all settled — no idle motion."""
        self._after_id = None
        busy = False
        for name in self._names:
            st = self._st[name]
            if abs(st["frac"] - st["shown"]) > 0.002:
                st["shown"] += (st["frac"] - st["shown"]) * self.LERP
                busy = True
            elif st["shown"] != st["frac"]:
                st["shown"] = st["frac"]
            else:
                continue  # settled — no redraw
            self._geom_fill(name)
        if busy:
            self._after_id = self.after(self.TICK_MS, self._tick)

    # ── pointer ───────────────────────────────────────────────────────
    def _at(self, y: int) -> str | None:
        return next((n for n, c in self._cells.items() if c["y0"] <= y < c["y1"]), None)

    def _click(self, e: tk.Event) -> None:
        if self._disabled:
            return
        # Tab column is the button; progress column stays display-only.
        if e.x >= scaled(self.PROG_W) and (name := self._at(e.y)) is not None:
            self._on_select(name)

    def _motion(self, e: tk.Event) -> None:
        name = self._at(e.y)  # hover over both columns
        if name == self._hovered:
            return
        old, self._hovered = self._hovered, name
        for n in (old, name):
            if n in self._cells:
                self._refresh(n)
        self.configure(cursor="hand2" if name and not self._disabled and e.x >= scaled(self.PROG_W) else "")
        if self._on_hover:
            self._on_hover(name)

    def _leave(self, _e: tk.Event) -> None:
        old, self._hovered = self._hovered, None
        if old in self._cells:
            self._refresh(old)
        self.configure(cursor="")
        if self._on_hover:
            self._on_hover(None)
