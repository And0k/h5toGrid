"""Config tree in tksheet ≥ 7 treeview."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from tkinter import ttk
from typing import Any

import numpy as np
from tksheet import Sheet

from tcm.config import ConfigInCoefs_InclProc

# Derive field order from dataclass declaration — single source of truth.
# Exclude ``dates`` / ``date`` which are handled as tree-level metadata, not row items.
_COEF_FIELDS = [f.name for f in dataclasses.fields(ConfigInCoefs_InclProc) if f.name not in ("dates", "date")]

_COEF_SHAPES: dict[str, tuple[int, ...]] = {
    "Ag": (3, 3), "Cg": (3,), "Ah": (3, 3), "Ch": (3,),
    "Rz": (3, 3), "kVabs": (6,), "P_t": (3, 3),
    "P": (2,), "PBattery": (2,), "PTemp": (2,),
    "azimuth_shift_deg": (), "g0xyz": (3,),
}
_1D_WITH_DATES = {"kVabs"}  # единственное 1D с датами → parent+child
_SUB = "₁₂₃₄₅₆₇₈₉"
_DATE_COL = 2  # sheet col: 0=tree 1=₁ 2=₂/date 3=₃…

# coef_sheet.py
from collections.abc import Mapping
from typing import Final, Literal

TkAlign = Literal["w", "center", "e"]
_ALIGN: Final[Mapping[str, TkAlign]] = {
    "left": "w",
    "right": "e",
    "center": "center",
    "w": "w",
    "e": "e",
}


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:g}"
    return str(v)


def _pf(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _is_date(s: str) -> bool:
    for f in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            datetime.strptime(s, f)
            return True
        except ValueError:
            continue
    return False

class ConfigSheet:
    """Wraps tksheet.Sheet(treeview=True) for config display / editing."""

    def __init__(self, parent) -> None:

        self.sh = Sheet(parent, treeview=True, show_horizontal_grid=False, show_vertical_grid=False)
        bg = ttk.Style().lookup("TFrame", "background") or "SystemButtonFace"
        self.sh.set_options(allow_cell_overflow=True, header_background=bg)
        self.sh.enable_bindings(["all"])
        self.sh.extra_bindings([("edit_validation", self._on_edit)])
        self._meta: dict[Any, dict] = {}
        self._nv = 6
        self._full = False
        self._cfg: dict = {}
        # Single snapshot tuple for dirty tracking — populated at end of load()
        self._snap: tuple[dict, dict, str] = ({}, {}, "")

    # ── public API ──────────────────────────────────────────────────

    def load(self, cfg: dict, full: bool = False) -> None:
        self._cfg, self._full = cfg, full
        self._meta.clear()
        self.sh.del_rows(rows=list(range(self.sh.total_rows())))
        self.sh.enable_bindings(["all"])
        self._nv = self._calc_nv(cfg, full)
        self.sh.headers(list(_SUB[: self._nv]))
        (self._build_full if full else self._build_coefs)(cfg)
        self._apply_styles()
        self.sh.redraw()
        self._take_snapshot()

    def get_edited_coefs(self) -> dict[str, Any]:
        """Read leaf values → coefs dict for YAML write-back."""
        out: dict[str, Any] = {}
        for iid, m in self._meta.items():
            t = m.get("type")
            if t == "2d":
                nr, nc = m["shape"]
                out[m["key"]] = [
                    [_pf((self.sh.item(c).get("values") or ())[j]) or 0.0 for j in range(nc)]
                    for c in m["children"][:nr]
                ]
            elif t == "1d":  # kVabs: parent+child
                vals = self.sh.item(m["child"]).get("values") or ()
                out[m["key"]] = [_pf(vals[j]) or 0.0 for j in range(min(m["len"], len(vals)))]
            elif t == "1d_flat":  # Cg, Ch, …: одна строка
                vals = self.sh.item(iid).get("values") or ()
                out[m["key"]] = [_pf(vals[j]) or 0.0 for j in range(min(m["len"], len(vals)))]
            elif t == "scalar":
                if v := (self.sh.item(iid).get("values") or ("",))[0]:
                    out[m["key"]] = float(v)
        return out

    def get_edited_dates(self) -> dict[str, str]:
        return {
            m["key"]: d
            for iid, m in self._meta.items()
            if m.get("has_date")
            and (vals := self.sh.item(iid).get("values") or ())
            and len(vals) > 1
            and (d := vals[1])   # date in values[1]
        }

    def get_edited_input_path(self) -> str:
        for iid, m in self._meta.items():
            if m.get("type") == "input":
                return (self.sh.item(iid).get("values") or ("",))[0]
        return ""

    # ── dirty tracking ───────────────────────────────────────────────

    def _current_state(self) -> tuple[dict, dict, str]:
        """Return full editable state as a comparable tuple."""
        return self.get_edited_coefs(), self.get_edited_dates(), self.get_edited_input_path()

    def _take_snapshot(self) -> None:
        """Capture current state as the *clean* baseline."""
        self._snap = self._current_state()

    @property
    def is_dirty(self) -> bool:
        """True when sheet content differs from last load/save snapshot."""
        return self._current_state() != self._snap

    def mark_clean(self) -> None:
        """Reset dirty flag after a successful write-back."""
        self._take_snapshot()

    # ── tree construction ───────────────────────────────────────────

    def _calc_nv(self, cfg: dict, full: bool) -> int:
        nv = 6
        for v in cfg.get("input", {}).get("coefs", {}).values():
            if isinstance(v, (list, np.ndarray)) and len(v):
                nv = max(nv, len(v[0]) if isinstance(v[0], (list, np.ndarray)) else len(v))
        if full:
            for sec in cfg.values():
                if isinstance(sec, dict):
                    for v in sec.values():
                        if isinstance(v, (list, np.ndarray)):
                            nv = max(nv, len(v))
        return nv


    def _build_coefs(self, cfg: dict) -> None:
        inp = cfg.get("input", {})
        path = _fmt(inp.get("path", ""))
        inp_iid = self._ins(
            "",
            "input",
            [path] + [""] * (self._nv - 1),
            "",
            meta={"key": "input", "type": "input", "is_string": True, "max_col": 1, "style": "node"},
            open_=True,  # ← раскрыт
        )
        if tr := inp.get("time_ranges", []):
            self._ins(
                inp_iid,
                "time_ranges",
                [_fmt(x) for x in tr] + [""] * (self._nv - len(tr)),
                "",
                meta={"is_string": True, "max_col": len(tr)},
            )
        coefs = inp.get("coefs", {})
        dates = coefs.get("dates", {})
        cdate_src = coefs.get("date")
        cdate = cdate_src or (max(dates.values()) if dates else "")
        coefs_iid = self._ins(
            inp_iid,
            "coefs",
            [""] * self._nv,
            cdate,
            meta={
                "key": "coefs",
                "has_date": True,
                "max_col": 0,
                "date_style": "blue" if not cdate_src else None,
            },
            open_=False,
        )
        for name in _COEF_FIELDS:
            if name in coefs:
                self._ins_coef(coefs_iid, name, coefs.get(name), dates.get(name, ""))


    def _build_full(self, cfg: dict) -> None:
        for sec, val in cfg.items():
            if sec == "input":
                self._build_input(val)
            elif isinstance(val, dict):
                sid = self._ins("", sec, [""] * self._nv, "")
                for k, v in val.items():
                    self._ins_generic(sid, k, v)
            else:
                self._ins_leaf("", sec, val)

    def _build_input(self, inp: dict) -> None:
        path = _fmt(inp.get("path", ""))
        inp_iid = self._ins(
            "",
            "input",
            [path] + [""] * (self._nv - 1),
            "",
            meta={"key": "input", "type": "input", "is_string": True, "max_col": 1, "style": "node"},
            open_=True,  # ← раскрыт
        )
        for k, v in inp.items():
            if k == "path":
                continue
            if k == "coefs":
                dates = v.get("dates", {})
                cdate = v.get("date") or (max(dates.values()) if dates else "")
                cid = self._ins(
                    inp_iid,
                    "coefs",
                    [""] * self._nv,
                    cdate,
                    meta={"key": "coefs", "has_date": True, "max_col": 0},
                )
                for name in _COEF_FIELDS:
                    if name in v:
                        self._ins_coef(cid, name, v.get(name), dates.get(name, ""))
            else:
                self._ins_generic(inp_iid, k, v)

    # ── coef inserters ──────────────────────────────────────────────
    def _ins_coef(self, par: Any, name: str, value: Any, date: str) -> None:
        shape = _COEF_SHAPES.get(name, ())
        if len(shape) == 2:
            self._ins_2d(par, name, value, shape, date)
        elif len(shape) == 1 and name in _1D_WITH_DATES:
            self._ins_1d(par, name, value, shape[0], date)
        elif len(shape) == 1:
            self._ins_1d_flat(par, name, value, shape[0])
        else:
            self._ins_scalar(par, name, value)

    def _ins_2d(self, par, name, value, shape, date) -> None:
        nr, nc = shape
        children: list = []
        pid = self._ins(
            par,
            name,
            [""] * self._nv,
            date,
            meta={
                "key": name,
                "type": "2d",
                "children": children,
                "has_date": True,
                "shape": shape,
                "max_col": 0,
            },
        )
        for i in range(nr):
            row = [_fmt(x) for x in value[i]] if value else [""] * nc
            children.append(
                self._ins(pid, f"{name}[{i}]", row + [""] * (self._nv - len(row)), "", meta={"max_col": nc})
            )

    def _ins_1d(self, par, name, value, n, date):
        """1D с датами (kVabs): parent + child."""
        pid = self._ins(
            par,
            name,
            [""] * self._nv,
            date,
            meta={"key": name, "type": "1d", "child": None, "has_date": True, "len": n, "max_col": 0},
        )
        row = [_fmt(x) for x in value] if value else [""] * n
        self._meta[pid]["child"] = self._ins(
            pid, name, row + [""] * (self._nv - len(row)), "", meta={"max_col": n}
        )

    def _ins_1d_flat(self, par, name, value, n):
        """1D без дат (Cg, Ch, P, …): одна строка, без детей."""
        row = [_fmt(x) for x in value] if value else [""] * n
        self._ins(
            par,
            name,
            row + [""] * (self._nv - len(row)),
            "",
            meta={"key": name, "type": "1d_flat", "len": n, "max_col": n},
        )

    def _ins_scalar(self, par, name, value):
        self._ins(
            par,
            name,
            [_fmt(value)] + [""] * (self._nv - 1),
            "",
            meta={"key": name, "type": "scalar", "max_col": 1},
        )
    def _ins_generic(self, par: Any, key: str, value: Any) -> None:
        if isinstance(value, dict):
            sid = self._ins(par, key, [""] * self._nv, "")
            for k, v in value.items():
                self._ins_generic(sid, k, v)
        elif isinstance(value, (list, np.ndarray)) and len(value):
            if isinstance(value[0], (list, np.ndarray)):
                sid = self._ins(par, key, [""] * self._nv, "")
                for i, row in enumerate(value):
                    self._ins(sid, f"{key}[{i}]", [_fmt(x) for x in row] + [""] * (self._nv - len(row)), "")
            else:
                self._ins(
                    par,
                    key,
                    [_fmt(x) for x in value] + [""] * (self._nv - len(value)),
                    "",
                    meta={"is_string": True, "max_col": len(value)},
                )
        else:
            self._ins_leaf(par, key, value)

    def _ins_leaf(self, par: Any, key: str, value: Any) -> None:
        self._ins(par, key, [_fmt(value)] + [""] * (self._nv - 1), "", meta={"is_string": True, "max_col": 1})

    # ── helpers ─────────────────────────────────────────────────────
    def _ins(self, parent_iid, text, vals, date="", meta=None, open_=False):
        if date and len(vals) > 1:
            vals = list(vals)
            vals[1] = date
            if meta is not None:
                meta.setdefault("meta_date_cols", []).append(2)
        # даты по содержимому
        if (dc := [c + 1 for c, v in enumerate(vals) if v and _is_date(str(v))]):
            meta.setdefault("date_cols", []).extend(dc)
        try:
            iid = self.sh.insert(parent=parent_iid, text=text, values=vals, open_=open_)
        except TypeError:
            iid = self.sh.insert(parent=parent_iid, text=text, values=vals)
            if open_:
                try:
                    self.sh.item(iid, open_=True)
                except Exception:
                    pass
        if meta:
            self._meta[iid] = meta
        return iid

    def _on_edit(self, event) -> str | None:
        """
        Validate: Col 0 Name (full only), Col ₂ date|num, rest num.
        за пределами размерности → запрет, кроме даты (обрабатывается отдельно)
        """
        c, val = event.c, event.value
        if c == 0:
            return val if self._full else None
        if not val or not val.strip():
            return val
        m = self._meta.get(getattr(event, "iid", None), {})
        if c == _DATE_COL and m.get("has_date"):
            return val if _is_date(val) else None
        if c > m.get("max_col", self._nv):
            return None
        if m.get("is_string"):
            return val
        if _is_num(val):
            return val
        return None

    # def _apply_styles(self) -> None:

    #     bg = ttk.Style().lookup("TFrame", "background") or "SystemButtonFace"
    #     self.sh.highlight_cells(row="all", column="all", canvas="header", bg=bg, fg="#0055CC", redraw=False)

    #     for iid, m in self._meta.items():
    #         print(f"{iid=}: e:", end=" ")
    #         mc = m.get("max_col", 0)
    #         # 1) числа, даты → право (дефолт)
    #         for c in range(1, mc + 1):
    #             self.sh.align_cells(row=iid, column=c, align="e", redraw=False)
    #             print(f"{c=}", end=' ')
    #         # 2) metadata dates → лево (высший приоритет)
    #         for c in m.get("meta_date_cols", []):
    #             self.sh.align_cells(row=iid, column=c, align="w", redraw=False)
    #             if m.get("date_style") == "blue":
    #                 self.sh.highlight_cells(row=iid, column=c, fg="#0055CC", redraw=False)
    #                 print(f"Blue({c=})!", end=" ")
    #             print(f"w({c=})!", end=" ")
    #         print()
    #     self.sh.redraw()

    def _style_header(self, bg: str, fg: str) -> None:

        # 1) глобальный цвет header — самый надёжный путь для tksheet 7.x
        try:
            (sh := self.sh).set_options(header_fg=fg, header_bg=bg)
        except (AttributeError, TypeError):
            pass

        # # 2) точечная подсветка header-ячеек: header row = 0, не "all"
        # for c in range(sh.total_columns()):
        #     sh.highlight_cells(
        #         row=0,
        #         column=c,
        #         canvas="header",
        #         bg=bg,
        #         fg=fg,
        #         redraw=False,
        #     )
        # sh.redraw()


    def _apply_styles(self) -> None:
        sh = self.sh

        # SystemButtonFace не везде валиден; #F0F0F0 безопаснее
        bg = str(ttk.Style().lookup("TFrame", "background") or "#F0F0F0")
        fg = "#0055CC"

        self._style_header(bg, fg)

        # treeview mode:
        # r — видимая row position.
        # Если есть сортировка/сворачивание строк, enumerate может разъехаться;
        # тогда нужен отдельный mapping iid -> visible row position.
        for r, (_, m) in enumerate(self._meta.items()):
            date_cols = tuple(int(c) for c in (m.get("meta_date_cols") or ()))
            date_set = frozenset(date_cols)

            # числа/даты → право
            # meta columns у вас 1-based, tksheet columns 0-based
            for c in range(1, int(m.get("max_col") or 0) + 1):
                if c not in date_set:
                    sh.align_cells(r, c - 1, align="e", redraw=False)

            # metadata dates → лево
            for c in date_cols:
                if (col := c - 1) >= 0:
                    sh.align_cells(r, col, align="w", redraw=False)

                    if m.get("date_style") == "blue":
                        sh.highlight_cells(
                            row=r,
                            column=col,
                            fg=fg,
                            highlight_fg=fg,
                            redraw=False,
                        )

        sh.redraw()