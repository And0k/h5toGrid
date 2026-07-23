"""Config tree in tksheet ≥ 7 treeview."""
from __future__ import annotations
from typing import Any
from datetime import datetime
import numpy as np

COEF_ORDER = ["Ag", "Cg", "Ah", "Ch", "Rz", "kVabs",
              "P_t", "P", "PBattery", "PTemp", "azimuth_shift_deg", "g0xyz"]

_COEF_SHAPES: dict[str, tuple[int, ...]] = {
    "Ag": (3, 3), "Cg": (3,), "Ah": (3, 3), "Ch": (3,),
    "Rz": (3, 3), "kVabs": (6,), "P_t": (3, 3),
    "P": (2,), "PBattery": (2,), "PTemp": (2,),
    "azimuth_shift_deg": (), "g0xyz": (3,),
}
_1D_WITH_DATES = {"kVabs"}          # единственное 1D с датами → parent+child
_SUB = "₁₂₃₄₅₆₇₈₉"
_DATE_COL = 2                        # sheet col: 0=tree 1=₁ 2=₂/date 3=₃…


def _fmt(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, (float, np.floating)): return f"{v:g}"
    return str(v)

def _pf(v: str) -> float | None:
    try: return float(v)
    except (ValueError, TypeError): return None

def _is_num(s: str) -> bool:
    try: float(s); return True
    except (ValueError, TypeError): return False

def _is_date(s: str) -> bool:
    for f in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
              "%d.%m.%Y", "%Y-%m-%d %H:%M:%S"):
        try: datetime.strptime(s, f); return True
        except ValueError: continue
    return False


class ConfigSheet:

    def __init__(self, parent) -> None:
        from tksheet import Sheet
        from tkinter import ttk
        self.sh = Sheet(parent, treeview=True,
                        show_horizontal_grid=False, show_vertical_grid=False)
        bg = ttk.Style().lookup("TFrame", "background") or "SystemButtonFace"
        self.sh.set_options(allow_cell_overflow=True, header_background=bg)
        self.sh.enable_bindings(["all"])
        self._meta: dict[Any, dict] = {}
        self._nv = 6
        self._full = False
        self._cfg: dict = {}
        self.sh.extra_bindings([("edit_validation", self._on_edit)])

    # ── public ──────────────────────────────────────────────────────

    def load(self, cfg: dict, full: bool = False) -> None:
        self._cfg, self._full = cfg, full
        self._meta.clear()
        self.sh.del_rows(rows="all")
        self.sh.enable_bindings(["all"])
        self._nv = self._calc_nv(cfg, full)
        self.sh.headers([""] + list(_SUB[:self._nv]))   # tree col без заголовка
        (self._build_full if full else self._build_coefs)(cfg)
        if full:
            self._set_all_open(True)
        else:
            self._set_all_open(False)
            self._expand_input()

    def get_edited_coefs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for iid, m in self._meta.items():
            t = m.get("type")
            if t == "2d":
                nr, nc = m["shape"]
                out[m["key"]] = [
                    [_pf((self.sh.item(c).get("values") or ())[j]) or 0.0
                     for j in range(nc)]
                    for c in m["children"][:nr]]
            elif t == "1d":                               # kVabs: parent+child
                vals = self.sh.item(m["child"]).get("values") or ()
                out[m["key"]] = [_pf(vals[j]) or 0.0
                                 for j in range(min(m["len"], len(vals)))]
            elif t == "1d_flat":                          # Cg, Ch, …: одна строка
                vals = self.sh.item(iid).get("values") or ()
                out[m["key"]] = [_pf(vals[j]) or 0.0
                                 for j in range(min(m["len"], len(vals)))]
            elif t == "scalar":
                if v := (self.sh.item(iid).get("values") or ("",))[0]:
                    out[m["key"]] = float(v)
        return out

    def get_edited_dates(self) -> dict[str, str]:
        return {m["key"]: d
                for iid, m in self._meta.items()
                if m.get("has_date")
                and (vals := self.sh.item(iid).get("values") or ())
                and len(vals) > 1 and (d := vals[1])}

    def get_edited_input_path(self) -> str:
        for iid, m in self._meta.items():
            if m.get("type") == "input":
                return (self.sh.item(iid).get("values") or ("",))[0]
        return ""

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
        inp_iid = self._ins("", "input",
                            [path] + [""] * (self._nv - 1), "",
                            meta={"key": "input", "type": "input",
                                  "is_string": True, "max_col": 1})
        if tr := inp.get("time_ranges", []):
            self._ins(inp_iid, "time_ranges",
                      [_fmt(x) for x in tr] + [""] * (self._nv - len(tr)), "",
                      meta={"is_string": True, "max_col": len(tr)})
        coefs = inp.get("coefs", {})
        dates = coefs.get("dates", {})
        cdate = coefs.get("date") or (max(dates.values()) if dates else "")
        coefs_iid = self._ins(inp_iid, "coefs", [""] * self._nv, cdate,
                              meta={"key": "coefs", "has_date": True, "max_col": 0})
        for name in COEF_ORDER:
            if name in coefs or name in _COEF_SHAPES:
                self._ins_coef(coefs_iid, name, coefs.get(name),
                               dates.get(name, ""))

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
        inp_iid = self._ins("", "input",
                            [path] + [""] * (self._nv - 1), "",
                            meta={"key": "input", "type": "input",
                                  "is_string": True, "max_col": 1})
        for k, v in inp.items():
            if k == "path": continue
            if k == "coefs":
                dates = v.get("dates", {})
                cdate = v.get("date") or (max(dates.values()) if dates else "")
                cid = self._ins(inp_iid, "coefs", [""] * self._nv, cdate,
                                meta={"key": "coefs", "has_date": True, "max_col": 0})
                for name in COEF_ORDER:
                    if name in v or name in _COEF_SHAPES:
                        self._ins_coef(cid, name, v.get(name),
                                       dates.get(name, ""))
            else:
                self._ins_generic(inp_iid, k, v)

    # ── coef inserters ──────────────────────────────────────────────

    def _ins_coef(self, par, name, value, date):
        shape = _COEF_SHAPES.get(name, ())
        if len(shape) == 2:
            self._ins_2d(par, name, value, shape, date)
        elif len(shape) == 1 and name in _1D_WITH_DATES:
            self._ins_1d(par, name, value, shape[0], date)
        elif len(shape) == 1:
            self._ins_1d_flat(par, name, value, shape[0])
        else:
            self._ins_scalar(par, name, value)

    def _ins_2d(self, par, name, value, shape, date):
        nr, nc = shape
        children: list = []
        pid = self._ins(par, name, [""] * self._nv, date,
                        meta={"key": name, "type": "2d", "children": children,
                              "has_date": True, "shape": shape, "max_col": 0})
        for i in range(nr):
            row = [_fmt(x) for x in value[i]] if value else [""] * nc
            children.append(self._ins(
                pid, f"{name}[{i}]", row + [""] * (self._nv - len(row)), "",
                meta={"max_col": nc}))

    def _ins_1d(self, par, name, value, n, date):
        """1D с датами (kVabs): parent + child."""
        pid = self._ins(par, name, [""] * self._nv, date,
                        meta={"key": name, "type": "1d", "child": None,
                              "has_date": True, "len": n, "max_col": 0})
        row = [_fmt(x) for x in value] if value else [""] * n
        self._meta[pid]["child"] = self._ins(
            pid, name, row + [""] * (self._nv - len(row)), "",
            meta={"max_col": n})

    def _ins_1d_flat(self, par, name, value, n):
        """1D без дат (Cg, Ch, P, …): одна строка, без детей."""
        row = [_fmt(x) for x in value] if value else [""] * n
        self._ins(par, name, row + [""] * (self._nv - len(row)), "",
                  meta={"key": name, "type": "1d_flat", "len": n, "max_col": n})

    def _ins_scalar(self, par, name, value):
        self._ins(par, name, [_fmt(value)] + [""] * (self._nv - 1), "",
                  meta={"key": name, "type": "scalar", "max_col": 1})

    def _ins_generic(self, par, key, value):
        if isinstance(value, dict):
            sid = self._ins(par, key, [""] * self._nv, "")
            for k, v in value.items():
                self._ins_generic(sid, k, v)
        elif isinstance(value, (list, np.ndarray)) and len(value):
            if isinstance(value[0], (list, np.ndarray)):
                sid = self._ins(par, key, [""] * self._nv, "")
                for i, row in enumerate(value):
                    self._ins(sid, f"{key}[{i}]",
                              [_fmt(x) for x in row]
                              + [""] * (self._nv - len(row)), "")
            else:
                self._ins(par, key,
                          [_fmt(x) for x in value]
                          + [""] * (self._nv - len(value)), "",
                          meta={"is_string": True, "max_col": len(value)})
        else:
            self._ins_leaf(par, key, value)

    def _ins_leaf(self, par, key, value):
        self._ins(par, key, [_fmt(value)] + [""] * (self._nv - 1), "",
                  meta={"is_string": True, "max_col": 1})

    # ── helpers ─────────────────────────────────────────────────────

    def _ins(self, parent_iid, text, vals, date="", meta=None):
        if date and len(vals) > 1:
            vals = list(vals)
            vals[1] = date
        iid = self.sh.insert(parent=parent_iid, text=text, values=vals)
        if meta:
            self._meta[iid] = meta
        return iid

    def _set_all_open(self, open_):
        for iid in self._meta:
            try: self.sh.item(iid, open_=open_)
            except Exception: pass

    def _expand_input(self):
        for iid, m in self._meta.items():
            if m.get("type") == "input":
                try: self.sh.item(iid, open_=True)
                except Exception: pass

    def _on_edit(self, event):
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