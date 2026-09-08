"""tksheet context-menu authorization — one policy for every leaf-mutation path.

``RowPolicy`` is the single source of truth; it is consulted by the per-popup
menu refresh, the ``MT.rc_add_rows`` shadow
(:meth:`MetadataNodeMixin._rc_add_rows`) and the :class:`SheetGuard` wrappers
on tksheet's real mutation methods.  Deny is explicit (``bell()``), never silent.

Permission matrix (``selected_ref()`` kinds; ``top`` = ``parent == ""``;
``user`` = extras-added rows tracked in ``host._user_rows``):

+------------+---------------+--------------------------------------+
| target     | insert above/ | delete                             |
|            | below         |                                      |
+------------+---------------+--------------------------------------+
| meta/setup | split (routed)| deny (removal = Undo of the split)   |
| meta leaf  | deny          | deny (fixed paired rows)             |
| top-level  | deny (never   | deny                                 |
|            | new top nodes)|                                      |
| hydra child| native        | deny (fixed hydra tree)              |
| user row   | deny          | allow iff selection ⊆ user rows      |
+------------+---------------+--------------------------------------+

Columns are append-only: native left/right entries go off via bindings, the
``Add column`` extra appends at the end (trailing indices ≥
``host._user_col_base`` are deletable); the ``Add row`` extra only parents a
child under the selected node, never a top-level node.  A read-only sheet
(simplified mode, no real data) gets every entry disabled per popup.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from ._i18n import STRINGS as _S
from ._sheet_undo import GroupUndoBridge, MetadataTree, suspend_native_pushes

lf = logging.getLogger(__name__)

__all__ = [
    "OFF_BINDINGS",
    "SORT_BINDINGS",
    "RowPolicy",
    "SheetGuard",
    "disable_sort_menus",
    "disable_unsafe_menus",
    "install_menu_patch",
    "refresh_insert_labels",
    "refresh_menu_state",
    "render_insert_labels",
    "target_label",
]

SORT_BINDINGS = ("sort_cells", "sort_row", "sort_column", "sort_columns", "sort_rows")
"""tksheet bindings whose menu entries sort — disabled once (re-disabled after each ``enable_bindings(["all"])``)."""

COL_INSERT_BINDINGS = ("rc_insert_column",)
"""Native column insert goes off globally — columns are append-only via the ``Add column`` extra."""

OFF_BINDINGS = SORT_BINDINGS + COL_INSERT_BINDINGS
"""One-time-off bindings (re-applied after each ``enable_bindings(["all"])`` in ``load()``)."""

_FALLBACK_ABOVE = "Insert setup {target} above"
_FALLBACK_BELOW = "Insert setup {target} below"
_NATIVE_ABOVE = "Insert rows above"
_NATIVE_BELOW = "Insert rows below"
_NATIVE_ADD_ROW = "Add row"
_NATIVE_ADD_COL = "Add column"
_NATIVE_DEL_ROW = "Delete rows"
_NATIVE_DEL_COL = "Delete columns"
_NATIVE_INS_COL_LEFT = "Insert columns left"
_NATIVE_INS_COL_RIGHT = "Insert columns right"
_FALLBACK_SPLIT, _FALLBACK_NATIVE, _FALLBACK_DENY = "split", "native", "deny"


def _as_tree(meta: Any) -> MetadataTree:
    """Sketch protocol — ``ConfigSheet`` hosts are wrapped, facades pass through."""
    return meta if isinstance(meta, MetadataTree) else MetadataTree(meta)


def target_label(meta: Any, ref: tuple | None = None) -> str:
    """New setup number for the split labels, or ``""`` to keep native labels.

    Non-empty only when *ref* (default: the tree's current selection) is the
    ``metadata`` root or a ``setup`` node and the model knows its autonumber.
    """
    tree = _as_tree(meta)
    if ref is None:
        with suppress(Exception):
            ref = tree.selected_ref()
    if not ref or ref[0] not in ("meta", "setup") or not tree.setups:
        return ""
    return "" if tree.next_num is None else str(tree.next_num)


def _fmt(key: str, fallback: str, target: str) -> str:
    """STR template formatted with *target* — English fallback on missing key/shape."""
    with suppress(Exception):
        return str(_S.get(key, fallback)).format(target=target)
    return fallback.format(target=target)


def render_insert_labels(meta: Any, ref: tuple | None = None) -> tuple[str | None, str | None]:
    """``(above, below)`` split labels, or ``(None, None)`` to keep native ones."""
    if not (target := target_label(meta, ref)):
        return (None, None)
    return (
        _fmt("sheet.insert_setup_above", _FALLBACK_ABOVE, target),
        _fmt("sheet.insert_setup_below", _FALLBACK_BELOW, target),
    )


def _native_labels(tree: MetadataTree) -> tuple[str, str]:
    """Current tksheet *Insert rows above/below* labels (custom ops honored)."""
    above, below = _NATIVE_ABOVE, _NATIVE_BELOW
    with suppress(Exception):
        if (ops := getattr(getattr(tree.sheet, "MT", None), "PAR", None)) is not None:
            ops = getattr(ops, "ops", ops)
        above = str(getattr(ops, "insert_rows_above_label", above))
        below = str(getattr(ops, "insert_rows_below_label", below))
    return (above, below)


def _ops_labels(tree: MetadataTree, *names: str, fallback: str = "") -> set[str]:
    """Native menu labels for *names* from sheet ops (custom labels honored)."""
    out = {fallback} if fallback else set()
    with suppress(Exception):
        if (ops := getattr(getattr(tree.sheet, "MT", None), "PAR", None)) is not None:
            ops = getattr(ops, "ops", ops)
            out |= {str(getattr(ops, n)) for n in names if getattr(ops, n, None) is not None}
    return {s for s in out if s}


# ------------------------------------------------------------- RowPolicy
class RowPolicy:
    """Single source of truth for row/column mutation permissions.

    Kinds resolve from ``host._meta`` (``meta``/``setup``/``row``/``top``/
    ``hydra``), ``host._user_rows`` (extras-added ``user`` rows) and the
    trailing-column base ``host._user_col_base``.  Unknown iids deny both ways.
    Per-row ``_meta`` flags (``protected``/``can_delete``/``can_insert``) are
    honored when present, deny by default.
    """

    def __init__(self, tree: MetadataTree) -> None:
        self._tree = tree
        self._host = tree.host

    # ------------------------------------------------- target resolution
    def _meta_of(self, iid: Any) -> dict:
        """``_meta`` record for *iid* (empty dict when unknown)."""
        with suppress(Exception):
            if isinstance(m := (getattr(self._host, "_meta", None) or {}).get(iid), dict):
                return m
        return {}

    def kind_of(self, iid: Any) -> str:
        """``meta``/``setup``/``row``/``top``/``hydra``/``user``/``unknown``."""
        if m := self._meta_of(iid):
            if m.get("is_metadata_root"):
                return "meta"
            if m.get("is_setup"):
                return "setup"
            if m.get("is_metadata"):
                return "row"
            return "top" if m.get("parent") == "" else "hydra"
        with suppress(Exception):
            if iid is not None and iid in (getattr(self._host, "_user_rows", None) or ()):
                return "user"
        return "unknown"

    def _display_iids(self) -> list:
        """Full visible walk (unlike ``_iid_by_disp`` this keeps unknown iids)."""
        with suppress(Exception):
            if (walk := getattr(self._host, "_walk", None)) is not None:
                return list(walk(visible=True))
        return []

    def _at_display(self, r: Any) -> Any | None:
        """Display row → iid (``None`` when out of range)."""
        with suppress(Exception, TypeError, ValueError, IndexError):
            if r is not None and (vis := self._display_iids()) and 0 <= int(r) < len(vis):
                return vis[int(r)]
        return None

    def _selection_iids(self) -> list:
        """Oracle + selected + popup-loc rows → iids (``None``-free)."""
        host, out = self._host, []
        with suppress(Exception):
            if (iid := getattr(host, "_rc_sel_iid", None)) is not None:
                out.append(iid)
        out += self._selected_iids()
        return out

    def _selected_iids(self) -> list:
        """Selected + popup-loc rows → iids — the rows a command would actually act on."""
        host, out = self._host, []
        with suppress(Exception):
            out += [i for r in (host.sh.get_selected_rows() or ()) if (i := self._at_display(r)) is not None]
        with suppress(Exception):
            loc = host.sh.MT.RI.popup_menu_loc if hasattr(host.sh.MT, "RI") else host.sh.RI.popup_menu_loc
            if isinstance(loc, int) and (i := self._at_display(loc)) is not None and i not in out:
                out.append(i)
        return out

    def selection_kinds(self) -> set[str]:
        """Kinds over the whole rc target set (empty selection → empty set)."""
        return {self.kind_of(i) for i in self._selection_iids()}

    def oracle_ref(self) -> tuple | None:
        """``selected_ref()``-shaped tuple for the oracle iid (or ``None``)."""
        with suppress(Exception):
            if (ref := self._tree.selected_ref()) is not None:
                return ref
        with suppress(Exception):
            if (iid := getattr(self._host, "_rc_sel_iid", None)) is not None:
                return (self.kind_of(iid), None)
        return None

    def is_readonly(self) -> bool:
        """Simplified mode without real data — every entry goes disabled."""
        return bool(getattr(self._host, "_readonly", False) or getattr(self._host, "_loading", False))

    # ------------------------------------------------- insert
    def insert_permission(self, ref: tuple | None) -> str:
        """``"split"`` (meta/setup) / ``"native"`` (hydra child) / ``"deny"``."""
        if not ref or self.is_readonly():
            return _FALLBACK_DENY
        if ref[0] in ("meta", "setup"):
            return _FALLBACK_SPLIT
        if ref[0] == "hydra":
            return _FALLBACK_NATIVE
        return _FALLBACK_DENY  # row/top/user/unknown: fixed structure, no new top nodes

    def can_rc_add_rows(self) -> bool:
        """``MT.rc_add_rows`` gate — split targets pass (routed), hydra passes native."""
        if self.is_readonly():
            return False
        kinds = self.selection_kinds()
        if not kinds:
            return False
        if kinds & {"meta", "setup"}:
            return True
        return kinds <= {"hydra"}

    def can_native_insert(self) -> bool:
        """Bare ``Sheet.insert_rows``/``insert_row`` gate — hydra-only, never split targets."""
        if self.is_readonly():
            return False
        kinds = self.selection_kinds()
        return bool(kinds) and kinds <= {"hydra"}

    def can_child_add(self) -> bool:
        """Extras ``Add row`` — a parentable (non-metadata, known-or-user) node is selected."""
        if self.is_readonly():
            return False
        return any(k in ("hydra", "top", "user") for k in self.selection_kinds())

    # ------------------------------------------------- delete
    def _flag(self, iid: Any, name: str, default: bool = False) -> bool:
        try:
            return bool(self._meta_of(iid).get(name, default))
        except Exception:
            return default

    def can_delete_rows(self) -> bool:
        """Only extras-added rows go — and only when *every* acted-on row is one.

        The bare oracle is not enough (an empty selection deletes nothing) —
        selected + popup-loc rows must be non-empty and all-user.
        """
        if self.is_readonly():
            return False
        iids = self._selected_iids()
        if not iids:
            return False
        with suppress(Exception):
            known = {i for i in iids if self._tree.sheet.exists(i)}
            if not known:
                return False
            return all(self.kind_of(i) == "user" and not self._flag(i, "protected", False) for i in known)
        return False

    def can_delete_columns(self) -> bool:
        """Only appended trailing columns go — every selected data-col must qualify."""
        if self.is_readonly():
            return False
        base = getattr(self._host, "_user_col_base", None)
        if base is None:
            return False
        cols: set[int] = set()
        with suppress(Exception):
            mt = self._host.sh.MT
            for c in mt.get_selected_cols() or ():
                with suppress(Exception):
                    cols.add(int(mt.datacn(c)))
            if isinstance(loc := mt.CH.popup_menu_loc, int):
                with suppress(Exception):
                    cols.add(int(mt.datacn(loc)))
        return bool(cols) and all(c >= int(base) for c in cols)

    def delete_setup_allowed(self) -> bool:
        """Controller-level setup deletion — not implemented (Undo covers split removal)."""
        return False


# ------------------------------------------------------------- SheetGuard
class SheetGuard:
    """Wrap tksheet's user-facing mutation methods with :class:`RowPolicy`.

    Wrapped on ``Sheet``: ``del_rows``(+aliases)/``del_columns``(+alias) —
    delete-only-yours must hold even for programmatic callers.  Deliberately
    NOT ``insert_rows``/``insert_row``: every user insert path funnels through
    the vetted ``MT.rc_add_rows`` shadow, the ``MT.rc_add_columns`` wrapper or
    the self-vetted extras, while programmatic tree construction
    (``Sheet.insert`` → ``insert_rows``, fixtures included) must keep working.
    Wrapped on ``MT``: ``rc_add_columns`` (always denied — append-only
    columns)/``delete_rows``/``delete_columns``.  ``MT.rc_add_rows`` stays owned
    by the mixin shadow (policy-vetted inside it).  Missing names are skipped,
    so the guard degrades gracefully across tksheet versions.  Internal
    rebuilds pass via ``host._loading`` or :meth:`suspended`.
    """

    _SHEET_DELETE = ("del_rows", "delete_row", "delete_rows", "del_columns", "delete_columns")
    _MT_DELETE = ("delete_rows", "delete_columns")

    def __init__(self, tree: MetadataTree, policy: RowPolicy) -> None:
        self._tree, self._policy, self._host = tree, policy, tree.host
        self._patched: list[str] = []
        with suppress(Exception):
            self._patch_owner(tree.sheet, tuple((n, self._rule(n)) for n in self._SHEET_DELETE))
        with suppress(Exception):
            if (mt := getattr(tree.sheet, "MT", None)) is not None:
                self._patch_owner(
                    mt,
                    (("rc_add_columns", self._never),) + tuple((n, self._rule(n)) for n in self._MT_DELETE),
                )

    @staticmethod
    def _never() -> bool:
        """Native column insert — always denied (append-only columns)."""
        return False

    def _rule(self, name: str):
        """Predicate for a wrapped method — column deletes need the column gate."""
        if "column" in name:
            return self._policy.can_delete_columns
        return self._policy.can_delete_rows

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Bypass the guard for internal rebuilds (counter-style, nestable)."""
        host = self._host
        host._guard_suspended = getattr(host, "_guard_suspended", 0) + 1
        try:
            yield
        finally:
            host._guard_suspended = max(0, getattr(host, "_guard_suspended", 0) - 1)

    def _bypassed(self) -> bool:
        return bool(getattr(self._host, "_loading", False) or getattr(self._host, "_guard_suspended", 0) > 0)

    def _bell(self) -> None:
        with suppress(Exception):
            self._tree.sheet.bell()

    def _patch_owner(self, owner: Any, rules: tuple[tuple[str, Any], ...]) -> None:
        for name, allowed in rules:
            self._wrap(owner, name, allowed)

    def _wrap(self, owner: Any, name: str, allowed: Any) -> None:
        if not callable(orig := getattr(owner, name, None)) or getattr(orig, "_sheet_guard_wrapped", False):
            return

        def wrapped(*args: Any, _orig: Any = orig, _ok: Any = allowed, **kwargs: Any) -> Any:
            if not self._bypassed() and not _ok():
                self._bell()
                lf.debug("sheet-guard: denied %s", wrapped.__name__)
                return None
            return _orig(*args, **kwargs)

        wrapped._sheet_guard_wrapped = True  # type: ignore[attr-defined]
        wrapped.__name__ = name
        with suppress(Exception):
            setattr(owner, name, wrapped)
            self._patched.append(f"{type(owner).__name__}.{name}")


# ------------------------------------------------------------- menu sweep
def _all_menus(tree: MetadataTree) -> list[Any]:
    """All four tksheet rc menus (table / index / header / empty) — Nones dropped."""
    out: list[Any] = []
    with suppress(Exception):
        for m in (
            tree.sheet.MT.rc_popup_menu,
            tree.sheet.RI.ri_rc_popup_menu,
            tree.sheet.CH.ch_rc_popup_menu,
            tree.sheet.MT.empty_rc_popup_menu,
        ):
            if m is not None:
                out.append(m)
    return out


def _entry_labels(menu: Any) -> Iterator[tuple[int, str]]:
    """``(index, label)`` over command entries — separators/cascades skipped silently."""
    try:
        end = menu.index("end")
    except Exception:
        return
    if end is None:
        return
    for i in range(end + 1):
        with suppress(Exception):
            if menu.type(i) != "command":
                continue
            yield i, str(menu.entrycget(i, "label"))


def _menu_state(menu: Any, i: int) -> str:
    """Current entry state (``"normal"`` when unreadable)."""
    with suppress(Exception):
        return str(menu.entrycget(i, "state"))
    return "normal"


def refresh_menu_state(menu: Any, meta: Any, policy: RowPolicy | None = None) -> bool:
    """Relabel + enable/disable a single menu from the policy.  Returns changed.

    tksheet rebuilds every menu fresh (native labels, ``normal`` state) before
    the pre-popup hook, so matching is by native label.  Split labels appear
    only when a meta/setup node is in the target set; insert entries stay
    native-but-disabled on leaves, top nodes, user rows and empty targets.
    Read-only sheets (simplified mode, no real data) get every entry disabled.
    """
    if menu is None:
        return False
    tree = _as_tree(meta)
    pol = policy if policy is not None else RowPolicy(tree)
    kinds = pol.selection_kinds()
    readonly = pol.is_readonly()
    native_above, native_below = _native_labels(tree)
    if not readonly and (kinds & {"meta", "setup"}):
        if (ref := pol.oracle_ref()) is None or ref[0] not in ("meta", "setup"):
            ref = ("setup", None)  # autonumber only — target_label ignores the index
        split_above, split_below = render_insert_labels(tree, ref)
    else:
        split_above, split_below = None, None
    want_above, want_below = split_above or native_above, split_below or native_below
    ins_state = "normal" if not readonly and (split_above or kinds <= {"hydra"} and kinds) else "disabled"
    can_del_rows, can_del_cols = pol.can_delete_rows(), pol.can_delete_columns()
    can_child = pol.can_child_add()
    extras_row = {_S.get("sheet.insert_row", _NATIVE_ADD_ROW), _NATIVE_ADD_ROW}
    extras_col = {_S.get("sheet.insert_col", _NATIVE_ADD_COL), _NATIVE_ADD_COL}
    dels = _ops_labels(tree, "delete_rows_label", fallback=_NATIVE_DEL_ROW)
    delcs = _ops_labels(tree, "delete_columns_label", fallback=_NATIVE_DEL_COL)
    ins_cols = _ops_labels(
        tree, "insert_columns_left_label", "insert_columns_right_label", "insert_column_label"
    ) | {_NATIVE_INS_COL_LEFT, _NATIVE_INS_COL_RIGHT}
    changed = False
    for i, label in _entry_labels(menu):
        if label == native_above:
            want_label, want_state = want_above, ins_state
        elif label == native_below:
            want_label, want_state = want_below, ins_state
        elif label in dels:
            want_label, want_state = label, ("normal" if can_del_rows else "disabled")
        elif label in delcs:
            want_label, want_state = label, ("normal" if can_del_cols else "disabled")
        elif label in ins_cols:
            want_label, want_state = label, "disabled"  # append-only columns
        elif label in extras_row:
            want_label, want_state = label, ("normal" if can_child else "disabled")
        elif label in extras_col:
            want_label, want_state = label, "normal"
        else:
            want_label, want_state = label, _menu_state(menu, i)
        if readonly:
            want_state = "disabled"
        if label != want_label or _menu_state(menu, i) != want_state:
            with suppress(Exception):
                menu.entryconfig(i, label=want_label, state=want_state)
                changed = True
    return changed


def refresh_insert_labels(menu: Any, meta: Any, ref: tuple | None = None) -> bool:
    """Legacy entry point — relabels above/below only (state via :func:`refresh_menu_state`)."""
    if menu is None:
        return False
    tree = _as_tree(meta)
    if ref is None:
        with suppress(Exception):
            ref = tree.selected_ref()
    new_above, new_below = render_insert_labels(tree, ref)
    native_above, native_below = _native_labels(tree)
    last = getattr(menu, "_meta_last", None) or {}
    want_above = new_above or native_above
    want_below = new_below or native_below
    known_above = {native_above, last.get("above", native_above)}
    known_below = {native_below, last.get("below", native_below)}
    changed = False
    for i, label in _entry_labels(menu):
        want = want_above if label in known_above else want_below if label in known_below else None
        if want is not None and label != want:
            with suppress(Exception):
                menu.entryconfig(i, label=want)
                changed = True
    with suppress(Exception):
        menu._meta_last = {"above": want_above, "below": want_below}
    return changed


def _menu_for(tree: MetadataTree, side: str) -> Any | None:
    """Popup menu about to show on *side* (``"ri"`` tree column / ``"mt"`` cells)."""
    with suppress(Exception):
        if side == "ri":
            return tree.sheet.RI.ri_rc_popup_menu
        return tree.sheet.MT.rc_popup_menu
    return None


def _popup_iid(tree: MetadataTree, side: str, event: Any) -> Any | None:
    """Tree iid under a right-click — display row via ``identify_row`` (drift-tolerant)."""
    try:
        mt = tree.sheet.MT
        r: Any = None
        for arg in (event, getattr(event, "y", None)):
            with suppress(Exception):
                if (v := mt.identify_row(arg)) is not None:
                    r = int(v)
                    break
        if r is None:
            return None
        with suppress(Exception):
            if (walk := getattr(tree.host, "_walk", None)) is not None:
                if vis := list(walk(visible=True)):
                    return vis[r] if 0 <= r < len(vis) else None
        return tree.iid_at_row(r)
    except Exception:
        return None


def _install_rc_hook(tree: MetadataTree, policy: RowPolicy, side: str) -> bool:
    """Refresh labels + state (and the rc oracle) in the pre-popup hook slot."""
    try:
        host, sheet = tree.host, tree.sheet
        owner = sheet.RI if side == "ri" else sheet.MT
    except Exception:
        return False
    if getattr(owner, "_meta_labels_hooked", False):
        return True
    prev = owner.extra_rc_func

    def hook(event: Any = None, _prev: Any = prev) -> None:
        if _prev is not None:
            with suppress(Exception):
                _prev(event)
        try:
            if (iid := _popup_iid(tree, side, event)) is not None and hasattr(host, "_rc_sel_iid"):
                host._rc_sel_iid = iid  # right-clicked row, not a stale selection
            for menu in _all_menus(tree):
                refresh_menu_state(menu, tree, policy)
        except Exception:
            lf.exception("sheet-popup: pre-popup refresh failed")

    hook._meta_labels_hooked = True  # type: ignore[attr-defined]
    with suppress(Exception):
        owner.extra_rc_func = hook
        owner._meta_labels_hooked = True
        return True
    return False


def _install_structure_hook(tree: MetadataTree) -> bool:
    """Rebuild row caches + clear hover status after any allowed insert.

    tksheet inserts shift display rows while ``_vis``/``_int_row_of`` still map
    the old layout, so the next hover would publish another row's status
    message.  ``extra_end_insert_*_rc_func`` fires at the end of every
    ``add_rows``/``add_columns`` (menu, extras and API alike), except undo
    replays — exactly the allowed-insert paths (denied ones return early).
    """
    try:
        host = tree.host
        mt = tree.sheet.MT
        if not all(hasattr(host, n) for n in ("_rebuild_row_caches", "_clear_status")):
            return False
    except Exception:
        return False
    if getattr(mt, "_meta_structure_hooked", False):
        return True
    prev_rows, prev_cols = mt.extra_end_insert_rows_rc_func, mt.extra_end_insert_cols_rc_func

    def _after(_event: Any = None) -> None:
        with suppress(Exception):
            host._hide_hover_field()
        with suppress(Exception):
            host._clear_status()
        with suppress(Exception):
            host._rebuild_row_caches()
        with suppress(Exception):
            host.sh.after_idle(host._rebuild_row_caches)

    def hook_rows(event: Any = None, _prev: Any = prev_rows) -> None:
        if _prev is not None:
            with suppress(Exception):
                _prev(event)
        _after(event)

    def hook_cols(event: Any = None, _prev: Any = prev_cols) -> None:
        if _prev is not None:
            with suppress(Exception):
                _prev(event)
        _after(event)

    with suppress(Exception):
        mt.extra_end_insert_rows_rc_func, mt.extra_end_insert_cols_rc_func = hook_rows, hook_cols
        mt._meta_structure_hooked = True
        return True
    return False


def disable_unsafe_menus(sheet: Any) -> bool:
    """Switch sort + native column-insert entries off — re-applied after each ``enable_bindings(["all"])``."""
    with suppress(Exception):
        sheet.disable_bindings(list(OFF_BINDINGS))
        # lf.debug("sheet-popup: unsafe bindings disabled: %s", OFF_BINDINGS)
        return True
    return False


def disable_sort_menus(sheet: Any) -> bool:
    """Backward-compat alias of :func:`disable_unsafe_menus`."""
    return disable_unsafe_menus(sheet)


def _register_extras(tree: MetadataTree) -> bool:
    """Append-only extras — ``Add column`` (append at end), ``Add row`` (child of selection)."""
    try:
        sheet, host = tree.sheet, tree.host
    except Exception:
        return False
    done = False
    with suppress(Exception):
        if (col_fn := getattr(host, "_insert_col_at_end", None)) is not None:
            sheet.popup_menu_add_command(str(_S.get("sheet.insert_col", _NATIVE_ADD_COL)), col_fn)
            done = True
    with suppress(Exception):
        if (row_fn := getattr(host, "_add_row_child", None)) is not None:
            sheet.popup_menu_add_command(str(_S.get("sheet.insert_row", _NATIVE_ADD_ROW)), row_fn)
            done = True
    return done


def install_menu_patch(meta: Any, bridge: GroupUndoBridge | None = None) -> bool:
    """Patch once: unsafe bindings off, append-only extras, guard, per-popup state.

    *meta* is a ``MetadataTree`` (``MetadataTree(sheet, rows=...)``) or a
    ``ConfigSheet`` directly; *bridge* defaults to a new ``GroupUndoBridge``.
    Safe to call once per sheet — repeat calls only re-link the bridge.
    """
    try:
        tree = _as_tree(meta)
        if tree.sheet is None:  # fail fast without a sheet
            return False
    except Exception:
        return False
    if getattr(tree.sheet, "_meta_menu_installed", False):
        if bridge is not None:
            with suppress(Exception):
                tree.host._undo_bridge = bridge
        return True
    disable_unsafe_menus(tree.sheet)
    _register_extras(tree)
    if bridge is None:
        with suppress(Exception):
            bridge = GroupUndoBridge(tree)
        if bridge is None:
            return False
    with suppress(Exception):
        tree.host._undo_bridge = bridge  # split_setup coalesces through it
    policy = RowPolicy(tree)
    with suppress(Exception):
        tree.host._row_policy = policy
    guard = SheetGuard(tree, policy)
    with suppress(Exception):
        tree.host._row_guard = guard
    _install_rc_hook(tree, policy, "ri")
    _install_rc_hook(tree, policy, "mt")
    _install_structure_hook(tree)
    with suppress(Exception):
        tree.sheet._meta_menu_installed = True
    # lf.debug("sheet-popup: menu patch installed (guard=%s)", guard._patched)
    return True
