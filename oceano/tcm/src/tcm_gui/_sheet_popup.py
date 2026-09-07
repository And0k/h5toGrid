"""tksheet context-menu patch — sort removal (once) + per-popup split labels.

Owns everything ``ConfigSheet`` does with the sheet's *existing* popup menus —
never replaces or deletes them wholesale:

* one-time setup (:func:`install_menu_patch`): sorting entries off via
  ``disable_bindings`` (tksheet rebuilds every menu from flags on each popup,
  so entry surgery would not stick), the ``sheet.insert_col``/``sheet.insert_row``
  extras registered, the :class:`GroupUndoBridge` installed, and pre-popup
  refresh hooks set on ``MT.extra_rc_func`` / ``RI.extra_rc_func``;
* per-popup refresh (:func:`refresh_insert_labels`): only the existing *Insert
  rows above/below* entries are relabeled — and only when the popup target is
  the ``metadata`` root or a ``setup`` node, with ``{target}`` (:func:`target_label`)
  being the number the split copy will take.  Anywhere else the native
  labels *and* commands run untouched (the split routing itself stays in
  :meth:`MetadataNodeMixin._rc_add_rows`, which the menu commands reach
  through the long-standing ``MT.rc_add_rows`` shadow).

Undo/Redo/Delete entries are never touched — group-undo interposition lives in
:mod:`tcm_gui._sheet_undo` and vets only group markers.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

from ._i18n import STRINGS as _S
from ._sheet_undo import GroupUndoBridge, MetadataTree

lf = logging.getLogger(__name__)

__all__ = [
    "SORT_BINDINGS",
    "disable_sort_menus",
    "install_menu_patch",
    "refresh_insert_labels",
    "target_label",
]

SORT_BINDINGS = ("sort_cells", "sort_row", "sort_column", "sort_columns", "sort_rows")
"""tksheet bindings whose menu entries sort — disabled once (re-disabled after each ``enable_bindings(["all"])``)."""

_FALLBACK_ABOVE = "Insert setup {target} above"
_FALLBACK_BELOW = "Insert setup {target} below"
_NATIVE_ABOVE = "Insert rows above"
_NATIVE_BELOW = "Insert rows below"


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


def refresh_insert_labels(menu: Any, meta: Any, ref: tuple | None = None) -> bool:
    """Relabel the existing above/below entries for a metadata/setup target.

    Entries are matched by exact label — the current native labels or the ones
    rendered by the previous call (kept on ``menu._meta_last``) — so other
    languages and customized ``ops`` keep working.  Commands and state are
    never touched.  Returns True when any label changed.
    """
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
        return tree.iid_at_row(r) if r is not None else None
    except Exception:
        return None


def _install_rc_hook(tree: MetadataTree, side: str) -> bool:
    """Refresh labels (and the rc oracle) in the pre-popup hook slot, chaining any previous func."""
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
            refresh_insert_labels(_menu_for(tree, side), tree)
        except Exception:
            lf.exception("sheet-popup: pre-popup refresh failed")

    hook._meta_labels_hooked = True  # type: ignore[attr-defined]
    with suppress(Exception):
        owner.extra_rc_func = hook
        owner._meta_labels_hooked = True
        return True
    return False


def disable_sort_menus(sheet: Any) -> bool:
    """Switch all sorting entries off — once per ``enable_bindings(["all"])`` (see ``load()``)."""
    with suppress(Exception):
        sheet.disable_bindings(list(SORT_BINDINGS))
        lf.debug("sheet-popup: sort entries disabled")
        return True
    return False


def _register_extras(tree: MetadataTree) -> bool:
    """Move of ``ConfigSheet``'s *Insert column/row* (append-at-end) menu additions — idempotent."""
    try:
        sheet, host = tree.sheet, tree.host
    except Exception:
        return False
    done = False
    with suppress(Exception):
        if (col_fn := getattr(host, "_insert_col_at_end", None)) is not None:
            sheet.popup_menu_add_command(str(_S.get("sheet.insert_col", "Insert column")), col_fn)
            done = True
    with suppress(Exception):
        sheet.popup_menu_add_command(
            str(_S.get("sheet.insert_row", "Insert row")), lambda e=None: sheet.insert_row()
        )
        done = True
    return done


def install_menu_patch(meta: Any, bridge: GroupUndoBridge | None = None) -> bool:
    """Patch once: sort off, extras on, group-undo bridged, labels refreshed per popup.

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
    disable_sort_menus(tree.sheet)
    _register_extras(tree)
    if bridge is None:
        with suppress(Exception):
            bridge = GroupUndoBridge(tree)
        if bridge is None:
            return False
    with suppress(Exception):
        tree.host._undo_bridge = bridge  # split_setup coalesces through it
    _install_rc_hook(tree, "ri")
    _install_rc_hook(tree, "mt")
    with suppress(Exception):
        tree.sheet._meta_menu_installed = True
    lf.debug("sheet-popup: menu patch installed")
    return True
