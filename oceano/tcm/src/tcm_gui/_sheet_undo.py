"""Group undo for metadata ``setup`` splits — interposed on tksheet's native chronology.

No second global undo system: native ``MT.undo``/``MT.redo`` (menu entries,
``Ctrl+Z``/``Ctrl+Shift+Z``) keep working for every ordinary edit.  Only the
metadata interval split (:meth:`MetadataNodeMixin.split_setup`, which rebuilds
the whole ``metadata`` subtree) is coalesced into a single step: the fragments
the rebuild pushes onto ``MT.undo_stack`` are discarded and one marker
(``GROUP_EVENT``) carrying ``before``/``after`` model snapshots takes their
place in the same native stack, so chronological order against cell edits is
preserved.

Interception point is ``MT.extra_begin_ctrl_z_func`` — tksheet calls it at the
top of both :meth:`undo` and :meth:`redo` (``begin_undo``/``begin_redo``).
``try_binding`` discards hook return values (only an exception vetoes the
native path), so the hook performs the snapshot restore itself and then raises
the private :class:`_VetoNativeUndo` to stop the native inversion, which could
not replay the custom payload.  Anything else (including hook internals
failing) falls through to the native behavior.

Sketch API::

    meta = MetadataTree(sheet, rows=current_metadata_rows)
    menu_patch = GroupUndoBridge(meta)
    install_menu_patch(meta, menu_patch)  # tcm_gui._sheet_popup
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext, suppress
from typing import Any

lf = logging.getLogger(__name__)

__all__ = ["GROUP_EVENT", "GroupUndoBridge", "MetadataTree", "suspend_native_pushes"]

GROUP_EVENT = "metadata_split"
"""``eventname`` of group markers pushed onto ``MT.undo_stack``/``MT.redo_stack``."""


class _VetoNativeUndo(Exception):
    """Control flow: snapshot already restored, skip tksheet's native inversion."""


def _take_snapshot(host: Any) -> Any:
    """Opaque model snapshot — ``MetadataTree.snapshot`` or ``_snapshot_group``."""
    if isinstance(host, MetadataTree):
        return host.snapshot()
    return host._snapshot_group()


def _apply_snapshot(host: Any, snap: Any) -> None:
    """Restore an opaque model snapshot — ``MetadataTree.restore`` or ``_restore_group``."""
    if isinstance(host, MetadataTree):
        host.restore(snap)
    else:
        host._restore_group(snap)


def _mt_of(meta: Any) -> Any:
    """Resolve tksheet's ``MainTable`` from a tree/host/``Sheet``/``MainTable``."""
    host = meta.host if isinstance(meta, MetadataTree) else meta
    with suppress(Exception):
        sh = getattr(host, "sh", None) or (
            None if hasattr(host, "undo_stack") else getattr(host, "sheet", None)
        )
        if sh is not None:
            host = sh
    if (mt := getattr(host, "MT", None)) is not None:
        return mt
    if hasattr(host, "undo_stack"):
        return host
    raise AttributeError(f"GroupUndoBridge: no tksheet MainTable on {type(meta).__name__}")


def _top_is_group(stack: Any) -> bool:
    """True when the top native-stack entry is a group marker."""
    with suppress(Exception, IndexError, AttributeError, TypeError, KeyError):
        if stack and stack[-1].get("data", {}).get("eventname") == GROUP_EVENT:
            return True
    return False


@contextmanager
def suspend_native_pushes(mt: Any) -> Iterator[None]:
    """Discard ``undo_stack`` fragments pushed inside the block; keep redo intact.

    Rebuilds (``Sheet.insert``/``del_rows`` with ``undo=True``) push one native
    entry per row — swapping in a scratch deque keeps the real chronology (and
    its oldest entries under ``max_undos``) untouched.  ``sheet_modified`` calls
    inside still purge/reassign ``redo_stack`` (native new-edit semantics), so
    the redo deque is pinned across the block and the caller decides explicitly
    (group push purges it, group undo/redo preserve it).
    """
    try:
        saved_undo, saved_redo = mt.undo_stack, mt.redo_stack
    except Exception:
        yield
        return
    mt.undo_stack = deque()
    try:
        yield
    finally:
        mt.undo_stack = saved_undo
        with suppress(Exception):
            mt.redo_stack = saved_redo


class MetadataTree:
    """Sketch-compatible facade over a metadata-owning host (normally ``ConfigSheet``).

    ``MetadataTree(sheet, rows=...)`` — *sheet* is the host; *rows* optionally
    seeds the model through the host's ``_metadata`` setter (single 11-array or
    ``[[num, arr], …]`` groups).  ``ConfigSheet`` itself satisfies the same
    protocol, so every consumer below also accepts it directly.
    """

    def __init__(self, sheet: Any, rows: Any = None) -> None:
        if rows is not None and hasattr(sheet, "_metadata"):
            with suppress(Exception):
                sheet._metadata = rows
        self._host = sheet

    @property
    def host(self) -> Any:
        """Wrapped model owner."""
        return self._host

    @property
    def sheet(self) -> Any:
        """tksheet ``Sheet`` for menu patching (``ConfigSheet.sh`` or the sheet itself)."""
        return getattr(self._host, "sh", self._host)

    @property
    def setups(self) -> Any:
        """``[[num, 11-array], …]`` model (or ``None`` before first build)."""
        return getattr(self._host, "_setups", None)

    @property
    def next_num(self) -> Any:
        """Autonumber the next split copy will take (``None`` when unknown)."""
        return getattr(self._host, "_next_setup_num", None)

    def iid_at_row(self, r: Any) -> Any | None:
        """Display row → tree iid (``None`` when unresolvable)."""
        with suppress(Exception):
            if (fn := getattr(self._host, "_iid_at_row", None)) is not None and r is not None:
                return fn(int(r))
        return None

    def selected_ref(self) -> tuple | None:
        """``("meta", None)`` / ``("setup", idx)`` / ``("row", idx)`` / ``None``."""
        m: Any = None
        with suppress(Exception):
            if (fn := getattr(self._host, "_rc_meta_ref", None)) is not None:
                m = fn()
        if not isinstance(m, dict):
            with suppress(Exception):
                m = (getattr(self._host, "_meta", None) or {}).get(getattr(self._host, "_rc_sel_iid", None))
        if not isinstance(m, dict):
            return None
        if m.get("is_metadata_root"):
            return ("meta", None)
        if m.get("is_setup"):
            return ("setup", m.get("setup_idx"))
        if m.get("is_metadata"):
            return ("row", m.get("setup_idx"))
        return None

    def insert(self, above: bool = True) -> bool:
        """Split the selected interval (``True`` ≈ *Insert rows above*)."""
        return bool(self._host.split_setup(above=above))

    def delete_selected_setup(self) -> bool:
        """No setup deletion exists — the native Delete entry stays; protocol stub."""
        return False

    def snapshot(self) -> Any:
        """Opaque model snapshot for the undo chronology."""
        return self._host._snapshot_group()

    def restore(self, snap: Any) -> None:
        """Restore a snapshot (rebuilds the subtree, native pushes suspended)."""
        self._host._restore_group(snap)


class GroupUndoBridge:
    """Coalesce split rebuilds into one native-chronology step; else delegate.

    Installs ``MT.extra_begin_ctrl_z_func`` once per ``MainTable`` (re-installs
    only swap the routed bridge).  :meth:`group` wraps the model mutation; the
    pre/post snapshots become the marker payload.  :meth:`undo`/``redo`` and
    :meth:`can_undo`/``can_redo` simply forward to the native stack, which
    routes group markers back here through the hook.
    """

    EVENT = GROUP_EVENT

    def __init__(self, meta: Any) -> None:
        self._meta = meta if isinstance(meta, MetadataTree) else MetadataTree(meta)
        mt = self._mt = _mt_of(self._meta)
        if not getattr(mt, "_group_undo_installed", False):
            mt._group_undo_prev = mt.extra_begin_ctrl_z_func
            mt.extra_begin_ctrl_z_func = self._hook
            mt._group_undo_installed = True
        mt._group_undo_bridge = self

    @property
    def meta(self) -> MetadataTree:
        """Wrapped metadata tree."""
        return self._meta

    @contextmanager
    def group(self) -> Iterator[GroupUndoBridge]:
        """Coalesce the wrapped block into a single native-chronology marker."""
        mt = self._mt
        before = _take_snapshot(self._meta)
        guard = getattr(self._meta.host, "_row_guard", None)
        ctx = guard.suspended() if guard is not None else nullcontext()
        with suspend_native_pushes(mt), ctx:
            yield self
        after = _take_snapshot(self._meta)
        with suppress(Exception):
            mt.purge_redo_stack()  # new edit invalidates redo — native semantics
        mt.undo_stack.append(
            {"name": GROUP_EVENT, "data": {"eventname": GROUP_EVENT, "before": before, "after": after}}
        )
        # lf.debug("group-undo: pushed marker (undo depth=%d)", len(mt.undo_stack))

    def can_undo(self) -> bool:
        """True when the native stack (group markers included) can undo."""
        with suppress(Exception):
            return bool(self._mt.undo_stack)
        return False

    def can_redo(self) -> bool:
        """True when the native stack (group markers included) can redo."""
        with suppress(Exception):
            return bool(self._mt.redo_stack)
        return False

    def undo(self, event: Any = None) -> Any:
        """Native undo — group markers on top are restored through the hook."""
        return self._mt.undo(event)

    def redo(self, event: Any = None) -> Any:
        """Native redo — group markers on top are restored through the hook."""
        return self._mt.redo(event)

    def _hook(self, event: Any) -> Any:
        """``extra_begin_ctrl_z_func`` — restore group markers, else delegate."""
        mt = self._mt
        name = event.get("eventname", "") if isinstance(event, dict) else getattr(event, "eventname", "")
        try:
            if (br := getattr(mt, "_group_undo_bridge", None)) is not None:
                if name == "begin_undo" and _top_is_group(getattr(mt, "undo_stack", None)):
                    try:
                        br._undo_group(mt)
                    except Exception:
                        lf.exception("group-undo: restore on undo failed")
                    raise _VetoNativeUndo()
                if name == "begin_redo" and _top_is_group(getattr(mt, "redo_stack", None)):
                    try:
                        br._redo_group(mt)
                    except Exception:
                        lf.exception("group-undo: restore on redo failed")
                    raise _VetoNativeUndo()
        except _VetoNativeUndo:
            raise
        except Exception:
            lf.exception("group-undo: hook routing failed; delegating to native")
        if (prev := getattr(mt, "_group_undo_prev", None)) is not None:
            result = prev(event)
            # Native undo/redo of a value edit — notify peers (hash guard in
            # _notify_metadata_changed makes coef-only undos a no-op).
            with suppress(Exception):
                mt._notify_metadata_changed()  # type: ignore[attr-defined]
            return result
        return None

    def _undo_group(self, mt: Any) -> None:
        """Pop the marker, restore ``before``, carry it to the redo stack."""
        marker = mt.undo_stack.pop()
        try:
            _apply_snapshot(self._meta, marker["data"]["before"])
        except Exception:
            mt.undo_stack.append(marker)  # failed restore keeps the chronology
            raise
        mt.redo_stack.append(marker)
        with suppress(Exception):
            mt.sheet_modified(marker["data"], purge_redo=False)
        with suppress(Exception):
            mt.PAR.emit_event("<<Undo>>", marker["data"])
        # Group undo restored the model — notify peers (split structure changed).
        with suppress(Exception):
            mt._notify_metadata_changed()  # type: ignore[attr-defined]

    def _redo_group(self, mt: Any) -> None:
        """Pop the marker, restore ``after``, carry it back to the undo stack."""
        marker = mt.redo_stack.pop()
        try:
            _apply_snapshot(self._meta, marker["data"]["after"])
        except Exception:
            mt.redo_stack.append(marker)
            raise
        mt.undo_stack.append(marker)
        with suppress(Exception):
            mt.sheet_modified(marker["data"], purge_redo=False)
        with suppress(Exception):
            mt.PAR.emit_event("<<Redo>>", marker["data"])
        # Group redo restored the model — notify peers.
        with suppress(Exception):
            mt._notify_metadata_changed()  # type: ignore[attr-defined]
