"""Config-sheet ``metadata`` node — device-metadata model, nesting & interval split.

Owns everything :class:`tcm_gui.coef_sheet.ConfigSheet` does with the
``metadata`` node:

* subtree build (:meth:`MetadataNodeMixin._build_metadata`) — paired rows
  (``tcm._meta_pairs.PAIRS``) directly under the root for a single
  deployment interval; autonumbered ``setup`` sublevels (``0``, ``1``, …)
  only when the device actually carries several intervals (nested
  ``info_devices.yaml`` entry — *Multiple intervals* in
  ``oceano/meta_finder/docs/reference/io_formats.md``);
* read-back (:meth:`MetadataNodeMixin.get_edited_metadata_setups`) and dirty
  tracking vs the load snapshot;
* intercepting the sheet's built-in *Insert rows above/below* commands
  (``MT.rc_add_rows``) when the ``metadata`` root or a ``setup`` node is
  selected (:meth:`MetadataNodeMixin.split_setup`): the selected interval is
  copied to an adjacent autonumbered ``setup`` node with the shared boundary
  pinned into the copy — ``above``: ``time_range[1] := time_range[0]``,
  ``below``: ``time_range[0] := time_range[1]``.  Paired rows listed directly
  under ``metadata`` (single interval) first move into a new ``0`` node; the
  copy takes the next free number (``1`` for a fresh split).

The model is ``self._setups = [[num, 11-array], …]`` — *num* doubles as the
``setup`` node label and the ``info_devices.yaml`` station key.
``_metadata`` stays available as a property alias for the first array, so
App/scan call sites keep working unchanged.

Menu presentation lives elsewhere: :mod:`tcm_gui._sheet_popup` removes the
sorting entries once and relabels *Insert rows above/below* per popup, while
:mod:`tcm_gui._sheet_undo` coalesces each split into one native-chronology
undo step (``_snapshot_group``/``_restore_group`` below are its protocol).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any

from tcm import _constants, _meta_pairs

lf = logging.getLogger(__name__)


class MetadataNodeMixin:
    """``metadata``-node owner for :class:`tcm_gui.coef_sheet.ConfigSheet`.

    Model & split contract: module docstring.
    """

    # ------------------------------------------------------------- init
    def _init_metadata_node(self) -> None:
        """State + rc-menu hook — call once from ``ConfigSheet.__init__``."""
        self._setups: list[list[Any]] | None = None  # [[num, 11-array], ...]
        self._next_setup_num = 0  # autonumber for the next split copy
        self._cur_setup = 0  # last selected setup — split anchor for the root node
        # New/autofilled metadata (absent file, or a split) — unsaved until Run
        self._metadata_unsaved = False
        self._snap_meta: tuple = ()
        self._rc_sel_iid: Any = None  # last selected iid — rc-menu target oracle
        # Group-undo bridge (tcm_gui._sheet_popup.install_menu_patch sets the real
        # one; None keeps splits ungrouped).  Menu labels/sort-off live there too.
        self._undo_bridge: Any = None
        # Cross-tab metadata sync — App wires ``on_metadata_changed`` per sheet;
        # ``_metadata_pub`` is the last-published ``_meta_snap`` (None → next mutation
        # publishes); ``_sync_guard`` suppresses echo while a peer applies a mirror.
        self.on_metadata_changed: Any = None
        self._metadata_pub: tuple | None = None
        self._sync_guard: bool = False
        # Row-mutation authorization (install_menu_patch links policy + guard).
        self._row_policy: Any = None  # RowPolicy — single source of truth
        self._row_guard: Any = None  # SheetGuard — wraps tksheet mutation methods
        self._guard_suspended = 0  # counter — internal rebuilds bypass the guard
        self._user_rows: set = set()  # extras-added row iids — the only deletable rows
        self._user_col_base: int | None = None  # first appended column — trailing cols deletable
        # Intercept built-in "Insert rows above/below": an instance attribute
        # shadows the bound method for the menu lambdas (`MT.rc_add_rows(...)`).
        mt = self.sh.MT
        self._rc_add_rows_orig = mt.rc_add_rows
        mt.rc_add_rows = lambda where="above": self._rc_add_rows(where)

    @property
    def _metadata(self) -> list[Any] | None:
        """First setup's 11-array — compat alias for App/scan/tint call sites."""
        if setups := getattr(self, "_setups", None):
            return setups[0][1]
        return None

    @_metadata.setter
    def _metadata(self, md: Any) -> None:
        """Ingest scan/browse input — single 11-array or ``[[num, arr], …]`` groups.

        ``None``/empty → autofilled stub on build.  A nested device-file entry
        arrives as groups (``[[num, 11-array], …]``); numeric station keys are
        preserved, non-numeric renumbered sequentially (labels stay stable).
        """
        if not md:
            self._setups = None
            self._next_setup_num = 0
            self._cur_setup = 0
            return
        if isinstance(md, dict):  # {sid: arr} → groups
            md = [(k, v) for k, v in md.items() if isinstance(v, (list, tuple))]
        if (
            md
            and isinstance(md[0], (list, tuple))
            and len(md[0]) == 2
            and isinstance(md[0][1], (list, tuple))
        ):
            if all(str(k).isdigit() for k, _ in md):
                self._setups = [[int(k), list(v)] for k, v in md]
                self._next_setup_num = max(num for num, _ in self._setups) + 1
            else:
                self._setups = [[i, list(v)] for i, (_, v) in enumerate(md)]
                self._next_setup_num = len(self._setups)
        else:  # single 11-array (scan/legacy)
            self._setups = [[0, list(md)]]
            self._next_setup_num = 1
        self._cur_setup = 0

    # ------------------------------------------------------------- hooks
    def _user_row_set(self) -> set:
        """Extras-added row iids (created on demand — ``__new__`` harnesses skip ``__init__``)."""
        if (us := getattr(self, "_user_rows", None)) is None:
            us = self._user_rows = set()
        return us

    def _rc_add_rows(self, where: str) -> None:
        """Intercepted ``MT.rc_add_rows`` — policy-routed: split / native / deny+bell.

        The :class:`SheetGuard` deliberately skips this method (the shadow owns
        it); the same :class:`RowPolicy` verdict drives the menu refresh, so UI
        state and execution can never disagree.
        """
        if (pol := getattr(self, "_row_policy", None)) is not None:
            with suppress(Exception):
                perm = pol.insert_permission(pol.oracle_ref())
                if perm == "split":
                    return self.split_setup(above=where != "below", ref=self._rc_meta_ref())
                if perm == "deny":
                    with suppress(Exception):
                        self.sh.bell()
                    return None
                return self._rc_add_rows_orig(where)  # native — guard re-vets below
        if (m := self._rc_meta_ref()) is None:  # legacy path (no policy installed)
            return self._rc_add_rows_orig(where)
        self.split_setup(above=where != "below", ref=m)

    def _rc_meta_ref(self) -> dict | None:
        """Meta of the rc-menu target — metadata root / setup node, else None."""
        cand = [getattr(self, "_rc_sel_iid", None)]
        with suppress(Exception):
            cand += [self._iid_at_row(r) for r in self.sh.get_selected_rows()]
        return next(
            (
                m
                for iid in cand
                if (m := self._meta.get(iid)) and (m.get("is_setup") or m.get("is_metadata_root"))
            ),
            None,
        )

    # ------------------------------------------------------------- split
    def split_setup(self, *, above: bool, ref: dict | None = None) -> bool:
        """Copy the selected interval to an adjacent autonumbered ``setup`` node.

        ``above`` pins ``time_range[1] := time_range[0]`` into the copy,
        ``below`` pins ``time_range[0] := time_range[1]`` — the copy shares
        the boundary with the original, keeping the intervals adjacent.
        The rebuild coalesces into one native-chronology step through the
        group-undo bridge (``_undo_bridge``, installed by
        ``tcm_gui._sheet_popup.install_menu_patch``).  Returns True when the
        tree was rebuilt.
        """
        if not getattr(self, "_setups", None):
            return False
        br = getattr(self, "_undo_bridge", None)
        ctx = br.group() if br is not None else nullcontext()
        with ctx:
            setups = self._setups
            k = max(0, min(self._ref_setup_idx(ref), len(setups) - 1))
            cp = list(setups[k][1])
            src, dst = (6, 7) if above else (7, 6)
            if len(cp) > dst:
                cp[dst] = cp[src]
            setups.insert(k + (0 if above else 1), [self._next_setup_num, cp])
            self._next_setup_num += 1
            self._metadata_unsaved = True
            self._rebuild_metadata_rows()
        self._notify_metadata_changed()
        return True

    def _snapshot_group(self) -> tuple:
        """Opaque group-undo snapshot — consumed only by ``GroupUndoBridge``."""
        return (
            copy.deepcopy(getattr(self, "_setups", None)),
            getattr(self, "_next_setup_num", 0),
            getattr(self, "_cur_setup", 0),
            bool(getattr(self, "_metadata_unsaved", False)),
        )

    def _restore_group(self, snap: tuple) -> None:
        """Restore a group-undo snapshot — rebuilds the subtree, native pushes suspended."""
        setups, nxt, cur, unsaved = snap
        self._setups = copy.deepcopy(setups)
        self._next_setup_num = nxt
        self._cur_setup = cur
        self._metadata_unsaved = unsaved
        try:  # failure propagates — the bridge keeps the marker on its stack
            from tcm_gui._sheet_undo import suspend_native_pushes

            with suspend_native_pushes(self.sh.MT), self._suspend_guard():
                self._rebuild_metadata_rows()
        except ImportError:
            self._rebuild_metadata_rows()
        self._notify_metadata_changed()

    @contextmanager
    def _suspend_guard(self) -> Iterator[None]:
        """Internal rebuilds bypass SheetGuard (counter-style, nestable)."""
        self._guard_suspended = getattr(self, "_guard_suspended", 0) + 1
        try:
            yield
        finally:
            self._guard_suspended = max(0, getattr(self, "_guard_suspended", 0) - 1)

    def _ref_setup_idx(self, ref: dict | None) -> int:
        """Setup index implied by *ref* — the node's own or the last selected."""
        if ref and ref.get("is_setup"):
            self._cur_setup = int(ref["setup_idx"])
        return self._cur_setup

    # ---------------------------------------------------- cross-tab sync
    def _notify_metadata_changed(self) -> None:
        """Publish metadata state to same-identity peers (App fan-out).

        No-op when nothing actually changed vs the last publication — the hash
        guard leaves coef-only edits, peer echoes (``_sync_guard``) and
        programmatic refills silent.  ``_take_metadata_snapshot`` resets
        ``_metadata_pub`` to ``None``, so every rebuilt baseline forces the
        next real mutation to publish.
        """
        stem = getattr(self, "_page_stem", "?")
        # lf.debug("_notify_metadata_changed called on %s", stem)
        if getattr(self, "_sync_guard", False):
            # lf.debug("  -> suppressed by sync_guard")
            return
        snap = self._meta_snap()
        # lf.debug("  -> snap=%s", snap)
        if snap == getattr(self, "_metadata_pub", None):
            # lf.debug("  -> snap unchanged, skipping")
            return
        self._metadata_pub = snap
        cb = getattr(self, "on_metadata_changed", None)
        # lf.debug("  -> publishing (cb=%s)", "set" if cb else "None")
        if cb is not None:
            with suppress(Exception):
                cb()

    def apply_metadata_setups(self, setups: list[list[Any]], *, dirty: bool = True) -> None:
        """Peer mirror: replace the model with *setups* and rebuild the subtree.

        *dirty* propagates the source tab's metadata-dirty flag — a sync that
        carries a user edit must mark the peer dirty too, so its Run writes
        the new metadata.  Structural-only syncs (splits on a clean tab) pass
        ``dirty=False`` — the peer keeps its prior dirty state.
        """
        # lf.debug("apply_metadata_setups on %s setups=%s dirty=%s", getattr(self, "_page_stem", "?"), len(setups), dirty)
        self._sync_guard = True
        prior_unsaved = getattr(self, "_metadata_unsaved", False)
        try:
            self._setups = copy.deepcopy(setups)
            self._next_setup_num = 1 + max(num for num, _ in self._setups) if self._setups else 0
            self._cur_setup = 0
            self._rebuild_metadata_rows()
            self._metadata_pub = self._meta_snap()
            # lf.debug("  -> applied, new snap=%s", self._metadata_pub)
        finally:
            self._sync_guard = False
            # Dirty if source was dirty OR peer was already dirty
            self._metadata_unsaved = prior_unsaved or dirty
        self._apply_metadata_dirty_label()

    # ------------------------------------------------------------- build
    def _build_metadata(self) -> None:
        """Top-level ``metadata`` node (sibling of ``input``) with paired rows.

        ``metadata`` itself is the device-file path (always editable,
        browseable — same floated field as ``input``).  Children are the 6
        paired rows from ``_meta_pairs.PAIRS``; empty cells show gray example
        ghosts that vanish on edit (``CellPlaceholder``), never persisted as
        ``"?"``.  Several intervals → autonumbered ``setup`` sublevels (flat
        display only while a single interval remains).
        """
        if getattr(self, "_setups", None) is None:
            # Autofilled from an absent info_devices.yaml — seed from time_ranges
            tr = (self._cfg.get("input", {}) or {}).get("time_ranges") or []
            md: list[Any] = [None] * 11
            if tr and len(tr) >= 2:
                md[6], md[7] = tr[0], tr[-1]
            self._setups, self._next_setup_num, autofilled = [[0, md]], 1, True
        else:
            autofilled = False

        # Device-file path — always editable (default when file absent).
        _path = getattr(self, "_metadata_path", None)
        if _path is None:
            _path = self._default_metadata_path()
        # Autofilled from an absent info_devices.yaml — unsaved only when
        # there's a real file to save to (non-empty derived path).  The old
        # guard ``any(not is_placeholder…)`` missed the empty stub with no
        # ``time_ranges``, and the unconditional ``autofilled`` kept the
        # default page (no input.path → _path=="") dirty forever.
        self._metadata_unsaved = autofilled and bool(_path and str(_path).strip())

        meta_iid = self._ins(
            "",
            "metadata",
            [_path] + [""] * (self._nv - 1),
            "",
            meta={
                "key": "metadata",
                "path": "metadata",
                "style": "node",
                "max_col": 1,
                "is_metadata_root": True,
                "is_string": True,
                "browse": True,
                "check": "exists",
                "metadata_path": _path,
            },
            open_=True,
        )
        if len(self._setups) == 1:  # nested level shown only when required
            self._ins_paired_rows(meta_iid, self._setups[0][1], 0)
        else:
            for idx, (num, arr) in enumerate(self._setups):
                sid = self._ins(
                    meta_iid,
                    str(num),
                    [""] * self._nv,
                    "",
                    meta={"style": "node", "is_setup": True, "setup_idx": idx, "max_col": 0},
                    open_=True,
                )
                self._ins_paired_rows(sid, arr, idx)
        self._take_metadata_snapshot()

    def _default_metadata_path(self) -> str:
        """device_dir/info_devices.yaml (parent of ``_raw``) for the loaded data."""
        try:
            from tcm import paths

            # Judge the RAW string before resolving — ``Path("").absolute()``
            # is the cwd, so a no-path GUI launch would otherwise probe the
            # launch directory and ``find_dir_raw_absolute`` would log the
            # misleading "Not standard input path" warning on the project
            # root.  Repo-internal paths are skipped too: their default
            # device file would land in the code tree — the scan owns that
            # error verdict, not this default derivation.
            probe_str = str((self._cfg.get("input", {}) or {}).get("path") or "").strip()
            probe = Path(probe_str).absolute() if probe_str and probe_str != "." else None
            if probe is not None and (probe == _constants.REPO_ROOT or _constants.REPO_ROOT in probe.parents):
                probe = None
            ddir = paths.find_dir_raw_absolute(probe).parent if probe is not None else None
            return str(ddir / "info_devices.yaml") if ddir else ""
        except Exception:
            return ""

    def _ins_paired_rows(self, parent_iid: Any, arr: list[Any], setup_idx: int) -> None:
        """Six paired rows (``_meta_pairs.PAIRS``) under *parent_iid* for 11-array *arr*."""
        arr = list(arr) + [None] * max(0, 11 - len(arr))
        paired = _meta_pairs.to_display(arr)
        for label, idxs in _meta_pairs.PAIRS:
            vals = paired.get(label, ["?"] * len(idxs))
            # Empty "?" → ghost via CellPlaceholder, not literal "?".
            # Distinguish vacuous vs valued "?" by checking arr indices.
            display_vals: list[str] = []
            ghost_cols: list[int] = []
            for j, v in enumerate(vals):
                idx = idxs[j]
                raw = arr[idx] if idx < len(arr) else None
                if v == "?" and _meta_pairs.is_placeholder(raw):
                    display_vals.append("")
                    ghost_cols.append(j)
                else:
                    display_vals.append(v)
            is_time = label == "time_range"
            iid = self._ins(
                parent_iid,
                label,
                display_vals + [""] * (self._nv - len(display_vals)),
                "",
                meta={
                    "label": label,
                    "path": f"metadata.{label.replace(', ', '_').replace('/', '_')}",
                    "is_string": True,
                    "is_metadata": True,
                    "max_col": len(idxs),
                    "has_date": is_time,
                    "setup_idx": setup_idx,
                    "_ghost_cols": ghost_cols,
                },
            )
            # Remember ghosts for placeholder pass — _ph.show needs row index later
            if ghost_cols:
                self._meta[iid]["_ghost_example"] = [
                    _meta_pairs.EXAMPLES.get(label, ["?", "?"])[j] for j in range(len(idxs))
                ]

    # --------------------------------------------------------- read-back
    def get_metadata_path(self) -> str:
        """Device-file path — edited root cell, else the resolved/loaded one."""
        for iid, m in self._meta.items():
            if m.get("is_metadata_root"):
                if not (s := self._cell_str(iid, 0)):
                    return str(getattr(self, "_metadata_path", "") or "")
                return s
        return str(getattr(self, "_metadata_path", "") or "")

    def get_edited_metadata_setups(self) -> list[list[Any]]:
        """Read every setup's paired rows → ordered 11-arrays for write-back.

        Ghost placeholders (``_ph``) read as ``""`` → ``None`` → ``~``
        (required) or trimmed tail — identical to the coefs date extraction.
        """
        row_of = self._row_map()
        ph = getattr(self, "_ph", None)
        out: list[list[Any]] = []
        for idx, (_, base) in enumerate(getattr(self, "_setups", None) or []):
            paired: dict[str, list[str]] = {}
            for iid, m in self._meta.items():
                if not m.get("is_metadata") or m.get("setup_idx") != idx:
                    continue
                r = row_of.get(iid)
                vals: list[str] = []
                for j in range(int(m.get("max_col", 1))):
                    if r is not None and ph is not None and hasattr(ph, "has") and ph.has(r, j):
                        vals.append("")
                    else:
                        raw = self.sh.item(iid).get("values") or ()
                        vals.append(str(raw[j]) if j < len(raw) else "")
                paired[m["label"]] = vals
            out.append(_meta_pairs.to_storage(paired, base=list(base)) if paired else list(base))
        return out

    def get_edited_metadata(self) -> list[Any]:
        """First setup's 11-array — single-interval write-back contract."""
        setups = self.get_edited_metadata_setups()
        return setups[0] if setups else []

    def get_edited_metadata_map(self) -> dict[str, list[Any]]:
        """``{station_key: 11-array}`` for ``info_devices.yaml`` write-back.

        Keys are the setup numbers (loaded station keys / split autonumbers) —
        ``App._write_metadata`` merges this per pcid, preserving nesting.
        """
        arrs = self.get_edited_metadata_setups()
        setups = getattr(self, "_setups", None) or []
        return {str(num): arr for (num, _), arr in zip(setups, arrs)}

    def autofill_burst(self, bdt: Any, bst: Any, *, time_ranges: list[Any] | None = None) -> bool:
        """Seed/backfill the burst pair (indices 8, 9) of the first setup.

        Creates an 11-array stub (time_ranges → 6, 7) when none exists, then
        fills the burst fields into placeholders.  ``True`` when anything
        changed — App marks the tab dirty so Run writes ``info_devices.yaml``.
        """
        arr: list[Any] = []
        fresh = False
        if getattr(self, "_setups", None) is None:
            arr = [None] * 11
            if time_ranges and len(time_ranges) >= 2:
                arr[6], arr[7] = time_ranges[0], time_ranges[1]
            self._setups = [[0, arr]]
            self._next_setup_num = 1
            fresh = True
        else:
            arr = self._setups[0][1]
        if len(arr) < 11:
            arr = list(arr) + [None] * (11 - len(arr))
            self._setups[0][1] = arr
        cur_bdt = arr[8] if len(arr) > 8 else None
        cur_bst = arr[9] if len(arr) > 9 else None
        if _meta_pairs.is_placeholder(cur_bdt) or _meta_pairs.is_placeholder(cur_bst):
            if fresh or str(cur_bdt) != str(bdt) or str(cur_bst) != str(bst):
                arr[8], arr[9] = bdt, bst
                self._metadata_unsaved = True
                return True
        return False

    def _meta_snap(self) -> tuple:
        """Hashable snapshot of all setups' edited rows."""
        return tuple(
            tuple("?" if v is None else str(v) for v in arr) for arr in self.get_edited_metadata_setups()
        )

    def is_metadata_dirty(self) -> bool:
        """True when any setup's rows differ from the load snapshot, or the
        metadata was autofilled from an absent info_devices.yaml (new —
        saved by ``App._write_metadata``)."""
        if getattr(self, "_metadata_unsaved", False):
            return True
        return (snap := getattr(self, "_snap_meta", None)) is not None and snap != self._meta_snap()

    def _take_metadata_snapshot(self) -> None:
        self._snap_meta = self._meta_snap()
        # Rebuilt baseline — next real mutation publishes to peers (None = arm).
        self._metadata_pub = None

    def mark_metadata_clean(self) -> None:
        self._metadata_unsaved = False
        self._take_metadata_snapshot()

    # -------------------------------------------------------- structure
    def _reload_metadata_from(self, path_str: str) -> None:
        """Load metadata from an existing file on browse select.

        Keeps **every** interval of the device entry — a nested
        ``info_devices.yaml`` structure becomes ``setup`` sublevels (station
        keys are reused as node numbers when numeric).
        """
        try:
            from meta_finder.io_info_files import read_metadata_file

            from tcm import format as _fmt

            data = read_metadata_file(Path(path_str).expanduser())
            stem = getattr(self, "_page_stem", "") or ""
            pcid = _fmt.to_pcid_from_name(_fmt.stem_to_pcid(stem)) if stem else None
            ent = None
            if pcid is not None:
                for cand in (pcid, pcid.replace("_", "")):
                    if cand in data:
                        ent = data[cand]
                        break
            if ent is None and data:
                ent = next(iter(data.values()))
            groups = (
                [(k, v) for k, v in ent.items() if isinstance(v, (list, tuple))]
                if isinstance(ent, dict)
                else [(0, ent)]
                if isinstance(ent, (list, tuple))
                else []
            )
            if not groups:
                return
            # Non-numeric station keys → renumber sequentially (file order kept)
            if all(str(k).isdigit() for k, _ in groups):
                self._setups = [[int(k), list(v)] for k, v in groups]
                self._next_setup_num = max(num for num, _ in self._setups) + 1
            else:
                self._setups = [[i, list(v)] for i, (_, v) in enumerate(groups)]
                self._next_setup_num = len(groups)
            self._metadata_path = path_str
            self._rebuild_metadata_rows()
            self._notify_metadata_changed()
        except Exception:
            lf.exception("Failed to reload metadata from %s", path_str)

    def _rebuild_metadata_rows(self) -> None:
        """Rebuild only the metadata subtree — keep node expanded and ghosts visible."""
        # lf.debug("_rebuild_metadata_rows on %s: _setups=%s", getattr(self, "_page_stem", "?"), [(n, arr[:3] if arr else None) for n, arr in (self._setups or [])])
        # Discard stale placeholders before structural change — row indices will shift
        self._ph.clear_all(self.sh)
        guard = getattr(self, "_row_guard", None)
        ctx = guard.suspended() if guard is not None else nullcontext()
        with ctx:  # internal deletes/inserts bypass SheetGuard (user paths stay vetted)
            to_del = [
                iid
                for iid, m in list(self._meta.items())
                if m.get("is_metadata") or m.get("is_metadata_root") or m.get("is_setup")
            ]
            for iid in to_del:
                with suppress(Exception):
                    self.sh.delete_row(self._row_map().get(iid, -1))
                self._meta.pop(iid, None)
            self._build_metadata()
        self._apply_open()
        self._rebuild_row_caches()
        self._apply_styles()
        self._apply_placeholders()
        self._apply_default_fg()
        self._apply_validations()
        with suppress(Exception):
            self.sh.redraw()
        # Rebuilt subtree got fresh iids — stale `_snap` would flag the tab
        # dirty (and rewrite YAML on Run) without any user edit.
        self._take_metadata_snapshot()
        self._take_snapshot()
        with suppress(Exception):
            self._apply_metadata_dirty_label()
