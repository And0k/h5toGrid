"""Sheet hover: status-bar publication and the floated PathField overlay.

Mixin for :class:`tcm_gui.coef_sheet.ConfigSheet`.

* **Status bar** (:meth:`SheetHoverMixin._publish_status`) — hovered row's doc
  short text from ``config_reference.md`` (:mod:`tcm_gui._help`), with a live
  ``time_ranges`` sync detail appended (recomputed per publish — never a stale
  scan-time message) and Shift-toggled dir/file hints for ``path`` under ``coefs``.
* **Floated PathField** (:meth:`SheetHoverMixin._show_hover_field`) — after
  ``_INTENT_MS`` dwell on a browse row, a reusable :class:`tcm_gui._path_field.PathField`
  covers the row plus a :class:`tcm_gui._browse_button.BrowseOverlay` at the
  right edge.  Empty commits propagate to :meth:`SheetHoverMixin._hover_write`
  which writes ``""`` and restores the ghost — deletion is a first-class value.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from tkinter import TclError
from types import SimpleNamespace
from typing import Any, Final

from tcm_gui import _help, _path_field

from ._browse_button import (
    COEF_FILETYPES,
    DATA_FILETYPES,
    BrowseOverlay,
    _is_shift_pressed,
    _pointer_inside,
)
from ._i18n import STRINGS as _S, fmt_status

_INTENT_MS: Final[int] = 120  # hover-intent delay for the floated PathField (ms)

# Coef rows with ``has_date=True`` fall back to the generic ``input.coefs.dates``
# doc short ("Per-component calibration dates").  Map coef key → i18n key so the
# status shows a component-specific label instead (e.g. "Accelerometer
# calibration date" for Ag, whose date cell is shared by Cg).
_COEF_DATE_LABELS: dict[str, str] = {
    "Ag": "input.coefs.date.accelerometer",
    "Ah": "input.coefs.date.magnetometer",
    "Rz": "input.coefs.date.alignment",
    "P_t": "input.coefs.date.pressure",
    "kVabs": "input.coefs.date.velocity",
}

_l = logging.getLogger(__name__)


class SheetHoverMixin:
    """Hover status publication + floated PathField edit surface for browse rows."""

    # No-op overlay — replaces PathField's internal BrowseOverlay so its
    # SheetHoverBinder never creates a second button.
    _NULL_OV = SimpleNamespace(
        visible=False,
        pending=False,
        show=lambda **_kw: None,
        hide=lambda: None,
        schedule_show=lambda _kw, **_a: None,
        schedule_hide=lambda **_a: None,
        cancel_show=lambda: None,
        cancel_hide=lambda: None,
    )

    # ── status publication ────────────────────────────────────────────

    def _clear_status(self) -> None:
        """Reset hover status tracking and clear the status bar."""
        self._status_iid = None
        self._status_source = None
        self._status_col = None
        self._hover_detail = ""
        self._publish_status(None)

    def _on_sheet_leave(self, _event) -> None:
        """``<Leave>`` also fires when the pointer steps onto the field —
        the delayed hide's pointer check decides; status preserved if the
        pointer merely moved onto the field (same row) or a show was pending."""
        had_pending_show = self._field_show_job is not None
        self._schedule_field_hide()
        if not self._pointer_in_field() and not had_pending_show:
            self._clear_status()

    def _on_sheet_wheel(self, _event) -> None:
        """Scroll changes row hit-testing — immediate hide, clear status."""
        self._hide_hover_field()
        self._clear_status()

    def _help_candidates(self, iid: Any, *, tree: bool = False, col: int | None = None) -> list[str]:
        """Ordered ``help_for_path`` candidates for *iid*.

        Single source for ``_on_tree_motion`` and ``_publish_status`` — paired
        metadata rows (``point, symbol``) try both ``metadata.point`` and
        ``metadata.symbol``, the ``metadata`` root tries ``metadata.path``,
        otherwise the node's own ``path``.  Tree motion shows the first field
        only; data-cell hover fans both.

        *col* selects the field for metadata paired rows: the candidate whose
        column is hovered is tried first, so col 1 (``symbol``) shows the
        ``metadata.symbol`` short instead of always the first field.
        """
        m = self._meta.get(iid, {})
        path = str(m.get("path") or "")
        if m.get("is_metadata_root"):
            # Tree column shows the ``metadata`` section (status + dwell tooltip);
            # the data cell (path field) keeps ``metadata.path``.
            return ["metadata"] if tree else ["metadata.path"]
        if m.get("is_metadata") and (lbl := m.get("label")):
            # Label separators: ``,`` pairs fields, ``/`` the burst row's
            # shortened second field (``burst_dt/t`` → ``bursts_t`` key).
            parts = [s.strip().replace(" ", "_") for s in re.split(r"[,/]", lbl) if s.strip()]
            field_cands = [f"metadata.{parts[0]}"]
            if not tree:
                # ``t`` is the label-shortened ``bursts_t`` doc key.
                field_cands += [f"metadata.{p if p != 't' else 'bursts_t'}" for p in parts[1:]]
            # Column-specific: put the hovered field first so it wins in _publish_status.
            if not tree and col is not None and 0 <= col < len(field_cands):
                field_cands = [field_cands[col]] + field_cands[:col] + field_cands[col + 1 :]
            cands = field_cands
            if path not in cands:
                cands.append(path)
            return cands
        cands: list[str] = []
        if not tree and m.get("has_date"):
            cands += [f"{path}.date", f"{path}.dates"]
            if (par := m.get("parent")) and (pp := self._meta.get(par, {}).get("path")):
                cands += [f"{pp}.dates", f"{pp}.date"]
        if path:
            # `input.path` lives in the `input` node's own cell (col 0), not as a
            # separate child row — its field docs must show on data-cell hover
            # (tree=False), but the tree column (tree=True) must show the section
            # `## input` so children never affect the parent. Other containers
            # (e.g. `coefs` with literal `path` child `input.coefs.path`) never
            # get the `.path` suffix — leaf row `path="input.coefs.path"` shows
            # its own `### input.coefs.path` docs via `cands=[path]`.
            if not tree and path == "input":
                cands += [f"{path}.path", path]
            else:
                cands.append(path)
        return cands

    def _on_tree_motion(self, event) -> None:
        """Hover over tree column (index canvas) — show help for that node.

        For metadata paired rows (``point, symbol`` …) the tree column shows
        the first field's short text; a ``", …"`` suffix is appended when the
        row documents more than one editable column, signalling the second
        field shown on data-cell hover.
        """
        if (hit := self._hover_resolve(event)) is None:
            self._clear_status()
            return

        iid, _row, _y = hit
        if iid == self._status_iid and self._status_source == "tree":
            return

        self._status_iid = iid
        self._status_source = "tree"
        m = self._meta.get(iid, {})
        multi = m.get("is_metadata") and m.get("max_col", 1) > 1
        for cand in self._help_candidates(iid, tree=True):
            if cand and (h := _help.help_for_path(cand)) and (txt := _help.section_body_short(h)):
                self._hover_detail = self._resolve_detail(cand)
                if self.on_hover_status is not None:
                    txt = f"{txt}{_S['metadata.tree_suffix']}" if multi else txt
                    self.on_hover_status(txt, True)
                return
        if self.on_hover_status is not None:
            self._hover_detail = ""
            self.on_hover_status(str(m.get("key") or m.get("label") or m.get("path") or ""), False)

    def _coefs_status_hint(self) -> str:
        """Mode-aware status hint for ``input.coefs`` (path) browse button.

        General = ``### `input.coefs.path` `` short body (pre-``####``) from
        ``config_reference.md``; suffix = mode-specific GUI hint from ``STR``
        (``input.coefs.path.status.dir/files``).  Shift held → ``file`` mode;
        default → ``dir`` mode.  Same augmentation pattern as ``path_field``
        and ``time_ranges.hover.*``.
        """

        def _suffix_for(m: str) -> str:
            if v := _S.get(f"input.coefs.path.status.{m}s"):
                return str(v)
            return ""

        mode = "file" if _is_shift_pressed() else "dir"
        # Doc general (fallback to _NO_MODE when mode-specific section absent)
        base = ""
        if (h := _help.help_for_path("input.coefs.path", mode=mode)) and isinstance(h.body, str) and h.body:
            base = str(h.body)
        elif (ge := _help.help_for_path("input.coefs.path")) and isinstance(
            getattr(ge, "body", None), Mapping
        ):
            raw = ge.body.get(_help._NO_MODE)  # type: ignore[attr-defined]
            if isinstance(raw, _help.ModeBody):
                base = raw.short
            elif isinstance(raw, str):
                base = raw
        suffix = _suffix_for(mode)
        if base and suffix:
            return f"{base} {suffix}"
        if base:
            return base
        if suffix:
            return suffix
        return _S["browse_btn.status_files" if mode == "file" else "browse_btn.status"]

    def _on_shift_toggle(self, _event) -> None:
        """Re-publish status when Shift is pressed/released while hovering input.coefs (path).

        Gate: ``_status_iid`` is only set while the pointer is actively on a
        row (cleared by ``_clear_status`` on leave/blank-area) — unlike
        ``_field_iid`` which survives hide for deferred ``after_idle`` commits.
        The ``_pointer_in_field`` branch covers the case where the pointer
        stepped onto the browse button or floated PathField.
        """
        iid = self._status_iid
        if iid is None:
            return
        if self._meta.get(iid, {}).get("key") not in ("path", "coefs"):
            return
        # Pointer on the hover button → its poll (_update_icon) re-publishes
        # the button-specific hint on this transition; publishing the row
        # text here would overwrite it.
        if self._hover_btn is not None and self._hover_btn._hovered:
            return
        if not (self._pointer_in_field() or _pointer_inside(self.sh.MT)):
            return
        self._publish_status(iid)

    def _f1_anchor(self) -> str:
        """Anchor of the ``config_reference`` section F1 should open.

        Target = the selected row (``sh.tree_selected`` — the current
        selection box's iid); if nothing selected but the mouse is inside
        the dwell tooltip widget (``_hover_field``), use that row
        (``_status_iid``).  Mouse over other sheet elements is not tracked
        for F1 — nothing selected and no tooltip hover means the App opens
        the readme.  Resolution fans :meth:`_help_candidates` (paired
        metadata rows try every split label) then walks ``meta["parent"]`` —
        child rows of an undocumented node inherit their ancestor's section
        (``Ag[0]`` → ``input.coefs.Ag`` → the ``input.coefs`` group).
        ``help_for_path`` strips array indices; ``entry.anchor`` is the
        GitHub-style slug mirroring ``browser/web/viewer.js::slugify``.
        ``""`` when nothing documents the chain — the App then opens the
        readme instead.
        """
        iid = self.sh.tree_selected or (self._status_iid if self._pointer_in_field() else None)
        return self._f1_anchor_for_iid(iid)

    def _f1_help_candidates(self, iid: Any) -> list[str]:
        """F1 candidate ordering — node's own path first for metadata rows.

        Status text uses :meth:`_help_candidates` (metadata.X first).  F1
        prefers the specific field (e.g. ``input.time_ranges``) over the
        generic ``metadata.time_ranges`` — the same row documents the field,
        not the metadata group.
        """
        cands = self._help_candidates(iid)
        m = self._meta.get(iid, {})
        path = str(m.get("path") or "")
        if path and m.get("is_metadata") and path in cands:
            cands = [path] + [c for c in cands if c != path]
        return cands

    def _f1_anchor_for_iid(self, iid: Any) -> str:
        """Resolve the F1 anchor for the given *iid* (or ``""`` if undocumented).

        Shared resolution logic: fan out :meth:`_f1_help_candidates` (node's
        own path first for metadata rows), then walk ``meta["parent"]`` — child
        rows of an undocumented node inherit their ancestor's section.  Used by
        :meth:`_f1_anchor` (selection / dwell tooltip) and by the App when the
        mouse is over the status label.
        """
        while iid is not None:
            for cand in self._f1_help_candidates(iid):
                if cand and (e := _help.help_for_path(cand)):
                    return e.anchor
            iid = self._meta.get(iid, {}).get("parent")
        return ""

    @staticmethod
    def _resolve_detail(path: str) -> str:
        """Resolve the dwell tooltip text — ``#### Detailed`` blocks only.

        Scans every ``###`` section of the field (mode-tagged or modeless)
        plus the field-level block; a tooltip exists ⟺ some section carries a
        ``Detailed`` block.  Section short bodies and group prose never arm
        the dwell — regression: every coef row showed the ``input.coefs``
        group text instead of nothing.  Returns ``""`` — the caller skips arming.

        A bare ``### Detailed`` heading (no backticks, e.g. the RU doc's
        ``metadata`` section) stores its content as the mode body's ``short``
        under the ``"Detailed"`` tag — check that too.
        """
        if (e := _help.help_for_path(path)) and isinstance(e.body, Mapping):
            for tag, val in e.body.items():
                if isinstance(val, _help.ModeBody):
                    if tag == "Detailed":
                        return val.short
                    if d := val.details.get("Detailed"):
                        return str(d)
        return ""

    def _publish_status(self, iid: Any, col: int | None = None) -> None:
        """Status text for the hovered element (data cells on MT canvas).

        Fallback chain: :func:`tcm_gui._help.section_body_short` (prefers the
        ``###`` section lead-in text below the table, falls back to the table
        row last cell) → ``key`` → ``label`` → ``path``.  The ``time_ranges``
        row appends the *live* sync detail from :meth:`_time_ranges_detail`;
        ``input.coefs`` (path) shows Shift-toggled dir/file content.
        Tree-column hover is handled by ``_on_tree_motion`` which always uses
        the section-level path.

        *col* selects which field of a metadata paired row is shown — without
        it the first field always wins.
        """
        if self.on_hover_status is None:
            return

        if iid is None:
            self._hover_detail = ""
            self.on_hover_status("", False)
            return

        m = self._meta.get(iid, {})
        ident = str(m.get("key") or m.get("path") or m.get("label") or "")

        # Instant-apply checkbox cells (outside max_col) — STR-driven, one box per row.
        # Ready → pending status; incomplete → fill-in hint; synced → silent.
        if col is not None and (boxes := getattr(self, "_apply_boxes", None)):
            for _kind, _box in boxes.items():
                if _box.get("iid") == iid and _box.get("col") == col:
                    _pre = f"input.calib.{_kind}"
                    _state = _box.get("state") or ("ready" if _box.get("pending") else "empty")
                    if _state == "ready":
                        self._hover_detail = _S.get(f"{_pre}.apply.detailed", "")
                        self.on_hover_status(_S.get(f"{_pre}.apply.status.pending", ""), True)
                        return
                    if _state == "incomplete":
                        self._hover_detail = _S.get(f"{_pre}.apply.detailed", "")
                        self.on_hover_status(_S.get(f"{_pre}.apply.status.incomplete", ""), True)
                        return

        # time_ranges: doc short + live sync detail — recomputed, never cached
        if m.get("label") == "time_ranges" or str(m.get("path") or "") == "input.time_ranges":
            sync = self._time_ranges_detail()
            for cand in self._help_candidates(iid, tree=False, col=col):
                if cand and (h := _help.help_for_path(cand)) and (txt := _help.section_body_short(h)):
                    txt = f"{txt} — {sync}" if sync else txt
                    self._hover_detail = self._resolve_detail(cand) or self._resolve_detail(
                        str(m.get("path") or "")
                    )
                    self.on_hover_status(txt, True)
                    return
            self._hover_detail = ""
            self.on_hover_status(sync, False)
            return

        # Mode-aware: input.coefs (path) shows dir/file content from config_reference.md
        # instead of the table-row short text.  Shift toggles mode.  For the
        # coefs node also append the GUI behaviour and recorded date suffixes.
        if ident in ("path", "coefs") and (txt := self._coefs_status_hint()):
            detail = self._resolve_detail("input.coefs.path")
            if gui_detailed := _S.get("input.coefs.path.detailed.gui", ""):
                detail = f"{detail} {gui_detailed}".strip() if detail else gui_detailed
            self._hover_detail = detail
            # status GUI behaviour
            if gui_status := _S.get("input.coefs.path.status.gui", ""):
                txt = f"{txt} {gui_status}".strip()
            if ident == "coefs" and (coefs_date := m.get("_coefs_date")):
                if suffix := fmt_status(_S.get("input.coefs.date.status", ""), date=coefs_date):
                    txt = f"{txt} {suffix}".strip()
            self.on_hover_status(txt, True)
            return

        for cand in self._help_candidates(iid, tree=False, col=col):
            if cand and (h := _help.help_for_path(cand)) and (txt := _help.section_body_short(h)):
                self._hover_detail = self._resolve_detail(cand) or self._resolve_detail(
                    str(m.get("path") or "")
                )
                # Coef row with a date cell → component-specific status
                # instead of the generic "Per-component calibration dates".
                if (
                    m.get("has_date")
                    and not m.get("is_metadata")
                    and (lbl := _COEF_DATE_LABELS.get(str(m.get("key") or "")))
                    and (date_txt := _S.get(lbl, ""))
                ):
                    self.on_hover_status(date_txt, True)
                    return
                # Append coefs date suffix when hovering the coefs node
                if m.get("key") == "coefs" and (coefs_date := m.get("_coefs_date")):
                    suffix = fmt_status(_S.get("input.coefs.date.status", ""), date=coefs_date)
                    if suffix:
                        self.on_hover_status(f"{txt} {suffix}".strip(), True)
                        return
                # Pending calib triggers hint at the instant-apply checkbox.
                # Ready → apply suffix; incomplete → fill-in suffix; empty → doc only.
                _path = str(m.get("path") or "")
                _kind = _path.rsplit(".", 1)[-1] if _path.startswith("input.calib.") else None
                if _kind not in ("g0xyz", "coordinates", "azimuth_add"):
                    _kind = None
                if _kind is not None:
                    _state = "empty"
                    with suppress(Exception):
                        _state = str(self.apply_state(_kind))
                    if _state in ("ready", "incomplete"):
                        _pre = f"input.calib.{_kind}"
                        _key = f"{_pre}.status.gui" if _state == "ready" else f"{_pre}.status.incomplete"
                        if sfx := _S.get(_key, ""):
                            txt = f"{txt} {sfx}".strip()
                self.on_hover_status(txt, True)
                return

        self._hover_detail = ""
        self.on_hover_status(str(m.get("key") or m.get("label") or m.get("path") or ""), False)

    # ── hovered-field value I/O ───────────────────────────────────────

    def _hover_write(self, text: str) -> None:
        """Write path to column 0 of the hovered row + restyle.

        ``text == ""`` is a deletion: the cell is emptied and the ghost
        placeholder restored — an empty commit must not leave the previous
        value in the model (regression: deleted path resurrected on hover).
        A non-empty commit drops stale ghost tracking first — otherwise
        ``_hover_read``/``_cell_str`` keep reading ``""`` for a cell that
        now holds real data (regression: retyped path showed the ghost).
        An empty commit also dismisses the overlay so the cell's own ghost
        is visible at once — the field lingering under the pointer made the
        row look blank (user had to click away to see the placeholder).
        """
        if (iid := self._field_iid) is None:
            return

        if (r := self._internal_row(iid)) is not None:
            with suppress(TclError):
                self.sh.set_cell_data(r, 0, text)
            if text:
                if (ph := getattr(self, "_ph", None)) is not None:
                    ph.untrack(self.sh, r, 0)
            else:
                self._restore_placeholder(iid, r, 0)

        self._apply_edit_value(iid, 0, text)

        # Content changed under the pointer (edit commit / browse) —
        # re-sync the overlay's content-aware align (ghost left, path right).
        if (f := self._hover_field) is not None and f.winfo_ismapped():
            f.update_idletasks()
            self._sync_floated_align()

        if not text:
            # The cell beneath now shows the ghost — hide the identical
            # overlay instead of leaving a second, covering surface.
            # (Hover dwell re-shows it if the pointer lingers on the row.)
            self._hide_hover_field()

        m = self._meta.get(iid, {})
        if m.get("is_metadata_root"):
            self._metadata_path = text
            if Path(text).expanduser().is_file():
                self.sh.after_idle(lambda p=text: self._reload_metadata_from(p))
        elif (m.get("key") == "path" or m.get("path") == "input.coefs") and self._mgr is not None and text:
            self.sh.after_idle(lambda: self._mgr.notify_path_changed(text))
        # Re-validate the written cell after browse — red fg if its check fails.
        if m.get("check"):
            self.sh.after_idle(lambda: self._apply_validations(iid))

    def _hover_read(self) -> str:
        """Read column 0 of the hovered row (for dialog initialdir). Ghost reads as ``""``."""
        if (iid := self._field_iid) is None:
            return ""
        if (r := self._internal_row(iid)) is not None:
            ph = getattr(self, "_ph", None)
            if ph is not None and ph.has(r, 0):
                return ""
            with suppress(TclError, IndexError):
                return self.sh.get_cell_data(r, 0) or ""
        return ""

    # ── floated PathField lifecycle ───────────────────────────────────

    def _sync_floated_align(self) -> None:
        """Content-aware align for the floated overlay field only.

        Real paths right-align + scroll to the filename end (it sits next
        to the browse button — long directory prefixes scroll away);
        ghosts stay left-aligned like the cell underneath.

        Overlay-only by construction: it touches the floated field's own
        1×1 sheet, never the standalone top field.  ``table_align`` resets
        any per-cell override; the follow-up scroll needs finalized geometry,
        so callers run it AFTER ``place`` + ``update_idletasks``.
        """
        f = self._hover_field
        if f is None:
            return
        if f._ph.has(0, 0):
            f.sh.table_align("w", redraw=False)
            f.sh.redraw()
            f._scroll_to_left()
        else:
            f.sh.table_align("e", redraw=False)
            f.sh.redraw()
            f._scroll_to_right()

    def _ensure_hover_field(self) -> _path_field.PathField:
        """The single PathField instance for the text surface + a separate
        ``BrowseOverlay`` button at the sheet's right edge.  Both are
        created lazily on first browse hover.  Focus is opt-in."""
        if self._hover_field is None:
            f = _path_field.PathField(
                self.sh,
                align="w",  # floated field: left-aligned like the cell it covers
                on_commit=self._hover_write,
                on_begin_edit=self._on_field_edit_start,
                on_end_edit=self._on_field_edit_end,
                dir_title="",  # files-only — no directory selection for per-probe paths
                files_title=_S["dialog.data_files"],
                filetypes=DATA_FILETYPES,
                shift_swap=False,  # no Shift placeholder swap for per-probe fields
                placeholder=_S.get("path_field.placeholder_probe", ""),  # per-probe data file hint
            )
            # Neuter PathField's own browse overlay — we use a separate one.
            # Replace overlay + binder so SheetHoverBinder never creates a
            # second button.
            f._ov.hide()
            f._ov = self._NULL_OV
            f._binder._ov = self._NULL_OV
            f._binder._last = None
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                f.sh.MT.bind(ev, lambda _e: self._hide_hover_field(), add="+")
            # PathField <Leave>: pointer moved to sheet MT or button on the
            # same row — re-publish row status instead of leaving it empty.
            # True sheet leave is handled by _on_sheet_leave.
            f.sh.MT.bind("<Leave>", self._on_field_leave, add="+")
            # Overlay-only: after PathField's own <Configure> handling, re-sync
            # the content-aware align — the HEAD handler scrolls LEFT for
            # align="w" fields, which pushes a right-aligned path off-screen
            # on any resize while the overlay is mapped.
            f.unbind("<Configure>")

            def _floated_configure(event, _f=f) -> None:
                _f._on_configure(event)
                if not _f._editing:
                    self._sync_floated_align()

            f.bind("<Configure>", _floated_configure, add="+")
            self._hover_field = f
            # Status callback for browse button Shift hint — wraps
            # on_hover_status(msg, md) into the on_status(text) signature.
            # Button <Leave> re-publishes row status instead of clearing:
            # the pointer typically moves to the PathField or MT area on the
            # same row (both children of MT → no MT <Motion> fires),
            # so clearing would leave the status empty until the next
            # motion event.  True sheet leave is handled by _on_sheet_leave.
            _on_status = self._hover_btn_status if self.on_hover_status is not None else None
            self._hover_btn = BrowseOverlay(
                self.sh,
                self._hover_write,
                self._hover_read,
                dir_title="",  # files-only — no directory selection for per-probe paths
                files_title=_S["dialog.data_files"],
                filetypes=DATA_FILETYPES,
                on_status=_on_status,
                status_hint=self._status_hint,
            )
        return self._hover_field

    def _show_hover_field(self, iid: Any, hit_row: int, fallback_y: int) -> None:
        f = self._ensure_hover_field()
        if f._editing:
            return  # safety: don't reposition while editing — Entry stays until Enter/Esc
        f.cancel_edit()  # stale editor from the previous row → Esc
        self._field_iid = iid
        self._field_row = hit_row  # stored for expand-on-edit
        self._field_y = fallback_y
        # Reconfigure browse button for the current row type.
        # input.coefs (path) → dir+file (coefs), metadata root → file-only *.yaml, others → data files.
        is_coefs = self._meta.get(iid, {}).get("key") in ("path", "coefs")
        is_meta_path = bool(self._meta.get(iid, {}).get("is_metadata_root"))
        if self._hover_btn is not None:
            if is_coefs:
                self._hover_btn._dir_title = _S["dialog.coefs_dir"]
                self._hover_btn._files_title = _S["dialog.coefs_files"]
                self._hover_btn._filetypes = COEF_FILETYPES
                self._hover_btn._files_only = False
                self._hover_btn._save_mode = False
                self._hover_btn._save_ext = ""
                self._hover_btn._status_hint = _S["browse_btn.status"]
                self._hover_btn._status_hint_files = _S["browse_btn.status_files"]
            elif is_meta_path:
                self._hover_btn._dir_title = ""
                self._hover_btn._files_title = _S.get("dialog.metadata_file", _S["dialog.coefs_files"])
                self._hover_btn._filetypes = [("YAML", "*.yaml"), (_S["dialog.filter_all"], "*.*")]
                self._hover_btn._files_only = True
                self._hover_btn._save_mode = True
                self._hover_btn._save_ext = ".yaml"
                self._hover_btn._status_hint = _S.get("metadata.status", self._status_hint)
                self._hover_btn._status_hint_files = ""
            else:
                self._hover_btn._dir_title = ""
                self._hover_btn._files_title = _S["dialog.data_files"]
                self._hover_btn._filetypes = DATA_FILETYPES
                self._hover_btn._files_only = True
                self._hover_btn._save_mode = False
                self._hover_btn._save_ext = ""
                self._hover_btn._status_hint = self._status_hint
                self._hover_btn._status_hint_files = ""
        # Ghost text mirrors the cell underneath (single source of truth:
        # ``_placeholder_for``) — the overlay must never show a different
        # hint than the covered cell.  Rows without a ghost keep the
        # field's construction-time hint.
        f.set_placeholder(self._placeholder_for(iid, 0) or "")
        val = self._hover_read()
        f.set(val)
        f.place(**self._field_place_kw(hit_row, fallback_y, val))
        f.lift()
        f.update_idletasks()  # finalize geometry before the content-aware scroll
        self._sync_floated_align()
        if self._hover_btn is not None:
            self._hover_btn.show(**self._btn_place_kw(hit_row, fallback_y))
        # Publish status so the help text is visible even when the pointer
        # went straight to the overlay without lingering on the tksheet cell.
        self._status_iid = iid
        self._status_source = "data"
        self._publish_status(iid)

    def _field_place_kw(self, hit_row: int, fallback_y: int, val: str) -> dict[str, Any]:
        """Text surface: from col 0, top-aligned, row-matching height.

        Width ends where the browse button starts.  Height is read from
        ``MT.row_positions`` so the field matches the sheet's actual rows
        regardless of index-label chrome.
        """
        mt = self.sh.MT
        with suppress(AttributeError, TypeError, IndexError, TclError):
            y1, y2 = mt.row_positions[hit_row], mt.row_positions[hit_row + 1]
            x0 = self._col0_widget_x() or 0
            btn_w = self._hover_btn_w()
            return {
                "in_": mt,
                "x": x0,
                "anchor": "nw",
                "y": y1 - mt.canvasy(0),
                "width": max(mt.winfo_width() - x0 - btn_w, 50),
                "height": y2 - y1,
            }
        return {"in_": mt, "x": 0, "y": fallback_y, "anchor": "nw", "width": 320}

    def _field_full_width_kw(self) -> dict[str, Any]:
        """Full row width (no button subtraction) — used when editing starts."""
        mt = self.sh.MT
        with suppress(AttributeError, TypeError, IndexError, TclError):
            y1, y2 = mt.row_positions[self._field_row], mt.row_positions[self._field_row + 1]
            x0 = self._col0_widget_x() or 0
            return {
                "in_": mt,
                "x": x0,
                "anchor": "nw",
                "y": y1 - mt.canvasy(0),
                "width": mt.winfo_width() - x0,
                "height": y2 - y1,
            }
        return {"in_": mt, "x": 0, "y": self._field_y, "anchor": "nw", "width": 320}

    def _btn_place_kw(self, hit_row: int, fallback_y: int) -> dict[str, Any]:
        """Browse button: right edge of the visible row, top-aligned with text field."""
        mt = self.sh.MT
        with suppress(AttributeError, TypeError, IndexError, TclError):
            y1 = mt.row_positions[hit_row]
            return {"in_": mt, "x": mt.winfo_width(), "y": y1 - mt.canvasy(0), "anchor": "ne"}
        return {"in_": mt, "x": mt.winfo_width(), "y": fallback_y, "anchor": "ne"}

    def _hover_btn_w(self) -> int:
        """Pixel width of the hover browse button.

        Uses the actual rendered width (``winfo_width``) when the button
        exists and has been placed; falls back to ``winfo_reqwidth`` of a
        temporary button otherwise.
        """
        if self._hover_btn is not None and self._hover_btn._button is not None:
            with suppress(TclError):
                btn = self._hover_btn._button
                btn.update_idletasks()
                w = btn.winfo_width()
                if w > 1:
                    return w
        if not hasattr(self, "_btn_w_cache"):
            from ._browse_button import browse_button_width

            self._btn_w_cache = browse_button_width(self.sh)
        return self._btn_w_cache

    def _on_field_leave(self, _event) -> None:
        """PathField ``<Leave>``: re-publish row status.

        The pointer moved from the edit overlay to the sheet MT or browse
        button on the same row.  Child-to-parent transition may not trigger
        a sheet MT ``<Motion>``/``<Enter>``, so the ``iid == _status_iid``
        guard in ``_on_sheet_motion`` would skip re-publish — leaving the
        status empty if ``_on_sheet_leave`` cleared it at the MT→field
        boundary.  True sheet leave is handled by :meth:`_on_sheet_leave`.
        """
        if self._status_iid is not None and self.on_hover_status is not None:
            self._publish_status(self._status_iid)

    def _hover_btn_status(self, text: str) -> None:
        """Hover button status callback.

        Non-empty *text* (button enter / Shift toggle) → publish directly.
        Empty *text* (button leave) → re-publish the current row's status
        instead of clearing: the pointer moved to the PathField or MT area
        on the same row (both children of MT, so no MT ``<Motion>`` fires).
        True sheet leave is handled by :meth:`_on_sheet_leave`.
        """
        if text:
            self.on_hover_status(text, True)
        elif self._status_iid is not None:
            self._publish_status(self._status_iid)
        else:
            self.on_hover_status("", True)

    def _on_field_edit_start(self) -> None:
        """User clicked the overlay field to edit — freeze status, hide button, expand field.

        Fires ``on_edit_begin`` (App freezes hover status + hides the dwell
        tip — full parity with cell editing) and cancels any pending hide
        (armed by ``<Leave>`` when the pointer stepped onto the Entry), then
        forces geometry so ``PathField`` reads the correct ``winfo_width()``
        for its column constraint.
        """
        self._cancel_field_hide_job()
        if (cb := self.on_edit_begin) is not None:
            cb()
        if self._hover_btn is not None:
            self._hover_btn.hide()
        if (f := self._hover_field) is not None and f.winfo_ismapped():
            f.place(**self._field_full_width_kw())
            f.update_idletasks()

    def _on_field_edit_end(self) -> None:
        """Entry edit finished (commit / Esc / click-away) — unfreeze status, restore width.

        Fires ``on_edit_end`` (App releases the edit-time status freeze —
        also reached via ``cancel_edit()`` from ``_hide_hover_field``),
        then restores hover width and re-shows the button.  Guarded like the
        ``_sheet_tint`` close funnel: end fires inside teardown paths where a
        callback error must not break the hide.
        """
        if (cb := self.on_edit_end) is not None:
            try:
                cb()
            except Exception:
                _l.exception("on_edit_end callback failed")
        self._restore_hover_placement()

    def _restore_hover_placement(self) -> None:
        if (f := self._hover_field) is not None and f.winfo_ismapped() and not f._editing:
            val = self._hover_read()
            f.place(**self._field_place_kw(self._field_row, self._field_y, val))
            f.update_idletasks()
            if self._hover_btn is not None:
                self._hover_btn.show(**self._btn_place_kw(self._field_row, self._field_y))

    def _schedule_field_show(self, iid: Any, hit_row: int, fallback_y: int) -> None:
        self._cancel_field_hide_job()
        if self._field_show_job is not None:
            self.sh.after_cancel(self._field_show_job)
        self._field_pending = (iid, hit_row, fallback_y)
        self._field_show_job = self.sh.after(_INTENT_MS, self._do_field_show)

    def _do_field_show(self) -> None:
        self._field_show_job = None
        if self._field_pending is not None:
            self._show_hover_field(*self._field_pending)

    def _schedule_field_hide(self) -> None:
        """Delayed hide — vetoed when the pointer has moved onto the field
        itself (MT fires ``<Leave>`` at exactly that crossing)."""
        if self._field_show_job is not None:
            self.sh.after_cancel(self._field_show_job)
            self._field_show_job = self._field_pending = None
        field_mapped = self._hover_field is not None and self._hover_field.winfo_ismapped()
        btn_visible = self._hover_btn is not None and self._hover_btn.visible
        if self._field_hide_job is None and (field_mapped or btn_visible):
            self._field_hide_job = self.sh.after(_INTENT_MS, self._do_field_hide)

    def _do_field_hide(self) -> None:
        self._field_hide_job = None
        if (f := self._hover_field) is not None and f._editing:
            return  # don't hide while editing — Entry fills the PathField
        if not self._pointer_in_field():
            self._hide_hover_field()

    def _hide_hover_field(self) -> None:
        """Immediate teardown: cancel jobs, cancel in-flight edit, unmap field + button.

        Deliberately keeps ``_field_iid`` — PathField commits via
        ``after_idle``, so a commit already queued must still land on its row.
        An open Entry must be cancelled (``_editing`` reset) or every
        ``_editing``-guarded path stays wedged — the overlay never reappears
        until a new scan rebuilds the sheet.  Unmap runs BEFORE cancel so
        ``_on_field_edit_end`` → ``_restore_hover_placement`` sees an unmapped
        field and skips the place/show round-trip just undone here.
        """
        for attr in ("_field_show_job", "_field_hide_job"):
            if (job := getattr(self, attr)) is not None:
                self.sh.after_cancel(job)
                setattr(self, attr, None)
        self._field_pending = None
        if (f := self._hover_field) is not None:
            if f.winfo_ismapped():
                f.place_forget()
            if f._editing:
                f.cancel_edit()
        if self._hover_btn is not None:
            self._hover_btn.hide()

    def _cancel_field_hide_job(self) -> None:
        job, self._field_hide_job = self._field_hide_job, None
        if job is not None:
            self.sh.after_cancel(job)

    def _pointer_in_field(self) -> bool:
        """True when the pointer is inside the floated PathField or its browse button."""
        f = self._hover_field
        if f is not None and f.winfo_ismapped() and _pointer_inside(f):
            return True
        btn = self._hover_btn
        return btn is not None and btn.visible and btn._button is not None and _pointer_inside(btn._button)

    def _on_sheet_motion(self, event) -> None:
        """Hover: status text for any visible row; floated field on browse rows."""
        if (hit := self._hover_resolve(event)) is None:
            self._schedule_field_hide()
            if self._status_iid is not None:
                self._clear_status()
            if self._empty_area_hint and self.on_hover_status is not None:
                self.on_hover_status(self._empty_area_hint, True)
            return

        iid, row, y = hit
        col = self._raw_col(event)

        # Re-publish when row, source (tree ↔ data), or (for metadata paired
        # rows) the hovered column changes — each column documents a different
        # field, so col 0 → metadata.point but col 1 → metadata.symbol.
        # Apply-checkbox cells also force re-publish: stepping onto/off the box
        # (same row, action col outside max_col) swaps data hint ↔ apply status.
        m = self._meta.get(iid, {})
        _is_apply = False
        with suppress(Exception):
            _is_apply = bool(getattr(self, "_is_apply_cell", lambda *_a: False)(iid, col))
        _was_apply = False
        with suppress(Exception):
            _was_apply = bool(getattr(self, "_is_apply_cell", lambda *_a: False)(iid, self._status_col))
        if (
            iid != self._status_iid
            or self._status_source != "data"
            or (m.get("is_metadata") and m.get("max_col", 1) > 1 and col != self._status_col)
            or _is_apply
            or _was_apply
        ):
            self._status_iid = iid
            self._status_source = "data"
            self._status_col = col
            self._publish_status(iid, col)

        # Readonly mode: no hover overlays (editing is blocked anyway).
        if self._readonly:
            self._schedule_field_hide()
            return

        if not self._meta.get(iid, {}).get("browse"):
            self._schedule_field_hide()
            return

        f = self._hover_field
        if iid == self._field_iid and (
            (f is not None and f.winfo_ismapped()) or self._field_show_job is not None
        ):
            self._cancel_field_hide_job()  # motion over target vetoes pending hide
            # During editing the field is at full width — don't shrink it back.
            if f is not None and f.winfo_ismapped() and not f._editing:
                val = self._hover_read()
                if f.get() != val:
                    f.set(val)  # cell may have changed underneath (editor commit, deletion)
                f.place(**self._field_place_kw(row, y, val))
                if self._hover_btn is not None:
                    self._hover_btn.show(**self._btn_place_kw(row, y))
            return

        # While editing, don't reposition or replace the field — it must stay
        # until the user commits (Enter) or cancels (Esc).
        if f is not None and f._editing:
            return

        self._schedule_field_show(iid, row, y)
