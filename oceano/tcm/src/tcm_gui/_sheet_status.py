"""Sheet hover: status-bar publication and the floated PathField overlay.

Mixin for :class:`tcm_gui.coef_sheet.ConfigSheet`.

* **Status bar** (:meth:`SheetHoverMixin._publish_status`) — hovered row's doc
  short text from ``config_reference.md`` (:mod:`tcm_gui._help`), with a live
  ``time_ranges`` sync detail appended (recomputed per publish — never a stale
  scan-time message) and Shift-toggled dir/file hints for ``coefs_path``.
* **Floated PathField** (:meth:`SheetHoverMixin._show_hover_field`) — after
  ``_INTENT_MS`` dwell on a browse row, a reusable :class:`tcm_gui._path_field.PathField`
  covers the row plus a :class:`tcm_gui._browse_button.BrowseOverlay` at the
  right edge.  Empty commits propagate to :meth:`SheetHoverMixin._hover_write`
  which writes ``""`` and restores the ghost — deletion is a first-class value.
"""

from __future__ import annotations

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
from ._i18n import STRINGS as _S

_INTENT_MS: Final[int] = 120  # hover-intent delay for the floated PathField (ms)


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

    def _help_candidates(self, iid: Any, *, tree: bool = False) -> list[str]:
        """Ordered ``help_for_path`` candidates for *iid*.

        Single source for ``_on_tree_motion`` and ``_publish_status`` — paired
        metadata rows (``point, symbol``) try both ``metadata.point`` and
        ``metadata.symbol``, the ``metadata`` root tries ``metadata.path``,
        otherwise the node's own ``path``.  Tree motion shows the first field
        only; data-cell hover fans both.
        """
        m = self._meta.get(iid, {})
        path = str(m.get("path") or "")
        if m.get("is_metadata_root"):
            return ["metadata.path"]
        if m.get("is_metadata") and (lbl := m.get("label")):
            parts = [s.strip().replace(" ", "_").replace("/", "_") for s in lbl.split(",")]
            cands = [f"metadata.{parts[0]}"]
            if not tree and len(parts) > 1:
                cands.append(f"metadata.{parts[1]}")
            if path not in cands:
                cands.append(path)
            return cands
        cands: list[str] = []
        if not tree and m.get("has_date"):
            cands += [f"{path}.date", f"{path}.dates"]
            if (par := m.get("parent")) and (pp := self._meta.get(par, {}).get("path")):
                cands += [f"{pp}.dates", f"{pp}.date"]
        cands += [f"{path}.path", path] if path else []
        return cands

    def _on_tree_motion(self, event) -> None:
        """Hover over tree column (index canvas) — show help for that node."""
        if (hit := self._hover_resolve(event)) is None:
            self._clear_status()
            return

        iid, _row, _y = hit
        if iid == self._status_iid and self._status_source == "tree":
            return

        self._status_iid = iid
        self._status_source = "tree"
        for cand in self._help_candidates(iid, tree=True):
            if cand and (h := _help.help_for_path(cand)) and h.short:
                self._hover_detail = self._resolve_detail(cand)
                if self.on_hover_status is not None:
                    self.on_hover_status(h.short, True)
                return
        m = self._meta.get(iid, {})
        if self.on_hover_status is not None:
            self._hover_detail = ""
            self.on_hover_status(str(m.get("key") or m.get("label") or m.get("path") or ""), False)

    def _coefs_status_hint(self) -> str:
        """Mode-aware status hint for ``coefs_path`` browse button.

        Shift held → ``file`` mode; default → ``dir`` mode.
        Content from ``config_reference.md`` ``<mode>`` sections.
        """
        mode = "file" if _is_shift_pressed() else "dir"
        if (h := _help.help_for_path("input.coefs_path", mode=mode)) and h.body:
            return str(h.body)
        return _S["browse_btn.status_files" if mode == "file" else "browse_btn.status"]

    def _on_shift_toggle(self, _event) -> None:
        """Re-publish status when Shift is pressed/released while hovering coefs_path.

        Gate: ``_status_iid`` is only set while the pointer is actively on a
        row (cleared by ``_clear_status`` on leave/blank-area) — unlike
        ``_field_iid`` which survives hide for deferred ``after_idle`` commits.
        The ``_pointer_in_field`` branch covers the case where the pointer
        stepped onto the browse button or floated PathField.
        """
        iid = self._status_iid
        if iid is None:
            return
        if self._meta.get(iid, {}).get("key") != "coefs_path":
            return
        # Pointer on the hover button → its poll (_update_icon) re-publishes
        # the button-specific hint on this transition; publishing the row
        # text here would overwrite it.
        if self._hover_btn is not None and self._hover_btn._hovered:
            return
        if not (self._pointer_in_field() or _pointer_inside(self.sh.MT)):
            return
        self._publish_status(iid)

    def _on_f1_help(self, _event=None) -> None:
        """F1 over a sheet row — open the doc browser at its ``config_reference`` heading.

        Uses the hovered row (``_status_iid``) so no pointer event is needed;
        ``help_for_path`` strips array indices and returns the section anchor
        (GitHub-style slug, mirrors ``browser/web/viewer.js::slugify``).  The
        doc MUST be the same localized file the entries were parsed from
        (``doc_path(resolve_lang())`` — exactly what ``_load`` reads); plain
        ``doc_path()`` always serves the English file and a localized anchor
        then finds no element — the page opens but never scrolls.
        """
        if (iid := self._status_iid) is None:
            return
        if not (path := str(self._meta.get(iid, {}).get("path") or "")):
            return
        anchor = entry.anchor if (entry := _help.help_for_path(path)) else ""
        from tcm_gui.browser import get_documentation_browser

        get_documentation_browser().open(_help.doc_path(_help.resolve_lang()), anchor=anchor or None)

    @staticmethod
    def _resolve_detail(path: str) -> str:
        """Resolve the dwell tooltip text — ``#### Detailed`` blocks only.

        Scans every ``###`` section of the field (mode-tagged or modeless)
        plus the field-level block; a tooltip exists ⟺ some section carries a
        ``Detailed`` block.  Section short bodies and group prose never arm
        the dwell — regression: every coef row showed the ``input.coefs``
        group text instead of nothing.  Returns ``""`` — the caller skips arming.
        """
        if (e := _help.help_for_path(path)) and isinstance(e.body, Mapping):
            for val in e.body.values():
                if isinstance(val, _help.ModeBody) and (d := val.details.get("Detailed")):
                    return str(d)
        return ""

    def _publish_status(self, iid: Any) -> None:
        """Status text for the hovered element (data cells on MT canvas).

        Fallback chain: ``help_for_path(path).short`` (from
        ``config_reference.md``) → ``key`` → ``label`` → ``path``.  The
        ``time_ranges`` row appends the *live* sync detail from
        :meth:`_time_ranges_detail`; ``coefs_path`` shows Shift-toggled
        dir/file content.  Tree-column hover is handled by ``_on_tree_motion``
        which always uses the section-level path.
        """
        if self.on_hover_status is None:
            return

        if iid is None:
            self._hover_detail = ""
            self.on_hover_status("", False)
            return

        m = self._meta.get(iid, {})
        ident = str(m.get("key") or m.get("path") or m.get("label") or "")

        # time_ranges: doc short + live sync detail — recomputed, never cached
        if m.get("label") == "time_ranges" or str(m.get("path") or "") == "input.time_ranges":
            sync = self._time_ranges_detail()
            for cand in self._help_candidates(iid, tree=False):
                if cand and (h := _help.help_for_path(cand)) and h.short:
                    txt = f"{h.short} — {sync}" if sync else h.short
                    self._hover_detail = self._resolve_detail(cand) or self._resolve_detail(
                        str(m.get("path") or "")
                    )
                    self.on_hover_status(txt, True)
                    return
            self._hover_detail = ""
            self.on_hover_status(sync, False)
            return

        # Mode-aware: coefs_path shows dir/file content from config_reference.md
        # instead of the table-row short text.  Shift toggles mode.
        if ident == "coefs_path" and (txt := self._coefs_status_hint()):
            self._hover_detail = self._resolve_detail("input.coefs_path")
            self.on_hover_status(txt, True)
            return

        for cand in self._help_candidates(iid, tree=False):
            if cand and (h := _help.help_for_path(cand)) and h.short:
                self._hover_detail = self._resolve_detail(cand) or self._resolve_detail(
                    str(m.get("path") or "")
                )
                self.on_hover_status(h.short, True)
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
        elif m.get("key") == "coefs_path" and self._mgr is not None and text:
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
        # coefs_path → dir+file (coefs), metadata root → file-only *.yaml, others → data files.
        is_coefs = self._meta.get(iid, {}).get("key") == "coefs_path"
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
        """User clicked the overlay field to edit — hide button, expand field.

        Cancels any pending hide (armed by ``<Leave>`` when the pointer
        stepped onto the Entry) and forces geometry so ``PathField``
        reads the correct ``winfo_width()`` for its column constraint.
        """
        self._cancel_field_hide_job()
        if self._hover_btn is not None:
            self._hover_btn.hide()
        if (f := self._hover_field) is not None and f.winfo_ismapped():
            f.place(**self._field_full_width_kw())
            f.update_idletasks()

    def _on_field_edit_end(self) -> None:
        """Entry edit finished — restore hover width, re-show button."""
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

        # Re-publish when row OR source (tree ↔ data) changes.
        if iid != self._status_iid or self._status_source != "data":
            self._status_iid = iid
            self._status_source = "data"
            self._publish_status(iid)

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
