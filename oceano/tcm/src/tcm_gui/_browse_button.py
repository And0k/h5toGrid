"""Floating browse button — one widget core, three trigger policies.

BrowseOverlay          host-agnostic lifecycle: create/place/destroy,
                       Shift-aware icon, dialog, hover-intent scheduling.
BrowseButtonManager    sheet-edit policy: retry-poll for tksheet's
                       TextEditor; attach/detach around cell editing.
bind_hover_browse      hover policy for plain Entry fields off the sheet.

tksheet reuses a single TextEditor across edits — identity can never
discriminate editors.  Correctness rests on the sheet-side protocol:
unconditional ``detach()`` at every ``begin_edit_cell`` cancels stale
retries before any foreign editor exists.

Teardown (sheet policies): the button is a child of the sheet placed
``in_=`` a foreign anchor; when tksheet destroys the anchor the button
survives at stale canvas coordinates — ``destroy()`` is the only safe
teardown, ``place_forget()`` cannot reliably unmap it.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Mapping
from contextlib import suppress
from tkinter import TclError, filedialog, ttk
from typing import Any

from tksheet import Sheet

COEF_FILETYPES = [("Coefs", "*.h5 *.nc *.yaml *.yml"), ("All", "*.*")]

_LBL_DIR = "…📁"  # no-Shift default: browse directory
_LBL_FILE = "…📄"  # Shift held: browse files
_POLL_MS = 80  # Shift-state icon polling interval
_INTENT_MS = 120  # hover-intent delay (show and hide)
_RETRY_MS = 50  # editor acquisition retry interval
_MAX_RETRIES = 20  # × _RETRY_MS ≈ 1 s budget for editor creation


def _is_shift_pressed() -> bool:
    """Return True when either Shift key is currently held."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
    except (AttributeError, OSError):
        return False


def _pointer_inside(w) -> bool:
    """True iff the OS pointer is currently within *w*'s bounds."""
    try:
        x, y = w.winfo_pointerxy()
        return 0 <= x - w.winfo_rootx() < w.winfo_width() and 0 <= y - w.winfo_rooty() < w.winfo_height()
    except TclError:
        return False


def browse_button_height(parent) -> int:
    """Natural height of the browse button — host fields size to this."""
    b = ttk.Button(parent, text=_LBL_DIR, width=4)
    try:
        return b.winfo_reqheight()
    finally:
        b.destroy()


class BrowseOverlay:
    """Host-agnostic floating browse button.

    *host* parents the button and lends ``after()``; *write*/*read*
    decouple the text sink (Text editor, Entry, sheet cell) from widget
    code.  Placement is caller-supplied — right-edge convention:
    ``in_=anchor, relx=1.0`` (button extends right of the anchor).
    ``leave_hides`` arms hover-policy teardown: button ``<Leave>`` then
    schedules a hide that the pointer-inside check vetoes.
    """

    def __init__(
        self,
        host,
        write: Callable[[str], None],
        read: Callable[[], str] | None = None,
        *,
        filetypes=COEF_FILETYPES,
        dir_title: str = "Coefficients directory",
        files_title: str = "Coefficient files",
        leave_hides: bool = False,
    ) -> None:
        self._host = host
        self._write, self._read = write, read
        self._filetypes, self._dir_title, self._files_title = filetypes, dir_title, files_title
        self._leave_hides = leave_hides
        self._button: ttk.Button | None = None
        self._icon_job: str | None = None
        self._show_job: str | None = None
        self._hide_job: str | None = None
        self._pending_place: dict[str, Any] = {}

    @property
    def visible(self) -> bool:
        return self._button is not None

    @property
    def pending(self) -> bool:
        """A show is scheduled but not yet realized."""
        return self._show_job is not None

    # ── show / hide ───────────────────────────────────────────────
    def show(self, **place_kw: Any) -> None:
        """Idempotent: create on first call, re-place on every call."""
        self.cancel_show()
        self.cancel_hide()
        if self._button is None:
            self._button = self._make_button()
        self._button.place(**place_kw)
        self._button.lift()
        self._start_icon_polling()

    def hide(self) -> None:
        """Destroy the button; cancel every pending job."""
        self.cancel_show()
        self.cancel_hide()
        job, self._icon_job = self._icon_job, None
        if job is not None:
            with suppress(TclError):
                self._host.after_cancel(job)
        btn, self._button = self._button, None
        if btn is not None:
            with suppress(TclError):
                btn.destroy()

    # ── hover-intent scheduling ───────────────────────────────────
    def schedule_show(self, place_kw: Mapping[str, Any], delay_ms: int = _INTENT_MS) -> None:
        """Delayed show — prevents flashing during pointer sweeps."""
        if self.visible:
            self.show(**place_kw)  # already committed: re-anchor at once
            return
        self.cancel_show()
        self._pending_place = dict(place_kw)
        self._show_job = self._host.after(delay_ms, self._do_show)

    def schedule_hide(self, delay_ms: int = _INTENT_MS) -> None:
        """Delayed hide — vetoed if the pointer still sits on the button."""
        self.cancel_hide()
        self._hide_job = self._host.after(delay_ms, self._hide_if_outside)

    def cancel_show(self) -> None:
        job, self._show_job = self._show_job, None
        if job is not None:
            with suppress(TclError):
                self._host.after_cancel(job)

    def cancel_hide(self) -> None:
        job, self._hide_job = self._hide_job, None
        if job is not None:
            with suppress(TclError):
                self._host.after_cancel(job)

    def _do_show(self) -> None:
        self._show_job = None
        self.show(**self._pending_place)

    def _hide_if_outside(self) -> None:
        self._hide_job = None
        if self._button is not None and not _pointer_inside(self._button):
            self.hide()

    # ── widget core ───────────────────────────────────────────────
    def _make_button(self) -> ttk.Button:
        btn = ttk.Button(self._host, text=_LBL_DIR, width=4, command=self._browse)
        btn.focus_set = lambda: None  # type: ignore[method-assign]
        btn.configure(takefocus=False)
        btn.bind("<Button-1>", lambda _e: (self._browse(), "break")[-1], add="+")
        if self._leave_hides:
            btn.bind("<Leave>", lambda _e: self.schedule_hide(), add="+")
        return btn

    def _start_icon_polling(self) -> None:
        job, self._icon_job = self._icon_job, None
        if job is not None:
            with suppress(TclError):
                self._host.after_cancel(job)
        self._update_icon()

    def _update_icon(self) -> None:
        if self._button is None:
            return
        self._button.configure(text=_LBL_FILE if _is_shift_pressed() else _LBL_DIR)
        self._icon_job = self._host.after(_POLL_MS, self._update_icon)

    # ── dialog ────────────────────────────────────────────────────
    def _browse(self) -> None:
        cur = (self._read() if self._read is not None else "") or ""
        start = os.path.dirname(cur.split(",")[0].strip()) or None
        try:
            if _is_shift_pressed():
                paths = filedialog.askopenfilenames(
                    title=self._files_title, filetypes=self._filetypes, initialdir=start
                )
                result = ",".join(paths) if paths else ""
            else:
                result = filedialog.askdirectory(title=self._dir_title, initialdir=start)
        except TclError:
            result = ""
        if result:
            self._write(result)


class BrowseButtonManager:
    """Sheet-edit trigger policy: button lives while the TextEditor is open.

    ``attach`` opens with ``detach()`` — idempotent reset that also
    destroys any orphaned button from a missed teardown.  The caller
    (``ConfigSheet._on_begin_edit_cell``) runs ``detach()`` on **every**
    edit onset, browse row or not — that pre-emption, not editor
    identity, is what keeps the button off foreign editors.

    Write target is always **column 0** (the path cell) regardless of
    which column the user clicked.  ``close_text_editor`` fires
    ``end_edit_cell`` for the original column.

    An optional *on_edit_restyler* callback ``(iid, col, value)`` lets
    the caller run ``_apply_edit_value`` for gray/blue restyling that
    ``set_cell_data`` bypasses.
    """

    def __init__(
        self,
        sheet: Sheet,
        on_path_changed: Callable[[str], None],
        on_edit_restyler: Callable[[Any, int, str], None] | None = None,
        editor_place: Callable[[Any], dict[str, Any]] | None = None,
    ) -> None:
        self._sheet = sheet
        self.notify_path_changed = on_path_changed
        self._on_edit_restyler = on_edit_restyler
        self._editor: Any = None
        self._target: tuple[int, int] | None = None
        self._target_iid: Any = None
        self._retries = 0
        self._retry_job: str | None = None
        self._ov = BrowseOverlay(sheet, self._write_cell, self._read_cell)
        self._editor_place = editor_place or (lambda ed: {
            "in_": ed, "relx": 1.0, "x": 0, "rely": 0, "y": 0, "height": ed.winfo_height()})

    def attach(self, row: int, col: int, iid: Any = None) -> None:
        self.detach()
        self._target = (row, col)
        self._target_iid = iid
        self._retries = 0
        self._retry_job = self._sheet.after(_RETRY_MS, self._acquire_and_place)

    def detach(self) -> None:
        job, self._retry_job = self._retry_job, None
        if job is not None:
            self._sheet.after_cancel(job)
        self._ov.hide()
        self._editor = None
        self._target_iid = None

    def _acquire_and_place(self) -> None:
        self._retry_job = None
        if (ed := self._sheet.get_text_editor_widget()) is None:
            self._retries += 1
            if self._retries < _MAX_RETRIES:
                self._retry_job = self._sheet.after(_RETRY_MS, self._acquire_and_place)
            return
        self._editor = ed
        self._ov.show(**self._editor_place(ed))

    def _read_cell(self) -> str:
        """Read column 0 of the target row (for dialog initialdir)."""
        if self._target is None:
            return ""
        row, _col = self._target
        try:
            return self._sheet.get_cell_data(row, 0) or ""
        except (TclError, IndexError):
            return ""

    def _write_cell(self, text: str) -> None:
        """Write to column 0, restyle via callback, close editor.

        ``set_cell_data`` bypasses the edit pipeline (no validation,
        no ``end_edit_cell``), so we call the restyler directly and
        close the editor to commit.
        """
        if self._target is None:
            return
        row, _col = self._target
        with suppress(TclError):
            self._sheet.set_cell_data(row, 0, text)
        if self._on_edit_restyler is not None and self._target_iid is not None:
            self._on_edit_restyler(self._target_iid, 0, text)
        with suppress(TclError):
            self._sheet.after_idle(self._sheet.close_text_editor)


class SheetHoverBinder:
    """Motion policy on a sheet's MT canvas → overlay show/hide.

    *resolve(event)* → ``place_kw`` to show, ``None`` to hide.  The
    mechanics live here once — intent scheduling, scroll/leave
    teardown, same-target churn veto; gating (browse meta, status
    publishing) is injected with the resolver.  Binds use ``add="+"``:
    never replace tksheet's own MT handlers (a replacing ``<MouseWheel>``
    bind kills scrolling).
    """

    def __init__(
        self,
        sheet,
        overlay: BrowseOverlay,
        resolve: Callable[[Any], Mapping[str, Any] | None],
    ) -> None:
        self._ov, self._resolve = overlay, resolve
        self._last: Mapping[str, Any] | None = None
        mt = sheet.MT
        mt.bind("<Motion>", self._on_motion, add="+")
        mt.bind("<Leave>", lambda _e: overlay.schedule_hide(), add="+")
        mt.bind("<MouseWheel>", lambda _e: overlay.hide(), add="+")

    def _on_motion(self, event) -> None:
        pk = self._resolve(event)
        if pk == self._last and pk is not None and (self._ov.visible or self._ov.pending):
            if self._ov.visible:
                self._ov.cancel_hide()  # motion over target vetoes pending hide
            return  # shown or pending on same target: no churn
        self._last = pk
        if pk is None:
            self._ov.hide()  # cancels a pending show too
        else:
            self._ov.schedule_show(pk)
