"""Browse button that attaches to tksheet's TextEditor during cell editing.

Generalised for any row whose meta carries ``"browse": True`` — currently
``input.path`` and ``input.coefs_path``.  The button icon toggles between
folder (default) and file (Shift held) via periodic Shift-key polling.

Three-layer teardown (any one suffices):
  * ``end_edit_cell`` → ``detach()`` — normal close path.
  * ``<Destroy>`` on editor → ``detach()`` — tree-arrow / ``load()`` rebuild.
  * polling in ``_update_icon`` → ``detach()`` — editor reuse without destroy.

Identity anchor: ``attach(row, col)`` snapshots the pre-existing editor
(``_stale``).  ``_acquire_and_place`` accepts a new editor only if it is
*not* the stale one — prevents a pending retry from landing on a foreign
editor that appeared after the target edit closed.
"""
from __future__ import annotations

import ctypes
from collections.abc import Callable
from tkinter import TclError, filedialog, ttk
from typing import Any

from tksheet import Sheet

COEF_FILETYPES = [("Coefs", "*.h5 *.nc *.yaml *.yml"), ("All", "*.*")]

# Button labels — ellipsis + icon, width=4 fits both glyphs.
_LBL_DIR = "…📁"  # no-Shift default: browse directory
_LBL_FILE = "…📄"  # Shift held: browse files
_POLL_MS = 80  # Shift-state icon polling + editor-aliveness interval
_RETRY_MS = 50  # editor acquisition retry interval
_MAX_RETRIES = 20  # × _RETRY_MS ≈ 1 s budget for editor creation


def _is_shift_pressed() -> bool:
    """Return True when either Shift key is currently held."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
    except (AttributeError, OSError):
        return False


def _alive(widget) -> bool:
    """Return True if *widget* still exists in the Tk interpreter."""
    try:
        return bool(widget.winfo_exists())
    except TclError:
        return False


class BrowseButtonManager:
    """Attaches a browse button to tksheet's TextEditor when editing
    path-type rows (meta ``"browse": True``).  Visible only while the
    editor is open.

    **Identity anchor** (F1): ``attach(row, col)`` snapshots the pre-existing
    editor as ``_stale``.  ``begin_edit_cell`` fires *before* editor creation,
    so anything already present is foreign.  ``_acquire_and_place`` accepts a
    new editor only if ``ε is not None ∧ ε is not _stale ∧ _alive(ε)``.

    **Three-layer teardown**:
      1. ``end_edit_cell`` → ``detach()`` — normal close.
      2. ``<Destroy>`` on editor → ``detach()`` — arrow / ``load()`` rebuild.
      3. ``_update_icon`` polling → ``detach()`` — editor reuse without destroy.

    **Idempotent reset** (F3): ``attach()`` opens with ``detach()``, so a live
    button from a previous cycle is destroyed before a new one is created.

    **Why ``after(_RETRY_MS)``** for the first attempt: tksheet fires
    ``begin_edit_cell`` before creating the TextEditor.
    """

    def __init__(self, sheet: Sheet, on_path_changed: Callable[[str], None]) -> None:
        self._sheet = sheet
        self.notify_path_changed = on_path_changed
        self._button: ttk.Button | None = None
        self._editor: Any = None
        self._target: tuple[int, int] | None = None
        self._retries: int = 0
        self._retry_job: str | None = None
        self._icon_job: str | None = None

    # ── attach / detach lifecycle ────────────────────────────────────

    def attach(self, row: int, col: int) -> None:
        """Begin retry-polling for the TextEditor widget.

        F3 — opens with ``detach()`` to destroy any live button from a
        previous cycle and cancel stale retries.
        """
        self.detach()  # F3 — idempotent reset
        self._target = (row, col)
        self._retries = 0
        self._retry_job = self._sheet.after(_RETRY_MS, self._acquire_and_place)

    def _acquire_and_place(self) -> None:
        """Timeout callback: try to grab the editor; retry if absent.

        Accepts whatever ``get_text_editor_widget()`` returns — tksheet may
        reuse a single TextEditor widget for every cell, so identity checks
        would reject the reused widget.
        """
        self._retry_job = None
        ed = self._sheet.get_text_editor_widget()
        if ed is not None and _alive(ed):
            self._place(ed)
            return
        self._retries += 1
        if self._retries < _MAX_RETRIES:
            self._retry_job = self._sheet.after(_RETRY_MS, self._acquire_and_place)

    def _place(self, editor) -> None:
        """Create button and position it next to the editor.

        F2 — binds ``<Destroy>`` on the editor so that any close path
        (including tree-arrow toggle and ``load()`` rebuild) triggers
        ``detach()``.
        """
        self._editor = editor
        editor.bind("<Destroy>", self._on_editor_destroy, add="+")  # F2
        self._button = ttk.Button(self._sheet, text=_LBL_DIR, width=4, command=self._browse)
        self._button.focus_set = lambda: None  # type: ignore[method-assign]
        self._button.configure(takefocus=False)
        self._button.bind("<Button-1>", lambda _e: (self._browse(), "break")[-1], add="+")
        self._button.place(
            in_=editor,
            relx=1.0,
            x=0,
            rely=0,
            y=0,
            height=editor.winfo_height(),
        )
        self._start_icon_polling()

    def _on_editor_destroy(self, event) -> None:
        """F2 — ``<Destroy>`` fires for every descendant; act only on the editor."""
        if event.widget is self._editor:
            self.detach()

    def detach(self) -> None:
        """Destroy button and cancel ALL pending jobs.

        Idempotent: safe to call multiple times or when nothing is active.
        Never touches the editor widget — only our own state.
        """
        self._cancel_retry()
        self._stop_icon_polling()
        btn, self._button = self._button, None
        if btn is not None:
            btn.destroy()
        self._editor = None
        self._target = None

    def _cancel_retry(self) -> None:
        job, self._retry_job = self._retry_job, None
        if job is not None:
            self._sheet.after_cancel(job)

    # ── Shift-aware icon + editor-aliveness polling ──────────────────

    def _start_icon_polling(self) -> None:
        self._update_icon()

    def _stop_icon_polling(self) -> None:
        job, self._icon_job = self._icon_job, None
        if job is not None:
            self._sheet.after_cancel(job)

    def _update_icon(self) -> None:
        """Refresh icon glyph; auto-detach if the editor is gone or replaced.

        Layer-3 teardown: tksheet may reuse a single TextEditor canvas window
        without firing ``end_edit_cell`` or ``<Destroy>``.  Polling
        ``get_text_editor_widget()`` detects this.
        """
        if self._editor is None or self._button is None:
            return
        if self._sheet.get_text_editor_widget() is not self._editor:
            self.detach()
            return
        self._button.configure(text=_LBL_FILE if _is_shift_pressed() else _LBL_DIR)
        self._icon_job = self._sheet.after(_POLL_MS, self._update_icon)

    # ── click handling ───────────────────────────────────────────────

    def _browse(self) -> None:
        if self._editor is None or self._target is None:
            return
        if _is_shift_pressed():
            paths = filedialog.askopenfilenames(title="Coefficient files", filetypes=COEF_FILETYPES)
            if not paths:
                return
            result = ",".join(paths)
        else:
            result = filedialog.askdirectory(title="Coefficients directory")
            if not result:
                return
        # Always write to column 0 — that's where the path lives, regardless
        # of which column the user clicked to open the editor.
        row, _col = self._target
        self._sheet.set_cell_data(row, 0, result)
        # Close the editor (was on whatever column the user clicked).
        # after_idle avoids re-entrancy from inside the button-click handler.
        self._sheet.after_idle(self._sheet.close_text_editor)
