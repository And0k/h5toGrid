"""LogRecord → queue → ScrolledText color tags.  No ANSI parsing.

Records buffered in the queue may be drained long after their originating
``lf.debug(...)`` call returned.  :class:`utils.init.LoggingStyleAdapter`
mutates a single ``Message`` instance across log calls (its ``fmt``/``args``
overwrite in place), so a deferred ``rec.getMessage()`` would render the
*latest* message instead of the one recorded at emit time — every record
from a given logger collapses to the last text it produced.

Hydra's ``job_logging/colorlog`` formatter sidesteps this by rendering
``record.getMessage()`` synchronously inside ``Formatter.format`` (writing
straight to its stream).  :class:`QueueHandler` mirrors that contract: it
formats the message once at emit time and freezes the result back onto the
record (``rec.msg = text; rec.args = ()``) so the GUI side, which only ever
calls ``rec.getMessage()`` lazily from the queue, sees the same immutable text
Hydra already wrote to console/file.

``emit`` degrades gracefully: a record whose message cannot render through
stdlib ``%``-formatting (e.g. a ``{}``-style string on a plain logger, or a
stray ``%`` in user content) still reaches the queue with its raw text — a
logging bug must never crash the GUI callback that produced it (the About
dialog used to die this way).  ``drain`` additionally renders ``exc_info``
records' **full traceback** (``traceback.format_exception``) so worker-side
``lf.exception(...)`` errors surface their exact location (file:line of each
frame) in the GUI log.

Message and traceback chunks render through :meth:`LogText.insert_linked`
when the widget provides it — bare filesystem paths under the allowed dir
(:mod:`tcm_gui._allowed_paths`) become clickable links there.
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
import traceback
from pathlib import Path
from queue import Empty, Queue

from tcm.stage_ctx import StageContextFilter

from . import _allowed_paths, const, theme
from .browser.browser import open_md_link

# Re-export for ``install()`` callers.
__all__ = ["QueueHandler", "install", "drain"]


class QueueHandler(logging.Handler):
    """Enqueue record; PauseGate checkpoint before enqueue; skip consecutive dupes.

    Has its own :class:`StageContextFilter` so ``emit()`` sees the prefixed
    message (boundary marks like ``[## …]``) **before** freezing.  Without
    it the dedup key would be the raw message, collapsing same-named
    boundaries.
    """

    def __init__(self, q: Queue, gate: PauseGate) -> None:  # noqa: F821
        super().__init__()
        self.q, self.gate = q, gate
        self._last_key: tuple[str, str] | None = None
        # Own filter — mark prefix must be visible to freeze + dedup.
        self.addFilter(StageContextFilter())

    def reset_dedup(self) -> None:
        """Clear consecutive-duplicate state — call at the start of each worker task."""
        self._last_key = None

    def emit(self, rec: logging.LogRecord) -> None:
        self.gate.wait()
        # Render once with live fmt/args, then freeze the result so later
        # drain-time getMessage() returns the same text — not whatever the
        # caller's reused Message object was last mutated to.
        try:
            text = rec.getMessage()
        except Exception:
            # fmt can't survive stdlib %-formatting ({} on plain logger, stray
            # % in user content) — keep the record's raw text and move on:
            # the GUI log must still receive it and the caller must never crash.
            text = str(rec.msg)
        rec.msg, rec.args = text, ()
        key = (rec.funcName, text)
        if key == self._last_key:
            return
        self._last_key = key
        self.q.put(rec)


def install(q: Queue, gate, level: int = logging.DEBUG) -> QueueHandler:
    """Attach a new :class:`QueueHandler` to the root logger and return it.

    Intended to be called **once** at GUI startup.  The returned handler
    reference should be stored (e.g. on :attr:`Runtime.queue_handler`) so
    that :meth:`QueueHandler.reset_dedup` can be called between worker
    tasks and the handler can be re-attached after Hydra's ``dictConfig``
    replaces root handlers.
    """
    h = QueueHandler(q, gate)
    h.setLevel(level)
    logging.getLogger().addHandler(h)
    return h


def drain(q: Queue, w, root: str = "") -> int:
    """Drain queue → Text.  Returns count of appended records.

    Boundary marks (``stage_fresh > 0``) are already baked into
    ``rec.getMessage()`` by the QueueHandler's own ``StageContextFilter``.
    Message and traceback chunks go through ``w.insert_linked(text, tags,
    root)`` when the widget provides it (path auto-linking under the *root*
    allowed dir); plain widgets fall back to a plain insert.
    """
    linked = getattr(w, "insert_linked", None)

    def insert(text: str, tags) -> None:
        if linked:
            linked(text, tags, root)
        else:
            w.insert("end", text, tags)

    n = 0
    while True:
        try:
            rec = q.get_nowait()
        except Empty:
            break
        n += 1
        ts = time.strftime("%H:%M:%S", time.localtime(rec.created))
        tag = rec.levelname.lower()
        w.insert("end", f"{ts}│", tag)
        w.insert("end", f"{rec.funcName}│", "func")
        insert(f"{rec.getMessage()}\n", tag)
        if rec.exc_info:  # exception records: full traceback — exact error location
            insert("".join(traceback.format_exception(*rec.exc_info)), tag)
    return n


class LogText(tk.Text):
    """Log ``tk.Text`` with clickable filesystem-path links.

    Linkable paths (:mod:`tcm_gui._allowed_paths`, restricted to the *root*
    allowed dir) are inserted by :meth:`insert_linked`: files display shrunk
    to the file name, directories in full.  Link chunks carry the shared
    ``loglink`` visual tag plus a per-target data tag ``link:<file: URI>``
    (URI-escaped → space-free → valid Tk tag name) — the same "tag IS the
    URL" convention as ``tcm._md_parse`` spans.  Tk adjusts tag ranges on any
    edit, so there is no index bookkeeping; :meth:`link_url_at` is the hook
    duck-typed by ``tcm_gui._rtf_clipboard`` (Ctrl+C exports hyperlinks).
    Clicks route through ``open_md_link`` → the OS-associated application.
    """

    def __init__(self, master=None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.tag_configure("loglink", foreground=theme.LINK_FG, underline=True)
        self.tag_bind("loglink", "<Button-1>", self._open_link)
        self.tag_bind("loglink", "<Enter>", lambda _e: self.configure(cursor="hand2"))
        self.tag_bind("loglink", "<Leave>", lambda _e: self.configure(cursor=""))

    def insert_linked(self, text: str, base_tags, root: str = "") -> None:
        """Insert *text*; linkable paths under the *root* allowed dir become links."""
        pos = 0
        for m in _allowed_paths.allowed_path_re(root).finditer(text):
            if not (kind := _allowed_paths.path_kind(p := m[0])):
                continue  # odd extension / nonexistent dir → stays plain
            self.insert("end", text[pos : m.start()], base_tags)
            display = Path(p).name if kind == "file" else p
            self.insert("end", display, (*base_tags, "loglink", f"link:{_allowed_paths.to_uri(p)}"))
            pos = m.end()
        self.insert("end", text[pos:], base_tags)

    def link_url_at(self, index: str) -> str | None:
        """``file:`` URI of the link span at *index* (``None`` outside links)."""
        return next(
            (t.removeprefix("link:") for t in self.tag_names(index) if t.startswith("link:")),
            None,
        )

    def link_at(self, x: int, y: int) -> str | None:
        """Target URL of the link span at widget coordinates ``(x, y)``."""
        return self.link_url_at(f"@{x},{y}")

    def _open_link(self, event: tk.Event) -> None:
        if url := self.link_url_at(f"@{event.x},{event.y}"):
            open_md_link(url)
