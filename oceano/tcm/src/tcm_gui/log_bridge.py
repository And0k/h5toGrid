"""LogRecord → queue → ScrolledText color tags.  No ANSI parsing.

Records buffered in the queue may be drained long after their originating
``lf.debug(...)`` call returned.  :class:`tcm.utils2init.LoggingStyleAdapter`
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
records' exception line (``format_exception_only``) so worker-side
``lf.exception(...)`` tracebacks surface their message in the GUI log.
"""

from __future__ import annotations

import logging
import time
import traceback
from queue import Empty, Queue

from tcm.stage_ctx import StageContextFilter

from . import const

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


def drain(q: Queue, w) -> int:
    """Drain queue → Text.  Returns count of appended records.

    Boundary marks (``stage_fresh > 0``) are already baked into
    ``rec.getMessage()`` by the QueueHandler's own ``StageContextFilter``.
    """
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
        w.insert("end", f"{rec.getMessage()}\n", tag)
        if rec.exc_info:  # exception records: show the exception line too
            w.insert("end", "".join(traceback.format_exception_only(*rec.exc_info[:2])), tag)
    return n
