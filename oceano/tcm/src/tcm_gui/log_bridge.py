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
"""

from __future__ import annotations

import logging
import time
from queue import Empty, Queue

FUNC_COLOR = "#0070A0"


class QueueHandler(logging.Handler):
    """Enqueue record; PauseGate checkpoint before enqueue; skip consecutive dupes.

    Freezes the rendered message onto the record (mirroring Hydra's colorlog
    formatter, which calls ``getMessage()`` once at emit time).  Subsequent
    lazy reads by :func:`drain` cannot be corrupted by the mutable ``Message``
    reused across log calls in :class:`~tcm.utils2init.LoggingStyleAdapter`.

    Installed **once** on the root logger at GUI startup so log calls from
    *any* thread — GUI callbacks (``_reload_coefs``, ``_scan``) and the
    worker pipeline alike — reach the ScrolledText.  Between worker tasks
    call :meth:`reset_dedup` to clear the consecutive-duplicate state so the
    first record of a new task is never swallowed as a "duplicate" of the
    last record of the previous task.
    """

    def __init__(self, q: Queue, gate: PauseGate) -> None:  # noqa: F821
        super().__init__()
        self.q, self.gate = q, gate
        self._last_key: tuple[str, str] | None = None

    def reset_dedup(self) -> None:
        """Clear consecutive-duplicate state — call at the start of each worker task."""
        self._last_key = None

    def emit(self, rec: logging.LogRecord) -> None:
        self.gate.wait()
        # Render once with live fmt/args, then freeze the result so later
        # drain-time getMessage() returns the same text — not whatever the
        # caller's reused Message object was last mutated to.
        text = rec.getMessage()
        rec.msg, rec.args = text, ()
        key = (rec.funcName, rec.getMessage())
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
    """Drain queue → Text. Returns count of appended records."""
    n = 0
    while True:
        try: rec = q.get_nowait()
        except Empty: break
        n += 1
        ts  = time.strftime("%H:%M:%S", time.localtime(rec.created))
        tag = rec.levelname.lower()
        w.insert("end", f"{ts}│", tag)
        w.insert("end", f"{rec.funcName}│", "func")
        w.insert("end", f"{rec.getMessage()}\n", tag)
    return n
