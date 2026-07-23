"""LogRecord → queue → ScrolledText color tags.  No ANSI parsing."""

from __future__ import annotations

import logging
import time
from queue import Empty, Queue

TAG_COLORS: dict[str, str] = {
    "debug": "#808080",
    "info": "#1a1a1a",
    "warning": "#CC7000",
    "error": "#CC0000",
    "critical": "#CC0000",
}
FUNC_COLOR = "#0070A0"


class QueueHandler(logging.Handler):
    """Enqueue record; PauseGate checkpoint before enqueue; skip consecutive dupes."""

    def __init__(self, q: Queue, gate: PauseGate) -> None:  # noqa: F821
        super().__init__()
        self.q, self.gate = q, gate
        self._last_key: tuple[str, str] | None = None

    def emit(self, rec: logging.LogRecord) -> None:
        self.gate.wait()
        key = (rec.funcName, rec.getMessage())
        if key == self._last_key:
            return
        self._last_key = key
        self.q.put(rec)


def install(q: Queue, gate, level: int = logging.DEBUG) -> QueueHandler:
    """Attach to root logger; returns handler for later removal."""
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
