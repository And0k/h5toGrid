"""Shared state: queues, progress snapshots, pause gate."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from queue import Queue


class PauseGate:
    """Freeze/resume worker at log/tqdm checkpoints — pipeline code untouched."""
    __slots__ = ("_ev",)

    def __init__(self) -> None:
        self._ev = threading.Event()
        self._ev.set()

    pause  = lambda s: s._ev.clear()
    resume = lambda s: s._ev.set()
    wait   = lambda s: s._ev.wait()

    @property
    def paused(self) -> bool:
        return not self._ev.is_set()


@dataclass
class ProgressState:
    """Worker writes, GUI reads (lock-guarded).

    :param _clear_status: one-shot flag — :meth:`clear_and_reset` sets it,
        :meth:`consume_clear` atomically reads-and-resets it.  Used at probe
        boundaries so the GUI poll can wipe stale stage text without touching
        it in every idle tick.
    """
    current: int = 0
    total:   int = 0
    desc:    str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _clear_status: bool = field(default=False, repr=False)

    def set(self, cur: int, tot: int, desc: str = "") -> None:
        with self._lock:
            self.current, self.total, self.desc = cur, tot, desc

    def snapshot(self) -> tuple[int, int, str]:
        with self._lock:
            return self.current, self.total, self.desc

    def clear_and_reset(self) -> None:
        """Reset to idle *and* signal the GUI to clear status text once."""
        with self._lock:
            self.current, self.total, self.desc = 0, 0, ""
            self._clear_status = True

    def consume_clear(self) -> bool:
        """Atomically check-and-reset the clear-status flag (GUI thread)."""
        with self._lock:
            if self._clear_status:
                self._clear_status = False
                return True
            return False


@dataclass
class Runtime:
    """Passed to every worker thread."""
    log_queue:        Queue         = field(default_factory=Queue)
    result_queue:     Queue         = field(default_factory=Queue)
    progress_overall: ProgressState = field(default_factory=ProgressState)
    progress_stage:   ProgressState = field(default_factory=ProgressState)
    pause_gate:       PauseGate     = field(default_factory=PauseGate)
    # QueueHandler installed once at GUI startup on the root logger so log
    # records from GUI callbacks (main thread) AND worker tasks (background
    # thread) both reach the ScrolledText.  Worker resets dedup state per task.
    queue_handler: QueueHandler | None = field(default=None, repr=False)  # noqa: F821