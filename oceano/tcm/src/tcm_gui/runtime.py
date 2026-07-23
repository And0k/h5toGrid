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
    """Worker writes, GUI reads (lock-guarded)."""
    current: int = 0
    total:   int = 0
    desc:    str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, cur: int, tot: int, desc: str = "") -> None:
        with self._lock:
            self.current, self.total, self.desc = cur, tot, desc

    def snapshot(self) -> tuple[int, int, str]:
        with self._lock:
            return self.current, self.total, self.desc


@dataclass
class Runtime:
    """Passed to every worker thread."""
    log_queue:        Queue         = field(default_factory=Queue)
    result_queue:     Queue         = field(default_factory=Queue)
    progress_overall: ProgressState = field(default_factory=ProgressState)
    progress_stage:   ProgressState = field(default_factory=ProgressState)
    pause_gate:       PauseGate     = field(default_factory=PauseGate)