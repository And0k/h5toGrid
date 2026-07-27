"""GuiTqdm: drop-in tqdm bar → ProgressState.  Module-level (cross-thread)."""

from __future__ import annotations

_runtime = None  # module-level: visible from dask worker threads
_tqdm_cls = None


def set_runtime(rt) -> None:
    global _runtime
    _runtime = rt


def get_runtime():
    return _runtime


def set_tqdm_class(cls) -> None:
    global _tqdm_cls
    _tqdm_cls = cls


def get_tqdm_class():
    return _tqdm_cls


def stage_desc(desc: str) -> None:
    """Update overall progress description without advancing the value.

    Called at each processing phase start so the GUI label reflects
    the current stage immediately (not only after the stage completes).
    No-op when GUI is not active.
    """
    if _runtime:
        cur, tot, _ = _runtime.progress_overall.snapshot()
        _runtime.progress_overall.set(cur, tot, desc)


class GuiTqdm:
    """Drop-in tqdm replacement routing progress to :class:`ProgressState`.

    Used in two contexts:
    - **TqdmCallback**: ``GuiTqdm(total=N, desc=...)`` — dask task-level bar.
      ``update()`` / ``close()`` called by the callback.
    - **Binning loop**: ``GuiTqdm(iterable, desc=...)`` — per-bin progress.
      Used as an iterator (``for item in bar:``).
    """

    def __init__(self, iterable=None, total=None, desc=None, **_kw) -> None:
        self._iterable = iterable
        if _runtime:
            self._ps = _runtime.progress_stage
            self._gate = _runtime.pause_gate
        else:
            self._ps = None
            self._gate = None
        # Infer total from iterable len when not given explicitly
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = 0
        self.total = total or 0
        self.n = 0
        self.desc = desc or ""

    def __iter__(self):
        """Yield items from wrapped iterable, updating progress per item."""
        for item in self._iterable:
            yield item
            self.update(1)

    def update(self, n: int = 1) -> None:
        self.n += n
        if self._ps:
            self._ps.set(self.n, self.total, self.desc)
        if self._gate:
            self._gate.wait()  # ← pause checkpoint

    def close(self) -> None:
        if self._ps:
            self._ps.set(self.total, self.total, self.desc)

    def set_description(self, d: str) -> None:
        self.desc = d

    # tqdm compat no-ops
    set_postfix = lambda self, **_kw: None
    set_postfix_str = lambda self, _s="": None
    refresh = lambda self: None
    __enter__ = lambda self: self
    __exit__ = lambda self, *a: None
