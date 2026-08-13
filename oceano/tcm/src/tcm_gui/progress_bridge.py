"""GuiTqdm: drop-in tqdm bar → ProgressState + ProgressBank.  Module-level (cross-thread)."""

from __future__ import annotations

from .progress_bank import canon_stage

_runtime = None  # module-level: visible from dask worker threads
_tqdm_cls = None
_current_cfg: str | None = None


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


def set_cfg(cfg: str | None) -> None:
    """Attribute progress to a configuration.

    Call at each worker task start — next to ``reset_dedup()`` — so
    ``GuiTqdm`` ticks and ``stage_desc`` boundaries land in the right
    tab cell of the ProgressBank.
    """
    global _current_cfg
    _current_cfg = cfg


def get_cfg() -> str | None:
    return _current_cfg


def _bank():
    return getattr(_runtime, "progress_bank", None) if _runtime is not None else None


def stage_desc(desc: str) -> None:
    """Update overall progress description and mark a stage boundary.

    Called at each processing phase start so the GUI reflects the current
    stage immediately.  Additionally advances the per-config state machine
    of :class:`ProgressBank` (tab strip fill).  No-op when GUI is inactive.
    """
    if not _runtime:
        return
    cur, tot, _ = _runtime.progress_overall.snapshot()
    _runtime.progress_overall.set(cur, tot, desc)
    if bank := _bank():
        bank.stage_start(_current_cfg, canon_stage(desc))


class GuiTqdm:
    """Drop-in tqdm replacement routing progress to ``ProgressState`` and
    ``ProgressState``-like sinks.

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
            self._bank = getattr(_runtime, "progress_bank", None)
        else:
            self._ps = self._gate = self._bank = None
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
        if self._bank:
            self._bank.inner(_current_cfg, self.n, self.total)
        if self._gate:
            self._gate.wait()  # ← pause checkpoint

    def close(self) -> None:
        if self._ps:
            self._ps.set(self.total, self.total, self.desc)
        if self._bank:
            self._bank.inner(_current_cfg, self.total, self.total)

    def set_description(self, d: str) -> None:
        self.desc = d

    # tqdm compat no-ops
    set_postfix = lambda self, **_kw: None
    set_postfix_str = lambda self, _s="": None
    refresh = lambda self: None
    __enter__ = lambda self: self
    __exit__ = lambda self, *a: None
