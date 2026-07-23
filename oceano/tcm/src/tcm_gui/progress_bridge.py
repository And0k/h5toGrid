"""GuiTqdm: drop-in tqdm bar → ProgressState.  Module-level (cross-thread)."""
from __future__ import annotations

_runtime = None          # module-level: visible from dask worker threads
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


class GuiTqdm:
    """Replaces tqdm bar inside TqdmCallback(tqdm_class=GuiTqdm)."""

    def __init__(self, total=None, desc=None, **_kw) -> None:
        if _runtime:
            self._ps = _runtime.progress_stage
            self._gate = _runtime.pause_gate
        else:
            self._ps = None
            self._gate = None
        self.total, = total or 0
        self.n = 0
        self.desc = desc or ""

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
    refresh = lambda self: None
    __enter__ = lambda self: self
    __exit__ = lambda self, *a: None
