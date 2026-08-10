Финализация: `tcm/journal.py`, дельта `stage_ctx.py`, тест.

## `tcm/journal.py`

```python
"""Recover processing state from stage-boundary marks in the file log.

A boundary is the first record after ``set_probe`` / ``set_stage``,
marked ``[## prefix]`` by :class:`~tcm.stage_ctx.StageContextFilter` at
whatever level that record was logged at.  Segments are line spans between
consecutive boundaries of a probe; within-stage ``[prefix]`` marks (WARNING+)
carry identity but do not affect segmentation.  Run outcome is best-effort
from the documented terminal line ``Done — …`` (how_it_works.md §processing.run).

Usage::

    rd = journal.Reader.open(raw_dir)   # None when no runs exist
    rd.probes["i90"].stages[3]          # StageSeg(file, a, name, b)
    list(rd.segment("i90", 3))          # physical lines of the stage
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Final, Iterator

LOG_SUBDIR: Final = Path("cfg_proc") / "log"   # hydra.run.dir convention

_PREFIX_BODY: Final = (r"probe (?P<pcid>\S+)(?: (?P<idx>[\d./]+))?"
                       r"(?: stage (?P<num>\d+) (?P<name>\S+))?")
_MARK_RE: Final = re.compile(rf"\[(?P<boundary>## )?{_PREFIX_BODY}\]")
_DONE_RE: Final = re.compile(
    r"Done — \d+ probes: (?P<ok>[\w,]*) ok.*failed \((?P<fail>[\w,]*)\)")


def parse_prefix(s: str) -> dict[str, str] | None:
    """Bare prefix ('probe i90 1/2 stage 3 proc') → identity dict; None when unmatched."""
    m = re.fullmatch(_PREFIX_BODY, s)
    return {k: v for k, v in m.groupdict().items() if v} if m else None


def parse_mark(line: str) -> tuple[bool, dict[str, str]] | None:
    """First [## probe …] / [probe …] in a log line → (is_boundary, identity)."""
    m = _MARK_RE.search(line)
    if not m:
        return None
    return (bool(m["boundary"]),
            {k: v for k, v in m.groupdict().items() if v and k != "boundary"})


@dataclass(slots=True)
class StageSeg:
    """Line span [a, b) of one stage occurrence in a concrete log file."""
    file: Path
    a: int
    name: str = ""
    b: int | None = None        # None only mid-fold; EOF closes every span


@dataclass(slots=True)
class ProbeState:
    pcid: str
    idx: str = ""
    stages: dict[int, StageSeg] = field(default_factory=dict)   # stage num → seg
    last: int = 0
    ok: bool | None = None      # None — no Done line named this probe


class Reader:
    """Fold of [## …] boundaries over the last WINDOW run dirs.

    Last occurrence wins per (pcid, stage num) — a partial re-run in a
    newer dir supersedes older segments.
    """
    WINDOW: Final = 5

    def __init__(self, log_root: Path) -> None:
        self.probes: dict[str, ProbeState] = {}
        self.ended = False
        dirs = sorted(d for d in log_root.iterdir() if d.is_dir())[-self.WINDOW:]
        self.latest = dirs[-1].name if dirs else ""
        for d in dirs:
            for f in sorted(d.glob("*.log")):
                self._fold(f)

    @staticmethod
    def open(raw_dir: Path) -> Reader | None:
        """Reader over <raw_dir>/cfg_proc/log; None when the dir is absent."""
        root = raw_dir / LOG_SUBDIR
        return Reader(root) if root.is_dir() else None

    def segment(self, pcid: str, num: int) -> Iterator[str]:
        """Physical lines of one stage occurrence; empty when absent."""
        if (p := self.probes.get(pcid)) and (seg := p.stages.get(num)):
            with open(seg.file, encoding="utf-8", errors="replace") as fh:
                yield from islice(fh, seg.a, seg.b)

    def _fold(self, path: Path) -> None:
        cur: dict[str, StageSeg] = {}      # pcid → seg open in this file
        n = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh):
                if "Done —" in line:
                    self._done(line)
                if (mk := parse_mark(line)) is None or not mk[0]:
                    continue               # no mark / within-stage [prefix]
                pf = mk[1]
                p = self.probes.setdefault(pf["pcid"], ProbeState(pcid=pf["pcid"]))
                p.idx = pf.get("idx", p.idx)
                if "num" not in pf:
                    continue               # probe-only boundary (before set_stage)
                if prev := cur.get(pf["pcid"]):
                    prev.b = n             # previous stage ends where this one starts
                seg = StageSeg(file=path, a=n, name=pf["name"])
                num = int(pf["num"])
                p.stages[num] = seg
                p.last = num
                cur[pf["pcid"]] = seg
        for seg in cur.values():
            seg.b = n + 1                  # EOF closes the span

    def _done(self, line: str) -> None:
        self.ended = True
        if not (m := _DONE_RE.search(line)):
            return
        for pcid in m["ok"].split(","):
            if pcid and (p := self.probes.get(pcid)):
                p.ok = True
        for pcid in m["fail"].split(","):
            if pcid and (p := self.probes.get(pcid)):
                p.ok = False
```

## Дельта `stage_ctx.py`

```python
_cv_fresh = contextvars.ContextVar("stage_fresh", default=False)
# _ALL_CVS: + (_cv_fresh, False)

def set_probe(probe_id, probe_idx=0, cfg_idx=0, n_probes=0, n_cfgs=0) -> None:
    """Set probe-level context — call once per config in the processing loop."""
    _cv_probe_id.set(probe_id)
    if probe_idx: _cv_probe_idx.set(probe_idx)
    if cfg_idx:   _cv_cfg_idx.set(cfg_idx)
    if n_probes:  _cv_n_probes.set(n_probes)
    if n_cfgs:    _cv_n_cfgs.set(n_cfgs)
    _cv_fresh.set(True)

def set_stage(stage_num: int, stage_name: str) -> None:
    """Set stage-level context — call at each processing phase boundary.

    The next log record (any level) carries the [## prefix] boundary mark;
    journal.Reader builds segments from these marks.
    """
    _cv_stage_num.set(stage_num)
    _cv_stage_name.set(stage_name)
    _cv_fresh.set(True)


class StageContextFilter(logging.Filter):
    """Inject record.stage_prefix; mark state boundaries with ##.

    First record after set_probe/set_stage (any level):  [## prefix] msg
    Subsequent WARNING+ within the stage:                [prefix] msg
    INFO/DEBUG within the stage:                         untouched

    record.stage_prefix / record.stage_fresh — for GUI readers.
    The decision runs once per record: a record passes several handlers,
    the contextvar must be consumed exactly once.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        prefix = _build_prefix()
        record.stage_prefix = prefix
        if getattr(record, "_stage_ctx_done", False):
            return True
        fresh = bool(prefix and _cv_fresh.get())
        if fresh:
            _cv_fresh.set(False)
        record.stage_fresh = fresh
        record._stage_ctx_done = True
        if not prefix or not (fresh or record.levelno >= logging.WARNING):
            return True
        mark = f"[## {prefix}]" if fresh else f"[{prefix}]"
        if _is_message(record.msg):
            record.msg = type(record.msg)(f"{mark} {record.msg.fmt}", record.msg.args)
        else:
            record.msg = f"{mark} {record.msg}"
        return True
```

## `tests/test_journal.py`

```python
"""Reader: segments from [## …] boundaries; WARNING marks don't break them."""
import logging
from pathlib import Path

from tcm import journal, stage_ctx

_FMT = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s|%(message)s", "%H:%M:%S")
_l = logging.getLogger("test.journal")


def _run_log(tmp: Path, fn) -> journal.Reader:
    """Emit fn()'s records through a hydra-formatted file → Reader over it."""
    log_dir = tmp / "cfg_proc" / "log" / "2026-07-29_14-15-03"
    log_dir.mkdir(parents=True)
    fh = logging.FileHandler(log_dir / "tcm_proc.log", encoding="utf-8")
    fh.setFormatter(_FMT)
    fh.addFilter(stage_ctx.StageContextFilter())  # фильтр на хендлере — как в продакшене
    _l.addHandler(fh)
    _l.setLevel(logging.DEBUG)
    try:
        fn()
    finally:
        fh.close()
        _l.removeHandler(fh)
        stage_ctx.clear()
    return journal.Reader.open(tmp)


def test_segments(tmp_path):
    def fn():
        stage_ctx.set_probe("i90", 1, 1, 2, 2)
        stage_ctx.set_stage(1, "load")
        _l.info("Loading data for i90…")       # [## probe i90 1/2 stage 1 load]
        _l.debug("Skipping i67 — covered")     # без марки
        _l.warning("Sparse region detected")   # [probe i90 1/2 stage 1 load] — без ##
        stage_ctx.set_stage(3, "proc")
        _l.info("Processing i90")              # [## … stage 3 proc]
        _l.info("Done — 1 probes: i90 ok | 0 skipped () | 0 failed ()")

    rd = _run_log(tmp_path, fn)
    p = rd.probes["i90"]
    assert p.idx == "1/2" and p.last == 3
    assert set(p.stages) == {1, 3}
    assert p.stages[1].name == "load"
    assert p.stages[1].b == p.stages[3].a      # передача диапазона на границе
    assert rd.ended and p.ok is True
    seg1 = list(rd.segment("i90", 1))
    assert any("Loading data" in l for l in seg1)
    assert any("Sparse region" in l for l in seg1)   # WARNING внутри сегмента
    assert not any("Processing i90" in l for l in seg1)


def test_boundary_at_debug(tmp_path):
    """Первая запись этапа на DEBUG тоже задаёт границу (видно в файле)."""
    def fn():
        stage_ctx.set_probe("i01")
        stage_ctx.set_stage(1, "load")
        _l.debug("chunk 0 loaded")             # [## …] на DEBUG
        _l.info("Loading done")
    rd = _run_log(tmp_path, fn)
    assert "chunk 0 loaded" in next(rd.segment("i01", 1))


def test_interrupted(tmp_path):
    def fn():
        stage_ctx.set_probe("i90")
        stage_ctx.set_stage(1, "load")
        _l.info("Loading…")
        # убит — строки Done нет
    rd = _run_log(tmp_path, fn)
    assert not rd.ended and rd.probes["i90"].ok is None
```

При накладывании проверить одно место: где сегодня прикреплён `StageContextFilter` — марки работают на всех хендлерах только при креплении к хендлерам (console/file/QueueHandler), не к логгеру; существующий WARNING-префикс работает — значит, крепление уже правильное, и `stage_fresh` дойдёт до очереди тем же путём. Следующая фаза — GUI: `_stage_tree.py` читает `rec.stage_fresh` + `parse_prefix(rec.stage_prefix)` из очереди, restore в `app._on_scan_ok` через `Reader.open`.