"""Recover processing state from boundary marks in the file log.

Boundaries are the first records after set_probe / set_stage / set_sublevel,
marked by StageContextFilter: [# probe …], [## … stage …], [### … / sub].
Segments are line spans between consecutive boundaries of a probe; sublevel
segments nest within the stage active at their boundary.  Within-context
[prefix] marks (WARNING+) carry identity but do not affect segmentation.
Run outcome is best-effort from the documented terminal line "Done — …"
(how_it_works.md §processing.run).

Usage::

    rd = Reader.open(data_dir)             # None when no runs exist
    rd.probes["i90"].stages[3]             # StageSeg(file, a, name, b)
    list(rd.segment("i90", 3))             # physical lines of the stage
    list(rd.segment("i90", 1, "read"))     # all occurrences of sublevel "read"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Final, Iterator

LOG_SUBDIR: Final = Path("cfg_proc") / "log"  # hydra.run.dir convention

_PREFIX_BODY: Final = (
    r"probe (?P<pcid>\S+)(?: (?P<idx>[\d./]+))?"
    r"(?: stage (?P<num>\d+) (?P<name>\S+))?"
    r"(?: / (?P<sub>.+?))?"
)
_MARK_RE: Final = re.compile(rf"\[(?P<marks>#+)? ?{_PREFIX_BODY}\]")
_DONE_RE: Final = re.compile(r"Done — \d+ probes: (?P<ok>[\w,]*) ok.*failed \((?P<fail>[\w,]*)\)")


def parse_prefix(s: str) -> dict[str, str] | None:
    """Bare prefix (record.stage_prefix) → identity dict; None when unmatched."""
    m = re.fullmatch(_PREFIX_BODY, s)
    return {k: v for k, v in m.groupdict().items() if v} if m else None


def parse_mark(line: str) -> tuple[int, dict[str, str]] | None:
    """First mark in a log line → (level, identity); level 0 = [prefix] without #."""
    m = _MARK_RE.search(line)
    if not m:
        return None
    return (
        len(m["marks"]) if m["marks"] else 0,
        {k: v for k, v in m.groupdict().items() if v and k != "marks"},
    )


@dataclass(slots=True)
class StageSeg:
    """Line span [a, b) of one stage occurrence in a concrete log file."""

    file: Path
    a: int
    name: str = ""
    b: int | None = None  # None only mid-fold; EOF closes every span


@dataclass(slots=True)
class SubSeg:
    """Line span [a, b) of one sublevel occurrence within its stage."""

    file: Path
    a: int
    name: str
    b: int | None = None


@dataclass(slots=True)
class ProbeState:
    pcid: str
    idx: str = ""
    stages: dict[int, StageSeg] = field(default_factory=dict)
    subs: dict[int, list[SubSeg]] = field(default_factory=dict)  # stage num → occurrences
    last: int = 0
    ok: bool | None = None  # None — no Done line named this probe


class Reader:
    """Fold of boundary marks over the last WINDOW run dirs.

    Last occurrence wins per (pcid, stage num) — a partial re-run in a
    newer dir supersedes older segments.
    """

    WINDOW: Final = 5

    def __init__(self, log_root: Path) -> None:
        self.probes: dict[str, ProbeState] = {}
        self.ended = False
        dirs = sorted(d for d in log_root.iterdir() if d.is_dir())[-self.WINDOW :]
        self.latest = dirs[-1].name if dirs else ""
        for d in dirs:
            for f in sorted(d.glob("*.log")):
                self._fold(f)

    @staticmethod
    def open(data_dir: Path) -> Reader | None:
        """Reader over <data_dir>/cfg_proc/log; None when the dir is absent."""
        root = data_dir / LOG_SUBDIR
        return Reader(root) if root.is_dir() else None

    def segment(self, pcid: str, num: int, sub: str | None = None) -> Iterator[str]:
        """Lines of a stage; of all occurrences of a sublevel when *sub* is given."""
        if not (p := self.probes.get(pcid)):
            return
        segs = (
            [p.stages[num]]
            if sub is None and num in p.stages
            else [s for s in p.subs.get(num, []) if s.name == sub]
            if sub
            else []
        )
        for seg in segs:
            with open(seg.file, encoding="utf-8", errors="replace") as fh:
                yield from islice(fh, seg.a, seg.b)

    def _fold(self, path: Path) -> None:
        cur_stage: dict[str, StageSeg] = {}  # pcid → open stage seg in this file
        cur_sub: dict[str, SubSeg] = {}  # pcid → open sublevel seg
        cur_num: dict[str, int] = {}  # pcid → open stage num
        n = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh):
                if "Done —" in line:
                    self._done(line)
                if (mk := parse_mark(line)) is None:
                    continue
                level, pf = mk
                pcid = pf["pcid"]
                p = self.probes.setdefault(pcid, ProbeState(pcid=pcid))
                p.idx = pf.get("idx", p.idx)
                # Stage boundary — or a sublevel whose stage mark was overridden
                # (set_stage without details immediately followed by set_sublevel).
                if level >= 2 and "num" in pf and (level == 2 or pcid not in cur_num):
                    if prev := cur_stage.get(pcid):
                        prev.b = n
                    if sprev := cur_sub.pop(pcid, None):
                        sprev.b = n
                    num = int(pf["num"])
                    p.stages[num] = StageSeg(file=path, a=n, name=pf["name"])
                    p.last = num
                    cur_stage[pcid] = p.stages[num]
                    cur_num[pcid] = num
                if level == 3 and "sub" in pf and (num := cur_num.get(pcid)):
                    if sprev := cur_sub.get(pcid):
                        sprev.b = n
                    sseg = SubSeg(file=path, a=n, name=pf["sub"])
                    p.subs.setdefault(num, []).append(sseg)
                    cur_sub[pcid] = sseg
                # level 0/1: identity only (WARNING / probe registration)
        for seg in (*cur_stage.values(), *cur_sub.values()):
            seg.b = n + 1  # EOF closes both

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
