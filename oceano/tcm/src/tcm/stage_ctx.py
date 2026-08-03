"""Context-var driven state tracking: log context, boundary marks, progress.

Three hierarchical scopes — probe / stage / sublevel.  The setters are the
single state entry point: each updates contextvars, arms a boundary mark,
and — when given details — emits the boundary record itself, so a state
transition is logged and GUI-visible from the call alone::

    set_stage(1, "load", "Loading %s", path.name)
    # INFO:  [## probe i90 stage 1 load] Loading @i90.TXT

    set_sublevel("read", "chunk %d/%d", k, n)
    # DEBUG: [### probe i90 stage 1 load / read] chunk 2/3

    tick()   # advance stage counter + progress bar (no log — pure state)

Progress bar positioning is also centralised here: ``set_probe(stem_idx=…,
n_cfgs_total=…)`` computes the per-probe base offset; ``set_stage_plan(n)``
records how many active stages exist; ``tick()`` auto-increments the counter
and updates ``progress_overall`` via an optional ``progress_bridge`` import.
"""

from __future__ import annotations

import contextvars
import logging

# Optional GUI bridge — no-op when GUI is not installed.
try:
    from tcm_gui import progress_bridge as _pb
except ImportError:
    _pb = None  # type: ignore[assignment]

# ── Context variables (task-local, thread-safe) ─────────────────────────

_cv_probe_id = contextvars.ContextVar("probe_id", default="")
_cv_probe_idx = contextvars.ContextVar("probe_idx", default=0)  # 1-based
_cv_cfg_idx = contextvars.ContextVar("cfg_idx", default=0)  # 1-based within probe
_cv_n_probes = contextvars.ContextVar("n_probes", default=0)
_cv_n_cfgs = contextvars.ContextVar("n_cfgs", default=0)
_cv_stage_num = contextvars.ContextVar("stage_num", default=0)
_cv_stage_name = contextvars.ContextVar("stage_name", default="")
_cv_sub = contextvars.ContextVar("sublevel", default="")
# Boundary mark armed by the setters: 1 probe / 2 stage / 3 sublevel;
# consumed by StageContextFilter on the first record that passes it.
_cv_fresh = contextvars.ContextVar("fresh", default=0)
# Progress positioning — set once per config by set_probe / set_stage_plan.
_cv_probe_base = contextvars.ContextVar("probe_base", default=0)
_cv_probe_total = contextvars.ContextVar("probe_total", default=0)
_cv_n_active = contextvars.ContextVar("n_active", default=0)
_cv_tick_idx = contextvars.ContextVar("tick_idx", default=0)

_ALL_CVS = (
    (_cv_probe_id, ""),
    (_cv_probe_idx, 0),
    (_cv_cfg_idx, 0),
    (_cv_n_probes, 0),
    (_cv_n_cfgs, 0),
    (_cv_stage_num, 0),
    (_cv_stage_name, ""),
    (_cv_sub, ""),
    (_cv_fresh, 0),
    (_cv_probe_base, 0),
    (_cv_probe_total, 0),
    (_cv_n_active, 0),
    (_cv_tick_idx, 0),
)

# Boundaries carry name=tcm.stage_ctx (framework event) and funcName of the
# caller (stacklevel=2) — distinct from module logs in the file format.
_lf = logging.getLogger(__name__)


# ── Setters ─────────────────────────────────────────────────────────────


def set_probe(
    probe_id: str,
    probe_idx: int = 0,
    cfg_idx: int = 0,
    n_probes: int = 0,
    n_cfgs: int = 0,
    stem_idx: int = 0,
    n_cfgs_total: int = 0,
) -> None:
    """Set probe-level context — call once per config in the processing loop.

    Clears stage/sublevel/tick scopes.  When *stem_idx* and *n_cfgs_total*
    are given, computes progress bar positioning (100 units per config).
    """
    _cv_probe_id.set(probe_id)
    if probe_idx:
        _cv_probe_idx.set(probe_idx)
    if cfg_idx:
        _cv_cfg_idx.set(cfg_idx)
    if n_probes:
        _cv_n_probes.set(n_probes)
    if n_cfgs:
        _cv_n_cfgs.set(n_cfgs)
    _cv_stage_num.set(0)
    _cv_stage_name.set("")
    _cv_sub.set("")
    _cv_fresh.set(1)
    _cv_tick_idx.set(0)
    _cv_probe_base.set((stem_idx - 1) * 100 if stem_idx else 0)
    _cv_probe_total.set(n_cfgs_total * 100 if n_cfgs_total else 0)


def set_stage_plan(n_active: int) -> None:
    """Record how many stages are active in this config (for ``tick()`` fraction)."""
    _cv_n_active.set(n_active)


def set_stage(stage_num: int, stage_name: str, details: str = "", *args) -> None:
    """Set stage-level context — call at each processing phase boundary.

    With *details* (printf fmt + args) the boundary record is emitted here
    at INFO: ``[## probe … stage N name] <rendered>``.  Without — the mark
    rides on the stage's next natural log record.  Clears the sublevel scope.
    """
    _cv_stage_num.set(stage_num)
    _cv_stage_name.set(stage_name)
    _cv_sub.set("")
    _cv_fresh.set(2)
    if details:
        _lf.info(details, *args, stacklevel=2)


def set_sublevel(name: str, details: str = "", *args) -> None:
    """Set sublevel context — a sub-stage within a stage (chunk pass, kernel).

    With *details* the boundary record is emitted here at DEBUG:
    ``[### … / name] <rendered>``.  Without — the mark rides on the
    sublevel's next record (fine for unimportant sublevels).
    """
    _cv_sub.set(name)
    _cv_fresh.set(3)
    if details:
        _lf.debug(details, *args, stacklevel=2)


def tick(stage_name: str = "") -> None:
    """Advance stage counter and update the overall progress bar.

    Optionally set stage context when *stage_name* is given (useful for
    the ``_tick`` callback inside ``_process_and_persist`` which combines
    stage change + progress in one call).  No log record is emitted —
    pure state transition.
    """
    idx = _cv_tick_idx.get() + 1
    _cv_tick_idx.set(idx)
    if stage_name:
        _cv_stage_num.set(idx)
        _cv_stage_name.set(stage_name)
        _cv_sub.set("")
        _cv_fresh.set(2)
    if _pb and (n_active := _cv_n_active.get()):
        frac = round(idx * 100 / n_active)
        base = _cv_probe_base.get()
        total = _cv_probe_total.get()
        if rt := _pb.get_runtime():
            rt.progress_overall.set(base + frac, total, _build_prefix())


def clear() -> None:
    """Reset all context vars to defaults (call after processing completes)."""
    for cv, default in _ALL_CVS:
        cv.set(default)


def snapshot() -> tuple[str, int, int, int, int, int, str, str]:
    """Current context as a plain tuple (for GUI / external readers)."""
    return (
        _cv_probe_id.get(),
        _cv_probe_idx.get(),
        _cv_cfg_idx.get(),
        _cv_n_probes.get(),
        _cv_n_cfgs.get(),
        _cv_stage_num.get(),
        _cv_stage_name.get(),
        _cv_sub.get(),
    )


# ── Logging filter ──────────────────────────────────────────────────────


def _build_prefix() -> str:
    """Human-readable prefix from current context vars.

    Format: ``probe {id} [{idx}] [stage {n} {name}] [/ {sublevel}]`` —
    sub-indexes only when the corresponding count > 1.
    """
    pid = _cv_probe_id.get()
    if not pid:
        return ""
    parts = [f"probe {pid}"]
    pi, ci = _cv_probe_idx.get(), _cv_cfg_idx.get()
    np_, nc = _cv_n_probes.get(), _cv_n_cfgs.get()
    if np_ > 1:
        parts.append(f"{pi}.{ci}/{np_}.{nc}" if nc > 1 else f"{pi}/{np_}")
    sn, sname = _cv_stage_num.get(), _cv_stage_name.get()
    if sname:
        parts.append(f"stage {sn} {sname}" if sn else sname)
    if sub := _cv_sub.get():
        parts.append(f"/ {sub}")
    return " ".join(parts)


class StageContextFilter(logging.Filter):
    """Inject record.stage_prefix; consume the armed boundary mark.

    First record after a setter (any level):   [## / ### prefix] msg
    Subsequent WARNING+ within the scope:      [prefix] msg
    INFO/DEBUG within the scope:               untouched

    Record attributes for readers: stage_prefix (always), stage_fresh
    (mark level 0-3), boundary_msg (pristine message of a marked record).
    The decision runs once per record: a record passes several handlers,
    the mark must be consumed exactly once.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        prefix = _build_prefix()
        record.stage_prefix = prefix  # type: ignore[attr-defined]
        if getattr(record, "_stage_ctx_done", False):
            return True
        level = _cv_fresh.get() if prefix else 0
        if level:
            _cv_fresh.set(0)
            record.boundary_msg = record.getMessage()  # type: ignore[attr-defined]
        record.stage_fresh = level  # type: ignore[attr-defined]
        record._stage_ctx_done = True  # type: ignore[attr-defined]
        if not prefix or not (level or record.levelno >= logging.WARNING):
            return True
        mark = f"{'#' * level} " if level else ""
        text = f"[{mark}{prefix}]"
        # Prepend — create a new Message copy when needed to avoid mutating
        # the shared singleton in LoggingStyleAdapter.
        if _is_message(record.msg):
            record.msg = type(record.msg)(f"{text} {record.msg.fmt}", record.msg.args)
        else:
            record.msg = f"{text} {record.msg}"
        return True


def _is_message(msg: object) -> bool:
    """True if *msg* is a :class:`tcm.utils2init.Message` instance."""
    return hasattr(msg, "fmt") and hasattr(msg, "args") and not isinstance(msg, str)
