"""Context-var driven stage tracking for logging and progress.

Uses :mod:`contextvars` so the current probe/stage identity is visible to
all loggers in the same task without passing parameters through every call.
:class:`StageContextFilter` injects ``record.stage_prefix`` into every
:class:`~logging.LogRecord` — INFO/DEBUG messages are *not* modified;
WARNING+ messages automatically receive ``[prefix]`` prepended to the text.

Typical usage::

    from tcm import stage_ctx

    stage_ctx.set_probe("i90", probe_idx=1, cfg_idx=1, n_probes=2, n_cfgs=4)
    stage_ctx.set_stage(1, "load")
    lf.info("Loading data…")              # clean — no prefix in log
    lf.warning("Sparse region detected")  # [probe i90 1/2 stage 1 load] Sparse region detected
"""

from __future__ import annotations

import contextvars
import logging

# ── Context variables (task-local, thread-safe) ─────────────────────────
_cv_probe_id = contextvars.ContextVar("probe_id", default="")
_cv_probe_idx = contextvars.ContextVar("probe_idx", default=0)  # 1-based
_cv_cfg_idx = contextvars.ContextVar("cfg_idx", default=0)  # 1-based within probe
_cv_n_probes = contextvars.ContextVar("n_probes", default=0)
_cv_n_cfgs = contextvars.ContextVar("n_cfgs", default=0)  # total valid configs
_cv_stage_num = contextvars.ContextVar("stage_num", default=0)
_cv_stage_name = contextvars.ContextVar("stage_name", default="")
_ALL_CVS = (
    (_cv_probe_id, ""),
    (_cv_probe_idx, 0),
    (_cv_cfg_idx, 0),
    (_cv_n_probes, 0),
    (_cv_n_cfgs, 0),
    (_cv_stage_num, 0),
    (_cv_stage_name, ""),
)


def set_probe(
    probe_id: str,
    probe_idx: int = 0,
    cfg_idx: int = 0,
    n_probes: int = 0,
    n_cfgs: int = 0,
) -> None:
    """Set probe-level context — call once per config in the processing loop."""
    _cv_probe_id.set(probe_id)
    if probe_idx:
        _cv_probe_idx.set(probe_idx)
    if cfg_idx:
        _cv_cfg_idx.set(cfg_idx)
    if n_probes:
        _cv_n_probes.set(n_probes)
    if n_cfgs:
        _cv_n_cfgs.set(n_cfgs)


def set_stage(stage_num: int, stage_name: str) -> None:
    """Set stage-level context — call at each processing phase boundary."""
    _cv_stage_num.set(stage_num)
    _cv_stage_name.set(stage_name)


def clear() -> None:
    """Reset all context vars to defaults (call after processing completes)."""
    for cv, default in _ALL_CVS:
        cv.set(default)


def snapshot() -> tuple[str, int, int, int, int, int, str]:
    """Return current context as a plain tuple (for GUI / external readers)."""
    return (
        _cv_probe_id.get(),
        _cv_probe_idx.get(),
        _cv_cfg_idx.get(),
        _cv_n_probes.get(),
        _cv_n_cfgs.get(),
        _cv_stage_num.get(),
        _cv_stage_name.get(),
    )


# ── Logging filter ──────────────────────────────────────────────────────


def _build_prefix() -> str:
    """Build human-readable stage prefix from current context vars."""
    pid = _cv_probe_id.get()
    pi, ci = _cv_probe_idx.get(), _cv_cfg_idx.get()
    np_, nc = _cv_n_probes.get(), _cv_n_cfgs.get()
    sn, sname = _cv_stage_num.get(), _cv_stage_name.get()

    parts: list[str] = []
    if pid:
        parts.append(f"probe {pid}")
        # Display sub-indexes only when more than one
        if np_ > 1:
            if nc > 1:
                parts.append(f"{pi}.{ci}/{np_}.{nc}")
            else:
                parts.append(f"{pi}/{np_}")
    if sname:
        if sn:
            parts.append(f"stage {sn} {sname}")
        else:
            parts.append(sname)
    return " ".join(parts)


class StageContextFilter(logging.Filter):
    """Inject ``record.stage_prefix`` from contextvars.

    INFO/DEBUG: ``stage_prefix`` attribute is set (for GUI) but message is untouched.
    WARNING+:   ``[prefix]`` is prepended to the message text (once, guarded by flag).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        prefix = _build_prefix()
        record.stage_prefix = prefix  # type: ignore[attr-defined]

        if prefix and record.levelno >= logging.WARNING and not getattr(record, "_stage_prefixed", False):
            # Prepend [prefix] — create a new Message copy if needed to avoid
            # mutating the shared singleton in LoggingStyleAdapter.
            record._stage_prefixed = True  # type: ignore[attr-defined]
            if _is_message(record.msg):
                record.msg = type(record.msg)(f"[{prefix}] {record.msg.fmt}", record.msg.args)
            else:
                record.msg = f"[{prefix}] {record.msg}"
        return True


def _is_message(msg: object) -> bool:
    """True if *msg* is a :class:`tcm.utils2init.Message` instance."""
    return hasattr(msg, "fmt") and hasattr(msg, "args") and not isinstance(msg, str)
