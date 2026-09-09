"""CLI argument parsing for the tcm processing pipeline.

Hydra handles all config keys (``input.path``, ``input.ids``, ``out.*``, etc.)
natively via ``compose`` — this module only handles pre-Hydra setup.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import logging
import os
import re
import sys
from collections.abc import Callable, Mapping
from functools import wraps
from io import StringIO
from pathlib import Path, PurePath
from types import FrameType, ModuleType
from typing import Any, NamedTuple
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

from tcm import format, policy, schema
from tcm.utils_time_corr import sanitize_time_ranges
from utils.log_init import LoggingStyleAdapter
from utils.logging_config import SafeStringFormatter, upgrade_loggers

# Optional GUI bridge for scan progress — no-op when GUI is not installed.
try:
    from tcm_gui import progress_bridge as _pb
except ImportError:
    _pb = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# argparse compatibility for Python 3.14 — must run before Hydra builds parser
# ---------------------------------------------------------------------------
# Python 3.14's ``_check_help`` calls ``_expand_help`` on every ``help=``
# value.  If help is a non-string, non-iterable (e.g. Hydra's
# ``LazyCompletionHelp``), ``_expand_help`` raises ``TypeError`` which
# ``_check_help`` re-raises as ``ValueError``.  Patch ``_get_help_string``
# to coerce non-string help values to ``str()``.
_orig_get_help_string = argparse.HelpFormatter._get_help_string


def _patched_get_help_string(self, action) -> str | None:
    help_string = _orig_get_help_string(self, action)
    if help_string is not None and not isinstance(help_string, str):
        return str(help_string)
    return help_string


argparse.HelpFormatter._get_help_string = _patched_get_help_string

import hydra  # noqa: E402

from tcm import _constants, config_yaml, paths, stage_ctx, to_omegaconf  # noqa: E402
from utils.init import (  # noqa: E402
    Ex_nothing_done,
    ini2dict,
    standard_error_info,
    this_prog_basename,
    update_cfg_time_ranges,
)  # noqa: E402

lf = LoggingStyleAdapter(__name__)


# Default glob pattern (Windows-first: uppercase I).
_TCM_DEFAULT_GLOB_PATTERN = "*I*.txt"
DEFAULT_GLOB = f"{_constants.RAW_DIR_NAME}/{_TCM_DEFAULT_GLOB_PATTERN}"


# ---------------------------------------------------------------------------
# File handler creation (replaces Hydra's default)
# ---------------------------------------------------------------------------

_UNSAFE_FILENAME: re.Pattern[str] = re.compile(r"[^\w.-]", re.ASCII)


def as_filename(s: str, fallback: str = "_") -> str:
    """Sanitize *s* for use as a filename component.

    Allow-list deletion (``\\w``, ``.``, ``-``), strip trailing dots,
    return *fallback* if empty.

    >>> as_filename('<cfg_from_args>')
    'cfg_from_args'
    """
    clean = _UNSAFE_FILENAME.sub("", s).rstrip(".")
    return clean or fallback


# ANSI/SGR escape sequence: ESC '[' params 'm' — matches colour, bold, italic…
_ANSI_RE: re.Pattern[str] = re.compile("\033\\[[0-9;]*m")

# File format mirrors the console's module.funcName:row style (prefix stripped by
# AnsiStrippedFormatter → SafeStringFormatter)
_FILE_LOG_FMT = "%(asctime)s|%(name)s.%(funcName)s:%(lineno)d|%(levelname)s|%(message)s"
_FILE_LOG_DATEFMT = "%H:%M:%S"


class AnsiStrippedFormatter(SafeStringFormatter):
    """File-safe formatter that strips ANSI colour codes from exception text.

    ``colorlog.ColoredFormatter`` (the *console* handler) colours tracebacks
    on Python ≥ 3.13 via ``traceback.print_exception(colorize=True)``.
    ``logging`` caches the coloured text on ``record.exc_text`` — every later
    handler reuses the cache.  Stripping here keeps console output coloured
    while guaranteeing a clean file regardless of handler order.
    Inherits ``package_prefix`` stripping and clickable tracebacks from
    :class:`utils.logging_config.SafeStringFormatter`.
    """

    def __init__(self, **kwargs):
        super().__init__(package_prefix="tcm.", **kwargs)

    def formatException(self, exc_info) -> str:  # noqa: D401
        return _ANSI_RE.sub("", super().formatException(exc_info))

    def format(self, record: logging.LogRecord) -> str:
        if record.exc_text:
            record.exc_text = _ANSI_RE.sub("", record.exc_text)
        return super().format(record)


def _setup_file_handler(cfg: Mapping[str, Any], enable: bool = True) -> None:
    """Create a FileHandler with the correct filename and ANSI-stripped formatting.

    Called from ``_store`` after Hydra initialization and override merge.
    Replaces Hydra's default file handler (removed from ``colorlog.yaml``)
    so that:

    * the filename is derived from ``program.return_`` (e.g.
      ``processing-cfg_from_args.log`` for config-generation-only runs),
    * ANSI escapes are stripped (``AnsiStrippedFormatter``),
    * stage context is injected (``StageContextFilter``).

    Removes any existing ``FileHandler``s first (e.g. from a previous
    ``call_in_raw_dir`` in the same process — worker re-entry).

    When *enable* is ``False`` (GUI scan, ``Return.CFG_FROM_ARGS`` with
    ``enable_file_logging=False``) the file handler is suppressed — no
    ``cfg_proc/log/*.log`` is created. Console + queue handlers remain.
    """
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.run.dir)
    job_name = hydra_cfg.job.name  # e.g. "processing"

    return_ = cfg.get("program", {}).get("return_")
    if return_ and str(return_) != schema.Return.END:
        filename = f"{job_name}-{as_filename(str(return_))}.log"
    else:
        filename = f"{job_name}.log"

    root = logging.getLogger()
    for h in root.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)

    if not enable:
        lf.debug("File logging disabled for {} (enable=False)", run_dir / filename)
        return

    fh = logging.FileHandler(str(run_dir / filename), encoding="utf-8")
    fh.setFormatter(AnsiStrippedFormatter(fmt=_FILE_LOG_FMT, datefmt=_FILE_LOG_DATEFMT))
    fh.setLevel(logging.DEBUG)
    fh.addFilter(stage_ctx.StageContextFilter())
    root.addHandler(fh)
    upgrade_loggers("tcm.")  # CustomLogger: exception-origin lineno, caller>callee
    # lf.debug("Log file: {}", run_dir / filename)


def parse_data_path(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Extract first positional arg (data path) from ``argv``, handling commas.

    Positional = non-flag, non-``key=value`` argument.  Consecutive positional
    arguments are **joined with commas** to reconstruct paths split by shell
    comma-handling (e.g. ``@i,t-chain`` split by PowerShell).  Only the first
    run of consecutive positionals is consumed as the data path; subsequent
    positionals pass through in ``remaining``.

    :param argv: ``sys.argv``-style list (includes script name at index 0).
    :returns: ``(path_in, remaining_argv)`` where ``remaining_argv`` has the
        consumed positional span stripped.  Returns ``None`` for *path_in*
        when no non-flag, non-``key=value`` argument is found — callers
        that need a concrete path should apply :data:`DEFAULT_GLOB` as
        fallback (see :func:`call_in_raw_dir`).
    """
    remaining = list(argv)

    # Collect consecutive non-flag, non-key=value args as path segments
    segments: list[str] = []
    for i, arg in enumerate(argv[1:], start=1):
        if arg.startswith("-") or "=" in arg:
            if segments:
                break  # path complete — stop at first flag/key=value after segments
            continue  # pre-path flag, skip
        segments.append(arg)

    if not segments:
        return None, remaining  # No positional found — caller applies fallback

    # Remove consumed segments from remaining argv
    for seg in segments:
        remaining.remove(seg)

    # Join with commas: shell split at commas — reinstate them to reconstruct path
    path_in = Path(",".join(segments))
    return path_in, remaining


def inside_repo(path: Path) -> bool:
    """True when *path* resolves inside the protected code project.

    The check runs on the **resolved** path, so an empty/relative input
    (empty ``input.path`` means ``./``) is judged by its real target.
    Covers dev checkouts, envs nested in the repo and the frozen
    distributive alike — see :data:`_constants.REPO_ROOT`.
    """
    resolved = path.resolve()
    return resolved == _constants.REPO_ROOT or _constants.REPO_ROOT in resolved.parents


def safe_cfg_dir(path: Path) -> Path:
    """Create *path* only if it is outside the code project.

    Universal guard for ``cfg_proc/`` subdirectories (``run/``, ``log/``, …).
    Call this instead of ``path.mkdir(parents=True, exist_ok=True)`` for any
    config/log directory derived from data paths.  Last line of defence —
    :func:`call_in_raw_dir` rejects repo-internal anchors up front.

    :raises SystemExit: if *path* resolves inside :data:`_constants.REPO_ROOT`.
    """
    if inside_repo(path):
        resolved = path.resolve()
        print(
            f"Error: refusing to create {resolved} inside code project {_constants.REPO_ROOT}.\n"
            "Move your data outside the project tree.",
            file=sys.stderr,
        )
        sys.exit(1)
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


_result: Any = None  # stores fun(cfg) return — @hydra.main doesn't propagate it


def hydra_main(
    fun,
    config_name: str = "config",
    config_path: str = _constants.BUNDLED_CFG_PKG,
    version_base: str = "1.3",
    overrides: Mapping[str, Any] | None = None,
    *,
    exit_on_error: bool = True,
    enable_file_logging: bool = True,
) -> Any:
    """Dispatch *fun* via Hydra, return its result (``@hydra.main`` swallows returns).

    Two code paths:

    * **No overrides** (default): standard ``@hydra.main`` — composes defaults
      from ConfigStore + ``sys.argv`` CLI overrides.
    * **With overrides**: ``@hydra.main`` composes defaults as usual, then
      a wrapper deep-merges the hierarchical *overrides* dict on top before
      calling *fun*.  CLI ``sys.argv`` overrides still apply (below the dict).

    The decorated function's return value is stored in :data:`_result` and
    returned to the caller — ``@hydra.main`` itself discards return values.

    **Log file naming**: inside ``_store`` (before any logging), the composed
    config's ``program.return_`` is inspected.  If non-default, the file
    handler is created with a suffixed filename (e.g.
    ``processing-cfg_from_args.log``).  See :func:`_setup_file_handler`.

    Note on ``hydra.job.name``: Hydra unwraps the ``@wraps`` chain on ``_store``
    to find the original *fun*, reads ``fun.__module__`` (``"tcm.processing"``),
    and takes the last dotted segment as the job name (``"processing"``).
    See ``hydra/_internal/utils.py:detect_calling_file_or_module_from_task_function``.

    :param fun: task function accepting one ``DictConfig`` argument.
    :param overrides: hierarchical dict to merge on top of composed defaults.
    :returns: whatever *fun* returned (``None`` if it returned nothing).
    """
    global _result
    _result = None  # reset before each dispatch

    # Force Hydra to re-raise original exceptions instead of swallowing them
    # with sys.exit(1). Without this, ``except BaseException`` below catches
    # only a bare ``SystemExit`` and the real traceback is lost.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    caller = caller_info(skip=2)

    @wraps(fun)
    def _store(cfg: DictConfig):
        """Run *fun*, stashing its return in :data:`_result`."""
        global _result

        # Merge overrides FIRST (worker passes program.return_ etc. here)
        if overrides:
            base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            cfg = OmegaConf.merge(base, overrides)

        # Create file handler with correct filename (before any logging)
        _setup_file_handler(cfg, enable=enable_file_logging)

        # Resolve use_h5: user preference × library availability
        policy.init_io(cfg)

        # Banner logging
        _io = policy.io()
        lf.info(
            "{}. {} calls {}.{}{}",
            "TCM",
            caller,
            getattr(fun, "__module__", repr(fun)),
            getattr(fun, "__name__", repr(fun)),
            "" if _io else f" ({_io.reason})",
        )
        lf.debug(
            "{} | argv={} | Working directory: {}",
            Path(sys.argv[0]).stem if sys.argv else "?",
            sys.argv[1:],
            os.getcwd(),
        )

        _result = fun(cfg)
        return _result

    try:
        m_fun = hydra.main(config_name=config_name, config_path=config_path, version_base=version_base)(
            _store
        )
        m_fun()
        return _result
    except SystemExit:
        raise  # Propagate Hydra's sys.exit (e.g. --help, --cfg) without logging
    except BaseException:
        lf.exception("Error. Exiting the entire process")
        if exit_on_error:
            sys.exit(1)
        raise


# Kwargs accepted by :func:`hydra_main` (excluding ``fun``) — everything else
# is treated as an override dict to merge on top of composed defaults.
_HYDRA_MAIN_PARAMS = {
    "config_name",
    "config_path",
    "version_base",
    "overrides",
    "exit_on_error",
    "enable_file_logging",
}


def _build_hydra_argv(data_dir: Path) -> list[str]:
    """Build Hydra argv overrides — only ``--config-dir`` (argparse layer).

    ``input.path`` is injected via :func:`_prepare_overrides` (OmegaConf merge)
    to entirely bypass Hydra's ANTLR override parser.

    ``--config-dir`` targets Hydra's **argparse** layer, which natively handles
    commas, backslashes, colons, parentheses, brackets, braces, equals signs,
    and other ANTLR special characters — no escaping or quoting needed.
    """
    cfg_dir = data_dir / "cfg_proc"
    return ["--config-dir", str(cfg_dir)] if cfg_dir.is_dir() else []


def _prepare_overrides(path_in: Path, overrides: dict) -> dict:
    """Inject ``input.path`` into *overrides* for OmegaConf merge after composition.

    This bypasses Hydra's CLI override parser entirely — the path string
    never passes through the ANTLR grammar, so commas, backslashes, quotes,
    and other special characters are handled correctly.
    """
    return OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.create(overrides), {"input": {"path": path_in.as_posix()}}),
        resolve=True,
    )


def _print_usage_error(data_dir: Path | None, path_in: Path | None) -> None:
    """Print a user-friendly usage error and ``sys.exit(1)``.

    Called by :func:`call_in_raw_dir` when no data path was provided **and** no
    Hydra flag was present, **or** the resolved ``_raw/`` anchor directory does
    not exist on disk.  Replaces the cryptic ``FileNotFoundError`` from
    ``os.chdir`` with a message explaining what the program needs, why, and how
    to supply it (positional path argument, ``--help`` for option list, docs
    link).

    :param data_dir:  directory that was supposed to be the runtime anchor
        (``None`` when the caller detected *no* positional path at all).
    :param path_in:   resolved input path passed to
        :func:`paths.find_dir_raw_absolute` (``None`` when no positional
        path was found and no flag was present).
    """
    if path_in is None:
        head = "No data path was provided as a positional argument."
        body = (
            "The first non-flag, non-'key=value' CLI argument must be a path"
            " to your raw data — a directory, glob, or regex."
        )
    else:
        head = f"Data directory not found: {data_dir}"
        body = (
            f"  looked for '{_constants.RAW_DIR_NAME}/' via: {path_in}\n"
            f"  The '{_constants.RAW_DIR_NAME}/' directory is the anchor for all"
            " relative processing paths (configs, logs, outputs)."
        )
    print(
        f"{'─' * 60}\n"
        f"Error: {head}\n{body}\n"
        f"{'─' * 60}\n"
        f"Usage:\n"
        f"  tcm_proc <path_to_data> [options]\n\n"
        f"Examples:\n"
        f'  tcm_proc "_raw/*i*.txt"\n'
        f"  tcm_proc \"_raw\" 'input.ids=[i01,i_p02]' out.text_path=./results\n"
        f"  tcm_proc --help           ← list all options\n"
        f"  tcm_proc --cfg job        ← show the composed config without running\n\n"
        f"See --help for all config fields, or the user guide:\n"
        f"  README.md\n"
        f"{'─' * 60}",
        file=sys.stderr,
    )
    sys.exit(1)


def _require_nonempty_path(raw: str) -> Path:
    """``Path(raw)`` or :exc:`FileNotFoundError` when *raw* is blank.

    An EMPTY path (the GUI's cleared search field) would mean ``./`` — but
    the process cwd is never a meaningful data anchor:
    :func:`call_in_raw_dir` chdirs to the last ``data_dir``, so ``./``
    would silently rescan the SAME directory (and pollute the code project
    when launched from the repo).  Same verdict as a wrong path — the user
    must point at their data.  Checked on the RAW string: ``Path("")``
    normalizes to ``Path(".")`` and would hide the emptiness.
    """
    if not str(raw).strip():
        raise FileNotFoundError(
            "Empty data path: the current directory is not a data anchor. "
            "Enter the path to your data — a directory, glob, or regex."
        )
    return Path(raw)


def call_in_raw_dir(fun, yaml_path: Path | None = None, enable_file_logging: bool = True, **kwargs) -> Any:
    """Bootstrap CLI → Hydra runtime for a processing entry point.

    :param enable_file_logging: when ``False``, suppress creation of
        ``cfg_proc/log/*.log`` (GUI scan ``CFG_FROM_ARGS``).

    1. **Flag bypass**: when any ``-`` prefixed argument is present in
       ``sys.argv``, the function delegates straight to :func:`hydra_main`
       without consuming positional args or resolving a data directory.
       This covers all Hydra flags (``--help``, ``--cfg job``, ``--info``,
       ``--version``, ``--run``, ``--multirun``, …) — ``parse_data_path``
       is skipped entirely so flag values like ``job`` (argument to ``--cfg``)
       are not mistakenly consumed as path segments.  When **no** positional
       path **and** no flags are present, a user-friendly usage error is
       printed via :func:`_print_usage_error`.
    2. Parses the raw-data path from ``sys.argv``: 1st non-flag, non-``key=value``
       argument treated as path; resolves to nearest ``_raw/`` ancestor via
       :func:`paths.find_dir_raw_absolute`.  If no positional path was given,
       prints a user-friendly usage error (via :func:`_print_usage_error`)
       instead of crashing with a cryptic :exc:`FileNotFoundError`.
    3. Verifies the resolved ``data_dir`` exists on disk; if not, prints a
       user-friendly error explaining what ``_raw/`` is and why it's needed.
    4. Changes the working directory to ``data_dir``.
    5. Injects ``input.path`` into the *overrides* dict via
       :func:`_prepare_overrides` (OmegaConf merge — bypasses Hydra's ANTLR
       override parser, so commas, backslashes, quotes are handled correctly).
       Injects ``--config-dir`` into ``sys.argv`` via
       :func:`_build_hydra_argv` (argparse layer — natively handles ALL
       ANTLR special characters in paths).
    6. Calls *fun* via :func:`hydra_main`.

    Any keyword argument whose name is **not** a :func:`hydra_main` parameter
    (``config_name``, ``config_path``, ``version_base``, ``overrides``) is
    collected into an *overrides* dict and passed to :func:`hydra_main` as
    hierarchical config overrides that layer **on top of** Hydra-composed
    defaults — preserving all ConfigStore defaults for unspecified groups.

    Args:
        fun: A callable accepting one ``DictConfig`` argument
            (typically a ``@hydra.main``-decorated or plain function).
        yaml_path: Optional per-probe run YAML (``cfg_proc/run/<name>.yaml``).
            Loaded via :func:`OmegaConf.load` and used as the **base** for
            dict overrides — explicit ``**kwargs`` win over YAML values.
        **kwargs: ``config_name``, ``config_path``, etc. forwarded to
            :func:`hydra_main`; remaining keys (e.g. ``input={...}``) are
            treated as config-group overrides.

    Returns:
        Whatever *fun* returned (``None`` if it didn't return anything).
        ``@hydra.main`` doesn't propagate return values, so :func:`hydra_main`
        stashes the result in :data:`_result` and returns it here.
        For flag-only invocations (``--help``, ``--cfg``, …): ``None`` (Hydra
        prints info and exits, *fun* is never called).

    Note:
        ConfigStore registration (structured-group dataclasses) must happen
        before ``@hydra.main`` resolves — for the processing pipeline,
        ``tcm.schema`` (imported above) does this at module level.
    """
    # Separate hydra_main params from override dicts.
    hydra_main_kwargs: dict[str, Any] = {"enable_file_logging": enable_file_logging}
    overrides: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _HYDRA_MAIN_PARAMS:
            hydra_main_kwargs[k] = v
        else:
            overrides[k] = v

    # When overrides dict is provided, extract input.path from it for
    # _prepare_overrides injection.  Remove from hydra_main_kwargs to avoid
    # duplicate 'overrides' in the final call.
    if "overrides" in hydra_main_kwargs:
        overrides = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(overrides), hydra_main_kwargs.pop("overrides")),
            resolve=True,
        )

    # Load per-probe YAML as base; explicit kwargs merge on top via OmegaConf.
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(Path(yaml_path))
        overrides_cfg = OmegaConf.create(overrides)
        overrides = OmegaConf.to_container(OmegaConf.merge(yaml_cfg, overrides_cfg), resolve=True)

    # Extract input.path: from overrides dict or from sys.argv.
    try:
        path_in = Path(overrides["input"]["path"])
    except KeyError:
        try:
            path_in = Path(hydra_main_kwargs["overrides"]["input"]["path"])
        except (KeyError, AttributeError, TypeError):
            # If any '-' prefixed arg is present, it's a Hydra/argparse flag.
            # Delegate to Hydra directly without consuming positional args
            # (which might be flag values like --cfg's 'job' argument).
            # Info flags (--help, --cfg, --info, …) print and exit; execution
            # flags (--run, --multirun) proceed with path=None from defaults.
            if any(a.startswith("-") for a in sys.argv[1:]):
                lf.debug("Flags detected — delegating to Hydra directly")
                return hydra_main(fun, overrides=overrides or None, **hydra_main_kwargs)

            path_in, remaining_argv = parse_data_path(sys.argv)
            if path_in is None:
                # No positional path and no flags → user error.
                _print_usage_error(data_dir=None, path_in=None)
            # if str(path_in) in ("", "."):
            #    # Positional ""/"." — Path("") normalizes to "."; same
            #    # verdict as an empty override (see _require_nonempty_path).
            #    raise FileNotFoundError(
            #        "Empty data path: the current directory is not a data anchor. "
            #        "Enter the path to your data — a directory, glob, or regex."
            #    )
        else:
            # overrides dict provided path — keep sys.argv as-is (Worker
            # sets it to [script] + hydra_args before each call).
            remaining_argv = list(sys.argv)
    else:
        remaining_argv = list(sys.argv)

    # Resolve relative inputs against the real cwd so the anchor check
    # below judges the actual target, and overrides-branch paths get the
    # same treatment the argv branch always had.
    path_in = path_in.resolve()

    # Reject repo-internal paths BEFORE anchor resolution — otherwise
    # find_dir_raw_absolute logs a misleading "Not standard input path"
    # warning for what is really a project-tree violation (a pasted repo
    # path, or a relative one while cwd is the repo — dev or distributive).
    # Otherwise cfg_proc/, Hydra outputs and logs would pollute the code
    # project.
    if inside_repo(path_in):
        raise FileNotFoundError(
            f"Not a data directory: '{path_in}' is inside the code project "
            f"({_constants.REPO_ROOT}). "
            "Point input.path at your data outside the project tree."
        )

    # Resolve the nearest `_raw/` ancestor.
    data_dir = paths.find_dir_raw_absolute(path_in)

    # Verify the anchor directory exists — find_dir_raw may return a `_raw/`
    # path whose name matches but doesn't exist on disk (e.g. from DEFAULT_GLOB
    # glob pattern embedded in the path), causing a cryptic FileNotFoundError
    # from os.chdir.  Give a user-friendly message instead.
    if not data_dir.is_dir():
        _print_usage_error(data_dir=data_dir, path_in=path_in)

    # Inject input.path into overrides dict (bypasses Hydra's ANTLR parser).
    overrides = _prepare_overrides(path_in, overrides)

    # Build sys.argv for @hydra.main: script name + searchpath + user options.
    # input.path is in overrides — never touches Hydra's override parser.
    remaining_argv[1:1] = _build_hydra_argv(data_dir)

    os.chdir(data_dir)

    sys.argv = remaining_argv

    return hydra_main(fun, overrides=overrides or None, **hydra_main_kwargs)


def process_loading_yaml(process_fun: Callable, base_cfg, dir_cfgs, cfgs, n_cfgs_existed):
    """Load per-probe run YAMLs, merge on top of *base_cfg*, call process_fun.

    Run YAMLs (``cfg_proc/run/*.yaml``) use ``@package _global_`` — their
    keys live at the root level.  :func:`OmegaConf.merge` applies the YAML
    as overrides on top of *base_cfg*, preserving all groups not mentioned
    in the YAML.

    **Pre-filtering**: before processing, all YAML stems are validated against
    ``input.path`` by comparing canonical identities (see :func:`tcm.format.pcid_key`) —
    permissive across pcid formatting variants and comment separators
    (``i_p05-маг`` ≡ ``INKL_P05_маг``).  Whitespace anywhere in the YAML stem
    → the config is ignored (normally excluded upstream in
    :func:`tcm.config_yaml.get_existed_cfgs` — a lone backup triggers
    regeneration instead of an empty run).  Full matching contract (what is
    ignored and why): :doc:`io_formats — Config-file matching
    </docs/reference/io_formats.md>`.
    One WARNING summarises all skips; the remaining valid configs drive the
    ``[idx/n_cfgs]`` numbering and GUI progress totals.

    :param process_fun: Callable(merged config)
    :param base_cfg: Hydra-composed config (from ``@hydra.main``).
    :param dir_cfgs: absolute path to the directory of run YAML.
    :param cfgs: map of pcid to its config stem from `dir_cfgs` dir
    :param n_cfgs_existed: number of existed configs for logging
    :return: processed_pcids, failed_pcids, last_cfg, collected
        where ``collected`` is ``[(stem, yaml_path_str, result), ...]``
    """
    processed_pcids: list[str] = []
    failed_pcids: list[str] = []
    last_cfg: DictConfig | None = None
    collected: list[tuple[str, str, Any]] = []
    if not cfgs:
        lf.info("No configs to run (available: {}, requested: {})", n_cfgs_existed, len(cfgs))
        return processed_pcids, failed_pcids, last_cfg, collected

    # ── Pre-filter: validate YAML stems against input.path ──────────────
    # Exclude backup copies and missing files BEFORE the processing loop so
    # that [idx/n_cfgs] numbering and GUI progress totals are correct.
    valid_cfgs: dict[str, list[str]] = {}  # pcid → [valid stems]
    loaded_cfgs: dict[str, DictConfig] = {}  # stem → merged DictConfig (cached)
    skipped_cfgs: list[tuple[str, str, str]] = []  # (stem, yaml_core, input_core)
    copied_cfgs: list[str] = []  # whitespace in stem → ignored
    missing_cfgs: list[tuple[str, str]] = []  # (stem, pcid)

    for pcid, stems in cfgs.items():
        valid_stems: list[str] = []
        for stem in stems:
            # Whitespace in the name → ignored (io_formats.md#config-file-matching)
            if any(ch.isspace() for ch in stem):
                copied_cfgs.append(stem)
                continue
            yaml_path = dir_cfgs / f"{stem}.yaml"
            if not yaml_path.is_file():
                missing_cfgs.append((stem, pcid))
                continue
            cfg_dc = OmegaConf.load(yaml_path)
            cfg_dc = OmegaConf.merge(base_cfg, cfg_dc)
            OmegaConf.update(cfg_dc, "_yaml_path", yaml_path, force_add=True)
            yaml_core = stem.rsplit("@", 1)[-1]
            input_core = Path(cfg_dc.input.path).stem.rsplit("@", 1)[-1]
            if format.pcid_key(yaml_core) != format.pcid_key(input_core):
                skipped_cfgs.append((stem, yaml_core, input_core))
                continue
            valid_stems.append(stem)
            loaded_cfgs[stem] = cfg_dc  # cache for processing loop
        if valid_stems:
            valid_cfgs[pcid] = valid_stems

    # Report all skips in one consolidated message
    if skipped_cfgs:
        details = "; ".join(f'"{s}" (stem "{yc}" ≠ path "{ic}")' for s, yc, ic in skipped_cfgs)
        lf.warning(
            "Skipping {} config{} — YAML stem ≠ input.path (manual copy?): {}",
            len(skipped_cfgs),
            "" if len(skipped_cfgs) == 1 else "s",
            details,
        )
    if copied_cfgs:
        lf.warning(
            "Ignoring {} config{} (whitespace in name): {}",
            len(copied_cfgs),
            "" if len(copied_cfgs) == 1 else "s",
            "; ".join(f'"{s}.yaml"' for s in copied_cfgs),
        )
    for stem, pcid in missing_cfgs:
        lf.warning("Config missing for {}: {}", pcid, dir_cfgs / f"{stem}.yaml")
        failed_pcids.append(pcid)

    cfgs = valid_cfgs
    n_cfgs = sum(len(s) for s in cfgs.values())
    n_probes = len(cfgs)

    # ── Process valid configs ───────────────────────────────────────────
    lf.info(
        "Running {} {}{} ({} config{})",
        n_probes,
        "probe" if n_probes == 1 else "probes",
        f" of {n_cfgs_existed} available" if n_probes != n_cfgs_existed else "",
        n_cfgs,
        "" if n_cfgs == 1 else "s",
    )
    stem_idx = 0
    _rt = _pb.get_runtime() if _pb else None
    for probe_i, (pcid, stems) in enumerate(cfgs.items(), start=1):
        for cfg_i, stem in enumerate(stems, start=1):
            stem_idx += 1
            if _rt:
                _rt.progress_stage.set(stem_idx - 1, n_cfgs, f"composing:{stem}")
            # Re-resolve per-stem path: ``yaml_path`` from the pre-filter pass
            # holds only the *last-iterated* stem — leaking it here would log
            # and return the wrong filename for every probe (see GH bug: log
            # "from {yaml_path.name}" and ``collected`` both referenced a
            # single stale variable). Bind fresh from the real stem.
            yaml_path = dir_cfgs / f"{stem}.yaml"
            # Reuse cached config from pre-filter pass
            cfg_dc = loaded_cfgs[stem]

            # Attribute progress to this config so GuiTqdm / stage_desc feed
            # the correct tab cell in ProgressBank.
            if _pb:
                _pb.set_cfg(stem)

            # Set stage context for logging prefix and GUI progress display
            stage_ctx.set_probe(
                pcid,
                probe_idx=probe_i,
                cfg_idx=cfg_i,
                n_probes=n_probes,
                n_cfgs=len(stems),
                stem_idx=stem_idx,
                n_cfgs_total=n_cfgs,
            )

            lf.info('[{}/{}] probe {} (from "{}")', stem_idx, n_cfgs, pcid, yaml_path.name)
            OmegaConf.update(cfg_dc, "_stem_idx", stem_idx, force_add=True)
            OmegaConf.update(cfg_dc, "_n_cfgs", n_cfgs, force_add=True)
            bank = getattr(_pb.get_runtime(), "progress_bank", None) if _pb else None
            ok = False
            try:
                result = process_fun(cfg_dc)
                processed_pcids.append(pcid)
                last_cfg = cfg_dc
                ok = True
                if result is not None:
                    # CFG_FROM_ARGS (scan): result=DictConfig — no data processed
                    collected.append((stem, str(yaml_path), result))
            except FileNotFoundError as e:
                lf.warning(
                    "[{}/{}] {}: source file missing ({}). Delete stale YAML",
                    stem_idx,
                    n_cfgs,
                    pcid,
                    e.filename or e,
                )
                failed_pcids.append(pcid)
            except Exception:
                lf.exception("[{}/{}] Processing failed for {}", stem_idx, n_cfgs, pcid)
                failed_pcids.append(pcid)
            finally:
                if bank:
                    bank.finish(stem, ok=ok)
                gc.collect()  # release previous probe's data before loading next

    # Clear stage context after all configs processed
    stage_ctx.clear()
    return processed_pcids, failed_pcids, last_cfg, collected


def sugar_expand_m(cfg_dict: dict[str, Any]) -> None:
    """Expand shorthand keys in-place within cfg sub-dicts. Sugar expansion

    For every nested ``min`` / ``max`` dict in *cfg_dict* (found recursively
    at the first nesting level only — not deeper), replaces shorthand keys
    (e.g. ``"M"``) with concrete axis keys (``"Mx"``, ``"My"``, ``"Mz"``),
    copying the value into each new key **only when the concrete key is absent**.

    Parameters
    ----------
    cfg_dict
        Configuration dict (e.g. ``cfg.input`` or ``cfg.filter``).
        Modified in-place.
    """

    _SHORTHAND_EXPANSIONS: dict[str, tuple[str, ...]] = {
        "M": ("Mx", "My", "Mz"),  # Shorthand column keys expanded to concrete axis keys
    }

    for sub_key in ("min", "max"):
        sub = cfg_dict.get(sub_key)
        if not isinstance(sub, dict):
            continue
        for short, expansions in _SHORTHAND_EXPANSIONS.items():
            if short not in sub:
                continue
            val = sub[short]
            for concrete in expansions:
                sub.setdefault(concrete, val)
            del sub[short]  # remove shorthand after expansion


def sugar_condense_lim_date(cfg_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    # """initialise program state, convert types."""
    # if cfg.input.path is None:
    #     raise ValueError("input.path must be provided")

    # # Build cfg_in: resolve input config, run sugar merge + M expansion
    # cfg_in = OmegaConf.to_container(cfg.input, resolve=True)
    update_cfg_time_ranges(
        cfg_dict,
        min_date=cfg_dict.pop("min_date", None),
        max_date=cfg_dict.pop("max_date", None),
    )
    # cfg_in["path"]
    # cfg = {
    #     "in": cfg_in,
    #     "filter": OmegaConf.to_container(cfg.filter, resolve=True) if cfg.filter else None,
    #     "proc": OmegaConf.to_container(cfg.proc, resolve=True) if OmegaConf.select(cfg, "proc") else None,
    # }
    # return cfg_dict


def main_init(
    cfg: DictConfig[str, DictConfig[str, Any]],
    program_name: str = "",
    __file__: str | None = None,
) -> Mapping[str, Any]:
    """Convert Hydra ``DictConfig`` to a plain dict with resolved types and paths.

    Centralised post-Hydra bootstrap shared by all entry points
    (``run_processing``, ``run_calibration``, etc.).  Every pipeline MUST call
    this once after Hydra composition and before any downstream consumption.

    Steps (in order):
      1. logs config summary (non-empty values only, debug level)
      2. Early return for sentinel ``program.return_`` values.
      3. ``ini2dict(cfg)`` — converts ``DictConfig`` → plain ``dict``, applying
         name-driven type conversions (see :func:`utils2ini2dict`):

         - ``dt_*`` prefix → ``timedelta`` (suffix becomes the unit, default seconds)
         - ``*_path`` / ``path_*`` → ``pathlib.Path``
         - ``*_date`` / ``*_time`` → ``datetime``
         - ``*_int`` / ``*_integer`` / ``*_index`` → ``int``
         - ``*_float`` → ``float``
         - ``*_bool`` / ``*_b`` → ``bool``
         - ``*_list`` / ``*_names`` → ``list`` (comma-split, recursive fix)
         - ``*_dict`` → ``dict`` (colon-split, recursive fix)
         - ``min_*`` / ``max_*`` / ``fixed_*`` / ``float_*`` (catch-all) → ``float``

      4. Sugar expansion: ``M`` shorthand → ``Mx/My/Mz`` in min/max dicts.
       5. ``min_date``/``max_date`` → merged into ``time_ranges``.
       6. Inverted ``time_ranges`` pairs (``start > end``) → dropped with a
          warning (full-file load fallback; dropped pairs stashed as
          ``_dropped_time_ranges`` for post-load min/max reporting).
       7. ``PathLayout`` resolves output paths (``db_path``, ``not_joined_db_path``,
         ``raw_db_path``, ``text_path``) and writes them into **both** the original
         ``cfg.out`` DictConfig **and** the returned ``cfg_t["out"]`` dict.

    Parameters
    ----------
    cfg
        Hydra-composed top-level ``DictConfig`` (all groups resolved).
    Returns
    -------
    dict
        Plain ``dict`` with type-converted values, sugar expansions applied,
        and output paths resolved.  Downstream code should use this instead of
        the original ``DictConfig``.
    """
    try:
        conf_, ignored_keys = to_omegaconf.to_omegaconf_merge_compatible(cfg, schema.Config)
        conf_ignored = {
            k0: (
                {k1: v1 for k1, v1 in v0.items() if v1}
                if hasattr(v0, "items")
                else str(v0)
                if isinstance(v0, PurePath)
                else v0
            )
            for k0, v0 in cfg.items()
            if k0 in ignored_keys
        }
        ru = config_yaml._ry()
        with StringIO() as s:
            s.writelines("--- Configuration (defaults excluded) ---\n")
            ru.dump(conf_, s)
            s.writelines("--- Additional arguments ---\n")
            ru.dump(conf_ignored, s)
            msg = s.getvalue()
        lf.debug(msg)
    except MissingMandatoryValue as e:
        lf.error(standard_error_info(e))
        raise Ex_nothing_done()

    if not cfg.program.return_:
        print("Can not initialise: provide non empty program.return_ value")
        return cfg
    elif cfg.program.return_ == schema.Return.CFG_FROM_ARGS:
        return cfg

    hydra.verbose = cfg.program.verbose == "DEBUG"
    print("\n" + this_prog_basename(__file__) if __file__ else program_name, end=" started. ")
    try:
        cfg_t = ini2dict(cfg)
    except MissingMandatoryValue as e:
        lf.error(standard_error_info(e))
        raise Ex_nothing_done()
    except Exception:
        lf.exception("startup error")

    sugar_expand_m(cfg_t["input"])
    sugar_condense_lim_date(cfg_t["input"])

    # Drop inverted time_ranges pairs (start > end match nothing downstream —
    # make_range_mask yields all-False → 100% filtered out). Fall back to
    # full-file load for their scope and stash the dropped pairs so
    # run_processing can report file min/max in the warning.
    if tr := cfg_t["input"].get("time_ranges"):
        valid_tr, dropped_tr = sanitize_time_ranges(tr)
        if dropped_tr:
            lf.warning(
                "Inverted time_ranges {} ignored for {} — full-file load to determine min/max",
                dropped_tr,
                cfg_t["input"].get("path", "?"),
            )
            cfg_t["input"]["time_ranges"] = valid_tr
            cfg_t["_dropped_time_ranges"] = [(str(s), str(e)) for s, e in dropped_tr]

    # Resolve output paths: raw_db_path, text_path, not_joined_db_path, db_path.
    # PathLayout operates on the original DictConfig; copy results to cfg_t["out"].
    try:
        paths.PathLayout.from_cfg(cfg.input, cfg.out).apply_to_cfg(cfg.out)
        for entity_name in paths.PathLayout.SCHEMA:
            key = f"{entity_name}_path"
            if (resolved := getattr(cfg.out, key, None)) is not None:
                cfg_t["out"][key] = resolved
    except (ValueError, OSError) as e:
        lf.debug("PathLayout resolution skipped: {}", e)

    return cfg_t


# The caller frame; derive module identity from frame globals.
# For PyInstaller, prefer module name as stable identity; file path may be synthetic or absent.


class Caller(NamedTuple):
    module: str | None
    package: str | None
    file: Path | None
    function: str | None
    line: int


def _path(value: str | Path | None) -> Path | None:
    return Path(value) if value and not str(value).startswith("<") else None


def _frozen_file(name: str | None, is_package: bool) -> Path | None:
    if (base := getattr(sys, "_MEIPASS", None)) and name:
        stem = Path(base).joinpath(*name.split("."))
        return stem / "__init__.pyc" if is_package else stem.with_suffix(".pyc")
    return None


def caller_info(skip: int = 0) -> Caller:
    """
    Return immediate caller info.
    skip=1 when calling from a wrapper/decorator/helper layer.
    """
    frm: FrameType = sys._getframe(skip + 1)
    mod: ModuleType | None = inspect.getmodule(frm)

    name, package = [getattr(mod, key, None) or frm.f_globals.get(key) for key in ("__name__", "__package__")]
    # todo: if function=='_run_code' change: skip-=1
    # or when module='_pydevd_bundle.pydevd_runpy', package='_pydevd_bundle'

    origin = getattr(getattr(mod, "__spec__", None), "origin", None)

    file = next(
        filter(
            None,
            (
                _path(getattr(mod, "__file__", None)),
                _path(origin) if origin not in {None, "built-in", "frozen", "namespace"} else None,
                _frozen_file(name, hasattr(mod, "__path__")),
                _path(frm.f_globals.get("__file__")),
                _path(frm.f_code.co_filename),
            ),
        ),
        None,
    )

    return Caller(name, package, file, frm.f_code.co_name, frm.f_lineno)
