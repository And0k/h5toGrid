"""
Handler/formatter layer of unified monorepo logging.

Split of purpose with :mod:`utils.log_init` (message-style adapters/filters used
at call sites): this module configures handlers, formatters and the root logger
only.  It cooperates with hydra ``dictConfig`` setups instead of fighting them:
:func:`setup_logging` attaches console/file handlers **only** when the root
logger is not configured yet (unless ``force=True``) — importing a library
module can never hijack e.g. tcm_gui's colorlog console or queue handlers.

Works with or without hydra; declarative per-subproject YAML configs are loaded
via :func:`load_yaml_logging`.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import traceback
from datetime import datetime
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Optional

# Caller-package info cached for CustomLogger/formatter (set by setup_logging)
_pkg_dir: Optional[Path] = None
_pkg_prefix: Optional[str] = ""


class SafeStringFormatter(logging.Formatter):
    """Sanitizes messages/args to UTF-8-safe text, strips own-package prefix from
    the record name, supports ``caller>callee`` display for same-module calls
    (record attr ``caller_function`` set by :class:`CustomLogger`) and
    VSCode-clickable tracebacks (``File "path":lineno``).
    """

    def __init__(
        self,
        fmt: str,
        datefmt: Optional[str] = None,
        package_prefix: str = "",
        caller_callee: bool = False,
        clickable_tb: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.package_prefix = package_prefix
        self.caller_callee = caller_callee
        self.clickable_tb = clickable_tb

    @staticmethod
    def sanitize_message(msg: Any) -> Any:
        """Replace characters unencodable in UTF-8 so logging never crashes."""
        if isinstance(msg, str):
            try:
                msg.encode("utf-8")
                return msg
            except UnicodeEncodeError:
                return msg.encode("utf-8", errors="replace").decode("utf-8")
        return msg

    def format(self, record: logging.LogRecord) -> str:
        record.msg = self.sanitize_message(record.msg)
        if record.args:  # sanitize string args only; keep non-strings (e.g. ints for %d) intact
            record.args = tuple(self.sanitize_message(a) if isinstance(a, str) else a for a in record.args)

        original_name, original_funcName = record.name, record.funcName
        record.name = original_name.replace(self.package_prefix, "") if self.package_prefix else original_name

        if self.caller_callee and getattr(record, "caller_function", None):
            # Same-module call: show "caller>callee" instead of duplicated funcName
            record.name = f"{record.caller_function}>{record.funcName}"
            record.funcName = ""

        swapped_lineno = None
        if hasattr(record, "_original_lineno"):  # exception-origin line from CustomLogger
            swapped_lineno, record.lineno = record.lineno, record._original_lineno

        formatted = super().format(record)

        record.name, record.funcName = original_name, original_funcName
        if swapped_lineno is not None:
            record.lineno = swapped_lineno
        return formatted

    def formatException(self, exc_info) -> str:
        tb_lines = traceback.format_exception(*exc_info)
        if not self.clickable_tb:
            return "".join(tb_lines)
        return "".join(re.sub(r'File "([^"]+)", line (\d+)', r'File "\1:\2"', ln) for ln in tb_lines)


class CustomLogger(logging.Logger):
    """Logger reporting real call sites: skips logging-internals frames, defaults
    ``stacklevel`` to 2, shows exception-origin lineno for ``exc_info`` records and
    adds same-module ``caller_function`` attr consumed by :class:`SafeStringFormatter`.
    """

    @staticmethod
    def _in_pkg(frame) -> bool:
        filename = Path(frame.f_code.co_filename)
        if _pkg_dir is None:
            return False
        try:
            if filename.resolve().is_relative_to(_pkg_dir.resolve()):
                return True
        except (ValueError, OSError):
            return False
        return frame.f_globals.get("__name__", "").startswith(_pkg_prefix or "\x00")

    def findCaller(self, stack_info=False, stacklevel=1):
        # Walk up to the first package frame — the actual callee (user code),
        # skipping importlib/logging internals and this module itself.
        frame = sys._getframe()
        while frame:
            if self._in_pkg(frame):
                if Path(frame.f_code.co_filename).name == "logging_config.py":
                    frame = frame.f_back
                    continue
                co = frame.f_code
                return (co.co_filename, frame.f_lineno, co.co_name, None)
            frame = frame.f_back
        return super().findCaller(stack_info, stacklevel)

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, stack_info=False):
        extra = dict(extra) if extra else {}
        if _pkg_dir is not None and not extra.get("caller_function"):
            frame = sys._getframe(2)  # skip makeRecord and its caller
            while frame:
                if self._in_pkg(frame):
                    if Path(frame.f_code.co_filename).name == "logging_config.py":
                        frame = frame.f_back
                        continue
                    caller = frame.f_back
                    if (
                        caller
                        and Path(caller.f_code.co_filename).resolve()
                        == Path(frame.f_code.co_filename).resolve()
                    ):
                        extra["caller_function"] = caller.f_code.co_name
                    break
                frame = frame.f_back
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, stack_info)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if exc_info:
            try:  # report where the exception was raised, not where it was logged
                _, _, exc_tb = sys.exc_info() if exc_info is True else exc_info
                tb = exc_tb
                while tb.tb_next:
                    tb = tb.tb_next
                extra = {**(extra or {}), "_original_lineno": tb.tb_lineno}
            except Exception:
                pass
        super()._log(level, msg, args, exc_info, extra, stack_info, 2 if stacklevel == 1 else stacklevel)


def get_formatter(
    name: bool = True,
    funcName: bool = True,
    datefmt: Optional[str] = None,
    msecs: bool = False,
    package_prefix: str = "",
    caller_callee: bool = False,
    clickable_tb: bool = True,
) -> SafeStringFormatter:
    """Formatter assembling ``asctime[.msecs] [name.]funcName:lineno LEVEL: message``."""
    return SafeStringFormatter(
        "".join(
            [
                "%(asctime)s",
                ".%(msecs)03d" if msecs else "",
                " %(name)s" if name else "",
                ".%(funcName)s:%(lineno)d" if funcName else "",
                " %(levelname)s: %(message)s",
            ]
        ),
        datefmt=datefmt,
        package_prefix=package_prefix,
        caller_callee=caller_callee,
        clickable_tb=clickable_tb,
    )


_LOG_COLORS = {"DEBUG": "purple", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red"}

try:
    from colorlog import ColoredFormatter as _ColorlogFormatter  # noqa: PLC0415
except ImportError:  # utils stays dependency-free; plain fallback below
    _ColorlogFormatter = None

if _ColorlogFormatter is not None:

    class ColoredSafeFormatter(SafeStringFormatter, _ColorlogFormatter):
        """:class:`SafeStringFormatter` (sanitize/prefix-strip/clickable-tracebacks)
        composed with colorlog colouring. MRO routes ``format`` through both."""

        def __init__(
            self, fmt: str, datefmt: Optional[str] = None, log_colors: Optional[dict] = None, **safe_opts
        ):
            SafeStringFormatter.__init__(self, fmt, datefmt, **safe_opts)
            _ColorlogFormatter.__init__(self, fmt, datefmt=datefmt, log_colors=log_colors or _LOG_COLORS)

else:
    ColoredSafeFormatter = None


def colored_formatter(
    fmt: str = "%(cyan)s%(asctime)s%(reset)s|%(blue)s%(funcName)s%(reset)s|%(log_color)s%(message)s",
    datefmt: str = "%H:%M:%S",
    log_colors: Optional[dict] = None,
    **safe_opts,
):
    """dictConfig-ready factory ('()' key) combining colorlog colouring with
    :class:`SafeStringFormatter` options (``package_prefix``, ``caller_callee``,
    ``clickable_tb``); plain fallback when colorlog is absent."""
    if ColoredSafeFormatter is None:
        return get_formatter(datefmt=datefmt, **safe_opts)
    return ColoredSafeFormatter(fmt, datefmt, log_colors=log_colors, **safe_opts)


def upgrade_loggers(prefix: str = "tcm.") -> int:
    """Retro-install :class:`CustomLogger` onto already-created loggers under *prefix*.

    Needed because ``dictConfig`` cannot set logger classes while subprojects create
    their loggers at import time — before Hydra applies ``job_logging``.  Enables
    exception-origin lineno reporting and same-module ``caller>callee`` display for
    the declarative YAML path.  Call **after** all target loggers exist (in tcm:
    from ``cli._setup_file_handler``, i.e. once every pipeline module is imported).
    Returns count upgraded.
    """
    global _pkg_dir, _pkg_prefix  # give CustomLogger the package context it needs
    if _pkg_dir is None:
        f = sys._getframe(1)
        _pkg_dir = Path(f.f_code.co_filename).parent
        _pkg_prefix = prefix

    candidates = [
        lg
        for name, lg in logging.Logger.manager.loggerDict.items()
        if name.startswith(prefix) and isinstance(lg, logging.Logger) and type(lg) is not CustomLogger
    ]
    for lg in candidates:
        lg.__class__ = CustomLogger  # stateless swap — safe on live instances
    return len(candidates)


def load_yaml_logging(path: str | Path) -> None:
    """Apply a declarative logging YAML (e.g. hydra job_logging snippet) via dictConfig."""
    from ruamel.yaml import YAML  # noqa: PLC0415

    def plain(obj):  # ruamel Commented* wrappers are rejected by dictConfig
        if isinstance(obj, dict):
            return {k: plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [plain(v) for v in obj]
        return obj

    cfg = plain(YAML(typ="safe").load(Path(path).read_text(encoding="utf-8")))
    dictConfig(cfg)


def add_file_handler(
    logger,
    dir: Optional[Path] = None,
    name: str = "{date}.log",
    level: int = logging.DEBUG,
) -> Path:
    """Attach a UTF-8 FileHandler to *logger*, replacing previous file handlers only
    (console handlers stay untouched).

    :param name: file name; ``{date}`` expands to start time ``%Y%m%d_%H%M``
    :return: created log file path
    """
    logger = getattr(logger, "logger", logger)  # accept LoggingStyleAdapter input
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    file = Path(name.format(date=f"{datetime.now():%Y%m%d_%H%M}"))
    if dir is not None:
        file = Path(dir) / file
    h = logging.FileHandler(file, encoding="utf-8")
    h.setLevel(level)
    h.setFormatter(get_formatter(datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return file


def setup_logging(
    name: Optional[str] = None,
    log_level: Optional[int] = None,
    log_file_dir: Optional[str | Path] = "logs",
    log_file_sfx: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    console_format_args: Optional[dict] = None,
    file_format_args: Optional[dict] = None,
    force: bool = False,
    use_custom_logger: bool = False,
    package_prefix: Optional[str] = None,
) -> logging.Logger:
    """Configure root handlers (console + optional timestamped file) and return *name* logger.

    Side-effect-safe by contract: when the root logger already has handlers
    (hydra/tcm_gui/``basicConfig`` own it) nothing is attached or cleared unless
    ``force=True`` — the returned logger just propagates to the existing config.

    :param name: logger name; ``None`` resolves to the calling module's ``__name__``
    :param log_file_dir: directory for ``{ts}_{sfx|name}.log`` (created on demand);
        ``None`` disables file logging. Relative → resolved against cwd.
    :param package_prefix: dotted prefix stripped from displayed record names;
        ``None`` auto-detects from the calling module's top-level package
    :param console_format_args / file_format_args: :func:`get_formatter` kwargs overrides
    :param use_custom_logger: install :class:`CustomLogger` (accurate call sites)
    """
    global _pkg_dir, _pkg_prefix

    caller_frame = sys._getframe(1)
    src = Path(caller_frame.f_code.co_filename)
    caller_mod: str = caller_frame.f_globals.get("__name__", "")
    if name is None:
        name = caller_mod
    if package_prefix is None:
        package_prefix = f"{caller_mod.split('.')[0]}." if caller_mod else ""
    _pkg_dir, _pkg_prefix = src.parent, package_prefix

    root = logging.getLogger()
    if not force and root.handlers:  # configured elsewhere (hydra/GUI) — propagate only
        return logging.getLogger(name)

    if use_custom_logger:
        logging.setLoggerClass(CustomLogger)
    if log_level is None:
        log_level = min(console_level, file_level)
    root.setLevel(log_level)
    for h in root.handlers[:]:
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(
        get_formatter(
            caller_callee=use_custom_logger,
            **{"datefmt": "%H:%M:%S", "package_prefix": package_prefix, **(console_format_args or {})},
        )
    )
    root.addHandler(console)

    if log_file_dir is not None:
        log_file_dir = Path(log_file_dir)
        if not log_file_dir.is_absolute():
            log_file_dir = Path.cwd() / log_file_dir
        log_file_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%y%m%d_%H%M")
        fh = logging.FileHandler(log_file_dir / f"{ts}_{log_file_sfx or name}.log", encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(
            get_formatter(
                **{"datefmt": "%H:%M:%S", "package_prefix": package_prefix}, **(file_format_args or {})
            )
        )
        root.addHandler(fh)

    logger = logging.getLogger(name)  # created with the custom class while it is still installed
    if use_custom_logger:
        logging.setLoggerClass(logging.Logger)  # do not leak the custom class globally
    return logger


def init_logging(logger="", log_file=None, level_file="INFO", level_console=None):
    """Legacy basicConfig-based file+console init (moved verbatim-ish from log_init.py).

    Logs next to the script by default (``log/&{script}.log``, '&' marks autoname),
    forces UTF-8 everywhere, console gets a simple ``%(message)s`` handler at
    ``level_console`` regardless of the file level.
    """
    from .init import dir_create_if_need, this_prog_basename  # noqa: PLC0415

    if logger is None:
        logger = sys._getframe(1).f_back.f_globals["__name__"]  # caller's name
    elif isinstance(logger, str) and __name__ == "__main__":
        logger = ""

    if log_file:
        if not Path(log_file).is_absolute():
            log_file = Path(sys.argv[0]).parent / log_file
    else:
        flogD = Path(sys.argv[0]).parent / "log"
        dir_create_if_need(str(flogD))
        log_file = flogD / f"&{this_prog_basename()}.log"

    lg = logging.getLogger(logger) if isinstance(logger, str) else logger
    lg.handlers.clear()

    filename = Path(log_file)
    if not filename.parent.exists():  # bad target dir → fall back to module-local logs/
        bad_path, filename = True, Path(__file__).parent / "logs" / filename.name
        filename.parent.mkdir(parents=True, exist_ok=True)
    else:
        bad_path = False

    if not lg.root.hasHandlers():
        # UTF-8 forced: system encoding (e.g. cp1251) can't encode e.g. ×
        logging.basicConfig(
            filename=filename, format="%(asctime)s %(message)s", level=level_file, encoding="utf-8"
        )
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")  # PyInstaller-frozen consoles
        console = logging.StreamHandler()
        console.setLevel(level_console or "INFO")  # default INFO regardless of file level
        console.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(console)

    if bad_path:
        lg.warning("Bad log path: %s! Using default dir: %s", log_file, filename)
    return lg
