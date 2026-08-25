"""Unit tests for utils.logging_config extras used by tcm's declarative YAML."""

import logging
import sys

import pytest

from utils.logging_config import ColoredSafeFormatter, CustomLogger, colored_formatter, upgrade_loggers


@pytest.fixture(autouse=True)
def _cleanup_pkg_ctx():
    """Reset module globals that upgrade_loggers caches across tests."""
    import utils.logging_config as lc

    yield
    lc._pkg_dir, lc._pkg_prefix = None, ""


def test_colored_formatter_strips_prefix_shows_module_func_lineno():
    fmtr = colored_formatter(
        fmt="%(asctime)s|%(name)s.%(funcName)s:%(lineno)d|%(levelname)s|%(message)s",
        datefmt="%H:%M:%S",
        package_prefix="tcm.",
    )
    rec = logging.LogRecord("tcm.gui._help", logging.DEBUG, "p.py", 425, "Loaded %d entries", (85,), None)
    rec.funcName = "_load"
    out = fmtr.format(rec)
    assert "gui._help._load:425" in out  # prefix stripped, location shown
    assert not out.startswith("tcm.")


def test_colored_formatter_is_colorlog_composed():
    if ColoredSafeFormatter is None:
        pytest.skip("colorlog not installed")
    fmtr = colored_formatter(fmt="%(log_color)s%(message)s", package_prefix="")
    assert isinstance(fmtr, ColoredSafeFormatter)
    rec = logging.LogRecord("m", logging.ERROR, "p.py", 1, "err", (), None)
    assert "\x1b[" in fmtr.format(rec)  # ANSI colour codes present


def test_upgrade_loggers_reclasses_existing(caplog):
    lg = logging.getLogger("tcm.upgrade_probe")
    assert type(lg) is not CustomLogger  # plain stdlib logger created at import time

    n = upgrade_loggers("tcm.")

    assert n >= 1
    assert type(lg) is CustomLogger


def test_custom_logger_reports_exception_origin_lineno():
    lg = logging.getLogger("tcm.lineno_probe")  # must exist before upgrade (see docstring)
    records: list[logging.LogRecord] = []

    class Cap(logging.Handler):
        def emit(self, record):
            records.append(record)

    lg.handlers[:] = [Cap()]
    lg.setLevel(logging.DEBUG)
    upgrade_loggers("tcm.")
    try:
        raise ValueError("boom")
    except ValueError:
        origin_lineno = sys.exc_info()[2].tb_lineno  # the ``raise`` line above
        lg.error("failed", exc_info=True)

    assert type(lg) is CustomLogger
    assert getattr(records[0], "_original_lineno", None) == origin_lineno
