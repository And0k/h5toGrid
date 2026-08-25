"""Regression: importing meta_finder must never configure logging.

The old `meta_finder.logging_config.setup_logging()` ran at module import time,
hijacking the host app (tcm_gui/hydra): it cleared root handlers, killed colored
console output and spawned an unwanted ``meta/`` dir in cwd.  Modules now use
plain ``logging.getLogger(__name__)`` and propagate to whatever configuration
exists; standalone CLIs call :func:`utils.logging_config.setup_logging` once.
"""
import logging

import pytest


@pytest.fixture
def _root_guard():
    """Snapshot root-logger state and restore it after the test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    root.handlers, root.level = handlers, level


def test_import_leaves_root_and_cwd_untouched(_root_guard, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handlers_before = list(_root_guard.handlers)

    import meta_finder.file_finder  # noqa: F401  (any submodule pulls the package in)
    import meta_finder.io_info_files  # noqa: F401

    assert not (tmp_path / "meta").exists()  # no unwanted log dir
    assert list(_root_guard.handlers) == handlers_before  # host config untouched


def test_records_propagate_to_host_config(_root_guard):
    import meta_finder.parse_data_file_name as m

    lg = logging.getLogger(m.__name__)
    assert lg.propagate  # records reach the host app's root handlers
    assert not lg.handlers  # module owns no private handlers


def test_setup_logging_noop_on_configured_root(_root_guard, tmp_path):
    """utils.logging_config.setup_logging is side-effect-safe on a pre-configured root."""
    from utils.logging_config import setup_logging

    marker = logging.NullHandler()
    _root_guard.addHandler(marker)
    lg = setup_logging("meta_finder.x", log_file_dir=tmp_path / "logs", force=False)
    assert marker in _root_guard.handlers  # existing handler survived
    assert not (tmp_path / "logs").exists()  # no file logging behind host's back


def test_setup_logging_force_reconfigures(_root_guard, tmp_path):
    from utils.logging_config import setup_logging

    marker = logging.NullHandler()
    _root_guard.addHandler(marker)
    setup_logging("meta_finder.y", log_file_dir=tmp_path / "logs", force=True)
    assert marker not in _root_guard.handlers  # cleared when explicitly forced
    assert list((tmp_path / "logs").glob("*.log"))  # file created on demand
