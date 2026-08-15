"""Shared fixtures for tcm_gui tests."""

from __future__ import annotations

import tkinter as tk

import pytest

from tcm import _constants, policy, schema


# Auto-configure use_h5 for all tests (mirrors policy._io.set(policy.IOPolicy.resolve(cfg)))
@pytest.fixture(autouse=True)
def _auto_use_h5_set():
    """Set use_h5 based on H5_AVAILABLE before each test.

    Resets to the default after the test, preventing state leakage from tests
    that call ``policy._io.set(value)`` explicitly.
    """
    policy._io.set(policy.IOPolicy(schema.UseH5.AUTO, _constants.H5_AVAILABLE))
    yield
    policy._io.set(policy.IOPolicy(schema.UseH5.AUTO, False))  # reset to unavailable
    import gc

    gc.collect()  # force HDF5 C-object cleanup while handles are still valid


@pytest.fixture(scope="session", autouse=True)
def _session_tk_root():
    """Keep the first Tk interpreter alive for the entire test session.

    tksheet caches ``PhotoImage`` objects (``sheet_ops.select_all_image`` etc.)
    in the **first** Tk interpreter that creates a ``Sheet``.  When that
    interpreter is destroyed, the cached images become stale — any later test
    that creates a *new* Tk interpreter and opens a text editor will crash
    with ``TclError: image "pyimageNN" doesn't exist``.

    Creating the session's first Tk root once and keeping it alive until the
    end ensures all tksheet PhotoImage objects remain valid throughout.
    A dummy ``Sheet`` is created in this interpreter to force ``sheet_ops``
    to bind its images here (before any test-specific interpreter exists).
    """
    try:
        root = tk.Tk()
        root.withdraw()
        # Force tksheet's sheet_ops to bind its PhotoImage objects in this
        # interpreter — the one that stays alive for the entire session.
        from tksheet import Sheet

        _init_sh = Sheet(root, show_header=False, show_horizontal_grid=False,
                         show_vertical_grid=False)
        _init_sh.destroy()
        yield root
        root.destroy()
    except tk.TclError:
        yield None
