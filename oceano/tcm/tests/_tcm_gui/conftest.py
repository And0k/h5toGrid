"""Shared fixtures for tcm_gui tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _auto_use_h5_set():
    """Set use_h5 based on H5_AVAILABLE before each test.

    Resets to the default after the test, preventing state leakage from tests
    that call ``_constants.use_h5_set(value)`` explicitly.
    """
    from tcm import _constants

    _constants.use_h5_set(True if _constants.H5_AVAILABLE else None)
    yield
    _constants.use_h5_set(None)  # reset to unresolved
    import gc

    gc.collect()  # force HDF5 C-object cleanup while handles are still valid
