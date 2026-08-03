"""Shared fixtures for tcm_gui tests."""
from __future__ import annotations

import pytest
from tcm import _constants, schema, policy


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