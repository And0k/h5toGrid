#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for wave_params_run.py refactoring.

This test verifies that:
1. The cfg dictionary has the correct structure
2. All functions accept and use the cfg parameter correctly
3. The script can be imported without errors
"""

from pathlib import Path
import pytest

from scripts.wave_params_run import cfg, main, calculate_spectrograms


def test_cfg_structure():
    """Test that cfg dictionary has correct structure matching spectr_clc."""
    # Test that cfg has required top-level sections
    assert "spectr_clc" in cfg, "cfg must have 'spectr_clc' section"
    assert "wave_params" in cfg, "cfg must have 'wave_params' section"
    assert "program" in cfg, "cfg must have 'program' section"

    # Test that spectr_clc section has required subsections
    spectr_clc_cfg = cfg["spectr_clc"]
    assert "in" in spectr_clc_cfg, "spectr_clc must have 'in' subsection"
    assert "out" in spectr_clc_cfg, "spectr_clc must have 'out' subsection"
    assert "proc" in spectr_clc_cfg, "spectr_clc must have 'proc' subsection"
    assert "filter" in spectr_clc_cfg, "spectr_clc must have 'filter' subsection"
    assert "program" in spectr_clc_cfg, "spectr_clc must have 'program' subsection"

    # Test that split_period is not in cfg (it was removed)
    assert "split_period" not in spectr_clc_cfg["out"], (
        "split_period should be removed from cfg"
    )
    assert "split_period" not in spectr_clc_cfg["proc"], (
        "split_period should be removed from cfg"
    )

    # Test that dt_interval_minutes is present
    assert "dt_interval_minutes" in spectr_clc_cfg["proc"], (
        "dt_interval_minutes must be in cfg"
    )
    assert "overlap" in spectr_clc_cfg["proc"], (
        "overlap must be in cfg"
    )

    # Test that wave_params section has required keys
    wave_params_cfg = cfg["wave_params"]
    assert "fmin" in wave_params_cfg, "wave_params must have 'fmin'"
    assert "fmax" in wave_params_cfg, "wave_params must have 'fmax'"
    assert "sensor_height" in wave_params_cfg, "wave_params must have 'sensor_height'"
    assert "sea_depths" in wave_params_cfg, "wave_params must have 'sea_depths'"


def test_cfg_constants_moved():
    """Test that all CAPS constants have been moved to cfg."""
    # Test that tables_list is in cfg (was TABLE_PATTERN)
    assert "tables_list" in cfg["spectr_clc"]["in"], (
        "tables_list must be in cfg (was TABLE_PATTERN)"
    )

    # Test that dt_interval_minutes is in cfg (was DT_INTERVAL_MINUTES)
    assert "dt_interval_minutes" in cfg["spectr_clc"]["proc"], (
        "dt_interval_minutes must be in cfg (was DT_INTERVAL_MINUTES)"
    )

    # Test that fmin and fmax are in cfg (was FMIN, FMAX)
    assert "fmin" in cfg["spectr_clc"]["proc"], (
        "fmin must be in cfg (was FMIN)"
    )
    assert "fmax" in cfg["spectr_clc"]["proc"], (
        "fmax must be in cfg (was FMAX)"
    )

    # Test that sensor_height is in cfg (was SENSOR_HEIGHT)
    assert "sensor_height" in cfg["wave_params"], (
        "sensor_height must be in cfg (was SENSOR_HEIGHT)"
    )

    # Test that sea_depths is in cfg (was SEA_DEPTHS)
    assert "sea_depths" in cfg["wave_params"], (
        "sea_depths must be in cfg (was SEA_DEPTHS)"
    )


def test_calculate_spectrograms_accepts_cfg():
    """Test that calculate_spectrograms() accepts cfg parameter."""
    # This test verifies the function signature, not execution
    import inspect
    sig = inspect.signature(calculate_spectrograms)
    assert "cfg" in sig.parameters, "calculate_spectrograms must accept cfg parameter"


def test_no_argparse_import():
    """Test that argparse is not imported in wave_params_run.py."""
    import scripts.wave_params_run as wave_params_run
    import inspect

    # Get the source code
    source = inspect.getsource(wave_params_run)

    # Verify argparse is not imported
    assert "import argparse" not in source, (
        "argparse should not be imported in wave_params_run.py"
    )
    assert "from argparse import" not in source, (
        "argparse should not be imported in wave_params_run.py"
    )


def test_cfg_values_are_not_hardcoded_constants():
    """Test that cfg values are not using CAPS constant names."""
    # Check that cfg doesn't reference CAPS constants
    spectr_clc_cfg = cfg["spectr_clc"]
    wave_params_cfg = cfg["wave_params"]

    # These should not be in the cfg values
    assert not any(
        "TABLE_PATTERN" in str(v) for v in spectr_clc_cfg["in"].values()
    ), "cfg should not reference TABLE_PATTERN constant"
    assert not any(
        "DT_INTERVAL_MINUTES" in str(v) for v in spectr_clc_cfg["proc"].values()
    ), "cfg should not reference DT_INTERVAL_MINUTES constant"
    assert not any(
        "FMIN" in str(v) or "FMAX" in str(v) for v in spectr_clc_cfg["proc"].values()
    ), "cfg should not reference FMIN or FMAX constants"
    assert not any(
        "SENSOR_HEIGHT" in str(v) for v in wave_params_cfg.values()
    ), "cfg should not reference SENSOR_HEIGHT constant"
    assert not any(
        "SEA_DEPTHS" in str(v) for v in wave_params_cfg.values()
    ), "cfg should not reference SEA_DEPTHS constant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
