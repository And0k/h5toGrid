#!/usr/bin/env python
"""Test script to verify YAML writing fix for devices with intervals key."""

import logging
import sys
from pathlib import Path

from meta_finder.logging_config import setup_logging
from meta_finder import create_info_files

setup_logging(__name__, console_level=logging.DEBUG, file_level=logging.DEBUG)
logger = setup_logging()


def test_yaml_intervals_fix():
    """Test that devices with intervals key are formatted correctly."""
    test_content = {
        "i10": {
            "point": "",
            "sea_depth": 85,
            "height_above_bottom": 0,
            "modification_symbol": "⭡",
            "lat": 55.8940,
            "lon": 19.0899,
            "intervals": [
                {
                    "time_st": "2018-10-17 16:30:00",
                    "time_en": "2018-10-18 07:15:00",
                    "burst_dt": None,
                    "bursts_t": None,
                    "coef_date": None,
                    "time_raw_st": None,
                    "time_raw_en": None,
                },
                {
                    "time_st": "2018-10-22T12:03:00",
                    "time_en": "2018-10-27T06:47:28",
                    "burst_dt": None,
                    "bursts_t": None,
                    "coef_date": None,
                    "time_raw_st": None,
                    "time_raw_en": None,
                },
            ],
            "data_paths": {},
            "cruise": None,
            "coef_date": None,
            "time_raw_st": None,
            "time_raw_en": None,
            "combined_comments": None,
        }
    }

    # Test _format_for_devices_meta_file
    logger.info("Testing _format_for_devices_meta_file...")
    formatted = create_info_files._format_for_devices_meta_file(test_content)
    logger.info(f"Formatted content: {formatted}")

    # Verify that device with intervals is passed through as nested dict
    assert "i10" in formatted, "Device i10 should be in formatted content"
    assert isinstance(formatted["i10"], dict), "Device i10 should be a dict (not a list)"
    assert "intervals" in formatted["i10"], "Device i10 should have intervals key"
    assert len(formatted["i10"]["intervals"]) == 2, "Device i10 should have 2 intervals"
    logger.info("✓ _format_for_devices_meta_file correctly handles devices with intervals")

    logger.info("\n✓ All tests passed!")


if __name__ == "__main__":
    try:
        test_yaml_intervals_fix()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
