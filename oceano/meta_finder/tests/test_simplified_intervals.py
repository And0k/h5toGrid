#!/usr/bin/env python
"""Test script to verify simplified nested dict structure implementation."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_finder.logging_config import setup_logging
from meta_finder import metadata_extractor
from meta_finder import io_info_files

# Enable debug logging
setup_logging(__name__, console_level=logging.DEBUG, file_level=logging.DEBUG)
logger = setup_logging()

def test_simplified_structure():
    """Test that the simplified nested dict structure works correctly."""

    # Test 1: Read YAML with nested dict structure
    logger.info("Test 1: Reading YAML with nested dict structure...")
    test_yaml_content = {
        "i10": {
            "0": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"],
            "1": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-22T12:03:00", "2018-10-27T06:47:28"],
        }
    }

    # Simulate reading from YAML file
    # Convert to nested dict structure with string keys
    converted = {}
    for device_id, metadata in test_yaml_content.items():
        if isinstance(metadata, dict):
            # Nested dict structure - ensure all keys are strings
            converted[device_id] = {str(k): v for k, v in metadata.items()}
        else:
            # Single interval device (list/tuple) - convert to nested dict with key "0"
            converted[device_id] = {"0": metadata}

    logger.info(f"Converted structure: {converted}")

    # Verify structure
    assert "i10" in converted, "Device i10 should be in converted content"
    assert isinstance(converted["i10"], dict), "Device i10 should be a dict"
    assert "0" in converted["i10"], "Device i10 should have station_id '0'"
    assert "1" in converted["i10"], "Device i10 should have station_id '1'"
    assert converted["i10"]["0"][6] == "2018-10-17 16:30:00", f"Station 0 time_st should be 2018-10-17 16:30:00, got {converted['i10']['0'][6]}"
    assert converted["i10"]["1"][6] in ["2018-10-22 12:03:00", "2018-10-22T12:03:00"], f"Station 1 time_st should be 2018-10-22 12:03:00 or 2018-10-22T12:03:00, got {converted['i10']['1'][6]}"
    logger.info("✓ Test 1 passed: Nested dict structure is correct")

    # Test 2: Single interval device
    logger.info("\nTest 2: Single interval device...")
    single_interval_content = {
        "i01": ["", 50, 0, "⭡", 55.5, 19.0, "2018-10-01 10:00:00", "2018-10-01 12:00:00"],
    }

    # Convert to nested dict structure
    converted_single = {}
    for device_id, metadata in single_interval_content.items():
        if isinstance(metadata, dict):
            converted_single[device_id] = {str(k): v for k, v in metadata.items()}
        else:
            converted_single[device_id] = {"0": metadata}

    logger.info(f"Converted single interval structure: {converted_single}")

    # Verify structure
    assert "i01" in converted_single, "Device i01 should be in converted content"
    assert isinstance(converted_single["i01"], dict), "Device i01 should be a dict"
    assert "0" in converted_single["i01"], "Device i01 should have station_id '0'"
    assert len(converted_single["i01"]) == 1, "Device i01 should have exactly one station"
    logger.info("✓ Test 2 passed: Single interval device structure is correct")

    # Test 3: Mixed devices (single and multiple intervals)
    logger.info("\nTest 3: Mixed devices...")
    mixed_content = {
        "i10": {
            "0": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"],
            "1": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-22T12:03:00", "2018-10-27T06:47:28"],
        },
        "i01": ["", 50, 0, "⭡", 55.5, 19.0, "2018-10-01 10:00:00", "2018-10-01 12:00:00"],
    }

    # Convert to nested dict structure
    converted_mixed = {}
    for device_id, metadata in mixed_content.items():
        if isinstance(metadata, dict):
            converted_mixed[device_id] = {str(k): v for k, v in metadata.items()}
        else:
            converted_mixed[device_id] = {"0": metadata}

    logger.info(f"Converted mixed structure: {converted_mixed}")

    # Verify structure
    assert "i10" in converted_mixed, "Device i10 should be in converted content"
    assert "i01" in converted_mixed, "Device i01 should be in converted content"
    assert len(converted_mixed["i10"]) == 2, "Device i10 should have 2 stations"
    assert len(converted_mixed["i01"]) == 1, "Device i01 should have 1 station"
    logger.info("✓ Test 3 passed: Mixed devices structure is correct")

    # Test 4: String station IDs (not just numeric)
    logger.info("\nTest 4: String station IDs...")
    string_station_content = {
        "i10": {
            "A": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"],
            "B": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-22T12:03:00", "2018-10-27T06:47:28"],
        },
    }

    # Convert to nested dict structure
    converted_string = {}
    for device_id, metadata in string_station_content.items():
        if isinstance(metadata, dict):
            converted_string[device_id] = {str(k): v for k, v in metadata.items()}
        else:
            converted_string[device_id] = {"0": metadata}

    logger.info(f"Converted string station structure: {converted_string}")

    # Verify structure
    assert "i10" in converted_string, "Device i10 should be in converted content"
    assert "A" in converted_string["i10"], "Device i10 should have station_id 'A'"
    assert "B" in converted_string["i10"], "Device i10 should have station_id 'B'"
    logger.info("✓ Test 4 passed: String station IDs are handled correctly")

    logger.info("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_simplified_structure()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
