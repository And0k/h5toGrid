#!/usr/bin/env python
"""Test script to verify YAML writing with simplified nested dict structure."""

import logging
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_finder.logging_config import setup_logging
from meta_finder import io_info_files
from ruamel.yaml import YAML

# Enable debug logging
setup_logging(__name__, console_level=logging.DEBUG, file_level=logging.DEBUG)
logger = setup_logging()

# Configure YAML parser for reading
yaml_parser = YAML()
yaml_parser.preserve_quotes = True

def test_yaml_writing():
    """Test that YAML writing works correctly with nested dict structure."""

    # Test content with mixed devices (single and multiple intervals)
    test_content = {
        "i10": {
            "0": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"],
            "1": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-22T12:03:00", "2018-10-27T06:47:28"],
        },
        "i01": {
            "0": ["", 50, 0, "⭡", 55.5, 19.0, "2018-10-01 10:00:00", "2018-10-01 12:00:00"],
        },
    }

    logger.info("Test 1: Writing YAML with nested dict structure...")

    # Create temporary file for testing
    temp_path = Path(tempfile.mktemp(suffix='.yaml'))

    try:
        # Write YAML file
        io_info_files.save_to_yaml_format(test_content, temp_path)

        # Read back and verify using ruamel.yaml
        with open(temp_path, 'r', encoding='utf-8') as f:
            written_content = f.read()

        logger.info(f"Written YAML content:\n{written_content}")

        # Parse back using ruamel.yaml to verify it's valid
        with open(temp_path, 'r', encoding='utf-8') as f:
            parsed_content = yaml_parser.load(f)

        # Verify structure by checking parsed content
        assert "i10" in parsed_content, "i10 should be in parsed YAML"
        assert "i01" in parsed_content, "i01 should be in parsed YAML"
        assert "0" in parsed_content["i10"], "Station 0 should be in i10"
        assert "1" in parsed_content["i10"], "Station 1 should be in i10"
        assert parsed_content["i10"]["0"][6] == "2018-10-17 16:30:00", "First interval time should match"
        assert parsed_content["i10"]["1"][6] == "2018-10-22T12:03:00", "Second interval time should match"

        logger.info("✓ Test 1 passed: YAML writing works correctly")

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

    # Test 2: Single interval device
    logger.info("\nTest 2: Writing YAML with single interval device...")
    single_interval_content = {
        "i01": {
            "0": ["", 50, 0, "⭡", 55.5, 19.0, "2018-10-01 10:00:00", "2018-10-01 12:00:00"],
        },
    }

    temp_path = Path(tempfile.mktemp(suffix='.yaml'))

    try:
        # Write YAML file
        io_info_files.save_to_yaml_format(single_interval_content, temp_path)

        # Read back and verify using ruamel.yaml
        with open(temp_path, 'r', encoding='utf-8') as f:
            written_content = f.read()

        logger.info(f"Written YAML content:\n{written_content}")

        # Parse back using ruamel.yaml to verify it's valid
        with open(temp_path, 'r', encoding='utf-8') as f:
            parsed_content = yaml_parser.load(f)

        # Verify structure by checking parsed content
        # Single-interval devices are written as flat lists (not nested {"0": [...]})
        assert "i01" in parsed_content, "i01 should be in parsed YAML"
        assert isinstance(parsed_content["i01"], list), (
            "Single-interval i01 should be a flat list in YAML"
        )
        assert parsed_content["i01"][6] == "2018-10-01 10:00:00", "Time should match"

        logger.info("✓ Test 2 passed: Single interval device YAML writing works correctly")

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

    # Test 3: String station IDs
    logger.info("\nTest 3: Writing YAML with string station IDs...")
    string_station_content = {
        "i10": {
            "A": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"],
            "B": ["", 85, 0, "⭡", 55.8940, 19.0899, "2018-10-22T12:03:00", "2018-10-27T06:47:28"],
        },
    }

    temp_path = Path(tempfile.mktemp(suffix='.yaml'))

    try:
        # Write YAML file
        io_info_files.save_to_yaml_format(string_station_content, temp_path)

        # Read back and verify using ruamel.yaml
        with open(temp_path, 'r', encoding='utf-8') as f:
            written_content = f.read()

        logger.info(f"Written YAML content:\n{written_content}")

        # Parse back using ruamel.yaml to verify it's valid
        with open(temp_path, 'r', encoding='utf-8') as f:
            parsed_content = yaml_parser.load(f)

        # Verify structure by checking parsed content
        assert "i10" in parsed_content, "i10 should be in parsed YAML"
        assert "A" in parsed_content["i10"], "Station A should be in i10"
        assert "B" in parsed_content["i10"], "Station B should be in i10"
        assert parsed_content["i10"]["A"][6] == "2018-10-17 16:30:00", "Station A time should match"
        assert parsed_content["i10"]["B"][6] == "2018-10-22T12:03:00", "Station B time should match"

        logger.info("✓ Test 3 passed: String station ID YAML writing works correctly")

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

    # Test 4: User's actual scenario (from log)
    logger.info("\nTest 4: Writing YAML with user's actual scenario...")
    user_scenario_content = {
        "i2": {"0": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22 12:00:00", "2018-10-27 06:47:30"]},
        "i3": {
            "0": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-17T16:30:00", "2018-10-18T07:15:00", "", "", "через 50м"],
            "1": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22T12:06:12", "2018-10-27T06:46:08"]
        },
        "i6": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22 12:07:35", "2018-10-27 06:46:47"],
        "i7": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22 12:08:15", "2018-10-27 06:46:44"],
        "i8": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22 12:00:00", "2018-10-27 06:47:14"],
        "i9": {
            "0": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"],
            "1": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22 12:06:29", "2018-10-27 06:45:15"]
        },
        "i16": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00", "", "", "слетевшие батарейки"],
        "i10": {
            "0": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-17 16:30:00", "2018-10-18 07:15:00"],
            "1": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22T12:03:00", "2018-10-27T06:47:28"]
        },
        "i11": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22 12:06:25", "2018-10-27 06:47:11"],
        "i12": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22T12:07:05", "2018-10-27 06:45:58"],
        "i19": ["", 85, 0, "⭡", 55.894, 19.0899, "2018-10-22 12:05:17", "2018-10-27 06:47:14"],
    }

    temp_path = Path(tempfile.mktemp(suffix='.yaml'))

    try:
        # Write YAML file
        io_info_files.save_to_yaml_format(user_scenario_content, temp_path)

        # Read back and verify using ruamel.yaml
        with open(temp_path, 'r', encoding='utf-8') as f:
            written_content = f.read()

        logger.info(f"Written YAML content:\n{written_content}")

        # Parse back using ruamel.yaml to verify it's valid
        with open(temp_path, 'r', encoding='utf-8') as f:
            parsed_content = yaml_parser.load(f)

        # Verify structure by checking parsed content
        assert "i2" in parsed_content, "i2 should be in parsed YAML"
        assert "i3" in parsed_content, "i3 should be in parsed YAML"
        assert "i10" in parsed_content, "i10 should be in parsed YAML"
        assert "0" in parsed_content["i10"], "Station 0 should be in i10"
        assert "1" in parsed_content["i10"], "Station 1 should be in i10"
        assert parsed_content["i10"]["0"][6] == "2018-10-17 16:30:00", "First interval time should match"
        assert parsed_content["i10"]["1"][6] == "2018-10-22T12:03:00", "Second interval time should match"
        assert "через 50м" in parsed_content["i3"]["0"][10], (
            "Comment 'через 50м' should be in i3 station 0 at index 10"
        )

        logger.info("✓ Test 4 passed: User's scenario YAML writing works correctly")

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

    logger.info("\n✓ All YAML writing tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_yaml_writing()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
