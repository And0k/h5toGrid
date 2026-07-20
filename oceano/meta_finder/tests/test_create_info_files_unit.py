#!/usr/bin/env python
"""
Unit tests for create_info_files.py functionality.
"""

from pathlib import Path
import pytest

# Import the functions to test
from meta_finder.io_info_files import all_vals_empty
import meta_finder.config as config
from meta_finder.file_finder import discover_datafiles_for_all_dev_in_dev_dir

x6questions = ["?"] * 6

class TestIsAllQuestionMarks:
    """Test the all_vals_empty function."""

    @pytest.mark.parametrize(
        "content,expected_result,comment",
        [
            ({"device1": ["?", "?", "?"], "device2": ["?", "?", "?"]}, True, "all values are question marks"),
            (
                {"device1": ["?", "value", "?"], "device2": ["?", "?", "?"]},
                False,
                "mixed values with one non-question mark",
            ),
            (
                {"device1": ["valid", "value", "here"], "device2": ["another", "valid", "value"]},
                False,
                "all values are valid",
            ),
            (
                {"?": x6questions},
                True,
                "single placeholder device with all question marks",
            ),
            (
                {"i01": x6questions + ["2023-01-01", "2023-01-02", "?", "?"]},
                False,
                "device with valid time values",
            ),
        ],
        ids=[
            "all_question_marks",
            "mixed_values",
            "all_valid_values",
            "placeholder_device",
            "valid_time_values",
        ],
    )
    def test_is_all_question_marks_various_inputs(self, content, expected_result, comment):
        """Test all_vals_empty with various input scenarios."""
        result = all_vals_empty(content)
        assert result == expected_result, f"Expected {expected_result} for {comment} but got {result}"


def test_discover_all_devices_integration():
    """Integration test for discovering all devices in a directory."""
    # Use existing test data directory structure
    test_device_dir = Path("test_data") / "test_device_directory_logic" / "230507_ABP53@i" / "230508_inclinometer@i03"

    # Only run if the test directory exists
    if test_device_dir.exists():
        result = discover_datafiles_for_all_dev_in_dev_dir(test_device_dir)

        # Result should be a dictionary mapping device IDs to lists of file tuples
        assert isinstance(result, dict), "Expected discover_all_devices to return a dictionary"

        # If there are devices, the keys should be strings (device IDs) and values should be lists of file tuples
        for device_id, file_list in result.items():
            assert isinstance(device_id, str), "Expected each device ID to be a string"
            assert isinstance(file_list, list), f"Expected file list for device {device_id} to be a list"
            # Each item in the file list should be a tuple of (Path, PurePosixPath) or similar
            for file_tuple in file_list:
                assert isinstance(file_tuple, tuple), f"Expected each file entry to be a tuple, got {type(file_tuple)}"
                assert len(file_tuple) == 2, f"Expected each file tuple to have 2 elements, got {len(file_tuple)}"
    else:
        # Skip if the test directory doesn't exist
        pytest.skip(f"Test directory {test_device_dir} does not exist, skipping integration test")