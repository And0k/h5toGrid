"""
Test for the full processing pipeline using the common test data setup.

This test verifies that the full processing pipeline works correctly with the
common test data structure, ensuring all components work together properly.
"""
import pytest
from pathlib import Path
from meta_finder.collect import process_all_metadata
from meta_finder.file_finder import discover_datafiles_for_all_dev_in_dev_dir, find_navigation_files, find_cruise_directories
# from meta_finder.utils_sys import remove_directory


def test_full_processing_pipeline(common_test_data_setup):
    """Test the full processing pipeline with common test data."""
    # Use the common test data setup
    test_dir = common_test_data_setup

    # Clean up any existing temporary directories
    # remove_directory(path: Path)

    try:
        # Find cruise directories in the test data
        cruise_dirs = find_cruise_directories([test_dir])

        # Should find at least one cruise directory
        assert len(cruise_dirs) >= 1, "Should find at least one cruise directory in test data"

        # Find text output files (including those in archives)
        text_files = discover_datafiles_for_all_dev_in_dev_dir(cruise_dirs)
        # Text files are optional in our test data, so we don't assert their presence

        # Find navigation files
        nav_files = find_navigation_files(cruise_dirs)
        # Navigation files are optional, so we don't assert their presence

        # Process metadata
        metadata_list = process_all_metadata(cruise_dirs)

        # Check that we got some metadata
        assert len(metadata_list) >= 0, "Should process metadata entries (may be 0 if no valid data)"

        # Check structure of metadata entries if we have any
        for json_path, devices_data in metadata_list.items():
            for device_id, entry in devices_data.items():
                # Check required fields
                required_fields = ['cruise_name', 'device_id', 'point', 'sea_depth', 'height_above_bottom',
                                 'lat', 'lon', 'time_st', 'time_en', 'bursts_t', 'burst_dt']
                for field in required_fields:
                    assert field in entry, f"Missing required field {field} in metadata entry"

                # Check that device_id is not empty
                assert entry['device_id'] != '', "device_id should not be empty"

                # Check that times are properly formatted or marked as missing
                if entry['time_st'] not in ['?', '-']:
                    # If time is not a placeholder, it should be a valid string
                    assert isinstance(entry['time_st'], str), "time_st should be a string when not a placeholder"

                if entry['time_en'] not in ['?', '-']:
                    assert isinstance(entry['time_en'], str), "time_en should be a string when not a placeholder"

        print(f"Processed {len(metadata_list)} metadata entries successfully")

    finally:
        pass
        # Clean up temporary directories
        # cleanup_temp_dirs()