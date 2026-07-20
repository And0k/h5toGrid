import os
from unittest.mock import patch
from pathlib import Path
from pathlib import PurePosixPath
import pytest

from meta_finder.file_finder import find_cruise_directories, discover_datafiles_for_all_dev_in_dev_dir

# Global constants for test file names
TEXT_FILE_NAME = '230508_1551bin2s@i03.tsv'
INFO_FILE_NAME = 'info_devices.json'
COMBINED_DIR_NAME = '230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6'


def test_find_cruise_directories_with_combined_paths(common_test_data_setup):
    """Test that find_cruise_directories finds combined paths."""
    # Test that find_cruise_directories finds the combined path
    cruise_dirs = find_cruise_directories([common_test_data_setup])
    expected_dir = common_test_data_setup / COMBINED_DIR_NAME
    assert expected_dir in cruise_dirs, "Combined directory should be found in cruise directories"


def test_find_text_output_files_with_combined_paths(common_test_data_setup):
    """Test that discover_datafiles_for_all_dev_in_dev_dir finds files in combined paths."""
    # Test that discover_datafiles_for_all_dev_in_dev_dir finds the file
    cruise_dirs = find_cruise_directories([common_test_data_setup])
    text_files = {}
    for cruise_dir in cruise_dirs:
        # Need to find device directories first, then look for text output in those
        from meta_finder.file_finder import find_device_dirs
        device_dirs = find_device_dirs(cruise_dir)
        for device_dir in device_dirs:
            text_output_dirs = list(device_dir.glob("text_output*"))
            for text_output_dir in text_output_dirs:
                text_files.update(discover_datafiles_for_all_dev_in_dev_dir(text_output_dir))

    expected_file = PurePosixPath(TEXT_FILE_NAME)
    found = False
    for dir_path, file_names in text_files.items():
        if expected_file in file_names:
            found = True
            break
    assert found, "Text file should be found in combined directory"


def test_process_all_metadata_with_combined_paths(common_test_data_setup):
    """Test that process_all_metadata processes combined paths correctly."""
    # Use the combined directory from common test data
    combined_dir = common_test_data_setup / COMBINED_DIR_NAME

    # Patch the search_dirs for this specific test
    with patch('meta_finder.config.search_dirs', [common_test_data_setup]):
        # Import inside the test to avoid issues with module-level search_dirs
        from meta_finder.collect import process_all_metadata

        # Run the metadata processing
        metadata = process_all_metadata()

        # Verify that metadata was returned
        assert metadata, "No metadata returned from process_all_metadata"

        # metadata is a dict where keys are info_devices.json paths and values are device metadata dicts
        # We need to check the device metadata within each file's data
        all_devices = []
        for devices_data in metadata.values():
            for dev_id, device_meta in devices_data.items():
                # Add device_id to the metadata for easier checking
                device_meta_with_id = device_meta.copy()
                device_meta_with_id['device_id'] = dev_id
                all_devices.append(device_meta_with_id)

        # Check that metadata for the combined path is processed
        found_i3 = any(entry.get('device_id') == 'i3' for entry in all_devices)
        found_i4 = any(entry.get('device_id') == 'i4' for entry in all_devices)
        assert found_i3, "Device i3 should be found in processed metadata"
        assert found_i4, "Device i4 should be found in processed metadata"


@pytest.mark.parametrize("time_content,should_be_included", [
    ("dummy data", False),  # Invalid time content should not be included
    ("Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2019-11-08 12:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2019-11-08 12:00:01\t0.15\t185\t0.06\t0.09\t5.3\t20.2", True)  # Valid time content should be included
], ids=["invalid_time_content", "valid_time_content"])
def test_write_files_list_with_combined_paths_time_content(common_test_data_setup, time_content, should_be_included, tmp_path):
    """Test that files with different time content are handled correctly in the output."""
    from meta_finder.file_writer import write_files_list

    # Use the combined directory from common test data but modify its content temporarily
    combined_dir = common_test_data_setup / COMBINED_DIR_NAME
    text_output_dir = combined_dir / 'text_output'
    text_file = text_output_dir / TEXT_FILE_NAME
    original_content = text_file.read_text()

    # Temporarily change the content of the text file
    text_file.write_text(time_content)

    # Write the files list using the process_all_metadata function
    output_file = tmp_path / 'test_files.tsv'

    # Patch the search_dirs for this specific test
    with patch('meta_finder.config.search_dirs', [common_test_data_setup]):
        # Import inside the patched context to ensure correct search_dirs
        from meta_finder.collect import process_all_metadata
        # Use process_all_metadata to generate the required json_metadata structure
        json_metadata = process_all_metadata()
        write_files_list(json_metadata, output_file)

    # Restore original content
    text_file.write_text(original_content)

    # Check the output file
    content = output_file.read_text()
    # Check for the info_devices.json file
    assert 'info_devices.json' in content, "info_devices.json should always be included in output"

    # Check if the text file is included based on time content validity
    if should_be_included:
        assert TEXT_FILE_NAME in content, "Text file with valid time content should be included in output"
    else:
        assert TEXT_FILE_NAME not in content, "Text file with invalid time content should not be included in output"
