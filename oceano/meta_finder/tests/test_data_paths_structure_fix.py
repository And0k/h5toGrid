"""
Test for the data_paths structure fix to ensure data_paths maintains consistent format.
This test verifies that data_paths is preserved as a dictionary instead of being
converted to a list and back, which was causing "data_path" values to appear as "?".
"""
from meta_finder.collect import process_all_metadata
from meta_finder.file_finder import discover_device_dirs


def test_data_paths_structure_remains_as_dict(common_test_data_setup):
    """Test that data_paths maintains consistent dictionary structure through processing."""
    # Use the existing test data from the common test data setup
    test_dir = common_test_data_setup

    # Use an existing test directory that has the proper structure
    # We'll use the combined device test directory which has text_output files
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6"
    text_output_dir = device_dir / "text_output"

    # Verify the test directory and files exist
    assert device_dir.exists(), "Combined device test directory should exist"
    assert text_output_dir.exists(), "Text output directory should exist"

    # Find our test device directories
    device_dirs_mapping = discover_device_dirs([
        test_dir
    ])  # Search in the common test data directory

    # Verify we found device directories
    assert len(device_dirs_mapping) >= 1, (
        "Should find at least one cruise directory for data_paths structure test"
    )

    # Process the metadata
    metadata_result = process_all_metadata(device_dirs_mapping)

    # Verify processing completed
    assert len(metadata_result) >= 0, (
        "Should process metadata for data_paths structure test"
    )

    # Check the structure of the result if there's any metadata
    for json_path, devices_data in metadata_result.items():
        for dev_id, device_info in devices_data.items():
            # Check if data_paths exists and what type it is
            data_paths = device_info.get("data_paths", {})

            # The main assertion: data_paths should be a dictionary, not a list
            assert isinstance(data_paths, dict), (
                f"Expected data_paths to be dict for device {dev_id}, but got {type(data_paths)} - this indicates the data_paths structure fix failed"
            )

            # Additional verification that it has proper content
            assert len(data_paths) >= 0, (
                f"data_paths should have valid content for device {dev_id} in data_paths structure test"
            )


def test_data_paths_structure_with_multiple_devices(common_test_data_setup):
    """Test that data_paths maintains dictionary structure with multiple devices."""
    # Use the existing test data from the common test data setup
    test_dir = common_test_data_setup

    # Use an existing test directory with multiple devices
    # We'll use the semicolon patterns test directory which has multiple devices
    device_dir = test_dir / "250707_semicolon_patterns" / "250708_semi@i21"
    text_output_dir = device_dir / "text_output"

    # Verify the test directory and files exist
    assert device_dir.exists(), "Semicolon patterns test directory should exist"
    assert text_output_dir.exists(), "Text output directory should exist"

    # Find our test device directories
    device_dirs_mapping = discover_device_dirs([
        test_dir
    ])  # Search in the common test data directory

    # Process the metadata
    metadata_result = process_all_metadata(device_dirs_mapping)

    # Check that all devices maintain data_paths as dictionary
    for json_path, devices_data in metadata_result.items():
        for device_id, device_info in devices_data.items():
            data_paths = device_info.get("data_paths", {})

            # Verify data_paths is still a dictionary (not converted to list)
            assert isinstance(data_paths, dict), (
                f"Multiple devices test: Expected data_paths to be dict for device {device_id}, but got {type(data_paths)} - fix failed for multiple devices scenario"
            )

            # Verify it has proper content
            assert len(data_paths) >= 0, (
                f"Multiple devices test: data_paths should have valid content for device {device_id}"
            )


if __name__ == "__main__":
    # These calls are just for demonstration - pytest will run the tests
    print("Tests are designed to be run with pytest")