import os
from pathlib import Path, PurePosixPath
import pytest

# Import the function to test
from meta_finder.data_proc_funcs import extract_time_ranges_from_combined_file

# Sample content for a combined file with device-specific column names
# Creating a more complex test case with 5x more rows and varying data availability
combined_content_with_devices = (
    "Time\tVabs_i03\tVdir_i03\tv_i03\tu_i03\tVabs_i04\tVdir_i04\tv_i04\tu_i04\tVabs_i37\tVdir_i37\tv_i37\tu_i37\t\n"
    "2023-06-18 18:00:00.000000\t1.0\t45.0\t0.5\t\t\t\t\t\n"  # i03 has data, i04 and i37 don't
    "2023-06-18 18:10:00.000000\t1.1\t46.0\t0.6\t\t\t\t\t\n"  # i03 has data, i04 and i37 don't
    "2023-06-18 18:20:00.0000\t\t\t\t2.0\t55.0\t1.5\t1.5\t\t\t\t\t\n"  # i03 doesn't, i04 has data, i37 doesn't
    "2023-06-18 18:30:00.000000\t\t\t\t\t2.1\t56.0\t1.6\t\t\t\t\t\n"  # i03 doesn't, i04 has data, i37 doesn't
    "2023-06-18 18:40:00.000000\t\t\t\t\t3.0\t65.0\t2.5\t\n"   # i03 and i04 don't, i37 has data
    "2023-06-18 18:50:00.000000\t\t\t\t\t3.1\t66.0\t2.6\t\n"   # i03 and i04 don't, i37 has data
    "2023-06-18 19:00:00.0000\t1.2\t47.0\t0.7\t\t\t\t\t\n"  # i03 has data again, i04 and i37 don't
    "2023-06-18 19:10:00.00000\t\t\t\t\t2.2\t57.0\t1.7\t\t\t\t\t\n"  # i03 doesn't, i04 has data again, i37 doesn't
    "2023-06-18 19:20:00.0000\t\t\t\t\t3.2\t67.0\t2.7\t\n"   # i03 and i04 don't, i37 has data again
    "2023-06-18 19:30:00.0000\t\t\t\t\n"              # All devices have no data
    "2023-06-18 19:40:00.000000\t1.3\t48.0\t0.8\t2.3\t58.0\t1.8\t3.3\t68.0\t2.8\t\n"  # All devices have data
)

# Sample content for a combined file with combined device column names
combined_content_with_combined_devices = (
    "Time\tVabs_i05_14\tVdir_i05_14\tv_i05_14\tTemp_i06\tVabs_i07\tVdir_i07\tv_i07\t\n"
    "2023-06-18 18:00:00.000000\t1.0\t45.0\t0.5\t25.0\t\t\t\t\n"  # i05_14 and i06 have data, i07 doesn't
    "2023-06-18 18:10:00.000000\t1.1\t46.0\t0.6\t25.5\t\t\t\n"  # i05_14 and i06 have data, i07 doesn't
    "2023-06-18 18:20:00.000000\t\t\t\t\t2.0\t55.0\t1.5\t\n"     # i05_14 and i06 don't, i07 has data
    "2023-06-18 18:30:00.00000\t\t\t\t\t2.1\t56.0\t1.6\t\n"     # i05_14 and i06 don't, i07 has data
    "2023-06-18 18:40:00.000000\t1.2\t47.0\t0.7\t26.0\t3.0\t65.0\t2.5\t\n"  # All have data
)

# Sample content for a combined file without matching device-specific column names
# This simulates a file that has device suffixes but none that match our requested devices
combined_content_without_matching_devices = (
    "Time\tVabs_i99\tVdir_i99\tv_i99\tu_i99\tVabs_i98\tVdir_i98\tv_i98\tu_i98\tVabs_i97\tVdir_i97\tv_i97\tu_i97\t\n"
    "2023-06-18 18:00:00.000000\t1.0\t45.0\t0.5\t0.5\t2.0\t55.0\t1.5\t3.0\t65.0\t2.5\t\n"
    "2023-06-18 18:10:00.00000\t1.1\t46.0\t0.6\t2.1\t56.0\t1.6\t3.1\t66.0\t2.6\t\n"
)


def test_extract_time_ranges_from_combined_file_with_device_columns(common_test_data_setup):
    """Test extracting time ranges for multiple devices from a combined file with device-specific column names."""
    # Use an existing test directory from the common test data setup
    test_dir = common_test_data_setup
    # Use an existing device directory that has the proper structure
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6"
    text_output_dir = device_dir / "text_output"

    # Verify the test directory and files exist
    assert device_dir.exists(), "Combined device test directory should exist"
    assert text_output_dir.exists(), "Text output directory should exist"

    # Use an existing combined file if available, otherwise create one temporarily
    combined_file_path = text_output_dir / "230508_1551bin2s@i03.tsv"

    # If the file doesn't exist or doesn't have the right content, create it temporarily
    if not combined_file_path.exists() or "Vabs_i03" not in combined_file_path.read_text():
        # Create a temporary file with the required content
        with open(combined_file_path, "w") as f:
            f.write(combined_content_with_devices)

    dev_ids = ['i3', 'i4', 'i37']

    # Call the function - convert string to PurePosixPath
    result, comb_comments = extract_time_ranges_from_combined_file(device_dir, PurePosixPath(combined_file_path.name), dev_ids)

    # Check that we got results for all devices
    assert len(result) == 3
    assert 'i3' in result
    assert 'i4' in result
    assert 'i37' in result

    # Check the time ranges for each device
    # i3: should have data from 18:00:00 to 19:40:00
    assert result['i3'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i3']
    assert start_time == "2023-06-18 18:00:00"
    assert end_time == "2023-06-18 19:40:00"

    # i4: should have data from 18:20:00 to 19:40:00
    assert result['i4'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i4']
    assert start_time == "2023-06-18 18:20:00"
    assert end_time == "2023-06-18 19:40:00"

    # i37: should have data from 18:40:00 to 19:40:00
    assert result['i37'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i37']
    assert start_time == "2023-06-18 18:40:00"
    assert end_time == "2023-06-18 19:40:00"


def test_extract_time_ranges_from_combined_file_single_device_with_columns(common_test_data_setup):
    """Test extracting time ranges for a single device from a combined file with device-specific column names."""
    # Use an existing test directory from the common test data setup
    test_dir = common_test_data_setup
    # Use an existing device directory that has the proper structure
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6"
    text_output_dir = device_dir / "text_output"

    # Verify the test directory and files exist
    assert device_dir.exists(), "Combined device test directory should exist"
    assert text_output_dir.exists(), "Text output directory should exist"

    # Use an existing combined file if available, otherwise create one temporarily
    combined_file_path = text_output_dir / "230508_1551bin2s@i03.tsv"

    # If the file doesn't exist or doesn't have the right content, create it temporarily
    if not combined_file_path.exists() or "Vabs\tVdir\tv\tu\tInclination\tTemp" not in combined_file_path.read_text():
        # Create a temporary file with the required content
        with open(combined_file_path, "w") as f:
            f.write(combined_content_with_devices)

    dev_ids = ['i3']

    # Call the function - convert string to PurePosixPath
    result, comb_comments = extract_time_ranges_from_combined_file(device_dir, PurePosixPath(combined_file_path.name), dev_ids)

    # Check that we got results for the device
    assert len(result) == 1
    assert 'i3' in result
    assert result['i3'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i3']
    assert start_time == "2023-06-18 18:00:00"
    assert end_time == "2023-06-18 19:40:00"


def test_extract_time_ranges_from_combined_file_nonexistent_device_with_columns(common_test_data_setup):
    """Test extracting time ranges for a device that doesn't exist in a file with device-specific column names."""
    # Use an existing test directory from the common test data setup
    test_dir = common_test_data_setup
    # Use an existing device directory that has the proper structure
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6"
    text_output_dir = device_dir / "text_output"

    # Verify the test directory and files exist
    assert device_dir.exists(), "Combined device test directory should exist"
    assert text_output_dir.exists(), "Text output directory should exist"

    # Use an existing combined file if available, otherwise create one temporarily
    combined_file_path = text_output_dir / "230508_1551bin2s@i03.tsv"

    # If the file doesn't exist or doesn't have the right content, create it temporarily
    if not combined_file_path.exists() or "Vabs_i03" not in combined_file_path.read_text():
        # Create a temporary file with the required content
        with open(combined_file_path, "w") as f:
            f.write(combined_content_with_devices)

    dev_ids = ['i99']  # Device that doesn't exist in the file

    # Call the function - convert string to PurePosixPath
    result, comb_comments = extract_time_ranges_from_combined_file(device_dir, PurePosixPath(combined_file_path.name), dev_ids)

    # Check that we got None for the nonexistent device
    assert len(result) == 1
    assert 'i99' in result
    assert result['i99'] is None


def test_extract_time_ranges_from_combined_file_mixed_existing_and_nonexistent_with_columns(common_test_data_setup):
    """Test extracting time ranges for a mix of existing and nonexistent devices with device-specific column names."""
    # Use an existing test directory from the common test data setup
    test_dir = common_test_data_setup
    # Use an existing device directory that has the proper structure
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6"
    text_output_dir = device_dir / "text_output"

    # Verify the test directory and files exist
    assert device_dir.exists(), "Combined device test directory should exist"
    assert text_output_dir.exists(), "Text output directory should exist"

    # Use an existing combined file if available, otherwise create one temporarily
    combined_file_path = text_output_dir / "230508_1551bin2s@i03.tsv"

    # If the file doesn't exist or doesn't have the right content, create it temporarily
    if not combined_file_path.exists() or "Vabs_i03" not in combined_file_path.read_text():
        # Create a temporary file with the required content
        with open(combined_file_path, "w") as f:
            f.write(combined_content_with_devices)

    dev_ids = ['i3', 'i99', 'i37']  # Mix of existing and nonexistent devices

    # Call the function - convert string to PurePosixPath
    result, comb_comments = extract_time_ranges_from_combined_file(device_dir, PurePosixPath(combined_file_path.name), dev_ids)

    # Check that we got results for existing devices and None for nonexistent ones
    assert len(result) == 3
    assert 'i3' in result
    assert 'i99' in result
    assert 'i37' in result

    # Existing devices should have time ranges
    assert result['i3'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i3']
    assert start_time == "2023-06-18 18:00:00"
    assert end_time == "2023-06-18 19:40:00"

    assert result['i37'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i37']
    assert start_time == "2023-06-18 18:40:00"
    assert end_time == "2023-06-18 19:40:00"

    # Nonexistent device should be None
    assert result['i99'] is None


def test_extract_time_ranges_from_combined_file_with_combined_device_columns(common_test_data_setup):
    """Test extracting time ranges for devices from a combined file with combined device column names."""
    # Use an existing test directory from the common test data setup
    test_dir = common_test_data_setup
    # Use an existing device directory that has the proper structure
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6"
    text_output_dir = device_dir / "text_output"

    # Verify the test directory and files exist
    assert device_dir.exists(), "Combined device test directory should exist"
    assert text_output_dir.exists(), "Text output directory should exist"

    # Use an existing combined file if available, otherwise create one temporarily
    combined_file_path = text_output_dir / "230508_1551bin2s@i03.tsv"

    # If the file doesn't exist or doesn't have the right content, create it temporarily
    if not combined_file_path.exists() or "Vabs_i05_14" not in combined_file_path.read_text():
        # Create a temporary file with the required content
        with open(combined_file_path, "w") as f:
            f.write(combined_content_with_combined_devices)

    # Request individual devices that are part of combined columns
    dev_ids = ['i5', 'i14', 'i6', 'i7']

    # Call the function - convert string to PurePosixPath
    result, comb_comments = extract_time_ranges_from_combined_file(device_dir, PurePosixPath(combined_file_path.name), dev_ids)

    # Check that we got results for all devices
    assert len(result) == 4
    assert 'i5' in result
    assert 'i14' in result
    assert 'i6' in result
    assert 'i7' in result

    # Check the time ranges for each device
    # i5 and i14: should have data from 18:00:00 to 18:40:00 (from combined column i05_14)
    assert result['i5'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i5']
    assert start_time == "2023-06-18 18:00:00"
    assert end_time == "2023-06-18 18:40:00"

    assert result['i14'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i14']
    assert start_time == "2023-06-18 18:00:00"
    assert end_time == "2023-06-18 18:40:00"

    # i6: should have data from 18:00:00 to 18:40:00
    assert result['i6'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i6']
    assert start_time == "2023-06-18 18:00:00"
    assert end_time == "2023-06-18 18:40:00"

    # i7: should have data from 18:20:00 to 18:40:00
    assert result['i7'] is not None
    start_time, end_time, bursts_t, burst_dt = result['i7']
    assert start_time == "2023-06-18 18:20:00"
    assert end_time == "2023-06-18 18:40:00"