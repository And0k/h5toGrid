#!/usr/bin/env python
"""
Test script to reproduce the HDF5 timestamp conversion issue.
This tests the specific issue where raw HDF5 dates are not properly converted to strings.
"""
import pytest
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

# Add the project source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_timestamp_conversion_issue_reproduction():
    """Test the specific issue with HDF5 raw dates not being converted to strings."""
    from meta_finder.hdf5_processor import _convert_single_timestamp, _convert_timestamps

    print("Testing the timestamp conversion issue...")
    print("Issue: Raw HDF5 timestamps like 15759684002000000 are not converted to proper date strings")
    print("="*80)

    # These are the exact values from the user's meta file that are showing as raw numbers
    problematic_timestamps = [
        15759684002000000,  # From row 2, time_raw_st column
        15773759980000000,  # From row 2, time_raw_en column
        157712561400000000,  # From row 6, time_raw_st column
        1578352607000000,  # From row 6, time_raw_en column
    ]

    print("Testing problematic nanosecond timestamps:")
    for i, ts_val in enumerate(problematic_timestamps):
        print(f" {i+1}. Raw timestamp: {ts_val}")

        # Test single timestamp conversion
        result = _convert_single_timestamp(ts_val, {})
        print(f"     Converted result: {result}")

        # Expected: should be a proper datetime string, not the raw number
        if str(result) == str(ts_val):
            print(f"     ❌ ISSUE: Raw timestamp not converted - this is the bug!")
        else:
            print(f"     ✅ OK: Timestamp properly converted")

        # Calculate expected datetime for verification
        expected_dt = datetime.fromtimestamp(ts_val / 1e9)  # Convert from nanoseconds to seconds
        expected_str = expected_dt.strftime('%Y-%m-%d %H:%M:%S')
        print(f"     Expected: {expected_str}")
        print()


@pytest.mark.parametrize("raw_timestamp,expected_date_str,comment", [
    (15759684002000000, "2019-12-02 00:20:00", "first timestamp from user's example should convert to proper date string"),
    (15773759998000000, "2019-12-26 12:59:59", "second timestamp from user's example should convert to proper date string"),
    (15771256140000000, "2019-12-24 18:26:54", "third timestamp from user's example should convert to proper date string"),
    (1578352607000000, "2020-01-06 23:16:47", "fourth timestamp from user's example should convert to proper date string"),
], ids=["first_ts", "second_ts", "third_ts", "fourth_ts"])
def test_timestamp_conversion_fix(raw_timestamp, expected_date_str, comment):
    """Test that problematic nanosecond timestamps are properly converted to datetime strings."""
    from meta_finder.hdf5_processor import _convert_single_timestamp

    result = _convert_single_timestamp(raw_timestamp, {})

    # Check that the result is not the raw timestamp number
    assert str(result) != str(raw_timestamp), f"Timestamp should not remain as raw number: {result} as per: {comment}"

    # Check that the result contains the expected date pattern
    assert expected_date_str.split()[0] in str(result), f"Result should contain expected date {expected_date_str.split()[0]}, got {result} as per: {comment}"

    # Check that the result is in proper datetime format
    try:
        # Try to parse the result as a datetime string
        parsed_result = datetime.strptime(str(result), '%Y-%m-%d %H:%M:%S')
        datetime_ok = True
    except ValueError:
        datetime_ok = False

    assert datetime_ok, f"Result should be parseable as datetime string, got {result} as per: {comment}"


def test_numpy_datetime64_conversion():
    """Test conversion with actual numpy datetime64 objects."""
    from meta_finder.hdf5_processor import _convert_single_timestamp
    import numpy as np

    # Create numpy datetime64 with nanosecond precision (like HDF5 files might have)
    np_timestamp = np.datetime64(15759684002000000, 'ns')

    result = _convert_single_timestamp(np_timestamp, {})

    print(f"Numpy datetime64 conversion: {np_timestamp} -> {result}")

    # Should not return the raw value
    assert str(result) != str(np_timestamp), f"NumPy datetime64 should be converted to string as per: numpy datetime64 conversion"

    # Should contain date information
    assert "2019-12" in str(result), f"Result should contain date information as per: numpy datetime64 conversion"


def test_convert_timestamps_array_with_nano_values():
    """Test the _convert_timestamps function with problematic nanosecond values."""
    from meta_finder.hdf5_processor import _convert_timestamps

    # Test with problematic nanosecond timestamps from the user's data
    time_values = [157596840020000, 157737599980000]  # Start and end times
    start_time, end_time = _convert_timestamps(time_values, {}, 0, -1)

    print(f"Array conversion - Start: {start_time}, End: {end_time}")

    # Both should be converted to proper datetime strings, not raw numbers
    assert str(start_time) != str(time_values[0]), f"Start time should be converted from raw timestamp as per: array nanosecond timestamp conversion"
    assert str(end_time) != str(time_values[1]), f"End time should be converted from raw timestamp as per: array nanosecond timestamp conversion"

    # Both should contain date patterns
    assert "2019-12" in str(start_time), f"Start time should contain date pattern as per: array nanosecond timestamp conversion"
    assert "2019-12" in str(end_time), f"End time should contain date pattern as per: array nanosecond timestamp conversion"


if __name__ == "__main__":
    test_timestamp_conversion_issue_reproduction()
    print("Running parametrized tests...")

    # Run the parametrized test manually for debugging
    test_timestamp_conversion_fix(15759684002000000, "2019-12-02 00:20:00", "first timestamp from user's example should convert to proper date string")
    test_timestamp_conversion_fix(1577375999800000000, "2019-12-26 12:59:59", "second timestamp from user's example should convert to proper date string")
    test_timestamp_conversion_fix(157712561400000000, "2019-12-24 18:26:54", "third timestamp from user's example should convert to proper date string")
    test_timestamp_conversion_fix(15783526070000000, "2020-01-06 23:16:47", "fourth timestamp from user's example should convert to proper date string")

    test_numpy_datetime64_conversion()
    test_convert_timestamps_array_with_nano_values()

    print("All tests completed.")