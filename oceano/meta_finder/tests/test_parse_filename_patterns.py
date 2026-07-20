import pytest
from meta_finder.parse_data_file_name import parse_filename_for_metadata


@pytest.mark.parametrize(
    "filename,expected_result,description",
    [
        # Original patterns
        (
            "230508_1200bin2s_i03.tsv",
            {"datetime": "230508_1200", "averaging_interval": 2, "devices": ["i3"], "device_id": "i3"},
            "test_parse_filename_original_4_digit_time_single_device_tsv: Original pattern with 4-digit time and single device"
        ),
        (
            "191210_12000bin300s@i07.tsv",
            {"datetime": "191210_12000", "averaging_interval": 300, "devices": ["i7"], "device_id": "i7"},
            "test_parse_filename_original_5_digit_time_single_device_at_separator_tsv: Original pattern with 5-digit time and single device with @ separator"
        ),
        (
            "191210_120bin30s@i07,23,30,32.tsv",
            {"datetime": "191210_120", "averaging_interval": 30, "devices": ["i7", "i23", "i30", "i32"]},
            "test_parse_filename_original_3_digit_time_multiple_devices_at_separator_tsv: Original pattern with 3-digit time and multiple devices with @ separator"
        ),

        # New # separator pattern
        (
            "191210#07,23,30,32-bin300s.zip",
            {"datetime": "191210", "averaging_interval": 300, "devices": ["i7", "i23", "i30", "i32"]},
            "test_parse_filename_hash_separator_multiple_devices_zip: New # separator pattern with multiple devices in ZIP archive"
        ),
        (
            "191210_1200bin300s#i07,23,30,32.tsv",
            {"datetime": "191210_1200", "averaging_interval": 300, "devices": ["i7", "i23", "i30", "i32"]},
            "test_parse_filename_hash_separator_multiple_devices_tsv: Hash separator in original format with multiple devices"
        ),
        (
            "191210_1200bin300s#i07.tsv",
            {"datetime": "191210_1200", "averaging_interval": 300, "devices": ["i7"], "device_id": "i7"},
            "test_parse_filename_hash_separator_single_device_tsv: Hash separator in original format with single device"
        ),

        # Simple device pattern without interval
        (
            "200113_000_i13.csv",
            {"datetime": "200113_000", "devices": ["i13"], "device_id": "i13"},
            "test_parse_filename_simple_device_pattern_3_digit_time_csv: Simple device pattern with 3-digit time in CSV"
        ),
        (
            "200113_0000_i13.csv",
            {"datetime": "200113_0000", "devices": ["i13"], "device_id": "i13"},
            "test_parse_filename_simple_device_pattern_4_digit_time_csv: Simple device pattern with 4-digit time in CSV"
        ),
        (
            "200113_00000_i13.csv",
            {"datetime": "200113_00000", "devices": ["i13"], "device_id": "i13"},
            "test_parse_filename_simple_device_pattern_5_digit_time_csv: Simple device pattern with 5-digit time in CSV"
        ),

        # @ separator pattern without bin
        (
            "200113_00@i13.csv",
            {"datetime": "200113_00", "devices": ["i13"], "device_id": "i13"},
            "test_parse_filename_at_separator_no_bin_csv: @ separator pattern without bin in CSV"
        ),

        # New pattern without binning (per user request)
        (
            "191210#07,23,30,32.zip",
            {"datetime": "191210", "devices": ["i7", "i23", "i30", "i32"]},
            "test_parse_filename_hash_separator_multiple_devices_no_bin_zip: # separator pattern without binning"
        ),
        (
            "191224#03,13.zip",
            {"datetime": "191224", "devices": ["i3", "i13"]},
            "test_parse_filename_hash_separator_two_devices_no_bin_zip: Another # separator pattern without binning"
        ),

        # Files that should not match
        (
            "non_matching_file.txt",
            {},
            "test_parse_filename_non_matching_txt: File with no matching pattern should not match"
        ),
    ],
    ids=[
        "original_4_digit_time_single_device_tsv",
        "original_5_digit_time_single_device_at_separator_tsv",
        "original_3_digit_time_multiple_devices_at_separator_tsv",
        "hash_separator_multiple_devices_zip",
        "hash_separator_multiple_devices_tsv",
        "hash_separator_single_device_tsv",
        "simple_device_3_digit_time_csv",
        "simple_device_4_digit_time_csv",
        "simple_device_5_digit_time_csv",
        "at_separator_no_bin_csv",
        "hash_separator_multiple_devices_no_bin_zip",
        "hash_separator_two_devices_no_bin_zip",
        "non_matching_txt"
    ]
)
def test_parse_filename_for_metadata_supports_all_patterns(filename, expected_result, description):
    """Test that parse_filename_for_metadata supports all required patterns including new # separator and simple device patterns."""
    result = parse_filename_for_metadata(filename)

    # Check each expected key-value pair in the result
    for key, expected_value in expected_result.items():
        assert key in result, f"Key '{key}' not found in result for filename '{filename}'. Expected: {expected_result}, Actual: {result}. {description}"
        assert result[key] == expected_value, f"Value mismatch for key '{key}' in filename '{filename}'. Expected: {expected_value}, Actual: {result[key]}. {description}"

    # If expected result is empty, ensure result is also empty
    if not expected_result:
        assert result == {}, f"Expected empty result for filename '{filename}' but got: {result}. {description}"

    # Additional assertion to ensure the test description is properly used
    assert description is not None, f"Test description should not be None. {description}"