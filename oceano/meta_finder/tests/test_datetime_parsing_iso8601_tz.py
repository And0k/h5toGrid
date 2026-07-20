"""
Test datetime parsing with ISO 8601 format including timezone offset without colon.
"""

import pytest
from datetime import datetime
from meta_finder.data_proc_funcs import parse_datetime_from_row


@pytest.mark.parametrize(
    "line, is_raw, sep, expected_result, description",
    [
        (
            "2016-02-19T16:02:30.000000+0200\t5.108148\t0.024317\t165.191746",
            False,
            "\t",
            datetime(2016, 2, 19, 16, 2, 30, 0),
            "ISO 8601 with timezone offset +0200 (no colon)",
        ),
        (
            "2016-02-19T16:02:30.000000-0500\t5.108148\t0.024317\t165.191746",
            False,
            "\t",
            datetime(2016, 2, 19, 16, 2, 30, 0),
            "ISO 8601 with timezone offset -0500 (no colon)",
        ),
        (
            "2016-02-19T16:02:30+0200\t5.108148\t0.024317\t165.191746",
            False,
            "\t",
            datetime(2016, 2, 19, 16, 2, 30, 0),
            "ISO 8601 with timezone offset +0200 (no microseconds)",
        ),
        (
            "2016-02-19T16:02:30.123456+02:00\t5.108148\t0.024317\t165.191746",
            False,
            "\t",
            datetime(2016, 2, 19, 16, 2, 30, 123456),
            "ISO 8601 with timezone offset +02:00 (with colon - standard format)",
        ),
        (
            "2016-02-19 16:02:30\t5.108148\t0.024317\t165.191746",
            False,
            "\t",
            datetime(2016, 2, 19, 16, 2, 30, 0),
            "Standard space-separated datetime format",
        ),
        (
            "NaN\t5.108148\t0.024317\t165.191746",
            False,
            "\t",
            None,
            "NaN timestamp should return None",
        ),
        (
            "",
            False,
            "\t",
            None,
            "Empty line should return None",
        ),
    ],
)
def test_parse_datetime_from_row_various_formats(
    line, is_raw, sep, expected_result, description
):
    """Test datetime parsing with various ISO 8601 formats."""
    result = parse_datetime_from_row(line, is_raw=is_raw, sep=sep)
    if expected_result is None:
        assert result is None, f"{description}: expected None but got {result}"
    else:
        assert result is not None, f"{description}: expected datetime but got None"
        # Compare year, month, day, hour, minute, second, microsecond
        assert (
            result.year,
            result.month,
            result.day,
            result.hour,
            result.minute,
            result.second,
            result.microsecond,
        ) == (
            expected_result.year,
            expected_result.month,
            expected_result.day,
            expected_result.hour,
            expected_result.minute,
            expected_result.second,
            expected_result.microsecond,
        ), f"{description}: expected {expected_result} but got {result}"


def test_parse_datetime_with_header():
    """Test parsing datetime from a line that starts with header text."""
    line = "Time\t P\t Vabs\t Vdir"
    result = parse_datetime_from_row(line, is_raw=False, sep="\t")
    assert result is None, "Header line should return None"


def test_parse_datetime_serial_fallback():
    """Test that serial MATLAB/Excel format still works as fallback."""
    line = "42430.66840\t5.108148\t0.024317\t165.191746"
    result = parse_datetime_from_row(line, is_raw=False, sep="\t")
    assert result is not None, "Serial format should parse successfully"
    # Serial 42430.66840 should be around March 2016
    assert result.year == 2016, f"Expected year 2016 but got {result.year}"
    assert result.month == 3, f"Expected month 3 but got {result.month}"
