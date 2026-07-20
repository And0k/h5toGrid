"""
Tests to reproduce the 'list' object has no attribute 'strip' error in _extract_device_time_ranges_from_combined_content
"""
import pytest
from unittest.mock import patch
from meta_finder.data_proc_funcs import _extract_device_time_ranges_from_combined_content
from pathlib import Path


def test_extract_device_time_ranges_from_combined_content_with_list_instead_of_string():
    """Test _extract_device_time_ranges_from_combined_content when columns contain lists instead of strings"""
    # Simulate the problematic scenario where columns contain lists instead of strings
    lines = [
        "Time\tVabs_i01\tVdir_i01",  # Header line
        "2023-01-01 12:00:00\t[['value1', 'value2']]\t[['value3', 'value4']]"  # Data line with lists as strings
    ]

    # This should handle the case gracefully where columns might contain lists
    result = _extract_device_time_ranges_from_combined_content(
        Path("/fake/dir"),
        Path("fake_file.txt"),
        lines,
        ["i01"]
    )

    # Should return a tuple with device_time_ranges and combined_comments
    assert isinstance(result, tuple)
    assert len(result) == 2

    device_time_ranges, combined_comments = result
    assert isinstance(device_time_ranges, dict)
    assert isinstance(combined_comments, dict)


def test_extract_device_time_ranges_from_combined_content_with_actual_lists():
    """Test _extract_device_time_ranges_from_combined_content when columns contain actual lists"""
    # Simulate the problematic scenario where columns contain actual lists instead of strings
    lines = [
        "Time\tVabs_i01\tVdir_i01",  # Header line
        ["2023-01-01 12:00:00", ["value1", "value2"], ["value3", "value4"]]  # Data line with actual lists
    ]

    # This should handle the case gracefully where lines might contain lists
    result = _extract_device_time_ranges_from_combined_content(
        Path("/fake/dir"),
        Path("fake_file.txt"),
        lines,
        ["i01"]
    )

    # Should return a tuple with device_time_ranges and combined_comments
    assert isinstance(result, tuple)
    assert len(result) == 2

    device_time_ranges, combined_comments = result
    assert isinstance(device_time_ranges, dict)
    assert isinstance(combined_comments, dict)


def test_extract_device_time_ranges_from_combined_content_with_mixed_types():
    """Test _extract_device_time_ranges_from_combined_content with mixed column types"""
    # Simulate a scenario with mixed types in columns
    lines = [
        "Time\tVabs_i01\tVdir_i01",  # Header line
        "2023-01-01 12:00:00\tvalue1\t[\"nested\", \"list\"]"  # Data line with mixed types
    ]

    # This should handle the case gracefully
    result = _extract_device_time_ranges_from_combined_content(
        Path("/fake/dir"),
        Path("fake_file.txt"),
        lines,
        ["i01"]
    )

    # Should return a tuple with device_time_ranges and combined_comments
    assert isinstance(result, tuple)
    assert len(result) == 2

    device_time_ranges, combined_comments = result
    assert isinstance(device_time_ranges, dict)
    assert isinstance(combined_comments, dict)


def test_extract_device_time_ranges_from_combined_content_exact_list_strip_error():
    """Test to reproduce the exact 'list' object has no attribute 'strip' error"""
    # Create a scenario where columns[col_index] is actually a list
    # This simulates the case where after splitting a line, one of the elements is a list
    lines = [
        "Time\tVabs_i01\tVdir_i01",  # Header line (string)
        "2023-01-01 12:00:00\tvalue1\tvalue2"  # Normal data line
    ]

    # Let's test with actual problematic input that could cause this
    # Create a scenario where the data lines themselves are lists instead of strings
    problematic_lines = [
        "Time\tVabs_i01\tVdir_i01",  # Header line (string)
        [["2023-01-01 12:00:00"], "value1", "value2"]  # Data line where first element is a list
    ]

    # This should handle the case gracefully
    result = _extract_device_time_ranges_from_combined_content(
        Path("/fake/dir"),
        Path("fake_file.txt"),
        problematic_lines,
        ["i01"]
    )

    # Should return a tuple with device_time_ranges and combined_comments
    assert isinstance(result, tuple)
    assert len(result) == 2

    device_time_ranges, combined_comments = result
    assert isinstance(device_time_ranges, dict)
    assert isinstance(combined_comments, dict)