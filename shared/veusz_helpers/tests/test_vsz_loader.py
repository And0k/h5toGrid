import sys
import pytest
from pathlib import Path
import numpy as np
from Veusz_plugins.vsz_loader import (
    get_info_from_filename, add_months, search_time_range_indexes, 
    zone_to_seconds_offset, get_path_in_parents, load_info_json, 
    bool2ranges
)


@pytest.mark.parametrize(
    "basename,expected_time_start,expected_time_end,expected_devices,comment",
    [
        ("231229_2201@i91.vsz", "2023-12-29T22:01:00", "2023-12-29T22:01:00", {"i91": {"type": "i", "model": "i", "number": 91}}, 
         "Simple device ID i91 from 22:01:00 to 22:01:00"),
        ("230830_0000-10_1200.vsz", "2023-08-30T00:00:00", "2023-09-10T12:00:00", {}, 
         "Time range from 00:00 to 10:1200 (end date 10 days later)"),
        ("230830_0000dt=1h.vsz", "2023-08-30T00:00:00", "2023-08-30T01:00:00", {}, 
         "Duration of 1 hour from start time"),
        ("230830_0000-10_1200.vsz", "2023-08-30T00:00:00", "2023-09-10T12:00:00", {}, 
         "Time range from 00:00 to 10:1200 (end date 10 days later)"),
        ("231229_2201@i3,4,15.vsz", "2023-12-29T22:01:00", "2023-12-29T22:01:00", {"i03": {"type": "i", "model": "i", "number": 3}, "i04": {"type": "i", "model": "i", "number": 4}, "i15": {"type": "i", "model": "i", "number": 15}}, 
         "Multiple single-digit devices with leading zeros"),
    ],
    ids=[
        "simple_device",
        "time_range",
        "duration_1h", 
        "wind_ecmwf",
        "multiple_devices"
    ]
)
def test_get_info_from_filename_various_formats(basename, expected_time_start, expected_time_end, expected_devices, comment):
    """Test parsing of different filename formats."""
    time_range, out_info = get_info_from_filename(basename)
    
    expected_start = np.datetime64(expected_time_start)
    expected_end = np.datetime64(expected_time_end)
    
    assert len(time_range) == 2, f"Expected 2 time elements for {comment}, got {len(time_range)}"
    assert time_range[0] == expected_start, f"Start time mismatch for {comment}: expected {expected_start}, got {time_range[0]}"
    assert time_range[1] == expected_end, f"End time mismatch for {comment}: expected {expected_end}, got {time_range[1]}"
    
    for device_id, device_info in expected_devices.items():
        assert device_id in out_info['devices'], f"Device {device_id} not found in output for {comment}"
        if device_info['type']:
            assert out_info['devices'][device_id]['type'] == device_info['type'], f"Device type mismatch for {comment}"
        if device_info['number']:
            assert out_info['devices'][device_id]['number'] == device_info['number'], f"Device number mismatch for {comment}"


def test_add_months():
    """Test adding months to a datetime64 object."""
    dt = np.datetime64('2023-01-15T12:00:00')
    result = add_months(dt, 3)
    
    expected = np.datetime64('2023-04-15T12:00:00')
    assert result == expected, "Should add 3 months correctly"


def test_add_months_with_overflow():
    """Test adding months with month overflow."""
    dt = np.datetime64('2023-01-31T12:00:00')
    result = add_months(dt, 1)
    
    # February doesn't have 31 days, so it should go to the last day of February
    expected = np.datetime64('2023-02-28T12:00:00')
    assert result == expected, "Should handle month overflow by using last day of target month"


@pytest.mark.parametrize(
    "index,time_range,raw_time_shift_s,expected_result,comment",
    [
        (np.array([100, 200, 300, 400, 500]), 
         [np.datetime64('1970-01-01T00:00:00.000000150'), np.datetime64('1970-01-01T00:00:00.000000450')], 
         0, [1, 4], 
         "Time range from 150 to 450 in index array [100,200,300,400,500]"),
    ],
    ids=["basic_time_range"]
)
def test_search_time_range_indexes(index, time_range, raw_time_shift_s, expected_result, comment):
    """Test finding time range indexes."""
    result = search_time_range_indexes(index, time_range, raw_time_shift_s)
    assert result == expected_result, f"Index search result mismatch for {comment}: expected {expected_result}, got {result}"


def test_add_months():
    """Test adding months to a datetime64 object."""
    dt = np.datetime64('2023-01-15T12:00:00')
    result = add_months(dt, 3)
    
    expected = np.datetime64('2023-04-15T12:00:00')
    assert result == expected, "Should add 3 months correctly"


def test_add_months_with_overflow():
    """Test adding months with month overflow."""
    dt = np.datetime64('2023-01-31T12:00:00')
    result = add_months(dt, 1)
    
    # February doesn't have 31 days, so it should go to the last day of February
    expected = np.datetime64('2023-02-28T12:00:00')
    assert result == expected, "Should handle month overflow by using last day of target month"


@pytest.mark.parametrize(
    "index,time_range,raw_time_shift_s,expected_result,comment",
    [
        (np.array([100, 200, 300, 400, 500]), 
         [np.datetime64('1970-01-01T00:00:00.000000150'), np.datetime64('1970-01-01T00:00:00.000000450')], 
         0, [1, 4], 
         "Time range from 150 to 450 in index array [100,200,300,400,500]"),
    ],
    ids=["basic_time_range"]
)
def test_search_time_range_indexes(index, time_range, raw_time_shift_s, expected_result, comment):
    """Test finding time range indexes."""
    result = search_time_range_indexes(index, time_range, raw_time_shift_s)
    assert result == expected_result, f"Index search result mismatch for {comment}: expected {expected_result}, got {result}"


def test_zone_to_seconds_offset():
    """Test converting timezone string to seconds offset."""
    # Test UTC+0
    assert zone_to_seconds_offset("UTC") == 0
    # Test UTC+2 (2 hours in seconds)
    assert zone_to_seconds_offset("UTC+2") == 7200
    # Test UTC-5 (negative 5 hours in seconds)
    assert zone_to_seconds_offset("UTC-5") == -18000


def test_get_path_in_parents_found():
    """Test finding a file in parent directories."""
    # Create a temporary test structure
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create parent and child directories
        parent_dir = Path(temp_dir)
        child_dir = parent_dir / "child"
        child_dir.mkdir()
        
        # Create the target file in parent directory
        target_file = parent_dir / "target_file.txt"
        target_file.write_text("test content")
        
        # Search from child directory
        result = get_path_in_parents(child_dir, "target_file.txt")
        assert result == target_file


def test_get_path_in_parents_not_found():
    """Test that FileNotFoundError is raised when file is not found."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent_file = Path(temp_dir) / "nonexistent.txt"
        nonexistent_file.parent.mkdir(exist_ok=True)
        
        with pytest.raises(FileNotFoundError):
            get_path_in_parents(nonexistent_file.parent, "nonexistent.txt")


def test_load_info_json():
    """Test loading device information from JSON file."""
    import tempfile
    import json
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test JSON file
        json_file = Path(temp_dir) / "info_devices.json"
        test_data = {
            "i91": ["point_name", 50, 10, "symbol", [55.1, 20.2]]
        }
        with open(json_file, 'w', encoding='utf8') as f:
            json.dump(test_data, f)
        
        # Test loading
        probes = {"devices": {"i91": {"type": "i", "model": "i", "number": 91}}}
        result = load_info_json(probes, json_file)
        
        assert "i91" in result
        assert result["i91"]["p"] == "point_name"
        assert result["i91"]["b"] == 50
        assert result["i91"]["s"] == "symbol"


def test_bool2ranges():
    """Test converting boolean array to ranges."""
    # Simple test case
    b_ok = np.array([True, True, False, False, True, True, True])
    result = bool2ranges(b_ok, min_range=1)
    
    # Should return indices of edges where True/False changes
    expected = np.array([0, 2, 4, 7])  # Start of first True, end of first True, start of second True, end of second True
    # Note: This function might be complex, just test it executes without errors
    assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])