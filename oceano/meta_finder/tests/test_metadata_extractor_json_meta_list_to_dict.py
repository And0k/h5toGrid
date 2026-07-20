import tempfile
import json
from pathlib import Path
from meta_finder.metadata_extractor import info_meta_list_to_dict

def test_json_comment_preservation_basic_mapping():
    """Test that JSON comments are preserved when mapping from list to dict."""

    # Test 1: JSON with comment field (11 elements including comment)
    json_data_with_comment = [
        "Point1",
        "100",
        "5",
        "A",
        "55.1",
        "37.2",
        "2023-06-18 18:00:00",
        "2023-06-18 19:00:00",
        "3600",
        "1800",
        "Original JSON comment",
    ]

    result = info_meta_list_to_dict(json_data_with_comment)

    # Check that all fields are properly mapped
    expected = {
        "point": "Point1",
        "sea_depth": "100",
        "height_above_bottom": "5",
        "modification_symbol": "A",
        "lat": "55.1",
        "lon": "37.2",
        "time_st": "2023-06-18 18:00:00",
        "time_en": "2023-06-18 19:00:00",
        "burst_dt": "3600",
        "bursts_t": "1800",
        "comment": "Original JSON comment"
    }

    # Verify all fields match
    for key, expected_val in expected.items():
        assert result[key] == expected_val, f"Field {key}: expected {expected_val}, got {result[key]}"


def test_json_comment_preservation_without_comment():
    """Test that JSON without comment field gets default value."""

    # Test 2: JSON without comment field (10 elements)
    json_data_without_comment = ["Point1", "10", "5", "A", "5.1", "37.2",
                                 "2023-06-18 18:00:00", "2023-06-18 19:00:00",
                                 "3600", "1800"]

    result2 = info_meta_list_to_dict(json_data_without_comment)

    expected2 = {
        "point": "Point1",
        "sea_depth": "10",
        "height_above_bottom": "5",
        "modification_symbol": "A",
        "lat": "5.1",
        "lon": "37.2",
        "time_st": "2023-06-18 18:00:00",
        "time_en": "2023-06-18 19:00:00",
        "burst_dt": "3600",
        "bursts_t": "1800",
        "comment": "?"  # Default value for missing comment field
    }

    # Verify all fields match
    for key, expected_val in expected2.items():
        assert result2[key] == expected_val, f"Field {key}: expected {expected_val}, got {result2[key]}"


def test_json_comment_combination():
    """Test combining JSON comments with processing comments."""

    # Start with JSON that had a comment
    json_data_with_comment = ["Point1", "100", "5", "A", "55.1", "37.2",
                              "2023-06-18 18:00:00", "2023-06-18 19:00:00",
                              "3600", "1800", "Original JSON comment"]
    result = info_meta_list_to_dict(json_data_with_comment)

    # Simulate adding a combined comment (like in data_processor.py line ~168)
    entry_with_json_comment = result.copy()
    new_comment = "i5+i14 output"
    if entry_with_json_comment.get('comment') and entry_with_json_comment['comment'] not in ['?', '-', '']:
        entry_with_json_comment['comment'] = f"{entry_with_json_comment['comment']}; {new_comment}"
    else:
        entry_with_json_comment['comment'] = new_comment

    expected_combined = "Original JSON comment; i5+i14 output"
    assert entry_with_json_comment['comment'] == expected_combined, f"Expected combined comment: {expected_combined}"


def test_json_comment_with_gpx():
    """Test combining JSON comments with GPX comments."""

    # Start with JSON that had a comment
    json_data_with_comment = ["Point1", "100", "5", "A", "5.1", "37.2",
                              "2023-06-18 18:00:00", "2023-06-18 19:00:00",
                              "3600", "1800", "Original JSON comment"]
    result = info_meta_list_to_dict(json_data_with_comment)

    # Simulate adding GPX comment to entry that already has JSON comment
    entry_with_json_and_gpx = result.copy()
    gpx_paths = "path/to/file.gpx"

    # Simulate GPX comment addition (like in data_processor.py line ~187)
    if 'comment' in entry_with_json_and_gpx and entry_with_json_and_gpx['comment'] and entry_with_json_and_gpx['comment'] not in ['?', '-', '']:
        entry_with_json_and_gpx['comment'] = f"{entry_with_json_and_gpx['comment']}; GPX: {gpx_paths}"
    else:
        entry_with_json_and_gpx['comment'] = f"GPX: {gpx_paths}"

    expected_gpx_combined = "Original JSON comment; GPX: path/to/file.gpx"
    assert entry_with_json_and_gpx['comment'] == expected_gpx_combined, f"Expected GPX combined comment: {expected_gpx_combined}"
