"""Test YAML underscore suffix handling in write/read round-trip.

Tests here verify that write_devices_meta_yaml and read_metadata_file handle
devices with underscore suffixes correctly in round-trip scenarios.
"""

import pytest
from pathlib import Path

from meta_finder.io_info_files import (
    write_devices_meta_yaml,
    read_metadata_file,
    _concatenate_additional_comments,
    _remove_trailing_empty_fields,
)


def _convert_to_read_format(content: dict) -> dict:
    """Convert flat {device_id: [list]} to nested {device_id: {"0": [list]}} format.

    Simulates what read_metadata_file returns: write applies _remove_trailing_empty_fields,
    read applies _concatenate_additional_comments.
    """
    result = {}
    for device_id, values in content.items():
        trimmed = list(_remove_trailing_empty_fields(values))
        processed = _concatenate_additional_comments(trimmed)
        result[device_id] = {"0": processed}
    return result


@pytest.mark.parametrize(
    "original_content,test_description",
    [
        (
            {
                "ip7": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', ''],
                "w2": ['?', '?', '?', '?', '?', '?', '2013-05-10 10:00:00', '2013-05-10 17:06:27', '-', '-', '?'],
                "ip7_": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', '?'],
                "ip7__": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', ''],
                "w1": ['?', '?', '?', '?', '?', '?', '2013-05-10 10:00:00', '2013-05-19 15:01:15', '-', '-', '?'],
                "i15": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', '?'],
                "ip1_": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', ''],
                "i1": ['?', '?', '?', '?', '?', '?', '2013-05-10 10:00:00', '2013-05-10 14:17:19:59', '-', '-', '?'],
            },
            "Round-trip transformation with multiple devices",
        ),
        (
            {
                "device1": ['a', 'b', 'c'],
                "device2": ['d', 'e', 'f'],
            },
            "Round-trip transformation without underscore suffixes",
        ),
    ],
)
def test_round_trip_yaml_write_read(tmp_path, original_content, test_description):
    """Test that writing and reading YAML preserves original content."""
    yaml_file = tmp_path / "test_devices.yaml"

    write_devices_meta_yaml(tmp_path, yaml_file, original_content)
    read_content = read_metadata_file(yaml_file)

    expected_read = _convert_to_read_format(original_content)
    assert read_content == expected_read, f"Failed: {test_description} - content not preserved after round-trip"


def test_yaml_file_structure(tmp_path):
    """Test that the YAML file preserves flat device entries (no grouping by underscore suffix)."""
    yaml_file = tmp_path / "test_structure.yaml"

    content = {
        "ip7": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', ''],
        "ip7_": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', '?'],
        "ip7__": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', ''],
        "w2": ['?', '?', '?', '?', '?', '?', '2013-05-10 10:00:00', '2013-05-10 17:06:27', '-', '-', '?'],
    }

    write_devices_meta_yaml(tmp_path, yaml_file, content)

    with open(yaml_file, 'r', encoding='utf-8') as f:
        yaml_content = f.read()

    # Since underscore suffix grouping was removed, devices are written as flat lists
    assert '"ip7":' in yaml_content, "ip7 should be in the YAML file"
    assert '"ip7_":' in yaml_content, "ip7_ should be in the YAML file as separate entry"
    assert '"ip7__":' in yaml_content, "ip7__ should be in the YAML file as separate entry"

    read_content = read_metadata_file(yaml_file)
    expected_read = _convert_to_read_format(content)
    assert read_content == expected_read, "Content should be preserved after round-trip"
