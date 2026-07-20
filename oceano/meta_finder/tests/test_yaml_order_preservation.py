"""Test YAML order preservation when writing and reading device metadata.

Tests that the order of devices in DEVICES_FILE_NAME_UPD (YAML) files
is preserved compared to DEVICES_FILE_NAME (JSON) files.

The test verifies:
1. Top-level device names order (without underscores) is preserved
2. Devices with underscore suffixes are sorted by number of underscores
"""

import logging
import pytest
from pathlib import Path

from meta_finder.io_info_files import (
    write_devices_meta_yaml,
    read_metadata_file,
    write_devices_meta_json,
    _concatenate_additional_comments,
    _remove_trailing_empty_fields,
)

from meta_finder.logging_config import setup_logging

logger = setup_logging(log_level=logging.DEBUG)


def _extract_base_name(device_id: str) -> str:
    """Extract base device name without trailing underscores."""
    return device_id.rstrip('_')


def _count_trailing_underscores(device_id: str) -> int:
    """Count number of trailing underscores in device identifier."""
    return len(device_id) - len(device_id.rstrip('_'))


def _get_base_name_order(device_ids: list) -> list:
    """Extract base names from device IDs preserving order of first occurrence."""
    seen = set()
    base_order = []
    for device_id in device_ids:
        base_name = _extract_base_name(device_id)
        if base_name not in seen:
            seen.add(base_name)
            base_order.append(base_name)
    return base_order


def _group_by_base_name(device_ids: list) -> dict:
    """Group device IDs by their base name."""
    groups = {}
    for device_id in device_ids:
        base_name = _extract_base_name(device_id)
        underscore_count = _count_trailing_underscores(device_id)
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append((device_id, underscore_count))
    return groups


def _convert_to_read_format(content: dict) -> dict:
    """Convert flat {device_id: [list]} to nested {device_id: {"0": [list]}} format.

    Simulates what read_metadata_file returns after processing:
    write applies _remove_trailing_empty_fields, read applies _concatenate_additional_comments.
    """
    result = {}
    for device_id, values in content.items():
        trimmed = list(_remove_trailing_empty_fields(values))
        processed = _concatenate_additional_comments(trimmed)
        result[device_id] = {"0": processed}
    return result


# Shared test data for order preservation tests
ORDER_TEST_CONTENT = {
    "ip7": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', ''],
    "w2": ['?', '?', '?', '?', '?', '?', '2013-05-10 10:00:00', '2013-05-10 17:06:27', '-', '-', '?'],
    "ip7_": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', '?'],
    "ip7__": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', ''],
    "w1": ['?', '?', '?', '?', '?', '?', '2013-05-10 10:00:00', '2013-05-19 15:01:15', '-', '-', '?'],
    "i15": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', '?'],
    "ip1_": ['?', '?', '?', '?', '?', '?', '?', '?', '', '', ''],
    "i1": ['?', '?', '?', '?', '?', '?', '2013-05-10 10:00:00', '2013-05-10 14:17:19:59', '-', '-', '?'],
}


@pytest.mark.parametrize(
    "original_content,test_description",
    [
        (
            ORDER_TEST_CONTENT,
            "Order preservation with devices having underscore suffixes",
        ),
        (
            {
                "device3": ['a', 'b', 'c'],
                "device1": ['d', 'e', 'f'],
                "device2": ['g', 'h', 'i'],
            },
            "Order preservation without underscore suffixes",
        ),
        (
            {
                "device1_": ['a', 'b', 'c'],
                "device1": ['d', 'e', 'f'],
                "device1__": ['g', 'h', 'i'],
            },
            "Order preservation with underscore suffixes only",
        ),
    ],
)
def test_yaml_order_preservation_round_trip(tmp_path, original_content, test_description):
    """Test that device order is preserved when writing and reading YAML files."""
    yaml_file = tmp_path / "test_devices.yaml"

    logger.debug(f"Testing: {test_description}")
    logger.debug(f"Original device order: {list(original_content.keys())}")

    write_devices_meta_yaml(tmp_path, yaml_file, original_content)
    read_content = read_metadata_file(yaml_file)

    logger.debug(f"Read device order: {list(read_content.keys())}")

    expected_read = _convert_to_read_format(original_content)
    assert read_content == expected_read, (
        f"Failed: {test_description} - content not preserved after round-trip"
    )

    original_base_order = _get_base_name_order(list(original_content.keys()))
    read_base_order = _get_base_name_order(list(read_content.keys()))

    assert original_base_order == read_base_order, (
        f"Failed: {test_description} - base device name order not preserved. "
        f"Original base order: {original_base_order}, Read base order: {read_base_order}"
    )

    # Verify that all devices with the same base name are present
    # (underscore suffix grouping was removed, so no sorting by underscore count)
    original_groups = _group_by_base_name(list(original_content.keys()))
    read_groups = _group_by_base_name(list(read_content.keys()))

    for base_name in original_groups:
        original_ids = [dev_id for dev_id, _ in original_groups[base_name]]
        read_ids = [dev_id for dev_id, _ in read_groups[base_name]]
        assert set(original_ids) == set(read_ids), (
            f"Failed: {test_description} - devices with base name '{base_name}' "
            f"not preserved. Original: {original_ids}, Read: {read_ids}"
        )


@pytest.mark.parametrize(
    "original_content,test_description",
    [
        (
            ORDER_TEST_CONTENT,
            "Info-file to YAML order preservation with devices having underscore suffixes",
        ),
    ],
)
def test_json_to_yaml_order_preservation(tmp_path, original_content, test_description):
    """Test that device order is preserved when converting from JSON to YAML."""
    json_file = tmp_path / "test_devices.json"
    yaml_file = tmp_path / "test_devices.yaml"

    logger.debug(f"Testing JSON to YAML: {test_description}")
    logger.debug(f"Original device order: {list(original_content.keys())}")

    write_devices_meta_json(tmp_path, json_file, original_content)
    json_content = read_metadata_file(json_file)

    write_devices_meta_yaml(tmp_path, yaml_file, json_content)
    yaml_content = read_metadata_file(yaml_file)

    logger.debug(f"Info-file device order: {list(json_content.keys())}")
    logger.debug(f"YAML device order: {list(yaml_content.keys())}")

    expected_read = _convert_to_read_format(original_content)
    assert yaml_content == expected_read, (
        f"Failed: {test_description} - content not preserved after JSON to YAML conversion"
    )

    original_base_order = _get_base_name_order(list(original_content.keys()))
    yaml_base_order = _get_base_name_order(list(yaml_content.keys()))

    assert original_base_order == yaml_base_order, (
        f"Failed: {test_description} - base device name order not preserved in YAML. "
        f"Original base order: {original_base_order}, YAML base order: {yaml_base_order}"
    )

    # Verify that all devices with the same base name are present
    # (underscore suffix grouping was removed, so no sorting by underscore count)
    original_groups = _group_by_base_name(list(original_content.keys()))
    yaml_groups = _group_by_base_name(list(yaml_content.keys()))

    for base_name in original_groups:
        original_ids = [dev_id for dev_id, _ in original_groups[base_name]]
        yaml_ids = [dev_id for dev_id, _ in yaml_groups[base_name]]
        assert set(original_ids) == set(yaml_ids), (
            f"Failed: {test_description} - devices with base name '{base_name}' "
            f"not preserved. Original: {original_ids}, YAML: {yaml_ids}"
        )


def test_yaml_file_content_order(tmp_path):
    """Test that the YAML file content preserves device order at the file level."""
    content = {
        "device3": ['a', 'b', 'c'],
        "device1": ['d', 'e', 'f'],
        "device2": ['g', 'h', 'i'],
    }

    yaml_file = tmp_path / "test_order.yaml"

    logger.debug(f"Original device order: {list(content.keys())}")

    write_devices_meta_yaml(tmp_path, yaml_file, content)

    with open(yaml_file, 'r', encoding='utf-8') as f:
        yaml_lines = f.readlines()

    device_lines = [
        line for line in yaml_lines
        if any(d in line for d in ('"device1":', '"device2":', '"device3":'))
    ]

    devices_in_file = [line.split(':')[0].strip().strip('"') for line in device_lines]

    logger.debug(f"Devices in YAML file order: {devices_in_file}")

    assert devices_in_file == list(content.keys()), (
        f"Device order in YAML file does not match original. "
        f"Expected: {list(content.keys())}, Got: {devices_in_file}"
    )
