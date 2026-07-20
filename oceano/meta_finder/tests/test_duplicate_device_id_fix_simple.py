"""
Simple test to verify that duplicate device IDs are not created.

This test specifically addresses the bug where devices from info_devices.yaml
would be added as duplicates when they already exist in the merged content.
"""
import pytest
from pathlib import Path
from meta_finder.create_info_files import _merge_device_metadata


def test_no_duplicate_device_ids_simple():
    """
    Test that _merge_device_metadata does not create duplicate device IDs.

    This is the core bug fix: when merging existing content with new content,
    devices that exist in both should not be duplicated.
    """
    # Existing content with some devices
    existing_content = {
        "i03": ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"],
        "i04": ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"],
    }

    # New content with the SAME devices (updated values)
    new_content = {
        "i03": ["1", "10", "5", "", "55.5", "21.5", "2023-01-01 00:00:00",
                 "2023-01-02 00:00:00", "300", "10", "Test"],
        "i04": ["2", "15", "7", "", "55.6", "21.6", "2023-01-01 00:00:00",
                 "2023-01-02 00:00:00", "300", "10", "Test2"],
        "w01": ["3", "20", "10", "", "55.7", "21.7", "2023-01-01 00:00:00",
                 "2023-01-02 00:00:00", "600", "5", "Test3"],
    }

    # Merge with normalize_keys=False (as when merging from existing info_devices@meta_finder.yaml)
    merged = _merge_device_metadata(
        existing_content,
        new_content,
        normalize_keys=False
    )

    # Verify no duplicates - this is the key test for the bug fix
    assert len(merged) == 3, (
        f"Expected 3 devices after merge, got {len(merged)}. "
        f"Devices: {list(merged.keys())}. "
        f"BUG: Devices were duplicated!"
    )

    # Verify that merged content has the new values
    # Merged devices get nested station dict structure: {"0": [values]}
    assert merged["i03"]["0"][0] == "1", "i03 should have new point value"
    assert merged["i04"]["0"][0] == "2", "i04 should have new point value"
    assert merged["w01"][0] == "3", "w01 should have new point value (new device, flat list)"


def test_new_devices_added_at_end():
    """
    Test that new devices (not in existing) are added at the end.
    """
    # Existing content with 2 devices
    existing_content = {
        "i03": ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"],
        "i04": ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"],
    }

    # New content with 3 devices (w01 is new)
    new_content = {
        "i03": ["1", "10", "5", "", "55.5", "21.5", "2023-01-01 00:00:00",
                 "2023-01-02 00:00:00", "300", "10", "Test"],
        "i04": ["2", "15", "7", "", "55.6", "21.6", "2023-01-01 00:00:00",
                 "2023-01-02 00:00:00", "300", "10", "Test2"],
        "w01": ["3", "20", "10", "", "55.7", "21.7", "2023-01-01 00:00:00",
                 "2023-01-02 00:00:00", "600", "5", "Test3"],
    }

    # Merge
    merged = _merge_device_metadata(
        existing_content,
        new_content,
        normalize_keys=False
    )

    # Verify all 3 devices are present
    assert len(merged) == 3, (
        f"Expected 3 devices after merge, got {len(merged)}"
    )

    # Verify order: existing devices first, then new devices
    device_order = list(merged.keys())
    assert device_order[0] == "i03", "i03 should be first (existing)"
    assert device_order[1] == "i04", "i04 should be second (existing)"
    assert device_order[2] == "w01", "w01 should be third (new)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
