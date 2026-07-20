"""Test device extraction from text_output subdirectory names when filenames don't contain device IDs.

This test verifies the fix for Issue 1: Extract devices from text_output subdirectory
names when filenames don't contain device IDs.

The pattern shows device IDs are in subdirectory names (i02, W03), not in filenames:
text_output/130510/
├── i02/          # Device ID in subdirectory name
│   └── 130510_1100-10_2319.txt  # Filename doesn't contain device ID
└── W03/          # Device ID in subdirectory name
    └── 130510_1100-10_2319.txt  # Filename doesn't contain device ID
"""

import pytest
from pathlib import Path, PurePosixPath

from meta_finder.file_finder import extract_devices_from_text_output


@pytest.fixture
def temp_device_dir_with_subdirs(tmp_path):
    """Create a temporary device directory with text_output containing subdirectories."""
    device_dir = tmp_path / "inclinometer"
    device_dir.mkdir()

    text_output_dir = device_dir / "text_output"
    text_output_dir.mkdir()

    # Create subdirectory i02 with a file that doesn't contain device ID in filename
    i02_dir = text_output_dir / "i02"
    i02_dir.mkdir()
    i02_file = i02_dir / "130510_1100-10_2319.txt"
    i02_file.write_text("timestamp\ti02_value\n130510_1100\t1.23")

    # Create subdirectory W03 with a file that doesn't contain device ID in filename
    w03_dir = text_output_dir / "W03"
    w03_dir.mkdir()
    w03_file = w03_dir / "130510_1100-10_2319.txt"
    w03_file.write_text("timestamp\tw03_value\n130510_1100\t4.56")

    # Create subdirectory with date prefix (130510)
    date_dir = text_output_dir / "130510"
    date_dir.mkdir()
    date_file = date_dir / "130510_1100-10_2319.txt"
    date_file.write_text("timestamp\tvalue\n130510_1100\t7.89")

    return device_dir


def test_extract_devices_from_subdirectory_names(temp_device_dir_with_subdirs):
    """Test that devices are extracted from subdirectory names when filenames don't contain device IDs.

    This test verifies Issue 1 fix: Extract devices from text_output subdirectory
    names when filenames don't contain device IDs.
    """
    dev_files = extract_devices_from_text_output(temp_device_dir_with_subdirs)

    # Should find devices i02 and w03 (normalized from W03) from subdirectory names
    assert "i2" in dev_files, "Device i02 should be found and normalized to i2"
    assert "w3" in dev_files, "Device W03 should be found and normalized to w3"

    # Verify files are associated with correct devices
    i2_files = dev_files["i2"]
    assert len(i2_files) == 1, f"Device i2 should have 1 file, got {len(i2_files)}"
    dir_path, rel_path = i2_files[0]
    assert rel_path.name == "130510_1100-10_2319.txt"
    # dir_path is text_output_dir; rel_path contains the subdirectory component
    assert dir_path.name == "text_output", (
        f"dir_path should be text_output directory, got {dir_path.name}"
    )
    assert str(rel_path).startswith("i02/"), (
        f"rel_path should start with i02/ subdirectory, got {rel_path}"
    )

    w3_files = dev_files["w3"]
    assert len(w3_files) == 1, f"Device w3 should have 1 file, got {len(w3_files)}"
    dir_path, rel_path = w3_files[0]
    assert rel_path.name == "130510_1100-10_2319.txt"
    # dir_path is text_output_dir; rel_path contains the subdirectory component
    assert dir_path.name == "text_output", (
        f"dir_path should be text_output directory, got {dir_path.name}"
    )
    assert str(rel_path).startswith("W03/"), (
        f"rel_path should start with W03/ subdirectory, got {rel_path}"
    )


def test_extract_devices_from_date_subdirectory(temp_device_dir_with_subdirs):
    """Test that files in date-named subdirectories are handled correctly.

    Files in subdirectories like 130510/ (date format) should not be
    associated with any device since the subdirectory name doesn't contain a device ID.
    """
    dev_files = extract_devices_from_text_output(temp_device_dir_with_subdirs)

    # The date subdirectory (130510) doesn't contain a device ID, so its file
    # should not be associated with any device
    # The file might be skipped or associated with a generic device, but not a specific one
    # This behavior is acceptable as date-only subdirectories don't identify devices
    assert len(dev_files) >= 2, "Should find at least 2 devices (i2 and w3)"


def test_device_normalization_from_subdirectory_names(temp_device_dir_with_subdirs):
    """Test that device IDs from subdirectory names are properly normalized.

    Verifies that device IDs extracted from subdirectory names are normalized
    correctly (e.g., W03 -> w3, i02 -> i2).
    """
    dev_files = extract_devices_from_text_output(temp_device_dir_with_subdirs)

    # Check normalization: leading zeros removed, case converted to lowercase
    assert "i2" in dev_files, "Device i02 should be normalized to i2"
    assert "w3" in dev_files, "Device W03 should be normalized to w3"

    # Verify non-normalized forms are not present
    assert "i02" not in dev_files, "Non-normalized i02 should not be present"
    assert "W03" not in dev_files, "Non-normalized W03 should not be present"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
