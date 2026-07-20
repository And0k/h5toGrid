"""
Test device ID extraction from subdirectory names as fallback when filenames don't contain device information.
"""

import pytest
from pathlib import Path, PurePosixPath
import tempfile
import shutil

from meta_finder.file_finder import extract_devices_from_text_output, _extract_device_id_from_subdirectory_name


@pytest.mark.parametrize(
    "dir_path, rel_path, expected_device_ids, test_id",
    [
        # Test case 1: Device ID in subdirectory name (i02)
        (Path("text_output/i02"), PurePosixPath("130510_1100-10_2319.txt"), ["i2"], "subdirectory_i02"),
        # Test case 2: Device ID in subdirectory name (W03)
        (Path("text_output/W03"), PurePosixPath("130510_1100-10_2319.txt"), ["w3"], "subdirectory_W03"),
        # Test case 3: Device ID with @ prefix in subdirectory name
        (Path("text_output/@i04"), PurePosixPath("130510_1100-10_2319.txt"), ["i4"], "subdirectory_at_i04"),
        # Test case 4: Device ID with model prefix (ib27)
        (Path("text_output/ib27"), PurePosixPath("130510_1100-10_2319.txt"), ["ib27"], "subdirectory_ib27"),
        # Test case 5: No device ID in subdirectory name
        (Path("text_output/data"), PurePosixPath("130510_1100-10_2319.txt"), [], "no_device_id"),
        # Test case 6: File without subdirectory (just filename)
        (Path("text_output"), PurePosixPath("130510_1100-10_2319.txt"), [], "no_subdirectory"),
    ],
)
def test_extract_device_id_from_subdirectory_name(dir_path, rel_path, expected_device_ids, test_id):
    """Test extraction of device IDs from subdirectory names as fallback."""
    result = _extract_device_id_from_subdirectory_name(dir_path, rel_path)
    assert result == expected_device_ids, (
        f"Test {test_id}: Expected device IDs {expected_device_ids}, "
        f"but got {result} for dir_path={dir_path}, rel_path={rel_path}"
    )


def test_extract_devices_from_text_output_with_subdirectory_fallback():
    """Test device extraction from text_output directories with subdirectory fallback."""
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory structure:
        # temp_dir/
        #   device_dir/
        #     text_output/
        #       i02/
        #         130510_1100-10_2319.txt
        #       W03/
        #         130510_1100-10_2319.txt

        device_dir = temp_path / "device_dir"
        text_output_dir = device_dir / "text_output"
        i02_dir = text_output_dir / "i02"
        w03_dir = text_output_dir / "W03"

        # Create directories
        i02_dir.mkdir(parents=True)
        w03_dir.mkdir(parents=True)

        # Create test files with filenames that don't contain device information
        i02_file = i02_dir / "130510_1100-10_2319.txt"
        w03_file = w03_dir / "130510_1100-10_2319.txt"

        # Write simple content to files with proper column names
        i02_file.write_text("Time\tInclination_i2\n")
        w03_file.write_text("Time\tInclination_w3\n")

        # Extract devices from text_output
        result = extract_devices_from_text_output(device_dir)

        # Verify that devices were extracted from subdirectory names
        assert "i2" in result, "Expected device i2 to be extracted from i02 subdirectory"
        assert "w3" in result, "Expected device w3 to be extracted from W03 subdirectory"

        # Verify that files are associated with correct devices
        i2_files = result["i2"]
        w3_files = result["w3"]

        assert len(i2_files) == 1, f"Expected 1 file for device i2, got {len(i2_files)}"
        assert len(w3_files) == 1, f"Expected 1 file for device w3, got {len(w3_files)}"

        # Verify file paths
        i2_dir_path, i2_rel_path = i2_files[0]
        w3_dir_path, w3_rel_path = w3_files[0]

        assert i2_rel_path.name == "130510_1100-10_2319.txt", (
            f"Expected file 130510_1100-10_2319.txt for device i2, got {i2_rel_path.name}"
        )
        assert w3_rel_path.name == "130510_1100-10_2319.txt", (
            f"Expected file 130510_1100-10_2319.txt for device w3, got {w3_rel_path.name}"
        )


def test_extract_devices_from_text_output_with_mixed_scenarios():
    """Test device extraction with mixed scenarios: some files have device in filename, some don't."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory structure:
        # temp_dir/
        #   device_dir/
        #     text_output/
        #       i02/
        #         130510_1100-10_2319.txt  # No device in filename, use subdirectory
        #       130510_i03_1100-10_2319.txt  # Device in filename, use filename

        device_dir = temp_path / "device_dir"
        text_output_dir = device_dir / "text_output"
        i02_dir = text_output_dir / "i02"

        # Create directories
        i02_dir.mkdir(parents=True)

        # Create test files
        i02_file = i02_dir / "130510_1100-10_2319.txt"
        i03_file = text_output_dir / "130510_1100-10_2319_i03.txt"

        # Write content with proper column names
        i02_file.write_text("Time\tInclination_i2\n")
        i03_file.write_text("Time\tInclination_i3\n")

        # Extract devices from text_output
        result = extract_devices_from_text_output(device_dir)

        # Verify both devices were extracted
        assert "i2" in result, "Expected device i2 to be extracted from i02 subdirectory"
        assert "i3" in result, "Expected device i3 to be extracted from filename"

        # Verify files are associated with correct devices
        i2_files = result["i2"]
        i3_files = result["i3"]

        assert len(i2_files) == 1, f"Expected 1 file for device i2, got {len(i2_files)}"
        assert len(i3_files) == 1, f"Expected 1 file for device i3, got {len(i3_files)}"


def test_extract_devices_from_text_output_no_fallback_needed():
    """Test that fallback is not used when filename already contains device information."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory structure:
        # temp_dir/
        #   device_dir/
        #     text_output/
        #       130510_i05_1100-10_2319.txt  # Device in filename

        device_dir = temp_path / "device_dir"
        text_output_dir = device_dir / "text_output"

        # Create directory
        text_output_dir.mkdir(parents=True)

        # Create test file with device in filename
        i05_file = text_output_dir / "130510_1100-10_2319_i05.txt"
        i05_file.write_text("Time\tInclination_i5\n")

        # Extract devices from text_output
        result = extract_devices_from_text_output(device_dir)

        # Verify device was extracted from filename
        assert "i5" in result, "Expected device i5 to be extracted from filename"
        assert len(result["i5"]) == 1, f"Expected 1 file for device i5, got {len(result['i5'])}"
