"""
Test that files with invalid filenames (like Vlegend.txt) are skipped.
"""

import pytest
from pathlib import Path
import tempfile

from meta_finder.file_finder import extract_devices_from_text_output


def test_skip_files_without_datetime():
    """Test that files without datetime in filename are skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory structure:
        # temp_dir/
        #   device_dir/
        #     text_output/
        #       i02/
        #         130510_1100-10_2319.txt  # Valid file (has datetime)
        #         Vlegend.txt  # Invalid file (no datetime)

        device_dir = temp_path / "device_dir"
        text_output_dir = device_dir / "text_output"
        i02_dir = text_output_dir / "i02"

        # Create directories
        i02_dir.mkdir(parents=True)

        # Create test files
        valid_file = i02_dir / "130510_1100-10_2319.txt"
        valid_file.write_text("Time\tInclination_i2\n")

        invalid_file = i02_dir / "Vlegend.txt"
        invalid_file.write_text("Legend information\n")

        # Extract devices from text_output
        result = extract_devices_from_text_output(device_dir)

        # Verify that only the valid file was processed
        assert "i2" in result, "Expected device i2 to be extracted from valid file"

        # Verify that only 1 file is associated with device i2 (not 2)
        i2_files = result["i2"]
        assert len(i2_files) == 1, f"Expected 1 file for device i2, got {len(i2_files)}"

        # Verify that the file is the valid one
        dir_path, rel_path = i2_files[0]
        assert rel_path.name == "130510_1100-10_2319.txt", (
            f"Expected valid file 130510_1100-10_2319.txt, got {rel_path.name}"
        )


def test_skip_files_without_datetime_in_root_text_output():
    """Test that files without datetime in root text_output are skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directory structure:
        # temp_dir/
        #   device_dir/
        #     text_output/
        #       130510_1100-10_2319.txt  # Valid file (has datetime)
        #       Vlegend.txt  # Invalid file (no datetime)

        device_dir = temp_path / "device_dir"
        text_output_dir = device_dir / "text_output"

        # Create directory
        text_output_dir.mkdir(parents=True)

        # Create test files
        valid_file = text_output_dir / "130510_1100-10_2319.txt"
        valid_file.write_text("Time\tInclination_i2\n")

        invalid_file = text_output_dir / "Vlegend.txt"
        invalid_file.write_text("Legend information\n")

        # Extract devices from text_output
        result = extract_devices_from_text_output(device_dir)

        # Verify that the valid file was processed
        # File has datetime and device info in column name (Inclination_i2), so it should be processed
        assert "i2" in result, "Expected device i2 to be extracted from file content"

        # Verify that only 1 file is associated with device i2
        i2_files = result["i2"]
        assert len(i2_files) == 1, f"Expected 1 file for device i2, got {len(i2_files)}"

        # Verify that the file is the valid one
        dir_path, rel_path = i2_files[0]
        assert rel_path.name == "130510_1100-10_2319.txt", (
            f"Expected valid file 130510_1100-10_2319.txt, got {rel_path.name}"
        )
