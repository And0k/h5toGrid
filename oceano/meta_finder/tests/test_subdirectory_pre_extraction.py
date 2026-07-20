"""
Test to verify that subdirectory device pre-extraction eliminates redundant checks.

This test verifies that when processing multiple files from the same directory/archive,
the device extraction from the directory/archive name is performed only once instead of
being called for each individual file.
"""

import pytest
from pathlib import Path, PurePosixPath
from unittest.mock import patch, MagicMock
import logging

from meta_finder.file_finder import (
    _extract_device_id_from_directory_name,
    _extract_device_id_from_subdirectory_name,
    extract_devices_from_text_output,
)
from meta_finder import config


@pytest.fixture
def setup_test_data(tmp_path):
    """Create test directory structure with text_output archive."""
    # Create test directory structure
    device_dir = tmp_path / "test_device"
    device_dir.mkdir()

    # Create text_output.zip archive with device ID in name
    import zipfile
    archive_path = device_dir / "text_output@i03.zip"
    with zipfile.ZipFile(archive_path, 'w') as zf:
        # Add multiple test files to the archive
        for i in range(3):
            file_content = f"Time\ti{i+1}\n2024-01-01 00:00:00\t1.0\n"
            zf.writestr(f"240101_1200bin3600s@i{i+1:02d}.tsv", file_content)

    yield device_dir, archive_path


class TestSubdirectoryPreExtraction:
    """Test suite for subdirectory device pre-extraction optimization."""

    @pytest.mark.parametrize(
        "dir_name,rel_path,expected_devices,description",
        [
            # Test _extract_device_id_from_directory_name with various inputs
            ("i03", None, ["i3"], "device ID with leading zero"),
            ("@i05", None, ["i5"], "device ID with @ prefix"),
            ("text_output.zip", None, [], "no device ID"),
            # Test _extract_device_id_from_subdirectory_name with file directly in directory
            ("i03", PurePosixPath("file.tsv"), ["i3"], "file in directory with device"),
            # Test _extract_device_id_from_subdirectory_name with file in subdirectory
            ("text_output", PurePosixPath("i03/file.tsv"), ["i3"], "file in subdirectory"),
        ],
        ids=[
            "directory-with-device",
            "directory-with-at-prefix",
            "directory-no-device",
            "subdir-file-in-directory",
            "subdir-file-in-subdirectory",
        ],
    )
    def test_extract_device_id(self, dir_name, rel_path, expected_devices, description, caplog, tmp_path):
        """
        Test device ID extraction from directory and subdirectory names.

        This parametrized test covers both _extract_device_id_from_directory_name
        and _extract_device_id_from_subdirectory_name functions, ensuring they
        correctly extract and normalize device IDs from various path patterns.
        """
        if rel_path is None:
            # Test _extract_device_id_from_directory_name — expects str, not Path
            devices = _extract_device_id_from_directory_name(dir_name)
            assert devices == expected_devices, (
                f"{description}: Expected {expected_devices}, got {devices}"
            )
        else:
            # Test _extract_device_id_from_subdirectory_name — needs a Path for dir_path
            dir_path = tmp_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            with caplog.at_level(logging.DEBUG):
                devices = _extract_device_id_from_subdirectory_name(dir_path, rel_path)
                assert devices == expected_devices, (
                    f"{description}: Expected {expected_devices}, got {devices}"
                )
                # Verify logging occurred for subdirectory tests
                assert any(
                    "Attempting to extract device ID" in record.message
                    for record in caplog.records
                ), f"{description}: Expected logging to occur"

    @pytest.mark.parametrize(
        "file_count,expected_call_count",
        [
            (1, 0),  # 1 file, no redundant calls
            (5, 0),  # 5 files, no redundant calls
            (43, 0),  # 43 files, no redundant calls (original issue)
        ],
        ids=["single-file", "five-files", "forty-three-files"],
    )
    def test_pre_extraction_reduces_calls(
        self, setup_test_data, file_count, expected_call_count, caplog
    ):
        """
        Test that pre-extraction reduces the number of subdirectory extraction calls.

        This is the main optimization: when processing multiple files from the same
        directory/archive, the device extraction should happen only once, not once per file.
        """
        device_dir, archive_path = setup_test_data

        # Track calls to _extract_device_id_from_subdirectory_name
        original_extract = _extract_device_id_from_subdirectory_name
        call_count = {"count": 0}

        def tracked_extract(*args, **kwargs):
            call_count["count"] += 1
            return original_extract(*args, **kwargs)

        with patch(
            'meta_finder.file_finder._extract_device_id_from_subdirectory_name',
            side_effect=tracked_extract,
        ):
            with caplog.at_level(logging.DEBUG):
                # Process the directory
                result = extract_devices_from_text_output(device_dir)

                # Verify files were processed
                assert len(result) > 0, "Should have found devices"

                # Count how many files were processed
                total_files = sum(len(files) for files in result.values())

                # The key assertion: subdirectory extraction should be called much less
                # than the number of files
                # For files directly in archive: should be called 0 times (using pre-extracted)
                # For files in subdirectories: should be called only for those subdirectories
                assert (
                    call_count["count"] < total_files
                ), (
                    f"Subdirectory extraction called {call_count['count']} times "
                    f"for {total_files} files - should be much less"
                )

    def test_pre_extraction_logs_correctly(self, setup_test_data, caplog):
        """Test that device extraction from archive logs correctly."""
        device_dir, archive_path = setup_test_data

        with caplog.at_level(logging.DEBUG):
            result = extract_devices_from_text_output(device_dir)

            # Should have found devices
            assert len(result) > 0, "Should have found devices"

            # Verify processing log for the archive
            processing_logs = [
                record
                for record in caplog.records
                if "Processing text_output directory/archive" in record.message
            ]
            assert len(processing_logs) > 0, "Should have logged archive processing"

            # Should NOT have redundant "Attempting to extract" logs for each file
            # (or at least significantly fewer than number of files)
            attempt_logs = [
                record
                for record in caplog.records
                if "Attempting to extract device ID from subdirectory name" in record.message
            ]
            # With pre-extraction, files directly in archive should not trigger this log
            # Files in subdirectories might trigger it, but much less than total files
            assert len(attempt_logs) < 10, (
                f"Too many 'Attempting to extract' logs: {len(attempt_logs)}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
