"""
Test to verify device ID normalization fix for _raw directory files.

This test verifies that device IDs extracted from _raw directory files are properly
normalized, ensuring that files like #01.txt produce normalized device IDs like i1
instead of non-normalized i01.

The bug was that Pattern 2 in extract_device_id_from_raw_file_name() didn't normalize
device IDs when matching files with @# prefix but no device type (e.g., #01.txt).
"""

import pytest
from pathlib import Path
from meta_finder.file_finder import extract_device_id_from_raw_file_name


class TestDeviceIdNormalizationFix:
    """Test device ID normalization for _raw directory files."""

    @pytest.mark.parametrize(
        "file_name, expected_device_id, description",
        [
            # Pattern 2: Files with @# prefix but no device type (the bug case)
            ("#01.txt", "i1", "file with # prefix and zero-padded number"),
            ("#04.txt", "i4", "file with # prefix and zero-padded number"),
            ("#06.txt", "i6", "file with # prefix and zero-padded number"),
            ("#07.txt", "i7", "file with # prefix and zero-padded number"),
            ("#11.txt", "i11", "file with # prefix and non-zero-padded number"),
            ("#13.txt", "i13", "file with # prefix and non-zero-padded number"),
            ("@01.txt", "i1", "file with @ prefix and zero-padded number"),
            ("@04.txt", "i4", "file with @ prefix and zero-padded number"),
            # Pattern 1: Files with device type (already working)
            ("i01.txt", "i1", "file with device type i and zero-padded number"),
            ("i04.txt", "i4", "file with device type i and zero-padded number"),
            ("w01.txt", "w1", "file with device type w and zero-padded number"),
            ("w04.txt", "w4", "file with device type w and zero-padded number"),
            ("ib27.txt", "ib27", "file with device type ib and number"),
            ("ip06.txt", "ip6", "file with device type ip and zero-padded number"),
            # Subdirectory paths with # prefix but no device type are not matched
            # by extract_device_ids_from_prefixed_name (needs type prefix)
            # These are handled by the _raw directory walker using parent dirs
            # ("t1_k01/#01.txt", None, "file in subdirectory with # prefix but no device type"),
            # ("t3_k01/#01.txt", None, "file in subdirectory with # prefix but no device type"),
            # Edge cases
            ("#1.txt", "i1", "file with # prefix and single digit"),
            ("#100.txt", "i100", "file with # prefix and three-digit number"),
            # Date prefix + #number without type: not matched by extract_device_ids_from_prefixed_name
            # ("130510#01.txt", "i1", "file with date prefix and #01"),
        ],
    )
    def test_extract_device_id_normalization(
        self, file_name: str, expected_device_id: str, description: str
    ):
        """
        Test that device IDs extracted from _raw file names are properly normalized.

        This test verifies the fix for the bug where Pattern 2 in
        extract_device_id_from_raw_file_name() didn't normalize device IDs,
        causing files like #01.txt to produce non-normalized i01 instead of i1.

        Args:
            file_name: The file name to extract device ID from
            expected_device_id: The expected normalized device ID
            description: Description of the test case for clarity
        """
        result = extract_device_id_from_raw_file_name(file_name)

        assert result == expected_device_id, (
            f"{description}: expected '{expected_device_id}' but got '{result}' "
            f"for file '{file_name}'"
        )

    @pytest.mark.parametrize(
        "file_name, description",
        [
            ("1.txt", "file without prefix or device type"),
            ("data.txt", "file with no device information"),
            ("test_123.txt", "file with text but no device pattern"),
            ("", "empty file name"),
        ],
    )
    def test_extract_device_id_no_match(self, file_name: str, description: str):
        """
        Test that files without valid device patterns return None.

        Args:
            file_name: The file name to extract device ID from
            description: Description of the test case for clarity
        """
        result = extract_device_id_from_raw_file_name(file_name)

        assert result is None, (
            f"{description}: expected None but got '{result}' for file '{file_name}'"
        )

    def test_normalization_consistency_across_sources(self):
        """
        Test that device IDs from different sources normalize consistently.

        This test verifies that the same device ID extracted from different
        file name patterns results in the same normalized device ID.
        """
        # All these should normalize to the same device ID: i1
        test_cases = [
            ("#01.txt", "Pattern 2 with # prefix"),
            ("@01.txt", "Pattern 2 with @ prefix"),
            ("i01.txt", "Pattern 1 with device type"),
            ("i1.txt", "Pattern 1 without zero padding"),
        ]

        results = []
        for file_name, description in test_cases:
            result = extract_device_id_from_raw_file_name(file_name)
            results.append((file_name, description, result))

        # All results should be the same normalized device ID
        expected = "i1"
        for file_name, description, result in results:
            assert result == expected, (
                f"{description}: expected '{expected}' but got '{result}' for file '{file_name}'"
            )
