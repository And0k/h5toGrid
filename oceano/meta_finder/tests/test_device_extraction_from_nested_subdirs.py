"""Test device ID extraction from nested subdirectories.

This test verifies that device IDs can be correctly extracted from nested
directory structures like text_output/130510/i03/file.txt where the device
directory (i03) is inside a date directory (130510).
"""

import pytest
from pathlib import Path, PurePosixPath
from meta_finder.file_finder import _extract_device_id_from_subdirectory_name


@pytest.mark.parametrize(
    "dir_path, rel_path, expected_device",
    [
        pytest.param(
            Path("/text_output/130510"),
            PurePosixPath("i03/130510_1100-10_2319.txt"),
            "i3",
            id="device_in_date_subdir",
        ),
        pytest.param(
            Path("/text_output/130511"),
            PurePosixPath("w01/130511_0000-11_2319.txt"),
            "w1",
            id="wavegauge_in_date_subdir",
        ),
        pytest.param(
            Path("/text_output/130512"),
            PurePosixPath("ib27/130512_0000-12_2319.txt"),
            "ib27",
            id="bottom_inclinometer_in_date_subdir",
        ),
        pytest.param(
            Path("/text_output/i03"),
            PurePosixPath("130510_1100-10_2319.txt"),
            "i3",
            id="device_direct_in_subdir",
        ),
        pytest.param(
            Path("/text_output/130510"),
            PurePosixPath("130510_1100-10_2319.txt"),
            [],
            id="date_subdir_no_device",
        ),
        pytest.param(
            Path("/text_output/130510"),
            PurePosixPath("other/130510_1100-10_2319.txt"),
            [],
            id="non_device_subdir",
        ),
    ],
)
def test_extract_device_id_from_nested_subdirs(dir_path, rel_path, expected_device):
    """Test device ID extraction from nested subdirectory structures.

    Args:
        dir_path: Full path to the directory containing the file
        rel_path: Relative path to the file
        expected_device: Expected device ID or empty list if no device found
    """
    result = _extract_device_id_from_subdirectory_name(dir_path, rel_path)

    if expected_device:
        assert result == [expected_device], (
            f"Expected device '{expected_device}' for rel_path '{rel_path}', "
            f"but got {result}"
        )
    else:
        assert result == [], (
            f"Expected no device for rel_path '{rel_path}', but got {result}"
        )
