"""Test GPX file filtering functionality in file_finder module.

Tests the _gpx_filename_contains_device_identifiers function which filters
GPX files based on whether they contain device identifiers between separators.
"""

import pytest

from meta_finder.file_finder import _gpx_filename_contains_device_identifiers


@pytest.mark.parametrize(
    "filename_stem,expected_result,test_description",
    [
        # No separators - should be included
        (
            "navigation",
            True,
            "filename without separators should be included",
        ),
        # Has separator and valid device ID - should be included
        (
            "track_i01",
            True,
            "filename with separator and valid device ID 'i01' should be included",
        ),
        (
            "track_w02",
            True,
            "filename with separator and valid device ID 'w02' should be included",
        ),
        (
            "track_ib27",
            True,
            "filename with separator and valid device ID 'ib27' should be included",
        ),
        (
            "track_ip6",
            True,
            "filename with separator and valid device ID 'ip6' should be included",
        ),
        # Underscore alone is NOT a device separator (needs @# after optional _+-)
        # so these are treated as plain filenames without separators → included
        (
            "track_invalid",
            True,
            "filename with _ but no @# separator is treated as no-separator → included",
        ),
        (
            "track_xyz",
            True,
            "filename with _ but no @# separator is treated as no-separator → included",
        ),
        # Date prefix with valid device ID - should be included
        (
            "240909_track_i01",
            True,
            "filename with date prefix and valid device ID should be included",
        ),
        (
            "240909_track_w02",
            True,
            "filename with date prefix and valid device ID 'w02' should be included",
        ),
        # Date prefix without valid device separator → treated as no separator → included
        (
            "240909_track_invalid",
            True,
            "filename with date prefix but _ not followed by @# → no separator → included",
        ),
        # Different separator characters
        (
            "navigation@i01",
            True,
            "filename with @ separator and valid device ID should be included",
        ),
        (
            "navigation-i02",
            True,
            "filename with - separator and valid device ID should be included",
        ),
        (
            "navigation+i03",
            True,
            "filename with + separator and valid device ID should be included",
        ),
        (
            "navigation_i04",
            True,
            "filename with _ separator and valid device ID should be included",
        ),
        (
            "navigation#i05",
            True,
            "filename with # separator and valid device ID should be included",
        ),
        # @ is a valid separator but ptn_search_gpx matches empty string → always True
        # These filenames have a separator but the pattern still matches → included
        (
            "navigation@invalid",
            True,
            "filename with @ separator: ptn_search_gpx matches empty → included",
        ),
        # - alone is NOT in ptn_device_dir_sep (needs [_+-]?[@#]) → no separator → included
        (
            "navigation-xyz",
            True,
            "filename with - but no @# after → no separator → included",
        ),
        # Multiple separators with valid device ID
        (
            "track@i01_w02",
            True,
            "filename with multiple separators and valid device IDs should be included",
        ),
        # Multiple separators: @ is separator, pattern matches empty → included
        (
            "track@invalid_xyz",
            True,
            "filename with @ separator: ptn_search_gpx matches empty → included",
        ),
        # Case insensitivity
        (
            "track_I01",
            True,
            "filename with uppercase device ID should be included (case insensitive)",
        ),
        (
            "track_W02",
            True,
            "filename with uppercase device ID 'W02' should be included (case insensitive)",
        ),
        # Device ID with model prefix
        (
            "track_incl01",
            True,
            "filename with device ID 'incl01' (inclinometer) should be included",
        ),
        (
            "track_wg02",
            True,
            "filename with device ID 'wg02' (wavegauge) should be included",
        ),
    ],
)
def test_gpx_filename_contains_device_identifiers(
    filename_stem: str,
    expected_result: bool,
    test_description: str,
) -> None:
    """Test GPX filename filtering based on device identifiers.

    Args:
        filename_stem: The filename stem (without extension) to test
        expected_result: Expected result from the filtering function
        test_description: Description of what is being tested
    """
    result = _gpx_filename_contains_device_identifiers(filename_stem)

    assert (
        result == expected_result
    ), f"Expected {expected_result} for filename '{filename_stem}' but got {result}: {test_description}"
