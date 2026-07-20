"""Tests for device ID extraction from various sources.

Covers:
- extract_device_ids_from_prefixed_name(): HDF5 group names, directory names
- parse_device_id_groups(): parsing device group strings
- normalize_device_id(): normalizing device IDs
- _extract_device_ids_from_groups_in_data(): column name extraction
- _find_specific_device_in_path_parts(): path-based device resolution
- extract_device_id_from_raw_file_name(): raw file name extraction
"""

import re
import pytest
from pathlib import PurePosixPath

from meta_finder import config
from meta_finder.parse_data_file_name import (
    extract_device_ids_from_prefixed_name,
    parse_device_id_groups,
    normalize_device_id,
)
from meta_finder.data_proc_funcs import _extract_device_ids_from_groups_in_data
from meta_finder.file_finder import (
    _find_specific_device_in_path_parts,
    extract_device_id_from_raw_file_name,
)


class TestExtractDeviceIdsFromPrefixedName:
    """Test extract_device_ids_from_prefixed_name with HDF5 group names and directory names."""

    @pytest.mark.parametrize(
        "group_name, expected",
        [
            ("i3", ["i3"]),
            ("i03", ["i3"]),
            ("i54", ["i54"]),
            ("i54bin2s", ["i54"]),  # regression: was incorrectly i5 before fix
            ("i04bin2s", ["i4"]),  # normalize strips leading zero from i04
            ("ib27bin300s", ["ib27"]),
            ("w01", ["w1"]),
            ("w1", ["w1"]),
            ("p05", ["p5"]),
            ("ip6", ["ip6"]),
            ("i_b30", ["ib30"]),
            ("incl01", ["i1"]),
            ("wg02", ["w2"]),
            ("i_b(27,28,29,30)", ["ib27", "ib28", "ib29", "ib30"]),
            ("i(38,37,59,60,58)", ["i38", "i37", "i59", "i60", "i58"]),
            ("(i03,i04,w5)", ["i3", "i4", "w5"]),
            ("i03,i04,i05", ["i3", "i4", "i5"]),
            ("i03,4,5", ["i3", "i4", "i5"]),
            ("i03-5,6", ["i3", "i4", "i5", "i6"]),
            ("i_b20,i_b28,29,i_b40", ["ib20", "ib28", "ib29", "ib40"]),
            ("ib27-30", ["ib27", "ib28", "ib29", "ib30"]),
            ("incl_p30,incl_b01", ["ip30", "ib1"]),
            ("i", []),  # type-only, no number — validate=True by default skips it
            ("w", []),  # type-only, no number
            ("text_output", []),  # no device pattern
        ],
        ids=lambda v: repr(v) if isinstance(v, str) else str(v),
    )
    def test_hdf5_group_names(self, group_name, expected):
        """Device IDs extracted from HDF5 group names match expected values."""
        result = extract_device_ids_from_prefixed_name(group_name, msg_what="group name ")
        assert result == expected, (
            f"extract_device_ids_from_prefixed_name({group_name!r}) "
            f"returned {result!r}, expected {expected!r}"
        )

    @pytest.mark.parametrize(
        "dir_name, expected",
        [
            # text_output@i03: @ is separator, devices extracted after @ → i03 → i3
            ("text_output@i03", ["i3"]),
            # 230508_inclinometer@i03: @ is separator, devices extracted after @ → i03 → i3
            ("230508_inclinometer@i03", ["i3"]),
            # 201202P1-5,I1-2@i3,...: @ is present, devices extracted only after 1st @
            ("201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6",
             ["i3", "i5", "i9", "i10", "i11", "i15", "i19", "i23", "i28", "i30", "i32", "i33",
              "w1", "w2", "w3", "w4", "w5", "w6"]),
        ],
        ids=lambda v: repr(v) if isinstance(v, str) else str(v),
    )
    def test_directory_names(self, dir_name, expected):
        """Device IDs extracted from directory names match expected values."""
        result = extract_device_ids_from_prefixed_name(dir_name, msg_what="directory name ")
        assert result == expected, (
            f"extract_device_ids_from_prefixed_name({dir_name!r}) "
            f"returned {result!r}, expected {expected!r}"
        )


class TestParseDeviceIdGroups:
    """Test parse_device_id_groups function."""

    @pytest.mark.parametrize(
        "devices_str, validate, expected",
        [
            ("i(38,37),w(1,2)", False, ["i38", "i37", "w1", "w2"]),
            ("i_b20,i_b28,29,i_b40", False, ["ib20", "ib28", "ib29", "ib40"]),
            ("i03,i04,i05", True, ["i3", "i4", "i5"]),
            ("i03,4,5", True, ["i3", "i4", "i5"]),
            ("i03-5,6", True, ["i3", "i4", "i5", "i6"]),
            ("ib27-30", True, ["ib27", "ib28", "ib29", "ib30"]),
        ],
        ids=lambda v: repr(v) if isinstance(v, str) else str(v),
    )
    def test_parse_device_id_groups(self, devices_str, validate, expected):
        """Parsed device ID groups match expected normalized list."""
        result = parse_device_id_groups(devices_str, validate=validate)
        assert result == expected, (
            f"parse_device_id_groups({devices_str!r}, validate={validate}) "
            f"returned {result!r}, expected {expected!r}"
        )


class TestNormalizeDeviceId:
    """Test normalize_device_id function."""

    @pytest.mark.parametrize(
        "device_id, prefix, validate, expected",
        [
            ("i03", None, False, "i3"),
            ("I03", None, False, "i3"),
            ("w01", None, False, "w1"),
            ("ib27", None, False, "ib27"),
            ("i_b30", None, False, "ib30"),
            ("incl01", None, False, "i1"),
            ("wg02", None, False, "w2"),
            ("ip6", None, False, "ip6"),
            ("i03", "", False, "3"),  # prefix="" returns number only
            ("", None, False, None),  # empty string
            # 'i' matches ptn_device_type so prefix_found='i', validate doesn't reject it
            ("i", None, True, "i"),
            ("i", None, False, "i"),  # type-only with validate=False
        ],
        ids=lambda v: repr(v) if isinstance(v, str) else str(v),
    )
    def test_normalize_device_id(self, device_id, prefix, validate, expected):
        """Normalized device ID matches expected value."""
        result = normalize_device_id(device_id, prefix=prefix, validate=validate)
        assert result == expected, (
            f"normalize_device_id({device_id!r}, prefix={prefix!r}, validate={validate}) "
            f"returned {result!r}, expected {expected!r}"
        )


class TestExtractDeviceIdsFromGroupsInData:
    """Test _extract_device_ids_from_groups_in_data from data_proc_funcs."""

    @pytest.mark.parametrize(
        "devices_str, expected",
        [
            ("i11", ["i11"]),
            ("i03_14", ["i3", "i14"]),
            ("i05_14_27", ["i5", "i14", "i27"]),
            ("i03", ["i3"]),
            ("w01", ["w1"]),
            ("p05", ["p5"]),
            ("ib27", ["ib27"]),
            ("i_b30", ["ib30"]),
            ("ip6", ["ip6"]),
            ("incl01", ["i1"]),
            ("wg02", ["w2"]),
        ],
        ids=lambda v: repr(v) if isinstance(v, str) else str(v),
    )
    def test_extract_from_groups_in_data(self, devices_str, expected):
        """Device IDs extracted from column/group names match expected values."""
        result = _extract_device_ids_from_groups_in_data(devices_str)
        assert result == expected, (
            f"_extract_device_ids_from_groups_in_data({devices_str!r}) "
            f"returned {result!r}, expected {expected!r}"
        )


class TestFindSpecificDeviceInPathParts:
    """Test _find_specific_device_in_path_parts from file_finder.

    This function scans rel_path.parent.parts in reverse, calling
    _extract_device_id_from_directory_name on each part. Only returns
    a single specific device (excludes generic types i/w/p/*).
    """

    @pytest.mark.parametrize(
        "rel_path_str, expected",
        [
            # text_output@i03: @ is separator, devices after @ → i03 → i3
            # But _find_specific_device_in_path_parts scans parent.parts only (not filename),
            # and text_output@i03 is a parent part → extracts i3
            ("text_output@i03/file.txt", "i3"),
            ("text_output@i03/subdir/file.txt", "i3"),
            # text_output@i03/subdir@i05: scans reversed, last part subdir@i05 → @ separator → i05 → i5
            ("text_output@i03/subdir@i05/file.txt", "i5"),
            # text_output/subdir: no device pattern in any part
            ("text_output/subdir/file.txt", None),
            # text_output@i03/subdir@i03,i05: last part has multiple devices → returns None
            # (requires exactly 1 specific device)
            ("text_output@i03/subdir@i03,i05/file.txt", None),
            # 230508 is a dated prefix, i03 is extracted as device from last part
            ("text_output/230508/i03/file.txt", "i3"),
        ],
        ids=lambda v: repr(v) if isinstance(v, str) else str(v),
    )
    def test_find_specific_device_in_path_parts(self, rel_path_str, expected):
        """Specific device ID found from path parts matches expected value."""
        rel_path = PurePosixPath(rel_path_str)
        result = _find_specific_device_in_path_parts(rel_path)
        assert result == expected, (
            f"_find_specific_device_in_path_parts({rel_path_str!r}) "
            f"returned {result!r}, expected {expected!r}"
        )


class TestExtractDeviceIdFromRawFileName:
    """Test extract_device_id_from_raw_file_name from file_finder.

    The function strips optional dated prefix and [@#_-] separators,
    then matches the remainder against ptn_device_id_named_parts.
    The pattern matches type+model+number greedily, so 'W1_130510'
    matches type='w', number='130510' (not just 'w1').
    """

    @pytest.mark.parametrize(
        "file_name, expected",
        [
            # '#' is separator, 'W1_130510.txt' remains — pattern matches W1, _130510 is comment
            ("#W1_130510.txt", "w1"),
            # '130510' is dated prefix, '#' is separator, 'W1_130510.txt' remains
            ("130510#W1_130510.txt", "w1"),
            # no separator prefix, ptn_device_id_named_parts matches i1
            ("i1.txt", "i1"),
            # no separator prefix, W1 matches type='w', number='1'
            ("W1.txt", "w1"),
            # '#' separator, '1.txt' remains — no type/model but sep present, defaults to inclinometer
            ("#1.txt", "i1"),
            # no separator, '1.txt' — no type/model, no sep → returns None
            ("1.txt", None),
        ],
        ids=lambda v: repr(v) if isinstance(v, str) else str(v),
    )
    def test_extract_device_id_from_raw_file_name(self, file_name, expected):
        """Device ID extracted from raw file name matches expected value."""
        result = extract_device_id_from_raw_file_name(file_name)
        assert result == expected, (
            f"extract_device_id_from_raw_file_name({file_name!r}) "
            f"returned {result!r}, expected {expected!r}"
        )


class TestExcludedDirPatterns:
    """Test that excluded directory patterns work correctly."""

    def test_config_has_excluded_patterns(self):
        """Config should have ptn_dir_exclude attribute with default pattern."""
        assert hasattr(config, "ptn_dir_exclude")
        assert isinstance(config.ptn_dir_exclude, list)
        assert len(config.ptn_dir_exclude) > 0

    @pytest.mark.parametrize(
        "dir_name, should_exclude",
        [
            ("test-", True),
            ("230616-", True),
            ("some-name-", True),
            ("a-", True),
            ("test", False),
            ("230616_data", False),
            ("some-name", False),
            ("test-1", False),
            ("name-2b", False),
        ],
    )
    def test_directory_exclusion_logic(self, dir_name, should_exclude):
        """Directory exclusion pattern correctly identifies excluded directories."""
        is_excluded = any(
            re.search(pattern, dir_name) for pattern in config.ptn_dir_exclude
        )
        assert is_excluded == should_exclude, (
            f"Directory {dir_name!r} exclusion status: "
            f"expected {should_exclude}, got {is_excluded}"
        )


class TestPatternBoundary:
    """Test regex pattern boundary behavior for device extraction.

    Regression test for the bug where i54bin2s was incorrectly matched as i5
    instead of i54 due to the (?![a-z]) boundary in ptn_devices_groups_part.

    Note: ptn_devices_groups_part extracts the raw match (e.g., 'i04' not 'i4').
    Normalization happens later in parse_device_id_groups().
    """

    @pytest.mark.parametrize(
        "input_str, expected_match",
        [
            ("i54bin2s", "i54"),  # regression: was i5 before fix
            ("i04bin2s", "i04"),  # raw match, normalization strips leading zero later
            ("i3bin300s", "i3"),
            ("ib27bin300s", "ib27"),
            ("i54", "i54"),
            ("i3", "i3"),
        ],
        ids=lambda v: repr(v) if isinstance(v, str) else str(v),
    )
    def test_pattern_matches_full_device_number_before_bin_suffix(
        self, input_str, expected_match
    ):
        """Pattern should match the full device number before 'bin' suffix."""
        match = re.match(config.ptn_devices_groups_part, input_str, re.IGNORECASE)
        assert match is not None, f"Pattern did not match {input_str!r}"
        matched = match.group()
        assert matched == expected_match, (
            f"Pattern matched {matched!r} from {input_str!r}, expected {expected_match!r}"
        )
