"""Tests for strict vs relaxed device group pattern split.

Verifies that the strict variant (used for directory discovery) rejects bare
numbers after @/# without device type prefix (e.g., CTD_SST_48Mc#1253), while
the relaxed variant (used for filename parsing) still matches them correctly
(e.g., 191210#07,23,30,32-bin300s.zip).

Also verifies that broader date prefixes and separators are supported for
directory discovery (e.g., 130510..0708@ADCP,CTD,i,w,
201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6).
"""

import re

import pytest

from meta_finder.config import Config
from meta_finder import file_finder

@pytest.fixture()
def cfg():
    """Create a fresh Config instance with resolved patterns."""
    return Config()


class TestStrictPatternRejectsBareNumbers:
    """Strict pattern must NOT match bare numbers after @/# without device type."""

    @pytest.mark.parametrize(
        "dirname,comment",
        [
            (
                "CTD_SST_48Mc#1253",
                "CTD directory with #number but no device type prefix",
            ),
            (
                "some_data@12345",
                "directory with @number but no device type prefix",
            ),
            (
                "raw_data#999",
                "directory with #number but no device type prefix",
            ),
        ],
    )
    def test_strict_rejects_bare_numbers_after_separator(
        self, cfg, dirname, comment,
    ):
        """Strict ptn_devices_groups_part_strict must not match bare numbers after @/#."""
        match = re.search(cfg.ptn_devices_groups_part_strict, dirname)
        assert match is None, (
            f"Strict pattern should not match '{dirname}': {comment}"
        )

    @pytest.mark.parametrize(
        "dirname,comment",
        [
            (
                "CTD_SST_48Mc#1253",
                "CTD directory with #number but no device type prefix",
            ),
            (
                "some_data@12345",
                "directory with @number but no device type prefix",
            ),
        ],
    )
    def test_strict_rejects_in_dir_search_context(
        self, cfg, dirname, comment,
    ):
        """ptn_device_dir_search must not match directories with bare numbers after @/#."""
        match = re.search(cfg.ptn_device_dir_search, dirname)
        assert match is None, (
            f"Directory search pattern should not match '{dirname}': {comment}"
        )


class TestStrictPatternAcceptsTypedDevices:
    """Strict pattern MUST still match device IDs with proper type prefix."""

    @pytest.mark.parametrize(
        "dirname,comment",
        [
            (
                "230508_inclinometer@i03",
                "standard device directory with @i prefix",
            ),
            (
                "230509_wavegauge@w01",
                "wave gauge directory with @w prefix",
            ),
            (
                "230507_inclinometer@i3,4,15,19,37,38;ib27-30,ip6",
                "combined device directory with typed IDs",
            ),
            (
                "230506_inclinometer_i03",
                "device directory with _i prefix",
            ),
            (
                "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6",
                "device dir with date 201202, station info P1-5,I1-2 and typed devices after @",
            ),
        ],
    )
    def test_strict_accepts_typed_device_ids(
        self, cfg, dirname, comment,
    ):
        """Strict pattern must still match device IDs with type prefix."""
        match = re.search(cfg.ptn_devices_groups_part_strict, dirname)
        assert match is not None, (
            f"Strict pattern should match '{dirname}': {comment}"
        )

    @pytest.mark.parametrize(
        "dirname,comment",
        [
            (
                "230508_inclinometer@i03",
                "standard device directory with @i prefix",
            ),
            (
                "230509_wavegauge@w01",
                "wave gauge directory with @w prefix",
            ),
            (
                "230507_inclinometer@i3,4,15,19,37,38;ib27-30,ip6",
                "combined device directory with typed IDs",
            ),
            (
                "230506_inclinometer_i03",
                "device directory with _i prefix",
            ),
            (
                "130510..0708@ADCP,CTD,i,w",
                "device dir with date range 130510..0708 and mixed device types after @",
            ),
            (
                "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6",
                "device dir with date 201202, station info P1-5,I1-2 and typed devices after @",
            ),
        ],
    )
    def test_dir_search_accepts_device_dirs(
        self, cfg, dirname, comment,
    ):
        """ptn_device_dir_search must match directories with device identifiers."""
        match = re.search(cfg.ptn_device_dir_search, dirname)
        assert match is not None, (
            f"Directory search pattern should match '{dirname}': {comment}"
        )
        assert match["device"], (
            f"Directory search pattern must capture device group in '{dirname}': {comment}"
        )


class TestRelaxedPatternAcceptsBareNumbers:
    """Relaxed pattern (ptn_devices_groups_part) must match bare numbers after @/#."""

    @pytest.mark.parametrize(
        "filename,comment",
        [
            (
                "191210#07,23,30,32-bin300s.zip",
                "filename with #number list after date prefix",
            ),
            (
                "data@3,5,7-output.tsv",
                "filename with @number list",
            ),
        ],
    )
    def test_relaxed_accepts_bare_numbers_after_separator(
        self, cfg, filename, comment,
    ):
        """Relaxed ptn_devices_groups_part must match bare numbers after @/#."""
        match = re.search(cfg.ptn_devices_groups_part, filename)
        assert match is not None, (
            f"Relaxed pattern should match '{filename}': {comment}"
        )


class TestStrictDerivedFromRelaxed:
    """Strict variant must be derived from relaxed by removing the lookbehind alternative."""

    def test_strict_lacks_lookbehind_alternative(self, cfg):
        """Strict pattern must not contain the (?<=[@#])| lookbehind alternative."""
        assert "(?<=[@#])|" not in cfg.ptn_devices_groups_part_strict, (
            "Strict pattern must not contain the relaxed lookbehind alternative "
            "'(?<=[@#])|' that allows bare numbers after @/#"
        )

    def test_relaxed_has_lookbehind_alternative(self, cfg):
        """Relaxed pattern must contain the (?<=[@#])| lookbehind alternative."""
        assert "(?<=[@#])|" in cfg.ptn_devices_groups_part, (
            "Relaxed pattern must contain the lookbehind alternative "
            "'(?<=[@#])|' for filename parsing flexibility"
        )

    def test_strict_is_subset_of_relaxed(self, cfg):
        """Every match of strict must also be a match of relaxed (strict is more restrictive)."""
        # Test with various valid device directory names
        test_names = [
            "230508_inclinometer@i03",
            "230509_wavegauge@w01",
            "230507_inclinometer@i3,4,15",
            "230506_inclinometer_i03",
            "data_i(3,5,7)",
            "output@ib27-30",
            "130510..0708@ADCP,CTD,i,w",
            "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6",
        ]
        for name in test_names:
            strict_match = re.search(cfg.ptn_devices_groups_part_strict, name)
            relaxed_match = re.search(cfg.ptn_devices_groups_part, name)
            if strict_match:
                assert relaxed_match is not None, (
                    f"Relaxed pattern must match '{name}' when strict does "
                    f"(strict matched '{strict_match.group()}')"
                )


class TestBroaderDatePrefixParsing:
    """Broader date prefixes and separators for cruise/device directory discovery."""

    @pytest.mark.parametrize(
        "dirname,expected_date,comment",
        [
            (
                "130510..0708@ADCP,CTD,i,w",
                {"YY": "13", "MM": "05", "DD": "10", "rest": "..0708@ADCP,CTD,i,w"},
                "date range prefix 130510 parsed as YYMMDD with range in rest",
            ),
            (
                "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6",
                {"YY": "20", "MM": "12", "DD": "02",
                 "rest": "P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6"},
                "date prefix 201202 parsed as YYMMDD with station info in rest",
            ),
            (
                "201202_BalticSpit",
                {"YY": "20", "MM": "12", "DD": "02", "rest": "_BalticSpit"},
                "cruise dir with YYYYMM date prefix and underscore separator",
            ),
        ],
    )
    def test_parse_dated_dir_handles_broader_formats(
        self, dirname, expected_date, comment,
    ):
        """parse_dated_dir must extract date components from broader date prefix formats."""
        from meta_finder.parse_cruise_dir_name import parse_dated_dir

        result = parse_dated_dir(dirname)
        assert result == expected_date, (
            f"parse_dated_dir('{dirname}') should return {expected_date}: {comment}, "
            f"got {result}"
        )

    @pytest.mark.parametrize(
        "dirname,comment",
        [
            (
                "130510..0708@ADCP,CTD,i,w",
                "date range prefix with 6 leading digits",
            ),
            (
                "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6",
                "date prefix with station info immediately after digits",
            ),
            (
                "201202_BalticSpit",
                "cruise dir with 6-digit date prefix and underscore",
            ),
        ],
    )
    def test_glob_dated_dir_matches_broader_formats(
        self, cfg, dirname, comment,
    ):
        """glob_dated_dir must match directory names with broader date prefixes."""
        import fnmatch

        assert fnmatch.fnmatch(dirname, cfg.glob_dated_dir), (
            f"glob_dated_dir '{cfg.glob_dated_dir}' should match '{dirname}': {comment}"
        )


class TestThreeLevelDirectoryDiscovery:
    """Verify find_device_dirs discovers device dirs in multi-level cruise structures.

    Tests the full directory discovery pipeline with temporary structures:
    cruise_dir / intermediate_dir / device_dir, where the device dir has a broader
    date prefix or station info before the @device separator.
    """

    @pytest.fixture()
    def _setup_dir(self, tmp_path, request):
        """Create directory structure from parametrized (dirs_to_create, has_text_output)."""
        dirs_to_create, has_text_output = request.param
        for d in dirs_to_create:
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
        if has_text_output:
            (tmp_path / dirs_to_create[-1] / "text_output").mkdir(exist_ok=True)
        return tmp_path

    @pytest.mark.parametrize(
        "_setup_dir,expected_device_dir,comment",
        [
            (
                (
                    ["201202_BalticSpit/inclinometers/"
                     "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6"],
                    True,
                ),
                "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6",
                "3-level: cruise/intermediate/device_dir with text_output",
            ),
            (
                (
                    ["130510..0708@ADCP,CTD,i,w"],
                    True,
                ),
                "130510..0708@ADCP,CTD,i,w",
                "2-level: cruise dir with date range and device list, text_output",
            ),
            (
                (
                    ["130510..0708@ADCP,CTD,i,w/inclinometer,wavegage,any_other/"
                     "130510_Sambian"],
                    True,
                ),
                "130510_Sambian",
                "3-level: cruise/comma-keywords/dated_device_dir with text_output",
            ),
        ],
        indirect=["_setup_dir"],
        ids=["3level_baltic_spit", "2level_date_range", "3level_comma_keywords"],
    )
    def test_find_device_dirs_discovers_broader_formats(
        self, _setup_dir, expected_device_dir, comment,
    ):
        """find_device_dirs must find device directories with broader date prefixes."""

        cruise_dir = list(_setup_dir.iterdir())[0]
        device_dirs = file_finder.find_device_dirs(cruise_dir)
        device_names = [d.name for d in device_dirs]
        assert expected_device_dir in device_names, (
            f"Device dir '{expected_device_dir}' should be found ({comment}), got: {device_names}"
        )

    @pytest.mark.parametrize(
        "_setup_dir,expected_cruise_dir,comment",
        [
            (
                (
                    ["201202_BalticSpit/inclinometer/"
                     "201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6"],
                    True,
                ),
                "201202_BalticSpit",
                "cruise dir with 6-digit date prefix in 3-level structure",
            ),
            (
                (
                    ["130510..0708@ADCP,CTD,i,w"],
                    True,
                ),
                "130510..0708@ADCP,CTD,i,w",
                "cruise dir with date range prefix 130510..0708",
            ),
        ],
        indirect=["_setup_dir"],
        ids=["3level_baltic_spit", "2level_date_range"],
    )
    def test_find_cruise_directories_discovers_broader_formats(
        self, _setup_dir, expected_cruise_dir, comment,
    ):
        """find_cruise_directories must find cruise directories with broader date prefixes."""

        cruise_dirs = file_finder.find_cruise_directories([_setup_dir])
        cruise_names = [d.name for d in cruise_dirs]
        assert expected_cruise_dir in cruise_names, (
            f"Cruise dir '{expected_cruise_dir}' should be found ({comment}), got: {cruise_names}"
        )
