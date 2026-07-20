import pytest
from meta_finder.parse_cruise_dir_name import add_dataset_name

class TestCruiseNameExtraction:
    """Test cruise name extraction from various directory name formats"""

    @pytest.mark.parametrize(
        "directory_name, device_dir_name, expected_name",
        [
            # Basic cruise directory names
            ("230616_Kulikovo", None, "Kulikovo23"),
            ("230825_Kulikovo@ADCP,ADV,i,tr", None, "Kulikovo23"),
            ("230616_ABP54", None, "ABP54"),
            ("230825_ABP54_test", None, "ABP54_test"),
            ("201202_BalticSpit", None, "BalticSpit20"),

            # Device directory names with specific dates
            ("230616_Kulikovo", "230508_inclinometer@i03", "Kulikovo23"),
            ("230616_Kulikovo", "230509_wavegauge@w01", "Kulikovo23"),
            ("201202_BalticSpit", "211008P7.5,15,E15m@i04,11,14,36,37,38,w2,5", "BalticSpit20"),

            # Names ending with digits
            ("230616_ABP54", None, "ABP54"),
            ("230825_ABP54_test", None, "ABP54_test"),

            # Directory patterns with device specifications
            ("230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6", None, "ABP53"),
            ("230507_ABP53@i", None, "ABP53"),

            # Directory names with minimal content
            ("230616_", None, "23"),
            ("230616", None, "230616"),
            ("invalid_name", None, "invalid_name"),
        ],
        ids=[
            "basic_cruise_name",
            "cruise_name_with_devices",
            "name_ending_with_digits",
            "name_with_suffix",
            "baltic_spit_cruise",
            "device_dir_with_inclinometer",
            "device_dir_with_wavegauge",
            "device_dir_with_specifications",
            "name_ending_with_digits_simple",
            "name_ending_with_digits_with_suffix",
            "device_dir_with_specifications_simple",
            "directory_name_with_minimal_content",
            "directory_name_without_underscore",
            "directory_name_without_date_format",
            "directory_name_with_invalid_format",
        ]
    )
    def test_extract_cruise_name_various_formats(
        self, directory_name: str, device_dir_name: str, expected_name: str
    ) -> None:
        """Test cruise name extraction from various directory name formats"""
        if "invalid" in directory_name:
            with pytest.raises(ValueError):
                result = add_dataset_name(device_dir_name, directory_name, None)
        else:
            result = add_dataset_name(device_dir_name, directory_name, None)
            assert result == expected_name, (
                f"Expected '{expected_name}' but got '{result}' for directory '{directory_name}'"
            )

    @pytest.mark.parametrize(
        "directory_name, device_dir_name, used_names, expected_name",
        [
            # Deduplication tests
            ("230616_Kulikovo", None, set(), "Kulikovo23"),
            ("230825_Kulikovo@ADCP,ADV,i,tr", None, {"Kulikovo23"}, "Kulikovo2308"),
            ("230616_ABP54", None, set(), "ABP54"),
            ("230825_ABP54_test", None, {"ABP54"}, "ABP54_test"),

            # Device directory deduplication
            ("230616_Kulikovo", "230708_inclinometer@i03", set(), "Kulikovo23"),
            ("230616_Kulikovo", "230709_wavegauge@w01", {"Kulikovo23"}, "Kulikovo2307"),

            # Directory patterns with deduplication
            ("230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6", None, set(), "ABP53"),
            ("230507_ABP53@i", None, {"ABP53"}, "ABP53_23"),
        ],
        ids=[
            "first_occurrence_no_dedup",
            "second_occurrence_with_dedup",
            "digits_name_no_dedup",
            "digits_name_with_dedup",
            "device_dir_first_occurrence",
            "device_dir_second_occurrence",
            "device_dir_with_specifications_no_dedup",
            "device_dir_with_specifications_with_dedup",
        ]
    )
    def test_extract_cruise_name_with_deduplication(
        self, directory_name: str, device_dir_name: str, used_names: set, expected_name: str
    ) -> None:
        """Test cruise name extraction with deduplication logic"""
        result = add_dataset_name(device_dir_name, directory_name, used_names)
        assert result == expected_name, (
            f"Expected '{expected_name}' but got '{result}' for directory '{directory_name}' "
            f"with used_names {used_names}"
        )

    def test_extract_cruise_name_directory_patterns(self) -> None:
        """Test cruise name extraction with various directory name patterns"""
        test_cases = [
            # Directory patterns with device types
            ("230507_ABP53_inclinometer", None, "ABP53"),
            ("230507_ABP53_wavegauge", None, "ABP53"),
            ("230616_Kulikovo_field", None, "Kulikovo_field23"),
            ("230825_Kulikovo@ADCP,ADV,i,tr", None, "Kulikovo23"),

            # Device directory patterns
            ("230507_ABP53_inclinometer", "230508_inclinometer@i03", "ABP53"),
            ("230507_ABP53_wavegauge", "230509_wavegauge@w01", "ABP53"),
            ("230616_Kulikovo_field", "230617_other_device", "Kulikovo_field23"),

            # Directory patterns with minimal content
            ("230507_", None, "23"),
            ("230507_test@", None, "test23"),
            ("230507_test@device", None, "test23"),
        ]

        for directory_name, device_dir_name, expected_name in test_cases:
            result = add_dataset_name(device_dir_name, directory_name, None)
            assert result == expected_name, (
                f"Expected '{expected_name}' but got '{result}' for directory '{directory_name}'"
            )