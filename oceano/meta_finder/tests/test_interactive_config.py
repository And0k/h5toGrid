"""Test interactive configuration mode functionality."""

import pytest
from pathlib import Path
from meta_finder.config import Config


class TestInteractiveConfig:
    """Test suite for interactive configuration mode."""

    @pytest.mark.parametrize(
        "user_input,expected_config,description",
        [
            # Test with wildcard to use all defaults
            (
                "*\n",
                {},
                "wildcard uses all defaults"
            ),
            # Test with single value then wildcard (skip first two fields)
            (
                "\n\ntrue\n*\n",
                {"create_info_files": True},
                "single value then wildcard"
            ),
            # Test with multiple values then wildcard (skip first two fields)
            (
                "\n\ntrue\nfalse\n*\n",
                {"create_info_files": True, "from_data": False},
                "multiple values then wildcard"
            ),
            # Test with empty inputs (use defaults)
            (
                "\n\n\n*\n",
                {},
                "empty inputs use defaults"
            ),
        ],
        ids=[
            "wildcard-all-defaults",
            "single-value-then-wildcard",
            "multiple-values-then-wildcard",
            "empty-inputs-use-defaults",
        ]
    )
    def test_interactive_prompt_with_wildcard(
        self,
        user_input: str,
        expected_config: dict,
        description: str,
        monkeypatch,
    ):
        """Test interactive prompting with wildcard to use defaults.

        Args:
            user_input: Simulated user input string
            expected_config: Expected configuration values
            description: Test case description
            monkeypatch: Pytest fixture for mocking
        """
        # Mock input to provide simulated user responses using iterator
        inputs = iter(user_input.splitlines())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        # Call the interactive prompt method
        result = Config._prompt_interactive()

        # Verify result contains expected values
        for key, value in expected_config.items():
            assert result[key] == value, (
                f"{description}: Expected {key}={value}, got {result.get(key)}"
            )

    @pytest.mark.parametrize(
        "user_input,field_name,field_type,expected_value,description",
        [
            # Test boolean parsing
            ("true", "create_info_files", bool, True, "boolean true"),
            ("false", "create_info_files", bool, False, "boolean false"),
            ("yes", "create_info_files", bool, True, "boolean yes"),
            ("no", "create_info_files", bool, False, "boolean no"),
            ("1", "create_info_files", bool, True, "boolean 1"),
            ("0", "create_info_files", bool, False, "boolean 0"),

            # Test integer parsing
            ("100", "max_burst_time_detection", int, 100, "integer value"),
            ("10800", "max_burst_time_detection", int, 10800, "integer default"),

            # Test float parsing
            ("3.5", "default_text_file_averaging", float, 3.5, "float value"),
            ("2.0001", "default_text_file_averaging", float, 2.0001, "float default"),

            # Test string parsing
            ("test_pattern", "ptn_device_dir_keywords", str, "test_pattern", "string value"),

            # Test Path parsing
            ("/tmp/test", "output_dir", Path, Path("/tmp/test"), "path value"),
        ],
        ids=[
            "bool-true",
            "bool-false",
            "bool-yes",
            "bool-no",
            "bool-1",
            "bool-0",
            "int-value",
            "int-default",
            "float-value",
            "float-default",
            "string-value",
            "path-value",
        ]
    )
    def test_parse_field_value(
        self,
        user_input: str,
        field_name: str,
        field_type: type,
        expected_value,
        description: str,
    ):
        """Test field value parsing for different types.

        Args:
            user_input: String input to parse
            field_name: Name of the field (for field object lookup)
            field_type: Expected type of the field
            expected_value: Expected parsed value
            description: Test case description
        """
        # Find the field object
        field_obj = None
        for fld in Config.__dataclass_fields__.values():
            if fld.name == field_name:
                field_obj = fld
                break

        assert field_obj is not None, f"Field {field_name} not found in Config"

        # Strip whitespace from input (simulating what _prompt_interactive does)
        user_input = user_input.strip()

        # Parse the value
        result = Config._parse_field_value(user_input, field_type, field_obj)

        # Verify the result
        assert result == expected_value, (
            f"{description}: Expected {expected_value}, got {result}"
        )

    @pytest.mark.parametrize(
        "user_input,field_name,field_type,description",
        [
            # Test invalid boolean
            ("invalid\n", "create_info_files", bool, "invalid boolean"),
            # Test invalid integer
            ("not_a_number\n", "max_burst_time_detection", int, "invalid integer"),
            # Test invalid float
            ("not_a_float\n", "default_text_file_averaging", float, "invalid float"),
        ],
        ids=[
            "invalid-boolean",
            "invalid-integer",
            "invalid-float",
        ]
    )
    def test_parse_field_value_invalid(
        self,
        user_input: str,
        field_name: str,
        field_type: type,
        description: str,
    ):
        """Test that invalid field values raise ValueError.

        Args:
            user_input: String input to parse
            field_name: Name of the field (for field object lookup)
            field_type: Expected type of the field
            description: Test case description
        """
        # Find the field object
        field_obj = None
        for fld in Config.__dataclass_fields__.values():
            if fld.name == field_name:
                field_obj = fld
                break

        assert field_obj is not None, f"Field {field_name} not found in Config"

        # Verify that parsing raises ValueError
        with pytest.raises(ValueError):
            Config._parse_field_value(user_input, field_type, field_obj)

    @pytest.mark.parametrize(
        "user_input,expected_items,description",
        [
            # Test space-separated values
            ("tsv csv", ["tsv", "csv"], "space-separated list"),
            # Test comma-separated values
            ("tsv,csv", ["tsv", "csv"], "comma-separated list"),
            # Test mixed separators
            ("tsv, csv json", ["tsv", "csv", "json"], "mixed separators"),
        ],
        ids=[
            "space-separated",
            "comma-separated",
            "mixed-separators",
        ]
    )
    def test_parse_field_value_sequence(
        self,
        user_input: str,
        expected_items: list,
        description: str,
    ):
        """Test parsing of sequence types (list, tuple).

        Args:
            user_input: String input to parse
            expected_items: Expected list of items
            description: Test case description
        """
        # Find the output_format field
        field_obj = None
        for fld in Config.__dataclass_fields__.values():
            if fld.name == "output_format":
                field_obj = fld
                break

        assert field_obj is not None, "Field output_format not found in Config"

        # Parse the value
        result = Config._parse_field_value(user_input, str, field_obj)

        # Verify the result is a list with expected items
        assert isinstance(result, list), f"{description}: Expected list, got {type(result)}"
        assert result == expected_items, (
            f"{description}: Expected {expected_items}, got {result}"
        )

    def test_interactive_mode_integration(self, monkeypatch):
        """Test that interactive mode integrates correctly with from_args.

        Args:
            monkeypatch: Pytest fixture for mocking
        """
        # Mock sys.argv to include --interactive flag
        test_args = ["script_name", "--interactive"]
        monkeypatch.setattr("sys.argv", test_args)

        # Mock input to provide responses (use wildcard for all defaults)
        monkeypatch.setattr("builtins.input", lambda prompt: "*")

        # Create config from args
        config = Config.from_args()

        # Verify config was created with defaults
        assert config is not None, "Config should be created"
        assert config.create_info_files == False, "Should use default value"

    def test_interactive_mode_with_values(self, monkeypatch):
        """Test interactive mode with actual user-provided values.

        Note: Input sequence skips first two Config fields (top_search_dirs, cruise_dir)
        with empty strings to reach create_info_files and from_data fields.

        Args:
            monkeypatch: Pytest fixture for mocking
        """
        # Mock sys.argv to include --interactive flag
        test_args = ["script_name", "--interactive"]
        monkeypatch.setattr("sys.argv", test_args)

        # Mock input to provide specific values then wildcard
        # Skip first two fields (top_search_dirs, cruise_dir) with empty strings
        inputs = iter(["", "", "true", "false", "*"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        # Create config from args
        config = Config.from_args()

        # Verify config was created with user-provided values
        assert config is not None, "Config should be created"
        assert config.create_info_files == True, "Should use user-provided value"
        assert config.from_data == False, "Should use user-provided value"


def test_interactive_config_help():
    """Test that interactive mode is documented in help."""
    parser = Config.create_argument_parser()
    help_text = parser.format_help()

    # Verify that interactive mode is documented
    assert "--interactive" in help_text or "-i" in help_text, (
        "Interactive mode should be documented in help"
    )
    assert "Interactive mode" in help_text, (
        "Interactive mode description should be in help"
    )
