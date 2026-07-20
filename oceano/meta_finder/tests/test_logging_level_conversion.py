import pytest
from meta_finder.config import Config


def test_logging_level_conversion():
    """Test that logging level can be specified as string or numeric value."""

    # Test string conversion
    assert Config._convert_logging_level("DEBUG") == 10
    assert Config._convert_logging_level("INFO") == 20
    assert Config._convert_logging_level("WARNING") == 30
    assert Config._convert_logging_level("WARN") == 30
    assert Config._convert_logging_level("ERROR") == 40
    assert Config._convert_logging_level("CRITICAL") == 50
    assert Config._convert_logging_level("FATAL") == 50

    # Test numeric values remain unchanged
    assert Config._convert_logging_level(10) == 10
    assert Config._convert_logging_level(20) == 20

    # Test invalid string defaults to INFO
    assert Config._convert_logging_level("INVALID") == 20  # INFO level

    # Test case insensitivity
    assert Config._convert_logging_level("debug") == 10
    assert Config._convert_logging_level("Info") == 20


def test_config_with_string_logging_level():
    """Test that config can be created with string logging level."""
    # This should work without error
    config = Config(logging_level="DEBUG")
    assert config.logging_level == 10  # DEBUG level

    config = Config(logging_level="INFO")
    assert config.logging_level == 20  # INFO level

    config = Config(logging_level=30)  # Numeric value
    assert config.logging_level == 30  # WARNING level


if __name__ == "__main__":
    test_logging_level_conversion()
    test_config_with_string_logging_level()
    print("All logging level conversion tests passed!")