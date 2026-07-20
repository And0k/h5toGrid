"""Test that directories ending with '-' are excluded from processing."""

import re
from meta_finder import config


def test_excluded_dir_patterns():
    """Test that ptn_dir_exclude configuration is properly set."""
    # Check that the configuration exists and has the default pattern
    assert hasattr(config, 'ptn_dir_exclude'), "Config should have ptn_dir_exclude attribute"
    assert isinstance(config.ptn_dir_exclude, list), "ptn_dir_exclude should be a list"
    assert len(config.ptn_dir_exclude) > 0, "ptn_dir_exclude should not be empty"

    # Check the default pattern matches directories ending with '-' or '-.'
    default_pattern = config.ptn_dir_exclude[0]
    assert re.search(default_pattern, "test-"), f"Pattern should match 'test-'"
    assert re.search(default_pattern, "some-name-"), f"Pattern should match 'some-name-'"


def test_directory_exclusion_logic():
    """Test that the exclusion logic works correctly."""
    # Test directories that should be excluded
    excluded_dirs = ["test-", "230616-", "some-name-", "a-"]
    for dir_name in excluded_dirs:
        is_excluded = any(re.search(pattern, dir_name) for pattern in config.ptn_dir_exclude)
        assert is_excluded, f"Directory '{dir_name}' should be excluded"

    # Test directories that should NOT be excluded
    included_dirs = ["test", "230616_data", "some-name", "a", "test-1", "name-2b"]
    for dir_name in included_dirs:
        is_excluded = any(re.search(pattern, dir_name) for pattern in config.ptn_dir_exclude)
        assert not is_excluded, f"Directory '{dir_name}' should NOT be excluded"
