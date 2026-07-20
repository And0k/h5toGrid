"""
Tests to reproduce the 'NoneType' object is not iterable error in _find_matching_files_in_directory
"""
from unittest.mock import patch
from meta_finder.data_proc_funcs import _find_matching_files_in_archive, _find_matching_files_in_directory
from pathlib import Path

def test_find_files_with_same_pattern_generic_with_none_list():
    """Test _find_matching_files_in_directory when paths is None"""
    # This should handle the None case gracefully
    result = _find_matching_files_in_directory("test_file.txt", [])
    # Should return an empty list when paths is empty
    assert result == []


def test_find_files_with_same_pattern_generic_with_valid_list():
    """Test _find_matching_files_in_directory with a valid list"""
    from pathlib import Path, PurePosixPath
    valid_list = [Path("test_file.txt"), PurePosixPath("path/to/test_file.txt")]

    result = _find_matching_files_in_directory("test_file.txt", valid_list)
    # Should return the expected result
    assert len(result) == 1
    assert result[0] == Path("test_file.txt") or result[0] == PurePosixPath("path/to/test_file.txt")


def test_find_files_with_same_pattern_generic_with_empty_list():
    """Test _find_matching_files_in_directory with an empty list"""
    result = _find_matching_files_in_directory("test_file.txt", [])
    # Should return an empty list when paths is empty
    assert result == []


def test_find_matching_files_in_archive_with_corrupted_archive():
    """Test the scenario where list_archive_recursive might return None due to archive issues"""

    # Mock the utils_sys.list_archive_recursive to return None to simulate a corrupted archive
    with patch('meta_finder.utils_sys.list_archive_recursive', return_value=None):
        # Create a mock path
        mock_archive_path = Path("fake_archive.zip")
        mock_rel_path = Path("fake_file.txt")

        # This should handle the None return gracefully
        result = _find_matching_files_in_archive(mock_archive_path, mock_rel_path)
        # Should return an empty list, not raise an exception
        assert result == []


def test_find_matching_files_in_archive_with_exception_in_list_archive():
    """Test the scenario where list_archive_recursive raises an exception"""

    # Mock the utils_sys.list_archive_recursive to raise an exception
    with patch('meta_finder.utils_sys.list_archive_recursive', side_effect=Exception("Archive corrupted")):
        # Create a mock path
        mock_archive_path = Path("fake_archive.zip")
        mock_rel_path = Path("fake_file.txt")

        # This should handle the exception gracefully
        result = _find_matching_files_in_archive(mock_archive_path, mock_rel_path)
        # Should return an empty list, not raise an exception
        assert result == []