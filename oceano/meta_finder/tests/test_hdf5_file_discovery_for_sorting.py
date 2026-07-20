"""
Test to verify that HDF5 files are properly discovered and included in data paths for sorting,
even when time metadata is already available from text files.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from meta_finder.collect import get_absent_meta


def test_hdf5_files_discovered_for_complete_data_path_collection():
    """Test that HDF5 files are discovered even when time metadata is already available."""
    # Mock device directory
    mock_device_dir = Path("/mock/device/dir")

    # Create mock input with a device that has time metadata
    meta_in = {
        "i1": {
            "metadata": {
                "time_st": "2023-01-01 00:00:00",
                "time_en": "2023-01-02 00:00:00"
            },  # Has time info in JSON
            "data_paths": {
                (Path("/text"), Path("file.txt")): {"devices": ["i1"], "averaging_interval": 2}
            }
        }
    }

    # Mock HDF5 results for the additional discovery call (no time info needed)
    mock_hdf5_discovery_result = {
        "i1": {
            "time_info": None,  # No time info needed in this call
            "data_paths": {
                (Path("/hdf5_file.h5"), Path("")): {"devices": ["i1"], "is_hdf5": True, "h5_file": "test.h5"}
            }
        }
    }

    # Mock HDF5 results for the regular time extraction call (should not be needed in this case)
    mock_hdf5_time_result = {}

    with patch('meta_finder.collect.config') as mock_config:
        mock_config.use_hdf5_fallback = True
        mock_config.raw_hdf5_cols = set()

        # Mock the HDF5 extraction to return our test data for both calls
        with patch('meta_finder.collect.extract_metadata_from_hdf5') as mock_hdf5:
            # Configure mock to return different results for different calls
            mock_hdf5.side_effect = [mock_hdf5_time_result, mock_hdf5_discovery_result]

            # Mock text output processing to return empty (so only our initial data exists)
            with patch('meta_finder.collect.process_text_output_directory') as mock_text_proc:
                mock_text_proc.return_value = None

                # Call get_absent_meta
                result = get_absent_meta(meta_in, mock_device_dir)

                # Verify that the device exists in results
                assert "i1" in result

                # Check that data_paths contains both the original text file AND the HDF5 file
                data_paths = result["i1"]["data_paths"]
                path_keys = list(data_paths.keys())

                # Should contain both the original text file and the HDF5 file
                assert len(path_keys) >= 2, f"Should have at least 2 data paths (text + HDF5), but got {len(path_keys)}"

                # Check that both text file and HDF5 file are present
                has_text_file = any("file.txt" in str(path[1]) for path in path_keys)
                has_hdf5_file = any(".h5" in str(path[0]) for path in path_keys)

                assert has_text_file, "Should contain the original text file"
                assert has_hdf5_file, "Should contain the HDF5 file for proper sorting"


def test_hdf5_discovery_works_when_no_time_metadata_available():
    """Test that HDF5 files are discovered and their time info is extracted when needed."""
    # Mock device directory
    mock_device_dir = Path("/mock/device/dir")

    # Create mock input with a device that has NO time metadata
    meta_in = {
        "i1": {
            "metadata": {"time_st": "?", "time_en": "?"},  # No time info in JSON
            "data_paths": {
                (Path("/text"), Path("file.txt")): {"devices": ["i1"], "averaging_interval": 2}
            }
        }
    }

    # Mock HDF5 results with time info that should be used (regular time extraction call)
    mock_hdf5_time_result = {
        "i1": {
            "time_info": ("2023-01-01 00:00:00", "2023-01-02 00:00:00", "-", "-"),
            "data_paths": {
                (Path("/hdf5_file.h5"), Path("")): {"devices": ["i1"], "is_hdf5": True, "h5_file": "test.h5"}
            }
        }
    }

    # Mock HDF5 results for the additional discovery call (path discovery only)
    mock_hdf5_discovery_result = {
        "i1": {
            "time_info": None,
            "data_paths": {
                (Path("/hdf5_file.h5"), Path("")): {"devices": ["i1"], "is_hdf5": True, "h5_file": "test.h5"}
            }
        }
    }

    with patch('meta_finder.collect.config') as mock_config:
        mock_config.use_hdf5_fallback = True
        mock_config.raw_hdf5_cols = set()

        # Mock the HDF5 extraction to return our test data for both calls
        with patch('meta_finder.collect.extract_metadata_from_hdf5') as mock_hdf5:
            # Configure mock to return different results for different calls
            mock_hdf5.side_effect = [mock_hdf5_time_result, mock_hdf5_discovery_result]

            # Mock text output processing to return empty (so only our initial data exists)
            with patch('meta_finder.collect.process_text_output_directory') as mock_text_proc:
                mock_text_proc.return_value = None

                # Call get_absent_meta
                result = get_absent_meta(meta_in, mock_device_dir)

                # Verify that the device exists in results
                assert "i1" in result

                # Check that data_paths contains both the original text file AND the HDF5 file
                data_paths = result["i1"]["data_paths"]
                path_keys = list(data_paths.keys())

                # Should contain both the original text file and the HDF5 file
                assert len(path_keys) >= 2, f"Should have at least 2 data paths (text + HDF5), but got {len(path_keys)}"

                # Check that both text file and HDF5 file are present
                has_text_file = any("file.txt" in str(path[1]) for path in path_keys)
                has_hdf5_file = any(".h5" in str(path[0]) for path in path_keys)

                assert has_text_file, "Should contain the original text file"
                assert has_hdf5_file, "Should contain the HDF5 file for proper sorting"

                # Check that time info was properly set from HDF5
                assert result["i1"]["time_info"] == ("2023-01-01 00:00:00", "2023-01-02 00:00:00", "-", "-")


if __name__ == "__main__":
    test_hdf5_files_discovered_for_complete_data_path_collection()
    test_hdf5_discovery_works_when_no_time_metadata_available()
    print("All tests passed - HDF5 files are properly discovered for complete data path collection!")