#!/usr/bin/env python
"""
Test script for HDF5 coef date extraction functionality.
"""
import pytest
from pathlib import Path
import tempfile
import sys
import numpy as np
from datetime import datetime

dev_id = "i7"
dev_id_to_norm = "i07"

# Add the project source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_extract_coef_date_from_hdf5(mocker):
    """Test extracting coef date from HDF5 file."""
    from meta_finder.hdf5_processor import extract_coef_date_from_hdf5

    # Mock HDF5 file operations
    mock_tables = mocker.patch('meta_finder.hdf5_processor.tables')
    mock_file = mocker.MagicMock()
    mock_date_node = mocker.MagicMock()

    # Create mock date data (timestamp in nanoseconds) - for 2019-12-01 23:17:20
    mock_date_values = np.array([np.nan, 1575235040000000])  # Unix timestamp in microseconds

    mock_date_node.__getitem__.return_value = mock_date_values

    # Setup mock hierarchy
    mock_file.get_node.return_value = mock_date_node
    mock_file.__contains__.return_value = True # Path exists in file

    mock_tables.open_file.return_value.__enter__.return_value = mock_file

    # Test the function with device group name that matches expected HDF5 path format
    result = extract_coef_date_from_hdf5(Path("dummy.h5"), dev_id_to_norm)

    # Should return the converted timestamp, not the NaN value
    assert result == "2019-12-01 23:17:20", f"Expected '2019-12-01 23:17:20', got {result}"


@pytest.mark.parametrize("date_values,expected_result,comment", [
    (np.array([np.nan, 1575235040000000]), "2019-12-01 23:17:20", "should return the converted timestamp when NaN and valid timestamp present"),
    (np.array([15752350400000, 1579650490000000]), "2020-01-22 01:48:10", "should return the maximum timestamp when multiple valid timestamps present"),
    (np.array([np.nan, np.nan]), None, "should return None when all values are NaN"),
    (np.array(['2020-01-01', '2020-02-01']), "2020-02-01 00:00:00", "should return the maximum date when string dates are present"),
], ids=["with_nan", "max_timestamp", "all_nan", "string_dates"])
def test_extract_coef_date_different_scenarios(mocker, date_values, expected_result, comment):
    """Test extracting coef date with different data scenarios."""
    from meta_finder.hdf5_processor import extract_coef_date_from_hdf5

    # Mock HDF5 file operations
    mock_tables = mocker.patch('meta_finder.hdf5_processor.tables')
    mock_file = mocker.MagicMock()
    mock_date_node = mocker.MagicMock()

    mock_date_node.__getitem__.return_value = date_values

    # Setup mock hierarchy
    mock_file.get_node.return_value = mock_date_node
    mock_file.__contains__.return_value = True  # Path exists in file

    mock_tables.open_file.return_value.__enter__.return_value = mock_file

    # Test the function with device ID that matches expected HDF5 path format
    result = extract_coef_date_from_hdf5(Path("dummy.h5"), dev_id_to_norm)

    assert result == expected_result, f"extract_coef_date_from_hdf5 should return {expected_result} as per: {comment}"


def test_extract_all_coef_dates_from_hdf5_files(mocker):
    """Test extracting all coef dates from HDF5 files."""
    from meta_finder.hdf5_processor import extract_coef_dates_from_raw_hdf5_files
    from meta_finder import config

    # Temporarily set raw_hdf5_cols for testing
    original_raw_cols = getattr(config, 'raw_hdf5_cols', None)
    config.raw_hdf5_cols = {"coef_date", "raw_date_range"}

    dev_ids = [dev_id, 'i23']  # device IDs to pass to the testing function (normalized)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a _raw subdirectory
            raw_dir = temp_path / "_raw"
            raw_dir.mkdir()

            # Create a dummy HDF5 file in the raw directory
            dummy_h5 = raw_dir / "dummy.h5"
            dummy_h5.touch()

            # Create list of HDF5 files
            h5_files = [dummy_h5]

            # Mock the file operations
            mock_tables = mocker.patch('meta_finder.hdf5_processor.tables')
            mock_file = mocker.MagicMock()
            mock_date_node = mocker.MagicMock()

            # Mock group nodes for extract_devices_from_hdf5_groups to find
            # Create a mock group node for the device
            mock_group_node = mocker.MagicMock()
            mock_group_node._v_pathname = f"/{dev_id}"  # Path for the device group

            # Mock walk_nodes to return our device group
            mock_file.walk_nodes.return_value = [mock_group_node]

            # Create mock date data (timestamp in nanoseconds) - for 2019-12-01 23:17:20
            mock_date_values = np.array([np.nan, 157523504000000])  # Unix timestamp in microseconds

            mock_date_node.__getitem__.return_value = mock_date_values

            # Setup mock hierarchy
            mock_file.get_node.return_value = mock_date_node
            mock_file.__contains__.return_value = True  # Path exists in file

            mock_tables.open_file.return_value.__enter__.return_value = mock_file

            # Test the function
            result = extract_coef_dates_from_raw_hdf5_files(h5_files, dev_ids)

            # Should find dates for the devices
            assert dev_id in result, f"should find date for {dev_id} as per: valid device in HDF5 file"
            assert result[dev_id] == "2019-12-01 23:17:20", f"should return correct date for {dev_id} as per: extracted from HDF5 file"
    finally:
        # Restore original setting
        if original_raw_cols is not None:
            config.raw_hdf5_cols = original_raw_cols
        elif hasattr(config, 'raw_hdf5_cols'):
            delattr(config, 'raw_hdf5_cols')


def test_extract_time_range_from_hdf5_index(mocker):
    """Test extracting time range from HDF5 index."""
    from meta_finder.hdf5_processor import extract_time_range_from_hdf5_index

    # Mock the _convert_timestamps function to return predictable values
    mock_convert_timestamps = mocker.patch('meta_finder.hdf5_processor._convert_timestamps',
                                          side_effect=lambda time_values, attrs, start_idx, end_idx: ("2020-01-01 10:00:00", "2020-01-01 11:00:00"))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a dummy HDF5 file that matches the expected pattern
        dummy_h5 = temp_path / "dummy.proc_noAvg.h5"
        dummy_h5.touch()

        # Create mock h5_files_by_type dictionary
        h5_files_by_type = {
            "proc_noAvg": [dummy_h5],
            "proc": [],
            "raw": []
        }

        # Mock the file operations
        mock_tables = mocker.patch('meta_finder.hdf5_processor.tables')
        mock_file = mocker.MagicMock()
        mock_table = mocker.MagicMock()
        mock_cols = mocker.MagicMock()
        mock_index_col = mocker.MagicMock()

        # Create mock time data
        mock_time_values = np.array([
            np.datetime64('2020-01-01T10:00:00'),
            np.datetime64('2020-01-01T11:00:00')
        ])

        mock_index_col.__getitem__.return_value = mock_time_values
        mock_cols._v_colnames = ['index', 'data']
        mock_cols.index = mock_index_col
        mock_cols.data = mocker.MagicMock()

        mock_table.cols = mock_cols
        mock_table._v_attrs = mocker.MagicMock()
        mock_table.nrows = 2  # Mock the nrows attribute that the function accesses

        # Mock the walk_nodes to return our test group and table
        def mock_walk_nodes(where, classname):
            if where == "/" and classname == "Group":
                # Return a mock node for the device group
                mock_node = mocker.MagicMock()
                mock_node._v_pathname = f"/{dev_id_to_norm}"
                return [mock_node]
            elif where == f"/{dev_id_to_norm}" and classname == "Table":
                # Return our mock table
                return [mock_table]
            else:
                return []

        mock_file.walk_nodes.side_effect = mock_walk_nodes

        mock_tables.open_file.return_value.__enter__.return_value = mock_file

        # Test the function with normalized dev_id
        result = extract_time_range_from_hdf5_index(h5_files_by_type, [dev_id])

        # Should find time range for the device where HDF5 group was 'i07' but normalized to 'i7'
        assert dev_id in result, f"should find time range for {dev_id} as per: valid device in HDF5 file"
        assert result[dev_id] == ("2020-01-01 10:00:00", "2020-01-01 11:00:00"), (
            f"should return correct time range for {dev_id} as per: extracted from HDF5 index"
        )
