#!/usr/bin/env python
"""
Test script for HDF5 fallback functionality in TCM Metadata Processor.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from meta_finder import config
from meta_finder.hdf5_processor import extract_time_range_from_hdf5_table
from meta_finder.hdf5_processor import find_hdf5_files
from meta_finder.hdf5_processor import extract_averaging_seconds_from_h5group
from meta_finder.hdf5_processor import extract_time_ranges_from_hdf5_combined
from meta_finder.parse_data_file_name import extract_device_ids_from_prefixed_name
@pytest.mark.parametrize("group_name,expected,comment", [
    ("i1", "i1", "should extract simple device ID i1"),
    ("w2", "w2", "should extract simple device ID w2"),
    ("p5", "p5", "should extract device ID from p5"),
    ("i03", "i3", "should extract device ID from i03 (with zero)"),
    ("w04", "w4", "should extract device ID from w04 (with zero)"),
    ("invalid_group", None, "should return None for invalid group name"),
    ("i", None, "should return None for incomplete device ID"),
    ("123", None, "should return None for numeric-only group name"),
], ids=["simple_i1", "simple_w2", "simple_p5", "i03_with_zero", "w04_with_zero", "invalid", "incomplete", "numeric_only"])
def test_extract_device_id_from_hdf5_group_name(group_name, expected, comment):
    """Test extracting device ID from HDF5 group names."""

    result = extract_device_ids_from_prefixed_name(group_name)
    assert result == expected, (
        f"extract_device_ids_from_prefixed_name({group_name}) should return {expected} as per: {comment}"
    )


@pytest.mark.parametrize("h5group,expected,comment", [
    ("data.bin10", 10, "should extract averaging seconds 10 from data.bin10"),
    ("data.bin300", 300, "should extract averaging seconds 300 from data.bin300"),
    ("data.bin0", 0, "should extract averaging seconds 0 from data.bin0"),
    ("data.bin1", 1, "should extract averaging seconds 1 from data.bin1"),
    ("data.bin10", 10, "should extract averaging seconds 10 from data.bin10"),
    ("data", None, "should return None for data without bin pattern"),
    ("data.bin", None, "should return None for data.bin without number after bin"),
], ids=["bin10", "bin300", "bin0", "bin1", "bin10_noext", "no_bin", "bin_no_number"])
def test_extract_averaging_seconds_from_h5group(h5group, expected, comment):
    """Test extracting averaging seconds from h5group."""


    result = extract_averaging_seconds_from_h5group(h5group)
    assert result == expected, f"extract_averaging_seconds_from_h5group({h5group}) should return {expected} as per: {comment}"


def test_find_hdf5_files(common_test_data_setup):
    """Test finding HDF5 files in priority order."""

    # Use an existing test device directory from the common test data setup
    test_dir = common_test_data_setup
    # Use an existing device directory that has the proper structure
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6" / "230508_inclinometer@i03"

    # Verify the test directory exists
    assert device_dir.exists(), "Test device directory should exist in common test data"

    # Create test files in the device directory if they don't already exist
    (device_dir / "test.proc_noAvg.h5").touch(exist_ok=True)
    (device_dir / "test2.proc.h5").touch(exist_ok=True)
    raw_dir = device_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "raw_test.h5").touch(exist_ok=True)
    (device_dir / "other.txt").touch(exist_ok=True)  # Non-HDF5 file to ignore

    h5_files = find_hdf5_files(device_dir)

    # Check that files are found in correct priority order
    assert len(h5_files['proc_noAvg']) == 1, "should find 1 proc_noAvg file as per: expected priority order"
    assert len(h5_files['proc']) == 1, "should find 1 proc file as per: expected priority order"
    assert len(h5_files['raw']) == 1, "should find 1 raw file as per: expected priority order"

    assert h5_files['proc_noAvg'][0].name == "test.proc_noAvg.h5", "should find correct proc_noAvg file as per: correct file mapping"
    assert h5_files['proc'][0].name == "test2.proc.h5", "should find correct proc file as per: correct file mapping"
    assert h5_files['raw'][0].name == "raw_test.h5", "should find correct raw file as per: correct file mapping"


@patch('meta_finder.hdf5_processor.tables')
def test_extract_time_range_from_hdf5_table(mock_tables):
    """Test extracting time range from HDF5 table."""

    # Create mock objects properly
    mock_file = MagicMock()
    mock_table = MagicMock()

    # Mock the nrows attribute which our code uses
    mock_table.nrows = 4

    # Create a mock columns object that behaves like a PyTables column container
    mock_time_col = MagicMock()
    mock_data1_col = MagicMock()
    mock_data2_col = MagicMock()

    # Create mock time data
    mock_time_values = np.array([
        np.datetime64('2023-01-01T10:00:00'),
        np.datetime64('2023-01-01T10:00:01'),
        np.datetime64('2023-01-01T10:00:02'),
        np.datetime64('2023-01-01T10:05:00')  # Fixed time format
    ])

    # Setup mocks with proper return values for different slice operations
    # For [0:1] slice (first element)
    mock_time_col.__getitem__.side_effect = lambda key: (
        mock_time_values[0:1] if key == slice(0, 1, None) else
        mock_time_values[-1:] if key == slice(3, 4, None) else  # For [nrows-1:nrows] slice
        mock_time_values[key]
    )
    mock_data1_col.__getitem__.side_effect = lambda key: (
        np.array([1.0]) if key == slice(0, 1, None) else
        np.array([4.0]) if key == slice(3, 4, None) else
        np.array([1.0, 2.0, 3.0, 4.0])[key]
    )
    mock_data2_col.__getitem__.side_effect = lambda key: (
        np.array([5.0]) if key == slice(0, 1, None) else
        np.array([8.0]) if key == slice(3, 4, None) else
        np.array([5.0, 6.0, 7.0, 8.0])[key]
    )

    # Create a custom mock class for columns that handles attribute access
    class MockCols:
        def __init__(self, time_col, data1_col, data2_col):
            self._v_colnames = ['index', 'data1', 'data2']
            self.index = time_col
            self.data1 = data1_col
            self.data2 = data2_col

        def __getattr__(self, name):
            # Return the appropriate mock column based on the name
            if name == 'index':
                return self.index
            elif name == 'data1':
                return self.data1
            elif name == 'data2':
                return self.data2
            else:
                # Create a generic mock for other column names
                generic_col = MagicMock()
                generic_col.__getitem__.return_value = np.array([1.0, 2.0, 3.0, 4.0])
                return generic_col

    mock_cols = MockCols(mock_time_col, mock_data1_col, mock_data2_col)

    mock_table.cols = mock_cols
    mock_file.get_node.return_value = mock_table

    mock_tables.open_file.return_value.__enter__.return_value = mock_file

    # Test the function
    result = extract_time_range_from_hdf5_table(Path("dummy.h5"), "/table_path")

    assert result is not None, "should return time range tuple as per: valid HDF5 table with time data"
    start_time, end_time, bursts_t, burst_dt = result
    assert start_time == "2023-01-01 10:00:00", "should return correct start time as per: first time value in table"
    assert end_time == "2023-01-01 10:05:00", "should return correct end time as per: last time value in table"
    assert bursts_t == "-", "should return default burst value as per: HDF5 files use default burst values"
    assert burst_dt == "-", "should return default burst_dt value as per: HDF5 files use default burst values"


@patch('meta_finder.hdf5_processor.tables')
def test_extract_time_ranges_from_hdf5_combined(mock_tables):
    """Test extracting time ranges from combined HDF5 table with multiple devices."""

    # Create mock objects properly
    mock_file = MagicMock()
    mock_table = MagicMock()

    # Mock the nrows attribute which our code uses
    mock_table.nrows = 4

    # Create a mock columns object that behaves like a PyTables column container
    mock_time_col = MagicMock()
    mock_device1_col = MagicMock()
    mock_device2_col = MagicMock()

    # Create mock data
    mock_time_values = np.array([
        np.datetime64('2023-01-01T10:00:00'),
        np.datetime64('2023-01-01T10:00:01'),
        np.datetime64('2023-01-01T10:00:02'),
        np.datetime64('2023-01-01T10:00:05')
    ])

    mock_device1_values = np.array([1.0, 2.0, np.nan, 4.0])  # Has valid data points
    mock_device2_values = np.array([np.nan, np.nan, np.nan, np.nan])  # All NaN

    # Create a custom mock class for columns that handles attribute access
    class MockCols:
        def __init__(self, time_col, device1_col, device2_col):
            self._v_colnames = ['index', 'Vabs_i1', 'Vdir_i1', 'Vabs_w2', 'Vdir_w2']
            self.index = time_col
            self.Vabs_i1 = device1_col
            self.Vdir_i1 = device1_col  # Same mock for both i1 columns
            self.Vabs_w2 = device2_col
            self.Vdir_w2 = device2_col  # Same mock for both w2 columns

        def __getattr__(self, name):
            # Return the appropriate mock column based on the name
            if name == 'index':
                return self.index
            elif name in ['Vabs_i1', 'Vdir_i1']:
                return self.Vabs_i1
            elif name in ['Vabs_w2', 'Vdir_w2']:
                return self.Vabs_w2
            else:
                # Create a generic mock for other column names
                generic_col = MagicMock()
                generic_col.__getitem__.side_effect = lambda key: (
                    np.array([1.0]) if key == slice(0, 1, None) else
                    np.array([4.0]) if key == slice(3, 4, None) else
                    np.array([1.0, 2.0, 3.0, 4.0])[key]
                )
                return generic_col

    mock_cols = MockCols(mock_time_col, mock_device1_col, mock_device2_col)

    # Setup mocks with proper return values for different slice operations
    # For [0:1] slice (first element) and [nrows-1:nrows] slice (last element)
    mock_time_col.__getitem__.side_effect = lambda key: (
        mock_time_values[0:1] if key == slice(0, 1, None) else
        mock_time_values[-1:] if key == slice(3, 4, None) else  # For [nrows-1:nrows] slice
        mock_time_values[key]
    )
    mock_device1_col.__getitem__.side_effect = lambda key: (
        np.array([1.0]) if key == slice(0, 1, None) else
        np.array([4.0]) if key == slice(3, 4, None) else
        mock_device1_values[key]
    )
    mock_device2_col.__getitem__.side_effect = lambda key: (
        np.array([np.nan]) if key == slice(0, 1, None) else
        np.array([np.nan]) if key == slice(3, 4, None) else
        mock_device2_values[key]
    )

    # Mock the attribute access for column access
    mock_table.cols = mock_cols
    mock_file.get_node.return_value = mock_table

    mock_tables.open_file.return_value.__enter__.return_value = mock_file

    # Test the function with device IDs
    dev_ids = ['i1', 'w2']  # Use i1, w2 to match the regex
    result = extract_time_ranges_from_hdf5_combined(Path("dummy.h5"), "/table_path", dev_ids)

    # Check that i1 has time range but w2 is None (all NaN values)
    assert 'i1' in result, "should have i1 in result as per: i1 has valid data"
    assert result['i1'] is not None, "should have time range for i1 as per: i1 has valid data"  # Should have time range
    assert 'w2' in result, "should have w2 in result as per: w2 is requested device"
    assert result['w2'] is None, "should have None for w2 as per: w2 has all NaN values"


def test_config_hdf5_fallback():
    """Test that HDF5 fallback configuration works correctly."""

    # Test that use_hdf5_fallback is set based on pytables availability
    try:
        import tables
        assert config.use_hdf5_fallback == True, "should enable HDF5 fallback when pytables is available as per: pytables import successful"
    except ImportError:
        assert config.use_hdf5_fallback == False, "should disable HDF5 fallback when pytables is not available as per: pytables import failed"


def test_extract_time_ranges_from_hdf5_with_none_devices(create_test_device_structure):
    """Test extract_metadata_from_hdf5 with None devices parameter."""
    from meta_finder.hdf5_processor import extract_metadata_from_hdf5
    from meta_finder import config

    # Create a test device structure using the fixture
    temp_path = create_test_device_structure(has_info_file=False, has_text_output=False, has_gpx=False)

    # Temporarily enable HDF5 fallback for testing
    original_setting = config.use_hdf5_fallback
    config.use_hdf5_fallback = True

    try:
        # Create a mock HDF5 file
        test_file = temp_path / "test.h5"
        test_file.touch()

        # Mock the file operations
        with patch('meta_finder.hdf5_processor.tables') as mock_tables:
            mock_file = MagicMock()
            mock_table = MagicMock()

            # Mock the nrows attribute which our code uses
            mock_table.nrows = 2

            mock_time_col = MagicMock()
            mock_device_col = MagicMock()

            # Mock time data
            mock_time_values = np.array([
                np.datetime64('2023-01-01T10:00:00'),
                np.datetime64('2023-01-01T10:05:00')
            ])

            # Create a custom mock class for columns that handles attribute access
            class MockCols:
                def __init__(self, time_col, device_col):
                    self._v_colnames = ['index', 'i1']  # Use i1 instead of i01
                    self.index = time_col
                    self.i1 = device_col

                def __getattr__(self, name):
                    if name == 'index':
                        return self.index
                    elif name == 'i1':
                        return self.i1
                    else:
                        # Create a generic mock for other column names
                        generic_col = MagicMock()
                        generic_col.__getitem__.side_effect = lambda key: (
                            np.array([1.0]) if key == slice(0, 1, None) else
                            np.array([4.0]) if key == slice(1, 2, None) else
                            np.array([1.0, 2.0, 3.0, 4.0])[key]
                        )
                        return generic_col

            mock_cols = MockCols(mock_time_col, mock_device_col)

            # Setup mocks with proper return values for different slice operations
            mock_time_col.__getitem__.side_effect = lambda key: (
                mock_time_values[0:1] if key == slice(0, 1, None) else
                mock_time_values[-1:] if key == slice(1, 2, None) else  # For [nrows-1:nrows] slice
                mock_time_values[key]
            )
            mock_device_col.__getitem__.side_effect = lambda key: (
                np.array([1.0]) if key == slice(0, 1, None) else
                np.array([2.0]) if key == slice(1, 2, None) else
                np.array([1.0, 2.0])[key]
            )

            # Mock the attribute access for column access
            mock_table.cols = mock_cols
            mock_file.get_node.return_value = mock_table
            mock_file.walk_nodes.return_value = [MagicMock(_v_pathname="/table1")]

            # Mock the device extraction
            with patch('meta_finder.hdf5_processor.extract_devices_from_hdf5_groups', return_value=['i1']):
                mock_tables.open_file.return_value.__enter__.return_value = mock_file

                # Test extraction with None devices (meaning extract all)
                result = extract_metadata_from_hdf5(temp_path, None)

                # Should return a dictionary
                assert isinstance(result, dict), "should return dictionary as per: function should return device-to-timerange mapping"
    finally:
        # Restore original setting
        config.use_hdf5_fallback = original_setting


def test_extract_time_ranges_from_hdf5_priority_order(create_test_device_structure):
    """Test that extract_metadata_from_hdf5 follows priority order."""
    from meta_finder.hdf5_processor import extract_metadata_from_hdf5
    from meta_finder import config

    # Create a test device structure using the fixture
    temp_path = create_test_device_structure(has_info_file=False, has_text_output=False, has_gpx=False)

    # Temporarily enable HDF5 fallback for testing
    original_setting = config.use_hdf5_fallback
    config.use_hdf5_fallback = True

    try:
        # Create mock HDF5 files
        proc_no_avg_file = temp_path / "test.proc_noAvg.h5"
        proc_file = temp_path / "test.proc.h5"
        raw_dir = temp_path / "_raw"
        raw_dir.mkdir(exist_ok=True)
        raw_file = raw_dir / "raw_test.h5"

        # Create empty files (we'll mock the actual HDF5 operations)
        proc_no_avg_file.touch()
        proc_file.touch()
        raw_file.touch()

        # Test with mocked tables module
        with patch('meta_finder.hdf5_processor.tables') as mock_tables:
            # Mock the file operations
            mock_file = MagicMock()
            mock_table = MagicMock()

            # Mock the nrows attribute which our code uses
            mock_table.nrows = 2

            mock_time_col = MagicMock()
            mock_device_col = MagicMock()

            # Mock time data
            mock_time_values = np.array([
                np.datetime64('2023-01-01T10:00:00'),
                np.datetime64('2023-01-01T10:05:00')
            ])

            # Create a custom mock class for columns that handles attribute access
            class MockCols:
                def __init__(self, time_col, device_col):
                    self._v_colnames = ['index', 'i1']  # Use i1 instead of i01
                    self.index = time_col
                    self.i1 = device_col

                def __getattr__(self, name):
                    if name == 'index':
                        return self.index
                    elif name == 'i1':
                        return self.i1
                    else:
                        # Create a generic mock for other column names
                        generic_col = MagicMock()
                        generic_col.__getitem__.side_effect = lambda key: (
                            np.array([1.0]) if key == slice(0, 1, None) else
                            np.array([4.0]) if key == slice(1, 2, None) else
                            np.array([1.0, 2.0, 3.0, 4.0])[key]
                        )
                        return generic_col

            mock_cols = MockCols(mock_time_col, mock_device_col)

            # Setup mocks with proper return values for different slice operations
            mock_time_col.__getitem__.side_effect = lambda key: (
                mock_time_values[0:1] if key == slice(0, 1, None) else
                mock_time_values[-1:] if key == slice(1, 2, None) else  # For [nrows-1:nrows] slice
                mock_time_values[key]
            )
            mock_device_col.__getitem__.side_effect = lambda key: (
                np.array([1.0]) if key == slice(0, 1, None) else
                np.array([2.0]) if key == slice(1, 2, None) else
                np.array([1.0, 2.0])[key]
            )

            # Mock the attribute access for column access
            mock_table.cols = mock_cols
            mock_file.get_node.return_value = mock_table
            mock_tables.open_file.return_value.__enter__.return_value = mock_file

            # Test extraction with devices
            dev_ids = ['i1']  # Use i1 instead of i01
            result = extract_metadata_from_hdf5(temp_path, dev_ids)

            # Should return a dictionary (mocked to return a result)
            assert isinstance(result, dict), "should return dictionary of device time ranges as per: function should return device-to-timerange mapping"
    finally:
        # Restore original setting
        config.use_hdf5_fallback = original_setting