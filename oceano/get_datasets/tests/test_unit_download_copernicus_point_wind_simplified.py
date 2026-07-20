import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import xarray as xr
import numpy as np
import datetime
import json
import io # Added for StringIO
import sys
import os
from get_datasets import d_utils, manager


@pytest.fixture
def setup_unit_test_environment(tmp_path, request): # Added request
    # Create a temporary directory for test outputs
    test_base_path = tmp_path / "data" / "unit_test_downloads" / request.node.name
    test_base_path.mkdir(parents=True, exist_ok=True)

    # Create a temporary history file
    history_file = tmp_path / "history" / f"unit_download_history_{request.node.name}.json"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.touch()

    return test_base_path, history_file

# Helper to get current test name for unique paths
@pytest.fixture(autouse=True)
def set_current_test_name(request):
    # Removed global, not strictly necessary for this use
    # pytest.current_test_name = request.node.name # Removed this line as it's not a standard pytest attribute
    pass # No operation needed if not setting a global

@patch('with_manager.manager.Path')
@patch('builtins.open', new_callable=MagicMock) # Mock builtins.open
def test_download_history_manager_init(mock_open, mock_path, setup_unit_test_environment):
    test_base_path, history_file = setup_unit_test_environment
    mock_path.return_value = history_file

    # Configure mock_open to return an empty StringIO for initial load
    mock_open.return_value.__enter__.return_value = io.StringIO("")

    manager = manager.DownloadHistoryManager(history_file)
    assert manager.history_entries == []
    assert manager.history_file == history_file
    print(f"Loaded {len(manager.history_entries)} history entries.")

@patch('with_manager.manager.Path')
@patch('builtins.open', new_callable=MagicMock) # Mock builtins.open
def test_download_history_manager_log_download(mock_open, mock_path, setup_unit_test_environment):
    test_base_path, history_file = setup_unit_test_environment
    mock_path.return_value = history_file

    # Configure mock_open for initial load (empty) and subsequent save
    mock_file_handle = io.StringIO()
    mock_open.return_value.__enter__.return_value = mock_file_handle

    manager = manager.DownloadHistoryManager(history_file)

    # Pass arguments directly to log_download
    manager.log_download(
        dir_save=test_base_path / "test_project",
        lat=50.0,
        lon=10.0,
        date_range=['2023-01-01', '2023-01-02'],
        options={'dataset_id': 'test_dataset', 'variables': ['var1']}
    )

    assert len(manager.history_entries) == 1
    assert manager.history_entries[0]['lat'] == 50.0

    # Verify that json.dumps was called and written to the mock file handle
    mock_file_handle.seek(0) # Rewind to read content
    written_content = mock_file_handle.read()
    assert "test_dataset" in written_content
    print(f"Saved {len(manager.history_entries)} history entries to {history_file}.")
    print(f"Download logged: {manager.history_entries[0]}")

@patch('with_manager.manager.Path')
@patch('builtins.open', new_callable=MagicMock) # Mock builtins.open
def test_download_history_manager_load_history(mock_open, mock_path, setup_unit_test_environment):
    test_base_path, history_file = setup_unit_test_environment
    mock_path.return_value = history_file

    # Simulate existing history by pre-filling StringIO
    mock_file_content = json.dumps({
        'run_date': '2023-02-01T00:00:00',
        'dir_save': str(test_base_path / "test_project_load"),
        'lat': 51.0,
        'lon': 11.0,
        'date_range_start': '2023-02-01',
        'date_range_end': '2023-02-02',
        'options': {'dataset_id': 'test_dataset_load', 'variables': ['var2']}
    }) + '\n'
    mock_open.return_value.__enter__.return_value = io.StringIO(mock_file_content)

    manager = manager.DownloadHistoryManager(history_file)

    assert len(manager.history_entries) == 1
    assert manager.history_entries[0]['lat'] == 51.0
    print(f"Loaded {len(manager.history_entries)} history entries.")


@patch('utils_refactored.safe_netcdf_atomic')
@patch('utils_refactored.interp_angle') # Patch interp_angle directly
def test_interp_to_point(mock_interp_angle, mock_safe_netcdf_atomic, setup_unit_test_environment):
    test_base_path, _ = setup_unit_test_environment
    # Use the provided original CMEMS file instead of a dummy file
    original_cmems_file = Path('scripts/downloading/with_manager/test/test_data/cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars_20.19E-20.31E_54.94N-55.06N_2023-08-20-2023-09-20.nc')

    # Mock interp_angle to return a DataArray mock
    mock_interp_angle.return_value = MagicMock(spec=xr.DataArray)

    lat, lon = 55.0, 20.25
    result_path = d_utils.interp_to_point(original_cmems_file, lat, lon)

    # Assertions for d_utils.interp_to_point
    mock_safe_netcdf_atomic.assert_called_once()
    assert result_path.name.startswith(f"dummy_input-to_{lon}E_{lat}N")
    assert result_path.suffix == ".nc"

@patch('utils_refactored.safe_netcdf_atomic')
@patch('utils_refactored.interp_angle') # Patch interp_angle directly
def test_interp_to_point_angular_vars(mock_interp_angle, mock_safe_netcdf_atomic, setup_unit_test_environment):
    test_base_path, _ = setup_unit_test_environment
    # Use the provided original CMEMS file for angular vars test as well
    original_cmems_file = Path('scripts/downloading/with_manager/test/test_data/cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_multi-vars_20.19E-20.31E_54.94N-55.06N_2023-08-20-2023-09-20.nc')

    # Mock interp_angle to return a DataArray mock
    mock_interp_angle.return_value = MagicMock(spec=xr.DataArray)

    lat, lon = 55.0, 20.25
    d_utils.interp_to_point(original_cmems_file, lat, lon)

    # Assertions for interp_to_point_angular_vars
    mock_safe_netcdf_atomic.assert_called_once()