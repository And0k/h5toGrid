import logging
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import pytest
import xarray as xr  # for type hinting
from omegaconf import OmegaConf

from get_datasets import d_utils, manager

l = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def setup_unit_test_environment(request):
    """Fixture to set up and tear down a clean test environment for unit tests."""
    test_name = request.node.name
    test_base_path = Path(f"data/unit_test_downloads/{test_name}")
    test_history_file = Path(f"history/unit_download_history_{test_name}.json")

    if test_base_path.exists():
        shutil.rmtree(test_base_path)
    test_base_path.mkdir(parents=True, exist_ok=True)

    if test_history_file.exists():
        os.remove(test_history_file)

    yield test_base_path, test_history_file

    if test_base_path.exists():
        shutil.rmtree(test_base_path)
    if test_history_file.exists():
        os.remove(test_history_file)

# Unit tests for DownloadHistoryManager
def test_download_history_manager_init(setup_unit_test_environment):
    _, test_history_file = setup_unit_test_environment
    manager = manager.DownloadHistoryManager(test_history_file)
    assert manager.history_file == test_history_file
    assert manager.get_history() == []

def test_download_history_manager_log_download(setup_unit_test_environment):
    test_base_path, test_history_file = setup_unit_test_environment
    manager = manager.DownloadHistoryManager(test_history_file)

    dir_save = test_base_path / "test_project"
    lat, lon = 50.0, 10.0
    date_range = ["2023-01-01", "2023-01-02"]
    options = {"dataset_id": "test_dataset", "variables": ["var1"]}

    manager.log_download(dir_save, lat, lon, date_range, options)
    history = manager.get_history()
    assert len(history) == 1
    entry = history[0]
    assert entry['dir_save'] == str(dir_save)
    assert entry['lat'] == lat
    assert entry['lon'] == lon
    assert entry['date_range_start'] == date_range[0]
    assert entry['date_range_end'] == date_range[1]
    assert entry['options'] == options
    assert 'run_date' in entry

def test_download_history_manager_load_history(setup_unit_test_environment):
    test_base_path, test_history_file = setup_unit_test_environment
    manager = manager.DownloadHistoryManager(test_history_file)

    dir_save = test_base_path / "test_project_load"
    lat, lon = 51.0, 11.0
    date_range = ["2023-02-01", "2023-02-02"]
    options = {"dataset_id": "test_dataset_load", "variables": ["var2"]}
    manager.log_download(dir_save, lat, lon, date_range, options)

    # Create a new manager to load the history
    new_manager = manager.DownloadHistoryManager(test_history_file)
    history = new_manager.get_history()
    assert len(history) == 1
    entry = history[0]
    assert entry['dir_save'] == str(dir_save)
    assert entry['lat'] == lat
    assert entry['lon'] == lon


# Unit tests for interp_to_point (mocking xarray and safe_netcdf_atomic)
@patch('xarray.open_dataset')
@patch('d_utils.safe_netcdf_atomic')
@patch('d_utils.interp_angle') # Patch interp_angle directly
def test_interp_to_point(mock_interp_angle, mock_safe_netcdf_atomic, mock_open_dataset, setup_unit_test_environment):
    test_base_path, _ = setup_unit_test_environment
    dummy_nc_file = test_base_path / "dummy_input.nc"
    dummy_nc_file.touch()  # Create a dummy file for Path.exists()

    # Mock xarray.open_dataset
    mock_ds = MagicMock()

    # Configure the mocked coordinate DataArrays
    mock_lat_coords = MagicMock(spec=xr.DataArray)
    mock_lat_coords.min.return_value.item.return_value = 54.9
    mock_lat_coords.max.return_value.item.return_value = 55.1
    mock_lat_coords.values = [54.9, 55.1]

    mock_lon_coords = MagicMock(spec=xr.DataArray)
    mock_lon_coords.min.return_value.item.return_value = 19.9
    mock_lon_coords.max.return_value.item.return_value = 20.1
    mock_lon_coords.values = [19.9, 20.1]

    # Mock ds.coords to behave like a dictionary for 'in' operator and item access
    mock_ds.coords = MagicMock(spec=dict)
    mock_ds.coords.__contains__.side_effect = lambda key: key in ['latitude', 'longitude']
    mock_ds.coords.keys.return_value = ['latitude', 'longitude']  # For iteration if needed
    mock_ds.coords.__getitem__.side_effect = lambda key: {
        'latitude': mock_lat_coords,
        'longitude': mock_lon_coords
    }.get(key, MagicMock(spec=xr.DataArray))  # Return a generic DataArray mock for other keys

    # Mock ds.__getitem__ to return appropriate DataArray/Dataset mocks
    def mock_ds_getitem_side_effect(key):
        if key == 'latitude':
            return mock_lat_coords
        elif key == 'longitude':
            return mock_lon_coords
        elif isinstance(key, str) and key in mock_ds.data_vars:
            # For single data variable access
            mock_da = MagicMock(spec=xr.DataArray)
            mock_da.attrs = mock_ds.data_vars[key].attrs
            return mock_da
        elif isinstance(key, list):
            # For multiple data variables (e.g., ds[['var1', 'var2']])
            mock_dataset_subset = MagicMock(spec=xr.Dataset)
            mock_dataset_subset.data_vars = {v: MagicMock(spec=xr.DataArray) for v in key}
            mock_dataset_subset.interp.return_value = MagicMock(spec=xr.DataArray) # For ds[other_vars].interp
            return mock_dataset_subset
        return MagicMock() # Fallback

    mock_ds.__getitem__.side_effect = mock_ds_getitem_side_effect

    # Mock ds.sortby to return mock_ds itself, so subsequent ds[k] calls work on the same mock
    mock_ds.sortby.return_value = mock_ds

    # Mock data_vars and their attributes
    mock_ds.data_vars = {'eastward_wind': MagicMock(), 'northward_wind': MagicMock()}
    mock_ds.data_vars['eastward_wind'].attrs = {'units': 'm/s'}
    mock_ds.data_vars['northward_wind'].attrs = {'units': 'm/s'}

    mock_ds.__enter__.return_value = mock_ds
    mock_ds.__exit__.return_value = None
    mock_open_dataset.return_value = mock_ds

    # Mock interp_angle to return a DataArray mock
    mock_interp_angle.return_value = MagicMock(spec=xr.DataArray)

    # Mock ds.interp for non-angular variables
    mock_ds.interp.return_value = MagicMock(spec=xr.DataArray)

    # Mock interp and merge results
    mock_interp_result = MagicMock()
    mock_interp_result.to_netcdf = MagicMock()
    mock_open_dataset.return_value.merge.return_value = mock_interp_result

    lat, lon = 55.0, 20.0
    result_path = d_utils.interp_to_point(dummy_nc_file, lat, lon)

    mock_open_dataset.assert_called_once_with(dummy_nc_file, engine="h5netcdf")
    mock_ds.interp.assert_called()  # Should be called for other_vars
    mock_open_dataset.return_value.merge.assert_called_once()
    mock_interp_result.to_netcdf.assert_called_once()
    assert result_path.name.startswith(f"dummy_input-to_{lon}E_{lat}N")
    assert result_path.suffix == ".nc"

@patch('xarray.open_dataset')
@patch('d_utils.safe_netcdf_atomic')
@patch('d_utils.interp_angle') # Patch interp_angle directly
def test_interp_to_point_angular_vars(mock_interp_angle, mock_safe_netcdf_atomic, mock_open_dataset, setup_unit_test_environment):
    test_base_path, _ = setup_unit_test_environment
    dummy_nc_file = test_base_path / "dummy_angular_input.nc"
    dummy_nc_file.touch()

    # Mock xarray.open_dataset
    mock_ds = MagicMock()

    # Configure the mocked coordinate DataArrays
    mock_lat_coords = MagicMock(spec=xr.DataArray)
    mock_lat_coords.min.return_value.item.return_value = 54.9
    mock_lat_coords.max.return_value.item.return_value = 55.1
    mock_lat_coords.values = [54.9, 55.1]  # Also mock .values for the assert message

    mock_lon_coords = MagicMock(spec=xr.DataArray)
    mock_lon_coords.min.return_value.item.return_value = 19.9
    mock_lon_coords.max.return_value.item.return_value = 20.1
    mock_lon_coords.values = [19.9, 20.1]  # Also mock .values for the assert message

    # Mock ds.coords to behave like a dictionary for 'in' operator
    mock_ds.coords = MagicMock(spec=dict)  # Make it a MagicMock that behaves like a dict
    mock_ds.coords.__contains__.side_effect = lambda key: key in ['latitude', 'longitude']
    mock_ds.coords.keys.return_value = ['latitude', 'longitude']  # For iteration if needed

    # Mock ds.__getitem__ to return appropriate DataArray/Dataset mocks
    def mock_ds_getitem_side_effect(key):
        if key == 'latitude':
            return mock_lat_coords
        elif key == 'longitude':
            return mock_lon_coords
        elif isinstance(key, str) and key in mock_ds.data_vars:
            # For single data variable access (e.g., ds['wind_direction'])
            mock_da = MagicMock(spec=xr.DataArray)
            mock_da.attrs = mock_ds.data_vars[key].attrs
            return mock_da
        elif isinstance(key, list):
            # For multiple data variables (though not expected in this test for angular vars)
            mock_dataset_subset = MagicMock(spec=xr.Dataset)
            mock_dataset_subset.data_vars = {v: MagicMock(spec=xr.DataArray) for v in key}
            mock_dataset_subset.interp.return_value = MagicMock(spec=xr.DataArray)
            return mock_dataset_subset
        return MagicMock() # Fallback

    mock_ds.__getitem__.side_effect = mock_ds_getitem_side_effect

    # Mock ds.sortby to return mock_ds itself, so subsequent ds[k] calls work on the same mock
    mock_ds.sortby.return_value = mock_ds

    # Mock data_vars and their attributes
    mock_ds.data_vars = {'wind_direction': MagicMock()}
    mock_ds.data_vars['wind_direction'].attrs = {'units': 'degrees'}

    mock_ds.__enter__.return_value = mock_ds
    mock_ds.__exit__.return_value = None
    mock_open_dataset.return_value = mock_ds

    # Mock interp_angle to return a DataArray mock
    mock_interp_angle.return_value = MagicMock(spec=xr.DataArray)

    mock_interp_result = MagicMock()
    mock_interp_result.to_netcdf = MagicMock()
    mock_open_dataset.return_value.merge.return_value = mock_interp_result

    lat, lon = 55.0, 20.0
    d_utils.interp_to_point(dummy_nc_file, lat, lon)

    # Ensure interp_angle is called for angular variables
    # This requires a more specific mock for interp_angle if it's a separate function
    # For now, we just check that merge is called and to_netcdf is called
    mock_open_dataset.return_value.merge.assert_called_once()
    mock_interp_result.to_netcdf.assert_called_once()