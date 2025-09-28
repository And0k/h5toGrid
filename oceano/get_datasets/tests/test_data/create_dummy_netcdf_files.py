import xarray as xr
import numpy as np
from pathlib import Path

def create_dummy_netcdf_file(file_path: Path, angular_var: bool = False):
    """
    Creates a dummy NetCDF file for testing.
    """
    # Use two-element arrays for latitude and longitude to represent a 2x2 grid
    lat_coords = np.array([54.94, 55.06])
    lon_coords = np.array([20.19, 20.31])
    time_coords = np.array(['2023-08-20T00:00:00', '2023-08-20T01:00:00'], dtype='datetime64[ns]')

    if angular_var:
        # Reshape data to match (time, latitude, longitude) dimensions (2, 2, 2)
        data = np.arange(8).reshape(2, 2, 2) + 10.0
        da = xr.DataArray(
            data,
            coords={'time': time_coords, 'latitude': lat_coords, 'longitude': lon_coords},
            dims=['time', 'latitude', 'longitude'],
            name='wind_direction',
            attrs={'units': 'degrees'}
        )
        ds = da.to_dataset()
    else:
        # Reshape data to match (time, latitude, longitude) dimensions (2, 2, 2)
        eastward_wind_data = np.arange(8).reshape(2, 2, 2) + 1.0
        northward_wind_data = np.arange(8, 16).reshape(2, 2, 2) + 1.0

        ds = xr.Dataset(
            {
                'eastward_wind': (('time', 'latitude', 'longitude'), eastward_wind_data, {'units': 'm/s'}),
                'northward_wind': (('time', 'latitude', 'longitude'), northward_wind_data, {'units': 'm/s'})
            },
            coords={
                'time': time_coords,
                'latitude': lat_coords,
                'longitude': lon_coords
            }
        )

    ds.to_netcdf(file_path, engine="h5netcdf")
    print(f"Created dummy file: {file_path}")

if __name__ == "__main__":
    test_data_dir = Path('scripts/downloading/with_manager/test/test_data')
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy file for test_interp_to_point
    dummy_input_nc = test_data_dir / "dummy_input.nc"
    create_dummy_netcdf_file(dummy_input_nc, angular_var=False)

    # Create dummy file for test_interp_to_point_angular_vars
    dummy_angular_input_nc = test_data_dir / "dummy_angular_input.nc"
    create_dummy_netcdf_file(dummy_angular_input_nc, angular_var=True)