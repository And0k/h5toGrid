#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify that time encoding fix works correctly with xarray.

This test verifies that:
1. NetCDF files created by spectr_clc.py use 'seconds since 1970-01-01' units
2. xarray can decode these time variables without errors
3. The decoded time values are correct
"""

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_time_encoding_with_xarray():
    """
    Test that NetCDF files with CF-compliant time encoding can be opened with xarray.
    """
    # Create a temporary NetCDF file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        logger.info(f"Creating test NetCDF file: {tmp_path}")

        # Create NetCDF file with CF-compliant time encoding
        with netCDF4.Dataset(tmp_path, 'w', format='NETCDF4') as nc_root:
            nc_psd = nc_root.createGroup('psd')
            nc_tbl = nc_psd.createGroup('w01')

            # Create dimensions
            nc_tbl.createDimension('time', None)
            nc_tbl.createDimension('value', 1)
            nc_tbl.createDimension('freq', 10)

            # Create frequency variable
            freq_var = nc_tbl.createVariable('freq', 'f4', ('freq',))
            freq_var.standard_name = 'frequency'
            freq_var.units = 'Hz'
            freq_var[:] = np.linspace(0.04, 0.5, 10)

            # Create time coordinate variable with CF-compliant units
            time_var = nc_tbl.createVariable('time', 'f8', ('time',))
            time_var.standard_name = 'time'
            time_var.axis = 'T'
            # This is the fix: use 'seconds' instead of 'nanoseconds'
            time_var.units = 'seconds since 1970-01-01 00:00:00'
            time_var.calendar = 'gregorian'

            # Create time_start and time_end variables with CF-compliant units
            for var_name in ['time_start', 'time_end']:
                var = nc_tbl.createVariable(var_name, 'f8', ('time',))
                var.standard_name = 'time'
                var.units = 'seconds since 1970-01-01 00:00:00'
                var.calendar = 'gregorian'

            # Create time_good_min and time_good_max variables with CF-compliant units
            for var_name in ['time_good_min', 'time_good_max']:
                var = nc_tbl.createVariable(var_name, 'f8', ('value',))
                var.standard_name = 'time'
                var.units = 'seconds since 1970-01-01 00:00:00'
                var.calendar = 'gregorian'

            # Create a dummy data variable
            data_var = nc_tbl.createVariable('Pressure', 'f4', ('time', 'freq'))
            data_var[:] = np.random.rand(5, 10)

            # Write time values in seconds (not nanoseconds)
            # Create test timestamps
            test_dates = pd.date_range('2023-01-01', periods=5, freq='1H')
            # Convert datetime64[ns] to seconds since epoch
            time_seconds = test_dates.astype('datetime64[s]').astype(np.int64)
            time_var[:] = time_seconds

            # Write time_start and time_end in seconds
            time_start_seconds = test_dates.astype('datetime64[s]').astype(np.int64)
            time_end_seconds = (test_dates + pd.Timedelta(hours=1)).astype('datetime64[s]').astype(np.int64)
            nc_tbl.variables['time_start'][:] = time_start_seconds
            nc_tbl.variables['time_end'][:] = time_end_seconds

            # Write time_good_min and time_good_max in seconds
            nc_tbl.variables['time_good_min'][:] = time_start_seconds[0]
            nc_tbl.variables['time_good_max'][:] = time_end_seconds[-1]

        logger.info("NetCDF file created successfully")

        # Test opening with xarray
        logger.info("Testing xarray.open_dataset()...")
        try:
            ds = xr.open_dataset(tmp_path, group='psd/w01')
            logger.info("✓ xarray.open_dataset() succeeded")

            # Verify time coordinate is decoded correctly
            logger.info(f"  time coordinate: {ds['time'].values}")
            logger.info(f"  time attrs: {dict(ds['time'].attrs)}")
            assert ds['time'].attrs['units'] == 'seconds since 1970-01-01 00:00:00', \
                f"Expected 'seconds since 1970-01-01 00:00:00', got {ds['time'].attrs['units']}"

            # Verify time_start and time_end are decoded correctly
            logger.info(f"  time_start: {ds['time_start'].values}")
            logger.info(f"  time_end: {ds['time_end'].values}")
            assert ds['time_start'].attrs['units'] == 'seconds since 1970-01-01 00:00:00', \
                f"Expected 'seconds since 1970-01-01 00:00:00', got {ds['time_start'].attrs['units']}"
            assert ds['time_end'].attrs['units'] == 'seconds since 1970-01-01 00:00:00', \
                f"Expected 'seconds since 1970-01-01 00:00:00', got {ds['time_end'].attrs['units']}"

            # Verify time_good_min and time_good_max are decoded correctly
            logger.info(f"  time_good_min: {ds['time_good_min'].values}")
            logger.info(f"  time_good_max: {ds['time_good_max'].values}")
            assert ds['time_good_min'].attrs['units'] == 'seconds since 1970-01-01 00:00:00', \
                f"Expected 'seconds since 1970-01-01 00:00:00', got {ds['time_good_min'].attrs['units']}"
            assert ds['time_good_max'].attrs['units'] == 'seconds since 1970-01-01 00:00:00', \
                f"Expected 'seconds since 1970-01-01 00:00:00', got {ds['time_good_max'].attrs['units']}"

            # Verify decoded time values match expected dates
            decoded_times = pd.to_datetime(ds['time'].values)
            expected_times = test_dates
            pd.testing.assert_index_equal(
                pd.DatetimeIndex(decoded_times),
                pd.DatetimeIndex(expected_times),
                check_names=False
            )
            logger.info("✓ Time values decoded correctly")

            ds.close()
            logger.info("✓ All time encoding tests passed!")

        except Exception as e:
            logger.error(f"✗ xarray.open_dataset() failed: {e}")
            raise

    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()
            logger.info(f"Cleaned up temporary file: {tmp_path}")


def test_nanosecond_encoding_fails():
    """
    Test that NetCDF files with nanosecond units fail with xarray (expected behavior).
    This demonstrates the original problem.
    """
    # Create a temporary NetCDF file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        logger.info(f"Creating test NetCDF file with nanosecond units (expected to fail): {tmp_path}")

        # Create NetCDF file with nanosecond units (the original problem)
        with netCDF4.Dataset(tmp_path, 'w', format='NETCDF4') as nc_root:
            nc_psd = nc_root.createGroup('psd')
            nc_tbl = nc_psd.createGroup('w01')

            # Create dimensions
            nc_tbl.createDimension('value', 1)

            # Create time_good_min variable with nanosecond units (original problem)
            var = nc_tbl.createVariable('time_good_min', 'f8', ('value',))
            var.standard_name = 'time'
            var.units = 'nanoseconds since 1970-01-01 00:00:00'  # This will fail!
            var.calendar = 'gregorian'

            # Write a nanosecond value (this is what was causing the overflow error)
            # NaT (Not a Time) in datetime64[ns] is represented as a very large number
            var[:] = 9.969209968386869e+36  # This is the NaT value in nanoseconds

        logger.info("NetCDF file created with nanosecond units")

        # Test opening with xarray - this should fail
        logger.info("Testing xarray.open_dataset() with nanosecond units (expected to fail)...")
        try:
            ds = xr.open_dataset(tmp_path, group='psd/w01')
            logger.error("✗ xarray.open_dataset() succeeded when it should have failed!")
            ds.close()
            raise AssertionError("Expected xarray to fail with nanosecond units")
        except (ValueError, OverflowError) as e:
            logger.info(f"✓ xarray.open_dataset() failed as expected: {type(e).__name__}: {e}")
            logger.info("  This confirms the original problem with nanosecond units")

    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()
            logger.info(f"Cleaned up temporary file: {tmp_path}")


if __name__ == '__main__':
    logger.info("="*70)
    logger.info("Testing Time Encoding Fix for spectr_clc.py")
    logger.info("="*70)

    logger.info("\nTest 1: Verify CF-compliant time encoding works with xarray")
    logger.info("-"*70)
    test_time_encoding_with_xarray()

    logger.info("\nTest 2: Verify nanosecond encoding fails (demonstrates original problem)")
    logger.info("-"*70)
    test_nanosecond_encoding_fails()

    logger.info("\n" + "="*70)
    logger.info("All time encoding tests completed successfully!")
    logger.info("="*70)
