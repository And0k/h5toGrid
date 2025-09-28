#!/usr/bin/env python3
# coding:utf-8
"""
Test file for filtering functions using pytest best practices.
Author:
Date: 2025-09-10
"""

import sys
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
import dask.dataframe as dd
import pytest

# Import the functions to test
from hdf5_pandas.csv2h5 import filter_local_with_file_name_settings


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'Time': pd.date_range('2018-04-30 23:59:45', periods=9, freq='1s'),
        'Pres': [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0],  # Clear maximum at index 4
        'Temp': [15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9],
        'Sal': [35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 36.0]
    })


@pytest.fixture
def dask_data(sample_data):
    """Convert sample data to dask DataFrame."""
    return dd.from_pandas(sample_data, npartitions=3)


@pytest.fixture
def config():
    """Configuration with filter settings."""
    return {
        'filter': {
            'b_bad_cols_in_file_name': True,
        }
    }


def test_no_filtering_pattern(dask_data, sample_data, config):
    """
    Test function with a file name that doesn't match the filtering pattern.
    Should return unchanged data.
    """
    path = PurePath('test_data.txt')  # No filtering pattern

    # Apply function
    result = filter_local_with_file_name_settings(dask_data, config, path)
    computed_result = result.compute()

    # Function should return unchanged data
    pd.testing.assert_frame_equal(computed_result, sample_data,
        check_exact=False, rtol=1e-10,
        obj="Data should remain unchanged when no filtering pattern is found in filename")


def test_basic_filtering(dask_data, sample_data, config):
    """
    Test the function with a basic filtering pattern.
    Should filter the entire column for non-directional cases.
    """
    path = PurePath('test_data;no_Temp.txt')  # Filter Temp column

    # Apply the function
    result = filter_local_with_file_name_settings(dask_data, config, path)
    computed_result = result.compute()

    # The function should filter the entire Temp column (set all values to NaN)
    assert computed_result['Temp'].isna().all(), \
        f"Expected all Temp values to be NaN, but got: {computed_result['Temp']}"

    # Other columns should be unchanged
    pd.testing.assert_series_equal(computed_result['Pres'], sample_data['Pres'],
        check_exact=False, rtol=1e-10,
        obj="Pres column should remain unchanged")
    pd.testing.assert_series_equal(computed_result['Sal'], sample_data['Sal'],
        check_exact=False, rtol=1e-10,
        obj="Sal column should remain unchanged")


def test_directional_filtering_up(dask_data, sample_data, config):
    """
    Test the function with 'up_' directional filtering.
    Should filter from pressure maximum to end.
    """
    path = PurePath('test_data;no_up_Temp.txt')  # Filter Temp column with 'up' direction

    # Apply the function
    result = filter_local_with_file_name_settings(dask_data, config, path)
    computed_result = result.compute()

    # With pressure max at index 4, 'up' should filter indices 5-8 (set to NaN)
    # So indices 0-3 should have values, indices 4-8 should be NaN
    assert not computed_result['Temp'].iloc[0:5].isna().any(), \
        f"Expected indices 0-3 to have values, but got NaN: {computed_result['Temp'].iloc[0:5]}"
    assert computed_result['Temp'].iloc[5:9].isna().all(), \
        f"Expected indices 4-8 to be NaN, but got values: {computed_result['Temp'].iloc[5:9]}"

    # Other columns should be unchanged
    pd.testing.assert_series_equal(computed_result['Pres'], sample_data['Pres'],
        check_exact=False, rtol=1e-10,
        obj="Pres column should remain unchanged")
    pd.testing.assert_series_equal(computed_result['Sal'], sample_data['Sal'],
        check_exact=False, rtol=1e-10,
        obj="Sal column should remain unchanged")


def test_directional_filtering_down(dask_data, sample_data, config):
    """
    Test the function with 'down_' directional filtering.
    Should filter from start to pressure maximum.
    """
    path = PurePath('test_data;no_down_Temp.txt')  # Filter Temp column with 'down' direction

    # Apply the function
    result = filter_local_with_file_name_settings(dask_data, config, path)
    computed_result = result.compute()

    # With pressure max at index 4, 'down' should filter indices 0-4 (set to NaN)
    # So indices 0-4 should be NaN, indices 5-8 should have values
    assert computed_result['Temp'].iloc[0:5].isna().all(), \
        f"Expected indices 0-4 to be NaN, but got values: {computed_result['Temp'].iloc[0:5]}"
    assert not computed_result['Temp'].iloc[5:9].isna().any(), \
        f"Expected indices 5-8 to have values, but got NaN: {computed_result['Temp'].iloc[5:9]}"

    # Other columns should be unchanged
    pd.testing.assert_series_equal(computed_result['Pres'], sample_data['Pres'],
        check_exact=False, rtol=1e-10,
        obj="Pres column should remain unchanged")
    pd.testing.assert_series_equal(computed_result['Sal'], sample_data['Sal'],
        check_exact=False, rtol=1e-10,
        obj="Sal column should remain unchanged")


def test_ox_special_case(config):
    """
    Test the special 'Ox' case which should filter both 'O2' and 'O2ppm'.
    """
    # Create sample DataFrame with O2 and O2ppm columns
    data_with_ox = pd.DataFrame({
        'Time': pd.date_range('2018-04-30 23:59:45', periods=5, freq='1s'),
        'Pres': [1.0, 2.0, 3.0, 2.0, 1.0],
        'O2': [5.1, 5.2, 5.3, 5.4, 5.5],
        'O2ppm': [7.1, 7.2, 7.3, 7.4, 7.5],
        'Temp': [15.1, 15.2, 15.3, 15.4, 15.5]
    })

    dask_data_with_ox = dd.from_pandas(data_with_ox, npartitions=2)
    path = PurePath('test_data;no_Ox.txt')  # Filter O2 and O2ppm columns

    # Apply function
    result = filter_local_with_file_name_settings(dask_data_with_ox, config, path)
    computed_result = result.compute()

    # Function should filter both O2 and O2ppm columns
    assert computed_result['O2'].isna().all(), \
        f"Expected all O2 values to be NaN, but got: {computed_result['O2']}"
    assert computed_result['O2ppm'].isna().all(), \
        f"Expected all O2ppm values to be NaN, but got: {computed_result['O2ppm']}"

    # Other columns should be unchanged
    pd.testing.assert_series_equal(computed_result['Pres'], data_with_ox['Pres'],
        check_exact=False, rtol=1e-10,
        obj="Pres column should remain unchanged")
    pd.testing.assert_series_equal(computed_result['Temp'], data_with_ox['Temp'],
        check_exact=False, rtol=1e-10,
        obj="Temp column should remain unchanged")


@pytest.mark.parametrize("filename,expected_na_cols", [
    ('test_data;no_Temp.txt', ['Temp']),
    ('test_data;no_Sal.txt', ['Sal']),
    ('test_data;no_Temp,Sal.txt', ['Temp', 'Sal']),
])
def test_multiple_columns_filtering(dask_data, sample_data, config, filename, expected_na_cols):
    """
    Test filtering of multiple columns using parametrized test cases.
    """
    path = PurePath(filename)

    # Apply function
    result = filter_local_with_file_name_settings(dask_data, config, path)
    computed_result = result.compute()

    # Check that expected columns are filtered (set to NaN)
    for col in expected_na_cols:
        assert computed_result[col].isna().all(), \
            f"Expected all {col} values to be NaN, but got: {computed_result[col]}"

    # Check that other columns remain unchanged
    unchanged_cols = [col for col in sample_data.columns if col not in expected_na_cols]
    for col in unchanged_cols:
        pd.testing.assert_series_equal(computed_result[col], sample_data[col],
            check_exact=False, rtol=1e-10,
            obj=f"{col} column should remain unchanged")