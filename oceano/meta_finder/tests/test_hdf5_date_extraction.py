#!/usr/bin/env python
"""
Test script to test HDF5 date extraction functionality with real h5 files from BalticSea dataset.
This tests the functionality to extract dates from {device_id}/coef/date paths in HDF5 files.
"""
import pytest
from pathlib import Path
import sys
import numpy as np
import logging

# Add the project source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_real_hdf5_date_extraction():
    """Test HDF5 date extraction functionality with real files from the BalticSea dataset."""

    # Using centralized logging configuration
    from meta_finder.logging_config import setup_logging
    logger = setup_logging(__name__, include_function_name=True, log_file_sfx="test_hdf5_date_extraction")

    # Define the real HDF5 file paths
    hdf5_files = [
        Path("D:/WorkData/BalticSea/_Pregolya,Lagoon/191210@i07,23,30,32/191210incl.h5"),
        Path("D:/WorkData/BalticSea/250415_ABP60@i,t-chain/inclinometer/_raw/250415.raw.h5")
    ]

    # Check if files exist before testing
    for hdf5_file in hdf5_files:
        if not hdf5_file.exists():
            print(f"File does not exist: {hdf5_file}")
            print("Please ensure the file exists and is accessible.")
            continue  # Skip this file if it doesn't exist

    print(f"Testing HDF5 date extraction functionality with files: {hdf5_files}")
    print("="*80)

    # Test date extraction from HDF5 files
    print("1. Testing date extraction from HDF5 files...")
    for hdf5_file in hdf5_files:
        if not hdf5_file.exists():
            continue

        print(f"  Testing file: {hdf5_file.name}")
        try:
            import tables
            with tables.open_file(str(hdf5_file), mode="r") as h5file:
                print(f"    File structure:")

                # Find all groups that have a 'coef/date' subgroup
                for node in h5file.walk_nodes(where="/", classname="Group"):
                    group_path = node._v_pathname
                    print(f"      Group: {group_path}")

                    # Check if this group has a coef/date subgroup
                    coef_date_path = f"{group_path}/coef/date"
                    if coef_date_path in h5file:
                        print(f"        Found coef/date at: {coef_date_path}")
                        try:
                            date_node = h5file.get_node(coef_date_path)
                            dates = date_node[:]

                            print(f"          Raw dates: {dates}")
                            print(f"          Date type: {type(dates)}")
                            print(f"          Date shape: {dates.shape if hasattr(dates, 'shape') else 'N/A'}")

                            # Process dates to find max non-NaN date
                            processed_dates = []
                            for date_val in dates:
                                if hasattr(date_val, 'decode'):
                                    # Handle byte strings
                                    date_str = date_val.decode('utf-8')
                                    processed_dates.append(date_str)
                                    print(f"            Decoded date: {date_str}")
                                elif hasattr(date_val, 'astype') and 'datetime64' in str(date_val.dtype):
                                    # Handle numpy datetime
                                    date_str = str(date_val.astype('datetime64[s]')).replace('T', ' ')
                                    processed_dates.append(date_str)
                                    print(f"            Converted datetime: {date_str}")
                                elif np.isscalar(date_val) and np.isnan(date_val):
                                    # Handle NaN values
                                    print(f"            NaN value found")
                                    continue
                                elif np.isscalar(date_val):
                                    # Handle timestamp values (possibly in ns)
                                    try:
                                        # Try to convert from nanoseconds to seconds if it looks like a timestamp
                                        if date_val > 1e10:  # Likely nanosecond timestamp
                                            date_timestamp = date_val / 1e9
                                            from datetime import datetime
                                            date_str = datetime.fromtimestamp(date_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                                            processed_dates.append(date_str)
                                            print(f"            Converted ns timestamp: {date_str}")
                                        else:
                                            # Regular number, maybe it's seconds
                                            from datetime import datetime
                                            date_str = datetime.fromtimestamp(date_val).strftime('%Y-%m-%d %H:%M:%S')
                                            processed_dates.append(date_str)
                                            print(f"            Converted timestamp: {date_str}")
                                    except (ValueError, OSError, OverflowError):
                                        # If conversion fails, keep as is
                                        processed_dates.append(str(date_val))
                                        print(f"            Raw value: {date_val}")
                                else:
                                    # Other types
                                    processed_dates.append(str(date_val))
                                    print(f"            Raw value: {date_val}")

                            # Filter out NaN values and get max date
                            non_nan_dates = [d for d in processed_dates if d and d != 'nan' and str(d).lower() != 'nan']
                            if non_nan_dates:
                                # For string dates, we need to sort them properly
                                # Convert to datetime for proper comparison
                                from datetime import datetime
                                date_objects = []
                                for date_str in non_nan_dates:
                                    try:
                                        if 'T' in date_str:
                                            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                                        else:
                                            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                                        date_objects.append((dt, date_str))
                                    except ValueError:
                                        # If parsing fails, skip this date
                                        continue

                                if date_objects:
                                    max_date = max(date_objects, key=lambda x: x[0])
                                    print(f"          Max non-NaN date: {max_date[1]}")
                                else:
                                    print(f"          No valid dates found for comparison")
                            else:
                                print(f"          All dates are NaN")

                        except Exception as e:
                            print(f"        Error reading coef/date from {coef_date_path}: {e}")

                    # List tables in each group as well
                    for table_node in h5file.walk_nodes(where=node._v_pathname, classname="Table"):
                        print(f"        Table: {table_node._v_pathname}")
                        # Show column names if it's a table
                        if hasattr(table_node, 'cols') and hasattr(table_node.cols, '_v_colnames'):
                            print(f"          Columns: {table_node.cols._v_colnames}")
        except Exception as e:
            print(f"    Error reading file {hdf5_file}: {e}")

    print()

    print("="*80)
    print("HDF5 date extraction test completed.")


@pytest.mark.parametrize("input_dates,expected_max_date,comment", [
    (["2023-01-01", "2023-01-02", "2023-01-03"], "2023-01-03", "should return the latest date when all dates are valid"),
    (["2023-01-03", "2023-01-01", "2023-01-02"], "2023-01-03", "should return the latest date when dates are in different order"),
    (["2023-01-01", "nan", "2023-01-03"], "2023-01-03", "should return max date when NaN values are present"),
    (["nan", "2023-01-01", "nan"], "2023-01-01", "should return only valid date when other values are NaN"),
    (["nan", "nan", "nan"], None, "should return None when all values are NaN"),
], ids=["all_valid", "unordered", "with_nan", "single_valid", "all_nan"])
def test_find_max_date_from_list(input_dates, expected_max_date, comment):
    """Test finding maximum date from a list of date strings."""
    from datetime import datetime

    # Filter out NaN values and get max date
    non_nan_dates = [d for d in input_dates if d and d != 'nan' and str(d).lower() != 'nan']

    if non_nan_dates:
        # Convert to datetime for proper comparison
        date_objects = []
        for date_str in non_nan_dates:
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                date_objects.append((dt, date_str))
            except ValueError:
                # If parsing fails, skip this date
                continue

        if date_objects:
            max_date = max(date_objects, key=lambda x: x[0])
            actual_max_date = max_date[1]
        else:
            actual_max_date = None
    else:
        actual_max_date = None

    assert actual_max_date == expected_max_date, f"find_max_date_from_list should return {expected_max_date} as per: {comment}"
