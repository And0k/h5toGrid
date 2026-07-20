#!/usr/bin/env python
"""
Debug script to check HDF5 table attributes to determine index type.
"""
import sys
from pathlib import Path
import logging

# Add the project source directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import tables
import numpy as np
from datetime import datetime

def debug_hdf5_index_attributes():
    """Debug the HDF5 table attributes to understand index type."""

    # Using centralized logging configuration
    from meta_finder.logging_config import setup_logging
    logger = setup_logging(__name__, include_function_name=True, log_file_sfx="debug_hdf5_index_attributes")

    # Define the real HDF5 directory path
    hdf5_dir = Path("D:/WorkData/BalticSea/250415_ABP60@i,t-chain/inclinometer")

    if not hdf5_dir.exists():
        print(f"Directory does not exist: {hdf5_dir}")
        return

    print("Debugging HDF5 table attributes...")
    print("="*80)

    # Test the proc_noAvg file specifically
    proc_file = hdf5_dir / "250415.proc_noAvg.h5"
    if proc_file.exists():
        print(f"Processing file: {proc_file}")

        with tables.open_file(str(proc_file), mode="r") as h5file:
            print("File opened successfully")

            # Look specifically at the table node
            table_node = h5file.get_node("/i63/table")
            print(f"\nTable node: {table_node._v_pathname}")

            # Check table attributes for index_kind
            print(f"Table attributes: {list(table_node._v_attrs._f_list())}")
            for attr_name in table_node._v_attrs._f_list():
                try:
                    attr_value = getattr(table_node._v_attrs, attr_name)
                    print(f" {attr_name}: {attr_value}")
                except:
                    print(f"  {attr_name}: <could not read>")

            # Check specifically for index_kind
            if hasattr(table_node._v_attrs, 'index_kind'):
                index_kind = table_node._v_attrs.index_kind
                print(f"\nFound index_kind: {index_kind}")

                # Now read the index column and convert based on the index_kind
                if 'datetime64[ns]' in str(index_kind):
                    print("Index is stored as nanosecond timestamps")

                    # Read the index values
                    index_col = table_node.cols.index
                    index_values = index_col[:10]  # Read first 10 values
                    print(f"Sample index values: {index_values}")

                    # Convert nanosecond timestamps to datetime strings
                    for i, val in enumerate(index_values[:3]):  # Just first 3
                        try:
                            # Convert nanosecond timestamp to datetime
                            timestamp_seconds = val / 1e9  # Convert from nanoseconds to seconds
                            dt = datetime.fromtimestamp(timestamp_seconds)
                            print(f"  Value {i}: {val} -> {dt} (timestamp: {timestamp_seconds})")
                        except Exception as e:
                            print(f" Value {i}: {val} -> ERROR: {e}")

            # Also check the pandas info attribute which has timezone info
            if hasattr(table_node._v_attrs, 'info'):
                info = table_node._v_attrs.info
                print(f"\nPandas info: {info}")

    print("\n" + "="*80)
    print("Debugging completed.")

if __name__ == "__main__":
    debug_hdf5_index_attributes()