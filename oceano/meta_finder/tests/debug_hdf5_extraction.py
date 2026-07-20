#!/usr/bin/env python
"""
Debug script to understand exactly how HDF5 time range extraction works with real files.
"""
import sys
from pathlib import Path
import logging
import numpy as np

# Add the project source directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import tables
from meta_finder.hdf5_processor import extract_time_range_from_hdf5_table

def debug_hdf5_time_extraction():
    """Debug the HDF5 time range extraction process step by step."""

    # Using centralized logging configuration
    from meta_finder.logging_config import setup_logging
    logger = setup_logging(__name__, include_function_name=True, log_file_sfx="debug_hdf5_extraction")

    # Define the real HDF5 directory path
    hdf5_dir = Path("D:/WorkData/BalticSea/250415_ABP60@i,t-chain/inclinometer")

    if not hdf5_dir.exists():
        print(f"Directory does not exist: {hdf5_dir}")
        return

    print("Debugging HDF5 time range extraction...")
    print("="*80)

    # Test the proc_noAvg file specifically
    proc_file = hdf5_dir / "250415.proc_noAvg.h5"
    if proc_file.exists():
        print(f"Processing file: {proc_file}")

        with tables.open_file(str(proc_file), mode="r") as h5file:
            print("File opened successfully")

            # List all nodes in the file
            print("\nAll nodes in the file:")
            for node in h5file.walk_nodes():
                print(f"  Node: {node._v_pathname} (type: {type(node).__name__})")

                # If it's a table, show more details
                if hasattr(node, 'cols') and hasattr(node.cols, '_v_colnames'):
                    print(f"    Columns: {node.cols._v_colnames}")

                    # Try to read the index column to see its content
                    if 'index' in node.cols._v_colnames:
                        try:
                            index_col = getattr(node.cols, 'index')
                            print(f"    Index column type: {type(index_col)}")
                            print(f"    Index column dtype: {index_col.dtype}")

                            # Read first few values
                            index_values = index_col[:10]  # Read first 10 values
                            print(f"    Index values (first 10): {index_values}")
                            print(f"    Index values type: {type(index_values)}")
                            if len(index_values) > 0:
                                print(f"    First value type: {type(index_values[0])}")
                                print(f"    First value: {index_values[0]}")

                                # Try to convert to string format as the function does
                                if hasattr(index_values[0], 'astype'):
                                    start_time = str(index_values[0].astype('datetime64[s]'))
                                    print(f"    Converted first value: {start_time}")

                                    if len(index_values) > 1:
                                        end_time = str(index_values[-1].astype('datetime64[s]'))
                                        print(f"    Converted last value: {end_time}")

                        except Exception as e:
                            print(f"    Error reading index column: {e}")

                    # Try to call the actual extraction function
                    print(f"\n    Calling extract_time_range_from_hdf5_table for path: {node._v_pathname}")
                    result = extract_time_range_from_hdf5_table(proc_file, node._v_pathname)
                    print(f"    Result: {result}")

    print("\n" + "="*80)
    print("Debugging completed.")

if __name__ == "__main__":
    debug_hdf5_time_extraction()