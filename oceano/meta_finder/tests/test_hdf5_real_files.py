#!/usr/bin/env python
"""
Test script to test HDF5 fallback functionality with real h5 files from BalticSea dataset.
"""
import sys
from pathlib import Path
import logging

# Add the project source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_finder.hdf5_processor import (
    extract_metadata_from_hdf5,
    extract_devices_from_hdf5_groups,
    find_hdf5_files,
    extract_time_range_from_hdf5_table,
    extract_time_ranges_from_hdf5_combined
)

def test_real_hdf5_files():
    """Test HDF5 fallback functionality with real files from the BalticSea dataset."""

    # Using centralized logging configuration
    from meta_finder.logging_config import setup_logging
    logger = setup_logging(__name__, include_function_name=True, log_file_sfx="test_hdf5_real_files")

    # Define the real HDF5 directory path
    hdf5_dir = Path("D:/WorkData/BalticSea/250415_ABP60@i,t-chain/inclinometer")

    if not hdf5_dir.exists():
        print(f"Directory does not exist: {hdf5_dir}")
        print("Please ensure the directory exists and is accessible.")
        return

    print(f"Testing HDF5 fallback functionality with files from: {hdf5_dir}")
    print("="*80)

    # 1. Test finding HDF5 files
    print("1. Finding HDF5 files in directory...")
    h5_files = find_hdf5_files(hdf5_dir)
    print(f"Found HDF5 files: {h5_files}")
    print()

    # 2. Test extracting devices from HDF5 groups
    print("2. Testing device extraction from HDF5 groups...")
    for h5_type, files in h5_files.items():
        for h5_file in files:
            print(f"  Testing file: {h5_file.name}")
            devices = extract_devices_from_hdf5_groups(h5_file)
            print(f"    Extracted devices: {devices}")
    print()

    # 3. Test extracting time ranges from HDF5 files (with no specific devices)
    print("3. Testing time range extraction (all devices)...")
    time_ranges_all = extract_metadata_from_hdf5(hdf5_dir, dev_ids=None)
    print(f"Time ranges for all devices: {time_ranges_all}")
    print()

    # 4. Test extracting time ranges with specific device list
    print("4. Testing time range extraction (specific devices)...")
    # Let's try to find some device IDs from the directory structure first
    all_devices = set()
    for h5_type, files in h5_files.items():
        for h5_file in files:
            devices = extract_devices_from_hdf5_groups(h5_file)
            all_devices.update(devices)

    print(f"All devices found in HDF5 files: {list(all_devices)}")

    if all_devices:
        time_ranges_specific = extract_metadata_from_hdf5(hdf5_dir, dev_ids=list(all_devices))
        print(f"Time ranges for specific devices: {time_ranges_specific}")
    else:
        print("No devices found in HDF5 files")
    print()

    # 5. Test individual file processing to see structure
    print("5. Testing individual file processing...")
    for h5_type, files in h5_files.items():
        for h5_file in files:
            print(f"  Processing file: {h5_file.name} (type: {h5_type})")
            try:
                import tables
                with tables.open_file(str(h5_file), mode="r") as h5file:
                    print(f"    File structure:")
                    for node in h5file.walk_nodes(where="/", classname="Group"):
                        print(f"      Group: {node._v_pathname}")
                        # List tables in each group
                        for table_node in h5file.walk_nodes(where=node._v_pathname, classname="Table"):
                            print(f"        Table: {table_node._v_pathname}")
                            # Show column names if it's a table
                            if hasattr(table_node, 'cols') and hasattr(table_node.cols, '_v_colnames'):
                                print(f"          Columns: {table_node.cols._v_colnames}")
            except Exception as e:
                print(f"    Error reading file {h5_file}: {e}")
    print()

    # 6. Performance and functionality summary
    print("6. HDF5 Fallback Functionality Summary:")
    print(f"   - Priority order followed: proc_noAvg -> proc -> raw")
    print(f"   - Found {sum(len(files) for files in h5_files.values())} HDF5 files")
    print(f"   - Total device IDs found: {len(all_devices)}")
    print(f"   - Time ranges extracted: {len(time_ranges_all) if time_ranges_all else 0}")

    if time_ranges_all:
        for dev_id, time_range in time_ranges_all.items():
            if time_range:
                start_time, end_time, bursts_t, burst_dt = time_range
                print(f"     Device {dev_id}: {start_time} to {end_time}")

    print("="*80)
    print("HDF5 fallback test completed.")
