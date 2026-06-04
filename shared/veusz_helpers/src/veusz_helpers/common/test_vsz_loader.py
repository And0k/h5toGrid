#!/usr/bin/env python3
"""
Test script to verify veusz_load_csv_tcm_raw function works correctly.
"""
import sys
import os
from pathlib import Path
import numpy as np

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import the module
import vsz_loader

def test_veusz_load_csv_tcm_raw():
    """Test the veusz_load_csv_tcm_raw function with mock data"""

    # Mock Veusz functions that are expected
    def mock_ImportFileCSV(*args, **kwargs):
        print(f"ImportFileCSV called with args: {args}, kwargs: {kwargs}")
        return None

    def mock_ImportFileHDF5(*args, **kwargs):
        print(f"ImportFileHDF5 called with args: {args}, kwargs: {kwargs}")
        return None

    def mock_AddCustom(*args, **kwargs):
        print(f"AddCustom called with args: {args}, kwargs: {kwargs}")
        return None

    # Add mock functions to the vsz_loader module
    vsz_loader.ImportFileCSV = mock_ImportFileCSV
    vsz_loader.ImportFileHDF5 = mock_ImportFileHDF5
    vsz_loader.AddCustom = mock_AddCustom

    # Create a simple test CSV file that matches the expected format
    test_file = Path("test_tcm_data.txt")
    test_content = """Inkl P01
Y,M,D,H,M,S,Ax,Ay,Az,Mx,My,Mz,Battery,Temp
2023,4,25,12,24,11,160,-2576,-13808,-111,62,547,5.42,21.63
2023,4,25,12,24,12,161,-2575,-13807,-110,63,548,5.43,21.64
"""

    with open(test_file, 'w') as f:
        f.write(test_content)

    try:
        # Test with different probe types
        probe_info_i = {"type": "i", "model": "i", "number": 3}
        probe_info_p = {"type": "i", "model": "p", "number": 3}

        print("Testing with probe type 'i':")
        result = vsz_loader.veusz_load_csv_tcm_raw(
            file=str(test_file),
            db="test_db.h5",
            time_range=[],
            time_shift_s=0,
            probe_info=probe_info_i
        )
        print(f"Result: {result}")

        print("\nTesting with probe type 'p':")
        result = vsz_loader.veusz_load_csv_tcm_raw(
            file=str(test_file),
            db="test_db.h5",
            time_range=[],
            time_shift_s=0,
            probe_info=probe_info_p
        )
        print(f"Result: {result}")

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

    print("Test completed successfully!")


def test_config_text_header_dtype():
    """Test the _config_text_header_dtype function"""
    print("\nTesting _config_text_header_dtype function:")

    # Test different probe types
    for probe_type in [None, "i", "p", "b", "d", "w"]:
        result = vsz_loader._config_text_header_dtype(probe_type)
        print(f"  Type {probe_type}: {result}")


if __name__ == "__main__":
    print("Testing vsz_loader functions...")

    # Test the configuration function
    test_config_text_header_dtype()

    # Test the main function
    test_veusz_load_csv_tcm_raw()