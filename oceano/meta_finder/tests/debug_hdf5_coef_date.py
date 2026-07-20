#!/usr/bin/env python
"""
Debug script to test HDF5 coef date extraction functionality with detailed logging.
"""
import tempfile
from pathlib import Path
import tables
import sys
import logging

# Add the project source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up logging to see debug messages
from meta_finder.logging_config import setup_logging
logger = setup_logging(__name__, include_function_name=True, log_file_sfx="debug_hdf5_coef_date",
    console_level=logging.DEBUG, file_level=logging.DEBUG)

def test_date_extraction():
    """Test date extraction with detailed logging."""
    from meta_finder.hdf5_processor import extract_coef_date_from_hdf5

    print("Creating test HDF5 file with date-only format...")

    # Create a simple test
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        h5file = tables.open_file(str(tmp_path), mode='w')
        try:
            print("Creating HDF5 structure...")
            incl03_group = h5file.create_group('/', 'incl03', 'Inclinometer 03 group')
            coef_group = h5file.create_group(incl03_group, 'coef', 'Coefficients group')
            print("Adding date array...")
            h5file.create_array(coef_group, 'date', ['2023-09-06'.encode('utf-8')], 'Date string')
        finally:
            h5file.close()

        print("Calling extract_coef_date_from_hdf5...")
        try:
            result = extract_coef_date_from_hdf5(tmp_path, 'incl03')
            print(f"Result: {result}")
        except Exception as e:
            print(f'Exception occurred: {e}')
            import traceback
            traceback.print_exc()

        tmp_path.unlink()

if __name__ == "__main__":
    test_date_extraction()