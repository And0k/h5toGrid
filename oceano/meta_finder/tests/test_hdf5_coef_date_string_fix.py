#!/usr/bin/env python
"""
Test script for HDF5 coef date string extraction functionality.
Tests specifically the case where dates are stored as strings in HDF5 files.
"""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from datetime import datetime
import tables

# Add the project source directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_hdf5_string_date_extraction():
    """Test extracting string dates from HDF5 files like in B:\\WorkData\\BalticSea\\_Pregolya,Lagoon\\230906@i3,38,54\\_raw\\230906.raw.h5"""

    from meta_finder.hdf5_processor import extract_coef_date_from_hdf5

    # Create a temporary HDF5 file with string date in /incl03/coef/date
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        # Create the HDF5 file structure
        h5file = tables.open_file(str(tmp_path), mode="w")
        try:
            # Create the incl03 group
            incl03_group = h5file.create_group("/", "incl03", "Inclinometer 03 group")

            # Create the coef group within incl03
            coef_group = h5file.create_group(incl03_group, "coef", "Coefficients group")

            # Create a dataset in /incl03/coef/date with string date
            date_string = "2023-09-06 12:00:00"  # ISO format string date
            h5file.create_array(coef_group, "date", [date_string.encode('utf-8')], "Date string")
        finally:
            h5file.close()

        # Now test the extraction
        result = extract_coef_date_from_hdf5(tmp_path, "incl03")

        # Should return the expected date
        assert result == "2023-09-06 12:00:00", f"Expected '2023-09-06 12:00:00', got {result}"

        # Clean up - use try-except to handle Windows file permission issues
        try:
            tmp_path.unlink()
        except PermissionError:
            # File might still be open on Windows, skip deletion
            import warnings
            warnings.warn(f"Could not delete temporary file {tmp_path}, may still be in use")


def test_hdf5_multiple_string_dates():
    """Test extracting the maximum date from multiple string dates in HDF5 files."""

    from meta_finder.hdf5_processor import extract_coef_date_from_hdf5

    # Create a temporary HDF5 file with multiple string dates in /incl03/coef/date
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        # Create the HDF5 file structure
        h5file = tables.open_file(str(tmp_path), mode="w")
        try:
            # Create the incl03 group
            incl03_group = h5file.create_group("/", "incl03", "Inclinometer 03 group")

            # Create the coef group within incl03
            coef_group = h5file.create_group(incl03_group, "coef", "Coefficients group")

            # Create a dataset in /incl03/coef/date with multiple string dates
            date_strings = [
                "2023-09-05 10:00:00".encode('utf-8'),
                "2023-09-07 14:30:00".encode('utf-8'),  # This should be the max
                "2023-09-06 12:00:00".encode('utf-8')
            ]
            h5file.create_array(coef_group, "date", date_strings, "Multiple date strings")
        finally:
            h5file.close()

        # Now test the extraction - should return the maximum date
        result = extract_coef_date_from_hdf5(tmp_path, "incl03")

        # Should return the maximum date from the array
        assert result == "2023-09-07 14:30:00", f"Expected '2023-09-07 14:30:00', got {result}"

        # Clean up - use try-except to handle Windows file permission issues
        try:
            tmp_path.unlink()
        except PermissionError:
            # File might still be open on Windows, skip deletion
            import warnings
            warnings.warn(f"Could not delete temporary file {tmp_path}, may still be in use")


def test_hdf5_string_date_with_nan():
    """Test extracting string dates when the array contains nan values."""

    from meta_finder.hdf5_processor import extract_coef_date_from_hdf5

    # Create a temporary HDF5 file with string date and nan in /incl03/coef/date
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        # Create the HDF5 file structure
        h5file = tables.open_file(str(tmp_path), mode="w")
        try:
            # Create the incl03 group
            incl03_group = h5file.create_group("/", "incl03", "Inclinometer 03 group")

            # Create the coef group within incl03
            coef_group = h5file.create_group(incl03_group, "coef", "Coefficients group")

            # Create a dataset in /incl03/coef/date with string dates and nan values
            # We'll use numpy array with bytes and nan-like values
            date_strings = np.array([
                b'2023-09-05 10:0:00',
                b'nan',  # This should be filtered out
                b'2023-09-07 14:30:00'  # This should be the max
            ], dtype='S20')  # Fixed length string array

            h5file.create_array(coef_group, "date", date_strings, "Date strings with nan")
        finally:
            h5file.close()

        # Now test the extraction
        result = extract_coef_date_from_hdf5(tmp_path, "incl03")

        # Should return the maximum valid date, ignoring 'nan'
        assert result == "2023-09-07 14:30:00", f"Expected '2023-09-07 14:30:00', got {result}"

        # Clean up - use try-except to handle Windows file permission issues
        try:
            tmp_path.unlink()
        except PermissionError:
            # File might still be open on Windows, skip deletion
            import warnings
            warnings.warn(f"Could not delete temporary file {tmp_path}, may still be in use")


if __name__ == "__main__":
    test_hdf5_string_date_extraction()
    test_hdf5_multiple_string_dates()
    test_hdf5_string_date_with_nan()
    print("All tests passed!")