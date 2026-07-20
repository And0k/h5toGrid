#!/usr/bin/env python
"""
Test script for HDF5 raw metadata integration functionality.
"""
import pytest
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch, MagicMock

# Add the project source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_collect_with_raw_hdf5_cols():
    """Test metadata associator integration with raw HDF5 columns."""
    from meta_finder.collect import get_absent_meta
    from meta_finder import config

    # Temporarily set raw_hdf5_cols for testing
    original_raw_cols = getattr(config, 'raw_hdf5_cols', None)
    config.raw_hdf5_cols = {"coef_date", "raw_date_range"}

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock the HDF5 processor functions
            with patch('meta_finder.collect.extract_raw_hdf5_metadata') as mock_extract_raw:
                # Mock return value for raw HDF5 metadata
                mock_extract_raw.return_value = {
                    'i07': {
                        'coef_date': '2019-12-01 23:17:20',
                        'time_raw_st': '2019-12-01 23:17:20',
                        'time_raw_en': '2019-12-01 23:18:20'
                    }
                }

                # Mock the text output processing to return empty results
                with patch('meta_finder.data_proc_funcs.process_text_output_directory') as mock_text_extract:
                    mock_text_extract.return_value = None  # process_text_output_directory doesn't return anything

                    # Mock HDF5 fallback to return empty results (so raw metadata is used)
                    with patch('meta_finder.collect.extract_metadata_from_hdf5') as mock_hdf5_extract:
                        mock_hdf5_extract.return_value = {}

                        # Call get_absent_metawith devices but no text_output files
                        result = get_absent_meta(['i07'], temp_path)

                        # Check that the result contains the raw HDF5 metadata
                        assert 'i07' in result, "should have device i07 in result as per: raw HDF5 metadata provided"
                        assert 'metadata' in result['i07'], "should have metadata for i07 as per: raw HDF5 metadata provided"

                        # Check that the metadata contains the raw HDF5 values
                        metadata = result['i07']['metadata']
                        # Metadata is now a dictionary, check specific keys
                        assert metadata['coef_date'] == '2019-12-01 23:17:20', "should have coef_date in metadata as per: raw_hdf5_cols configuration"
                        assert metadata['time_raw_st'] == '2019-12-01 23:17:20', "should have time_raw_st in metadata as per: raw_hdf5_cols configuration"
                        assert metadata['time_raw_en'] == '2019-12-01 23:18:20', "should have time_raw_en in metadata as per: raw_hdf5_cols configuration"
    finally:
        # Restore original setting
        if original_raw_cols is not None:
            config.raw_hdf5_cols = original_raw_cols
        elif hasattr(config, 'raw_hdf5_cols'):
            delattr(config, 'raw_hdf5_cols')
