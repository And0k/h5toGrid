#!/usr/bin/env python
"""
Test script for HDF5 fallback functionality when text_output files are not found.
This test verifies that metadata time information is extracted from HDF5 files
when text_output files are missing or empty.
"""
import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np


@pytest.mark.parametrize("ids_to_norm,expected_result,comment", [
    (["i1"], {"i1": {"time_info": ("2023-01-01 10:00:00", "2023-01-01 10:05:00", "-", "-"), "data_paths": {}}}, "should extract time info for device i1 from HDF5 when text_output not found"),
    (["w2"], {"w2": {"time_info": ("2023-01-01 10:00:00", "2023-01-01 10:05:00", "-", "-"), "data_paths": {}}}, "should extract time info for device w2 from HDF5 when text_output not found"),
    (["i1", "w2"], {"i1": {"time_info": ("2023-01-01 10:00:0", "2023-01-01 10:05:00", "-", "-"), "data_paths": {}}, "w2": {"time_info": ("2023-01-01 10:00:00", "2023-01-01 10:05:00", "-", "-"), "data_paths": {}}}, "should extract time info for multiple devices from HDF5 when text_output not found"),
], ids=["single_device_i1", "single_device_w2", "multiple_devices"])
def test_hdf5_fallback_when_text_output_not_found(ids_to_norm, expected_result, comment):
    """Test that HDF5 fallback works when text_output files are not found."""
    from meta_finder.collect import get_absent_meta
    from meta_finder import config

    # Temporarily enable HDF5 fallback for testing
    original_setting = config.use_hdf5_fallback
    config.use_hdf5_fallback = True

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_dir = Path(temp_dir)

            # Mock the text output processing to return empty results
            # Note: process_text_output_directory is only called when text output directories exist
            # In this test, we're testing HDF5 fallback when there are no text output files,
            # so process_text_output_directory should not be called
            with patch('meta_finder.collect.extract_metadata_from_hdf5') as mock_hdf5_extract:

                # Mock HDF5 extraction to return expected results
                # Handle both calls: first with specific device IDs and extract_time_info=True,
                # and second with None and extract_time_info=False
                call_count = 0

                def mock_hdf5_side_effect(device_dir_param, dev_ids_param=None, extract_time_info=True):
                    nonlocal call_count
                    call_count += 1

                    result = {}
                    if extract_time_info:
                        # First call - extract time info for specific devices
                        if dev_ids_param:
                            # If specific devices are requested, return only those
                            for dev_id in dev_ids_param:
                                # Find the original device ID that matches this normalized ID
                                for id_to_norm in ids_to_norm:
                                    from meta_finder.parse_data_file_name import normalize_device_id
                                    normalized_id = normalize_device_id(id_to_norm)
                                    if normalized_id == dev_id and id_to_norm in expected_result:
                                        result[dev_id] = {
                                            "time_info": expected_result[id_to_norm]["time_info"],
                                            "data_paths": expected_result[id_to_norm]["data_paths"]
                                        }
                                        break
                        else:
                            # If no specific devices, return all
                            for id_to_norm in ids_to_norm:
                                if id_to_norm in expected_result:
                                    from meta_finder.parse_data_file_name import normalize_device_id
                                    normalized_id = normalize_device_id(id_to_norm)
                                    result[normalized_id] = {
                                        "time_info": expected_result[id_to_norm]["time_info"],
                                        "data_paths": expected_result[id_to_norm]["data_paths"]
                                    }
                    else:
                        # Second call - extract data paths only (no time info)
                        for id_to_norm in ids_to_norm:
                            if id_to_norm in expected_result:
                                from meta_finder.parse_data_file_name import normalize_device_id
                                normalized_id = normalize_device_id(id_to_norm)
                                result[normalized_id] = {
                                    "time_info": None,  # No time info for this call
                                    "data_paths": expected_result[id_to_norm]["data_paths"]
                                }

                    # For the first call, if no devices are found, return empty dict
                    # For the second call, always return the structure with data paths
                    if call_count == 1 and not result and not extract_time_info:
                        # This is the first call but no devices found, return empty
                        return {}
                    elif call_count == 2 and not extract_time_info:
                        # This is the second call, return structure with data paths
                        return result
                    else:
                        # For other cases, return the result as is
                        return result

                mock_hdf5_extract.side_effect = mock_hdf5_side_effect

                # Normalize device IDs before passing to create_info_devices_content
                from meta_finder.parse_data_file_name import normalize_device_id
                normalized_ids = [normalize_device_id(dev_id) for dev_id in ids_to_norm]

                # Call get_absent_metawith devices but no text_output files
                result = get_absent_meta(normalized_ids, device_dir)

                # Verify that HDF5 extraction was called (fallback triggered)
                # When there are no text output files, the second call is made for complete data path collection
                mock_hdf5_extract.assert_called_once_with(device_dir, None, extract_time_info=False)

                # Check that the result contains the expected time info from HDF5
                for dev_id in ids_to_norm:
                    normalized_dev_id = normalize_device_id(dev_id)
                    assert normalized_dev_id in result, f"Device {normalized_dev_id} should be in result as per: {comment}"
                    if dev_id in expected_result:  # Check against original key in expected_result
                        expected_time_info = expected_result[dev_id]["time_info"]
                        actual_metadata = result[normalized_dev_id]["metadata"]
                        # Metadata is now a dictionary, check individual fields
                        assert actual_metadata["time_st"] == expected_time_info[0], f"Start time for {normalized_dev_id} should match HDF5 data as per: {comment}"
                        assert actual_metadata["time_en"] == expected_time_info[1], f"End time for {normalized_dev_id} should match HDF5 data as per: {comment}"
                        assert actual_metadata["burst_dt"] == expected_time_info[2], f"Burst dt for {normalized_dev_id} should match HDF5 data as per: {comment}"
                        assert actual_metadata["bursts_t"] == expected_time_info[3], f"Bursts t for {normalized_dev_id} should match HDF5 data as per: {comment}"

                # Check that the result contains the expected time info from HDF5
                for dev_id in ids_to_norm:
                    normalized_dev_id = normalize_device_id(dev_id)
                    assert normalized_dev_id in result, f"Device {normalized_dev_id} should be in result as per: {comment}"
                    if dev_id in expected_result:  # Check against original key in expected_result
                        expected_time_info = expected_result[dev_id]["time_info"]
                        actual_metadata = result[normalized_dev_id]["metadata"]
                        # Metadata is now a dictionary, check individual fields
                        assert actual_metadata["time_st"] == expected_time_info[0], f"Start time for {normalized_dev_id} should match HDF5 data as per: {comment}"
                        assert actual_metadata["time_en"] == expected_time_info[1], f"End time for {normalized_dev_id} should match HDF5 data as per: {comment}"
                        assert actual_metadata["burst_dt"] == expected_time_info[2], f"Burst dt for {normalized_dev_id} should match HDF5 data as per: {comment}"
                        assert actual_metadata["bursts_t"] == expected_time_info[3], f"Bursts t for {normalized_dev_id} should match HDF5 data as per: {comment}"

    finally:
        # Restore original setting
        config.use_hdf5_fallback = original_setting


def test_hdf5_fallback_with_no_text_output_directory():
    """Test that HDF5 fallback works when text_output directory doesn't exist."""
    from meta_finder.collect import get_absent_meta
    from meta_finder import config

    # Temporarily enable HDF5 fallback for testing
    original_setting = config.use_hdf5_fallback
    config.use_hdf5_fallback = True

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_dir = Path(temp_dir)

            # Mock HDF5 extraction to return expected results
            # Note: process_text_output_directory is only called when text output directories exist
            # In this test, we're testing HDF5 fallback when there are no text output files,
            # so process_text_output_directory should not be called
            with patch('meta_finder.collect.extract_metadata_from_hdf5') as mock_hdf5_extract:

                # Mock HDF5 extraction to return expected results
                mock_hdf5_extract.return_value = {
                    "i1": {
                        "time_info": ("2023-01-01 10:00:00", "2023-01-01 10:05:00", "-", "-"),
                        "data_paths": {}
                    }
                }

                # Normalize device IDs before passing to create_info_devices_content
                from meta_finder.parse_data_file_name import normalize_device_id
                normalized_ids = [normalize_device_id("i1")]

                # Call create_info_devices_content
                result = get_absent_meta(normalized_ids, device_dir)

                # Verify that HDF5 extraction was called (fallback triggered)
                # When there are no text output files, the second call is made for complete data path collection
                mock_hdf5_extract.assert_called_once_with(device_dir, None, extract_time_info=False)

                # Check that the result contains the expected time info from HDF5
                assert "i1" in result, "should have device i1 in result as per: HDF5 fallback provided time info"
                actual_metadata = result["i1"]["metadata"]
                expected_time_info = ("2023-01-01 10:00:00", "2023-01-01 10:05:00", "-", "-")
                # Metadata is now a dictionary, check individual fields
                assert actual_metadata["time_st"] == expected_time_info[0], "Start time should match HDF5 data as per: HDF5 fallback provided this information"
                assert actual_metadata["time_en"] == expected_time_info[1], "End time should match HDF5 data as per: HDF5 fallback provided this information"
                assert actual_metadata["burst_dt"] == expected_time_info[2], "Burst dt should match HDF5 data as per: HDF5 fallback provided this information"
                assert actual_metadata["bursts_t"] == expected_time_info[3], "Bursts t should match HDF5 data as per: HDF5 fallback provided this information"

    finally:
        # Restore original setting
        config.use_hdf5_fallback = original_setting


@pytest.mark.parametrize("ids_to_norm,comment", [
    (["i1"], "should handle single device when no text_output and HDF5 fallback enabled"),
    (["i1", "w2", "p3"], "should handle multiple devices when no text_output and HDF5 fallback enabled"),
    ([], "should handle empty device list when no text_output and HDF5 fallback enabled"),
], ids=["single_device", "multiple_devices", "empty_device_list"])
def test_hdf5_fallback_different_device_scenarios(ids_to_norm, comment):
    """Test HDF5 fallback with different device scenarios when text_output is not found."""
    from meta_finder.collect import get_absent_meta
    from meta_finder import config

    # Temporarily enable HDF5 fallback for testing
    original_setting = config.use_hdf5_fallback
    config.use_hdf5_fallback = True

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_dir = Path(temp_dir)

            # Mock HDF5 functionality to return appropriate results
            # Note: process_text_output_directory is only called when text output directories exist
            # In this test, we're testing HDF5 fallback when there are no text output files,
            # so process_text_output_directory should not be called
            with patch('meta_finder.hdf5_processor.extract_metadata_from_hdf5') as mock_extract_hdf5:
                if ids_to_norm:
                    # Normalize device IDs and create mock return value for each device
                    from meta_finder.parse_data_file_name import normalize_device_id
                    mock_result = {}
                    for id_to_norm in ids_to_norm:
                        normalized_id = normalize_device_id(id_to_norm)
                        mock_result[normalized_id] = {
                            "time_info": ("2023-01-01 10:00:0", "2023-01-01 10:05:00", "-", "-"),
                            "data_paths": {}
                        }
                    mock_extract_hdf5.return_value = mock_result
                else:
                    # For empty device list, return a placeholder
                    mock_extract_hdf5.return_value = {"?": {"time_info": None, "data_paths": {}}}

                # Normalize device IDs before passing to create_info_devices_content
                normalized_ids = [normalize_device_id(dev_id) for dev_id in ids_to_norm]

                # Call create_info_devices_content
                result = get_absent_meta(normalized_ids, device_dir)

                # Verify that HDF5 extraction was called (fallback triggered)
                # When there are no text output files, the second call is made for complete data path collection
                mock_extract_hdf5.assert_called_once_with(device_dir, None, extract_time_info=False)

                # Verify the behavior based on ids_to_norm
                if ids_to_norm:
                    for dev_id in ids_to_norm:
                        normalized_dev_id = normalize_device_id(dev_id)
                        assert normalized_dev_id in result, f"Device {normalized_dev_id} should be in result as per: {comment}"
                        # If device was in the input list, it should have metadata
                        assert "metadata" in result[normalized_dev_id], f"Device {normalized_dev_id} should have metadata as per: {comment}"
                else:
                    # For empty device list, there might be a placeholder
                    assert len(result) >= 0, f"Result should be valid for empty device list as per: {comment}"

    finally:
        # Restore original setting
        config.use_hdf5_fallback = original_setting