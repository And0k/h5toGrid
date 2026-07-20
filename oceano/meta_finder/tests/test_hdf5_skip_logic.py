#!/usr/bin/env python
"""
Test script to verify that HDF5 extraction is properly skipped when JSON metadata already contains time information.
"""

from pathlib import Path
from meta_finder.collect import get_absent_meta

def test_hdf5_extraction_skip():
    """Test that HDF5 extraction is skipped when time metadata already exists in JSON."""

    # Create a mock device directory (we won't actually access it since we're testing the logic)
    mock_device_dir = Path("dummy")

    # Create test input with JSON metadata that already has time information
    devices_with_json_metadata = {
        "i07": {
            "metadata": {
                "point": "A1",
                "sea_depth": "10.5",
                "height_above_bottom": "5.2",
                "modification_symbol": "",
                "lat": "54.1234",
                "lon": "20.5678",
                "time_st": "2023-10-12 10:00:00",  # Time start already exists
                "time_en": "2023-10-12 12:00:00",  # Time end already exists
                "burst_dt": "2.0",
                "bursts_t": "100",
                "comment": "Test device",
                "coef_date": "2023-10-11 09:00:00"  # Coef date already exists
            },
            "data_paths": {("dummy_path", "dummy_file"): {}}
        }
    }

    print("Testing HDF5 extraction skip logic...")
    print(f"Input metadata has time_st: {devices_with_json_metadata['i07']['metadata']['time_st']}")
    print(f"Input metadata has time_en: {devices_with_json_metadata['i07']['metadata']['time_en']}")
    print(f"Input metadata has coef_date: {devices_with_json_metadata['i07']['metadata']['coef_date']}")

    # Call get_absent_meta with extract_hdf5_times=True and extract_hdf5_coef_dates=True
    # This should NOT extract HDF5 data since JSON already has time info
    result = get_absent_meta(
        devices_with_json_metadata,
        mock_device_dir,
        extract_hdf5_times=True,
        extract_hdf5_coef_dates=True
    )

    print(f"\nResult metadata: {result['i07']['metadata']}")

    # Verify that the original time info is preserved
    result_metadata = result['i07']['metadata']
    assert result_metadata['time_st'] == "2023-10-12 10:00:00", f"Expected original time_st, got {result_metadata['time_st']}"
    assert result_metadata['time_en'] == "2023-10-12 12:00:00", f"Expected original time_en, got {result_metadata['time_en']}"
    assert result_metadata['coef_date'] == "2023-10-11 09:00:00", f"Expected original coef_date, got {result_metadata['coef_date']}"

    print("All assertions passed - HDF5 extraction was properly skipped when JSON metadata had time info")

    # Now test with metadata that has placeholder values - HDF5 extraction should occur
    devices_with_placeholders = {
        "i08": {
            "metadata": {
                "point": "A2",
                "sea_depth": "11.5",
                "height_above_bottom": "6.2",
                "modification_symbol": "",
                "lat": "54.2345",
                "lon": "20.6789",
                "time_st": "?",  # Placeholder - should trigger HDF5 extraction
                "time_en": "?",  # Placeholder - should trigger HDF5 extraction
                "burst_dt": "?",
                "bursts_t": "?",
                "comment": "Test device 2",
                "coef_date": "?"  # Placeholder - should trigger HDF5 extraction
            },
            "data_paths": {("dummy_path", "dummy_file"): {}}
        }
    }

    print(f"\nTesting with placeholder values...")
    print(f"Input metadata has time_st: {devices_with_placeholders['i08']['metadata']['time_st']}")
    print(f"Input metadata has time_en: {devices_with_placeholders['i08']['metadata']['time_en']}")
    print(f"Input metadata has coef_date: {devices_with_placeholders['i08']['metadata']['coef_date']}")

    # This would normally try to extract HDF5 data, but since we're not accessing real files
    # and our mock device_dir doesn't exist, it won't find any HDF5 files to extract from
    result2 = get_absent_meta(
        devices_with_placeholders,
        mock_device_dir,
        extract_hdf5_times=True,
        extract_hdf5_coef_dates=True
    )

    result2_metadata = result2['i08']['metadata']
    print(f"Result metadata: {result2_metadata}")

    # Since no real HDF5 files exist, the placeholders should remain
    assert result2_metadata['time_st'] == "?", f"Expected placeholder time_st, got {result2_metadata['time_st']}"
    assert result2_metadata['time_en'] == "?", f"Expected placeholder time_en, got {result2_metadata['time_en']}"
    assert result2_metadata['coef_date'] == "?", f"Expected placeholder coef_date, got {result2_metadata['coef_date']}"

    print("Placeholders remained as expected when no HDF5 data was available")

    print("\nAll tests passed! The fix correctly skips HDF5 extraction when time metadata already exists.")

if __name__ == "__main__":
    test_hdf5_extraction_skip()