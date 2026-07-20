import pytest
from meta_finder.parse_data_file_name import parse_filename_for_metadata

def test_parse_filename_returns_star_for_combined_files():
    """Test that combined files like '191108_1200bin600s.tsv' return devices = ['*']"""
    metadata = parse_filename_for_metadata("191108_1200bin600s.tsv")
    assert metadata.get('devices') == ["*"]

def test_parse_filename_returns_correct_devices_for_single_device():
    """Test that single device files still work correctly"""
    metadata = parse_filename_for_metadata("191108_1200bin600s@i07.tsv")
    assert metadata.get('device_id') == 'i7'
    assert metadata.get('devices') == ['i7']