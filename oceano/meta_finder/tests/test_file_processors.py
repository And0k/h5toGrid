import pytest
from pathlib import Path, PurePosixPath
from meta_finder.data_proc_funcs import extract_time_info_from_text_file

def test_extract_time_info_from_text_file(common_test_data_setup):
    """Test extracting time range from text file with proper format."""
    # Use the common test data setup
    test_dir = common_test_data_setup

    # Use an existing test directory with files that have similar patterns
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6" / "text_output"
    rel_path = PurePosixPath("230508_1551bin2s@i03.tsv")

    result = extract_time_info_from_text_file(device_dir, rel_path)
    if result:
        start_time, end_time, bursts_t, burst_dt = result
        assert start_time == "2019-11-08 12:00:00"
        assert end_time == "2019-11-08 12:00:01"
    else:
        assert False, "extract_time_info_from_text_file returned None"
