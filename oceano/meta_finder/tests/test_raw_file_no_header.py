"""Test for raw files with tab-separated format and no headers."""
import pytest
from pathlib import Path
from meta_finder.data_proc_funcs import extract_time_info_from_text_file


@pytest.mark.parametrize(
    "file_content,expected_start,expected_end,description",
    [
        (
            "2013\t05\t10\t10\t00\t00\t00129\t24681\t16320\t32448\t35392\t32895\t32701\t28672\n"
            "2013\t05\t10\t10\t00\t00\t00212\t24640\t16448\t32448\t35392\t32867\t32715\t28672\n",
            "2013-05-10 10:00:00",
            "2013-05-10 10:00:00",
            "raw file with tab-separated format and no headers"
        ),
    ],
    ids=["raw_tab_no_header"]
)
def test_raw_file_no_header(tmp_path, file_content, expected_start, expected_end, description):
    """Test that raw files with tab-separated format and no headers are parsed correctly."""
    # Create a temporary directory structure with _raw subdirectory
    raw_dir = tmp_path / "_raw"
    raw_dir.mkdir()

    # Create test file in _raw directory
    test_file = raw_dir / "130510.txt"
    test_file.write_text(file_content)

    # Extract time info
    result = extract_time_info_from_text_file(tmp_path, Path("_raw/130510.txt"))

    # Verify result
    assert result is not None, f"{description}: Should extract time info from raw file"
    start_time, end_time, burst_dt, bursts_t = result
    assert start_time == expected_start, f"{description}: Expected start time {expected_start}, got {start_time}"
    assert end_time == expected_end, f"{description}: Expected end time {expected_end}, got {end_time}"
