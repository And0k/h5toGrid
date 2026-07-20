import pytest
from pathlib import Path, PurePosixPath
from meta_finder.data_proc_funcs import read_file_lines_universal

def test_read_file_lines_universal_with_split_files(common_test_data_setup):
    """Test reading lines from split files - first lines from first file, last line from last file."""
    # Use the common test data setup
    test_dir = common_test_data_setup

    # Use an existing test directory with files that have similar patterns
    # Find a directory with multiple TSV files that can be used for this test
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6" / "text_output"

    # Test without max_lines (read all lines)
    lines, last_line = read_file_lines_universal(device_dir, PurePosixPath("230508_1551bin2s@i03.tsv"))

    # Should read all lines from first file
    assert len(lines) == 3  # Header + 2 data lines (based on the content in common_test_data_setup)
    assert lines[0] == "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n"
    assert lines[1] == "2019-11-08 12:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n"
    assert lines[2] == "2019-11-08 12:00:01\t0.15\t185\t0.06\t0.09\t5.3\t20.2\n"

    # Last line should be from the last file
    assert last_line == "2019-11-08 12:00:01\t0.15\t185\t0.06\t0.09\t5.3\t20.2\n"

def test_read_file_lines_universal_with_split_files_and_max_lines(common_test_data_setup):
    """Test reading lines from split files with max_lines limit."""
    # Use the common test data setup
    test_dir = common_test_data_setup

    # Use an existing test directory with files that have similar patterns
    device_dir = test_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6" / "text_output"

    # Test with max_lines=1 (read only first data line)
    lines, last_line = read_file_lines_universal(device_dir, PurePosixPath("230508_1551bin2s@i03.tsv"), max_lines=1)

    # Should read only first line from first file (header + 0 data lines)
    assert len(lines) == 1
    assert lines[0] == "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n"

    # Last line should still be from the last file
    assert last_line == "2019-11-08 12:00:01\t0.15\t185\t0.06\t0.09\t5.3\t20.2\n"