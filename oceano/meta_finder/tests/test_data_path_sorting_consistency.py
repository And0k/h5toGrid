"""
Test for consistent data path sorting using the same rules as sort_text_outputs function.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock
from meta_finder.data_processor import sort_data_paths
from meta_finder import config

@pytest.mark.parametrize(
    "text_outputs, json_devices, expected_order_desc",
    [
        # Test with different source types priority: directories > archives > proc_noAvg HDF5 > proc HDF5 > raw HDF5 > raw text files
        (
            {
                (Path("dir"), Path("file.tsv")): {"devices": ["i1"], "averaging_interval": 2},
                (Path("archive.zip"), Path("file.tsv")): {"devices": ["i1"], "averaging_interval": 2},
                (Path("test.proc_noAvg.h5"), Path("")): {"devices": ["i1"], "averaging_interval": 2, "is_hdf5": True},
                (Path("test.proc.h5"), Path("")): {"devices": ["i1"], "averaging_interval": 2, "is_hdf5": True},
                (Path("_raw/test.h5"), Path("")): {"devices": ["i1"], "averaging_interval": 2, "is_hdf5": True},
                (Path("raw_dir"), Path("raw_file.txt")): {"devices": ["i1"], "averaging_interval": 2},
            },
            {"i1"},
            "directory file should be first, then archive, then proc_noAvg HDF5, then proc HDF5, then raw HDF5, then raw text file as per: source type priority order"
        ),
        # Test with averaging interval priority (lower interval has higher priority)
        (
            {
                (Path("dir"), Path("file600.tsv")): {"devices": ["i1"], "averaging_interval": 600},
                (Path("dir"), Path("file2.tsv")): {"devices": ["i1"], "averaging_interval": 2},
                (Path("dir"), Path("file60.tsv")): {"devices": ["i1"], "averaging_interval": 60},
            },
            {"i1"},
            "file with 2s averaging should be first, then 60s, then 600s as per: averaging interval priority (lower is better)"
        ),
        # Test with specificity (dedicated files better than combined)
        (
            {
                (Path("dir"), Path("file.tsv")): {"devices": ["i1"], "averaging_interval": 2},
                (Path("dir"), Path("combined.tsv")): {"devices": ["i1", "i2"], "averaging_interval": 2},
                (Path("dir"), Path("generic.tsv")): {"devices": ["*"], "averaging_interval": 2},
            },
            {"i1"},
            "dedicated file for i1 should be first, then combined file, then generic file as per: specificity priority (dedicated better than combined)"
        ),
        # Test with number of devices (fewer is better)
        (
            {
                (Path("dir"), Path("file_i1.tsv")): {"devices": ["i1"], "averaging_interval": 2},
                (Path("dir"), Path("file_i1_i2.tsv")): {"devices": ["i1", "i2"], "averaging_interval": 2},
                (Path("dir"), Path("file_i1_i2_i3.tsv")): {"devices": ["i1", "i2", "i3"], "averaging_interval": 2},
            },
            {"i1"},
            "file with only i1 device should be first, then i1+i2, then i1+i2+i3 as per: fewer devices have higher priority"
        ),
        # Test with unmatched devices (fewer unmatched is better)
        (
            {
                (Path("dir"), Path("file_i1.tsv")): {"devices": ["i1"], "averaging_interval": 2},
                (Path("dir"), Path("file_i1_i2.tsv")): {"devices": ["i1", "i2"], "averaging_interval": 2},
                (Path("dir"), Path("file_i2_i3.tsv")): {"devices": ["i2", "i3"], "averaging_interval": 2},
            },
            {"i1"},
            "file with only matched device should be first, then file with 1 match 1 unmatched, then 0 matches 2 unmatched as per: fewer unmatched devices have higher priority"
        ),
    ],
    ids=[
        "source_type_priority",
        "averaging_interval_priority",
        "specificity_priority",
        "number_of_devices_priority",
        "unmatched_devices_priority"
    ]
)
def test_sort_text_outputs_priority_order(text_outputs, json_devices, expected_order_desc):
    """
    Test that sort_text_outputs sorts files according to the specified priority order.
    """
    sorted_outputs = sort_data_paths(text_outputs, json_devices)

    sorted_paths = [item[0] for item in sorted_outputs]
    original_paths = list(text_outputs.keys())

    # Verify that the result is actually sorted (this test is mainly to ensure the function works without errors
    # and to document the expected behavior)
    assert len(sorted_outputs) == len(text_outputs), f"all items should be preserved in sort as per: {expected_order_desc}"

    # Check that it's the same items, just reordered
    assert set(sorted_paths) == set(original_paths), f"all items should be present after sort as per: {expected_order_desc}"

    # Verify the actual order based on the test case
    if "source type priority" in expected_order_desc:
        # Expected order: directories (0) > archives (1) > proc_noAvg HDF5 (2) > proc HDF5 (3) > raw HDF5 (4) > raw text files (6)
        expected_order = [
            (Path("dir"), Path("file.tsv")),  # directory - priority 0
            (Path("archive.zip"), Path("file.tsv")),  # archive - priority 1
            (Path("test.proc_noAvg.h5"), Path("")),  # proc_noAvg HDF5 - priority 2
            (Path("test.proc.h5"), Path("")),  # proc HDF5 - priority 3
            (Path("_raw/test.h5"), Path("")),  # raw HDF5 - priority 4
            (Path("raw_dir"), Path("raw_file.txt")),  # raw text file - priority 6
        ]
        assert sorted_paths == expected_order, f"source type priority not respected. Expected: {expected_order}, Got: {sorted_paths} as per: {expected_order_desc}"

    elif "averaging interval priority" in expected_order_desc:
        # Expected order: lowest averaging interval first
        expected_order = [
            (Path("dir"), Path("file2.tsv")),  # averaging_interval: 2
            (Path("dir"), Path("file60.tsv")),  # averaging_interval: 60
            (Path("dir"), Path("file600.tsv")),  # averaging_interval: 600
        ]
        assert sorted_paths == expected_order, f"averaging interval priority not respected. Expected: {expected_order}, Got: {sorted_paths} as per: {expected_order_desc}"

    elif "specificity priority" in expected_order_desc:
        # Expected order: dedicated files better than combined
        expected_order = [
            (Path("dir"), Path("file.tsv")),  # dedicated to i1
            (Path("dir"), Path("combined.tsv")),  # combined i1, i2
            (Path("dir"), Path("generic.tsv")),  # generic ["*"]
        ]
        assert sorted_paths == expected_order, f"specificity priority not respected. Expected: {expected_order}, Got: {sorted_paths} as per: {expected_order_desc}"

    elif "number of devices priority" in expected_order_desc:
        # Expected order: fewer devices better
        expected_order = [
            (Path("dir"), Path("file_i1.tsv")),  # 1 device
            (Path("dir"), Path("file_i1_i2.tsv")),  # 2 devices
            (Path("dir"), Path("file_i1_i2_i3.tsv")),  # 3 devices
        ]
        assert sorted_paths == expected_order, f"number of devices priority not respected. Expected: {expected_order}, Got: {sorted_paths} as per: {expected_order_desc}"

    elif "unmatched devices priority" in expected_order_desc:
        # Expected order: fewer unmatched devices better
        expected_order = [
            (Path("dir"), Path("file_i1.tsv")),  # 0 unmatched (i1 matches)
            (Path("dir"), Path("file_i1_i2.tsv")),  # 1 unmatched (i1 matches, i2 doesn't)
            (Path("dir"), Path("file_i2_i3.tsv")),  # 2 unmatched (i2, i3 don't match)
        ]
        assert sorted_paths == expected_order, f"unmatched devices priority not respected. Expected: {expected_order}, Got: {sorted_paths} as per: {expected_order_desc}"


def test_sort_text_outputs_empty_inputs():
    """Test sorting with empty inputs."""
    result = sort_data_paths({}, set())
    assert result == [], "should return empty list for empty input as per: empty input handling"

    result = sort_data_paths({}, {"i1"})
    assert result == [], "should return empty list when text_outputs is empty as per: empty text_outputs handling"


def test_sort_text_outputs_default_averaging():
    """Test that files without averaging_interval use default value."""
    default_avg = config.default_text_file_averaging

    text_outputs = {
        (Path("dir"), Path("file.tsv")): {"devices": ["i1"]},  # No averaging_interval specified
    }

    sorted_outputs = sort_data_paths(text_outputs, {"i1"})

    # The file should be processed using the default averaging value
    assert len(sorted_outputs) == 1, "file without averaging should still be included in output as per: default averaging handling"
    assert sorted_outputs[0][0] == (Path("dir"), Path("file.tsv")), "file should be preserved in output as per: default averaging handling"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])