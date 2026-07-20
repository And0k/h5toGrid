"""
Test for the averaging interval logic in file sorting.
"""

import pytest
from pathlib import PurePosixPath
from meta_finder.data_processor import sort_data_paths

def test_sort_with_high_averaging_intervals():
    """Test sorting when there are files with high averaging intervals."""
    json_devices = {'i1', 'i2', 'i3'}
    text_outputs = {
        (PurePosixPath('path/to/dir'), PurePosixPath('file1.tsv')): {'averaging_interval': 600, 'devices': ['i1', 'i2']},
        (PurePosixPath('path/to/dir'), PurePosixPath('file2.tsv')): {'averaging_interval': 300, 'devices': ['i1']},
        (PurePosixPath('path/to/dir'), PurePosixPath('file3.tsv')): {'averaging_interval': 2, 'devices': ['i1']},  # 2 seconds
    }

    # Expected order based on priority:
    # 1. file3: 2s, dedicated, 1 device, 0 unmatched -> (2, False, 1, 0)
    # 2. file2: 300s, dedicated, 1 device, 0 unmatched -> (300, False, 1, 0)
    # 3. file1: 600s, dedicated, 2 devices, 0 unmatched -> (600, False, 2, 0)

    sorted_items = sort_data_paths(text_outputs, json_devices)
    sorted_filenames = [item[0][1].name for item in sorted_items]

    expected_order = ['file3.tsv', 'file2.tsv', 'file1.tsv']
    assert sorted_filenames == expected_order

def test_sort_with_only_low_averaging_intervals():
    """Test sorting when there are only files with low averaging intervals."""
    json_devices = {'i1', 'i2', 'i3'}
    text_outputs = {
        (PurePosixPath('path/to/dir'), PurePosixPath('file1.tsv')): {'averaging_interval': 2, 'devices': ['i1', 'i2']},  # 2 seconds
        (PurePosixPath('path/to/dir'), PurePosixPath('file2.tsv')): {'averaging_interval': 1, 'devices': ['i1']},      # 1 second
    }

    # Since all files have averaging <= 2 seconds and there are no files with averaging > 2 seconds,
    # they should all be treated as having no averaging (0) for sorting purposes.
    # Expected order based on priority:
    # 1. file2: 0s (treated as), dedicated, 1 device, 0 unmatched -> (0, False, 1, 0)
    # 2. file1: 0s (treated as), dedicated, 2 devices, 0 unmatched -> (0, False, 2, 0)

    sorted_items = sort_data_paths(text_outputs, json_devices)
    sorted_filenames = [item[0][1].name for item in sorted_items]

    expected_order = ['file2.tsv', 'file1.tsv']
    assert sorted_filenames == expected_order