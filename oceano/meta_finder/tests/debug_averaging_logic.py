"""
Debug script to test the averaging interval logic in file sorting.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import PurePosixPath
from meta_finder.data_processor import sort_data_paths

def test_sort_with_low_averaging_vs_no_averaging():
    """Test sorting when there are files with low averaging vs no averaging."""
    json_devices = {'i1', 'i2', 'i3'}
    text_outputs = {
        (PurePosixPath('path/to/dir'), PurePosixPath('file1.tsv')): {'averaging_interval': 0, 'devices': ['i1']},  # No averaging
        (PurePosixPath('path/to/dir'), PurePosixPath('file2.tsv')): {'averaging_interval': 2, 'devices': ['i1']},  # 2 seconds
        (PurePosixPath('path/to/dir'), PurePosixPath('file3.tsv')): {'averaging_interval': 1, 'devices': ['i1']},  # 1 second
    }

    print("Test: Sorting with low averaging vs no averaging")
    print("Files:")
    for path_tuple, metadata in text_outputs.items():
        print(f"  {path_tuple[1].name}: {metadata['averaging_interval']}s")

    sorted_items = sort_data_paths(text_outputs, json_devices)
    sorted_filenames = [item[0][1].name for item in sorted_items]

    print("Sorted order:")
    for i, filename in enumerate(sorted_filenames):
        print(f"  {i+1}. {filename}")

    # Since there are files with low averaging (1s and 2s), files with no averaging (0s)
    # should have lower priority
    # Expected order based on priority:
    # 1. file3: 1s -> (1, False, 1, 0)
    # 2. file2: 2s -> (2, False, 1, 0)
    # 3. file1: 0s (treated as inf) -> (inf, False, 1, 0)

    expected_order = ['file3.tsv', 'file2.tsv', 'file1.tsv']
    print(f"Expected order: {expected_order}")
    print(f"Actual order:   {sorted_filenames}")
    print(f"Match: {sorted_filenames == expected_order}")
    print()

def test_sort_with_only_no_averaging():
    """Test sorting when there are only files with no averaging."""
    json_devices = {'i1', 'i2', 'i3'}
    text_outputs = {
        (PurePosixPath('path/to/dir'), PurePosixPath('file1.tsv')): {'averaging_interval': 0, 'devices': ['i1']},  # No averaging
        (PurePosixPath('path/to/dir'), PurePosixPath('file2.tsv')): {'averaging_interval': 0, 'devices': ['i1', 'i2']},  # No averaging
    }

    print("Test: Sorting with only no averaging")
    print("Files:")
    for path_tuple, metadata in text_outputs.items():
        print(f"  {path_tuple[1].name}: {metadata['averaging_interval']}s")

    sorted_items = sort_data_paths(text_outputs, json_devices)
    sorted_filenames = [item[0][1].name for item in sorted_items]

    print("Sorted order:")
    for i, filename in enumerate(sorted_filenames):
        print(f"  {i+1}. {filename}")

    # Since there are only files with no averaging, they should be sorted normally
    # Expected order based on priority:
    # 1. file1: 0s, dedicated, 1 device, 0 unmatched -> (0, False, 1, 0)
    # 2. file2: 0s, dedicated, 2 devices, 0 unmatched -> (0, False, 2, 0)

    expected_order = ['file1.tsv', 'file2.tsv']
    print(f"Expected order: {expected_order}")
    print(f"Actual order:   {sorted_filenames}")
    print(f"Match: {sorted_filenames == expected_order}")
    print()

if __name__ == '__main__':
    test_sort_with_low_averaging_vs_no_averaging()
    test_sort_with_only_no_averaging()