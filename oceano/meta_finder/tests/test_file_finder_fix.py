"""Test to verify the file_finder.py fix for re.subn issue."""

import re
from meta_finder import config
from meta_finder.file_finder import extract_device_id_from_raw_file_name


def test_dated_prefix_removal():
    """Test that dated prefix is correctly removed from filenames."""
    # Test cases from the docstring examples
    test_cases = [
        ("#W1_130510.txt", "w1"),
        ("130510#W1_130510.txt", "w1"),
        ("i1.txt", "i1"),
        ("W1.txt", "w1"),
        ("#1.txt", "i1"),
        ("1.txt", None),
    ]

    for original, expected_device in test_cases:
        print(f"Original: {original}")

        # Use the actual function to extract device ID
        device_id = extract_device_id_from_raw_file_name(original)

        print(f"Device ID: {device_id}")
        assert device_id == expected_device, (
            f"Device ID extraction failed for {original}: "
            f"expected {expected_device!r}, got {device_id!r}"
        )
        print("✓ Test passed\n")


if __name__ == "__main__":
    print("Testing file_finder.py fix for re.subn issue\n")
    print("=" * 60)
    test_dated_prefix_removal()
    print("=" * 60)
    print("\n✓ All tests passed!")
