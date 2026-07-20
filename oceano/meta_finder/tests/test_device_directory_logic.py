import pytest
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from meta_finder.file_finder import find_device_dirs


def test_device_directory_logic(common_test_data_setup):
    """Test the logic for finding device directories based on cruise directory name and subdirectories."""
    base_path = common_test_data_setup

    test_cases = [
        (
            "cruise_with_device_name_and_subdirs",
            base_path / "230507_ABP53_inclinometer",
            ["230508_inclinometer@i03", "230509_wavegauge@w01"],  # Should return device subdirs, NOT cruise dir
            "Should return device subdirs when cruise dir has device name and device subdirs"
        ),
        (
            "cruise_with_device_name_no_device_subdirs",
            base_path / "230507_cruise_inclinometer",
            ["230508_other"], # Should return the dated subdir since there are no other device subdirs
            "Should return dated subdirs when it has only dated subdirs and no other device subdirs"
        ),
        (
            "cruise_with_ati_and_subdirs",
            base_path / "230507_ABP53@i",
            ["230508_inclinometer@i03", "230509@i04"],  # Should return device subdirs, NOT cruise dir
            "Should return device subdirs when cruise dir has @i and device subdirs"
        ),
        (
            "cruise_with_device_name_dated_subdirs_no_device_files",
            base_path / "230507_test_inclinometer",
            ["230508_test"],   # Expected device subdirs with date prefixes
            "Should return dated subdirs when it has only dated subdirs and no other device subdirs"
        )
    ]

    for test_name, cruise_dir_path, expected_device_dir_names, description in test_cases:
        # Call the function to find device directories
        device_dirs = find_device_dirs(cruise_dir_path)

        # Get the names of the found device directories
        found_device_dir_names = [d.name for d in device_dirs]

        # Check if the found device directories match exactly what we expect
        assert sorted(found_device_dir_names) == sorted(expected_device_dir_names), \
            f"Test {test_name} ({description}): Expected device dirs={expected_device_dir_names}, " \
            f"but got {found_device_dir_names}. Found device dirs: {[d.name for d in device_dirs]}"


@pytest.mark.parametrize("test_name,cruise_dir_path,expected_device_dir_names,description", [
    ("cruise_with_device_name_and_subdirs",
     "test_data/test_device_directory_logic/230507_ABP53_inclinometer",
     ["230508_inclinometer@i03", "230509_wavegauge@w01"],  # Should return device subdirs, NOT cruise dir
     "Should return device subdirs when cruise dir has device name and device subdirs"),
    ("cruise_with_device_name_no_device_subdirs",
     "test_data/test_device_directory_logic/230507_cruise_inclinometer",
     ["230508_other"], # Should return the dated subdir since there are no other device subdirs
     "Should return dated subdirs when it has only dated subdirs and no other device subdirs"),
    ("cruise_with_ati_and_subdirs",
     "test_data/test_device_directory_logic/230507_ABP53@i",
     ["230508_inclinometer@i03", "230509@i04"],  # Should return device subdirs, NOT cruise dir
     "Should return device subdirs when cruise dir has @i and device subdirs"),
    ("cruise_with_device_name_dated_subdirs_no_device_files",
     "test_data/test_device_directory_logic/230507_test_inclinometer",
     ["230508_test"],   # Expected device subdirs with date prefixes
     "Should return dated subdirs when it has only dated subdirs and no other device subdirs"),
])
def test_device_directory_logic(test_name, cruise_dir_path, expected_device_dir_names, description):
    """Test the logic for finding device directories based on cruise directory name and subdirectories."""
    cruise_dir = Path(cruise_dir_path)

    # Call the function to find device directories
    device_dirs = find_device_dirs(cruise_dir)

    # Get the names of the found device directories
    found_device_dir_names = [d.name for d in device_dirs]

    # Check if the found device directories match exactly what we expect
    assert sorted(found_device_dir_names) == sorted(expected_device_dir_names), \
        f"Test {test_name} ({description}): Expected device dirs={expected_device_dir_names}, " \
        f"but got {found_device_dir_names}. Found device dirs: {[d.name for d in device_dirs]}"
