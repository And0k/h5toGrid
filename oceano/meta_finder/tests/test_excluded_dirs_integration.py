"""Integration test for directory exclusion in file_finder functions."""

import tempfile
import shutil
from pathlib import Path
from meta_finder import config
from meta_finder.file_finder import find_cruise_directories, find_device_dirs


def test_cruise_directories_exclusion():
    """Test that find_cruise_directories excludes directories ending with '-'."""
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test cruise directories
        included_dir1 = tmpdir / "230616_normal_cruise"
        included_dir2 = tmpdir / "230617_another_cruise"
        excluded_dir1 = tmpdir / "230618_excluded-"
        excluded_dir2 = tmpdir / "230619_test-"

        for d in [included_dir1, included_dir2, excluded_dir1, excluded_dir2]:
            d.mkdir()

        # Find cruise directories
        found_dirs = find_cruise_directories([tmpdir])

        # Check that excluded directories are not in the results
        found_names = [d.name for d in found_dirs]

        assert "230616_normal_cruise" in found_names, "Normal cruise directory should be found"
        assert "230617_another_cruise" in found_names, "Another normal cruise directory should be found"
        assert "230618_excluded-" not in found_names, "Directory ending with '-' should be excluded"
        assert "230619_test-" not in found_names, "Directory ending with '-' should be excluded"

        print(f"✓ Found {len(found_dirs)} cruise directories (excluded {4 - len(found_dirs)})")


def test_device_directories_exclusion():
    """Test that find_device_dirs excludes directories ending with '-'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a cruise directory
        cruise_dir = tmpdir / "230616_test_cruise"
        cruise_dir.mkdir()

        # Create device directories with proper structure
        included_dev1 = cruise_dir / "230616_inclinometer@i03"
        included_dev2 = cruise_dir / "230617_wavegauge@w01"
        excluded_dev1 = cruise_dir / "230618_device-"
        excluded_dev2 = cruise_dir / "230619_test-"

        for d in [included_dev1, included_dev2, excluded_dev1, excluded_dev2]:
            d.mkdir()
            # Create _raw subdirectory to make it a valid device directory
            (d / "_raw").mkdir()

        # Find device directories
        found_dirs = find_device_dirs(cruise_dir=cruise_dir)

        # Check that excluded directories are not in the results
        found_names = [d.name for d in found_dirs]

        assert "230616_inclinometer@i03" in found_names, "Normal device directory should be found"
        assert "230617_wavegauge@w01" in found_names, "Another normal device directory should be found"
        assert "230618_device-" not in found_names, "Device directory ending with '-' should be excluded"
        assert "230619_test-" not in found_names, "Device directory ending with '-' should be excluded"

        print(f"✓ Found {len(found_dirs)} device directories (excluded {4 - len(found_dirs)})")


def test_nested_device_directories_exclusion():
    """Test exclusion works for nested dated directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create cruise directory with dated subdirectories
        cruise_dir = tmpdir / "230616_cruise"
        cruise_dir.mkdir()

        # Create dated directory
        dated_dir = cruise_dir / "230616_data"
        dated_dir.mkdir()

        # Create device directories inside dated directory
        included_dev = dated_dir / "inclinometer@i03"
        excluded_dev = dated_dir / "device-"

        for d in [included_dev, excluded_dev]:
            d.mkdir()
            (d / "_raw").mkdir()

        # Find device directories
        found_dirs = find_device_dirs(cruise_dir=cruise_dir)

        # Check results
        found_names = [d.name for d in found_dirs]

        assert "inclinometer@i03" in found_names, "Normal device directory should be found"
        assert "device-" not in found_names, "Device directory ending with '-' should be excluded"

        print(f"✓ Found {len(found_dirs)} nested device directories (excluded {2 - len(found_dirs)})")


if __name__ == "__main__":
    print("Testing cruise directory exclusion...")
    test_cruise_directories_exclusion()

    print("\nTesting device directory exclusion...")
    test_device_directories_exclusion()

    print("\nTesting nested device directory exclusion...")
    test_nested_device_directories_exclusion()

    print("\n✅ All integration tests passed!")
