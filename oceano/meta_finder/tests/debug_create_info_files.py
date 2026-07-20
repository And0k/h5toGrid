#!/usr/bin/env python
"""
Debug script for create_info_files.py functionality.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def debug_create_info_files():
    """Debug the create_info_files functionality."""
    print("Debugging create_info_files.py functionality")

    # Test directory
    test_dir = Path(__file__).parent.parent / "test_data"
    print(f"Test directory: {test_dir}")

    # Check what directories are in test_dir
    print("Directories in test_dir:")
    if test_dir.exists():
        for item in test_dir.iterdir():
            if item.is_dir():
                print(f"  {item.name}")
    else:
        print("Test directory does not exist!")

    # Import and test the functions
    try:
        from meta_finder.file_finder import find_cruise_directories
        from meta_finder.file_finder import discover_device_dirs

        # Test find_cruise_directories
        print("\nTesting find_cruise_directories:")
        cruise_dirs = find_cruise_directories([test_dir])
        print(f"Found {len(cruise_dirs)} cruise directories:")
        for dir in cruise_dirs:
            print(f"  {dir}")

        # Test scan_cruises_for_devices
        print("\nTesting scan_cruises_for_devices:")
        cruise_devices = discover_device_dirs([test_dir])
        print(f"Found {len(cruise_devices)} cruises with device directories:")
        for cruise_dir, device_dirs in cruise_devices.items():
            print(f"  Cruise: {cruise_dir}")
            print(f"  Device dirs: {device_dirs}")

    except Exception as e:
        print(f"Error running debug: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    debug_create_info_files()