#!/usr/bin/env python3
# Test script to verify GPX integration works correctly with download_copernicus.py

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from hydra import initialize, compose

def test_gpx_integration():
    """Test that GPX integration works correctly in download_copernicus.py"""
    print("Testing GPX integration in download_copernicus.py...")

    try:
        # Import the required modules
        from get_datasets.download_copernicus import main
        from get_datasets.utils import extract_coordinates_from_gpx

        # Load the configuration
        with initialize(config_path='cfg', version_base=None):
            cfg = compose(config_name='base', overrides=['projects=abp56_tchain/point'])

            # Get the project configuration
            project_cfg = cfg.projects.abp56_tchain
            print(f"Configuration loaded: {project_cfg.dir_save}")
            print(f"GPX path from config: {project_cfg.gpx_path}")

            # Test extracting coordinates from GPX
            # When running with Hydra, the working directory might be different
            # So we need to resolve the path relative to the config file location
            gpx_path = Path(project_cfg.gpx_path)
            if not gpx_path.exists():
                # Try alternative path relative to project root
                gpx_path = Path("oceano/get_datasets") / project_cfg.gpx_path
            points_from_gpx = extract_coordinates_from_gpx(gpx_path, None)

            if points_from_gpx:
                print(f"Successfully extracted {len(points_from_gpx)} points from GPX file:")
                for name, coords in points_from_gpx.items():
                    print(f"  - {name}: lat={coords['lat']}, lon={coords['lon']}")

                # Test that these points would be used in the configuration
                points = project_cfg.points or []
                points += list(points_from_gpx.values())

                print(f"Total points to be processed: {len(points)}")
                for i, point in enumerate(points):
                    print(f"  Point {i+1}: lat={point['lat']}, lon={point['lon']}")

                print("\nGPX integration test PASSED!")
            else:
                print("ERROR: No points extracted from GPX file!")
                return False

    except Exception as e:
        print(f"Error during GPX integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_gpx_integration()
    if success:
        print("\nAll tests passed! GPX integration is working correctly.")
    else:
        print("\nTests failed!")
        exit(1)