#!/usr/bin/env python3
# Test script to verify full integration with the yaml configuration

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from hydra import initialize, compose

def test_full_integration():
    """Test full integration with the yaml configuration"""
    print("Testing full integration with the yaml configuration...")

    try:
        # Import the required modules
        from get_datasets.download_copernicus import main
        from get_datasets.d_utils import extract_coordinates_from_gpx

        # Load the configuration
        with initialize(config_path='cfg', version_base=None):
            cfg = compose(config_name='base', overrides=['projects=abp56_tchain/point'])

            # Get the project configuration
            project_cfg = cfg.projects.abp56_tchain
            print(f"Configuration loaded: {project_cfg.dir_save}")
            print(f"Date range: {project_cfg.date_range}")
            print(f"GPX path from config: {project_cfg.gpx_path}")
            print(f"Dataset variables: {project_cfg.dataset_vars}")
            print(f"Depth range: {project_cfg.depth_min} to {project_cfg.depth_max}m")

            # Test extracting coordinates from GPX
            gpx_path = Path(project_cfg.gpx_path)
            if not gpx_path.exists():
                # Try alternative path relative to project root
                gpx_path = Path("oceano/get_datasets") / project_cfg.gpx_path
            points_from_gpx = extract_coordinates_from_gpx(gpx_path, None)

            if points_from_gpx:
                print(f"Successfully extracted {len(points_from_gpx)} points from GPX file:")
                for name, coords in points_from_gpx.items():
                    print(f"  - {name}: lat={coords['lat']}, lon={coords['lon']}")

                # Check if the points would be processed according to the logic in download_copernicus.py
                points = project_cfg.points or []
                points += list(points_from_gpx.values())

                print(f"\nTotal points to be processed: {len(points)}")
                for i, point in enumerate(points):
                    print(f"  Point {i+1}: lat={point['lat']}, lon={point['lon']}")

                # Verify the configuration matches requirements:
                # - date_range: ["2024-06-25", "2023-06-26"] (should be fixed to have correct order)
                # - gpx file: tests\test_data\points_85m_for_cmems.gpx
                # - dataset: 'cmems_mod_bal_phy_anfc_PT1H-i'
                # - variables: ('thetao','so','sob')
                # - depth range: [0..100m]

                date_range_correct = project_cfg.date_range == ["2024-06-25", "2024-06-26"]
                gpx_path_correct = "points_85m_for_cmems.gpx" in str(project_cfg.gpx_path)
                dataset_correct = "cmems_mod_bal_phy_anfc_PT1H-i" in project_cfg.dataset_vars
                variables_correct = set(project_cfg.dataset_vars["cmems_mod_bal_phy_anfc_PT1H-i"]) == {"thetao", "so", "sob"}
                depth_range_correct = project_cfg.depth_min == 0 and project_cfg.depth_max == 100

                print(f"\nConfiguration verification:")
                print(f"  Date range correct: {date_range_correct}")
                print(f"  GPX path correct: {gpx_path_correct}")
                print(f"  Dataset correct: {dataset_correct}")
                print(f"  Variables correct: {variables_correct}")
                print(f"  Depth range correct: {depth_range_correct}")

                all_correct = all([date_range_correct, gpx_path_correct, dataset_correct,
                                 variables_correct, depth_range_correct, len(points) > 0])

                if all_correct:
                    print("\n[SUCCESS] All configuration requirements are met!")
                    print("[SUCCESS] GPX points are correctly loaded and will be processed!")
                    print("[SUCCESS] The configuration matches all specified requirements!")
                    return True
                else:
                    print("\n[ERROR] Some configuration requirements are not met!")
                    return False
            else:
                print("ERROR: No points extracted from GPX file!")
                return False

    except Exception as e:
        print(f"Error during full integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_integration()
    if success:
        print("\n[SUCCESS] All integration tests passed! The configuration is working correctly.")
    else:
        print("\n[ERROR] Integration tests failed!")
        exit(1)