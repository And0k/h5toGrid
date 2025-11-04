#!/usr/bin/env python3
# Test script to verify the configuration loading works correctly

import os
from hydra import initialize, compose

def test_config_loading():
    """Test that the Hydra configuration loads correctly"""
    print("Testing Hydra config loading...")

    # Change to the correct directory
    os.chdir("oceano/get_datasets")

    with initialize(config_path='cfg', version_base=None):
        cfg = compose(config_name='base', overrides=['projects=abp56_tchain/point'])
        print(f"Config loaded successfully: {hasattr(cfg, 'projects') and 'abp56_tchain' in cfg.projects}")

        # Print some important config values
        if hasattr(cfg, 'projects') and 'abp56_tchain' in cfg.projects:
            project_cfg = cfg.projects.abp56_tchain
            print(f"Project dir_save: {project_cfg.dir_save}")
            print(f"Project date_range: {project_cfg.date_range}")
            print(f"Project gpx_path: {getattr(project_cfg, 'gpx_path', 'NOT SET')}")
            print(f"Project dataset_vars: {project_cfg.dataset_vars}")
            print("Configuration loaded correctly!")

if __name__ == "__main__":
    test_config_loading()