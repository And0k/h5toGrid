#!/usr/bin/env python3
# Debug script to check how the configuration is structured

import os
from pathlib import Path
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

def debug_config():
    """Debug configuration structure"""
    print("Debugging configuration structure...")

    try:
        GlobalHydra.instance().clear()

        with initialize(config_path='cfg', version_base=None):
            cfg = compose(config_name='base', overrides=['projects=abp56_tchain/point'])
            print("Configuration loaded successfully")

            print(f"Type of cfg: {type(cfg)}")
            print(f"cfg keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'No keys method'}")

            # Check projects config
            if hasattr(cfg, 'projects'):
                print(f"cfg.projects exists: {cfg.projects}")
                print(f"cfg.projects type: {type(cfg.projects)}")
                if hasattr(cfg.projects, 'abp56_tchain'):
                    print(f"cfg.projects.abp56_tchain exists: {cfg.projects.abp56_tchain}")
                    print(f"abp56_tchain keys: {list(cfg.projects.abp56_tchain.keys())}")

            # Check copernicus config
            if hasattr(cfg, 'copernicus'):
                print(f"cfg.copernicus exists: {cfg.copernicus}")
                print(f"copernicus keys: {list(cfg.copernicus.keys())}")

            # Print the whole config structure
            print(f"\nFull config structure:")
            print(cfg)

    except Exception as e:
        print(f"Error during config debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config()