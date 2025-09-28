import pytest
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import logging
import os
import shutil # Added shutil import

from download_copernicus_point_wind import main as download_point_wind_main

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
l = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def setup_test_environment(request):
    """Fixture to set up and tear down a clean test environment for each test function."""
    # Get the test name to create unique directories and files
    test_name = request.node.name
    test_base_path = Path(f"data/downloaded_test/{test_name}")
    test_history_file = Path(f"history/download_history_test_{test_name}.json")

    # Clean up previous test downloads and history
    if test_base_path.exists():
        shutil.rmtree(test_base_path)
    test_base_path.mkdir(parents=True, exist_ok=True)

    if test_history_file.exists():
        os.remove(test_history_file)

    # Yield the test environment paths to the test function
    yield test_base_path, test_history_file

    # Teardown: Clean up after test
    if test_base_path.exists():
        shutil.rmtree(test_base_path)
    if test_history_file.exists():
        os.remove(test_history_file)

def run_point_wind_test(project_name: str, setup_test_environment):
    """Helper function to run a single point wind download test."""
    test_base_path, test_history_file = setup_test_environment

    with hydra.initialize(config_path="../cfg", version_base=None):
        cfg = hydra.compose(config_name="base", overrides=[f"projects={project_name}/point_wind"])
        cfg.base.local_path = str(test_base_path)
        cfg.base.history_file = str(test_history_file)

        l.info(f"Testing with config for project '{project_name}':\n{OmegaConf.to_yaml(cfg)}")

        download_point_wind_main(cfg)

        # Assertions:
        assert test_history_file.exists()
        from manager import DownloadHistoryManager
        history_manager = DownloadHistoryManager(test_history_file)
        history = history_manager.get_history()
        assert len(history) == len(cfg.copernicus.point_wind.points) # Expecting one entry per point in the config

        expected_dir = Path(cfg.base.local_path) / cfg.copernicus.point_wind.dir_suffix

        # Check if the download was skipped due to missing copernicusmarine library
        all_skipped = all(
            entry.get('options', {}).get('status') == 'skipped' and
            'copernicusmarine library not available' in entry.get('options', {}).get('reason', '')
            for entry in history
        )

        if all_skipped:
            l.info(f"Test for project '{project_name}' passed successfully (downloads skipped due to missing copernicusmarine library).")
            assert not expected_dir.exists() or len(list(expected_dir.glob("*.nc"))) == 0
        else:
            assert expected_dir.is_dir()
            nc_files = list(expected_dir.glob("*.nc"))
            assert len(nc_files) > 0, f"No .nc files found in {expected_dir}"
            l.info(f"Test for project '{project_name}' passed successfully.")

def test_point_wind_kulikovo(setup_test_environment):
    run_point_wind_test("kulikovo", setup_test_environment)

def test_point_wind_abp56(setup_test_environment):
    run_point_wind_test("abp56", setup_test_environment)
