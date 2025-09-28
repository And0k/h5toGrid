import pytest
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import logging
import os
import shutil

# Temporarily add the scripts/downloading directory to the Python path
# to allow importing download_copernicus_region
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ..download_copernicus_region import main as download_region_main

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
l = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def setup_test_environment(request):
    """Fixture to set up and tear down a clean test environment for each test function."""
    test_name = request.node.name
    test_base_path = Path(f"data/downloaded_test_region/{test_name}")
    test_history_file = Path(f"history/download_history_test_region_{test_name}.json")

    # Clean up previous test downloads and history
    if test_base_path.exists():
        shutil.rmtree(test_base_path)
    test_base_path.mkdir(parents=True, exist_ok=True)

    if test_history_file.exists():
        os.remove(test_history_file)

    yield test_base_path, test_history_file

    # Teardown: Clean up after test
    if test_base_path.exists():
        shutil.rmtree(test_base_path)
    if test_history_file.exists():
        os.remove(test_history_file)

def run_region_test(project_name: str, setup_test_environment):
    """Helper function to run a single region download test."""
    test_base_path, test_history_file = setup_test_environment

    with hydra.initialize(config_path="../cfg", version_base=None):
        cfg = hydra.compose(config_name="base", overrides=[f"projects={project_name}/region"])
        cfg.base.local_path = str(test_base_path)
        cfg.base.history_file = str(test_history_file)

        l.info(f"Testing with config for project '{project_name}':\n{OmegaConf.to_yaml(cfg)}")

        download_region_main(cfg)

        # Assertions:
        assert test_history_file.exists()
        from ..manager import DownloadHistoryManager
        history_manager = DownloadHistoryManager(test_history_file)
        history = history_manager.get_history()
        assert len(history) == 1 # Expecting one entry per project run

        expected_dir = Path(cfg.base.local_path) / cfg.copernicus.region.dir_suffix
        assert expected_dir.is_dir()
        nc_files = list(expected_dir.glob("*.nc"))
        assert len(nc_files) > 0, f"No .nc files found in {expected_dir}"
        l.info(f"Test for project '{project_name}' passed successfully.")

def test_region_mariculture_1(setup_test_environment):
    run_region_test("mariculture_1", setup_test_environment)

def test_region_mariculture_2(setup_test_environment):
    run_region_test("mariculture_2", setup_test_environment)

def test_region_mariculture_3(setup_test_environment):
    run_region_test("mariculture_3", setup_test_environment)

def test_region_abp56_tchain(setup_test_environment):
    run_region_test("abp56_tchain", setup_test_environment)

def test_region_inflow(setup_test_environment):
    run_region_test("inflow", setup_test_environment)
