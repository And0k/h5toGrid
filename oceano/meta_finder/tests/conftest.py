import os
import pytest
from pathlib import Path
import shutil


# Set the environment variable at the module level, before any imports that might trigger logging
os.environ['META_FINDER_TEST_MODE'] = '1'


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_output_dir_session():
    """
    Session-level fixture to ensure test output directory is clean at the beginning of the test session.
    """
    output_dir = Path("test_data") / "meta_temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up all files at the start of the session
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path, ignore_errors=True)

    yield  # Run the entire test session

    # Optionally, we can preserve results after the session completes


@pytest.fixture(scope="session")
def test_output_dir():
    """
    Global fixture that provides a consistent output directory for all tests.

    This fixture ensures that all tests write their output files (like
    {yymmdd_HHMM}_files_TCM.tsv and {yymmdd_HHMM}_meta_TCM.tsv.tsv) to the
    test_data/meta_temp/ directory as required.
    """
    output_dir = Path("test_data") / "meta_temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def mock_setup_logging(mocker, test_output_dir):
    """
    Session-level fixture that mocks the setup_logging function to use the test output directory.
    This ensures that all log files are written to the test_data/meta_temp/logs/ directory
    instead of the default meta/ directory for the entire test session.
    """
    from meta_finder.logging_config import setup_logging

    def patched_setup_logging(
        name=__name__,
        log_level=None,
        include_function_name=True,
        log_file_dir=None,
        log_file_sfx="meta_finder",
        console_level=None,
        file_level=None
    ):
        # Override log_file_dir to use test output directory
        log_file_dir = test_output_dir / "logs"
        log_file_dir.mkdir(exist_ok=True)
        return setup_logging(
            name=name,
            log_level=log_level,
            include_function_name=include_function_name,
            log_file_dir=log_file_dir,
            log_file_sfx=log_file_sfx,
            console_level=console_level,
            file_level=file_level
        )

    # Mock the setup_logging function in the module where it's used
    mocker.patch('meta_finder.logging_config.setup_logging', side_effect=patched_setup_logging)
    return patched_setup_logging  # Return for potential use in tests if needed


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary test directory for individual tests."""
    return tmp_path


@pytest.fixture
def sample_info_content():
    """Provide sample info_devices.json content for testing."""
    return {
        "i3": ["p1", 50.5, 1.2, "A", 54.5, 20.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600],
        "i4": ["p2", 51.5, 1.3, "B", 55.5, 21.1, "2019-11-08T12:00:00", "2019-11-08T13:00:0", 300, 600]
    }


@pytest.fixture
def create_test_device_structure(temp_test_dir):
    """Create a standard device directory structure for testing."""
    def _create_structure(device_name="230508_inclinometer@i03",
                         has_text_output=True,
                         has_info_file=True,
                         has_gpx=False):
        # Create device directory
        device_dir = temp_test_dir / "230507_ABP53_inclinometer" / device_name
        device_dir.mkdir(parents=True, exist_ok=True)

        # Create info_devices.json if requested
        if has_info_file:
            info_file = device_dir / "info_devices.json"
            if not info_file.exists():
                info_content = {
                    "i3": ["p1", 50.5, 1.2, "A", 54.5, 20.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600],
                    "i4": ["p2", 51.5, 1.3, "B", 55.5, 21.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]
                }
                import json
                info_file.write_text(json.dumps(info_content))

        # Create text_output directory if requested
        if has_text_output:
            text_output_dir = device_dir / "text_output"
            text_output_dir.mkdir(exist_ok=True)

            # Create a sample text file if it doesn't exist
            text_file = text_output_dir / "230508_1551bin2s@i03.tsv"
            if not text_file.exists():
                text_file.write_text("Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n2019-11-08 12:00:00\t0.1\t180\t0.05\t0.08\t5.2\t20.1\n2019-11-08 12:00:01\t0.15\t185\t0.06\t0.09\t5.3\t20.2")

            # Always create the combined data file if it doesn't exist (for tests that need it)
            combined_file = text_output_dir / "210618_180bin10s.tsv"
            if not combined_file.exists():
                combined_content = (
                    "Time\tVabs_i05_14\tVdir_i05_14\tv_i05_14\tTemp_i05_14\t\n"
                    "2023-06-18 18:00:00.000000\t1.0\t45.0\t0.5\t25.0\t\n"
                    "2023-06-18 18:10:00.00000\t1.1\t46.0\t0.6\t25.5\t\n"
                )
                combined_file.write_text(combined_content)

        # Create navigation directory with GPX if requested
        if has_gpx:
            nav_dir = device_dir.parent / "navigation"
            nav_dir.mkdir(exist_ok=True)
            gpx_file = nav_dir / "track.gpx"
            if not gpx_file.exists():
                gpx_file.write_text('<?xml version="1.0"?><gpx></gpx>')

        return device_dir

    return _create_structure

@pytest.fixture
def create_test_cruise_structure(temp_test_dir):
    """Create a standard cruise directory structure for testing."""
    def _create_structure(cruise_name="230507_ABP53_inclinometer", device_dirs=None):
        cruise_dir = temp_test_dir / cruise_name
        cruise_dir.mkdir(parents=True, exist_ok=True)

        # Create device directories if specified
        if device_dirs:
            for device_name in device_dirs:
                device_dir = cruise_dir / device_name
                device_dir.mkdir(exist_ok=True)

                # Create info_devices.json for each device directory
                info_file = device_dir / "info_devices.json"
                info_content = {
                    "i3": ["p1", 50.5, 1.2, "A", 54.5, 20.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]
                }
                import json
                info_file.write_text(json.dumps(info_content))

        return cruise_dir

    return _create_structure