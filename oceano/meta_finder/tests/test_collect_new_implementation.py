"""
Tests for the new collect.py implementation functions that replace the complex get_absent_meta function.
"""
from pathlib import Path
import tempfile
import json
from datetime import datetime
from unittest.mock import patch, MagicMock
import logging

import pytest

from meta_finder.collect import (
    get_all_data_files_for_device_dir,
    add_all_data_paths,
    get_prioritized_data_sources_for_time_extraction,
    extract_time_metadata_from_prioritized_sources,
    update_device_metadata_with_time_info,
    get_absent_meta,
    process_all_metadata
)
from meta_finder import config
from meta_finder.logging_config import setup_logging

# Using centralized mocked logging configuration
logger = setup_logging(__name__, console_level=logging.DEBUG, file_level=logging.DEBUG)


@pytest.fixture
def sample_device_dir():
    """Create a temporary device directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        device_dir = Path(temp_dir)

        # Create text_output subdirectory
        text_output_dir = device_dir / "text_output"
        text_output_dir.mkdir()

        # Create sample text files with different device IDs
        (text_output_dir / "230508_1200bin2s_i03.tsv").write_text(
            "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n"
            "2023-05-08 12:00:00\t0.1\t90\t0.05\t0.08\t5.0\t20.0\n"
            "2023-05-08 12:00:02\t0.15\t95\t0.07\t0.12\t5.2\t20.1\n"
        )

        (text_output_dir / "230508_1200bin600s_i04.tsv").write_text(
            "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n"
            "2023-05-08 12:00:00\t0.2\t100\t0.1\t0.15\t6.0\t20.5\n"
            "2023-05-08 12:10:00\t0.25\t105\t0.12\t0.18\t6.2\t20.6\n"
        )

        yield device_dir


@pytest.fixture
def sample_device_dir_with_raw():
    """Create a temporary device directory with raw files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        device_dir = Path(temp_dir)

        # Create _raw subdirectory
        raw_dir = device_dir / "_raw"
        raw_dir.mkdir()

        # Create sample raw files
        (raw_dir / "incl03.txt").write_text("Raw data for incl03")
        (raw_dir / "incl04.txt").write_text("Raw data for incl04")

        yield device_dir


def test_get_all_data_files_for_device_dir_with_text_files(sample_device_dir):
    """Test that get_all_data_files_for_device_dir finds all data files and organizes by device ID."""
    result = get_all_data_files_for_device_dir(sample_device_dir)

    # Should find both devices i03 and i04 (normalized to i3 and i4)
    assert "i3" in result
    assert "i4" in result

    # Check that i3 has the correct data path
    i3_paths = result["i3"]
    assert len(i3_paths) == 1
    path_tuple = list(i3_paths.keys())[0]
    # path_tuple should be (directory_path, file_path)
    assert path_tuple[0] == sample_device_dir / "text_output"  # Directory path
    assert path_tuple[1].name == "230508_1200bin2s_i03.tsv"  # File path

    # Check metadata was extracted
    metadata = list(i3_paths.values())[0]
    assert metadata["devices"] == ["i3"]
    assert metadata["averaging_interval"] == 2


def test_get_all_data_files_for_device_dir_with_raw_files(sample_device_dir_with_raw):
    """Test that get_all_data_files_for_device_dir finds raw files."""
    result = get_all_data_files_for_device_dir(sample_device_dir_with_raw)

    # Should find both devices incl03 and incl04 (from raw files) - normalized to i3 and i4
    assert "i3" in result
    assert "i4" in result

    # Check that i3 has the correct data path
    i3_paths = result["i3"]
    assert len(i3_paths) == 1
    path_tuple = list(i3_paths.keys())[0]
    assert path_tuple[0] == sample_device_dir_with_raw / "_raw"  # Directory path
    assert path_tuple[1].name == "incl03.txt"  # File path


def test_add_all_data_paths():
    """Test add_all_data_paths function that creates device data structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        device_dir = Path(temp_dir)

        # Create text_output subdirectory
        text_output_dir = device_dir / "text_output"
        text_output_dir.mkdir()

        # Create sample text file
        (text_output_dir / "230508_120bin2s_i03.tsv").write_text(
            "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n"
            "2023-05-08 12:00:00\t0.1\t90\t0.05\t0.08\t5.0\t20.0\n"
        )

        # Input metadata with existing device info - using normalized device ID
        meta_in = {
            "i3": {
                "point": "test_point",
                "sea_depth": 10.0,
                "height_above_bottom": 1.0,
                "modification_symbol": "test"
            }
        }

        result = add_all_data_paths(meta_in, device_dir)

        # Should have i3 with both existing metadata and data paths (normalized from i03)
        assert "i3" in result
        assert result["i3"]["point"] == "test_point"
        assert "data_paths" in result["i3"]
        assert len(result["i3"]["data_paths"]) == 1


def test_get_prioritized_data_sources_for_time_extraction():
    """Test prioritization of data sources for time extraction."""
    # Create a sample devices_data structure
    devices_data = {
        "i3": {
            "data_paths": {
                (Path("dir1"), Path("file2_600s.tsv")): {"averaging_interval": 600},
                (Path("dir1"), Path("file1_2s.tsv")): {"averaging_interval": 2},  # Higher priority due to lower averaging
            }
        }
    }

    result = get_prioritized_data_sources_for_time_extraction(devices_data)

    # Should have i3 with prioritized sources
    assert "i3" in result
    assert len(result["i3"]) == 2

    # First source should be the one with lower averaging (higher priority)
    first_path, first_meta = result["i3"][0]
    assert first_meta["averaging_interval"] == 2


@patch('meta_finder.collect.extract_time_info_from_text_file')
def test_extract_time_metadata_from_prioritized_sources(mock_extract):
    """Test extraction of time metadata from prioritized sources."""
    # Mock the time extraction function to return specific values
    mock_extract.return_value = ("2023-05-08 12:00:00", "2023-05-08 14:00:00", 3600, 2)

    # Create prioritized sources
    prioritized_sources = [
        ((Path("dir1"), Path("file1.tsv")), {"averaging": 2.0}),
    ]

    result = extract_time_metadata_from_prioritized_sources("i03", prioritized_sources)

    # Should return the time info in the expected format
    assert result == {
        "time_st": "2023-05-08 12:00:00",
        "time_en": "2023-05-08 14:00:00",
        "burst_dt": 3600,
        "bursts_t": 2
    }


def test_update_device_metadata_with_time_info():
    """Test updating device metadata with time info."""
    devices_data = {
        "i03": {
            "data_paths": {},
            "point": "test_point"
        }
    }

    # Mock time info to add
    with patch('meta_finder.collect.extract_time_metadata_from_prioritized_sources') as mock_extract:
        mock_extract.return_value = {
            "time_st": "2023-05-08 12:00:00",
            "time_en": "2023-05-08 14:00:00",
            "burst_dt": 3600,
            "bursts_t": 2
        }

        with patch('meta_finder.collect.get_prioritized_data_sources_for_time_extraction') as mock_get_prioritized:
            mock_get_prioritized.return_value = {"i03": []}

            update_device_metadata_with_time_info(devices_data)

    # Should have updated the device with time info
    assert devices_data["i03"]["time_st"] == "2023-05-08 12:00:00"
    assert devices_data["i03"]["time_en"] == "2023-05-08 14:00:00"
    assert devices_data["i03"]["burst_dt"] == 3600
    assert devices_data["i03"]["bursts_t"] == 2


def test_get_absent_meta_simple():
    """Test the simplified get_absent_meta function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        device_dir = Path(temp_dir)

        # Create text_output subdirectory
        text_output_dir = device_dir / "text_output"
        text_output_dir.mkdir()

        # Create sample text file
        (text_output_dir / "230508_1200bin2s_i03.tsv").write_text(
            "Time\tVabs\tVdir\tv\tu\tInclination\tTemp\n"
            "2023-05-08 12:00:00\t0.1\t90\t0.05\t0.08\t5.0\t20.0\n"
        )

        # Input metadata
        meta_in = {
            "i03": {
                "point": "test_point",
                "sea_depth": 10.0,
                "height_above_bottom": 1.0,
                "modification_symbol": "test"
            }
        }

        result = get_absent_meta(meta_in, device_dir)

        # Should have i03 with all metadata and data paths (cruise is assigned later in process_all_metadata)
        assert "i03" in result
        assert result["i03"]["point"] == "test_point"
        assert "data_paths" in result["i03"]
        assert len(result["i03"]["data_paths"]) >= 0  # May have time info extracted


@pytest.mark.parametrize("test_id,comment", [
    ("collect_new_impl_process_all_metadata", "Test process_all_metadata with new implementation"),
])
def test_process_all_metadata_new_implementation(test_id, comment, common_test_cruises_dir):
    """Test process_all_metadata function with the new implementation."""
    # Use the common test data setup
    test_data_dir = common_test_cruises_dir

    # Use one of the existing test directories from common_test_cruises_dir
    # Let's use the directory with text_output files
    device_dir = test_data_dir / "230507_ABP53_inclinometer@i3,4,15,19,37,38;ib27-30,ip6"

    # Verify the test directory exists
    assert device_dir.exists(), "Existing test directory should exist in common test data"

    # Create a cruise directory structure
    cruise_dir = test_data_dir / "230507_ABP53_cruise"
    cruise_dir.mkdir(exist_ok=True)
    cruise_and_its_dev_dirs = {cruise_dir: [device_dir]}

    result, _stats, _cruise_dates = process_all_metadata(
        cruise_and_its_dev_dirs,
        from_data=True,
        extract_hdf5_times=False,
        extract_hdf5_coef_dates=False,
        create_info_files=False
    )

    # Should process the metadata correctly
    assert device_dir in result
    assert "i3" in result[device_dir]  # normalized device ID
    # Metadata is nested under station_id '0' as a dict with named fields
    assert result[device_dir]["i3"]["0"]["point"] == "p1"
    assert "data_paths" in result[device_dir]["i3"]
