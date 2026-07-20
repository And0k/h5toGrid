"""Test metadata file search priority with info_devices.yaml support."""

import pytest
from pathlib import Path
from meta_finder.metadata_extractor import read_metadata_files, read_metadata_files_to_dict
from meta_finder.io_info_files import write_metadata_file, read_metadata_file


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary test directory with sample metadata files."""
    test_dir = tmp_path / "test_device"
    test_dir.mkdir()

    # Sample metadata content
    sample_metadata = {
        "i01": ["?", "?", "?", "?", 55.5, 12.3, "2024-01-01 00:00:00", "2024-01-01 01:00:00", "", "", ""],
        "i02": ["?", "?", "?", "?", 55.6, 12.4, "2024-01-01 02:00:00", "2024-01-01 03:00:00", "", "", ""],
    }

    # Different metadata for each file type to distinguish them
    metadata_json = {
        "i01": ["JSON", "?", "?", "?", 55.5, 12.3, "2024-01-01 00:00:00", "2024-01-01 01:00:00", "", "", ""],
        "i02": ["JSON", "?", "?", "?", 55.6, 12.4, "2024-01-01 02:00:00", "2024-01-01 03:00:00", "", "", ""],
    }

    metadata_yaml = {
        "i01": ["YAML", "?", "?", "?", 55.5, 12.3, "2024-01-01 00:00:00", "2024-01-01 01:00:00", "", "", ""],
        "i02": ["YAML", "?", "?", "?", 55.6, 12.4, "2024-01-01 02:00:00", "2024-01-01 03:00:00", "", "", ""],
    }

    metadata_upd = {
        "i01": ["UPD", "?", "?", "?", 55.5, 12.3, "2024-01-01 00:00:00", "2024-01-01 01:00:00", "", "", ""],
        "i02": ["UPD", "?", "?", "?", 55.6, 12.4, "2024-01-01 02:00:00", "2024-01-01 03:00:00", "", "", ""],
    }

    return test_dir, metadata_json, metadata_yaml, metadata_upd


def test_priority_yaml_only(temp_test_dir):
    """Test that info_devices.yaml is used when only it exists."""
    test_dir, metadata_json, metadata_yaml, metadata_upd = temp_test_dir

    # Only create YAML file
    yaml_path = test_dir / "info_devices.yaml"
    write_metadata_file(test_dir, yaml_path, metadata_yaml)

    # Read metadata
    json_path = test_dir / "info_devices.json"
    result = read_metadata_files(json_path)

    # Verify YAML content was loaded (nested dict: device_id -> station_id -> list)
    assert result is not None
    assert len(result) == 2
    assert result["i01"]["0"][0] == "YAML"
    assert result["i02"]["0"][0] == "YAML"


def test_priority_json_only(temp_test_dir):
    """Test that info_devices.json is used when only it exists."""
    test_dir, metadata_json, metadata_yaml, metadata_upd = temp_test_dir

    # Only create JSON file
    json_path = test_dir / "info_devices.json"
    write_metadata_file(test_dir, json_path, metadata_json)

    # Read metadata
    result = read_metadata_files(json_path)

    # Verify JSON content was loaded (nested dict: device_id -> station_id -> list)
    assert result is not None
    assert len(result) == 2
    assert result["i01"]["0"][0] == "JSON"
    assert result["i02"]["0"][0] == "JSON"


def test_priority_yaml_over_json(temp_test_dir):
    """Test that info_devices.yaml takes priority over info_devices.json when both exist."""
    test_dir, metadata_json, metadata_yaml, metadata_upd = temp_test_dir

    # Create both JSON and YAML files
    json_path = test_dir / "info_devices.json"
    write_metadata_file(test_dir, json_path, metadata_json)

    yaml_path = test_dir / "info_devices.yaml"
    write_metadata_file(test_dir, yaml_path, metadata_yaml)

    # Read metadata
    result = read_metadata_files(json_path)

    # Verify YAML content was loaded (higher priority than JSON)
    assert result is not None
    assert len(result) == 2
    assert result["i01"]["0"][0] == "YAML", "YAML should take priority over JSON"
    assert result["i02"]["0"][0] == "YAML", "YAML should take priority over JSON"


def test_priority_upd_highest(temp_test_dir):
    """Test that info_devices@meta_finder.yaml has highest priority."""
    test_dir, metadata_json, metadata_yaml, metadata_upd = temp_test_dir

    # Create all three files
    json_path = test_dir / "info_devices.json"
    write_metadata_file(test_dir, json_path, metadata_json)

    yaml_path = test_dir / "info_devices.yaml"
    write_metadata_file(test_dir, yaml_path, metadata_yaml)

    upd_path = test_dir / "info_devices@meta_finder.yaml"
    write_metadata_file(test_dir, upd_path, metadata_upd)

    # Read metadata
    result = read_metadata_files(json_path)

    # Verify UPD content was loaded (highest priority)
    assert result is not None
    assert len(result) == 2
    assert result["i01"]["0"][0] == "UPD", "UPD should have highest priority"
    assert result["i02"]["0"][0] == "UPD", "UPD should have highest priority"


def test_priority_upd_over_yaml(temp_test_dir):
    """Test that info_devices@meta_finder.yaml takes priority over info_devices.yaml."""
    test_dir, metadata_json, metadata_yaml, metadata_upd = temp_test_dir

    # Create only UPD and YAML files (no JSON)
    yaml_path = test_dir / "info_devices.yaml"
    write_metadata_file(test_dir, yaml_path, metadata_yaml)

    upd_path = test_dir / "info_devices@meta_finder.yaml"
    write_metadata_file(test_dir, upd_path, metadata_upd)

    # Read metadata
    json_path = test_dir / "info_devices.json"
    result = read_metadata_files(json_path)

    # Verify UPD content was loaded (higher priority than YAML)
    assert result is not None
    assert len(result) == 2
    assert result["i01"]["0"][0] == "UPD", "UPD should take priority over YAML"
    assert result["i02"]["0"][0] == "UPD", "UPD should take priority over YAML"


def test_no_metadata_files(temp_test_dir):
    """Test that empty dict is returned when no metadata files exist."""
    test_dir, metadata_json, metadata_yaml, metadata_upd = temp_test_dir

    # Don't create any files
    json_path = test_dir / "info_devices.json"
    result = read_metadata_files(json_path)

    # Verify empty dict is returned
    assert result == {}


def test_yaml_file_reading_directly(temp_test_dir):
    """Test that YAML files can be read directly using read_metadata_file."""
    test_dir, metadata_json, metadata_yaml, metadata_upd = temp_test_dir

    # Create YAML file
    yaml_path = test_dir / "info_devices.yaml"
    write_metadata_file(test_dir, yaml_path, metadata_yaml)

    # Read directly
    result = read_metadata_file(yaml_path)

    # Verify content (nested dict: device_id -> station_id -> list)
    assert result is not None
    assert len(result) == 2
    assert result["i01"]["0"][0] == "YAML"
    assert result["i02"]["0"][0] == "YAML"


class TestSpacePrefixedDeviceFiltering:
    """Test that device IDs starting with a space are skipped on load from info files."""

    @pytest.fixture
    def metadata_with_space_prefixed(self):
        """Metadata containing a space-prefixed device ID alongside valid ones."""
        return {
            "i01": ["?", "?", "?", "?", 55.5, 12.3, "2024-01-01 00:00:00", "2024-01-01 01:00:00", "", "", ""],
            " i02": ["?", "?", "?", "?", 55.6, 12.4, "2024-01-01 02:00:00", "2024-01-01 03:00:00", "", "", ""],
            "i03": ["?", "?", "?", "?", 55.7, 12.5, "2024-01-01 04:00:00", "2024-01-01 05:00:00", "", "", ""],
        }

    @pytest.mark.parametrize(
        "file_ext,format_label",
        [(".yaml", "YAML"), (".json", "JSON")],
        ids=["yaml-format", "json-format"],
    )
    def test_read_metadata_file_skips_space_prefixed_devices(
        self, tmp_path, metadata_with_space_prefixed, file_ext, format_label
    ):
        """read_metadata_file should exclude device IDs that start with a space."""
        info_file = tmp_path / f"info_devices{file_ext}"
        write_metadata_file(tmp_path, info_file, metadata_with_space_prefixed)

        result = read_metadata_file(info_file)

        assert " i02" not in result, (
            f"Space-prefixed device ' i02' must not appear in result from {format_label} file"
        )
        assert len(result) == 2, (
            f"Expected 2 devices (i01, i03) after filtering space-prefixed from {format_label}, got {len(result)}"
        )
        assert "i01" in result and "i03" in result, (
            f"Non-space-prefixed devices must be preserved in {format_label} output"
        )

    @pytest.mark.parametrize(
        "file_ext,format_label",
        [(".yaml", "YAML"), (".json", "JSON")],
        ids=["yaml-format", "json-format"],
    )
    def test_read_metadata_files_to_dict_skips_space_prefixed_devices(
        self, tmp_path, metadata_with_space_prefixed, file_ext, format_label
    ):
        """read_metadata_files_to_dict should also exclude space-prefixed device IDs."""
        info_file = tmp_path / f"info_devices{file_ext}"
        write_metadata_file(tmp_path, info_file, metadata_with_space_prefixed)

        result = read_metadata_files_to_dict(info_file)

        assert " i02" not in result, (
            f"Space-prefixed device ' i02' must not appear in dict result from {format_label} file"
        )
        assert len(result) == 2, (
            f"Expected 2 devices after filtering from {format_label}, got {len(result)}"
        )

    def test_all_space_prefixed_devices_returns_empty(self, tmp_path):
        """When all device IDs are space-prefixed, result should be empty."""
        all_space_prefixed = {
            " i01": ["?", "?", "?", "?", 55.5, 12.3, "", "", "", "", ""],
            " i02": ["?", "?", "?", "?", 55.6, 12.4, "", "", "", "", ""],
        }
        info_file = tmp_path / "info_devices.yaml"
        write_metadata_file(tmp_path, info_file, all_space_prefixed)

        result = read_metadata_file(info_file)

        assert len(result) == 0, "All space-prefixed devices should result in empty dict"

    def test_no_space_prefixed_devices_unchanged(self, tmp_path):
        """When no device IDs are space-prefixed, all should be present."""
        normal_devices = {
            "i01": ["?", "?", "?", "?", 55.5, 12.3, "", "", "", "", ""],
            "i02": ["?", "?", "?", "?", 55.6, 12.4, "", "", "", "", ""],
        }
        info_file = tmp_path / "info_devices.yaml"
        write_metadata_file(tmp_path, info_file, normal_devices)

        result = read_metadata_file(info_file)

        assert len(result) == 2, "Normal devices should all be present"
        assert "i01" in result and "i02" in result
