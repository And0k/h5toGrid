"""
Test that info_devices.yaml content is preserved when creating info_devices@meta_finder.yaml
"""

from pathlib import Path
import pytest
from ruamel.yaml import YAML
import tempfile


from meta_finder.create_info_files import update_devices_meta_file
from meta_finder.io_info_files import (
    read_metadata_file,
    _concatenate_additional_comments,
    _remove_trailing_empty_fields,
)
yaml = YAML(typ='full', pure=True)


def _simulate_write_read(values: list) -> list:
    """Simulate write-then-read pipeline: trim then concatenate comments."""
    trimmed = list(_remove_trailing_empty_fields(values))
    return _concatenate_additional_comments(trimmed)


def test_yaml_content_preserved_when_creating_upd_file():
    """When info_devices.yaml exists but @meta_finder.yaml doesn't,
    YAML content is preserved in the newly created file.

    Note: _merge_device_metadata with normalize_keys=True normalizes existing YAML
    device IDs (e.g. i01 → i1). New content must use normalized IDs to match.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        device_dir = Path(temp_dir) / "test_device"
        device_dir.mkdir()

        yaml_file = device_dir / "info_devices.yaml"
        yaml_content = {
            "i01": [
                "point1", "depth1", "height1", "mod1", "lat1", "lon1",
                "2023-01-01T00:00:00", "2023-01-01T01:00:00", "300", "600", "comment1",
            ],
            "i02": [
                "point2", "depth2", "height2", "mod2", "lat2", "lon2",
                "2023-02-01T00:00:00", "2023-02-01T01:00:00", "400", "800", "comment2",
            ],
        }
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)

        # Use normalized device IDs (i1, i3) so merge can match existing YAML IDs
        new_content = {
            "i1": {
                "point": "?", "sea_depth": "?",
                "height_above_bottom": "?", "modification_symbol": "?",
                "lat": "?", "lon": "?",
                "time_st": "2023-01-01T00:00:00", "time_en": "2023-01-01T01:00:00",
                "burst_dt": "?", "bursts_t": "?", "comment": "?",
            },
            "i3": {
                "point": "point3", "sea_depth": "depth3",
                "height_above_bottom": "height3", "modification_symbol": "mod3",
                "lat": "lat3", "lon": "lon3",
                "time_st": "2023-03-01T00:00:00", "time_en": "2023-03-01T01:00:00",
                "burst_dt": "?", "bursts_t": "?", "comment": "comment3",
            },
        }

        result = update_devices_meta_file(device_dir=device_dir, content=new_content)

        assert result is True, "Expected update to succeed"

        upd_file = device_dir / "info_devices@meta_finder.yaml"
        assert upd_file.exists(), "Expected info_devices@meta_finder.yaml to be created"

        created_content = read_metadata_file(upd_file)

        # i01 in YAML normalizes to i1; existing nested dict preserved over new flat list
        assert "i1" in created_content, "i1 (from i01 in YAML) should be in created file"
        expected_i1 = _simulate_write_read(yaml_content["i01"])
        assert created_content["i1"]["0"] == expected_i1, (
            "i1 should preserve its original data from YAML file"
        )

        # i02 in YAML normalizes to i2; not in new content, preserved as-is
        assert "i2" in created_content, "i2 (from i02 in YAML) should be in created file"
        expected_i2 = _simulate_write_read(yaml_content["i02"])
        assert created_content["i2"]["0"] == expected_i2, (
            "i2 should preserve its original data from YAML file"
        )

        # i3: new device, added from new content
        assert "i3" in created_content, "i3 should be added from new content"
        expected_i3 = _simulate_write_read([
            "point3", "depth3", "height3", "mod3", "lat3", "lon3",
            "2023-03-01T00:00:00", "2023-03-01T01:00:00", "?", "?", "comment3",
        ])
        assert created_content["i3"]["0"] == expected_i3, (
            "i3 should match expected values including comment"
        )

        # Verify that info_devices.yaml was not modified
        original_yaml = read_metadata_file(yaml_file)
        for device_id in yaml_content:
            assert device_id in original_yaml, f"{device_id} should still be in original YAML"
            expected_original = _simulate_write_read(yaml_content[device_id])
            assert original_yaml[device_id]["0"] == expected_original, (
                f"{device_id} content should not be modified in original YAML"
            )


def test_yaml_content_merged_selectively():
    """When both info_devices.yaml and new content have data for the same device,
    existing non-placeholder values are preserved (nested dict takes precedence).

    Note: _merge_device_metadata with normalize_keys=True normalizes existing YAML
    device IDs. New content uses normalized IDs to match.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        device_dir = Path(temp_dir) / "test_device"
        device_dir.mkdir()

        yaml_file = device_dir / "info_devices.yaml"
        yaml_content = {
            "i01": [
                "point1", "depth1", "height1", "mod1", "lat1", "lon1",
                "2023-01-01T00:00:00", "2023-01-01T01:00:00", "300", "600", "comment1",
            ],
            "i02": [
                "?", "?", "?", "?", "?", "?",
                "2023-02-01T00:00:00", "2023-02-01T01:00:00", "?", "?",
            ],
        }
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)

        # Use normalized device IDs (i1, i2) so merge can match existing YAML IDs
        new_content = {
            "i1": {
                "point": "?", "sea_depth": "?",
                "height_above_bottom": "?", "modification_symbol": "?",
                "lat": "?", "lon": "?",
                "time_st": "?", "time_en": "?",
                "burst_dt": "?", "bursts_t": "?", "comment": "?",
            },
            "i2": {
                "point": "point2", "sea_depth": "depth2",
                "height_above_bottom": "height2", "modification_symbol": "mod2",
                "lat": "lat2", "lon": "lon2",
                "time_st": "2023-02-01T00:00:00", "time_en": "2023-02-01T01:00:00",
                "burst_dt": "400", "bursts_t": "800", "comment": "comment2",
            },
        }

        result = update_devices_meta_file(device_dir=device_dir, content=new_content)

        assert result is True, "Expected update to succeed"

        upd_file = device_dir / "info_devices@meta_finder.yaml"
        created_content = read_metadata_file(upd_file)

        # i01 in YAML normalizes to i1; existing nested dict preserved over new flat list
        expected_i1 = _simulate_write_read(yaml_content["i01"])
        assert created_content["i1"]["0"] == expected_i1, (
            "i1 should preserve its original non-placeholder data from YAML"
        )

        # i02 in YAML normalizes to i2; existing has placeholder point/depth, new has real values.
        # Merge fills placeholders with new values, keeps existing non-placeholders (time_st, time_en).
        expected_i2 = _simulate_write_read([
            "point2", "depth2", "height2", "mod2", "lat2", "lon2",
            "2023-02-01T00:00:00", "2023-02-01T01:00:00", "400", "800", "comment2",
        ])
        assert created_content["i2"]["0"] == expected_i2, (
            "i2 should have merged data: new values fill placeholders, existing non-placeholders kept"
        )
