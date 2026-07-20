"""Verify that info_devices@meta_finder.yaml is not recorded when content equals info_devices.yaml.

Tests two aspects:
1. Existing test data: @meta_finder.yaml files must not duplicate info_devices.yaml
   (after normalizing device IDs and comparing station data).
2. Functional test: update_devices_meta_file() must not write @meta_finder.yaml
   when the merged result is identical to info_devices.yaml content.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from meta_finder.parse_data_file_name import normalize_device_id
from meta_finder.io_info_files import read_metadata_file, _station_metadata_to_list, iter_station_id_items
from meta_finder.create_info_files import _content_data_equals, update_devices_meta_file

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data" / "Cruises"


def _collect_paired_dirs():
    """Find all device dirs that have both info_devices.yaml and info_devices@meta_finder.yaml."""
    pairs = []
    for meta_file in sorted(TEST_DATA_DIR.rglob("info_devices@meta_finder.yaml")):
        yaml_file = meta_file.parent / "info_devices.yaml"
        if yaml_file.exists():
            pairs.append((meta_file, yaml_file))
    return pairs


def _normalize_content(content: dict) -> dict:
    """Build {normalized_device_id: {station_id: [values]}} from raw file content."""
    normalized = {}
    for dev_id, device_entry in content.items():
        norm_id = normalize_device_id(dev_id)
        if norm_id is None:
            norm_id = dev_id
        stations = {}
        if isinstance(device_entry, dict):
            for sid, meta in iter_station_id_items(device_entry):
                stations[sid] = _station_metadata_to_list(meta)
        elif isinstance(device_entry, (list, tuple)):
            stations["0"] = list(device_entry)
        normalized[norm_id] = stations
    return normalized


def _values_equal(a: list, b: list) -> bool:
    """Compare two metadata value lists, treating string/int/float coercion loosely."""
    if len(a) != len(b):
        return False
    return all(str(va) == str(vb) for va, vb in zip(a, b))


PAIRED_DIRS = _collect_paired_dirs()


@pytest.mark.parametrize(
    ("meta_file", "yaml_file"),
    [pytest.param(mf, yf, id=str(mf.relative_to(TEST_DATA_DIR))) for mf, yf in PAIRED_DIRS],
)
def test_meta_finder_does_not_duplicate_yaml_content(meta_file: Path, yaml_file: Path):
    """Each @meta_finder.yaml must not have identical station data for any normalized device_id."""
    yaml_content = read_metadata_file(yaml_file)
    meta_content = read_metadata_file(meta_file)

    norm_yaml = _normalize_content(yaml_content)
    norm_meta = _normalize_content(meta_content)

    duplicates = []
    for dev_id, meta_stations in norm_meta.items():
        if dev_id not in norm_yaml:
            continue
        yaml_stations = norm_yaml[dev_id]
        for sid, meta_vals in meta_stations.items():
            if sid in yaml_stations and _values_equal(meta_vals, yaml_stations[sid]):
                duplicates.append((dev_id, sid))

    assert not duplicates, (
        f"{meta_file.relative_to(TEST_DATA_DIR.parent)}: "
        f"Duplicate entries found (same normalized device_id and identical data): {duplicates}"
    )


class TestContentDataEquals:
    """Unit tests for _content_data_equals comparing merged vs source content."""

    def test_identical_flat_lists(self):
        """Same device with identical flat list values must be equal."""
        a = {"i3": {"0": ["p1", 50.5, 1.2, "A", 54.5, 20.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]}}
        b = {"i3": ["p1", 50.5, 1.2, "A", 54.5, 20.1, "2019-11-08T12:00:00", "2019-11-08T13:00:00", 300, 600]}
        assert _content_data_equals(a, b, normalize_keys=True), "Flat list vs nested station '0' should be equal"

    def test_different_values(self):
        """Same device with different values must not be equal."""
        a = {"i3": {"0": ["p1", 50.5]}}
        b = {"i3": ["p2", 51.5]}
        assert not _content_data_equals(a, b, normalize_keys=True), "Different values should not be equal"

    def test_extra_device_in_merged(self):
        """Merged content with an extra device must not be equal to source."""
        a = {"i3": ["p1", 50.5], "i01": ["point1", "depth1"]}
        b = {"i3": ["p1", 50.5]}
        assert not _content_data_equals(a, b, normalize_keys=True), "Extra device should make them different"

    def test_normalized_key_match(self):
        """Non-normalized key in source must match normalized key in merged."""
        a = {"i03": ["p1", 50.5]}
        b = {"i3": ["p1", 50.5]}
        assert _content_data_equals(a, b, normalize_keys=True), "i03 normalized to i3 should match"

    def test_empty_dicts_equal(self):
        assert _content_data_equals({}, {}, normalize_keys=True)


class TestUpdateDevicesMetaFileSkipsDuplicate:
    """Functional test: update_devices_meta_file must not write when merged equals info_devices.yaml."""

    def test_skips_write_when_content_same_as_yaml(self, tmp_path):
        """When new content merges to same data as info_devices.yaml, no file is written."""
        # Create info_devices.yaml with device data
        yaml_content = "i3: [p1, 50.5, 1.2, A, 54.5, 20.1, '2019-11-08T12:00:00', '2019-11-08T13:00:00', 300, 600]\n"
        (tmp_path / "info_devices.yaml").write_text(yaml_content, encoding="utf-8")

        # New content that would merge to the same data (all placeholders, so existing is kept)
        new_content = {
            "i3": {
                "point": "?", "sea_depth": "?", "height_above_bottom": "?",
                "modification_symbol": "?", "lat": "?", "lon": "?",
                "time_st": "?", "time_en": "?", "burst_dt": "?", "bursts_t": "?", "comment": "?",
            }
        }

        with patch("meta_finder.config.overwrite_bad_devs_in_info_files", True):
            result = update_devices_meta_file(tmp_path, new_content)

        assert not result, "Should return False when merged content equals info_devices.yaml"
        assert not (tmp_path / "info_devices@meta_finder.yaml").exists(), (
            "File should not be created when content unchanged"
        )

    def test_writes_when_content_differs_from_yaml(self, tmp_path):
        """When new content adds data not in info_devices.yaml, file is written."""
        yaml_content = "i3: ['?', '?', '?', '?', '?', '?', '2019-11-08T12:00:00', '2019-11-08T13:00:00', 300, 600]\n"
        (tmp_path / "info_devices.yaml").write_text(yaml_content, encoding="utf-8")

        # New content with real point/depth data that fills placeholders
        new_content = {
            "i3": {
                "point": "p1", "sea_depth": 50.5, "height_above_bottom": 1.2,
                "modification_symbol": "A", "lat": 54.5, "lon": 20.1,
                "time_st": "?", "time_en": "?", "burst_dt": "?", "bursts_t": "?", "comment": "?",
            }
        }

        with patch("meta_finder.config.overwrite_bad_devs_in_info_files", True):
            result = update_devices_meta_file(tmp_path, new_content)

        assert result, "Should return True when merged content differs from info_devices.yaml"
        assert (tmp_path / "info_devices@meta_finder.yaml").exists(), "File should be created with new data"

    def test_deletes_preexisting_duplicate_meta_finder(self, tmp_path):
        """Pre-existing @meta_finder.yaml identical to info_devices.yaml must be deleted."""
        yaml_text = (
            "# Instrument_ID: [Point, Sea_depth, H_above_bot, Symbol, Lat, Lon,"
            " Time_st, Time_en, Burst_dt, Bursts_t, Comment]\n"
            "i3: [p1, 50.5, 1.2, A, 54.5, 20.1, '2019-11-08T12:00:00',"
            " '2019-11-08T13:00:00', 300, 600]\n"
        )
        (tmp_path / "info_devices.yaml").write_text(yaml_text, encoding="utf-8")
        # Create @meta_finder with identical data (same normalized device i3, same values)
        meta_text = (
            "# Instrument_ID: [Point, Sea_depth, H_above_bot, Symbol, Lat, Lon,"
            " Time_st, Time_en, Burst_dt, Bursts_t, Comment]\n"
            "i3: [p1, 50.5, 1.2, A, 54.5, 20.1, '2019-11-08T12:00:00',"
            " '2019-11-08T13:00:00', 300, 600]\n"
        )
        (tmp_path / "info_devices@meta_finder.yaml").write_text(meta_text, encoding="utf-8")

        # New content with placeholders — merged result equals existing yaml
        new_content = {
            "i3": {
                "point": "?", "sea_depth": "?", "height_above_bottom": "?",
                "modification_symbol": "?", "lat": "?", "lon": "?",
                "time_st": "?", "time_en": "?", "burst_dt": "?", "bursts_t": "?", "comment": "?",
            }
        }

        with patch("meta_finder.config.overwrite_bad_devs_in_info_files", True):
            result = update_devices_meta_file(tmp_path, new_content)

        assert not result, "Should return False when merged content equals source"
        assert not (tmp_path / "info_devices@meta_finder.yaml").exists(), (
            "Pre-existing @meta_finder.yaml identical to info_devices.yaml should be deleted"
        )

    def test_keeps_meta_finder_when_no_yaml_exists(self, tmp_path):
        """Pre-existing @meta_finder.yaml must NOT be deleted when no info_devices.yaml exists."""
        # No info_devices.yaml — @meta_finder is the authoritative source
        meta_text = (
            "# Instrument_ID: [Point, Sea_depth, H_above_bot, Symbol, Lat, Lon,"
            " Time_st, Time_en, Burst_dt, Bursts_t, Comment]\n"
            "i3: [p1, 50.5, 1.2, A, 54.5, 20.1, '2019-11-08T12:00:00',"
            " '2019-11-08T13:00:00', 300, 600]\n"
        )
        (tmp_path / "info_devices@meta_finder.yaml").write_text(meta_text, encoding="utf-8")

        # New content with placeholders — merge keeps existing @meta_finder data
        new_content = {
            "i3": {
                "point": "?", "sea_depth": "?", "height_above_bottom": "?",
                "modification_symbol": "?", "lat": "?", "lon": "?",
                "time_st": "?", "time_en": "?", "burst_dt": "?", "bursts_t": "?", "comment": "?",
            }
        }

        with patch("meta_finder.config.overwrite_bad_devs_in_info_files", True):
            result = update_devices_meta_file(tmp_path, new_content)

        assert not result, "Should return False when merged content unchanged"
        assert (tmp_path / "info_devices@meta_finder.yaml").exists(), (
            "Pre-existing @meta_finder.yaml must NOT be deleted when no info_devices.yaml exists"
        )
