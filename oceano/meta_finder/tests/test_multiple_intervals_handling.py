"""Tests for handling multiple intervals for the same device.

This test suite verifies that devices with multiple time intervals
(represented as nested dicts with station-ID keys like "0", "1") are
correctly handled and written as separate rows in TSV output.
"""
import pytest
from pathlib import Path
from typing import Dict, Any

from meta_finder.io_info_files import (
    info_devices_field_names_extended,
)
from meta_finder.file_writer import write_metadata_table


def _make_station_meta(
    point="", sea_depth=85, height_above_bottom=0, modification_symbol="⭡",
    lat=55.8940, lon=19.0899, time_st="", time_en="",
    burst_dt="", bursts_t="", comment="",
) -> dict:
    """Build a field-name dict for a single station/interval."""
    return {
        field_name: value
        for field_name, value in zip(
            info_devices_field_names_extended[:11],
            [point, sea_depth, height_above_bottom, modification_symbol,
             lat, lon, time_st, time_en, burst_dt, bursts_t, comment],
        )
    }


@pytest.fixture
def sample_metadata_dict_with_intervals():
    """Sample metadata dict with station-ID-keyed nested structure.

    Devices use the current format: ``{device_id: {"0": {...}, "1": {...}, "setup_name": ...}}``.
    """
    return {
        "i10": {
            "0": _make_station_meta(
                time_st="2018-10-17 16:30:00", time_en="2018-10-18 07:15:00",
            ),
            "1": _make_station_meta(
                time_st="2018-10-22 12:03:00", time_en="2018-10-27 06:47:28",
            ),
            "setup_name": "ABP44",
            "data_paths": {},
        },
        "i05": {
            "0": _make_station_meta(
                sea_depth=90, height_above_bottom=5, lat=56.1234, lon=20.5678,
                time_st="2018-10-15 08:00:00", time_en="2018-10-16 18:30:00",
            ),
            "setup_name": "ABP44",
            "data_paths": {},
        },
        "i07": {
            "point": "",
            "sea_depth": 80,
            "height_above_bottom": 10,
            "modification_symbol": "⭡",
            "lat": 57.3456,
            "lon": 21.2345,
            "time_st": "2018-10-20 10:00:00",
            "time_en": "2018-10-25 15:00:00",
            "burst_dt": "",
            "bursts_t": "",
            "comment": "",
            "setup_name": "ABP44",
            "data_paths": {},
        },
    }


class TestWriteMetadataTableWithMultipleIntervals:
    """Test the write_metadata_table function with multiple intervals."""

    def test_creates_multiple_rows_for_device_with_intervals(
        self, sample_metadata_dict_with_intervals, tmp_path
    ):
        """Test that device with intervals creates multiple rows in TSV."""
        dev_dir_meta = {Path("/test/device_dir"): sample_metadata_dict_with_intervals}
        cruise_date_map = {"ABP44": "181017"}

        write_metadata_table(dev_dir_meta, tmp_path / "meta_TCM.tsv", cruise_date_map)

        content = (tmp_path / "meta_TCM.tsv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        # Check header
        header = lines[0].split("\t")
        assert "device_id" in header, "Header should contain device_id"
        assert "time_st" in header, "Header should contain time_st"
        assert "time_en" in header, "Header should contain time_en"

        # Count rows for i10 device
        i10_rows = [line for line in lines[1:] if line.strip() and "i10" in line.split("\t")[1]]
        assert len(i10_rows) == 2, f"Should have 2 rows for i10 device, got {len(i10_rows)}"

        # Check first interval row
        first_row = i10_rows[0].split("\t")
        device_id_idx = header.index("device_id")
        time_st_idx = header.index("time_st")
        time_en_idx = header.index("time_en")

        assert (
            first_row[device_id_idx] == "i10",
            f"First row should have device_id=i10, got {first_row[device_id_idx]}",
        )
        assert (
            first_row[time_st_idx] == "2018-10-17 16:30:00",
            f"First row should have time_st=2018-10-17 16:30:00, got {first_row[time_st_idx]}",
        )
        assert (
            first_row[time_en_idx] == "2018-10-18 07:15:00",
            f"First row should have time_en=2018-10-18 07:15:00, got {first_row[time_en_idx]}",
        )

        # Check second interval row
        second_row = i10_rows[1].split("\t")
        assert (
            second_row[device_id_idx] == "i10",
            f"Second row should have device_id=i10, got {second_row[device_id_idx]}",
        )
        assert (
            second_row[time_st_idx] == "2018-10-22 12:03:00",
            f"Second row should have time_st=2018-10-22 12:03:00, got {second_row[time_st_idx]}",
        )
        assert (
            second_row[time_en_idx] == "2018-10-27 06:47:28",
            f"Second row should have time_en=2018-10-27 06:47:28, got {second_row[time_en_idx]}",
        )

    def test_single_interval_device_creates_one_row(
        self, sample_metadata_dict_with_intervals, tmp_path
    ):
        """Test that single interval device creates one row in TSV."""
        dev_dir_meta = {Path("/test/device_dir"): sample_metadata_dict_with_intervals}
        cruise_date_map = {"ABP44": "181017"}

        write_metadata_table(dev_dir_meta, tmp_path / "meta_TCM.tsv", cruise_date_map)

        content = (tmp_path / "meta_TCM.tsv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        header = lines[0].split("\t")

        # Count rows for i07 device (single interval)
        i07_rows = [line for line in lines[1:] if line.strip() and "i07" in line.split("\t")[1]]
        assert len(i07_rows) == 1, f"Should have 1 row for i07 device, got {len(i07_rows)}"

        # Check row values
        row = i07_rows[0].split("\t")
        device_id_idx = header.index("device_id")
        time_st_idx = header.index("time_st")
        time_en_idx = header.index("time_en")

        assert (
            row[device_id_idx] == "i07",
            f"Row should have device_id=i07, got {row[device_id_idx]}",
        )
        assert (
            row[time_st_idx] == "2018-10-20 10:00:00",
            f"Row should have time_st=2018-10-20 10:00:00, got {row[time_st_idx]}",
        )
        assert (
            row[time_en_idx] == "2018-10-25 15:00:00",
            f"Row should have time_en=2018-10-25 15:00:00, got {row[time_en_idx]}",
        )

    def test_non_time_fields_preserved_across_intervals(
        self, sample_metadata_dict_with_intervals, tmp_path
    ):
        """Test that non-time fields are preserved across all interval rows."""
        dev_dir_meta = {Path("/test/device_dir"): sample_metadata_dict_with_intervals}
        cruise_date_map = {"ABP44": "181017"}

        write_metadata_table(dev_dir_meta, tmp_path / "meta_TCM.tsv", cruise_date_map)

        content = (tmp_path / "meta_TCM.tsv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        header = lines[0].split("\t")

        sea_depth_idx = header.index("sea_depth")
        lat_idx = header.index("lat")
        lon_idx = header.index("lon")

        # Check both i10 rows have same non-time field values
        i10_rows = [line for line in lines[1:] if line.strip() and "i10" in line.split("\t")[1]]

        for row_str in i10_rows:
            row = row_str.split("\t")
            assert (
                row[sea_depth_idx] == "85",
                f"Row should have sea_depth=85, got {row[sea_depth_idx]}",
            )
            assert (
                row[lat_idx] == "55.894",
                f"Row should have lat=55.894, got {row[lat_idx]}",
            )
            assert (
                row[lon_idx] == "19.0899",
                f"Row should have lon=19.0899, got {row[lon_idx]}",
            )

    def test_interval_index_field_present(self, sample_metadata_dict_with_intervals, tmp_path):
        """Test that multi-interval devices produce multiple rows with distinct time values."""
        dev_dir_meta = {Path("/test/device_dir"): sample_metadata_dict_with_intervals}
        cruise_date_map = {"ABP44": "181017"}

        write_metadata_table(dev_dir_meta, tmp_path / "meta_TCM.tsv", cruise_date_map)

        content = (tmp_path / "meta_TCM.tsv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        header = lines[0].split("\t")

        time_st_idx = header.index("time_st")
        time_en_idx = header.index("time_en")

        # Check i10 rows have distinct time values (proves intervals are separate rows)
        i10_rows = [line for line in lines[1:] if line.strip() and "i10" in line.split("\t")[1]]
        assert len(i10_rows) == 2, f"i10 should have 2 rows, got {len(i10_rows)}"

        first_ts = i10_rows[0].split("\t")[time_st_idx]
        second_ts = i10_rows[1].split("\t")[time_st_idx]
        assert first_ts != second_ts, (
            f"Two i10 intervals should have different time_st values, got {first_ts!r} and {second_ts!r}"
        )

        # Check single interval device i07 has exactly 1 row
        i07_rows = [line for line in lines[1:] if line.strip() and "i07" in line.split("\t")[1]]
        assert len(i07_rows) == 1, "i07 should have 1 row"
