"""Test that data_paths and HDF5-derived fields are written to meta_TCM.tsv.

Verifies:
- _process_device_entry passes data_paths via kwargs for single-interval devices
  (no nested station structure).
- coef_date, time_raw_st, time_raw_en columns are present in the header and
  their values are faithfully written from metadata to the output TSV.
- For multi-interval devices, HDF5 fields set at device level (not inside
  station dicts) are propagated to every station row via fallback.

When no info_devices@meta_finder.yaml or info_devices.yaml exists, `get_absent_meta`
discovers devices from data files and populates data_paths correctly.
The `write_metadata_table` should write these paths and HDF5 fields to the output.
"""

from pathlib import Path, PurePosixPath

import pytest

from meta_finder.file_writer import write_metadata_table
from meta_finder.io_info_files import info_devices_field_names_extended


@pytest.fixture
def single_interval_device_meta():
    """Device metadata with data_paths but no nested station structure.

    Simulates the output of get_absent_meta for a device_dir with no info file,
    where devices are discovered from data files and resulting metadata is flat dict with
    data_paths at the top level (not nested under station IDs keys).
    """
    return {
        "i03": {
            "point": "?",
            "sea_depth": "?",
            "height_above_bottom": "?",
            "modification_symbol": "?",
            "lat": "?",
            "lon": "?",
            "time_st": "2024-06-16 18:00:00",
            "time_en": "2024-06-18 01:10:35",
            "burst_dt": "?",
            "bursts_t": "?",
            "comment": "?",
            "coef_date": "?",
            "time_raw_st": "?",
            "time_raw_en": "?",
            "data_paths": {
                (Path("/some/device_dir/_raw/140617i.h5"), PurePosixPath("/i03")): {
                    "h5_type": "raw"
                },
                (Path("/some/device_dir/_raw/txt"), PurePosixPath("i03.txt")): {},
            },
            "setup_name": "1406/BalticSpit",
            "combined_comments": {},
        },
    }


@pytest.fixture
def dev_dir_meta(single_interval_device_meta):
    """Create dev_dir_meta dict mapping device dir path to device metadata."""
    return {Path("/some/device_dir"): single_interval_device_meta}


@pytest.fixture
def device_meta_with_hdf5_times():
    """Device metadata with real coef_date, time_raw_st, time_raw_en values.

    Simulates the output of get_absent_meta after HDF5 extraction populates
    the coefficient date and raw time range fields.
    """
    return {
        "i03": {
            "point": "?",
            "sea_depth": "?",
            "height_above_bottom": "?",
            "modification_symbol": "?",
            "lat": "?",
            "lon": "?",
            "time_st": "2024-06-16 18:00:00",
            "time_en": "2024-06-18 01:10:35",
            "burst_dt": "?",
            "bursts_t": "?",
            "comment": "?",
            "coef_date": "2024-06-15 12:30:00",
            "time_raw_st": "2024-06-15 12:30:10",
            "time_raw_en": "2024-06-18 01:15:20",
            "data_paths": {
                (Path("/some/device_dir/_raw/140617i.h5"), PurePosixPath("/i03")): {
                    "h5_type": "raw"
                },
                (Path("/some/device_dir/_raw/txt"), PurePosixPath("i03.txt")): {},
            },
            "setup_name": "1406/BalticSpit",
            "combined_comments": {},
        },
    }


@pytest.fixture
def dev_dir_meta_with_hdf5_times(device_meta_with_hdf5_times):
    """Create dev_dir_meta with real HDF5-derived time values."""
    return {Path("/some/device_dir"): device_meta_with_hdf5_times}


class TestDataPathsInMetaTsvNoInfoFile:
    """Verify data_paths are written to meta_TCM.tsv for single-interval devices without info files."""

    @pytest.mark.parametrize(
        "write_1st_paths, data_path_col_name",
        [(True, "data_file_path"), (False, "data_paths")],
        ids=["1st_path_only", "all_paths"],
    )
    def test_single_interval_data_paths_written(
        self, tmp_path, dev_dir_meta, write_1st_paths, data_path_col_name
    ):
        """Verify data_paths are written to meta_TCM.tsv for single-interval devices."""
        meta_tcm_path = tmp_path / "meta_TCM.tsv"
        dataset_date_map = {"1406/BalticSpit": "240616"}

        write_metadata_table(
            dev_dir_meta,
            meta_tcm_path,
            dataset_date_map=dataset_date_map,
            write_1st_paths=write_1st_paths,
        )

        content = meta_tcm_path.read_text()
        lines = content.strip().split("\n")
        header = lines[0].strip().split("\t")

        data_path_col_idx = header.index(data_path_col_name)
        assert data_path_col_idx >= 0, (
            f"Column '{data_path_col_name}' not found in headers: {header}"
        )

        # Find our device row
        device_row = None
        for line in lines[1:]:
            if line.strip() and "i03" in line:
                device_row = line.strip().split("\t")
                break

        assert device_row is not None, "No row found for device i03"

        data_path_value = device_row[data_path_col_idx]
        assert data_path_value != "?", (
            f"data_file_path should not be '?' (data_paths were not passed to entry), "
            f"got '{data_path_value}'"
        )
        assert "i03" in data_path_value or "140617i.h5" in data_path_value, (
            f"data path should reference i03 and h5 file, got '{data_path_value}'"
        )

    def test_single_interval_quality_is_set(self, tmp_path, dev_dir_meta):
        """Verify quality column is set correctly when data_paths are present."""
        meta_tcm_path = tmp_path / "meta_TCM.tsv"

        write_metadata_table(dev_dir_meta, meta_tcm_path, write_1st_paths=True)

        content = meta_tcm_path.read_text()
        lines = content.strip().split("\n")
        header = lines[0].strip().split("\t")
        quality_idx = header.index("quality")

        for line in lines[1:]:
            if line.strip() and "i03" in line:
                row = line.strip().split("\t")
                quality = row[quality_idx]
                assert quality != "-", (
                    f"Quality should not be '-' when data_paths exist, got '{quality}'"
                )
                break


class TestHdf5TimeFieldsInMetaTsv:
    """Verify coef_date, time_raw_st, time_raw_en are written by write_metadata_table."""

    @pytest.mark.parametrize(
        "field_name",
        ["coef_date", "time_raw_st", "time_raw_en"],
        ids=["coef_date", "time_raw_st", "time_raw_en"],
    )
    def test_hdf5_field_columns_in_header(self, tmp_path, dev_dir_meta, field_name):
        """Verify coef_date, time_raw_st, time_raw_en columns appear in the TSV header."""
        meta_tcm_path = tmp_path / "meta_TCM.tsv"

        write_metadata_table(dev_dir_meta, meta_tcm_path, write_1st_paths=True)

        header = meta_tcm_path.read_text().strip().split("\n")[0].strip().split("\t")
        assert field_name in header, (
            f"Column '{field_name}' not found in header: {header}"
        )

    def test_hdf5_field_placeholder_values_preserved(self, tmp_path, dev_dir_meta):
        """Verify '?' placeholder values for coef_date/time_raw_st/time_raw_en are written."""
        meta_tcm_path = tmp_path / "meta_TCM.tsv"

        write_metadata_table(dev_dir_meta, meta_tcm_path, write_1st_paths=True)

        lines = meta_tcm_path.read_text().strip().split("\n")
        header = lines[0].strip().split("\t")

        device_row = None
        for line in lines[1:]:
            if line.strip() and "i03" in line:
                device_row = line.strip().split("\t")
                break

        assert device_row is not None, "No row found for device i03"

        for field_name in ("coef_date", "time_raw_st", "time_raw_en"):
            col_idx = header.index(field_name)
            assert device_row[col_idx] == "?", (
                f"Expected '{field_name}' to be '?', got '{device_row[col_idx]}'"
            )

    def test_hdf5_field_real_values_written(
        self, tmp_path, dev_dir_meta_with_hdf5_times
    ):
        """Verify real coef_date/time_raw_st/time_raw_en values are written to TSV."""
        expected_values = {
            "coef_date": "2024-06-15 12:30:00",
            "time_raw_st": "2024-06-15 12:30:10",
            "time_raw_en": "2024-06-18 01:15:20",
        }
        meta_tcm_path = tmp_path / "meta_TCM.tsv"

        write_metadata_table(
            dev_dir_meta_with_hdf5_times, meta_tcm_path, write_1st_paths=True
        )

        lines = meta_tcm_path.read_text().strip().split("\n")
        header = lines[0].strip().split("\t")

        device_row = None
        for line in lines[1:]:
            if line.strip() and "i03" in line:
                device_row = line.strip().split("\t")
                break

        assert device_row is not None, "No row found for device i03"

        for field_name, expected in expected_values.items():
            col_idx = header.index(field_name)
            assert device_row[col_idx] == expected, (
                f"Expected '{field_name}' to be '{expected}', "
                f"got '{device_row[col_idx]}'"
            )


def _make_station_meta(
    point="",
    sea_depth=85,
    height_above_bottom=0,
    modification_symbol="",
    lat=55.894,
    lon=19.089,
    time_st="",
    time_en="",
    burst_dt="",
    bursts_t="",
    comment="",
    coef_date="",
    time_raw_st="",
    time_raw_en="",
) -> dict:
    """Build a field-name dict for a single station/interval.

    Mirrors the structure that ``get_absent_meta`` returns when reading
    from an info file: all 14 fields of ``info_devices_field_names_extended``
    are present.  ``coef_date``, ``time_raw_st``, ``time_raw_en`` typically
    contain empty strings (initialized by ``meta_dev_default``), while the
    real values are set later at the device level by
    ``update_device_metadata_with_time_info`` (collect.py:306-311).
    """
    return dict(
        zip(
            info_devices_field_names_extended,
            [
                point,
                sea_depth,
                height_above_bottom,
                modification_symbol,
                lat,
                lon,
                time_st,
                time_en,
                burst_dt,
                bursts_t,
                comment,
                coef_date,
                time_raw_st,
                time_raw_en,
            ],
        )
    )


@pytest.fixture
def multi_interval_device_with_hdf5_fields():
    """Multi-interval device with HDF5 fields at device level.

    Mirrors the real structure returned by ``get_absent_meta`` when an info
    file defines station-level metadata and
    ``update_device_metadata_with_time_info`` (collect.py:306-311) sets
    ``coef_date``, ``time_raw_st``, ``time_raw_en`` at the device level
    (not inside station dicts).
    """
    return {
        "i10": {
            "0": _make_station_meta(
                point="A1",
                time_st="2024-06-16 18:00:00",
                time_en="2024-06-17 06:00:00",
            ),
            "1": _make_station_meta(
                point="A2",
                time_st="2024-06-22 12:00:00",
                time_en="2024-06-25 08:30:00",
            ),
            # HDF5 fields set at device level by update_device_metadata_with_time_info
            "coef_date": "2024-06-15 12:30:00",
            "time_raw_st": "2024-06-15 12:30:10",
            "time_raw_en": "2024-06-25 09:00:00",
            "data_paths": {
                (Path("/dev_dir/_raw/240616i.h5"), PurePosixPath("/i10")): {
                    "h5_type": "raw"
                },
            },
            "setup_name": "1406/BalticSpit",
            "combined_comments": {},
        },
    }


@pytest.fixture
def multi_interval_dev_dir_meta(multi_interval_device_with_hdf5_fields):
    """dev_dir_meta with multi-interval device."""
    return {Path("/dev_dir"): multi_interval_device_with_hdf5_fields}


class TestMultiIntervalHdf5TimeFields:
    """Verify HDF5 time fields propagate from device level to all station rows.

    ``get_absent_meta`` sets ``coef_date``, ``time_raw_st``, ``time_raw_en``
    at the device level (collect.py:306-311), not inside station dicts.
    ``write_metadata_table`` must propagate these via the fallback
    ``meta.get(field_name, "?")`` (file_writer.py:205) to every station row.
    """

    def test_multi_interval_creates_rows_with_hdf5_fields(
        self, tmp_path, multi_interval_dev_dir_meta
    ):
        """Each station row must inherit device-level HDF5 time fields."""
        expected = {
            "coef_date": "2024-06-15 12:30:00",
            "time_raw_st": "2024-06-15 12:30:10",
            "time_raw_en": "2024-06-25 09:00:00",
        }
        meta_tcm_path = tmp_path / "meta_TCM.tsv"

        write_metadata_table(
            multi_interval_dev_dir_meta, meta_tcm_path, write_1st_paths=True
        )

        lines = meta_tcm_path.read_text().strip().split("\n")
        header = lines[0].strip().split("\t")

        # Must produce two rows for i10 (one per station)
        i10_rows = [
            line.strip().split("\t")
            for line in lines[1:]
            if line.strip() and "i10" in line
        ]
        assert len(i10_rows) == 2, f"Expected 2 rows for i10, got {len(i10_rows)}"

        # Every row must carry the device-level HDF5 values
        for row_idx, row in enumerate(i10_rows):
            for field_name, expected_val in expected.items():
                col_idx = header.index(field_name)
                assert row[col_idx] == expected_val, (
                    f"Row {row_idx}: expected '{field_name}'='{expected_val}', "
                    f"got '{row[col_idx]}'"
                )

    def test_multi_interval_station_empty_falls_back_to_device_level(self, tmp_path):
        """Station-level empty values must not suppress device-level HDF5 fields.

        Regression test: ``dict.get(key, default)`` only falls back when the
        key is *absent*.  When the station dict contains ``coef_date: ""``
        (empty string), the old code returned ``""`` instead of the device-level
        value ``"2023-11-26 10:17:36"``.
        """
        device_meta = {
            "i53": {
                # Station dict: has all 14 fields, but extended fields are empty
                "0": _make_station_meta(
                    point="7733",
                    sea_depth=225,
                    height_above_bottom=0,
                    modification_symbol="↟",
                    lat=76.782617,
                    lon=63.923183,
                    time_st="2023-11-19 10:52:00",
                    time_en="2023-11-25 03:08:56",
                    # coef_date="" (default), time_raw_st="" (default),
                    # time_raw_en="" (default) — keys present, values empty
                ),
                # Device level: real values set by update_device_metadata_with_time_info
                "coef_date": "2023-11-26 10:17:36",
                "time_raw_st": "2023-11-19 12:52:00",
                "time_raw_en": "2023-11-25 05:08:56",
                "data_paths": {},
                "setup_name": "AMK93",
                "combined_comments": {},
            },
        }
        dev_dir_meta = {Path("/dev_dir"): device_meta}
        expected = {
            "coef_date": "2023-11-26 10:17:36",
            "time_raw_st": "2023-11-19 12:52:00",
            "time_raw_en": "2023-11-25 05:08:56",
        }
        meta_tcm_path = tmp_path / "meta_TCM.tsv"

        write_metadata_table(dev_dir_meta, meta_tcm_path, write_1st_paths=True)

        lines = meta_tcm_path.read_text().strip().split("\n")
        header = lines[0].strip().split("\t")

        device_row = [
            line.strip().split("\t")
            for line in lines[1:]
            if line.strip() and "i53" in line
        ]
        assert len(device_row) == 1, f"Expected 1 row for i53, got {len(device_row)}"

        for field_name, expected_val in expected.items():
            col_idx = header.index(field_name)
            assert device_row[0][col_idx] == expected_val, (
                f"Expected '{field_name}'='{expected_val}', "
                f"got '{device_row[0][col_idx]}'"
            )
