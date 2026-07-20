"""Standalone script to create and verify test HDF5 files for the test suite.

Creates files in test_data/Cruises/240303_hdf5_fallback_test/240304_device@i05/:
- test.proc_noAvg.h5: standard file with group i05, time range + data
- test2.proc.h5: combined file with device columns Vabs_i05, Vdir_i05
- test.proc_Avg.h5: standard file with group i05bin2s (averaging metadata)
- _raw/raw_test.h5: raw file with group i07, coef/date array, time range

Run directly to create and verify: pixi run -e test python tests/_create_test_hdf5.py
"""

from pathlib import Path

import numpy as np
import tables

# Base path for test HDF5 files
BASE_DIR = Path(__file__).parent.parent / "test_data" / "Cruises" / "240303_hdf5_fallback_test" / "240304_device@i05"

# Human-readable datetime strings → np.datetime64 → float64 epoch seconds for Time64Col
_TIME_STRINGS = ["2023-01-01T10:00:00", "2023-01-01T10:00:01", "2023-01-01T10:00:02", "2023-01-01T10:05:00"]
TIME_VALUES_SEC = np.array(
    [np.datetime64(s, "s").astype("datetime64[us]").astype(np.float64) / 1e6 for s in _TIME_STRINGS],
    dtype=np.float64,
)

# Coef date: 2019-12-02T02:17:20 UTC
_COEF_DATE_STR = "2019-12-02T02:17:20"
COEF_DATE_VALUES = np.array([
    np.nan,
    np.datetime64(_COEF_DATE_STR, "s").astype("datetime64[us]").astype(np.float64) / 1e6,
])


def _datetime64_to_epoch_sec(dt_str: str) -> float:
    """Convert a datetime string to float64 epoch seconds (for Time64Col)."""
    return np.datetime64(dt_str, "s").astype("datetime64[us]").astype(np.float64) / 1e6


def _create_hdf5_with_device_group(
    h5_path: Path, group_name: str, time_values: np.ndarray, data_columns: dict,
    coef_date_values: np.ndarray | None = None,
) -> None:
    """Create an HDF5 file with a root-level device group containing a table.

    The table uses Time64Col for the 'index' column (float64 epoch seconds) plus data columns.
    Optionally creates a coef/date array under the device group.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with tables.open_file(str(h5_path), mode="w") as h5file:
        grp = h5file.create_group("/", group_name)
        desc = {"index": tables.Time64Col(pos=0)}
        for pos, (col_name, arr) in enumerate(data_columns.items(), start=1):
            desc[col_name] = tables.Col.from_dtype(arr.dtype)
            desc[col_name]._v_pos = pos

        tbl = h5file.create_table(grp, "data", description=desc)
        row = tbl.row
        for i in range(len(time_values)):
            row["index"] = time_values[i]
            for col_name, arr in data_columns.items():
                row[col_name] = arr[i]
            row.append()
        tbl.flush()

        if coef_date_values is not None:
            coef_grp = h5file.create_group(grp, "coef")
            h5file.create_array(coef_grp, "date", coef_date_values)


def _create_hdf5_proc_combined(
    h5_path: Path, time_values: np.ndarray, device_columns: dict,
) -> None:
    """Create a .proc.h5 file with a single root-level table containing combined device columns.

    The table uses Time64Col for the 'index' column (float64 epoch seconds) plus device-specific
    columns like 'Vabs_i05', 'Vdir_i05'.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with tables.open_file(str(h5_path), mode="w") as h5file:
        desc = {"index": tables.Time64Col(pos=0)}
        for pos, (col_name, arr) in enumerate(device_columns.items(), start=1):
            desc[col_name] = tables.Col.from_dtype(arr.dtype)
            desc[col_name]._v_pos = pos

        tbl = h5file.create_table("/", "data", description=desc)
        row = tbl.row
        for i in range(len(time_values)):
            row["index"] = time_values[i]
            for col_name, arr in device_columns.items():
                row[col_name] = arr[i]
            row.append()
        tbl.flush()


def create_all_test_hdf5_files(base_dir: Path = BASE_DIR) -> None:
    """Create all HDF5 test files needed by the test suite."""
    base_dir.mkdir(parents=True, exist_ok=True)

    # --- test.proc_noAvg.h5: standard file with device group i05 ---
    proc_noAvg_path = base_dir / "test.proc_noAvg.h5"
    if not proc_noAvg_path.exists():
        _create_hdf5_with_device_group(
            proc_noAvg_path, "i05", TIME_VALUES_SEC,
            {"Vabs": np.array([0.1, 0.15, 0.2, 0.25]), "Vdir": np.array([180.0, 185.0, 190.0, 195.0])},
        )

    # --- test2.proc.h5: combined file with device columns ---
    proc_path = base_dir / "test2.proc.h5"
    if not proc_path.exists():
        _create_hdf5_proc_combined(
            proc_path, TIME_VALUES_SEC,
            {
                "Vabs_i05": np.array([0.1, 0.15, 0.2, 0.25]),
                "Vdir_i05": np.array([180.0, 185.0, 190.0, 195.0]),
            },
        )

    # --- test.proc_Avg.h5: standard file with averaging group name ---
    proc_avg_path = base_dir / "test.proc_Avg.h5"
    if not proc_avg_path.exists():
        _create_hdf5_with_device_group(
            proc_avg_path, "i05bin2s", TIME_VALUES_SEC,
            {"Vabs": np.array([0.1, 0.15, 0.2, 0.25]), "Vdir": np.array([180.0, 185.0, 190.0, 195.0])},
        )

    # --- _raw/raw_test.h5: raw file with coef/date and device group i07 ---
    raw_dir = base_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    raw_path = raw_dir / "raw_test.h5"
    if not raw_path.exists():
        _create_hdf5_with_device_group(
            raw_path, "i07", TIME_VALUES_SEC,
            {"Vabs": np.array([1.0, 2.0, 3.0, 4.0]), "Vdir": np.array([45.0, 46.0, 47.0, 48.0])},
            coef_date_values=COEF_DATE_VALUES,
        )


def verify_all_test_hdf5_files(base_dir: Path = BASE_DIR) -> None:
    """Verify all created HDF5 files have correct structure and readable data."""
    from datetime import datetime

    from meta_finder.hdf5_processor import (
        extract_coef_date_from_hdf5,
        extract_devices_from_hdf5_groups,
        extract_metadata_from_hdf5,
        extract_time_range_from_hdf5_table,
        extract_time_ranges_from_hdf5_combined,
        find_hdf5_files,
    )

    print(f"Verifying HDF5 files in: {base_dir.resolve()}")

    # Test find_hdf5_files
    files = find_hdf5_files(base_dir)
    print(f"\nfind_hdf5_files result:")
    for category, paths in files.items():
        for p in paths:
            print(f"  {category}: {p.name}")

    assert len(files["proc_noAvg"]) == 1, f"Expected 1 proc_noAvg file, got {len(files['proc_noAvg'])}"
    assert len(files["proc_Avg"]) == 1, f"Expected 1 proc_Avg file, got {len(files['proc_Avg'])}"
    assert len(files["proc"]) == 1, f"Expected 1 proc file, got {len(files['proc'])}"
    assert len(files["raw"]) == 1, f"Expected 1 raw file, got {len(files['raw'])}"

    # Compute expected times using same conversion as production code (local timezone)
    expected_start = datetime.fromtimestamp(TIME_VALUES_SEC[0]).strftime("%Y-%m-%d %H:%M:%S")
    expected_end = datetime.fromtimestamp(TIME_VALUES_SEC[-1]).strftime("%Y-%m-%d %H:%M:%S")

    # Test extract_time_range_from_hdf5_table on proc_noAvg file
    proc_noAvg_path = files["proc_noAvg"][0]
    time_range = extract_time_range_from_hdf5_table(proc_noAvg_path, "/i05/data")
    print(f"\nextract_time_range_from_hdf5_table('/i05/data'): {time_range}")
    assert time_range is not None, "Expected time range from proc_noAvg file"
    assert time_range[0] == expected_start, f"Expected start '{expected_start}', got '{time_range[0]}'"
    assert time_range[1] == expected_end, f"Expected end '{expected_end}', got '{time_range[1]}'"

    # Test extract_time_ranges_from_hdf5_combined on proc file
    proc_path = files["proc"][0]
    combined_result = extract_time_ranges_from_hdf5_combined(proc_path, "/data", ["i5"])
    print(f"\nextract_time_ranges_from_hdf5_combined('/data', ['i5']): {combined_result}")
    assert "i5" in combined_result, f"Expected 'i5' in result, got {combined_result.keys()}"
    assert combined_result["i5"] is not None, "Expected non-None time range for i5"

    # Test extract_devices_from_hdf5_groups on raw file
    raw_path = files["raw"][0]
    groups = extract_devices_from_hdf5_groups(raw_path)
    print(f"\nextract_devices_from_hdf5_groups(raw): {groups}")
    assert "i7" in groups, f"Expected 'i7' in groups, got {groups.keys()}"

    # Test extract_coef_date_from_hdf5 on raw file
    coef_date = extract_coef_date_from_hdf5(raw_path, "i07")
    print(f"\nextract_coef_date_from_hdf5('i07'): {coef_date}")
    assert coef_date is not None, "Expected coef date from raw file"
    expected_coef_dt = datetime.fromtimestamp(COEF_DATE_VALUES[1]).strftime("%Y-%m-%d")
    assert coef_date.startswith(expected_coef_dt), f"Expected '{expected_coef_dt}', got '{coef_date}'"

    # Test extract_metadata_from_hdf5
    metadata = extract_metadata_from_hdf5(base_dir, None)
    print(f"\nextract_metadata_from_hdf5(None):")
    for dev_id, data in metadata.items():
        print(f"  {dev_id}: time_info={data.get('time_info')}, data_paths={list(data.get('data_paths', {}).keys())}")

    print("\n✅ All verifications passed!")


if __name__ == "__main__":
    create_all_test_hdf5_files()
    verify_all_test_hdf5_files()
