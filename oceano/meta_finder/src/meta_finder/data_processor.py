"""
High-level data processing orchestration for TCM Metadata Processor.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path, PurePath

from . import config
from .logging_config import setup_logging
logger = setup_logging()
h5_sfx_priority_map = {".proc_noAvg": 2, ".proc_Avg": 3, ".proc": 4, ".raw": 5}

def get_h5_type_and_priority(h5_file_path: Path):
    """Determine HDF5 file type and priority from file path.

    Analyzes the HDF5 file path to determine its type and corresponding priority
    for sorting data sources. The priority follows the scheme:
    - proc_noAvg: priority 2 - highest for HDF5
    - proc_Avg: priority 3 - devices in group names (e.g., i04bin2s), no device suffixes in columns
    - proc: priority 4 - devices in column names like Vabs_i03
    - raw: priority 5 - lowest for HDF5
    Args:
        h5_file_path: Path to the HDF5 file (can be Path object or string)

    Returns:
        Tuple of (h5_type, priority) where:
            - h5_type: str, one of 'proc_noAvg', 'proc_Avg', 'proc', or 'raw'
            - priority: int, the sorting priority (2, 3, 4, or 5)

    Raises:
        ValueError: If the file path does not match any known HDF5 pattern
    """


    # Convert to Path object if string is provided
    if not isinstance(h5_file_path, PurePath):
        h5_file_path = PurePath(h5_file_path)

    # Determine h5_type and priority based on filename patterns
    *other_suffixes, last_suffix = h5_file_path.suffixes

    if last_suffix == ".h5":
        for sfx in other_suffixes:
            try:
                priority = h5_sfx_priority_map[sfx]
                h5_type = sfx[1:]
                break
            except KeyError:
                continue
        else:
            if "_raw" in h5_file_path.parts:
                h5_type = "raw"
                priority = 5
            else:
                # Raise ValueError for unrecognized HDF5 files as documented
                raise ValueError(
                    f"Unrecognized HDF5 file pattern: {h5_file_path}: name must contain one of "
                    f"{list(h5_sfx_priority_map)} suffix or file must be under _raw directory")

    elif last_suffix == ".mat":
        h5_type = "raw"
        priority = 6  # .mat files treated as raw HDF5 files
    else:
        # Raise ValueError for non-HDF5/.mat files as documented
        raise ValueError(f"File is not an HDF5 or .mat file: {h5_file_path}")

    return h5_type, priority


def sort_input_dirs(path):
    stem = path.name.split('.')[0]  # базовое имя без расширения
    suffix = path.suffix
    # Задаем приоритет: папка < .zip < .7z
    priority = {'': 0, '.zip': 1, '.7z': 2}.get(suffix, 0)
    return (stem, priority)


def _is_raw_source(parent_path: Path, rel_path: PurePath) -> bool:
    """Check whether a data source path points to raw (unprocessed) data."""
    return (
        "_raw" in parent_path.parts
        or parent_path.stem.startswith("_raw")
        or parent_path.stem.endswith("_raw")
        or "_raw" in str(rel_path).lower()
    )


def sort_data_paths(paths_meta: dict, devices_needed: set) -> list:
    """Sort data sources by a three-level priority scheme.

    Priority levels (highest to lowest):
      1. **Raw vs processed**: processed data (text_output, .proc.h5) comes before
         raw data (_raw dirs, raw .h5/.mat). This ensures we prefer already
         extracted/processed files over raw instrument dumps.
      2. **Averaging interval**: 2 s bin is optimal for text files. Priority
         decreases with distance from 2 s — both coarser (300 s) and finer (< 2 s)
         bins rank lower. For HDF5 files, averaging ≤ 2 s is treated as equal to
         2 s (high-resolution HDF5 data is as good as 2 s bin). Files without
         explicit averaging use the configured default (2.0001 s).
      3. **Source type**: among files with same raw/processed level and same
         averaging distance, prefer directories > archives > proc HDF5 > raw HDF5.

    After these three levels, ties are broken by:
      4. Specificity (dedicated files beat combined / wildcard files).
      5. Number of devices in filename (fewer is better).
      6. Number of unmatched devices (fewer is better).
    """

    # The optimal bin size — closest to this value wins for text files.
    _OPTIMAL_BIN_S = 2.0

    def sort_key(item):
        (parent_path, rel_path), dataname_metadata = item
        if not isinstance(parent_path, Path) or not isinstance(rel_path, PurePath):
            raise ValueError(
                f"(parent_path, rel_path)=({parent_path} {rel_path}) must be of (Path, PureFile) type!"
            )

        # Level 1: raw (True=1) vs processed (False=0). Processed first.
        is_raw = _is_raw_source(parent_path, rel_path)

        # Level 2: averaging interval — distance from optimal 2 s bin
        avg_interval = dataname_metadata.get('averaging_interval', config.default_text_file_averaging)
        if avg_interval is None:
            avg_interval = config.default_text_file_averaging
        try:
            avg_interval = float(avg_interval)
        except (TypeError, ValueError):
            avg_interval = float(config.default_text_file_averaging)

        # Level 3: source type priority (within the same raw/processed level)
        priority = 10  # default to lowest priority
        container_suffix = parent_path.suffix.lower()
        is_hdf5 = container_suffix in config.extensions_hdf5
        if is_hdf5:
            _, priority = get_h5_type_and_priority(parent_path)
        else:
            if not rel_path.name:
                raise ValueError(
                    f"Empty rel_path is not allowed: dir_path={parent_path}, rel_path={rel_path}"
                )

            if container_suffix in ['.zip', '.7z', '.rar']:
                priority = 9 if is_raw else 1  # raw archives lowest; text_output archives above HDF5
            elif container_suffix in config.extensions_hdf5:
                priority = 7
            else:
                # Directory container
                priority = 8 if is_raw else 0  # raw dirs low; text_output dirs highest

        # For HDF5 files, averaging ≤ 2 s is equivalent to 2 s (high-res is as good as bin2s).
        # For text files, distance from 2 s in either direction lowers priority.
        if is_hdf5 and avg_interval <= _OPTIMAL_BIN_S:
            avg_distance = 0.0
        else:
            avg_distance = abs(avg_interval - _OPTIMAL_BIN_S)

        filename_devices = set(dataname_metadata.get('devices', []))

        # Level 4: specificity — combined files (wildcards) are lower priority
        is_combined = any(d in filename_devices for d in ("*", "i", "w", "p"))

        # Level 5: number of devices in filename (fewer is better)
        num_devices = len(filename_devices)

        # Level 6: number of unmatched devices (fewer is better)
        unmatched_devices_count = len(filename_devices - devices_needed)

        return (is_raw, avg_distance, priority, is_combined, num_devices, unmatched_devices_count)

    sorted_items = sorted(paths_meta.items(), key=sort_key)
    return sorted_items
