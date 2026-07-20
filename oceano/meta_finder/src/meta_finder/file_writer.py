"""
File writing functions for TCM Metadata Processor.
"""

import logging
from typing import List, Dict, Any, Union
from pathlib import Path

from .config import config
from .logging_config import setup_logging
from . import io_info_files
from .io_info_files import info_devices_field_names_extended

logger = setup_logging()


def write_files_list(
    dev_dir_meta: Dict[Path, Dict[str, Any]],
    out_path: Path,
    dataset_date_map: Dict[str, str] = None,
    write_1st_paths=False,
) -> None:
    """
    Write a structured list of found files to files_TCM.tsv.

    The output is grouped by dataset. Each group starts with the path to the
    device directory.
    Dataset groups are separated by a blank line.

    Args:
        dev_dir_meta: A dictionary where keys are paths to device directories
            and values are dictionaries of devices metadata.
        out_path: Path to the output files_TCM.tsv file.
        dataset_date_map: Dictionary mapping dataset names to their dates (YYMMDD format).
        write_1st_paths: write only the 1st data file path for each device if True,
            otherwise write all collected paths
    """
    logger.info(
        "Writing files list to %s with %s",
        str(out_path),
        "1st priority data file only" if write_1st_paths else "all collected paths",
    )

    # First, group all the data by setup_name.
    dataset = {}
    for dev_dir_path, devices_data in dev_dir_meta.items():
        # All devices under one dev_dir share the same setup_name.
        # We can get the setup_name from the first device.
        if not devices_data:
            continue
        try:
            name = next(iter(devices_data.values()))["setup_name"]
        except KeyError:
            name = "?"

        # Each item in the list will be the device dir path and its associated devices
        if name not in dataset:
            dataset[name] = []
        dataset[name].append((dev_dir_path, devices_data))

    # Sort datasets by date (dataset_date_map values) for consistent output
    # If dataset_date_map is not provided or setup_name not in map, use dataset name as fallback
    sorted_names = sorted(
        dataset.keys(),
        key=(lambda x: (dataset_date_map.get(x, ""), x))
        if dataset_date_map
        else (lambda x: ""),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(sorted_names):
            if i > 0:
                f.write("\n")  # Add a blank line between datasets

            # For all items for this dataset sorted by device dir path for consistent order
            for dev_dir_path, devices_data in sorted(dataset[name], key=lambda x: x[0]):
                f.write(f"{dev_dir_path.as_posix()}\n")

                # For all devices of current json sorted by device_id (skip None keys from malformed YAML)
                for device_id in sorted(
                    k for k in devices_data.keys() if k is not None
                ):
                    device_info = devices_data[device_id]
                    # data_paths is a dictionary mapping path tuples to metadata
                    b_one_written = False
                    for path_tuple, filename_meta in device_info["data_paths"].items():
                        if not isinstance(path_tuple, tuple) and len(path_tuple) == 2:
                            logger.warning("Bad format of path_tuple: {path_tuple}")
                            continue
                        parent_path, rel_path = path_tuple
                        # remove root slash for hdf5/mat group to able construct relative path
                        if parent_path.suffix.lower() in (".h5", ".mat"):
                            rel_path = rel_path.as_posix().lstrip("/")

                        full_path = parent_path / rel_path
                        f.write(f"{full_path.as_posix()}\n")

                        # Write only the 1st data path per device when write_1st_paths is enabled
                        if not b_one_written:
                            b_one_written = True
                            if write_1st_paths:
                                break

    logger.info(f"Finished writing files list to {out_path}")


def write_metadata_table(
    dev_dir_meta: Dict[Path, Dict[str, Any]],
    meta_tcm_path: Path,
    dataset_date_map: Dict[str, str] = None,
    write_1st_paths: bool = True,
) -> None:
    """
    Write the final metadata to meta_TCM.tsv.

    Flatten the nested metadata structure, handle missing burst values,
    and optionally limit data file paths to the 1st (highest priority) per device.
    Handle devices with multiple intervals by creating one row per interval.

    write_1st_paths: when True (default) write only the 1st (highest priority) data file path per device,
    else write all collected paths and use 'data_paths' column name
    """
    # Count total devices across all directories for early logging
    total_devices = sum(len(devices_data) for devices_data in dev_dir_meta.values())
    logger.info(
        "Writing metadata table to %s (%s) with %s",
        meta_tcm_path,
        f"{total_devices} devices from {len(dev_dir_meta)} directories",
        "1st priority data file only" if write_1st_paths else "all collected paths",
    )

    # Determine which field to use for data paths
    data_path_field = "data_file_path" if write_1st_paths else "data_paths"

    # Start with all the fields from the extended list (excluding 'data_paths')
    all_fields = [
        field for field in info_devices_field_names_extended if field != "data_paths"
    ]

    # Create headers with data_file_path, quality (new field), comment, modification_symbol in that order
    # starting after the position of the element where comment originally was

    fields_without_moving_items = [
        field for field in all_fields if field not in ["comment", "modification_symbol"]
    ]
    comment_original_idx = info_devices_field_names_extended.index("comment")
    adjusted_comment_idx = comment_original_idx
    if (
        info_devices_field_names_extended.index("modification_symbol")
        < comment_original_idx
    ):
        adjusted_comment_idx -= 1
    # Insert quality column before comment
    headers = (
        ["setup_name", "device_id"]
        + fields_without_moving_items[:adjusted_comment_idx]
        + [data_path_field, "quality", "comment", "modification_symbol"]
        + fields_without_moving_items[adjusted_comment_idx:]
    )

    # Flatten the metadata from the nested dictionary `dev_dir_meta`
    all_devices = []

    def _process_device_entry(device_id: str, meta: Dict[str, Any], **kwargs) -> None:
        """Copy field values from `meta` into a flat entry dict and append to `all_devices`.

        Fields listed in `info_devices_field_names_extended` are copied individually,
        with burst-specific defaults. Since `data_paths` is not in that list,
        it must be passed via kwargs to be included in the entry.

        Args:
            device_id: Device identifier.
            meta: Device metadata dictionary.
            kwargs: Extra keys merged into the entry (overriding meta values):
                interval_index: Optional interval index for multi-interval devices.
                data_paths: Data-path mapping to copy from device metadata.
        """
        entry = {}
        for field_name in info_devices_field_names_extended:
            field_value = meta.get(field_name, "?")
            # Handle burst values: if missing or empty, set to '-'
            if field_name in ["bursts_t", "burst_dt"]:
                entry[field_name] = (
                    "-" if (not field_value or field_value == "?") else field_value
                )
            else:
                entry[field_name] = "" if field_value is None else field_value

        entry["device_id"] = device_id
        entry.update(kwargs)

        # Copy setup_name from source metadata if present (for nested dict structures)
        # This is needed because setup_name is not in info_devices_field_names_extended
        if "setup_name" in meta:
            entry["setup_name"] = meta["setup_name"]

        # Handle comment field: if missing, set to '-'
        if not entry.get("comment") or entry.get("comment") == "?":
            entry["comment"] = ""

        all_devices.append(entry)

    for devices_data in dev_dir_meta.values():
        for device_id, meta in devices_data.items():
            logger.debug(
                f"meta for {device_id} found: {str([k for k, v in meta.items() if v != '?'])}"
            )
            # Try to process as nested dict structure (multiple intervals)
            # If no station items are found, fall back to single interval structure
            has_nested = False
            for station_id, station_metadata in io_info_files.iter_station_id_items(
                meta
            ):
                has_nested = True
                # Build field-name dict from station_metadata (list or dict)
                # When station_metadata is already a field-name dict (from
                # info_meta_list_to_dict), use it directly; otherwise convert
                # from index-based list, falling back to device-level meta.
                if isinstance(station_metadata, dict):
                    station_metadata_dict = {}
                    for field_name in info_devices_field_names_extended:
                        val = station_metadata.get(field_name)
                        if val not in ("?", "", None, "-", "None"):
                            station_metadata_dict[field_name] = val
                        else:
                            station_metadata_dict[field_name] = meta.get(
                                field_name, "?"
                            )
                else:
                    station_metadata_dict = {}
                    for idx, field_name in enumerate(info_devices_field_names_extended):
                        if idx < len(station_metadata):
                            val = station_metadata[idx]
                            if val not in ("?", "", None, "-", "None"):
                                station_metadata_dict[field_name] = val
                            else:
                                station_metadata_dict[field_name] = meta.get(
                                    field_name, "?"
                                )
                        else:
                            station_metadata_dict[field_name] = meta.get(
                                field_name, "?"
                            )
                # Preserve all special metadata keys (setup_name, data_paths, combined_comments)
                for special_key in io_info_files._SPECIAL_METADATA_KEYS:
                    if special_key in meta:
                        station_metadata_dict[special_key] = meta[special_key]
                _process_device_entry(
                    device_id,
                    station_metadata_dict,
                    interval_index=int(station_id),
                    data_paths=meta.get("data_paths", {}),
                )
            if (
                not has_nested
            ):  # Device has single interval structure - create one entry
                logger.debug(f"  Device {device_id}: single interval structure")
                _process_device_entry(
                    device_id, meta, data_paths=meta.get("data_paths", {})
                )

    # Process data paths and quality for all device entries
    logger.debug(f"Processing {len(all_devices)} device entries for TCV output")
    for entry in all_devices:
        logger.debug(
            f"  Processing entry: device_id={entry.get('device_id')}, has_data_paths={'data_paths' in entry}"
        )
        if entry.get("data_paths"):
            # data_paths is a dictionary mapping path tuples to metadata
            collected_paths = []
            for path_tuple, filename_meta in entry["data_paths"].items():
                if not isinstance(path_tuple, tuple) or len(path_tuple) != 2:
                    logger.warning(f"strange data in path tuple: {path_tuple}")
                    continue
                parent_path, rel_path = path_tuple

                # remove root slash for hdf5/mat group to able construct relative path
                if parent_path.suffix.lower() in (".h5", ".mat"):
                    rel_path = rel_path.as_posix().lstrip("/")

                full_path = (parent_path / rel_path).as_posix()

                # Handle data paths based on configuration
                collected_paths.append(full_path)
                if write_1st_paths:
                    break

            # all collected data paths we will write as a comma-separated string
            data_path_value = ",".join(collected_paths) if collected_paths else "?"
        else:
            data_path_value = "?"

        entry[data_path_field] = data_path_value

        # Determine quality based on data_file_path
        if data_path_value and data_path_value != "?":
            entry["quality"] = "?" if "_raw" in data_path_value else "+"
        else:
            # data_file_path not found or is '?'
            entry["quality"] = "-"

    # Sort devices by dataset date
    if dataset_date_map:

        def _sort_key(x):
            return (dataset_date_map.get(x.get("setup_name", ""), ""),)

        all_devices.sort(key=_sort_key)

    # no sense to sort by setup_name - better by setup_name data min time,
    # and not sort by device ID if data comes in metadata files order, that has better sorting
    #     def _sort_key(x):
    #         return (
    #             x.get("setup_name", ""),
    #             x.get("device_id", ""),
    #         )

    with open(meta_tcm_path, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for entry in all_devices:
            f.write("\t".join(str(entry.get(header, "")) for header in headers) + "\n")

    logger.info(f"Finished writing metadata table to {meta_tcm_path}")


def find_latest_meta_file(output_dir: Union[str, Path]) -> str:
    """Find the latest meta_TCM_*.tsv file in the output directory."""
    logger.info(f"Finding latest meta_TCM_*.tsv file in {output_dir}")
    output_dir_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
    files = list(output_dir_path.glob("meta_TCM_*.tsv"))
    if not files:
        logger.info("No existing meta_TCM_*.tsv files found")
        return None
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    logger.info(
        f"Found {len(files)} existing meta_TCM_*.tsv files, returning the newest one"
    )
    return str(files[0])


def load_existing_metadata(
    meta_file_path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Load existing metadata from a meta_TCM_*.tsv file.

    Args:
        meta_file_path: Path to existing meta_TCM_*.tsv file

    Returns:
        Dictionary mapping device keys to metadata dictionaries
    """
    logger.info(f"Loading existing metadata from {meta_file_path}")
    meta_file_path_obj = (
        Path(meta_file_path) if isinstance(meta_file_path, str) else meta_file_path
    )

    existing_metadata = {}
    try:
        with open(meta_file_path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            logger.info("Existing metadata file is empty")
            return existing_metadata

        # Parse header
        headers = lines[0].strip().split("\t")

        # Parse data rows
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split("\t")
                row_data = {
                    header: values[i] if i < len(values) else "?"
                    for i, header in enumerate(headers)
                }
                if "device_id" in row_data and row_data["device_id"] != "?":
                    device_key = (
                        f"{row_data.get('setup_name', '?')}_{row_data['device_id']}"
                    )
                    existing_metadata[device_key] = row_data

        logger.info(f"Loaded existing metadata for {len(existing_metadata)} devices")
        return existing_metadata
    except Exception as e:
        logger.error(f"Error loading existing metadata from {meta_file_path}: {e}")
        return existing_metadata
