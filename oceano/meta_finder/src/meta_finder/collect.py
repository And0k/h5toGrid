from typing import List, Dict, Tuple, Any, Optional, Union, Sequence
from pathlib import Path, PurePosixPath
from datetime import datetime
import sys

from meta_finder import config
from meta_finder.config import Config, DEVICES_FILE_NAME, DEVICES_FILE_NAME_YAML, DEVICES_FILE_NAME_UPD
from .file_finder import (
    find_raw_directory_files,
    find_navigation_files,
    discover_device_dirs,
    extract_devices_from_text_output,
)
from .parse_data_file_name import parse_filename_for_metadata, normalize_device_id
from .metadata_extractor import read_metadata_files_to_dict, info_meta_list_to_dict
from . import io_info_files
from .create_info_files import update_devices_meta_file
from .data_processor import sort_data_paths, get_h5_type_and_priority
from .data_proc_funcs import extract_time_info_from_text_file
from .hdf5_processor import (
    extract_metadata_from_hdf5,
    extract_raw_hdf5_metadata,
    extract_time_range_from_hdf5_table,
)
from .parse_cruise_dir_name import add_dataset_name
from .logging_config import setup_logging
from .file_writer import write_files_list, write_metadata_table


logger = setup_logging()

def get_all_data_files_for_device_dir(
    device_dir: Path,
) -> Dict[str, Dict[Tuple[Path, PurePosixPath], Dict[str, Any]]]:
    """
    Find all data files in a device directory and organize them by device ID.

    Uses extract_devices_from_text_output() which includes fallback mechanism to extract
    device IDs from subdirectory names when filenames don't contain device information.

    Args:
        device_dir: Directory containing device data

    Returns:
        Dictionary mapping device IDs to their data_paths and metadata:
        {
            device_id: {(path, rel_path): name_meta}
        }
    """
    logger.debug(f"Getting all data files for device directory: {device_dir}")

    # Use extract_devices_from_text_output which includes fallback mechanism
    # This function returns Dict[str, List[Tuple[Path, PurePosixPath]]]
    text_output_devices = extract_devices_from_text_output(device_dir)
    logger.debug(f"Devices from text_output: {list(text_output_devices.keys())}")

    # Convert to the expected format: Dict[str, Dict[Tuple[Path, PurePosixPath], Dict[str, Any]]]
    data_paths = {}
    for device_id, file_tuples in text_output_devices.items():
        if device_id not in ['*', 'i', 'w', 'p']:  # Skip generic device types
            if device_id not in data_paths:
                data_paths[device_id] = {}
            for dir_path, rel_path in file_tuples:
                path_tuple = (Path(dir_path), rel_path)
                # Parse filename for metadata
                try:
                    name_meta = parse_filename_for_metadata(rel_path.name)
                    data_paths[device_id][path_tuple] = name_meta
                except Exception as e:
                    logger.error(f"Error parsing filename {rel_path.name}: {e}", exc_info=True)
                    data_paths[device_id][path_tuple] = {}

    # Also look for raw directory files
    raw_devices_dict = find_raw_directory_files(device_dir)
    for device_id, paths_dict in raw_devices_dict.items():
        if device_id not in ['*', 'i', 'w', 'p']:  # Skip generic device types
            if device_id not in data_paths:
                data_paths[device_id] = {}
            for dir_path, file_paths in paths_dict.items():
                for file_path in file_paths:
                    path_tuple = (Path(dir_path), file_path)
                    # Create minimal metadata for raw files
                    data_paths[device_id][path_tuple] = {}

    # If HDF5 fallback is enabled, also look for HDF5 files
    if config.extract_hdf5_times:
        # Get {device_id: {"data_paths": path_tuple}} dict,
        # Convert it to same output format as the original code
        for dev_id, h5_data in extract_metadata_from_hdf5(device_dir, extract_time_info=False).items():
            # Skip generic device types
            if dev_id in ['*', 'i', 'w', 'p']:
                logger.warning(f"generic device type ({dev_id}) found in HDF5 extraction: skip")
                continue
            if dev_id not in data_paths:
                data_paths[dev_id] = {}

            # Add all data paths for this device
            for path_tuple, metadata in h5_data.get("data_paths", {}).items():
                # Use unified function to determine h5_type based on the file path
                h5_file_path, h5_group = path_tuple
                try:
                    h5_type, _ = get_h5_type_and_priority(h5_file_path)
                except ValueError:
                    # Fallback to 'raw' if file pattern is not recognized
                    h5_type = 'raw'

                data_paths[dev_id][path_tuple] = {'h5_type': h5_type}
    logger.debug(f"Found data for devices: {str(data_paths)}")
    return data_paths


def add_all_data_paths(meta_in: Dict[str, Any], device_dir: Path) -> Dict[str, Any]:
    """
    Build device data structure with metadata and data_paths for all devices found in device_dir.

    Args:
        meta_in: Input metadata dictionary
        device_dir: Device directory path

    Returns:
        Dictionary with device data structure containing metadata and data_paths for all devices found.
    """
    all_data_paths = get_all_data_files_for_device_dir(device_dir)
    meta_dev_default = {field: "?" for field in io_info_files.info_devices_field_names_extended}

    # Preserve order: devices from meta_in first, then devices only in all_data_paths
    all_dev_ids = list(meta_in.keys()) + [dev_id for dev_id in all_data_paths.keys() if dev_id not in meta_in]
    return {
        dev_id: {
            **meta_in.get(dev_id, meta_dev_default),
            "data_paths": all_data_paths.get(dev_id, {}),
        } for dev_id in all_dev_ids
    }

def get_prioritized_data_sources_for_time_extraction(devices_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[Tuple[Path, PurePosixPath], Dict]]]:
    """
    Get prioritized data sources for time extraction for each device, using the sort_data_paths function for proper prioritization.

    Args:
        devices_data: Dictionary with device data containing data_paths

    Returns:
        Dictionary mapping device IDs to sorted list of (path_tuple, metadata) tuples
    """
    prioritized_sources = {}

    for dev_id, dev_data in devices_data.items():
        data_paths = dev_data.get("data_paths", {})
        if data_paths:
            # Use the existing sort_data_paths function to get prioritized list
            # This will properly sort by priority: text files > proc_noAvg > proc > raw HDF5, etc.
            sorted_paths = sort_data_paths(data_paths, {dev_id})  # Only this device for specificity check
            prioritized_sources[dev_id] = sorted_paths
        else:
            prioritized_sources[dev_id] = []

    return prioritized_sources


def extract_time_metadata_from_prioritized_sources(
    device_id: str,
    prioritized_sources: List[Tuple[Tuple[Path, PurePosixPath], Dict]],
    extract_hdf5_times: bool = True,
    extract_hdf5_coef_dates: bool = False,
) -> Dict[str, Any]:
    """
    Try to extract time metadata from prioritized data sources until successful

    Args:
        device_id: Device ID to extract metadata for
        prioritized_sources: List of (path_tuple, metadata) tuples in priority order
        extract_hdf5_times: Whether to extract time metadata from HDF5 files
        extract_hdf5_coef_dates: Whether to extract coefficient dates from HDF5 files

    Returns:
        Dict with fields (start_time, end_time, burst_dt, bursts_t) or {} if not found
    """
    time_info = {}
    # First try to extract from text files in priority order
    for (parent_path, rel_path), metadata in prioritized_sources:
        # Check if this is an HDF5 or MAT file
        if parent_path.suffix.lower() in (".h5", ".mat"):
            if extract_hdf5_times:
                # For HDF5 files, use the HDF5 extraction function
                try:
                    # For a specific HDF5 file and table path, extract time range directly
                    time_info = extract_time_range_from_hdf5_table(parent_path, str(rel_path / "table"))
                    if time_info:  # we got valid time_info = (start_time, end_time, burst_dt, bursts_t)
                        break
                except Exception as e:
                    logger.warning(f"Could not extract time from HDF5 file {parent_path}: {e}")
        else:
            # This is a text file, try to extract time info
            try:
                # Extract time info from text file with averaging_interval for burst detection
                # This function returns a single tuple, not a dict with multiple devices
                # Get averaging_interval from metadata if available, otherwise use default
                averaging_interval = metadata.get("averaging_interval")
                if averaging_interval is None:
                    # If metadata doesn't have averaging_interval, use default from config
                    averaging_interval = config.default_text_file_averaging
                time_info = extract_time_info_from_text_file(parent_path, rel_path, averaging_interval)
                if time_info:  # time_info is (start_time, end_time, burst_dt, bursts_t)
                    break
            except Exception as e:
                logger.warning(f"Could not extract time from text file {parent_path / rel_path}: {e}")

    return dict(
        zip(
            ["time_st", "time_en", "burst_dt", "bursts_t"],
            time_info or ["?", "?", "", ""],
        )
    )


def _update_time_fields(target: Dict[str, Any], time_info: Dict[str, Any]) -> None:
    """Update time fields in target dict only when current values are placeholders.

    Args:
        target: Dictionary to update (device entry or interval dict)
        time_info: Dictionary with extracted time fields
    """
    for key, value in time_info.items():
        current_value = target.get(key, "?")
        if current_value in ["?", "", None, "-", "None"] and value not in ["?", "", None, "-", "None"]:
            target[key] = value


def update_device_metadata_with_time_info(devices_meta: Dict[str, Dict[str, Any]], extract_hdf5_times: bool = True, extract_hdf5_coef_dates: bool = False):
    """
    Update devices_data with time metadata extracted from prioritized data sources, while preserving all data paths for each device.

    Preserves existing metadata values (from info-file) when extraction returns placeholder values.
    Only updates fields when extracted values are valid (not placeholders: "?", "", "-", None).
    Handles devices with nested dict structure (multiple intervals) by updating each station_id separately.

    Args:
        devices_data: Dictionary with device data to update
        extract_hdf5_times: Whether to extract time metadata from HDF5 files
        extract_hdf5_coef_dates: Whether to extract coefficient dates from HDF5 files
    """
    # Get prioritized data sources for each device
    prioritized_sources_map = get_prioritized_data_sources_for_time_extraction(devices_meta)

    # Extract time metadata for each device
    for dev_id, prioritized_sources in prioritized_sources_map.items():
        try:
            device_entry = devices_meta[dev_id]
        except KeyError:
            pass
        else:
            time_info = extract_time_metadata_from_prioritized_sources(
                dev_id,
                prioritized_sources,
                extract_hdf5_times,
                extract_hdf5_coef_dates
            )

            # Try to update time fields for nested dict structure (multiple intervals)
            # If no station items are found, fall back to single interval structure
            has_nested = False
            for station_id, station_metadata in io_info_files.iter_station_id_items(device_entry):
                has_nested = True
                # Convert to field-name dict if station_metadata is a list;
                # if already a dict (from info_meta_list_to_dict), use directly
                if isinstance(station_metadata, dict):
                    _update_time_fields(station_metadata, time_info)
                else:
                    station_metadata_dict = info_meta_list_to_dict(station_metadata)
                    _update_time_fields(station_metadata_dict, time_info)
                    # Convert dict back to list and update in-place
                    updated_list = [
                        station_metadata_dict.get(field_name, "?")
                        for field_name in io_info_files.info_devices_field_names_extended
                    ]
                    for idx, new_value in enumerate(updated_list):
                        if idx < len(station_metadata):
                            station_metadata[idx] = new_value

            if not has_nested:
                # Single interval device - update device entry directly
                _update_time_fields(device_entry, time_info)

    # Also extract any additional HDF5 metadata if needed
    # todo: use discovered devices dir and found data sources, not rediscover
    if extract_hdf5_coef_dates or (
        hasattr(config, "raw_hdf5_cols") and config.raw_hdf5_cols and "raw_date_range" in config.raw_hdf5_cols
    ):
        # Get the device directory from one of the data paths
        device_dir = None
        for dev_id, dev_meta in devices_meta.items():
            for path_tuple, _ in dev_meta.get("data_paths", {}).items():
                dir_path, _ = path_tuple
                device_dir = dir_path.parent
                break
            if device_dir:
                break

        if device_dir:
            raw_hdf5_metadata = extract_raw_hdf5_metadata(
                device_dir,
                list(devices_meta.keys()),
                config.raw_hdf5_cols
            )

            # Update devices_data with raw HDF5 metadata
            for dev_id, metadata_dict in raw_hdf5_metadata.items():
                if dev_id in devices_meta:
                    for key, value in metadata_dict.items():
                        if value not in ["?", "", None, "-", "None"]:
                            devices_meta[dev_id][key] = value


def get_absent_meta(
    meta_in: Dict[str, Any],
    device_dir: Path,
    extract_hdf5_times: bool = None,
    extract_hdf5_coef_dates: bool = None,
) -> Dict[str, Any]:
    """
    Search for metadata typically present in DEVICES_FILE_NAME files but absent in input metadata dict.
    This function ensures all available data files are included for each device, regardless of whether
    higher-priority files (like text_output) are missing for that device ID.

    Args:
        meta_in: Dictionary mapping device IDs to their data {device_id: {**metadata, "data_paths":
            datafiles}} (metadata will be initialized with placeholders)
            where
                - metadata - dict with keys of info-file metadata described in readme
                - datafiles - dict of {(directory_path, file_path): metadata}
        device_dir: Directory containing the devices
        extract_hdf5_times: Whether to extract time metadata from HDF5 files (defaults to config.extract_hdf5_times)
        extract_hdf5_coef_dates: Whether to extract coefficient dates from HDF5 files (defaults to presence of
            "coef_date" in config.raw_hdf5_cols)

    Returns:
        Dictionary representing DEVICES_FILE_NAME content with associated data files
    """
    if extract_hdf5_times is None:
        extract_hdf5_times = config.extract_hdf5_times
    if extract_hdf5_coef_dates is None:
        extract_hdf5_coef_dates = config.extract_hdf5_coef_dates

    # Create the initial devices_data structure with all available data paths for all devices
    devices_data = add_all_data_paths(meta_in, device_dir)

    # Update device metadata with time information extracted from data sources
    update_device_metadata_with_time_info(devices_data, extract_hdf5_times, extract_hdf5_coef_dates)

    # Prepare the final content — ensure all devices have their complete data paths
    content = {}
    for dev_id, dev_data in devices_data.items():
        sorted_data_paths = dict(sort_data_paths(dev_data["data_paths"], {dev_id}))
        content[dev_id] = {
            **dev_data,
            "data_paths": sorted_data_paths,
            "combined_comments": {},
        }
    logger.debug(f"get_absent_meta result: {content}")
    return content

def _apply_combined_comments_to_devices(content: Dict[str, Any]) -> None:
    """
    Apply combined file comments to device metadata where appropriate.

    Args:
        content: The device content dictionary to update
    """

    for device_id, entry in content.items():
        # If there are text_outputs and combined_comments, apply them
        # combined_comments is a dict at this point in get_absent_meta
        combined_comments_all = entry.get("combined_comments", {})

        # Process each combined comment entry to find relevant ones for this device
        for path_str, combined_comments_per_file in combined_comments_all.items():
            # Check if this device is part of any combined comment in this file
            for combined_key, comment in combined_comments_per_file.items():
                if device_id in combined_key:
                    # Add the special comment to indicate combined data
                    # If there's already a comment from metadata file, append to it
                    new_comment = (
                        comment
                        if (cmt := entry.get("comment", None)) in (None, "", "-", "?")
                        else f"{cmt}; {comment}"
                    )
                    entry["comment"] = new_comment
                    break


def process_all_metadata(
        cruise_and_its_dev_dirs: Dict[Path, List[Path]],
        from_data: bool = True,
        extract_hdf5_times: bool = True,
        extract_hdf5_coef_dates: bool = False,
        create_info_files: bool = False,
    ) -> Tuple[Dict[Path, Dict[str, Any]], Dict[str, Any]]:
    """
    Process all metadata from Cruises directories of known structure using the new simplified functions.
    This function replicates functionality of original process_all_metadata but using the new
    simplified functions that inherently handle of bug where missing text_output and .proc.h5 files
    cause other files to be excluded from data_paths.

    Args:
        cruise_and_its_dev_dirs: Dictionary mapping cruise directories to their device directories
        create_info_files: Whether to update DEVICES_FILE_NAME files if they don't exist

    Returns:
        Tuple of (processed_meta, stats) where:
        - processed_meta: Dictionary mapping device directories to their metadata
        - stats: Dictionary with statistics about processing including new devices found
    """

    total_device_dirs = sum(len(device_dirs) for device_dirs in cruise_and_its_dev_dirs.values())
    logger.info(f"Processing {len(cruise_and_its_dev_dirs)} cruises having {total_device_dirs} device directories")
    processed_meta = {}

    # Statistics collection
    stats = {
        "info_files_modified": [],
        "new_devices_found": {},  # Maps device_dir path to list of new device IDs
        "devices_with_time_data": set(),
        "processing_errors": [],  # Track processing errors for logging
    }

    # Extract metadata from data files and info-files.
    for cruise_dir, device_dirs in cruise_and_its_dev_dirs.items():
        logger.info(f"Processing cruise: {cruise_dir.name}")

        for device_dir in device_dirs:
            try:
                # Read metadata from info-file (try names in priority order: UPD -> YAML -> original)
                meta_from_info_file = {}
                for file_name in (
                    info_files := (DEVICES_FILE_NAME_UPD, DEVICES_FILE_NAME_YAML, DEVICES_FILE_NAME)
                ):
                    if (info_device_file := device_dir / file_name).exists():
                        meta_from_info_file = read_metadata_files_to_dict(info_device_file)
                        break
                else:
                    logger.warning(f"No any of {info_files} metadata info files found in {device_dir}")

                if from_data:
                    # Extract metadata from data files and combine with info-file metadata
                    extracted_devices_data = get_absent_meta(
                        meta_from_info_file,
                        device_dir,
                        extract_hdf5_times=extract_hdf5_times,
                        extract_hdf5_coef_dates=extract_hdf5_coef_dates,
                    )
                    # Track new devices found (in extracted data but not in info-file metadata)
                    new_devices_in_dir = [d_id for d_id in extracted_devices_data if d_id not in meta_from_info_file]
                    if new_devices_in_dir:
                        stats["new_devices_found"][str(device_dir)] = new_devices_in_dir
                else:
                    # Only use metadata from existing info-files without extracting from data files
                    extracted_devices_data = {
                        dev_id: {**meta, "data_paths": {}}
                        for dev_id, meta in meta_from_info_file.items()
                    }

                # Process combined file comments for each device based on their sorted data paths
                _apply_combined_comments_to_devices(extracted_devices_data)

                # Process metadata to handle GPX fallback
                coord_fallback_from_gpx(cruise_dir, device_dir, extracted_devices_data)

                # Track devices with time data
                for dev_id, dev_data in extracted_devices_data.items():
                    if dev_data.get("time_st") not in ["?", "", None, "-"]:
                        stats["devices_with_time_data"].add(f"{device_dir}:{dev_id}")

                # If create_info_files is True, update/create it using the extracted metadata
                if create_info_files and update_devices_meta_file(
                    device_dir=device_dir, content=extracted_devices_data
                ):
                    stats["info_files_modified"].append(str(device_dir))

                processed_meta[device_dir] = extracted_devices_data

            except Exception as e:
                logger.error(f"Error processing {device_dir}: {e}", exc_info=True)
                stats["processing_errors"].append({
                    "device_dir": str(device_dir), "error": str(e), "error_type": type(e).__name__
                })
                continue


    # Build unique dataset names per device dir from cruise and device dir names.
    # add_dataset_name may rename earlier entries during disambiguation.
    # DD inferral from extracted time metadata is handled inside add_dataset_name.
    used_datasets_paths: Dict[str, Path] = {}
    dev_dir_dates: Dict[Path, str] = {}
    for cruise_dir, device_dirs in cruise_and_its_dev_dirs.items():
        for device_dir in device_dirs:
            try:
                _, dev_dir_dates[device_dir] = add_dataset_name(
                    device_dir, cruise_dir, used_datasets_paths, processed_meta
                )
            except Exception as e:
                logger.error(f"Error building dataset name for {device_dir}: {e}", exc_info=True)
                stats["processing_errors"].append({
                    "device_dir": str(device_dir), "error": str(e), "error_type": type(e).__name__
                })

    # Set processed_meta items "setup_name" to final dataset names
    # Also map setup_names (used_datasets_paths keys) to to their dates (dev_dir_dates values) to sort later
    dataset_date_map = {}
    for final_name, dev_dir in used_datasets_paths.items():
        for dev_data in processed_meta.get(dev_dir, {}).values():
            dev_data["setup_name"] = final_name
        dataset_date_map[final_name] = dev_dir_dates[dev_dir]

    logger.info(f"Processed {len(processed_meta)} device directories")
    if stats["processing_errors"]:
        logger.warning(f"Encountered {len(stats['processing_errors'])} errors during processing")

    return (processed_meta, stats, dataset_date_map)


def _add_gpx_reference_to_metadata(
    metadata_dict: Dict[str, Any],
    cruise_dir: Path,
    device_dir: Path,
    dev_id: str,
    station_id: str = None
) -> bool:
    """Add GPX file reference to metadata when coordinates are missing.

    Args:
        metadata_dict: Device metadata dictionary to update
        cruise_dir: Cruise directory path for GPX search
        device_dir: Device directory path for GPX search
        dev_id: Device ID for logging
        station_id: Station ID for logging (None for single-interval devices)

    Returns:
        True if GPX reference was added, False otherwise
    """
    lat = metadata_dict.get("lat", "?")
    lon = metadata_dict.get("lon", "?")

    if lat != '?' and lon != '?':
        return False

    gpx_files = find_navigation_files(device_dir)
    if not gpx_files:
        gpx_files = find_navigation_files(cruise_dir)

    if not gpx_files:
        location_desc = f"device {dev_id} station {station_id}" if station_id else f"device {dev_id}"
        logger.warning(
            f"No GPX files found for {location_desc} in cruise directory {cruise_dir}. "
            f"Searched in device directory: {device_dir}"
        )
        return False

    # Store comma-separated paths in 'comment' as relative to cruise directory
    gpx_paths = ",".join(str(p.relative_to(cruise_dir)) for p in gpx_files)
    new_comment = (
        f"{cmt}; GPX: {gpx_paths}"
        if (cmt := metadata_dict.get("comment", "")) and cmt not in ['?', '-', '']
        else f"GPX: {gpx_paths}"
    )
    metadata_dict["comment"] = new_comment
    return True


def coord_fallback_from_gpx(cruise_dir: Path, device_dir: Path, devices_data: Dict[str, Any]) -> None:
    """Add GPX file references to device metadata when coordinates are missing.

    Handles both single-interval and multi-interval (nested) device structures.
    """
    for dev_id, entry in devices_data.items():
        # Try to process nested dict structure (multiple intervals) first
        has_nested = False
        for station_id, station_metadata in io_info_files.iter_station_id_items(entry):
            has_nested = True
            # Convert to field-name dict if station_metadata is a list;
            # if already a dict (from info_meta_list_to_dict), use directly
            if isinstance(station_metadata, dict):
                _add_gpx_reference_to_metadata(
                    station_metadata, cruise_dir, device_dir, dev_id, station_id
                )
            else:
                station_metadata_dict = info_meta_list_to_dict(station_metadata)
                if _add_gpx_reference_to_metadata(
                    station_metadata_dict, cruise_dir, device_dir, dev_id, station_id
                ):
                    # Convert dict back to list and update in-place
                    updated_list = [
                        station_metadata_dict.get(field_name, "?")
                        for field_name in io_info_files.info_devices_field_names_extended
                    ]
                    for idx, new_value in enumerate(updated_list):
                        if idx < len(station_metadata):
                            station_metadata[idx] = new_value

        # Single interval device - check lat/lon directly
        if not has_nested:
            _add_gpx_reference_to_metadata(entry, cruise_dir, device_dir, dev_id, station_id=None)


def main(**kwargs):
    """
    Main workflow function that orchestrates the entire metadata processing
    Args:
        top_search_dirs: List of directories to search for cruise data
        config_data: Configuration data (will use global config if not provided)

    Returns:
        Dictionary containing the results of the processing workflow
    """

    # Use provided kwargs or global config: update global config with provided values
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Log all configuration parameters at start of program execution
    logger.info("=" * 80)
    logger.info("Configuration parameters:")
    logger.info("=" * 80)
    from dataclasses import fields
    for field in fields(Config):
        value = getattr(config, field.name)
        logger.info(f"  {field.name}: {value}")
    logger.info("=" * 80)

    # Initialize variables for error handling
    meta = {}
    stats = {
        "info_files_modified": [],
        "new_devices_found": {},
        "devices_with_time_data": set(),
        "processing_errors": [],
    }

    # Generate timestamp at the start for consistent output file naming
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    try:
        # Find cruise directories
        # Use provided cruise_dir if available, otherwise search in top_search_dirs
        cruise_dir_list = config.input_dirs if config.input_dirs else None
        cruise_to_dev_dirs = discover_device_dirs(config.top_search_dirs, input_dirs=cruise_dir_list)

        # Process all metadata
        meta, stats, dataset_date_map = process_all_metadata(
            cruise_to_dev_dirs,
            **{
                k: getattr(config, k)
                for k in [
                    "from_data",
                    "extract_hdf5_times",
                    "extract_hdf5_coef_dates",
                    "create_info_files",
                ]
            },
        )

    except Exception as e:
        # Catch critical errors at main level
        logger.error(f"Critical error in main processing: {e}", exc_info=True)
        stats["processing_errors"].append({
            "location": "main",
            "error": str(e),
            "error_type": type(e).__name__
        })
        dataset_date_map = {}

    finally:
        # Always attempt to save collected data, even if errors occurred
        # Print statistics and warnings at the end
        if stats.get('info_files_modified'):
            logger.info("\n".join(["Info-file files modified: "] + [
                f" {f}" for f in stats['info_files_modified']]))
        if stats.get('new_devices_found'):
            # New devices were discovered and processed in this run - time data already extracted
            logger.info(
                f"New devices found and processed in {len(stats['new_devices_found'])} directories:"
            )
            for device_dir, new_devices in stats['new_devices_found'].items():
                logger.info(f"  {device_dir}: {', '.join(new_devices)}")

        # Log processing errors if any
        if stats.get('processing_errors'):
            logger.error(f"*** Processing completed with {len(stats['processing_errors'])} errors: ***")
            for error_info in stats['processing_errors']:
                logger.error(
                    "  {}: {}".format(
                        error_info["device_dir" if "device_dir" in error_info else "location"],
                        error_info["error"],
                    )
                )

        logger.info(
            f"Processing complete: "
            f"{len(stats.get('info_files_modified', []))} info-files modified, "
            f"{len(stats.get('devices_with_time_data', set()))} devices with time data extracted"
        )

        # Save collected data if any exists
        if meta:
            # Prepare output directory
            output_dir = Path(config.output_dir if (hasattr(config, 'output_dir') and config.output_dir) else "meta")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Write files list (using timestamp generated at start)
            if "tsv" in config.output_format:
                try:
                    files_tcm_path = output_dir / f"{timestamp}_files_TCM.tsv"
                    write_files_list(meta, files_tcm_path, dataset_date_map)
                    logger.info(f"Saved files list to {files_tcm_path}")

                    # Write metadata table
                    meta_tcm_path = output_dir / f"{timestamp}_meta_TCM.tsv"
                    write_metadata_table(meta, meta_tcm_path, dataset_date_map)
                    logger.info(f"Saved metadata table to {meta_tcm_path}")
                except Exception as write_error:
                    logger.error(f"Error writing output files: {write_error}", exc_info=True)
                    stats["processing_errors"].append({
                        "location": "output_writing",
                        "error": str(write_error),
                        "error_type": type(write_error).__name__
                    })
        else:
            logger.warning("No metadata collected - skipping output file generation")
            files_tcm_path = None
            meta_tcm_path = None

    print(">")
    return {
        "device_dir_metadata": meta,
        "output_files": {
            "files_tcm": str(files_tcm_path) if files_tcm_path is not None else None,
            "meta_tcm": str(meta_tcm_path) if meta_tcm_path is not None else None,
        },
        "stats": stats,
    }


if __name__ == "__main__":
    sys.exit(not main())  # have data <=> success
