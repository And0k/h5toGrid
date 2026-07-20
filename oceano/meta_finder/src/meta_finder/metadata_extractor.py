"""
Metadata extraction functions for TCM Metadata Processor.
"""

import json
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from .parse_data_file_name import normalize_device_id
from .logging_config import setup_logging
from . import io_info_files
from .config import DEVICES_FILE_NAME, DEVICES_FILE_NAME_YAML, DEVICES_FILE_NAME_UPD
logger = setup_logging()

def info_meta_list_to_dict(json_metadata_list):
    """Add fields from metadata list"""
    entry = {
        field_name: val
        for field_name, val in zip(io_info_files.info_devices_field_names_extended, json_metadata_list)
    }
    for field_name in io_info_files.info_devices_field_names_extended[len(json_metadata_list):]:
        entry[field_name] = (
            "" if field_name in ("burst_dt", "bursts_t", "coef_date", "time_raw_st", "time_raw_en") else "?"
        )

    # Parse and format datetime fields correctly
    from datetime import datetime
    datetime_fields = ["time_st", "time_en", "coef_date", "time_raw_st", "time_raw_en"]
    for field_name in datetime_fields:
        if field_name in entry and entry[field_name] not in ["?", "", None, "-", "None"]:
            try:
                # Try to parse the datetime string
                dt_str = entry[field_name]
                # Handles both 'YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS' formats
                dt_obj = datetime.fromisoformat(dt_str)

                # Format with space separator and ensure seconds are always two digits
                formatted_dt = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                entry[field_name] = formatted_dt
            except TypeError as e:
                # If parsing fails due to type error, check if it's already a datetime object
                if isinstance(entry[field_name], datetime):
                    # Already a datetime object - just format it
                    formatted_dt = entry[field_name].strftime("%Y-%m-%d %H:%M:%S")
                    entry[field_name] = formatted_dt
                else:
                    logger.warning(f"Could not parse datetime field {field_name} with value '{entry[field_name]}': {e}")
            except Exception as e:
                # If parsing fails for other reasons, leave as-is but log warning
                logger.warning(f"Could not parse datetime field {field_name} with value '{entry[field_name]}': {e}")

    return entry

def read_metadata_files(json_path: Path) -> Dict[str, Any]:
    """Extract metadata from metadata files in priority order:
    1. info_devices@meta_finder.yaml (highest priority)
    2. info_devices.yaml (replaces info_devices.json if present)
    3. info_devices.json (fallback)

    Args:
        json_path: Path to DEVICES_FILE_NAME file

    Returns:
        Dictionary where keys are device IDs and values are metadata lists
    """
    # Define the files to check in order of priority
    metadata_paths = [
        json_path.with_name(DEVICES_FILE_NAME_UPD),  # Highest priority: @meta_finder.yaml
        json_path.with_name(DEVICES_FILE_NAME_YAML),  # Second priority: .yaml
        json_path  # Lowest priority: original .json
    ]

    meta = None
    have_reading_error = False

    # Loop through files in priority order
    for file_path in metadata_paths:
        logger.debug(f"Attempting to extract metadata from {file_path}")

        try:
            meta = io_info_files.read_metadata_file(file_path)
            logger.debug(f"Successfully loaded metadata from {file_path}")
            break  # Break if successful
        except FileNotFoundError:
            continue
        except Exception as e:  # for example UnicodeDecodeError
            have_reading_error = True
            logger.warning(f"Error reading {file_path}: {e}")
            continue

    # Log results
    if logger.isEnabledFor(logging.DEBUG):
        if meta:
            logger.debug(f"Successfully extracted metadata from file containing {len(meta)} device entries: {list(meta.keys())}")
            # Log the actual content for debugging, focusing on important fields like time
            for device_id, metadata_list in meta.items():
                logger.debug(f'  Device "{device_id}": {metadata_list}')
        else:
            logger.debug("No valid metadata found in any checked files")

    return meta if meta is not None else {}

def read_metadata_files_to_dict(json_path: Path) -> Dict[str, Dict[str, Any]]:
    """Extract metadata from json_path (or DEVICES_FILE_NAME_UPD instead if exists) file and convert lists to dictionaries.

    Args:
        json_path: Path to DEVICES_FILE_NAME file

    Returns:
        Dictionary where keys are device IDs and values are metadata dictionaries
    """
    logger.debug(f"Extracting metadata from {json_path} and converting to dict format")
    result = {}
    json_data = read_metadata_files(json_path)
    if not json_data:
        logger.warning(
            "No metadata found in %s, %s, or %s",
            DEVICES_FILE_NAME_UPD,
            DEVICES_FILE_NAME_YAML,
            json_path.name,
        )
        return result

    # Check if all values are placeholders
    if io_info_files.all_vals_empty(json_data):
        logger.warning(
            "All values in %s, %s, or %s are placeholders",
            DEVICES_FILE_NAME_UPD,
            DEVICES_FILE_NAME_YAML,
            json_path.name,
        )
        return result

    # Convert metadata lists to dictionaries
    # Space-prefixed device IDs are already filtered by read_metadata_file()
    for device_id, metadata_list in json_data.items():
        # Normalize device_id, to prevent overwriting, only if has no trailing underscores
        result_device_id = device_id if device_id.endswith('_') else normalize_device_id(device_id)

        # Handle formats:
        # 1. List/tuple: Single interval device, convert to dict
        # 2. Dict with string keys: Multiple intervals (nested dict structure), convert all keys to strings
        if isinstance(metadata_list, (list, tuple)):
            # Single interval device - convert to dict with key "0"
            converted_metadata = {"0": info_meta_list_to_dict(metadata_list)}
        elif isinstance(metadata_list, dict):
            # Nested dict structure from read_metadata_file: {station_id: metadata_list}
            # Convert each metadata list to a field-name dict
            converted_metadata = {
                str(k): (
                    info_meta_list_to_dict(v) if isinstance(v, (list, tuple)) else v
                )
                for k, v in metadata_list.items()
            }
        else:
            logger.warning(
                f"Unexpected metadata type for device {device_id}: {type(metadata_list)}, skipping"
            )
            continue

        result[result_device_id] = converted_metadata
        logger.debug(
            "Converted metadata for device %s%s: %s",
            result_device_id,
            " (normalized)" if result_device_id != device_id else "",
            converted_metadata,
        )
    return result

# Not used: Process devices with trailing '_' suffixes to construct comments for corresponding devices
# result = process_trailing_underscore_devices(result)

# if logger.isEnabledFor(logging.DEBUG):
#     logger.debug("Final converted metadata result with %d entries: %s", len(result), result)
#     # Log the final converted content details for each device
#     for device_id, metadata_dict in result.items():
#         logger.debug(
#             "  Final converted data for device %s: %s",
#             device_id,
#             {k: v for k, v in metadata_dict.items() if v not in [None, "", "?"]},
#         )

def extract_coordinates_from_gpx(gpx_path: Path, points) -> Optional[Tuple[float, float]]:
    """Extract coordinates from a .gpx file.

    Args:
        gpx_path: Path to .gpx file

    Returns:
        Tuple of (latitude, longitude) or None if no coordinates found
    """
    logger.info(f"Extracting coordinates from {gpx_path}")
    points = points.copy()
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()

        # Define namespace
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

        # Try to find waypoints first
        waypoints = root.findall('.//gpx:wpt', ns)
        for waypoint in waypoints:
            if waypoint in points:
                points[waypoint]["lat"] = lat = float(waypoint.get('lat'))
                points[waypoint]["lon"] = lon = float(waypoint.get('lon'))
                logger.debug(f"Found coordinates in waypoint: lat={lat}, lon={lon}")

        # Check if any coordinates were found
        if not waypoints:
            logger.warning(f"No waypoints found in {gpx_path}")
            return None

        logger.debug(f"Successfully processed {len(waypoints)} waypoints from {gpx_path}")
        return points
    except Exception as e:
        logger.error(f"Error reading {gpx_path}: {e}")
        return None
