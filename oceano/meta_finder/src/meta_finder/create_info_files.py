#!/usr/bin/env python
"""
Script to create DEVICES_FILE_NAME files for cruises with inclinometer or wavegauge directories.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from itertools import zip_longest

from . import config
from . import io_info_files
from .config import DEVICES_FILE_NAME_UPD, DEVICES_FILE_NAME_YAML
from datetime import datetime
from .logging_config import setup_logging
logger = setup_logging()

def _format_for_devices_meta_file(content: Dict[str, Any]) -> Dict[str, List]:
    """
    Format content for JSON output by converting content values to the lists of expected format,
    trims metadata to expected format (11 elements)
    While the internal metadata may have additional elements for raw HDF5 data (coef_date, time_raw_st,
    time_raw_en), the expected format should have 11 elements:
    0: point
    1: sea_depth
    2: height above bottom
    3: modification symbol
    4: latitude
    5: longitude
    6: time_st (start time)
    7: time_en (end time)
    8: burst_dt
    9: bursts_t
    10: comment

    Args:
        content: Dictionary with device IDs as keys and metadata (list or dict) as values

    Returns:
        Formatted content with metadata trimmed to expected format
        with datetime objects converted to "%Y-%m-%d %H:%M:%S" string
    """
    formatted_content = {}
    for device_id, device_entry in content.items():
        logger.debug(f"Formatting device {device_id}: type={type(device_entry)}, keys={list(device_entry.keys())}")
        if not isinstance(device_entry, dict):
            raise ValueError("device_entry must be a dict")

        # Check if device has nested dict structure (multiple intervals)
        # Nested dict: keys are station IDs (any string/int), values are metadata lists
        # Single interval: keys are field names (point, sea_depth, etc.), values are field values
        #
        # Challenge: Device entries may have BOTH station IDs AND metadata keys (e.g., 'data_paths', 'cruise')
        # Example: {'0': [...], 'data_paths': {...}, 'cruise': '...'}
        #
        # Solution: Check if there are any list/tuple values (metadata lists)
        # If present, extract those as nested dict structure, ignoring metadata keys

        # Check if device has nested dict structure (multiple intervals)
        # Nested dict: keys are station IDs, values are metadata lists
        # Single interval: keys are field names, values are field values
        has_nested_structure = any(io_info_files.iter_station_id_items(device_entry))

        if has_nested_structure:
            # Nested dict structure - multiple intervals (station_id -> metadata)
            # Extract only the station ID -> metadata entries, ignore metadata keys
            # Convert field-name dicts to lists for output format
            nested_dict = {
                k: io_info_files._station_metadata_to_list(v)
                for k, v in io_info_files.iter_station_id_items(device_entry)
            }
            logger.debug(
                f"Device {device_id} has nested dict structure with {len(nested_dict)} intervals, "
                f"ignoring metadata keys: {[k for k in device_entry.keys() if k not in nested_dict]}"
            )
            formatted_content[device_id] = nested_dict
        else:
            # Single interval device - create a list of values in expected order
            # Extract only the fields that are part of the expected JSON format
            metadata_list = [
                value.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(value := device_entry.get(field_name, "?"), datetime)
                else value
                for field_name in io_info_files.info_devices_field_names_extended[:11]
            ]
            logger.debug(f"Formatted metadata for {device_id}: {metadata_list}")
            formatted_content[device_id] = metadata_list
    return formatted_content


def _content_data_equals(a: Dict[str, Any], b: Dict[str, Any], normalize_keys: bool = False) -> bool:
    """Compare two device metadata dicts for data equality, ignoring structure differences.

    Handles the case where one dict has nested station structure {'0': [vals]}
    and the other has flat list [vals]. Compares values element-by-element
    after converting both to a canonical form.

    Args:
        a: First device metadata dictionary
        b: Second device metadata dictionary
        normalize_keys: If True, normalize device IDs before comparing

    Returns:
        True if both dicts contain the same device data
    """
    from .parse_data_file_name import normalize_device_id

    def _to_station_lists(entry: Any) -> Dict[str, list]:
        """Convert any device entry to {station_id: [values]} form."""
        if isinstance(entry, dict) and any(io_info_files.iter_station_id_items(entry)):
            return {sid: io_info_files._station_metadata_to_list(v) for sid, v in io_info_files.iter_station_id_items(entry)}
        if isinstance(entry, (list, tuple)):
            return {"0": list(entry)}
        return {}

    def _strip_trailing_empty(vals: list) -> list:
        """Remove trailing placeholder/empty values after time_en (index 6)."""
        time_en_idx = 6
        if len(vals) <= time_en_idx + 1:
            return vals
        end = len(vals)
        while end > time_en_idx + 1 and (vals[end - 1] in ("", "?", "-", None)):
            end -= 1
        return vals[:end]

    def _vals_equal(va: list, vb: list) -> bool:
        sa, sb = _strip_trailing_empty(va), _strip_trailing_empty(vb)
        if len(sa) != len(sb):
            return False
        return all(str(x) == str(y) for x, y in zip(sa, sb))

    def _norm_key(k: str) -> str:
        return normalize_device_id(k) or k

    keys_a = {_norm_key(k) if normalize_keys else k for k in a}
    keys_b = {_norm_key(k) if normalize_keys else k for k in b}
    if keys_a != keys_b:
        return False

    for k in a:
        nk = _norm_key(k) if normalize_keys else k
        matching_b_key = next((bk for bk in b if (_norm_key(bk) if normalize_keys else bk) == nk), None)
        if matching_b_key is None:
            return False
        sl_a = _to_station_lists(a[k])
        sl_b = _to_station_lists(b[matching_b_key])
        if sl_a.keys() != sl_b.keys():
            return False
        for sid in sl_a:
            if not _vals_equal(sl_a[sid], sl_b[sid]):
                return False
    return True


def _device_sort_key(device_id: str):
    """Sort key extracting (type_model_prefix, numeric) from device ID for deterministic ordering.

    Uses config.ptn_device_id_named_parts (same pattern as normalize_device_id) to extract
    named groups (type, model, number). Examples: 'i03' -> ('i', 3), 'w01' -> ('w', 1),
    'ib27' -> ('ib', 27). Placeholder keys like '?' sort last.
    """
    if not device_id or device_id.strip() in ("?", "-", ""):
        return ("zzz", 0)
    d = device_id.lower().strip()
    if not (m := re.match(f"^{config.config.ptn_device_id_named_parts}$", d, re.IGNORECASE)):
        return ("zzz" + device_id, 0)
    type_part = m["type"] or ""
    model_part = m["model"] or ""
    prefix = f"{type_part}{model_part}"
    num = int(m["number"]) if m["number"] else 0
    return (prefix, num)


def _sort_devices_by_key(content: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with devices sorted by type/model prefix then number."""
    return dict(sorted(content.items(), key=lambda item: _device_sort_key(item[0])))


def _merge_device_metadata(
    existing_content: Dict[str, Any],
    new_content: Dict[str, Any],
    normalize_keys: bool = False
) -> Dict[str, Any]:
    """
    Merge device metadata from new content into existing content.

    Preserves order of existing devices and adds new devices sorted by type/model and number.
    Handles both single interval and nested dict (multiple intervals) structures.

    Args:
        existing_content: Existing device metadata dictionary
        new_content: New device metadata dictionary to merge in
        normalize_keys: If True, normalize device IDs from existing_content for comparison
                      (used when merging from info_devices.yaml which may have non-normalized IDs)

    Returns:
        Merged device metadata dictionary
    """
    from .parse_data_file_name import normalize_device_id

    merged_content = {}

    # First, process existing devices in their original order
    for d_id, vals in existing_content.items():
        # Normalize device ID for comparison if requested
        lookup_id = normalize_device_id(d_id) if normalize_keys else d_id

        if lookup_id in new_content:
            # Device exists in both files, merge values
            existing_vals = vals
            new_vals = new_content[lookup_id]

            # Check if existing_vals or new_vals is a nested dict (multiple intervals)
            existing_is_nested = (
                isinstance(existing_vals, dict)
                and any(io_info_files.iter_station_id_items(existing_vals))
            )
            new_is_nested = (
                isinstance(new_vals, dict)
                and any(io_info_files.iter_station_id_items(new_vals))
            )
            logger.debug(
                f"Device {d_id} (lookup: {lookup_id}): existing_is_nested={existing_is_nested}, "
                f"new_is_nested={new_is_nested}"
            )

            # Normalize both sides to {station_id: [values]} form for uniform merging
            # A flat list is treated as {"0": flat_list}
            def _to_station_dict(entry: Any) -> Dict[str, list]:
                if isinstance(entry, dict) and any(io_info_files.iter_station_id_items(entry)):
                    return {
                        sid: io_info_files._station_metadata_to_list(v)
                        for sid, v in io_info_files.iter_station_id_items(entry)
                    }
                if isinstance(entry, (list, tuple)):
                    return {"0": list(entry)}
                return {}

            existing_stations = _to_station_dict(existing_vals)
            new_stations = _to_station_dict(new_vals)

            if existing_stations and new_stations:
                # Merge station-by-station, keeping existing non-placeholder values
                logger.debug(
                    f"Device {d_id}: merging {len(existing_stations)} existing stations "
                    f"with {len(new_stations)} new stations"
                )
                merged_dict = {}
                for station_id, existing_list in existing_stations.items():
                    if station_id in new_stations:
                        new_list = new_stations[station_id]
                        merged_dict[station_id] = [
                            new_val if existing_val in ["?", "", None, "-"] else existing_val
                            for existing_val, new_val in zip_longest(
                                existing_list, new_list, fillvalue="?"
                            )
                        ]
                    else:
                        merged_dict[station_id] = existing_list
                # Add any new stations not present in existing
                for station_id, new_list in new_stations.items():
                    if station_id not in merged_dict:
                        merged_dict[station_id] = new_list
                merged_content[lookup_id] = merged_dict
            elif new_stations:
                # Only new has valid station data
                logger.debug(f"Device {d_id}: using new values (existing has no station data)")
                merged_content[lookup_id] = new_vals
            else:
                # Only existing has valid station data, preserve it
                logger.debug(f"Device {d_id}: preserving existing (new has no station data)")
                merged_content[lookup_id] = existing_vals
        else:
            # Device only exists in existing file, preserve it
            output_id = normalize_device_id(d_id) if normalize_keys else d_id
            merged_content[output_id] = vals

    # Collect new devices not yet in merged_content, then append them sorted by type/model and number
    new_device_items = {
        d_id: vals for d_id, vals in new_content.items() if d_id not in merged_content
    }
    for d_id, vals in sorted(new_device_items.items(), key=lambda item: _device_sort_key(item[0])):
        merged_content[d_id] = vals

    return merged_content


def update_devices_meta_file(device_dir: Path, content: Dict[str, List]) -> bool:
    """Update or create DEVICES_FILE_NAME_UPD file in the specified directory.

    This function handles selective updating of individual device entries:
    - If the file contains only placeholder values ("?", "-", or ""), it will be fully overwritten
    - If both existing and new content have real data, merges values field-by-field:
      * Keeps existing non-placeholder values
      * Updates placeholder fields with new values
    - Preserves the order of existing devices and adds new devices at the end
    - Treats "-", "?", "", None as placeholders for individual device values

    Args:
        device_dir: Directory where to create the file
        content: Dictionary with device IDs as keys and metadata (list or dict) as values

    Returns:
        True if file was created or updated, False otherwise
    """
    # Validate that device_dir is actually a directory
    if not device_dir.is_dir():
        logger.warning(f"Skipping {device_dir} as it is not a directory")
        return False


    # Check if file already exists
    file_out = device_dir / DEVICES_FILE_NAME_UPD
    file_keep = device_dir / DEVICES_FILE_NAME_YAML
    # Whether info_devices.yaml exists — determines if @meta_finder can be deleted as duplicate
    have_file_keep = file_keep.exists()

    if (have_file_out := file_out.exists()):
        if not config.overwrite_bad_devs_in_info_files:
            logger.info(
                f"Skipping update existed {file_out.name}: config.overwrite_bad_devs_in_info_files is 0"
            )
            return False
        content_use = _format_for_devices_meta_file(content)

        # Check if new content has valid device names (not "?")
        # Don't allow overwriting existing device names with placeholder "?"
        content_is_bad = not content_use or all(d_id == "?" for d_id in content_use)
        if content_is_bad:
            # New content is just a placeholder, don't overwrite existing with device info
            logger.warning(
                f"New content {str(content)} has only placeholders, skipping overwrite of "
                f"existing {file_out}"
            )
            # Log the content of existing file that will be preserved
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    existing_content = io_info_files.read_metadata_file(file_out)
                    logger.debug(f"Content of existing file being preserved: {existing_content}")
                except Exception as e:
                    logger.error(f"Error reading existing info file {file_out.name}: {e}", exc_info=True)
            return False
        else:
            # New content has real data, now check existing file
            try:
                existing_content = io_info_files.read_metadata_file(file_out)

                # Check if existing content has only empty values
                if io_info_files.all_vals_empty(existing_content):
                    logger.info(
                        f"Overwrite existing file {file_out.name} that has only placeholder values: "
                        f"{existing_content}"
                    )
                    content_write = _sort_devices_by_key(content_use)
                else:
                    # Both existing and new content have real data, do selective overwrite
                    logger.debug(f"Analyze {file_out.name} for selective update")
                    # Merge new values with existing, updating only placeholder fields
                    # Preserve the order of existing devices and add new devices at the end
                    content_write = _merge_device_metadata(
                        existing_content, content_use, normalize_keys=False
                    )
            except UnicodeDecodeError as e:
                logger.error(
                    f"Encoding error reading existing info file {file_out}: {e}. "
                    f"Problematic byte at position {e.start} is "
                    f"0x{ord(e.object[e.start : e.start + 1]) if e.start < len(e.object) else 0:02x}. "
                    "This often occurs with files containing non-ASCII characters (e.g., Russian letters) "
                    "saved with different encodings.",
                    exc_info=True,
                )
                import sys, traceback
                print(
                    f"ERROR: Encoding error reading existing info file {file_out.name}: {e}", file=sys.stderr
                )
                print("Traceback (most recent call last):", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return False  # If we can't read the file, don't overwrite
            except Exception as e:
                logger.error(f"Error reading existing info file {file_out}: {e}", exc_info=True)
                import sys, traceback
                print(f"ERROR: Error reading existing info file {file_out.name}: {e}", file=sys.stderr)
                print("Traceback (most recent call last):", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return False  # If we can't read the file, don't overwrite
    else:
        logger.debug(f"Creating new file: {file_out.name}")
        # Format the content for JSON output - trim metadata to expected format (11 elements)
        content_use = _format_for_devices_meta_file(content)
        existing_content = {}

        # Check if info_devices.yaml exists and read it to preserve its content
        if file_keep.exists():
            logger.info(f"Reading existing {file_keep.name} to preserve its content")
            try:
                yaml_content = io_info_files.read_metadata_file(file_keep)
                if yaml_content:
                    # Use YAML content as existing content for selective update
                    existing_content = yaml_content
                    have_file_keep = True
                    logger.debug(f"Loaded {len(yaml_content)} devices from {file_keep}")
            except Exception as e:
                logger.warning(
                    f"Error reading {file_keep.name}: {e}, will create {file_out.name} from scratch",
                    exc_info=True,
                )
                existing_content = {}

        # If we have existing YAML content, do selective update
        if existing_content:
            # Check if existing content has only empty values
            if io_info_files.all_vals_empty(existing_content):
                logger.info(f"Existing {file_keep.name} has only placeholder values: ignoring")
                content_write = _sort_devices_by_key(content_use)
            else:
                # Both existing YAML and new content have real data, do selective overwrite
                logger.debug(f"Merging new content with existing {file_keep.name}")
                content_write = _merge_device_metadata(
                    existing_content, content_use, normalize_keys=True
                )
        else:
            # No existing YAML file, use new content sorted by type/model and number
            content_write = _sort_devices_by_key(content_use)

    # Determine whether the merged content differs from the source (info_devices.yaml or existing @meta_finder)
    # Use structural comparison to handle nested vs flat list representation differences
    # Always normalize keys for robust comparison (normalizing already-normalized keys is idempotent)
    content_unchanged = _content_data_equals(content_write, existing_content, normalize_keys=True)

    # Write file only if content changed from the source
    if not content_unchanged:
        if existing_content:
            logger.info(
                f"Writing to {file_out.name} by selective update existed content:\n{existing_content}\n"
                f"with:\n{content_write}"
            )
        b_recorded = io_info_files.write_metadata_file(device_dir, file_out, content_write)
        return b_recorded

    # Content is unchanged — decide whether to keep or delete @meta_finder file
    if have_file_out:
        if have_file_keep:
            logger.info(
                f"Deleting {file_out.name} in {device_dir}: "
                f"content is identical to info_devices.yaml"
            )
            file_out.unlink()
        else:
            logger.debug(
                f"Keeping {file_out.name} in {device_dir}: "
                f"no info_devices.yaml to compare against"
            )
    else:
        logger.info(
            f"Not creating {file_out.name} as content is unchanged from existing source"
        )
    return False
