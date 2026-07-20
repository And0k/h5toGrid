import functools
from datetime import datetime
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Any
import json
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ
from ruamel.yaml.representer import RoundTripRepresenter
from typing import Mapping, Sequence
from .logging_config import setup_logging
from .config import DEVICES_FILE_NAME, DEVICES_FILE_NAME_UPD

logger = setup_logging()

# Common encodings to try for reading metadata files with special characters
# Includes UTF-8 variants and common legacy encodings for international character sets
COMMON_ENCODINGS = ['utf-8', 'cp1251', 'iso-8859-1', 'utf-8-sig']

# Configure YAML parser/dumper with preferred settings (global for reuse)
yaml = YAML()
yaml.default_flow_style = False
yaml.allow_unicode = True
yaml.preserve_quotes = True
yaml.sort_base_mapping_type_on_output = False
yaml.width = 4096  # practical “no wrap” #  yaml.width = None  # works in newer ruamel.yaml versions

def represent_none_as_tilde(representer, data):
    return representer.represent_scalar("tag:yaml.org,2002:null", "~")

yaml.representer.add_representer(type(None), represent_none_as_tilde)

# Field names of info_devices*.json devices metadata files
info_devices_field_names_extended = [
    "point",  # index 0
    "sea_depth",
    "height_above_bottom",
    "modification_symbol",
    "lat",
    "lon",
    "time_st",
    "time_en",
    "burst_dt",
    "bursts_t",
    "comment",
    "coef_date", # index 11 - coef date from HDF5 files
    "time_raw_st",  # index 12 - raw start time from HDF5 files
    "time_raw_en",  # index 13 - raw end time from HDF5 files
]
max_fields_number = 11
comment_field_index = info_devices_field_names_extended.index('comment')

yaml_array_header = ", ".join([
    (
        "h_above_bot" if n == "height_above_bottom" else "symbol" if n == "modification_symbol" else n
    ).capitalize()
    for n in info_devices_field_names_extended[:max_fields_number]
])
placeholders = {"?", "-", ""}

# Special keys that are not part of the metadata structure
_SPECIAL_METADATA_KEYS = ["data_paths", "setup_name", "combined_comments"]


def iter_station_id_items(device_entry: Dict[str, Any]):
    """Iterate over station ID items in a device entry, filtering out special metadata keys.

    Yields (station_id, metadata) for entries that are either:
    - list/tuple values (raw metadata lists from YAML/JSON read)
    - dict values containing field-name keys (field-name dicts from info_meta_list_to_dict)

    Special keys (data_paths, cruise, combined_comments) are always excluded.
    """
    for k, v in device_entry.items():
        if k in _SPECIAL_METADATA_KEYS:
            continue
        if isinstance(v, (list, tuple)):
            yield k, v
        elif isinstance(v, dict) and v.keys() & info_devices_field_names_extended:
            yield k, v


def _station_metadata_to_list(station_metadata) -> list:
    """Convert station metadata to a flat list for sequence-based operations.

    Args:
        station_metadata: Either a list/tuple (raw metadata list) or a
            dict with field-name keys (from info_meta_list_to_dict).

    Returns:
        Flat list of metadata values in field-name order.
    """
    if isinstance(station_metadata, dict):
        return [
            station_metadata.get(field_name, "?")
            for field_name in info_devices_field_names_extended
        ]
    return list(station_metadata)


def all_vals_empty(content: Dict[str, Any]) -> bool:
    """Check if the content contains only placeholder values.

    Args:
        content: Dictionary with device IDs as keys and metadata (list or dict) as values

    Returns:
        True if all values are placeholders or lists of these values, False otherwise
    """
    logger.debug(f"all_vals_empty called with {len(content)} devices: {list(content.keys())}")
    for device_id, values in content.items():
        logger.debug(f"  Checking device {device_id}: type={type(values).__name__}")
        # Don't consider device names as values - only check the metadata values
        if isinstance(values, dict):
            logger.debug(f"    Device {device_id} is dict with keys: {list(values.keys())}")
            # Check nested dict structure (station_id -> metadata) via iter_station_id_items
            for station_id, station_metadata in iter_station_id_items(values):
                metadata_list = _station_metadata_to_list(station_metadata)
                logger.debug(f"      Checking station_id {station_id}: type={type(station_metadata).__name__}")
                preview = metadata_list[:3]
                if len(metadata_list) > 3:
                    preview.append(f"... ({len(metadata_list) - 3} more)")
                logger.debug(f"        Values: {preview}")
                for idx, value in enumerate(metadata_list):
                    if value not in placeholders:
                        logger.debug(
                            f"        Found non-placeholder value {value} "
                            f"(type={type(value).__name__}) at index {idx}, returning False"
                        )
                        return False

            # Single interval with field names as keys
            for field_name in info_devices_field_names_extended:
                field_value = values.get(field_name, "?")
                if field_value not in placeholders:
                    logger.debug(f"      Found non-placeholder field {field_name}={field_value}, returning False")
                    return False
        elif isinstance(values, list):
            logger.debug(f"    Device {device_id} is list: {values}")
            for idx, value in enumerate(values):
                logger.debug(f"      Value {idx}: {value} (in placeholders={value in placeholders})")
                if value not in placeholders:
                    logger.debug(f"      Found non-placeholder value {value} at index {idx}, returning False")
                    return False
        else:
            logger.debug(f"    Device {device_id} is scalar: {values}")
            if values not in placeholders:
                logger.debug(f"    Found non-placeholder value {values}, returning False")
                return False
    logger.debug(f"All values are placeholders, returning True")
    return True


def atomic_write(write_func: Callable[[Path, dict], None]) -> Callable[[Path, Path, dict], bool]:
    """Decorator to handle atomic file writing with temporary file pattern.

    Wraps a write function that writes to a temporary file, then moves it atomically
    to the final location. Handles error logging and temporary file cleanup.

    Args:
        write_func: Function that takes (temp_file_path, content_write) and writes content

    Returns:
        Decorated function that takes (device_dir, info_file, content_write) and returns bool
    """
    @functools.wraps(write_func)
    def wrapper(device_dir: Path, info_file: Path, content_write: dict) -> bool:
        try:
            temp_file_path = info_file.with_suffix(info_file.suffix + '.tmp')
            write_func(temp_file_path, content_write)
            shutil.move(str(temp_file_path), str(info_file))
            logger.info(f"Created {info_file.name} in {device_dir} with {len(content_write)} devices")
            return True
        except Exception as e:
            logger.error(f"Error creating file {info_file}: {e}", exc_info=True)
            print(f"ERROR: Error creating file {info_file}: {e}", file=sys.stderr)
            print("Traceback (most recent call last):", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            temp_file_path = info_file.with_suffix(info_file.suffix + '.tmp')
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except:
                    pass
            return False
    return wrapper


def read_with_encoding_retry(file_format: str):
    """Decorator to handle file reading with multiple encoding attempts.

    Wraps a read function that takes a file object and returns parsed content.
    Attempts to read the file with different encodings to handle files with
    special characters or different encodings. Logs errors and preserves
    original exceptions for proper debugging.

    Args:
        file_format: Name of the file format (e.g., 'JSON', 'YAML') for logging

    Returns:
        Decorated function that takes (file_path: Path) and returns parsed content
    """
    def decorator(read_func: Callable[[Any], dict]) -> Callable[[Path], dict]:
        @functools.wraps(read_func)
        def wrapper(file_path: Path) -> dict:
            for encoding in COMMON_ENCODINGS:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return read_func(f)
                except UnicodeDecodeError:
                    continue
                except (FileNotFoundError, PermissionError, OSError):
                    raise
                except json.JSONDecodeError:
                    raise

            # Fallback: try with utf-8 and error replacement for corrupted files
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return read_func(f)
            except Exception as e:
                logger.error(
                    f"Failed to read {file_format} file {file_path} after trying encodings: "
                    f"{COMMON_ENCODINGS}",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


@atomic_write
def write_devices_meta_json(path: Path, content: dict) -> None:
    """Write JSON file atomically using temporary file pattern.

    Creates a temporary file, writes content, then moves atomically to final location.
    Uses compact format with one device per line for readability.

    Args:
        path: Path to the file location
        content_write: Dictionary content to write as JSON
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write("{\n")
        items = list(content.items())
        for i, (device_id, values) in enumerate(items):
            line = f'  "{device_id}": {json.dumps(values, ensure_ascii=False)}'
            if i < len(items) - 1:
                line += ","
            f.write(line + "\n")
        f.write("}\n")


def _concatenate_additional_comments(metadata_list: list) -> list:
    """Concatenate elements after comment field with ". " separator.

    Args:
        metadata_list: List of metadata values
        comment_index: Index of the comment field in the list

    Returns:
        List with additional comments concatenated to the comment field
    """
    # Pad list if shorter than comment_index to avoid index errors
    metadata_list.extend([""] * (comment_field_index + 1 - len(metadata_list)))

    # Extract non-empty/placeholder values from elements AFTER comment field
    additional_comments = [
        str(elem)
        for elem in metadata_list[comment_field_index + 1:]
        if elem not in placeholders and elem not in [None, ""]
    ]

    # Combine existing comment with additional comments
    combined_comments = [metadata_list[comment_field_index]] + additional_comments
    filtered_comments = [c for c in combined_comments if c not in placeholders and c not in [None, ""]]

    # Return list with all fields preserved up to comment field, then concatenated comment
    return metadata_list[:comment_field_index] + [". ".join(filtered_comments)]


def force_dq(obj):
    match obj:
        # scalars
        case str() as s:
            return DQ(s)  # s if s in placeholders else

        case datetime() as s:  # not used as we already converted datetime to str. todo: don't convert before
            return DQ(s.isoformat(sep=" "))

        # ruamel containers (must be before builtins)
        case CommentedSeq() as seq:
            for i, v in enumerate(seq):
                seq[i] = force_dq(v)
            return seq

        case CommentedMap() as mapping:
            for k, v in mapping.items():
                mapping[k] = force_dq(v)
            return mapping

        # builtin containers (fallback)
        case list() as seq:
            return [force_dq(v) for v in seq]

        case dict() as mapping:
            return {k: force_dq(v) for k, v in mapping.items()}

        case _:
            return obj


def _remove_trailing_empty_fields(seq: Sequence[Any]) -> Sequence[Any]:
    """Remove trailing empty/null values from a sequence.

    Args:
        seq: Sequence of values to filter

    Returns:
        Sequence with trailing empty values removed
    """
    # Find the index of time_en (field index 6)
    time_en_index = 6

    # If sequence is shorter than time_en index, return as-is
    if len(seq) <= time_en_index:
        return seq

    # Find the last non-empty value starting from the end
    last_non_empty = len(seq) - 1
    while last_non_empty > time_en_index:
        value = seq[last_non_empty]
        # Check if value is empty (None, empty string, or placeholder)
        if value is None or value == "" or value in placeholders:
            last_non_empty -= 1
        else:
            break

    # Return sequence up to last non-empty value
    return seq[:last_non_empty + 1]


def save_to_yaml_format(content: Mapping[str, Mapping[str, Sequence[Any]] | Sequence[Any]], output_path):
    root = CommentedMap()
    commented_1d = False
    commented_2d = False
    for name, map_or_seq in content.items():
        logger.debug(f"Prepare device {name} data of type={type(map_or_seq)}")
        if isinstance(map_or_seq, Sequence):
            if not commented_1d:
                commented_1d = True
                root.yaml_set_start_comment("Instrument_ID: [{}]".format(yaml_array_header))
            # Remove trailing empty fields from sequence
            filtered_seq = _remove_trailing_empty_fields(map_or_seq)
            value = CommentedSeq(filtered_seq)
            value.fa.set_flow_style()  # <-- inline [a, b, c]
        elif isinstance(map_or_seq, Mapping):
            logger.debug(f"  Device {name} is a Mapping with keys: {list(map_or_seq.keys())}")
            # Check if this is a nested dict structure (multiple intervals) or single interval (field names)
            # Nested dict: keys are station IDs (any string/int), values are metadata lists
            # Single interval: keys are field names (point, sea_depth, etc.), values are field values
            # We can distinguish by checking if there are any list/tuple values (metadata lists)
            # In nested dict, at least one value is a list/tuple
            # In single interval, values are mixed types (strings, numbers, dicts, etc.)
            has_metadata_list_values = any(iter_station_id_items(map_or_seq))
            logger.debug(f"  has_metadata_list_values={has_metadata_list_values} for device {name}")

            if has_metadata_list_values:
                # Count only station entries (exclude special keys already filtered by iter_station_id_items)
                station_entries = dict(iter_station_id_items(map_or_seq))
                # Check if there's only one interval (station_id '0') - if so, write as simple list
                if len(station_entries) == 1 and '0' in station_entries:
                    # Single interval - write as simple list for backward compatibility
                    logger.debug(f"  Device {name} has single interval, writing as simple list")
                    if not commented_1d:
                        commented_1d = True
                        root.yaml_set_start_comment("Instrument_ID: [{}]".format(yaml_array_header))
                    metadata_seq = _station_metadata_to_list(station_entries['0'])
                    filtered_seq = _remove_trailing_empty_fields(metadata_seq)
                    value = CommentedSeq(filtered_seq)
                    value.fa.set_flow_style()  # <-- inline [a, b, c]
                else:
                    # Multiple intervals - write as nested dict
                    logger.debug(
                        f"  Device {name} has multiple intervals ({len(station_entries)}), writing as nested dict"
                    )
                    if not commented_2d:
                        commented_2d = True
                        root.yaml_set_start_comment("Instrument_ID:\n  Setup_ID: [{}]".format(yaml_array_header))
                    value = CommentedMap()
                    for station_id, station_metadata in station_entries.items():
                        logger.debug(
                            f"    Processing station_id={station_id}, metadata type={type(station_metadata).__name__}"
                        )
                        metadata_seq = _station_metadata_to_list(station_metadata)
                        filtered_seq = _remove_trailing_empty_fields(metadata_seq)
                        seq_obj = CommentedSeq(filtered_seq)
                        seq_obj.fa.set_flow_style()  # <-- inline [a, b, c]
                        value[station_id] = seq_obj
            else:
                # Single interval with field names as keys (old format)
                # This shouldn't happen with the simplified structure, but handle it for backward compatibility
                if not commented_1d:
                    commented_1d = True
                    root.yaml_set_start_comment("Instrument_ID: [{}]".format(yaml_array_header))
                # Convert dict to list format
                metadata_list = [
                    map_or_seq.get(field_name, "")
                    for field_name in info_devices_field_names_extended
                ]
                filtered_seq = _remove_trailing_empty_fields(metadata_list)
                value = CommentedSeq(filtered_seq)
                value.fa.set_flow_style()  # <-- inline [a, b, c]
        else:
            # Unexpected type, skip
            logger.warning(f"Unexpected type for {name}: {type(map_or_seq)}, skipping")
            continue

        root[DQ(name)] = force_dq(value)

    # Create a fresh YAML instance for each call to avoid DocumentStartEvent errors
    fresh_yaml = YAML()
    fresh_yaml.default_flow_style = False
    fresh_yaml.allow_unicode = True
    fresh_yaml.preserve_quotes = True
    fresh_yaml.sort_base_mapping_type_on_output = False
    fresh_yaml.width = 4096
    fresh_yaml.representer.add_representer(type(None), represent_none_as_tilde)

    with open(output_path, "w", encoding="utf-8") as f:
        fresh_yaml.dump(root, f)


@atomic_write
def write_devices_meta_yaml(path: Path, content: dict) -> None:
    """Write YAML file atomically using temporary file pattern.

    Creates a temporary file, writes content in YAML format, then moves atomically
    to final location. Uses ruamel.yaml for better formatting control.
    Content should already be in nested dict structure (device_id -> station_id -> metadata_list).

    Args:
        path: Path to the temporary file location
        content_write: Dictionary content to write as YAML
    """
    logger.debug(f"write_devices_meta_yaml called with content keys: {list(content.keys())}")
    for device_id, device_entry in content.items():
        logger.debug(f"  Device {device_id}: type={type(device_entry)}, keys={list(device_entry.keys()) if isinstance(device_entry, dict) else 'N/A'}")
    # Content should already be in nested dict structure, write directly
    save_to_yaml_format(content, path)


def read_metadata_file(file_path: Path) -> dict:
    """Read metadata file, selecting format based on file extension.

    Supports JSON (.json) and YAML (.yaml, .yml) formats.
    For both formats, converts to unified nested dict structure (device_id -> station_id -> metadata_list)
    and concatenates elements after comment field with ". " separator.

    Args:
        file_path: Path to the metadata file to read

    Returns:
        The loaded content as dictionary in nested dict structure

    Raises:
        Exception: If the file cannot be read or format is not supported
    """
    suffix = file_path.suffix.lower()
    # Use CommentedMap to preserve order of devices from file
    if (data := (
        read_with_encoding_retry(file_format='JSON')(json.load)(file_path)
        if suffix == ".json"
        else read_with_encoding_retry(file_format='YAML')(yaml.load)(file_path)
        if suffix in (".yaml", ".yml")
        else None
    )):
        # Skip device entries whose ID starts with a space — convention for "no data yet"
        filtered = CommentedMap()
        skipped = []
        for device_id, metadata in data.items():
            if device_id.startswith(" "):
                skipped.append(device_id)
                continue
            filtered[device_id] = _convert_metadata_to_nested_dict(metadata)
        if skipped:
            logger.debug(
                "Skipped %s space-prefixed device(s) from %s: %s",
                len(skipped), file_path.name, skipped,
            )
        return filtered
    raise Exception(f"Unsupported file format: {suffix}. Supported formats: .json, .yaml, .yml")

def _convert_metadata_to_nested_dict(
    metadata: Dict[str | int, Dict[str | int, Sequence[str | int | float | datetime]]]
    | Dict[str | int, Sequence[str | int | float | datetime]],
):
    """Convert metadata to nested dict structure with concatenated comments.

    Args:
        metadata: Metadata in list, tuple, or dict format
        comment_field_index: Index of comment field in field names list

    Returns:
        CommentedMap with nested dict structure (station_id -> metadata_list)
    """
    device_entry = CommentedMap()
    for station_id, meta_seq in (
        # Nested dict structure with station_id keys
        metadata.items()
        if isinstance(metadata, dict)
        # Single interval device (list/tuple) - convert to nested dict with key "0"
        else [("0", metadata)]
    ):
        device_entry[str(station_id)] = _concatenate_additional_comments(list(meta_seq))
    return device_entry


def write_metadata_file(device_dir: Path, info_file: Path, content_write: dict) -> bool:
    """Write metadata file, selecting format based on file extension.

    Supports JSON (.json) and YAML (.yaml, .yml) formats.
    Uses atomic write pattern with temporary file.

    Args:
        device_dir: Directory where the file will be created
        info_file: Path to the final file location
        content_write: Dictionary content to write

    Returns:
        True if successful, False otherwise

    Raises:
        Exception: If the file format is not supported
    """
    logger.debug(f"write_metadata_file called for {info_file.name} with {len(content_write)} devices")
    for device_id, device_entry in content_write.items():
        logger.debug(f"  Device {device_id}: type={type(device_entry)}, keys={list(device_entry.keys()) if isinstance(device_entry, dict) else 'N/A'}")
    suffix = info_file.suffix.lower()

    if suffix == '.json':
        return write_devices_meta_json(device_dir, info_file, content_write)
    elif suffix in ('.yaml', '.yml'):
        return write_devices_meta_yaml(device_dir, info_file, content_write)
    else:
        raise Exception(f"Unsupported file format: {suffix}. Supported formats: .json, .yaml, .yml")
