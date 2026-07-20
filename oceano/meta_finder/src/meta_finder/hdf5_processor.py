"""
HDF5 processing functions for TCM Metadata Processor.
"""
from pathlib import Path, PurePosixPath
from typing import List, Dict, Optional, Tuple, Any
import re
import tables
import numpy as np
from datetime import datetime, timedelta
import warnings
from .data_proc_funcs import _extract_device_ids_from_column_name
from .logging_config import setup_logging
from .parse_data_file_name import extract_device_ids_from_prefixed_name
from . import config
logger = setup_logging()

warnings.filterwarnings("ignore", category=tables.exceptions.DataTypeWarning, module="tables")
warnings.filterwarnings("ignore", message=r"dtype\(\).*align=0", module="tables|numpy")

def _convert_timestamps(time_values, table_attrs, start_idx=0, end_idx=None):
    """
    Convert timestamps from HDF5/PyTables data to datetime strings.

    Args:
        time_values: Array of timestamp values from PyTables
        table_attrs: Table attributes containing index_kind metadata
        start_idx: Start index for time range (default: 0)
        end_idx: End index for time range (default: last element)

    Returns:
        tuple: (start_time_str, end_time_str) in 'YYYY-MM-DD HH:MM:SS' format
    """

    # Handle empty arrays and index bounds
    if not len(time_values):
        return ("", "")

    # Normalize end_idx
    if end_idx is None:
        end_idx = len(time_values) - 1
    elif end_idx < 0:
        end_idx = len(time_values) + end_idx

    # Get start and end values
    start_val = time_values[start_idx]
    end_val = time_values[end_idx] if len(time_values) > 1 else start_val

    # Convert based on type
    start_time = _convert_single_timestamp(start_val, table_attrs)
    end_time = _convert_single_timestamp(end_val, table_attrs)

    return start_time, end_time


def _convert_single_timestamp(value, table_attrs):
    """
    Convert a single timestamp value to datetime string.
    Uses numpy datetime64 consistently and falls back to nanoseconds always.

    Args:
        value: Single timestamp value
        table_attrs: Table attributes for type detection

    Returns:
        str: Formatted datetime string 'YYYY-MM-DD HH:MM:SS'
    """

    # Check for numpy datetime64 objects first
    if hasattr(value, "dtype") and np.issubdtype(value.dtype, np.datetime64):
        logger.debug(f"converting {value.dtype} of dtype {value}, with table_attrs: {table_attrs}")
        return _convert_numpy_datetime(value)

    # Check for datetime64 in table attributes (including both datetime64 and datetime64[ns])
    index_kind = getattr(table_attrs, "index_kind", None)
    if isinstance(index_kind, bytes):
        index_kind = index_kind.decode()

    if index_kind and str(index_kind) in ("datetime64[ns]", "datetime64"):
        logger.debug(f"converting {value.dtype} dtype {value}, with index_kind: {index_kind}")
        return _convert_nanosecond_timestamp(value)

    # Handle numeric timestamps - check for different units based on magnitude
    if isinstance(value, (int, float, np.int64)) and value > 0:
        logger.debug(f"converting {type(value)} type {value}")
        return _convert_numeric_timestamp_fallback(value)

    # Fallback to string representation
    return str(value).replace("T", " ")


def _convert_numeric_timestamp_fallback(value):
    """
    Convert numeric timestamp with fallback to microseconds.
    The raw HDF5 values in the TSV are microsecond timestamps, so convert accordingly.

    Args:
        value: Numeric timestamp value

    Returns:
        str: Formatted datetime string 'YYYY-MM-DD HH:MM:SS'
    """
    try:
        # Handle numpy scalars properly
        if hasattr(value, 'item'):
            value = value.item()  # Convert numpy scalar to Python scalar

        if value >= 1e17:
            # Values with 18+ digits are likely nanosecond timestamps (e.g., 1694023126000000)
            timestamp = value / 1e9  # Convert from nanoseconds to seconds
        elif value >= 1e14:  # Values with 15-16 digits are likely microsecond timestamps
            timestamp = value / 1e6  # Convert from microseconds to seconds
        elif value >= 1e11:  # Values with 11-13 digits might be millisecond timestamps
            timestamp = value / 1e3  # Convert from milliseconds to seconds
        else:  # Smaller values are likely seconds
            timestamp = value

        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError) as e:
        logger.warning(f"Error converting numeric timestamp {value}: {e}")
        return str(value)


def _convert_nanosecond_timestamp(value):
    """Convert nanosecond timestamp to datetime string."""
    try:
        timestamp = float(value) / 1e9
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError) as e:
        logger.warning(f"Error converting nanosecond timestamp {value}: {e}")
        return str(value)


def _convert_numpy_datetime(value, units="s"):
    """Convert numpy datetime64 to datetime string."""
    try:
        # Convert to seconds precision and format
        dt_str = str(np.datetime64(value, units))
        return dt_str.replace("T", " ")
    except Exception as e:
        logger.warning(f"Error converting numpy datetime {value}: {e}")
        return str(value)


def _convert_matlab_serial_date(matlab_datenum):
    day = datetime.fromordinal(int(matlab_datenum))
    dayfrac = timedelta(days=matlab_datenum % 1) - timedelta(days=366)
    return day + dayfrac


def _validate_and_format_date(dt: datetime, h5_file_path: Path, dev_group_name: str, source_description: str) -> Optional[str]:
    """
    Validate date is in reasonable range and format it as string.

    Args:
        dt: Datetime object to validate
        h5_file_path: Path to the HDF5 file (for logging)
        dev_group_name: Device group name (for logging)
        source_description: Description of where date came from (for logging)

    Returns:
        Formatted date string in 'YYYY-MM-DD HH:MM:SS' format or None if invalid
    """
    current_year = datetime.now().year
    if dt.year < 2010 or dt.year > current_year:
        logger.warning(
            f"Extracted coef date from {source_description} "
            f"{dt.strftime('%Y-%m-%d %H:%M:%S')} in {h5_file_path}/{dev_group_name} "
            f"is outside the reasonable range (2010 to {current_year}). Date year: {dt.year}"
        )
        return None
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def extract_devices_from_hdf5_groups(h5_file_path: Path) -> Dict[str, List[PurePosixPath]]:
    """
    Extract device IDs from HDF5 file group names.
    Only searches in root groups, not in subgroups.

    Args:
        h5_file_path: Path to the HDF5 file

    Returns:
        Dictionary mapping normalized device IDs to lists of group names that contain the device
    """

    device_groups = {}
    try:
        with tables.open_file(str(h5_file_path), mode="r") as h5file:
            # Only get group names in the root, not in subgroups
            for node in h5file.walk_nodes(where="/", classname="Group"):
                # Check if the node is directly under root (e.g., /group_name, not /some/sub/group_name)
                if node._v_pathname == "/":
                    continue  # Skip root
                # Check if this is a direct child of root by counting path separators
                path_parts = node._v_pathname.strip('/').split('/')
                if len(path_parts) == 1:  # Only one part means it's directly under root
                    group_path = PurePosixPath(node._v_pathname)
                    # Extract device name from group name using the same logic as for text files
                    for device_id in extract_device_ids_from_prefixed_name(group_path.name, msg_what="group name "):
                        if device_id not in device_groups:
                            device_groups[device_id] = []
                        device_groups[device_id].append(group_path)
    except Exception:
        logger.exception(f"Error reading HDF5 file {h5_file_path}")

    return device_groups



def _extract_time_range_from_table_path(h5_file_path: Path, table_path: str) -> Optional[Tuple[str, str]]:
    """
    Internal helper function to extract time range from a specific table in an HDF5 file.
    This extracts just the start and end times without additional metadata.

    Args:
        h5_file_path: Path to the HDF5 file
        table_path: Path to the table within the HDF5 file

    Returns:
        A tuple containing the start and end time strings, or None if not found.
    """

    try:
        with tables.open_file(str(h5_file_path), mode="r") as h5file:
            # Check if the table path exists before attempting to get the node
            if table_path not in h5file:
                logger.error(f"Table path '{table_path}' does not exist in HDF5 file {h5_file_path}")
                return None

            # Get the table
            table = h5file.get_node(table_path)

            if hasattr(table, 'cols') and hasattr(table.cols, '_v_colnames'):
                # For pandas HDF5 stores, the datetime index is often in a column named 'index'
                # or the first column if it's a datetime
                time_col = None

                # Look for common time column names
                for col_name in table.cols._v_colnames:
                    if 'time' in col_name.lower() or 'date' in col_name.lower() or col_name == 'index':
                        time_col = col_name
                        break

                # If no obvious time column found, check if first column is datetime
                if time_col is None and table.cols._v_colnames:
                    first_col_name = table.cols._v_colnames[0]
                    first_col = getattr(table.cols, first_col_name)
                    # Check if it's a datetime type
                    if hasattr(first_col, 'dtype') and 'time' in str(first_col.dtype).lower():
                        time_col = first_col_name

                if time_col:
                    # Read only first and last time values to avoid memory issues
                    n_rows = table.nrows
                    if n_rows > 0:
                        try:
                            # Read first time value
                            first_time_values = getattr(table.cols, time_col)[0:1]
                            # Read last time value
                            last_time_values = getattr(table.cols, time_col)[n_rows-1:n_rows]

                            # Get the actual values from the arrays
                            first_time_val = first_time_values[0]
                            last_time_val = last_time_values[0]

                            # Create a combined array with just first and last values for _convert_timestamps
                            time_values = [first_time_val, last_time_val] if n_rows > 1 else [first_time_val]

                            if len(time_values) > 0:
                                # Convert timestamps using the helper function
                                start_time, end_time = _convert_timestamps(
                                    time_values, table._v_attrs, 0, -1 if n_rows > 1 else 0
                                )

                                return start_time, end_time
                            else:
                                logger.warning(f"No time values in HDF5 file {h5_file_path}, table: {table_path}")
                                return None
                        except MemoryError as me:
                            logger.error(f"MemoryError when allocating array for HDF5 file {h5_file_path}, table: {table_path}: {me}")
                            logger.error(f"Unable to allocate memory for an array with shape ({n_rows},) and data type {getattr(table.cols, time_col).dtype}")
                            return None
                        except Exception as e:
                            logger.error(f"Error reading time column from HDF5 file {h5_file_path}, table: {table_path}: {e}")
                            return None
                    else:
                        logger.warning(f"No rows in HDF5 file {h5_file_path}, table: {table_path}")
                        return None
                else:
                    logger.warning(
                        f"Not found time col in HDF5 file {h5_file_path}, table: {table_path}"
                    )
            else:
                logger.warning(
                    f"Not found any cols in HDF5 file {h5_file_path}, table: {table_path}"
                )
    except MemoryError as me:
        logger.error(f"MemoryError when accessing HDF5 file {h5_file_path}, table: {table_path}: {me}")
        return None
    except Exception as e:
        logger.error(f"Error extracting time range from HDF5 file {h5_file_path}, table: {table_path}: {e}")

    return None



def extract_time_range_from_hdf5_table(
    h5_file_path: Path, table_path: str
) -> Optional[Tuple[str, str, str, str]]:
    """
    Extract time range from a specific table in an HDF5 file.

    Args:
        h5_file_path: Path to the HDF5 file
        table_path: Path to the table within the HDF5 file

    Returns:
        A tuple containing the start and end time strings, and burst info (bursts_t, burst_dt),
        or None if not found.
    """

    time_range_result = _extract_time_range_from_table_path(h5_file_path, table_path)
    if time_range_result:
        start_time, end_time = time_range_result
        # Default burst info for HDF5 files
        bursts_t, burst_dt = "-", "-"
        return start_time, end_time, burst_dt, bursts_t

    return None


def add_device_result(
    device_results: Dict[str, Dict[str, Dict[Tuple[Path, PurePosixPath], Dict[str, Any]]]],
    device_id: str,
    h5_file_path: Path,
    table_path: PurePosixPath | str,
    time_range: Optional[Tuple[str, str, str, str]] = None,
    averaging_interval: Optional[int] = None,
) -> bool:
    """
    Helper function to add a device result to the `device_id` metadata to `device_results` dictionary.

    Args:
        device_results: Dictionary to store device results
        device_id: The device ID to add
        table_path: hdf5 group with the data for the device
        h5_file_path: Path to the HDF5 file
        time_range: Optional time range tuple (start_time, end_time, burst_dt, bursts_t)
        averaging_interval: Optional averaging interval in seconds (extracted from group names like i04bin2s)
    """

    if (b_new := device_id not in device_results):
        device_results[device_id] = {"time_info": time_range, "data_paths": {}}
    # Create metadata for the HDF5 file with averaging interval if available
    metadata: Dict[str, Any] = {"devices": [device_id]}
    if averaging_interval is not None:
        metadata["averaging_interval"] = averaging_interval
    device_results[device_id]["data_paths"][(h5_file_path, PurePosixPath(table_path))] = metadata
    return b_new


def extract_device_ids_from_hdf5_table_columns(h5_file_path: Path, table_path: str) -> List[str]:
    """
    Extract device IDs from column names in a HDF5 table.
    This function is used to find device IDs in combined HDF5 files where device information
    is stored in column names rather than group names.

    Args:
        h5_file_path: Path to the HDF5 file
        table_path: Path to the table within the HDF5 file

    Returns:
        A list of normalized device IDs extracted from column names.
    """
    device_ids = []

    try:
        with tables.open_file(str(h5_file_path), mode="r") as h5file:
            # Check if the table path exists before attempting to get the node
            if table_path not in h5file:
                logger.error(f"Table path '{table_path}' does not exist in HDF5 file {h5_file_path}")
                return []

            table = h5file.get_node(table_path)
            if hasattr(table, 'cols') and hasattr(table.cols, '_v_colnames'):
                # For combined HDF5 tables, columns typically represent different devices
                col_names = table.cols._v_colnames

                # Process each column name to extract device IDs
                for col_name in col_names:
                    # Look for time column and skip it
                    if 'time' in col_name.lower() or 'date' in col_name.lower() or col_name == 'index':
                        continue

                    # Extract device ID from column name
                    for device_id in _extract_device_ids_from_column_name(col_name):
                        if device_id not in device_ids:  # Avoid duplicates
                            device_ids.append(device_id)

    except Exception as e:
        logger.error(f"Error extracting device IDs from HDF5 table columns {h5_file_path}:{table_path}: {e}")

    return device_ids


def extract_time_ranges_from_hdf5_combined(h5_file_path: Path, table_path: str, dev_ids: List[str]) -> Dict[str, Optional[Tuple[str, str, str, str]]]:
    """
    Extract device-specific time ranges from a combined HDF5 table that contains data for multiple devices.

    Args:
        h5_file_path: Path to the HDF5 file
        table_path: Path to the table within the HDF5 file
        dev_ids: List of pre-normalized device IDs to extract time ranges for

    Returns:
        A dictionary mapping device IDs to tuples of (start_time, end_time, bursts_t, burst_dt), or None if not found.
    """

    result = {dev_id: None for dev_id in dev_ids}

    try:
        with tables.open_file(str(h5_file_path), mode="r") as h5file:
            # Check if the table path exists before attempting to get the node
            if table_path not in h5file:
                logger.error(f"Table path '{table_path}' does not exist in HDF5 file {h5_file_path}")
                return {dev_id: None for dev_id in dev_ids}

            table = h5file.get_node(table_path)

            if hasattr(table, 'cols') and hasattr(table.cols, '_v_colnames'):
                # For combined HDF5 tables, columns typically represent different devices
                col_names = table.cols._v_colnames
                time_col = None

                # Look for time column (usually 'index' in pandas HDF5 stores)
                for col_name in col_names:
                    if 'time' in col_name.lower() or 'date' in col_name.lower() or col_name == 'index':
                        time_col = col_name
                        break

                if time_col:
                    # Read only first and last time values to avoid memory issues
                    n_rows = table.nrows
                    time_values = []  # Initialize time_values as empty

                    if n_rows > 0:
                        try:
                            # Read first time value
                            first_time_values = getattr(table.cols, time_col)[0:1]
                            # Read last time value
                            last_time_values = getattr(table.cols, time_col)[n_rows-1:n_rows]

                            # Get the actual values from the arrays
                            if len(first_time_values) > 0:
                                first_time_val = first_time_values[0]
                            else:
                                logger.warning(f"No first time value found in HDF5 file {h5_file_path}, table: {table_path}")
                                return {dev_id: None for dev_id in dev_ids}

                            if len(last_time_values) > 0:
                                last_time_val = last_time_values[0]
                            else:
                                logger.warning(f"No last time value found in HDF5 file {h5_file_path}, table: {table_path}")
                                return {dev_id: None for dev_id in dev_ids}

                            # Create a combined array with just first and last values for _convert_timestamps
                            time_values = [first_time_val, last_time_val] if n_rows > 1 else [first_time_val]
                        except MemoryError as me:
                            logger.error(f"MemoryError when allocating time array for HDF5 file {h5_file_path}, table: {table_path}: {me}")
                            logger.error(f"Unable to allocate memory for time column with shape ({n_rows},) and data type {getattr(table.cols, time_col).dtype}")
                            return {dev_id: None for dev_id in dev_ids}
                        except Exception as e:
                            logger.error(f"Error reading time column from HDF5 file {h5_file_path}, table: {table_path}: {e}")
                            return {dev_id: None for dev_id in dev_ids}

                    # Process each device column
                    for col_name in col_names:
                        if col_name == time_col:
                            continue # Skip time column

                        # Extract device ID from column name
                        col_device_ids = _extract_device_ids_from_column_name(col_name)

                        for device_id in col_device_ids:
                            if device_id in dev_ids:
                                try:
                                    # Read only first and last device data values to avoid memory issues
                                    n_rows = table.nrows
                                    device_data = []  # Initialize device_data as empty

                                    if n_rows > 0:
                                        # Get the data for this device column (read first and last values only)
                                        first_device_values = getattr(table.cols, col_name)[0:1]
                                        last_device_values = getattr(table.cols, col_name)[n_rows-1:n_rows]

                                        # Get the actual values from the arrays
                                        first_device_val = first_device_values[0]
                                        last_device_val = last_device_values[0]

                                        # Create a combined array with just first and last values
                                        device_data = [first_device_val, last_device_val] if n_rows > 1 else [first_device_val]

                                    # Check if there's valid data (not all NaN or empty)
                                    has_data = False
                                    if hasattr(device_data, '__len__') and len(device_data) > 0:
                                        # For our limited data, check first and last values
                                        for val in device_data:
                                            if hasattr(val, 'dtype') and np.issubdtype(val.dtype, np.number):
                                                # For numeric data, check if not NaN
                                                if not np.isnan(val):
                                                    has_data = True
                                                    break
                                            elif val and str(val).lower() not in ['nan', 'null', '']:
                                                # For other data types, check if not empty/NaN
                                                has_data = True
                                                break

                                    if has_data and len(time_values) > 0:
                                        # For combined files, if we have valid data, we use the full time range
                                        # since we only read first and last values
                                        start_time, end_time = _convert_timestamps(time_values, table._v_attrs, 0, -1)

                                        # Default burst info for HDF5 files
                                        bursts_t, burst_dt = "-", "-"

                                        result[device_id] = (start_time, end_time, burst_dt, bursts_t)
                                except MemoryError as me:
                                    logger.error(f"MemoryError when allocating device data array for HDF5 file {h5_file_path}, table: {table_path}, column: {col_name}: {me}")
                                    logger.error(f"Unable to allocate memory for device column with shape ({n_rows},) and data type {getattr(table.cols, col_name).dtype}")
                                    # Continue with other columns
                                    continue
                                except Exception as e:
                                    logger.error(f"Error reading device column {col_name} from HDF5 file {h5_file_path}, table: {table_path}: {e}")
                                    # Continue with other columns
                                    continue

    except MemoryError as me:
        logger.error(f"MemoryError when accessing HDF5 file {h5_file_path}, table: {table_path}: {me}")
        return {dev_id: None for dev_id in dev_ids}
    except Exception as e:
        logger.error(f"Error extracting time ranges from combined HDF5 table {h5_file_path}:{table_path}: {e}")

    return result


def _is_excluded_by_dir_patterns(name: str) -> bool:
    """Check if a file or directory name matches any ptn_dir_exclude pattern."""
    return any(re.search(pattern, name) for pattern in config.ptn_dir_exclude)


def _is_valid_hdf5_file(
    file_path: Path, bad_stem_ends: Tuple[str] = ("not_sorted", "-", "~", "psd"), bad_string: str = "копия"
) -> bool:
    """Check if file is valid HDF5 (not a copy, unfinished, or excluded by dir patterns)."""
    return (
        file_path.is_file()
        and file_path.suffix.lower() in config.extensions_hdf5
        and not file_path.stem.lower().endswith(bad_stem_ends)
        and bad_string not in file_path.name
        and not _is_excluded_by_dir_patterns(file_path.name)
    )


def _categorize_hdf5_file(file_path: Path) -> Optional[str]:
    """
    Categorize HDF5 file by its name suffix.

    Categories based on filename:
    - 'proc_noAvg': files ending with .proc_noAvg.h5
    - 'proc_Avg': files ending with .proc_Avg.h5 (devices in group names, no device suffixes in columns)
    - 'proc': files ending with .proc.h5 (devices in column names like Vabs_i03)
    - 'raw': all other .h5 files (excluding proc variants)

    Returns None if file doesn't end with .h5 extension.
    """
    name = file_path.name.lower()
    if not name.endswith('.h5'):
        return None
    if name.endswith('.proc_noavg.h5'):
        return 'proc_noAvg'
    if name.endswith('.proc_avg.h5'):
        return 'proc_Avg'
    if name.endswith('.proc.h5'):
        return 'proc'
    return 'raw'


def _iter_hdf5_search_directories(device_dir: Path):
    """
    Iterate over directories to search for HDF5 files.

    Yields tuples of (directory_path, is_raw_dir) where:
    - directory_path: Path to search for HDF5 files
    - is_raw_dir: True if all .h5 files should be treated as raw

    Searches:
    - device_dir itself (is_raw_dir=False)
    - _raw subdirectory (is_raw_dir=True)
    - Any subdirectory containing whole word 'h5' in its name (is_raw_dir=False)
    - Subdirectories with 'h5' in name inside _raw (is_raw_dir=True)

    Args:
        device_dir: Base device directory

    Yields:
        Tuples of (Path, bool) for each search directory
    """
    yield (device_dir, False)

    if not device_dir.is_dir():
        return

    for entry in device_dir.iterdir():
        if not entry.is_dir() or _is_excluded_by_dir_patterns(entry.name):
            continue
        if entry.name == "_raw":
            yield (entry, True)
            # Also search h5 subdirectories inside _raw (they're also raw)
            for subentry in entry.iterdir():
                if (
                    subentry.is_dir()
                    and not _is_excluded_by_dir_patterns(subentry.name)
                    and re.search(r'\bh5\b', subentry.name, re.IGNORECASE)
                ):
                    yield (subentry, True)
        elif re.search(r'\bh5\b', entry.name, re.IGNORECASE):
            yield (entry, False)


def find_hdf5_files(device_dir: Path) -> Dict[str, List[Path]]:
    """
    Find HDF5 files in device directory according to the specified priority order:
    1. *.proc_noAvg.h5 - processed files without averaging
    2. *.proc_Avg.h5 - processed files with devices in group names (e.g., i04bin2s)
    3. *.proc.h5 - processed files with devices in column names (e.g., Vabs_i03)
    4. *.h5 - raw HDF5 files (excluding proc variants)

    Searches in device_dir itself, _raw subdirectory, and any subdirectory containing
    the whole word 'h5' in its name (e.g., 'h5', 'h5_files', 'processed_h5').

    Files in _raw directory and its h5 subdirectories are automatically categorized as 'raw'.
    Files in device_dir and other h5 subdirectories are categorized by filename suffix.

    Returns dict with keys "proc_noAvg", "proc_Avg", "proc", "raw" for processing in priority order.
    """
    result: Dict[str, List[Path]] = {"proc_noAvg": [], "proc_Avg": [], "proc": [], "raw": []}

    for search_dir, is_raw_dir in _iter_hdf5_search_directories(device_dir):
        if not search_dir.is_dir():
            continue

        for file_path in search_dir.iterdir():
            if not _is_valid_hdf5_file(file_path):
                continue

            if is_raw_dir:
                # All valid .h5 files in _raw and its h5 subdirs are raw files
                result['raw'].append(file_path)
            # Categorize by filename suffix
            elif (category := _categorize_hdf5_file(file_path)):
                    result[category].append(file_path)

    return result


def extract_averaging_seconds_from_h5group(h5group: str) -> Optional[int]:
    """
    Extract averaging seconds from HDF5 group name using the pattern "bin{averaging_seconds}".

    Args:
        h5group: HDF5 group name to extract averaging from

    Returns:
        Averaging seconds as integer or None if not found
    """
    match = re.search(r'bin(\d+)', h5group, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _process_proc_h5_file(
    h5_file_path: Path,
    dev_ids_list: List[str],
    extract_time_info: bool,
    device_results: Dict[str, Any],
) -> None:
    """
    Process .proc.h5 files where devices are stored in column names rather than group names.

    Args:
        h5_file_path: Path to the HDF5 file
        dev_ids_list: List of device IDs to filter (empty means extract all)
        extract_time_info: Whether to extract time information
        device_results: Dictionary to store device results
    """
    try:
        with tables.open_file(h5_file_path, mode="r") as h5file:
            for node in h5file.walk_nodes(where="/", classname="Table"):
                table_path = node._v_pathname
                group_path = PurePosixPath(table_path).parent
                if group_path.name.startswith("log"):
                    continue  # log tables has different format which processing is not implemented
                table_device_ids = extract_device_ids_from_hdf5_table_columns(h5_file_path, table_path)

                if extract_time_info:
                    # When dev_ids_list is empty (None was passed = "extract all"),
                    # use discovered device IDs so extract_time_ranges_from_hdf5_combined doesn't return {}
                    ids_for_extraction = dev_ids_list or table_device_ids
                    time_ranges_result = extract_time_ranges_from_hdf5_combined(
                        h5_file_path, table_path, ids_for_extraction
                    )
                    for device_id, time_range in time_ranges_result.items():
                        if device_id != "_combined_comments" and time_range:
                            if not dev_ids_list or device_id in dev_ids_list:
                                add_device_result(
                                    device_results, device_id, h5_file_path, group_path, time_range
                                )
                else:
                    for device_id in table_device_ids:
                        if not dev_ids_list or device_id in dev_ids_list:
                            add_device_result(device_results, device_id, h5_file_path, group_path, None)
    except Exception as e:
        logger.error(f"Error reading .proc.h5 file {h5_file_path}: {e}")


def _process_standard_h5_file(
    h5_file_path: Path,
    dev_ids_list: List[str],
    extract_time_info: bool,
    device_results: Dict[str, Any],
) -> None:
    """
    Process standard HDF5 files where devices are stored as root-level groups.

    Extracts averaging interval from group names (e.g., i04bin2s -> 2 seconds) for .proc_Avg.h5 files.
    This averaging metadata is used for priority-based sorting of data sources.

    Args:
        h5_file_path: Path to the HDF5 file
        dev_ids_list: List of device IDs to filter (empty means extract all)
        extract_time_info: Whether to extract time information
        device_results: Dictionary to store device results
    """
    try:
        h5_devices = extract_devices_from_hdf5_groups(h5_file_path)

        for device_id, group_paths in h5_devices.items():
            if dev_ids_list and device_id not in dev_ids_list:
                continue

            for group_path in group_paths:
                time_range = (
                    extract_time_range_from_hdf5_table(h5_file_path, group_path)
                    if extract_time_info
                    else None
                )
                if not extract_time_info or time_range:
                    # Extract averaging interval from group name for .proc_Avg.h5 files
                    averaging_interval = None
                    if h5_file_path.name.endswith('.proc_Avg.h5'):
                        averaging_interval = extract_averaging_seconds_from_h5group(group_path.name)

                    add_device_result(
                        device_results, device_id, h5_file_path, group_path, time_range, averaging_interval
                    )

            if dev_ids_list and all(did in device_results for did in dev_ids_list):
                break
    except Exception as e:
        logger.warning(f"Error reading HDF5 file {h5_file_path}: {e}")


def extract_metadata_from_hdf5(device_dir: Path, dev_ids: List[str] = None, extract_time_info: bool = True) -> Dict[str, Any]:
    """
    Extract devices paths and optionally time ranges from HDF5 files. Searches all sources in device
    directory, and collects metadata according to priority order.

    Args:
        device_dir: Directory to search for HDF5 files
        dev_ids: List of pre-normalized device IDs to extract time ranges for (None means extract all)
        extract_time_info: Whether to extract time information (if False, only extracts device groups and paths)

    Returns:
        Dictionary mapping device IDs to their time info and associated file paths
        Format: {device_id: {"time_info": (start_time, end_time, bursts_t, burst_dt), "data_paths": {(dir_path, rel_path): metadata}}}
    """
    if not config.extract_hdf5_times:
        return {}

    logger.debug(
        "Trying HDF5 files for %s",
        'time range extraction' if extract_time_info else 'device and path discovery'
    )
    h5_files = find_hdf5_files(device_dir)
    logger.debug(f"Found HDF5 files: {h5_files}")

    device_results = {}
    dev_ids_list = dev_ids if dev_ids is not None else []

    # Process HDF5 files in priority order: proc_noAvg, proc_Avg, proc, raw
    for h5_type, h5_file_paths in h5_files.items():
        for h5_file_path in h5_file_paths:
            logger.debug(f"Trying HDF5 file: {h5_file_path} for type: {h5_type}")
            # .proc.h5: devices in column names; .proc_Avg.h5 and others: devices in group names
            if h5_file_path.name.endswith('.proc.h5'):
                _process_proc_h5_file(h5_file_path, dev_ids_list, extract_time_info, device_results)
            else:
                _process_standard_h5_file(h5_file_path, dev_ids_list, extract_time_info, device_results)

            # If required devices then check that all have been found
            if dev_ids_list and all(device_id in device_results for device_id in dev_ids_list):
                return device_results
    return device_results

def extract_coef_date_from_hdf5(h5_file_path: Path, dev_group_name: str) -> Optional[str]:
    """
    Extract the coef date from HDF5 file in hdf5 group that contains date: in the path {dev_group}/coef/date.
    Falls back to extracting from the "timestamp" attribute of /coef/Vabs0 if standard paths are not available.

    Args:
        h5_file_path: Path to the HDF5 file
        dev_group_name: hdf5 group name to extract date

    Returns:
        Date string in 'YYYY-MM-DD HH:MM:SS' format or None if not found
    """
    try:
        with tables.open_file(str(h5_file_path), mode="r") as h5file:
            # Construct the path to the coef date
            coef_date_path = f"/{str(dev_group_name).strip('/')}/coef/date"

            if coef_date_path in h5file:
                try:
                    date_node = h5file.get_node(coef_date_path)
                    # Check if the node is a dataset (Array, CArray, EArray, etc.) that can be indexed
                    if hasattr(date_node, 'shape'):
                        # It's a dataset, read all values
                        if date_node.shape == ():
                            # Scalar dataset - read as scalar
                            dates = [date_node[()]]
                        else:
                            # Array dataset - read all values
                            dates = date_node[:]
                    else:
                        # If it doesn't have a shape attribute, it might be a different type of node
                        # Try to read as a scalar or attribute
                        dates = [date_node[()]] if hasattr(date_node, '__getitem__') else []

                    # Process dates to find max non-NaN date
                    processed_dates = []
                    for date_val in dates:
                        if hasattr(date_val, 'decode'):
                            # Handle byte strings
                            date_str = date_val.decode('utf-8')
                            try:
                                # Try to parse the string date to datetime object
                                dt = datetime.fromisoformat(date_str)
                                processed_dates.append(dt)
                            except ValueError:
                                # If parsing fails, skip this date
                                continue
                        elif np.isscalar(date_val):
                            # Handle scalar values
                            if isinstance(date_val, (str, np.str_)):
                                # Handle string dates
                                try:
                                    # Try to parse the string date to datetime object
                                    dt = datetime.fromisoformat(date_val)
                                    processed_dates.append(dt)
                                except ValueError:
                                    # If parsing fails, skip this date
                                    continue
                            elif np.isnan(date_val):
                                # Handle NaN values
                                continue
                            else:
                                # Handle timestamp values (possibly in ns or microseconds)
                                try:
                                    # Try to convert from nano/microseconds to seconds if it looks like a timestamp
                                    if date_val > 1e14:  # Likely microsecond (values like 1.57523904e+15)
                                        date_timestamp = date_val / 1e6
                                        dt = datetime.fromtimestamp(date_timestamp)
                                        processed_dates.append(dt)
                                    elif date_val > 1e10:  # Likely second timestamp
                                        date_timestamp = date_val
                                        dt = datetime.fromtimestamp(date_timestamp)
                                        processed_dates.append(dt)
                                    else:
                                        # Regular number, maybe it's seconds
                                        dt = datetime.fromtimestamp(date_val)
                                        processed_dates.append(dt)
                                except (ValueError, OSError, OverflowError):
                                    # If conversion fails, skip this date
                                    continue
                        elif hasattr(date_val, 'astype') and 'datetime64' in str(date_val.dtype):
                            # Handle numpy datetime
                            dt = date_val[~np.isnan(date_val)].astype("datetime64[s]").item()
                            processed_dates.append(dt)
                        else:
                            # Other types - try to convert to string and then to datetime
                            date_str = str(date_val)
                            try:
                                dt = datetime.fromisoformat(date_str)
                                processed_dates.append(dt)
                            except ValueError:
                                # If parsing fails, skip this date
                                continue

                    # Filter out None values and get max date
                    valid_dates = [d for d in processed_dates if d is not None]
                    if valid_dates:
                        max_date = max(valid_dates)
                        return _validate_and_format_date(max_date, h5_file_path, dev_group_name, "coef/date dataset")
                    else:
                        return None
                except Exception as e:
                    logger.warning(f"Error reading coef/date from {coef_date_path} in {h5_file_path}: {e}")
                    return None
            else:
                logger.debug(f"Path {coef_date_path} not found in {h5_file_path}")
                # Try old format: Matlab serial date:
                coef_date_path = f"/{str(dev_group_name).strip('/')}/coef/TimeProcessed"
                if coef_date_path in h5file:
                    try:
                        date_node = h5file.get_node(coef_date_path)
                        max_date = max(_convert_matlab_serial_date(d.item()) for d in date_node[:])
                        return _validate_and_format_date(max_date, h5_file_path, dev_group_name, "coef/TimeProcessed dataset")
                    except Exception as e:
                        logger.warning(
                            f"Error reading coef. date from {coef_date_path} in {h5_file_path}: {e}"
                        )
                        return None

                # Fallback: Try to extract from "timestamp" attribute of /coef/Vabs0
                vabs0_path = f"/{str(dev_group_name).strip('/')}/coef/Vabs0"
                if vabs0_path in h5file:
                    try:
                        vabs0_node = h5file.get_node(vabs0_path)
                        timestamp_attr = getattr(vabs0_node._v_attrs, "timestamp", None)
                        if timestamp_attr is not None:
                            # Handle byte strings
                            if isinstance(timestamp_attr, bytes):
                                timestamp_str = timestamp_attr.decode('utf-8')
                            else:
                                timestamp_str = str(timestamp_attr)

                            # Parse the timestamp string
                            try:
                                dt = datetime.fromisoformat(timestamp_str)
                                return _validate_and_format_date(dt, h5_file_path, dev_group_name, "Vabs0 timestamp attribute")
                            except ValueError as e:
                                logger.warning(
                                    f"Failed to parse timestamp attribute '{timestamp_str}' from "
                                    f"{vabs0_path} in {h5_file_path}: {e}"
                                )
                                return None
                    except Exception as e:
                        logger.warning(
                            f"Error reading timestamp attribute from {vabs0_path} in {h5_file_path}: {e}"
                        )
                        return None

                logger.debug(f"No coef date found in {h5_file_path}/{dev_group_name}")
                return None
    except Exception as e:
        logger.error(f"Error opening HDF5 file {h5_file_path}: {e}")
        return None


def extract_coef_dates_from_raw_hdf5_files(h5_files: List[Path], dev_ids: List[str]) -> Dict[str, str]:
    """
    Extract coef dates from all HDF5 files in device directory for specific devices.
    This function looks specifically in _raw/*.h5 files when raw_hdf5_cols is set.

    Args:
        h5_files: list of raw files to search coefficients
        dev_ids: List of pre-normalized device IDs to extract dates for

    Returns:
        Dictionary mapping device IDs to their coef date strings
    """
    results = {}
    dev_ids_set = set(dev_ids)  # Convert to set for faster lookups

    for h5_file_path in h5_files:
        logger.debug(f"Checking HDF5 file for coef dates: {h5_file_path}")
        # Extract all device groups from the HDF5 file and normalize them
        device_groups = extract_devices_from_hdf5_groups(h5_file_path)

        # Process all device groups in this file at once, rather than iterating through dev_ids
        for normalized_group_id, group_names in device_groups.items():
            # Only process if this normalized_group_id is one we're looking for
            if normalized_group_id in dev_ids_set and normalized_group_id not in results:
                # Found a match, try to extract coef date from any of the group names
                for group_name in group_names:
                    coef_date = extract_coef_date_from_hdf5(h5_file_path, group_name)
                    if coef_date:
                        if normalized_group_id in results:
                            logger.warning(
                                f"Found next group {group_name} corresponding to the same device "
                                f"{normalized_group_id}. The coef date {results[normalized_group_id]} will "
                                f"be overwritten with {coef_date}!"
                            )
                        else:
                            logger.debug(
                                f"Found coef date for {normalized_group_id} in group {group_name}: "
                                f"{coef_date}"
                            )
                        results[normalized_group_id] = coef_date
                        break  # Found a date for this device, move to next device

    return results


def extract_raw_hdf5_metadata(
    device_dir: Path, dev_ids: List[str], raw_hdf5_cols: set
    ) -> Dict[str, Dict[str, str]]:
    """
    Extract raw HDF5 metadata for devices based on the raw_hdf5_cols configuration.
    from data that have raw time resolution (no averaging, i.e. "proc_noAvg.h5" and "raw.h5" files)
    Args:
        device_dir: Directory to search for HDF5 files
        dev_ids: List of pre-normalized device IDs to extract metadata for
        raw_hdf5_cols: Set of columns to extract (e.g., {"coef_date", "raw_date_range"})

    Returns:
        Dictionary mapping device IDs to their metadata dictionaries
    """
    logger.debug(f"Extracting raw HDF5 metadata: {raw_hdf5_cols} for devices {dev_ids}")
    results = {}

    # Look in all HDF5 files
    h5_files_by_type = find_hdf5_files(device_dir)
    logger.debug(f"Found HDF5 files by type: {h5_files_by_type}")
    # Remove not required sources (which are in priority order already: proc_noAvg, proc_Avg, proc, raw)
    h5_files_by_type = {k: h5_files_by_type[k] for k in ["proc_noAvg", "raw"]}
    for dev_id in dev_ids:
        results[dev_id] = {}

        # Extract coef_date if requested
        if "coef_date" in raw_hdf5_cols:
            coef_date = extract_coef_dates_from_raw_hdf5_files(h5_files_by_type["raw"], [dev_id])
            if dev_id in coef_date:
                results[dev_id]["coef_date"] = coef_date[dev_id]

        # Extract time_raw_st and time_raw_en if requested in priority order :
        if "raw_date_range" in raw_hdf5_cols:
            time_ranges = extract_time_range_from_hdf5_index(h5_files_by_type, [dev_id])
            try:
                results[dev_id]["time_raw_st"], results[dev_id]["time_raw_en"] = time_ranges[dev_id]
            except KeyError:
                pass

    return results


def extract_time_range_from_hdf5_index(
    h5_files_by_type: Dict[str, List[Path]], dev_ids: List[str]
) -> Dict[str, Tuple[str, str]]:
    """
    Extract the minimum and maximum time from HDF5 index data for each device.
    This function finds the time_raw_st and time_raw_en values.

    Args:
        h5_files_by_type: dict of List of HDF5 files with keys in priority order
        dev_ids: List of pre-normalized device IDs to extract time ranges for

    Returns:
        Dictionary mapping device IDs to tuples of (min_time, max_time)
        as strings in ISO format

    Note:
        HDF5 group names should match `config.ptn_devices_groups_part` with optional prefix (see
        `extract_device_ids_from_prefixed_name()`). This means that group names of generic devices (i.e.
        only device types) are ignored
    """
    total_files = sum(len(files) for files in h5_files_by_type.values())
    logger.debug(f"extract_time_range_from_hdf5_index for dev_ids={dev_ids} in {total_files} files")
    results = {}

    # Track which devices have already been found in higher priority files
    found_devices = set()

    for hdf5_type, h5_files in h5_files_by_type.items():
        for h5_file_path in h5_files:
            logger.debug(f"Checking HDF5 file for index time ranges: {h5_file_path}")
            try:
                with tables.open_file(str(h5_file_path), mode="r") as h5file:
                    # Walk through all nodes to find device groups with tables
                    for node in h5file.walk_nodes(where="/", classname="Group"):
                        group_name = Path(node._v_pathname).name
                        if node._v_pathname in ("/", "coef"):
                            continue  # Skip root and coef which must be always subgroup of device group

                        # Only process root-level groups (not subgroups), regardless of file type
                        # This ensures we only get direct children of root like /i01, /i04, etc., not /i01/coef
                        path_parts = node._v_pathname.strip('/').split('/')
                        if len(path_parts) > 1:
                            continue  # only processing root-level groups

                        # Check if this group name matches any of our device IDs
                        dev_ids_cur = extract_device_ids_from_prefixed_name(
                            group_name, msg_what="group name "
                        )
                        logger.debug(f"Extracted device_ids: {dev_ids_cur} from group: {group_name}")

                        # Filter to only requested device IDs not yet found in higher priority files
                        if not (dev_ids_cur := [
                            did for did in dev_ids_cur
                            if did in dev_ids and did not in found_devices
                        ]):
                            continue

                        # Look for tables within this group that might have time index
                        for table_node in h5file.walk_nodes(where=node._v_pathname, classname="Table"):
                            # Extract time range from this specific table
                            table_path = table_node._v_pathname
                            group_path = PurePosixPath(table_path).parent
                            if group_path.name.startswith("log"):
                                continue  # log tables processing is not implemented

                            logger.debug(f"Found table node: {table_node._v_pathname}")


                            time_range_result = _extract_time_range_from_table_path(
                                h5_file_path, table_path
                            )
                            if not time_range_result:
                                continue
                            start_time, end_time = time_range_result
                            logger.debug(f"Converted times: {start_time} to {end_time}")

                            # Update results if this is the first time for the device or if we find a wider range
                            for device_id in dev_ids_cur:
                                if device_id not in results:
                                    results[device_id] = (start_time, end_time)
                                    found_devices.add(device_id)  # Mark this device as found in this priority level
                                    logger.debug(f"Found initial time range for {device_id}: {start_time} to {end_time}")
                                else:
                                    # Compare with existing range and extend if necessary
                                    existing_start, existing_end = results[device_id]
                                    new_start = min(start_time, existing_start)
                                    new_end = max(end_time, existing_end)
                                    results[device_id] = (new_start, new_end)
                                    logger.debug(f"Updated time range for {device_id}: {new_start} to {new_end}")
            except Exception as e:
                logger.error(f"Error reading HDF5 file {h5_file_path}: {e}")

    logger.debug(f"extract_time_range_from_hdf5_index returning: {results}")
    return results
