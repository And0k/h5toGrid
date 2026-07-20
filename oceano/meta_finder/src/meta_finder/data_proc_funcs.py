"""
Specific file processing functions for TCM Metadata Processor.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path, PurePosixPath
import re
from datetime import datetime, timedelta, timezone

from . import config
from meta_finder.parse_data_file_name import parse_device_id_groups, parse_filename_for_metadata
from . import utils_sys
from .parse_data_file_name import normalize_device_id
from .logging_config import setup_logging

logger = setup_logging()

def serial_to_datetime(serial: float) -> str:
    """
    Преобразует Serial Date (Excel или MATLAB) в строку даты/времени в формате ISO.
    Если serial < 100_000, считается Excel (эпоха 1900), иначе — MATLAB (эпоха 0001).

    Args:
        serial: Serial date number (Excel or MATLAB format)

    Returns:
        str: ISO format datetime string
    """
    total_seconds_per_day = 86400
    base_epoch_datetime: datetime
    adjusted_serial: float

    if serial < 100_000:
        # Excel Serial Date (эпоха: 1900-01-01), компенсация фиктивного 29.02.1900
        base_epoch_datetime = datetime(1900, 1, 1)
        adjusted_serial = serial - 2
    else:
        # MATLAB Serial Date (эпоха: 0001-01-01)
        base_epoch_datetime = datetime.min.replace(year=1)
        adjusted_serial = serial - 366 - 1   # -1 - поправка в данных для совместимости с Excel

    days_part = int(adjusted_serial)
    fractional_day = adjusted_serial - days_part
    result_datetime = base_epoch_datetime + timedelta(
        days=days_part, seconds=round(fractional_day * total_seconds_per_day)
    )
    return result_datetime.isoformat()

# Cache for _find_dated_files_with_same_pattern_generic to avoid reprocessing same file groups
# Key: (parent_dir_path, device_id, extension) -> Result: [path_min, path_max]
_find_dated_files_cache: Dict[Tuple[str, str, str], List[Any]] = {}

# Cache for read_file_lines_universal results to avoid re-reading same file groups
# Key: (parent_dir_path, device_id, extension, max_lines, skip_nan_rows) -> Result: (lines, last_line, error)
_read_file_lines_cache: Dict[
    Tuple[str, str, str, Optional[int], bool], Tuple[List[str], Optional[str], Optional[str]]
] = {}


def clear_find_dated_files_cache() -> None:
    """
    Clear the cache for _find_dated_files_with_same_pattern_generic and read_file_lines_universal.

    This should be called at the beginning of processing each device directory
    to ensure that cache doesn't persist across different directories.
    """
    global _find_dated_files_cache, _read_file_lines_cache
    _find_dated_files_cache.clear()
    _read_file_lines_cache.clear()


def _build_dated_mask_pattern(filename: str) -> re.Pattern:
    """Build regex by replacing the dated prefix with digit mask.

    Replaces leading YYMMDD[_HHMM][-DD_HHMM] part with `\\d` equivalents while
    keeping the rest of the filename unchanged. Handles time ranges like
    ``180418_0710-18_2304`` by masking the entire prefix up to the device
    separator. This avoids complex pattern construction from config patterns.

    Masking strategy:
      - Each digit in the dated prefix becomes `\\d`
      - Hyphens and underscores between digit groups are kept literally
      - Everything after the dated prefix (device id, extension) is escaped

    """
    # Match the full dated prefix: YYMMDD[_HHMM][-DD[_HHMM]]
    # This covers: 180418, 180418_0710, 180418_0710-18_2304, etc.
    prefix_match = re.match(
        r'^(\d{4,6}(?:\.\.\d{2,6})?(?:_\d+(?:-\d+(?:_\d+)?)?)?)', filename
    )
    if not prefix_match:
        return re.compile(rf'^{re.escape(filename)}$', re.IGNORECASE)

    dated_prefix = prefix_match.group(1)
    suffix = filename[len(dated_prefix):]

    # Build mask: each digit -> \d, keep separators literal
    masked_prefix = re.sub(r'\d', r'\\d', dated_prefix)

    # Escape the remaining suffix for regex safety
    return re.compile(rf'^{masked_prefix}{re.escape(suffix)}$', re.IGNORECASE)


def _find_dated_files_with_same_pattern_generic(
    base_name: str, paths: list,
) -> list:
    r"""Find files sharing the same dated-part pattern, returning first and last.

    Groups files by the mask built from *base_name* (replacing date/time digits
    with ``\d``), then returns at most two paths: the ones with the smallest
    and largest timestamps among matching files.

    Args:
        base_name: Filename used to build the dated-part mask.
        paths: Iterable of Path or PurePosixPath objects to search.

    Returns:
        List with 1 or 2 elements: [path_min] or [path_min, path_max].
    """
    pattern = _build_dated_mask_pattern(base_name)

    # Extract timestamp prefix (digits before first non-digit separator) for ordering
    ts_re = re.compile(r'^(\d{4,6}(?:\.\.\d{2,6})?(?:_\d+(?:-\d+(?:_\d+)?)?)?)')

    ts_min: Optional[str] = None
    ts_max: Optional[str] = None
    path_min = path_max = None

    for path in paths:
        name = path.name if hasattr(path, 'name') else str(path)
        if not pattern.match(name):
            continue

        ts_match = ts_re.match(name)
        if not ts_match:
            continue
        ts = ts_match.group(1)

        if ts_min is None or ts < ts_min:
            ts_min, path_min = ts, path
            if ts_max is None:
                ts_max, path_max = ts_min, path_min
        elif ts > ts_max:
            ts_max, path_max = ts, path

    # Fallback: if pattern matched nothing, try exact name match
    if path_min is None:
        for path in paths:
            name = path.name if hasattr(path, 'name') else str(path)
            if name == base_name:
                return [path]
        return []

    result = [p for p in (path_min, path_max) if p is not None and p is not path_min]
    return [path_min] + result


def _find_matching_files_in_directory(parent_dir: Path, base_name: str) -> List[Path]:
    """
    Find files with the same pattern in a directory. Returns at most 2 files:
    the ones with smallest and largest timestamps that match the pattern.

    Args:
        parent_dir: The directory to search in
        base_name: The base filename to match against

    Returns:
        List of file paths with at most 2 elements: the files with smallest and largest timestamps.
        Length 1 if there's only one matching file, length 2 if there are multiple.
    """
    # Convert to Path object if it's a PurePosixPath for file system operations
    if isinstance(parent_dir, PurePosixPath):
        parent_dir = Path(parent_dir)
    if parent_dir.exists():
        paths = [file_path for file_path in parent_dir.iterdir() if file_path.is_file()]
        return _find_dated_files_with_same_pattern_generic(base_name, paths)
    else:
        return []


def _find_matching_files_in_archive(dir_archive: Path, rel_path: PurePosixPath) -> List[PurePosixPath]:
    """
    Find all files with the same pattern in an archive.

    Args:
        dir_archive: The archive file to search in
        rel_path: The relative path of the file to match against

    Returns:
        List of file paths with smallest and largest timestamp that match base_name suffix.
        Always returns at least one item if called with arguments from existing files.
    """
    base_name = rel_path.name
    rel_parent = PurePosixPath(rel_path.parent.as_posix())
    try:
        # use all files in the archive that are is in the same directory
        rel_paths = [
            rel_path
            for item in utils_sys.gen_from_archive(dir_archive)
            if not item["is_folder"] and (rel_path := item["rel_path"]).parent == rel_parent
        ]
    except Exception as e:
        rel_paths = []
        logger.error(f"Error listing archive contents {dir_archive}: {e}", exc_info=True)
    return _find_dated_files_with_same_pattern_generic(base_name, rel_paths)


def _get_last_lines_efficiently(
    file_handle, num_lines: int = 1, skip_nan_rows: bool = True
) -> List[str]:
    """
    Read the last N lines of a file efficiently without loading the entire file into memory.
    Optionally skips trailing rows that contain only NaN data.

    Args:
        file_handle: Open file handle positioned at the end of the file
        num_lines: Number of lines to read from the end of the file
        skip_nan_rows: If True, skip trailing rows with only NaN data (default: True)

    Returns:
        List of the last N lines from the file (empty list if file is empty)
    """
    try:
        file_handle.seek(0, 2)  # Seek to end of file
        file_size = file_handle.tell()

        if file_size == 0:
            # Empty file
            return []

        # Read chunks from end until we have enough lines
        chunk_size = 1024
        buffer = ""
        pos = file_size
        lines_found = 0

        while pos > 0 and lines_found < num_lines:
            # Calculate how much to read
            read_size = min(chunk_size, pos)
            pos -= read_size

            file_handle.seek(pos)
            chunk = file_handle.read(read_size)
            buffer = chunk + buffer

            # Count newlines in buffer
            lines_found = buffer.count('\n')

        # Split buffer into lines
        lines = buffer.split('\n')

        # If requested, skip trailing rows with only NaN data
        if skip_nan_rows and lines:
            # Determine separator from the last non-empty line
            sep = "\t"
            for line in reversed(lines):
                if line.strip():
                    sep = "\t" if "\t" in line else "," if "," in line else " "
                    break

            # Filter out trailing NaN rows
            filtered_lines = []
            for line in reversed(lines):
                if not line.strip():
                    continue  # Skip empty lines
                if _row_has_only_nan_data(line, sep):
                    continue  # Skip NaN rows
                filtered_lines.append(line)
                if len(filtered_lines) >= num_lines:
                    break

            # Reverse to get correct order
            return list(reversed(filtered_lines))

        # Return the last num_lines lines
        # Filter out empty lines at the end
        result = []
        for line in reversed(lines):
            if line.strip():
                result.append(line)
                if len(result) >= num_lines:
                    break
        return list(reversed(result))

    except (OSError, IOError) as e:
        logger.error(f"Error reading last lines from file: {e}")
        # Fallback to reading all lines if seeking fails (for very small files)
        try:
            file_handle.seek(0)
            all_lines = file_handle.readlines()
            if all_lines:
                if skip_nan_rows:
                    # Determine separator from first data line
                    sep = "\t" if "\t" in all_lines[0] else "," if "," in all_lines[0] else " "
                    # Find last num_lines lines with valid data
                    filtered_lines = []
                    for line in reversed(all_lines):
                        if line.strip() and not _row_has_only_nan_data(line.rstrip('\n'), sep):
                            filtered_lines.append(line.rstrip('\n'))
                            if len(filtered_lines) >= num_lines:
                                break
                    return list(reversed(filtered_lines))
                # Return last num_lines lines
                return [line.rstrip('\n') for line in all_lines[-num_lines:]]
        except MemoryError:
            logger.error("MemoryError when reading file. File may be too large.")
            raise

    return []


def _get_last_line_efficiently(file_handle, skip_nan_rows: bool = True, is_raw: bool = False) -> Optional[str]:
    """
    Read the last line of a file efficiently without loading the entire file into memory.
    Optionally skips trailing rows that contain only NaN data.
    Searches backward through multiple lines to find one with a valid timestamp.

    Args:
        file_handle: Open file handle positioned at the end of the file
        skip_nan_rows: If True, skip trailing rows with only NaN data (default: True)
        is_raw: Whether the file is in raw inclinometer format (default: False)

    Returns:
        The last line of the file with a valid timestamp, or None if file is empty
    """
    # Read multiple lines from the end of the file to find one with a valid timestamp
    # We read 10 lines to have enough context to find a valid line
    lines = _get_last_lines_efficiently(file_handle, num_lines=10, skip_nan_rows=skip_nan_rows)
    if not lines:
        return None

    # Determine separator from the last non-empty line
    sep = "\t"
    for line in reversed(lines):
        if line.strip():
            sep = "\t" if "\t" in line else "," if "," in line else " "
            break

    # Search backward through the lines to find one with a valid timestamp
    start_idx = len(lines) - 1
    max_attempts = 10
    time_result = _find_valid_time_line(
        lines, start_idx, direction="backward",
        max_attempts=max_attempts, is_raw=is_raw, sep=sep
    )
    if time_result:
        # Return the line with valid timestamp
        return lines[time_result[0]]
    else:
        # Fallback: return the last line even if it doesn't have a valid timestamp
        return lines[-1]


def _is_raw_format(dir_archive: Path, rel_path: PurePosixPath) -> bool:
    """
    Determine if a file is in raw inclinometer format.

    Args:
        dir_archive: Path to archive or directory
        rel_path: Relative path to the file

    Returns:
        True if the file is in raw format, False otherwise
    """
    return "_raw" in str(dir_archive).lower() or "_raw" in str(rel_path).lower()


def _read_first_last_lines(matching_files: List[Path], max_lines: Optional[int] = None, skip_nan_rows: bool = True) -> Tuple[List[str], Optional[str]]:
    """
    Read lines from split files in a directory, optionally skipping trailing NaN rows.

    Args:
        matching_files: List of file paths
        max_lines: Optional maximum number of lines to read from first file
        skip_nan_rows: If True, skip trailing rows with only NaN data when reading last line (default: True)

    Returns:
        Tuple of (list of lines from first file, last line from last file)
    Note: We assume that files has lines
    """
    lines = []
    last_line = None

    if matching_files:
        first_file_path = matching_files[0]
        # Determine if this is a raw format file
        is_raw = _is_raw_format(first_file_path.parent, first_file_path.name)
        # Read first lines from the first file
        with open(first_file_path, "r", errors="ignore") as f:
            if max_lines is not None:
                # Read first max_lines lines
                for i in range(max_lines):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                if len(matching_files) == 1:
                    # Read the last line as this is the last file too
                    last_line = _get_last_line_efficiently(f, skip_nan_rows=skip_nan_rows, is_raw=is_raw)
                    return lines, last_line
            else:
                f.seek(0)  # Reset file pointer to beginning
                lines = f.readlines()

                if len(matching_files) == 1:
                    # Read the last line as this is the last file too
                    if lines:
                        last_line = lines[-1]
                    return lines, last_line

        # Read the last line from the last file
        last_file_path = matching_files[-1]
        # Determine if this is a raw format file
        is_raw = _is_raw_format(last_file_path.parent, last_file_path.name)
        with open(last_file_path, "r", errors="ignore") as f:
            # Read last line efficiently without loading entire file
            last_line = _get_last_line_efficiently(f, skip_nan_rows=skip_nan_rows, is_raw=is_raw)
            if not last_line and lines:
                last_line = lines[-1]

    return lines, last_line


def _read_first_last_lines_from_archived_files(dir_archive: Path, matching_files: List[PurePosixPath], max_lines: Optional[int] = None, skip_nan_rows: bool = True) -> Tuple[List[str], Optional[str]]:
    """
    Read lines from split files in an archive, optionally skipping trailing NaN rows.

    Tries to read directly from archive using libarchive, falling back to
    extracting files to temporary directory for consistent NaN filtering behavior.

    Args:
        dir_archive: The archive file to read from
        matching_files: List of file paths sorted by timestamp
        max_lines: Optional maximum number of lines to read from first file
        skip_nan_rows: If True, skip trailing rows with only NaN data when reading last line (default: True)

    Returns:
        Tuple of (list of lines from first file, last line from last file)
    """
    lines = []
    last_line = None

    if matching_files:
        # Try to read directly from archive using libarchive
        if utils_sys.HAS_LIBARCHIVE:
            try:
                import libarchive as la

                # Determine which files to read (first and last, if different)
                files_to_read = [matching_files[0]]
                if len(matching_files) > 1:
                    files_to_read.append(matching_files[-1])

                # Read first file
                first_file_path = str(files_to_read[0])
                head_lines, _ = utils_sys._read_from_libarchive(
                    dir_archive, first_file_path, n_head=max_lines, skip_header=0
                )
                lines = head_lines

                # Read last file if different from first
                if len(files_to_read) > 1:
                    last_file_path = str(files_to_read[1])
                    _, last_line = utils_sys._read_from_libarchive(
                        dir_archive, last_file_path, n_head=None, skip_header=0
                    )
                else:
                    # Same file, get last line from what we already read
                    last_line = lines[-1] if lines else None

                return lines, last_line
            except Exception as e:
                logger.debug(
                    f"libarchive reading failed for {dir_archive}: {e}, falling back to extraction method"
                )

        # Fallback: extract files to temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Extract only the files we need (first and last, if different)
            files_to_extract = set()
            files_to_extract.add(matching_files[0])
            if len(matching_files) > 1:
                files_to_extract.add(matching_files[-1])

            # Extract the needed files from archive to temporary directory
            if dir_archive.suffix.lower() in config.extensions_archive:
                if dir_archive.suffix.lower() == ".zip":
                    import zipfile
                    with zipfile.ZipFile(dir_archive) as zf:
                        for file_path in files_to_extract:
                            zf.extract(str(file_path), path=temp_dir_path)
                elif dir_archive.suffix.lower() == ".7z":
                    import py7zr
                    with py7zr.SevenZipFile(dir_archive, mode="r") as archive:
                        archive.extract(path=temp_dir_path, targets=[str(fp) for fp in files_to_extract])
            else:
                logger.error(f"Unsupported archive format: {dir_archive.suffix}")
                return lines, last_line

            # Convert only the extracted files to regular Path objects
            # We only extracted matching_files[0] and matching_files[-1] (if different)
            extracted_files = [temp_dir_path / file_path for file_path in files_to_extract]

            # Use _read_first_last_lines on the extracted files
            lines, last_line = _read_first_last_lines(extracted_files, max_lines, skip_nan_rows=skip_nan_rows)

    return lines, last_line


def read_file_lines_universal(
    dir_archive: Path,
    rel_path: PurePosixPath,
    max_lines: Optional[int] = None,
    skip_nan_rows: bool = True,
    max_burst_time_detection: Optional[int] = None,
    seconds_per_line: Optional[int|float] = None,
) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Read lines from a file, handling both regular files and archives.
    - reads the last line regardless of max_lines.
    - For files that may be split by time, reads first lines from first file and last line from last file.
    - Optionally skips trailing rows with only NaN data.

    Args:
        dir_archive: The path to the text file or archive file
        rel_path: relative path to the data file in archive/directory
        max_lines: Optional maximum number of lines to read (None for all lines). Ignored if max_burst_time_detection is provided.
        skip_nan_rows: If True, skip trailing rows with only NaN data when reading last line (default: True)
        max_burst_time_detection: Optional maximum time in seconds for burst detection. When provided,
            reads lines 1 and 20 to calculate time interval, then computes how many lines are needed
            to cover this time span. Overrides max_lines if both are provided.

    Returns:
        Tuple of (list of lines, last line, error message) from the file.
        Last line is None if file is empty. Error message is None if no error occurred.
    """
    lines = []
    last_line = None
    last_error: Optional[str] = None

    # Calculate max_lines from max_burst_time_detection if provided
    if max_burst_time_detection is not None:
        is_raw = _is_raw_format(dir_archive, rel_path)
        skip_header = 4 if is_raw else 0
        calculated_max_lines = utils_sys.calculate_lines_for_burst_time(
            dir_archive, rel_path, max_burst_time_detection,
            parse_datetime_from_row, skip_header=skip_header, is_raw=is_raw, seconds_per_line=seconds_per_line
        )
        if calculated_max_lines is not None:
            max_lines = calculated_max_lines

    try:
        base_name = rel_path.name
        # Handle files checking if there are multiple files with the same pattern (split by time)
        if Path(dir_archive).is_dir():
            # Regular files
            parent_dir = dir_archive / rel_path.parent
            if (matching_files := _find_matching_files_in_directory(parent_dir, base_name)):
                # Check cache for reading results using the same key as file discovery
                # For split files, all files in the same group will have same device and extension
                # Extract device_id and extension from base_name for cache key
                base_meta = parse_filename_for_metadata(base_name)
                device_id = (
                    base_meta["devices"][0]
                    if base_meta
                    and "devices" in base_meta
                    and base_meta["devices"]
                    and base_meta["devices"][0] != "*"
                    else "*"
                )
                cache_key = (str(parent_dir), device_id, rel_path.suffix, max_lines, skip_nan_rows)

                if cache_key in _read_file_lines_cache:
                    logger.debug(f"Using cached read result for {base_name}-like files in {parent_dir}")
                    return _read_file_lines_cache[cache_key]

                lines, last_line = _read_first_last_lines(matching_files, max_lines, skip_nan_rows=skip_nan_rows)
                _read_file_lines_cache[cache_key] = (lines, last_line, last_error)
            else:
                logger.warning(
                    f"No matching files in directory {dir_archive.name}/{rel_path.parent} to "
                    f"filename '{base_name}' when looking second time: bug"
                )
        else:
            # Archive files
            if (matching_files := _find_matching_files_in_archive(dir_archive, rel_path)):
                # Check cache for reading results using the same key as file discovery
                # Extract device_id and extension from base_name for cache key
                base_meta = parse_filename_for_metadata(base_name)
                device_id = (
                    base_meta["devices"][0]
                    if base_meta
                    and "devices" in base_meta
                    and base_meta["devices"]
                    and base_meta["devices"][0] != "*"
                    else "*"
                )
                cache_key = (str(dir_archive), device_id, rel_path.suffix, max_lines, skip_nan_rows)

                if cache_key in _read_file_lines_cache:
                    logger.debug(f"Using cached read result for {base_name}-like files in {dir_archive}")
                    return _read_file_lines_cache[cache_key]

                lines, last_line = _read_first_last_lines_from_archived_files(
                    dir_archive, matching_files, max_lines, skip_nan_rows=skip_nan_rows
                )
                _read_file_lines_cache[cache_key] = (lines, last_line, last_error)
            else:
                logger.warning(
                    f"No matching files in archive {dir_archive.name}/{rel_path.parent} to "
                    f"filename '{base_name}' when looking second time: bug"
                )

        if matching_files:
            # Format file names for logging: single file or list of files
            file_names_str = (
                f": {', '.join(f.name for f in matching_files)}"
                if (n_files := len(matching_files)) > 1
                else ""
            )
            logger.info(
                "".join([
                    "Have read lines",
                    f" (max: {max_lines})" if max_lines else "",
                    " from ",
                    (f"{n_files} {rel_path}-like files" if n_files > 1 else f"{rel_path}"),
                    f" in {dir_archive.name}/" if dir_archive.name != "text_output" else "",
                    file_names_str
                ])
            )
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        logger.error(f"Error reading file {dir_archive / rel_path}: {e}", exc_info=True)

    return lines, last_line, last_error


def _row_has_only_nan_data(line: str, sep: str) -> bool:
    """
    Check if a data row has a valid timestamp but all data columns are NaN/empty.

    Such rows represent gaps in measurement periods when the instrument was not
    actively measuring data, but still logged timestamps.

    Args:
        line: A data row (excluding header)
        sep: Separator character (tab, comma, or space)

    Returns:
        True if all data columns (except first Time column) are NaN/empty, False otherwise
    """
    parts = line.split(sep)
    if len(parts) < 2:
        return False  # Not enough columns to be a valid data row

    # Skip first column (Time), check all data columns
    for value in parts[1:]:
        stripped = value.strip()
        if stripped and stripped not in ('NaN', 'nan', '-', '?', ''):
            return False  # Found valid data
    return True  # All data columns are NaN/empty


def _extract_burst_info_from_lines(
    lines: List[str], averaging_interval: int, is_raw: bool = False
) -> Tuple[int | str, int | str]:
    """
    Extracts burst information (bursts_t and burst_dt) from lines by analyzing gaps in the time data.

    Args:
        lines: List of lines from the data file (including header)
        averaging_interval: The averaging interval in seconds from the filename
        is_raw: Whether the file is in raw inclinometer format (Year,Month,Day,Hour,Minute,Second)

    Returns:
        A tuple containing (bursts_t, burst_dt) as integers or "-", "-" if not found.
        bursts_t: Interval between bursts (time from start of one burst to start of next burst)
        burst_dt: Duration of work within a single burst (from start to end of burst)
    """
    # Default values
    bursts_t = "-"
    burst_dt = "-"

    if len(lines) <= 1:  # Only header or no data
        logger.debug("Not enough lines to analyze gaps")
        return bursts_t, burst_dt

    # Parse time data
    sep = "\t" if "\t" in lines[0] else "," if "," in lines[0] else " "
    n_bad = 0
    try:
        # Skip header and parse timestamps as tuples of (string, datetime)
        timestamps = []
        for line in lines[1:]:
            dt = parse_datetime_from_row(line, is_raw=is_raw, sep=sep)
            if dt is None:
                n_bad += 1
            else:
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                timestamps.append((timestamp_str, to_utc_naive(dt)))
    except Exception as e:
        logger.error(f"Error converting lines: {e}", exc_info=True)

    if len(timestamps) < 2:
        logger.warning(
            f"Not enough valid timestamps to analyze gaps. Number of bad lines: {n_bad}/{len(lines)}"
        )
        return bursts_t, burst_dt
    elif n_bad:
        logger.info(f"Skipped {n_bad}/{len(lines)} bad lines, ...")
    # Find gaps in the data
    gaps = []
    # gap_threshold is in seconds - use max of 10s and 2x averaging interval
    gap_threshold = max(10, averaging_interval * 2)  # Sensible and longer than averaging interval

    # Track minimum time difference for validation
    min_time_diff = float('inf')

    for i in range(1, len(timestamps)):
        prev_time = timestamps[i-1]
        curr_time = timestamps[i]
        # timestamps contains tuples of (timestamp_str, datetime), use datetime part (index 1)
        time_diff = (curr_time[1] - prev_time[1]).total_seconds()

        # Track minimum time difference
        if time_diff < min_time_diff:
            min_time_diff = time_diff

        if time_diff > gap_threshold:
            gaps.append({
                'index': i,
                'prev_timestamp': prev_time[0],
                'curr_timestamp': curr_time[0],
                'gap_duration': time_diff
            })

    # Validate that minimum delta time is not significantly larger than averaging interval
    if min_time_diff > averaging_interval:
        logger.warning(
            f"Minimum delta time ({min_time_diff}s) is larger than averaging interval "
            f"({averaging_interval}s). This may indicate incorrect burst detection or "
            f"filename parsing. Gap threshold used: {gap_threshold}s, found {len(gaps)} gaps."
        )

    # Track the first data point for burst calculations
    first_data_point = timestamps[0][0] if timestamps else None

    # If we found at least 2 gaps, we can calculate burst info
    # We analyze all gaps to find the maximum data interval for burst_dt and bursts_t
    if len(gaps) >= 2 and first_data_point:
        time_first_data_point = datetime.strptime(first_data_point, "%Y-%m-%d %H:%M:%S")

        # Calculate burst intervals from all gap pairs and take the maximum
        max_bursts_t_seconds = 0
        max_burst_dt_seconds = 0

        for i in range(len(gaps) - 1):
            gap_i = gaps[i]
            gap_j = gaps[i + 1]

            # Extract timestamps from gaps
            time_before_gap_i = datetime.strptime(gap_i['prev_timestamp'], "%Y-%m-%d %H:%M:%S")
            time_after_gap_i = datetime.strptime(gap_i['curr_timestamp'], "%Y-%m-%d %H:%M:%S")
            time_before_gap_j = datetime.strptime(gap_j['prev_timestamp'], "%Y-%m-%d %H:%M:%S")
            time_after_gap_j = datetime.strptime(gap_j['curr_timestamp'], "%Y-%m-%d %H:%M:%S")

            # Calculate burst interval (time between consecutive burst starts)
            # For the first gap pair, use first data point as burst start
            if i == 0:
                bursts_t_seconds = (time_after_gap_i - time_first_data_point).total_seconds()
            else:
                bursts_t_seconds = (time_after_gap_j - time_after_gap_i).total_seconds()

            # Calculate burst duration (time from start to end of a burst)
            burst_dt_seconds = (time_before_gap_j - time_after_gap_i).total_seconds()

            # Track maximum intervals
            if bursts_t_seconds > max_bursts_t_seconds:
                max_bursts_t_seconds = bursts_t_seconds
            if burst_dt_seconds > max_burst_dt_seconds:
                max_burst_dt_seconds = burst_dt_seconds

        # If burst_dt is 0 or negative, it means there's no valid data between gaps
        # This can happen when gaps are adjacent or overlapping
        # In this case, we should not report burst info as it's not a valid burst pattern
        # If there's no valid burst (burst_dt <= 0), then bursts_t is also meaningless
        if max_burst_dt_seconds <= 0:
            logger.debug(
                f"Invalid burst duration ({max_burst_dt_seconds}s) - gaps may be adjacent or overlapping. "
                f"Found {len(gaps)} gaps. Since there's no valid data between gaps, burst pattern is invalid."
            )
            bursts_t = "-"
            burst_dt = "-"
        else:
            bursts_t = int(max_bursts_t_seconds)
            burst_dt = int(max_burst_dt_seconds)

    return bursts_t, burst_dt


def to_utc_naive(dt):
    if dt.tzinfo is None:
        # Already naive, assume it's UTC or return as-is
        return dt
    else:
        # Has timezone, convert to UTC then make naive
        return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _get_time_search_params(is_raw: bool) -> Tuple[int, int]:
    """
    Get the starting index and max attempts for time search.

    For raw format: Start from line 0, skip up to 4 lines to find valid time
    For standard format: Start from line 1, skip up to 1 line to find valid time

    Args:
        is_raw: Whether the file is in raw inclinometer format

    Returns:
        Tuple of (start_index, max_attempts)
    """
    if is_raw:
        return (0, 4)  # Raw format: start at line 0, skip up to 4 lines
    else:
        return (1, 1)  # Standard format: start at line 1, skip up to 1 line


def _find_valid_time_line(
    lines: List[str],
    start_idx: int,
    direction: str = "forward",
    max_attempts: int = 10,
    is_raw: bool = False,
    sep: str = "\t"
) -> Optional[Tuple[int, datetime]]:
    """
    Find a line with a valid timestamp, searching forward or backward from start index.

    This handles bad lines at the start or end of files, particularly in raw format.

    Args:
        lines: List of lines to search
        start_idx: Starting index for search
        direction: "forward" to search ahead, "backward" to search behind
        max_attempts: Maximum number of lines to check
        is_raw: Whether the file is in raw inclinometer format
        sep: Separator character for parsing

    Returns:
        Tuple of (line_index, datetime) if found, None otherwise
    """
    # Search forward/backward from start_idx
    for i in range(
        start_idx, *(
            [min(start_idx + max_attempts, len(lines))] if direction == "forward" else
            [max(start_idx - max_attempts, -1), -1]
        )):
        if (datetime_out := parse_datetime_from_row(lines[i], is_raw=is_raw, sep=sep)):
            return (i, datetime_out)

    return None


def parse_inclinometer_time_format(time_row: List[str]) -> Optional[datetime]:
    """
    Parse time from the inclinometer format: Year,Month,Day,Hour,Minute,Second

    Args:
        time_row: List of time components [year, month, day, hour, minute, second]

    Returns:
        datetime object or None if parsing fails
    """
    try:
        return datetime(*[int(component) for component in time_row[:6]])
    except (ValueError, IndexError) as e:
        logger.debug(f"Bad inclinometer time format {time_row}: {e}, continue...")
    return None


def parse_datetime_from_row(line: str, is_raw: bool = False, sep="\t") -> Optional[datetime]:
    """
    Parse datetime from a data row, handling both standard and raw inclinometer formats,
    plus Serial MATLAB/Excel DateTime format as fallback.

    Args:
        line: A line from the data file
        is_raw: Whether the line is in raw inclinometer format (Year,Month,Day,Hour,Minute,Second)

    Returns:
        datetime object or None if parsing fails
    """
    if not line.strip():
        return None

    if is_raw:
        # For raw inclinometer format, use regex to split by multiple possible separators at once
        # This handles both comma and tab separators commonly used in raw format
        # Split by comma, tab or space (most common separators for raw format)
        time_parts = re.split(r'[,\t ]', line.strip())
        if len(time_parts) >= 6:
            return parse_inclinometer_time_format(time_parts)
        else:
            logger.debug(f'Raw inclinometer format line doesn\'t have 6 time components: "{line[:100]}"...')
            return None
    else:
        # For standard format, extract timestamp from first column
        timestamp_str = line.split(sep, 1)[0].strip()

        try:
            # Try standard ISO format parsing
            return datetime.fromisoformat(timestamp_str)
        except ValueError as e:
            # return None without attempting further parsing for intentionally not time values
            if timestamp_str.upper() == 'NAN':
                return None
            # Detect header lines by checking if the first column contains non-numeric text
            # Headers typically have column names like "Time", "Date", "Year", etc.
            # Check if the first column looks like a header (contains letters but not a valid datetime format)
            if timestamp_str and not any(char.isdigit() for char in timestamp_str):
                # This is likely a header line with text like "Time", "Date", etc.
                return None
            # Check if the error is due to timezone offset format without colon
            # Format like: 2016-02-19T16:02:30.000000+0200 (should be +02:00)
            if '+' in timestamp_str or timestamp_str.endswith('Z'):
                # Try to fix timezone offset format by adding colon if needed
                # Pattern: +HHMM or -HHMM at the end of the string
                tz_pattern = re.compile(r'([+-])(\d{2})(\d{2})$')
                match = tz_pattern.search(timestamp_str)
                if match:
                    # Reconstruct timestamp with colon in timezone offset
                    sign, hours, minutes = match.groups()
                    fixed_timestamp = tz_pattern.sub(rf'{sign}{hours}:{minutes}', timestamp_str)
                    try:
                        return datetime.fromisoformat(fixed_timestamp)
                    except ValueError:
                        pass  # Fall through to next attempt

            # Try Serial MATLAB/Excel DateTime format as fallback
            try:
                serial_value = float(timestamp_str)
                iso_datetime_str = serial_to_datetime(serial_value)
                return datetime.fromisoformat(iso_datetime_str)
            except (ValueError, TypeError) as serial_e:
                logger.debug(
                    f"Failed parsing datetime from {line}: standard format failed ({e}), "
                    f"Serial MATLAB/Excel format failed ({serial_e})"
                )
                return None


def extract_time_info_from_text_file(
    dir_archive: Path, rel_path: PurePosixPath, averaging_interval: Optional[int] = None
) -> Optional[Tuple[str, str, int | str, int | str]]:
    """
    Extracts the time range from a text file, excluding trailing NaN data rows.

    Args:
        dir_archive: The path to the text file or archive file with format
        rel_path: relative path to the data file in archive/directory `dir_archive`
        averaging_interval: Optional averaging interval (delta time between adjacent rows in seconds)
            for burst detection (used for gap threshold calculation).

    Returns:
        A tuple containing the start and end time strings, and burst info (bursts_t, burst_dt),
        or None if not found. Burst values are integers or "-" if not found.
    """
    start_time, end_time = None, None
    bursts_t, burst_dt = "-", "-"

    try:
        # For burst detection, use time-based calculation via max_burst_time_detection
        # The read_file_lines_universal function will read lines 1 and 20 to calculate
        # the time interval, then compute how many lines are needed to cover the configured time
        # Read lines using universal function (with skip_nan_rows=True to exclude trailing NaN rows)
        lines, last_line, read_error = read_file_lines_universal(
            dir_archive, rel_path, max_burst_time_detection=config.max_burst_time_detection,
            skip_nan_rows=True, seconds_per_line=averaging_interval
        )
        if not lines:
            logger.warning(
                f"{dir_archive.name}/{rel_path} - no lines read"
                + (f". Error: {read_error}" if read_error else "")
            )
            return None

        # Determine if this is a raw inclinometer format file based on path
        is_raw = _is_raw_format(dir_archive, rel_path)

        # Extract time range from lines first
        sep = "\t" if "\t" in lines[0] else "," if "," in lines[0] else " "

        # Find start time by using helper function to search for valid time entry
        start_idx, max_attempts = _get_time_search_params(is_raw)
        start_result = _find_valid_time_line(
            lines, start_idx, direction="forward",
            max_attempts=max_attempts, is_raw=is_raw, sep=sep
        )
        if start_result:
            start_time = to_utc_naive(start_result[1])

        # Find end time from last_line (which is now validated to have a valid timestamp)
        # The _get_last_line_efficiently function already searched through multiple lines
        # to find one with a valid timestamp, so we can directly parse it
        if last_line:
            end_result = parse_datetime_from_row(last_line, is_raw=is_raw, sep=sep)
            if end_result:
                end_time = to_utc_naive(end_result)

        # If start_time is None but end_time is found, use end_time as start_time
        if start_time is None and end_time is not None:
            start_time = end_time

        # Only extract burst information if we have a valid start time
        if start_time and averaging_interval is not None and len(lines) > 1:
            bursts_t, burst_dt = _extract_burst_info_from_lines(lines, averaging_interval, is_raw=is_raw)

        if start_time and end_time:
            out = (
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S"),
                burst_dt,
                bursts_t,
            )
            # Log the extracted time range and burst info (always show all values)
            logger.debug(
                f"Extracted from {dir_archive.name}/{rel_path}: "
                f"time_st={out[0]}, time_en={out[1]}, burst_dt={burst_dt}, bursts_t={bursts_t}"
            )
            return out

        logger.warning(
            f"Time extraction is not successful from {dir_archive / rel_path}, ..."
            # f": 2 1st lines = {lines[:2]},\nlast_line = {last_line}\n"
            # f"start_time={start_time} and end_time={end_time},",
            # exc_info=True,
        )
    except Exception as e:
        logger.error(f"Error reading or processing file {dir_archive / rel_path}: {e}", exc_info=True)
    return None


def _extract_device_ids_from_groups_in_data(devices_str: str) -> List[str]:
    """
    Extract device IDs specified in column name suffix or in hdf5 group.

    Only processes column names that match ptn_devices_groups_part.
    Handles arbitrary number of devices by calling parse_device_group()
    after replacing all underscores between digits to commas.

    Examples:
        'i03' -> ['i3']
        'i05_14' -> ['i5', 'i14'] (underscore between digits replaced with comma)
        'i_bin7200s', 'i', 'param' -> [] (doesn't match device pattern)
    Returns:
        List of normalized device IDs extracted from the column name
    """


    # Replace ALL underscores between ALL digits with commas for parse_device_group
    # This handles cases like 'i05_14_27' -> 'i05,14,27'
    dev_groups_str = re.sub(r"(\d)_(\d)", r"\1,\2", devices_str)

    # Extract the full matched device part suffix
    if not (
        match := re.match(config.ptn_devices_groups_part, dev_groups_str, re.IGNORECASE)
        ) or not match.group(0):  # type regex group is required
        return []
    dev_groups_validated = match.group()
    try:
        return parse_device_id_groups(dev_groups_validated)
    except Exception as e:
        logger.debug(f"Failed to parse devices ({dev_groups_validated}) from column '{devices_str}': {e}")
        return []


def _extract_device_ids_from_column_name(column_name: str) -> List[str]:
    """
    Extract device IDs from a column name.
    Only processes column names that match ptn_devices_groups_part.
    Handles arbitrary number of devices (see parse_device_group())
    after replacing all underscores between digits to commas.

    Examples:
        'param_i03' -> ['i3']
        'param_i05_14' -> ['i5', 'i14'] (underscore between digits replaced with comma)
        'param_i_bin7200s' -> [] (doesn't match device pattern)
        'param' -> [] (doesn't match device pattern)

    Returns:
        List of normalized device IDs extracted from the column name
    """
    device_ids_str = re.split("[@#_]+", column_name, 1)[-1]  # on separator, prefix or both
    return _extract_device_ids_from_groups_in_data(device_ids_str)


def extract_time_ranges_from_combined_file(
    dir_archive: Path, rel_path: PurePosixPath, dev_ids: List[str],
    averaging_interval: Optional[int|float] = None
) -> Tuple[Dict[str, Optional[Tuple[str, str, int | str, int | str]]], Dict[str, Any]]:
    """
    Extracts device-specific time ranges from a combined data file that contains data for multiple devices.

    Args:
        dir_archive: The path to the text file or archive file with format
        rel_path: relative path to the combined data file in archive/directory `dir_archive`
        dev_ids: List of pre-normalized device IDs to extract time ranges for
        averaging_interval: Optional averaging interval for burst detection

    Returns:
        A tuple containing:
        - A dictionary mapping device IDs to tuples of (start_time, end_time, bursts_t, burst_dt), or None if not found.
          Burst values are integers or "-" if not found.
        - _combined_comments: A dictionary about combined device columns.
    """
    logger.info(
        f"Extracting time ranges from combined file {dir_archive / rel_path} for devices {dev_ids}"
    )
    logger.debug(f"Processing combined file: {dir_archive / rel_path}, requested devices: {dev_ids}")

    result = {}
    lines = []
    try:
        # Handle archive files
        if not Path(dir_archive).is_dir():
            # Use utils_sys to read all lines from archive

            archive_lines = utils_sys.read_first_last_lines(
                dir_archive, rel_path, skip_header=0
            )  # Don't skip header yet
            if archive_lines and archive_lines[0] and archive_lines[1]:
                # For combined files, we need all lines, not just first and last
                # We'll need to extract the full content
                lines, last_line = utils_sys.read_archive_file_lines(dir_archive, rel_path)

                # For archive files, use the original approach since we already have the lines
                if not lines:
                    logger.warning(f"No lines read from {dir_archive / rel_path}")
                    # Return None for all requested devices
                    for dev_id in dev_ids:
                        result[dev_id] = None
                    _combined_comments = {}
                    return result, {}

                # Extract time ranges for all devices and combined column information
                device_time_ranges, _combined_comments = _extract_device_time_ranges_from_combined_content(
                    dir_archive, rel_path, lines, dev_ids
                )
        else:
            # Handle regular files - check file size first to determine how to process
            file_path = dir_archive / rel_path
            file_size = file_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB threshold
                # For large files, use the streaming approach to extract time ranges
                device_time_ranges, _combined_comments = _extract_device_time_ranges_from_large_combined_file(
                    dir_archive, rel_path, dev_ids
                )
            else:
                # For smaller files, use the original approach
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                if not lines:
                    logger.warning(f"No lines read from {dir_archive / rel_path}")
                    # Return None for all requested devices
                    for dev_id in dev_ids:
                        result[dev_id] = None
                    _combined_comments = {}
                    return result, {}

                # Extract time ranges for all devices and combined column information
                device_time_ranges, _combined_comments = _extract_device_time_ranges_from_combined_content(
                    dir_archive, rel_path, lines, dev_ids
                )

            # Extract burst information if averaging_interval is provided
            bursts_info = {}
            if averaging_interval is not None and averaging_interval > 0:
                # For burst detection, use time-based calculation via max_burst_time_detection
                # The read_file_lines_universal function will read lines 1 and 20 to calculate
                # the time interval, then compute how many lines are needed to cover the configured time
                burst_lines, _, _ = read_file_lines_universal(
                    dir_archive, rel_path, max_burst_time_detection=config.max_burst_time_detection,
                    seconds_per_line=averaging_interval
                )
                if len(burst_lines) > 1:
                    # Determine if this is a raw inclinometer format file based on path
                    is_raw = _is_raw_format(dir_archive, rel_path)
                    bursts_t, burst_dt = _extract_burst_info_from_lines(burst_lines, averaging_interval, is_raw=is_raw)
                    # Store burst info for all devices (same for all in combined file)
                    bursts_info = {"bursts_t": bursts_t, "burst_dt": burst_dt}

            # Merge results with burst information
            for dev_id, time_range in device_time_ranges.items():
                if time_range is not None and len(time_range) == 2:
                    start_time, end_time = time_range
                    burst_dt = bursts_info.get("burst_dt", "-")
                    bursts_t = bursts_info.get("bursts_t", "-")
                    result[dev_id] = (start_time, end_time, burst_dt, bursts_t)
                    # Log the extracted time range and burst info for each device
                    logger.debug(
                        f"Extracted from combined file {dir_archive.name}/{rel_path} for {dev_id}: "
                        f"time_st={start_time}, time_en={end_time}, burst_dt={burst_dt}, bursts_t={bursts_t}"
                    )
                else:
                    result[dev_id] = time_range  # Keep as None or original value

    except MemoryError as me:
        logger.error(f"MemoryError when processing combined file {dir_archive / rel_path}: {me}")
        logger.error(f"File size: {file_path.stat().st_size if 'file_path' in locals() else 'unknown'} bytes")
        # Return None for all requested devices
        for dev_id in dev_ids:
            result[dev_id] = None
        _combined_comments = {}
        return result, _combined_comments
    except Exception as e:
        logger.error(f"Error processing combined file {dir_archive / rel_path}: {e}", exc_info=True)
        # Return None for all requested devices
        for dev_id in dev_ids:
            result[dev_id] = None
        _combined_comments = {}

    return result, _combined_comments


def _extract_device_time_ranges_from_large_combined_file(dir_archive, rel_path, dev_ids):
    """
    Helper function to extract device-specific time ranges from large file content using chunked processing.

    Args:
        dir_archive: The path to the text file or archive file with format
        rel_path: relative path to the combined data file in archive/directory `dir_archive`
        dev_ids: List of pre-normalized device IDs to extract time ranges for
    Returns:
        A tuple containing:
        - A dictionary mapping device IDs to tuples of (start_time, end_time) or None if not found
        - A dictionary mapping combined device pairs to special comment strings
    """
    # Initialize result dictionary
    device_time_ranges = {dev_id: None for dev_id in dev_ids}

    # Check if we need to extract all devices (when dev_ids is empty)
    extract_all_devices = len(dev_ids) == 0

    # Initialize tracking dictionaries with device IDs
    if extract_all_devices:
        # We'll initialize these after we know what devices we have
        device_start_times = {}
        device_end_times = {}
    else:
        device_start_times = {dev_id: None for dev_id in dev_ids}
        device_end_times = {dev_id: None for dev_id in dev_ids}

    # Track combined device column information
    combined_comments = {}  # Maps device pairs to special comment strings

    try:
        file_path = dir_archive / rel_path
        # Determine if this is an raw inclinometer format file (from _raw directory)
        is_raw = _is_raw_format(dir_archive, rel_path)

        # First, read and process the header to identify device columns
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header_line = f.readline()

            if not header_line.strip():
                logger.warning(f"File {file_path} is empty, skipping")
                return device_time_ranges, combined_comments

            # Parse header line to identify device columns
            header = header_line.strip().split('\t')

            # Identify device-specific columns
            device_columns = {}  # Maps device_id to list of column indices

            # Look for device identifiers in column headers
            # Examples: "Vabs_i01", "Vdir_i01", "Inclination_i01", etc.

            # If dev_ids is empty, we want to extract all device IDs from column names
            extract_all_devices = len(dev_ids) == 0

            for i, column_name in enumerate(header):
                col_device_ids = _extract_device_ids_from_column_name(column_name)
                # If we're extracting all devices or if the device is in our list
                # Use pre-normalized device IDs
                for device_id in col_device_ids:
                    normalized_device_id = normalize_device_id(device_id)
                    if extract_all_devices or normalized_device_id in dev_ids:
                        if normalized_device_id not in device_columns:
                            device_columns[normalized_device_id] = []
                        device_columns[normalized_device_id].append(i)

                # Check if this column contains data for multiple devices (combined column)
                # If so, create a special comment for those devices
                if len(col_device_ids) > 1:
                    combined_key = "+".join(sorted(col_device_ids))
                    combined_comments[combined_key] = f"{combined_key} output"

            # Special case: if no device-specific columns found but we have device IDs,
            # check if this is a combined file without device suffixes in column names
            # In this case, we should not assign columns to devices that don't exist
            # Use pre-normalized device IDs
            if not any(norm_id in device_columns for norm_id in dev_ids) and dev_ids:
                # For combined files without device-specific column names, we can't determine
                # which columns belong to which devices, so we return None for all devices
                logger.warning(f"No device-specific columns found in {file_path} for requested devices {dev_ids}")
                for dev_id in dev_ids:
                    device_time_ranges[dev_id] = None

            # If we're extracting all devices, initialize the tracking dictionaries now
            if extract_all_devices and device_columns:
                device_ids = list(device_columns.keys())
                device_time_ranges = {dev_id: None for dev_id in device_ids}
                device_start_times = {dev_id: None for dev_id in device_ids}
                device_end_times = {dev_id: None for dev_id in device_ids}

            # Now process the data lines in chunks to avoid memory issues
            # Use _find_valid_time_line to find first valid data line without warning
            start_line, max_attempts = _get_time_search_params(is_raw)
            sep = "\t" if "\t" in header_line else "," if "," in header_line else " "

            # Read first few lines to find first valid data line
            sample_lines = [header_line]
            for _ in range(max_attempts):
                line = f.readline()
                if not line:
                    break
                sample_lines.append(line)

            time_result = _find_valid_time_line(
                sample_lines, start_line, direction="forward",
                max_attempts=max_attempts, is_raw=is_raw, sep=sep
            )
            if time_result:
                # Found first valid data line, skip to it
                lines_to_skip = time_result[0]
                for _ in range(lines_to_skip):
                    f.readline()

            time_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)")

            # Process file line by line to avoid memory issues
            for line_num, line in enumerate(f, 1):  # Start counting from 1 for the first data line

                if not line.strip():
                    continue

                if is_raw:
                    columns = line.strip().split(',')
                else:
                    columns = line.strip().split('\t')
                if not columns:
                    continue

                # Extract timestamp based on format
                if is_raw:
                    # For inclinometer format, first 6 columns are Year,Month,Day,Hour,Minute,Second
                    if len(columns) >= 6:
                        time_row = columns[:6]
                        parsed_time = parse_inclinometer_time_format(time_row)
                        if parsed_time:
                            timestamp = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            continue  # Skip this line if time parsing fails
                    else:
                        continue  # Skip if not enough columns for time info
                else:
                    # For standard format, use the existing time pattern matching
                    time_match = time_pattern.match(columns[0]) if columns else None
                    if not time_match:
                        continue
                    timestamp = time_match.group(1)

                # Check each device for data in this row
                for dev_id, column_indices in device_columns.items():
                    # Check if any of the device's columns have data in this row
                    has_data = False
                    for col_index in column_indices:
                        if col_index < len(columns) and columns[col_index].strip():
                            # Consider non-empty values as data
                            value = columns[col_index].strip()
                            if value and value.lower() not in ['nan', 'null']:
                                has_data = True
                                break

                    if has_data:
                        # Update start time if not set
                        if device_start_times[dev_id] is None:
                            device_start_times[dev_id] = timestamp

                        # Always update end time
                        device_end_times[dev_id] = timestamp

        # Compile results
        for dev_id, start_time in device_start_times.items():
            if start_time is not None:
                if device_end_times[dev_id] is None:
                    # If file has only one line (header), or just header + one data line, use same start/end time
                    device_end_times[dev_id] = start_time
                device_time_ranges[dev_id] = (start_time, device_end_times[dev_id])

    except Exception as e:
        logger.error(f"Error processing large file content from {file_path}: {e}", exc_info=True)
        return device_time_ranges, combined_comments

    return device_time_ranges, combined_comments


def _extract_device_time_ranges_from_combined_content(dir_archive, rel_path, lines, dev_ids):
    """
    Helper function to extract device-specific time ranges from file content.

    Args:
        dir_archive: The path to the text file or archive file with format
        rel_path: relative path to the combined data file in archive/directory `dir_archive`
        lines: List of lines from the file
        dev_ids: List of pre-normalized device IDs to extract time ranges for
    Returns:
        A tuple containing:
        - A dictionary mapping device IDs to tuples of (start_time, end_time) or None if not found
        - A dictionary mapping combined device pairs to special comment strings
    """
    # Initialize result dictionary
    device_time_ranges = {dev_id: None for dev_id in dev_ids}

    # Check if we need to extract all devices (when dev_ids is empty)
    extract_all_devices = len(dev_ids) == 0

    # Initialize tracking dictionaries with device IDs
    if extract_all_devices:
        # We'll initialize these after we know what devices we have
        device_start_times = {}
        device_end_times = {}
    else:
        device_start_times = {dev_id: None for dev_id in dev_ids}
        device_end_times = {dev_id: None for dev_id in dev_ids}

    # Track combined device column information
    combined_comments = {}  # Maps device pairs to special comment strings

    time_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)")

    if len(lines) < 1:
        logger.warning(f"File {dir_archive / rel_path} is empty, skipping")
        return device_time_ranges, combined_comments

    try:
        # Determine if this is an raw inclinometer format file (from _raw directory)
        is_raw = _is_raw_format(dir_archive, rel_path)

        # Parse header line to identify device columns
        header = lines[0].strip().split('\t')

        # Identify device-specific columns
        device_columns = {}  # Maps device_id to list of column indices

        # Look for device identifiers in column headers
        # Examples: "Vabs_i01", "Vdir_i01", "Inclination_i01", etc.

        # If dev_ids is empty, we want to extract all device IDs from column names
        extract_all_devices = len(dev_ids) == 0

        for i, column_name in enumerate(header):
            col_device_ids = _extract_device_ids_from_column_name(column_name)
            # If we're extracting all devices or if the device is in our list
            # Use pre-normalized device IDs
            for device_id in col_device_ids:
                normalized_device_id = normalize_device_id(device_id)
                if extract_all_devices or normalized_device_id in dev_ids:
                    if normalized_device_id not in device_columns:
                        device_columns[normalized_device_id] = []
                    device_columns[normalized_device_id].append(i)

            # Check if this column contains data for multiple devices (combined column)
            # If so, create a special comment for those devices
            if len(col_device_ids) > 1:
                combined_key = "+".join(sorted(col_device_ids))
                combined_comments[combined_key] = f"{combined_key} output"

        # Special case: if no device-specific columns found but we have device IDs,
        # check if this is a combined file without device suffixes in column names
        # In this case, we should not assign columns to devices that don't exist
        # Use pre-normalized device IDs
        if not any(norm_id in device_columns for norm_id in dev_ids) and dev_ids:
            # For combined files without device-specific column names, we can't determine
            # which columns belong to which devices, so we return None for all devices
            logger.warning(f"No device-specific columns found in {dir_archive / rel_path} for requested devices {dev_ids}")
            for dev_id in dev_ids:
                device_time_ranges[dev_id] = None

        # If we're extracting all devices, initialize the tracking dictionaries now
        if extract_all_devices and device_columns:
            device_ids = list(device_columns.keys())
            device_time_ranges = {dev_id: None for dev_id in device_ids}
            device_start_times = {dev_id: None for dev_id in device_ids}
            device_end_times = {dev_id: None for dev_id in device_ids}

        # Process data lines - skip header lines for raw inclinometer format
        # Use _find_valid_time_line to find first valid data line without warning
        start_line, max_attempts = _get_time_search_params(is_raw)
        sep = "\t" if "\t" in lines[0] else "," if "," in lines[0] else " "
        time_result = _find_valid_time_line(
            lines, start_line, direction="forward",
            max_attempts=max_attempts, is_raw=is_raw, sep=sep
        )
        if time_result:
            start_line = time_result[0]

        for line_num, line in enumerate(lines[start_line:], start_line):  # Skip header line(s)
            if not line.strip():
                continue

            if is_raw:
                columns = line.strip().split(',')
            else:
                columns = line.strip().split('\t')
            if not columns:
                continue

            # Extract timestamp based on format
            if is_raw:
                # For inclinometer format, first 6 columns are Year,Month,Day,Hour,Minute,Second
                if len(columns) >= 6:
                    time_row = columns[:6]
                    parsed_time = parse_inclinometer_time_format(time_row)
                    if parsed_time:
                        timestamp = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        continue  # Skip this line if time parsing fails
                else:
                    continue # Skip if not enough columns for time info
            else:
                # For standard format, use the existing time pattern matching
                time_match = time_pattern.match(columns[0]) if columns else None
                if not time_match:
                    continue
                timestamp = time_match.group(1)

            # Check each device for data in this row
            for dev_id, column_indices in device_columns.items():
                # Check if any of the device's columns have data in this row
                has_data = False
                for col_index in column_indices:
                    if col_index < len(columns) and columns[col_index].strip():
                        # Consider non-empty values as data
                        value = columns[col_index].strip()
                        if value and value.lower() not in ['nan', 'null']:
                            has_data = True
                            break

                if has_data:
                    # Update start time if not set
                    if device_start_times[dev_id] is None:
                        device_start_times[dev_id] = timestamp

                    # Always update end time
                    device_end_times[dev_id] = timestamp

        # Compile results
        for dev_id, start_time in device_start_times.items():
            if start_time is not None:
                if device_end_times[dev_id] is None:
                    # If file has only one line (header), or just header + one data line, use same start/end time
                    device_end_times[dev_id] = start_time
                device_time_ranges[dev_id] = (start_time, device_end_times[dev_id])

    except Exception as e:
        logger.error(f"Error processing file content from {dir_archive / rel_path}: {e}", exc_info=True)
        return device_time_ranges, combined_comments

    return device_time_ranges, combined_comments


def process_text_output_directory(
    device_results: Dict[str, Any],
    combined_comments: Dict[str, Any],
    text_files_dict: Dict[PurePosixPath, List[PurePosixPath]],
    dev_ids_list: List[str],
):
    """
    Process a single text output directory or archive.
    Args:
        device_results: Dictionary to store device results:
        {device_id: {"time_info": , {"data_paths": {path_tuple: metadata}}}
        text_output_dir: Path to the text output directory or archive
        combined_comments: Dictionary to store combined comments
        dev_ids_list: List of device IDs to look for
    """

    # First, collect all possible files for each device (not just the first one we find)
    all_device_candidates = {}

    # Look for all files and extract devices from them
    for dir_path, rel_paths in text_files_dict.items():
        for rel_path in rel_paths:
            file_name = rel_path.name
            try:
                metadata = parse_filename_for_metadata(file_name)
                if metadata and "devices" in metadata:
                    devices_in_file = metadata["devices"]
                    path_tuple = (dir_path, rel_path)

                    # Handle combined files (marked with "*") or generic prefix files (i, w, p)
                    if any(d in devices_in_file for d in ["*", "i", "w", "p"]):
                        # Extract time ranges for all devices from combined file
                        # Extract device IDs from column names in the file
                        device_time_ranges, _combined_comments = extract_time_ranges_from_combined_file(
                            dir_path, rel_path, dev_ids_list
                        )

                        # Process combined comments if they exist
                        if _combined_comments:
                            path_str = f"{dir_path}/{rel_path}"
                            combined_comments[path_str] = device_time_ranges[""]

                        # Add all devices with time ranges to our collection
                        for device_id, time_info in device_time_ranges.items():
                            # If looking for specific devices, only add if in the list
                            if not dev_ids_list or device_id in dev_ids_list:
                                if device_id not in device_results:
                                    device_results[device_id] = {"time_info": time_info, "data_paths": {}}
                                device_results[device_id]["data_paths"][path_tuple] = metadata
                    elif dev_ids_list:
                        # Handle specific device files with priority logic
                        # dev_ids_list already contains normalized IDs
                        for dev_id in devices_in_file:
                            if dev_id not in dev_ids_list:
                                continue  # skip devices not in the list
                            # Collect all file candidates for each device, then prioritize
                            if dev_id not in all_device_candidates:
                                all_device_candidates[dev_id] = []
                            all_device_candidates[dev_id].append((path_tuple, metadata))
                    else:
                        # Handle case where we're discovering devices from files skipping generic markers
                        for dev_id in devices_in_file:

                            is_combined = dev_id in ["*", "i", "w", "p"]
                            if not is_combined:
                                # Collect all file candidates for each device, then prioritize
                                if dev_id not in all_device_candidates:
                                    all_device_candidates[dev_id] = []
                                all_device_candidates[dev_id].append((path_tuple, metadata))
            except Exception as e:
                logger.warning(f"Error parsing filename {file_name}: {e}")

    # For devices that have multiple file candidates, apply prioritization and process the best one first
    for dev_id, candidates in all_device_candidates.items():
        # Sort candidates by priority (lower averaging interval is better)
        def candidate_priority(item):
            path_tuple, metadata = item
            try:
                avg_interval = metadata["averaging_interval"]
            except KeyError:  # Files without averaging interval get the configured default value.
                avg_interval = config.default_text_file_averaging

            # Additional priority factors
            devices_in_file = metadata.get('devices', []) if metadata else []
            is_combined = any(char in devices_in_file for char in "*iwp")
            num_devices = len(devices_in_file) if devices_in_file else float('inf')
            return (avg_interval, is_combined, num_devices)

        # Sort candidates by priority (ascending - best first)
        sorted_candidates = sorted(candidates, key=candidate_priority)

        # Process candidates in priority order until we find one with valid time data
        # For split files, extract_time_info_from_text_file will automatically find
        # first and last files with same pattern, so we just need to call it once
        if sorted_candidates:
            path_tuple, metadata = sorted_candidates[0]
            dir_path, rel_path = path_tuple
            # Get averaging_interval from metadata if available, otherwise use default
            averaging_interval = metadata.get("averaging_interval")
            if averaging_interval is None:
                averaging_interval = config.default_text_file_averaging
            time_info = extract_time_info_from_text_file(dir_path, rel_path, averaging_interval)
            if time_info:  # Only keep if we get valid time info
                if dev_id not in device_results:
                    device_results[dev_id] = {"time_info": time_info, "data_paths": {}}
                # Store all candidate files as data paths
                for path_tuple, metadata in sorted_candidates:
                    device_results[dev_id]["data_paths"][path_tuple] = metadata
            else:
                # If the highest priority file has no valid time data, try to next one
                for path_tuple, metadata in sorted_candidates:
                    dir_path, rel_path = path_tuple
                    # Get averaging_interval from metadata if available, otherwise use default
                    averaging_interval = metadata.get("averaging_interval")
                    if averaging_interval is None:
                        averaging_interval = config.default_text_file_averaging
                    time_info = extract_time_info_from_text_file(dir_path, rel_path, averaging_interval)
                    if time_info:  # Only keep if we get valid time info
                        if dev_id not in device_results:
                            device_results[dev_id] = {"time_info": time_info, "data_paths": {}}
                        # Store all candidate files as data paths
                        for path_tuple, metadata in sorted_candidates:
                            device_results[dev_id]["data_paths"][path_tuple] = metadata
                        break  # Found a file with valid data, move to next device
                # If no file produced valid time data but we still want to track the files
                if dev_id not in device_results and sorted_candidates:
                    # Just track the file paths without time info
                    device_results[dev_id] = {"time_info": None, "data_paths": {}}
                    for path_tuple, metadata in sorted_candidates:
                        device_results[dev_id]["data_paths"][path_tuple] = metadata
