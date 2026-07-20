"""
File finding functions for TCM Metadata Processor.
"""

from pathlib import Path, PurePosixPath
from typing import List, Dict, Callable, Any, Mapping, Tuple, Union, Optional
import logging
import re
from itertools import chain
from collections import defaultdict

from meta_finder import config
from meta_finder.hdf5_processor import extract_devices_from_hdf5_groups, find_hdf5_files
from meta_finder.parse_data_file_name import (
    parse_filename_for_metadata,
    normalize_device_id,
    extract_device_ids_from_prefixed_name,
)
from meta_finder.parse_cruise_dir_name import parse_dated_dir
from . import utils_sys
from .logging_config import setup_logging

logger = setup_logging()


def find_cruise_directories(top_search_dirs: List[Union[Path, str]]) -> List[Path]:
    """Find all valid cruise directories matching the pattern YYMMDD_{cruise_name}.
    Only search 1 level deep
    Args:
        top_search_dirs: List of directories to search in (can be Path objects or strings)

    Returns:
        List of paths to valid cruise directories sorted by date
    """
    logger.info(f"Finding cruise directories in {top_search_dirs}")
    cruise_dirs = []
    excluded_count = 0

    for root_dir_input in top_search_dirs:
        root_dir = Path(root_dir_input)
        for dir_path in root_dir.glob(config.glob_dated_dir):
            if not dir_path.is_dir():
                continue
            # Check if directory matches any exclusion pattern
            if any(
                re.search(pattern, dir_path.name) for pattern in config.ptn_dir_exclude
            ):
                logger.debug(
                    f"Excluding directory '{dir_path.name}': matches exclusion pattern"
                )
                excluded_count += 1
                continue
            cruise_dirs.append(dir_path)

    if excluded_count > 0:
        logger.info(
            f"Excluded {excluded_count} directories based on exclusion patterns"
        )
    logger.info(f"Found {len(cruise_dirs)} cruise directories")
    # Sort by date extracted from directory name (YYMMDD pattern)
    return sorted(cruise_dirs, key=lambda x: x.name[:6])


def is_valid_device_dir(device_dir):
    """Check whether device directory has _raw or text_output subdirectory or archive."""
    if (device_dir / "_raw").is_dir() or (device_dir / "text_output").is_dir():
        return True
    stem_matches = {"_raw", "text_output"}
    for f in device_dir.iterdir():
        if f.stem in stem_matches and f.suffix.lower() in config.extensions_archive:
            return True
    return False


def find_device_dirs(
    cruise_dir: Optional[Path] = None, device_dirs_top: Optional[List[Path]] = None
) -> List[Path]:
    """Find device directories in cruise dir or device_dirs_top or/also use them itself as device directories. The condition for directory to be considered is a existence of "_raw" subdir

    Args:
        cruise_dir: Cruise directory to search in
        device_dirs_top: Optional top device directories instead of searching for them in `cruise_dir`

    Returns:
        List of device directories (as Path objects)
    """
    # Use device_dirs_top if provided, otherwise use cruise_dir
    if not device_dirs_top:
        if not cruise_dir.is_dir():
            logger.warning(
                f'Not a dir "{cruise_dir}" passed to find device directories inside'
            )
            return []

        # Search for top level device directories in the cruise directory - `{cruise_dir}\*device*`

        # Pattern (use with re.search to skip known date and other possible prefixes)
        device_dir_search_re = re.compile(config.ptn_device_dir_search, re.IGNORECASE)

        def _collect_subdevice_dirs(dated_parent: Path) -> None:
            """Search inside a dated directory for device-matching subdirectories and append them."""
            for subpath in dated_parent.iterdir():
                if not subpath.is_dir() or any(
                    re.search(pattern, subpath.name)
                    for pattern in config.ptn_dir_exclude
                ):
                    continue
                if device_dir_search_re.search(subpath.name) is not None:
                    device_dirs_top.append(subpath)
                    logger.debug(
                        f"Found device directory under dated subdir: {subpath}"
                    )

        device_dirs_top = []  # Possible directories to search for data
        for path in cruise_dir.iterdir():
            # Check if directory matches any exclusion pattern
            if not path.is_dir() or any(
                re.search(pattern, path.name) for pattern in config.ptn_dir_exclude
            ):
                continue
            is_dated = re.match(config.glob_dated_dir, path.name, re.IGNORECASE)
            if (m := device_dir_search_re.search(path.name)) is not None and m[
                "device"
            ]:
                device_dirs_top.append(path)
                # A dated directory (e.g. "130822@ADCP,t-chain,i") can match the device
                # pattern via broader_dev_list but not be a valid device dir itself — its
                # _raw/text_output may be nested inside a keyword subdirectory
                # (e.g. "inclinometer").  Always look inside dated dirs for sub-devices.
                if is_dated:
                    _collect_subdevice_dirs(path)
            elif is_dated:
                _collect_subdevice_dirs(path)
            elif path.name not in ["text_output", "_raw"] and not path.stem.startswith(
                "vsz"
            ):
                logger.debug(f"Skipping non-device directory: {path}")

        # If the cruise directory name contains device identifiers, add the cruise directory itself
        # using the same pattern (we can not exclude it in advance even if it has device subdirs)
        if device_dir_search_re.search(cruise_dir.name) is not None:
            # Add the cruise directory itself if it has device identifiers and no device subdirs exist
            device_dirs_top.append(cruise_dir)

    # Check if the above found directories has a "_raw" subdirectory - if so, treat it
    # as a device directory
    device_dirs = []

    # Search 1 level deeper also (i.e. for data in the {device_dirs_top}\{YYMMDD}*)
    # Look for directories that start with 6 digits followed by anything
    for device_dir in device_dirs_top:
        # Check if this device_dir has "_raw" subdir - if so, treat it as a device directory
        # even if they also contain date-like subdirectories
        if is_valid_device_dir(device_dir):
            device_dirs.append(device_dir)

        # Also include device directories that have dated subdirectories
        dated_device_dirs = []
        for path in device_dir.glob(config.glob_dated_dir):
            if path.is_dir() and is_valid_device_dir(path):
                dated_device_dirs.append(path)
        if dated_device_dirs:
            device_dirs.extend(sorted(dated_device_dirs, key=lambda x: x.name[:6]))

    # Deduplicate preserving order — a subdir may appear both as a direct device dir
    # and as a dated child of the cruise dir
    return list(dict.fromkeys(device_dirs))


def find_raw_directory_files(
    device_dir: Path,
) -> Dict[str, Dict[PurePosixPath, List[PurePosixPath]]]:
    """Find files in _raw directory and its subdirectories

    Args:
        device_dir: Directory to search for _raw subdirectory

    Returns:
        Dictionary mapping normalized device names to dictionaries of:
        - key: PurePosixPath of raw file directory or archive name
        - value: List of PurePosixPath of paths relative to file directory or archive name path
    """

    # Look for _raw directory
    raw_dir = device_dir / "_raw"
    if not raw_dir.exists():
        logger.debug(f"No _raw directory found in {device_dir}")
        return {}

    # Initialize defaultdict with nested defaultdict structure
    devices = defaultdict(lambda: defaultdict(list))
    # Process files in _raw directory and its subdirectories
    try:
        # Use rglob to recursively find all files in _raw directory
        for entry in raw_dir.rglob("*"):
            if not entry.is_file():
                continue
            # Only process supported file types
            if entry.suffix.lower() in (
                config.extensions_text | config.extensions_hdf5
            ):
                dev_id = extract_device_id_from_raw_file_name(entry.name)
                if entry.suffix.lower() in config.extensions_hdf5:
                    # HDF5 files usually contain device names internally only.
                    # save path, but entirely in key to fill relative
                    # path from their internal structure later
                    devices[dev_id or "*"][
                        PurePosixPath(entry.as_posix())
                    ]  # init empty list
                    # If they not encode it in file name, then we saved them under generic id
                elif dev_id:
                    # Group files by their parent directory to maintain directory structure
                    parent_dir = PurePosixPath(entry.parent.as_posix())
                    # The relative_path is just the filename (relative to parent_dir)
                    relative_path = PurePosixPath(entry.name)
                    devices[dev_id][parent_dir].append(relative_path)
            # Also check for files with archive extensions
            elif entry.suffix.lower() in config.extensions_archive:
                # Add the file paths within the archive
                archive_path = PurePosixPath(entry)
                try:
                    for item in utils_sys.gen_from_archive(archive_path):
                        if (
                            not item["is_folder"]
                            and (rel_path := item["rel_path"]).suffix.lower()
                            in config.extensions_text
                        ):
                            if dev_id := extract_device_id_from_raw_file_name(
                                rel_path.name
                            ):
                                devices[dev_id][archive_path].append(rel_path)
                except Exception:
                    logger.exception(f"Error listing contents of archive {entry}")
    except Exception:
        logger.exception(f"Error reading _raw directory {raw_dir}")

    # Convert defaultdict to regular dict for return
    if result := {dev_id: dict(paths_dict) for dev_id, paths_dict in devices.items()}:
        device_list = ", ".join(sorted(k for k in result.keys() if k != "*"))
        logger.info(
            f"Found %s in {device_dir.name}/_raw directory",
            " + ".join(
                (
                    ([f"{len(result)} device(s): {device_list}"] if device_list else [])
                    + (
                        [f"{len(result['*'])} general file(s)"]
                        if "*" in result.keys()
                        else []
                    )
                )
            ),
        )
        for dev_id, paths in result.items():
            logger.debug(f"Device {dev_id}: {len(paths)} paths")
    else:
        logger.info(f"Found no devices from {device_dir.name}/_raw directory")
    return result


def extract_device_id_from_raw_file_name(file_name):
    """
    Check whether file name matches device ID pattern allowed in "_raw" dir:
    - Single device only
    - Matches config.device_id with optional  If starts with device type [iw] or prefix [@#] following device type

    Args:
        file_name: Name of the file to extract device ID from
        default_device_type: Default device type when no type specified (default: 'i' for inclinometer)

    Return:
        normalized device ID

    Handles cases like:
    - #W1_130510.txt -> w1
    - 130510#W1_130510.txt -> w1
    - i1.txt -> i1
    - W1.txt -> w1
    - #1.txt -> i1 (# prefix removed, defaults to "i" device type)
    - 1.txt ->  i1
    """
    if m := re.match(r"^[@#_]+(\d+)(?:\.|$)", file_name):
        return normalize_device_id(m.group(1))
    dev_ids = extract_device_ids_from_prefixed_name(file_name, msg_what="file name ")
    return dev_ids[0] if len(dev_ids) == 1 else None

    # # Remove optional dated prefix and track if separator prefix was present
    # # Files with [@#_-] prefix should use Pattern 2 (defaults to inclinometer type)
    # prefix_match = re.match(rf"^({config.ptn_dated_prefix})?(?P<sep>[@#_-])*", file_name, re.IGNORECASE)
    # name = file_name[prefix_match.end():]

    # # Pattern 1: Files with device type (prefix optional) - use config pattern
    # # Only use Pattern 1 if file didn't have [@#_-] separator prefix
    # if (match := re.match(
    #     config.ptn_device_id_named_parts, name, re.IGNORECASE
    # )):
    #     groups = match.groupdict()
    #     if not (groups["type"] or groups["model"]) and prefix_match.group("sep") is None:
    #         return None
    #     device_id = "{type}{model}{number}".format_map(groups)
    #     # Normalize device ID to handle no type or incl -> i, etc.
    #     return normalize_device_id(device_id)
    # # Pattern didn't match - no valid device ID found
    # return None


def _resolve_device_ids_from_file_content(
    dir_path: Path, rel_path: PurePosixPath, generic_devices: List[str]
) -> List[str]:
    """Resolve actual device IDs from a file that contains generic device types by reading the file content.

    Args:
        dir_path: Directory containing the file
        rel_path: Relative path to the file
        generic_devices: List of generic device types found in the filename

    Returns:
        List of resolved device IDs
    """
    logger.debug(f"Resolving device IDs from file content: {dir_path / rel_path.name}")

    # Import necessary functions here to avoid circular imports
    from .data_proc_funcs import (
        read_file_lines_universal,
        _extract_device_ids_from_column_name,
    )
    from .parse_data_file_name import normalize_device_id

    try:
        # Read just the header line to identify devices from column names
        lines, _, read_error = read_file_lines_universal(
            dir_path, rel_path, max_lines=1
        )

        if not lines:
            logger.warning(
                f"No lines read from {dir_path / rel_path.name}"
                + (f". Error: {read_error}" if read_error else "")
            )
            return []

        # Parse the header line to identify device columns
        header = lines[0].strip()

        # Look for device identifiers in column names
        if "\t" in header or "," in header:  # Check if it's a structured format
            delimiter = "\t" if "\t" in header else ","
            columns = header.split(delimiter)

            resolved_devices = []
            for col in columns:
                resolved_devices.extend(_extract_device_ids_from_column_name(col))

            # Normalize all resolved device IDs
            normalized_devices = [
                normalize_device_id(dev_id) for dev_id in resolved_devices
            ]
            unique_devices = list(set(normalized_devices))
            logger.debug(f"Resolved unique devices from content: {unique_devices}")
            return unique_devices
        else:
            # If it's not a structured format, return empty list
            logger.debug(
                f"File {dir_path / rel_path.name} is not in a structured format"
            )
            return []

    except Exception as e:
        logger.error(
            f"Error resolving devices from {dir_path / rel_path.name}: {e}",
            exc_info=True,
        )
        return []


_GENERIC_DEVICE_TYPES = frozenset({"*", "i", "w", "p"})

# Regex to extract the dated timestamp prefix from a filename for ordering comparisons
_TS_PREFIX_RE = re.compile(r"^(\d{4,6}(?:\.\.\d{2,6})?(?:_\d+(?:-\d+(?:_\d+)?)?)?)")


class _PatternMinMaxTracker:
    """Track first and last files per dated-part pattern for a single device.

    For each unique mask pattern encountered, keeps only the entry with the
    smallest and largest timestamp prefix — all others are discarded inline.
    This avoids collecting all files then filtering in a second pass.

    Each entry also carries its ``base_dir`` (the directory or archive Path
    that contains the file) so that the caller can reconstruct the full
    (directory, rel_path) pair without a separate mapping that would be
    overwritten when the same device appears in multiple sources.
    """

    __slots__ = ("_buckets", "_total_seen")

    def __init__(self) -> None:
        # Each bucket: (compiled_pattern, ts_min, entry_min, ts_max, entry_max)
        # where each entry is (base_dir, rel_path, meta).
        self._buckets: list[tuple[re.Pattern, str, tuple, str, tuple]] = []
        self._total_seen: int = 0

    def add(self, base_dir: Path, rel_path: PurePosixPath, meta: dict) -> None:
        """Offer a file entry; the tracker keeps it only if it extends the min/max range."""
        from .data_proc_funcs import _build_dated_mask_pattern

        self._total_seen += 1
        name = rel_path.name
        ts_match = _TS_PREFIX_RE.match(name)
        ts = ts_match.group(1) if ts_match else name
        entry = (base_dir, rel_path, meta)

        for i, (pat, ts_min, entry_min, ts_max, entry_max) in enumerate(self._buckets):
            if pat.match(name):
                # Update min/max in-place
                if ts < ts_min:
                    ts_min, entry_min = ts, entry
                elif ts > ts_max:
                    ts_max, entry_max = ts, entry
                self._buckets[i] = (pat, ts_min, entry_min, ts_max, entry_max)
                return
        # No existing pattern matched — start a new bucket
        self._buckets.append((_build_dated_mask_pattern(name), ts, entry, ts, entry))

    def result(self) -> list[tuple[PurePosixPath, Path]]:
        """Return collected (base_dir, rel_path) pairs: at most 2 per pattern group."""
        out: list[tuple[PurePosixPath, Path]] = []
        for _pat, ts_min, entry_min, ts_max, entry_max in self._buckets:
            out.append((entry_min[0], entry_min[1]))
            if entry_max is not entry_min:
                out.append((entry_max[0], entry_max[1]))
        return out

    @property
    def total_seen(self) -> int:
        return self._total_seen


def _resolve_specific_devices(
    dir_path: Path,
    rel_path: PurePosixPath,
    devices: list[str],
) -> list[str]:
    """Resolve generic device types to specific IDs, returning only non-generic ones."""
    if any(d in _GENERIC_DEVICE_TYPES for d in devices):
        resolved = _resolve_device_ids_from_file_content(dir_path, rel_path, devices)
        return [d for d in resolved if d not in _GENERIC_DEVICE_TYPES]
    return [d for d in devices if d not in _GENERIC_DEVICE_TYPES]


def extract_devices_from_text_output(
    device_dir: Path,
) -> Dict[str, List[Tuple[Path, PurePosixPath]]]:
    """Extract device IDs from text_output file names, keeping only the earliest and latest
    file per device per dated-part pattern.

    Single-pass approach: as each file is encountered during the directory/archive walk,
    its device IDs are resolved and the file is offered to a per-device min/max tracker
    that keeps only the first and last entry per pattern group. No post-hoc filtering.

    Each tracker entry stores its own ``base_dir`` (the directory or archive that
    contains the file), so devices that span both loose files and archives are
    handled correctly without a separate dir_map that would be overwritten.

    Args:
        device_dir: Directory containing text_output subdirectory.

    Returns:
        Dictionary mapping normalized device IDs to lists of (directory_path, file_path) tuples.
        Each device has at most 2 entries per pattern group.
    """
    logger.debug(f"Extracting devices from text_output files in: {device_dir}")
    # Per-device min/max trackers — base_dir is stored inside each entry
    trackers: Dict[str, _PatternMinMaxTracker] = {}

    text_output_dirs = list(device_dir.glob("text_output*"))
    logger.debug(
        f"Found {len(text_output_dirs)} text_output directories/archives: {text_output_dirs}"
    )

    for text_output_dir in text_output_dirs:
        logger.debug(f"Processing text_output directory/archive: {text_output_dir}")

        # Determine directories and archives to scan
        if (
            text_output_dir.is_file()
            and text_output_dir.suffix.lower() in config.extensions_archive
        ):
            directories: list[Path] = []
            archives = [text_output_dir]
        elif text_output_dir.is_dir():
            directories = [text_output_dir]
            # Find nested archives using targeted globs per extension (avoids rglob('*'))
            archives = [
                item
                for ext in sorted(config.extensions_archive)
                for item in text_output_dir.rglob(f"*{ext}")
                if item.is_file()
            ]
        else:
            continue

        def _process_file(base_dir: Path, rel_path: PurePosixPath) -> None:
            """Parse metadata, resolve devices, and feed to per-device tracker."""
            if not (filename_meta := parse_path_for_metadata(rel_path)):
                return
            if not filename_meta.get("devices"):
                return
            specific = _resolve_specific_devices(
                base_dir, rel_path, filename_meta["devices"]
            )
            for dev in specific:
                trackers.setdefault(dev, _PatternMinMaxTracker()).add(
                    base_dir, rel_path, filename_meta
                )

        # Walk directories in a single pass
        for subdir in sorted(directories):
            try:
                for entry in subdir.rglob("*"):
                    if (
                        entry.is_file()
                        and entry.suffix.lower() in config.extensions_text
                    ):
                        _process_file(
                            text_output_dir,
                            PurePosixPath(
                                entry.relative_to(text_output_dir).as_posix()
                            ),
                        )
            except Exception:
                logger.exception(f"Error reading directory {subdir}")

        # Walk archives in a single pass
        for archive_path in sorted(archives):
            for item in utils_sys.gen_from_archive(archive_path):
                if (
                    not item["is_folder"]
                    and item["rel_path"].suffix.lower() in config.extensions_text
                ):
                    _process_file(archive_path, item["rel_path"])

    # Collect results from trackers — each entry carries its own base_dir
    dev_files: Dict[str, List[Tuple[Path, PurePosixPath]]] = {}
    for dev, tracker in trackers.items():
        entries = tracker.result()
        if entries:
            dev_files[dev] = [(bd, rp) for bd, rp in entries]
            logger.debug(
                f"Device {dev}: {tracker.total_seen} files scanned, {len(entries)} kept "
                "(first/last per pattern)"
            )

    if dev_files:
        device_info = ", ".join(
            f"{device}: {len(files)}" for device, files in dev_files.items()
        )
        logger.info(f"  Device files found from text_output sources: {device_info}")
    return dev_files


def parse_path_for_metadata(rel_path: PurePosixPath) -> Dict[str, Any]:
    """Parse file path for metadata, resolving device patterns from path parts.

    Extracts metadata from filename and replaces generic device patterns (like "*")
    with specific device IDs found in the relative path parts, scanning from last
    to first for priority.

    Args:
        rel_path: Relative path to the file

    Returns:
        Metadata dictionary with resolved device IDs, or None if no valid datetime
    """
    filename_meta = parse_filename_for_metadata(rel_path.name)
    # Only include files with valid datetime metadata
    if filename_meta:
        if "datetime" in filename_meta:
            # Continue resolving device if it is not an exact device
            has_generic_device = any(
                device in ["*", "i", "w", "p"] for device in filename_meta["devices"]
            )
            if has_generic_device and rel_path.parent != PurePosixPath("."):
                if resolved_device := _find_specific_device_in_path_parts(rel_path):
                    logger.debug(
                        f"Found specific device from path parts: {resolved_device}"
                    )
                    # Replace generic device patterns with the resolved specific device
                    filename_meta["devices"] = [
                        resolved_device if device in ["*", "i", "w", "p"] else device
                        for device in filename_meta["devices"]
                    ]
            return filename_meta
        else:
            logger.debug(f"Skipping file {rel_path.name} - no valid datetime found")
    return None


def validate_dev_ids(
    pre_extracted_devices, dev_files, rel_path, filename_meta, dir_path
):
    """not used"""
    filename_devices = [
        dev for dev in filename_meta["devices"] if dev not in ["*", "i", "w", "p"]
    ]
    # Only extract subdirectory devices if we need to validate
    # Use pre-extracted devices if file is directly in the directory/archive
    if rel_path.parent == PurePosixPath("."):
        subdir_devices = pre_extracted_devices
    else:
        # File is in a subdirectory, extract from that subdirectory
        subdir_devices = _extract_device_id_from_subdirectory_name(
            Path(dir_path), rel_path
        )

    if subdir_devices and filename_devices:
        # Both sources have device information - verify they match
        # Only warn if filename devices are not a subset of subdirectory devices
        # This handles cases where subdirectory contains multiple devices (e.g., @i3,5,9,w1-6)
        # and filename contains one of them (e.g., i5)
        if not set(filename_devices).issubset(set(subdir_devices)):
            logger.warning(
                f"Device mismatch for file {rel_path.name}: "
                f"filename contains {filename_devices}, "
                f"subdirectory {dir_path.name} contains {subdir_devices}. "
                f"Using devices from filename."
            )
        else:
            logger.debug(
                f"Device validation passed for {rel_path.name}: "
                f"filename and subdirectory both contain {filename_devices}"
            )

            # Add all specific devices from the file
            # Note: metadata['devices'] from parse_filename_for_metadata are already normalized
    for device in filename_devices:
        if device not in ["*", "i", "w", "p"]:  # Skip generic device types
            # Add file to the device's list
            if device not in dev_files:
                dev_files[device] = []
                # Store as tuple (Path to parent directory/archive, PurePosixPath of relative file path)
            dev_files[device].append((Path(dir_path), rel_path))


def _extract_device_id_from_directory_name(dir_name: str) -> List[str]:
    """
    Extract all device IDs from directory/archive name

    This function uses a dedicated pattern for directory/archive names, which handles
    complex device lists without requiring fake filenames.

    Args:
        dir_name: directory or archive name

    Returns:
        List of normalized device IDs extracted from directory name, or empty list if not found
    """
    # Try to match @device_list pattern first using the dedicated directory name pattern

    return extract_device_ids_from_prefixed_name(dir_name, msg_what="directory name ")

    # at_prefix_match = re.search(
    #     f"@(?P<devices>{config.ptn_devices_groups_part})|(?P<type>{config.ptn_device_type})",
    #     dir_name, re.IGNORECASE
    # )

    # # Check if we matched the devices groups
    # if at_prefix_match and (device_list_str := at_prefix_match['devices']):
    #     devices = [
    #         normalize_device_id(d)
    #         for group in split_top_level(device_list_str)
    #         for d in parse_device_group(group)
    #     ]
    #     # Filter out generic device types
    #     if (specific_devices := [d for d in devices if d not in ['*', 'i', 'w', 'p']]):
    #         return specific_devices

    # # Fallback: Try to find individual device IDs using ptn_device_id
    # # This handles cases like "230508_inclinometer@i03" where @i03 is embedded
    # if (matches := list(re.finditer(config.ptn_device_id, dir_name, re.IGNORECASE))):
    #     device_ids = [match.group(0) for match in matches]
    #     normalized_devices = [normalize_device_id(device_id) for device_id in device_ids]
    #     # Remove duplicates while preserving order
    #     seen = set()
    #     unique_devices = []
    #     for device in normalized_devices:
    #         if device not in seen:
    #             seen.add(device)
    #             unique_devices.append(device)
    #     return unique_devices
    # return []


def _find_specific_device_in_path_parts(rel_path: PurePosixPath) -> Optional[str]:
    """Scan rel_path parts in reverse order to find first specific device_id pattern.

    When parse_filename_for_metadata returns general device pattern ["*"], this function
    scans the relative path parts from last to first to find a specific device identifier.
    This prioritizes the deepest subdirectory over parent directories.

    If the deepest part that contains devices has multiple specific devices, the file's
    device is ambiguous — returns None without falling back to shallower parts.  All
    path parts are still scanned so that a warning can be emitted when a shallower part
    lists devices not contained in the deeper part's set.

    Args:
        rel_path: Relative path from dir_path to the file

    Returns:
        Single specific device ID if unambiguously found, None otherwise.
        Only returns specific devices (excludes generic types like '*', 'i', 'w', 'p').
    """
    # Collect devices from every path part (deepest first) to enable cross-level warnings.
    parts_with_devices: list[tuple[str, list[str]]] = []
    for part in reversed(rel_path.parent.parts):
        devices = _extract_device_id_from_directory_name(part)
        specific = [d for d in devices if d not in ["*", "i", "w", "p"]]
        if specific:
            parts_with_devices.append((part, specific))

    if not parts_with_devices:
        return None

    # The first entry is the deepest part that contains any devices.
    deepest_part, deepest_devices = parts_with_devices[0]

    if len(deepest_devices) == 1:
        logger.debug(
            f"Found specific device {deepest_devices[0]} in path part: {deepest_part}"
        )
        return deepest_devices[0]

    # Deepest part is ambiguous — warn if shallower parts mention devices not in the
    # deeper set, which may indicate a misconfiguration.
    deepest_set = set(deepest_devices)
    for shallower_part, shallower_devices in parts_with_devices[1:]:
        unexpected = [d for d in shallower_devices if d not in deepest_set]
        if unexpected:
            logger.warning(
                "Shallower path part '%s' lists devices %s not in deeper part '%s' devices %s",
                shallower_part,
                unexpected,
                deepest_part,
                deepest_devices,
            )

    logger.debug(
        f"Ambiguous devices {deepest_devices} in deepest path part: {deepest_part}, returning None",
    )
    return None


def _extract_device_id_from_subdirectory_name(
    dir_path: Path, rel_path: PurePosixPath
) -> List[str]:
    """
    Extract device ID from subdirectory name as a fallback when filename doesn't contain device information.

    This function determines which directory/archive name to extract devices from based on the
    file's relative path, then delegates to _extract_device_id_from_directory_name()
    for the actual pattern matching. Individual file processing still needs this function because
    files may be in different subdirectories (e.g., text_output/130510/i03/file.txt
    where i03 is the device directory), requiring the rel_path parameter to
    determine the correct parent directory.

    Args:
        dir_path: Full path to the directory containing the file
        rel_path: Relative path to the file (from the text_output directory)

    Returns:
        List of normalized device IDs extracted from subdirectory name, or empty list if not found
    """
    logger.debug(
        f"Attempting to extract device ID from subdirectory name for: {rel_path}"
    )

    # Determine the directory/archive name to extract devices from
    # This handles cases like text_output/130510/i03/file.txt where i03 is the device directory
    # rel_path.parent gives us the parent directory relative to dir_path
    if rel_path.parent != PurePosixPath("."):
        # File is in a subdirectory, extract from that subdirectory
        subdir_path = dir_path / rel_path.parent
        logger.debug(f"Subdirectory name: {rel_path.parent.name}")
    else:
        # File is directly in dir_path, extract from dir_path itself
        subdir_path = dir_path
        logger.debug(f"Using parent directory name: {dir_path.name}")

    # Delegate to core extraction function for pattern matching
    devices = _extract_device_id_from_directory_name(subdir_path.name)

    if devices:
        logger.debug(f"Extracted devices from subdirectory: {devices}")
    else:
        logger.debug(f"No device ID found in subdirectory name for: {rel_path}")

    return devices


def discover_datafiles_for_all_dev_in_dev_dir(
    device_dir: Path,
) -> Dict[str, List[Tuple[Path, PurePosixPath]]]:
    """Discover all devices from text_output files or _raw directory or HDF5 files.

    Args:
        device_dir: Directory to search for devices (should be a standard device directory)

    Returns:
        Dictionary mapping device IDs to lists of (directory_path, file_path) tuples
        where directory_path is Path to the parent directory/archive and file_path is PurePosixPath
        relative to the parent directory/archive
    """
    logger.info(f"Discovering all devices in: {device_dir}")

    # Clear the cache for split files processing at the beginning of each device directory
    # This ensures that cache doesn't persist across different directories
    from .data_proc_funcs import clear_find_dated_files_cache

    clear_find_dated_files_cache()

    # First try to extract devices from text_output files
    dev_files = extract_devices_from_text_output(device_dir)
    # Check raw files - always check to supplement text_output or as fallback
    raw_devices_dict = find_raw_directory_files(device_dir)

    # Log appropriate message based on what we found
    if not dev_files and not raw_devices_dict:
        logger.warning("No devices found in text_output or _raw directory")
    else:
        logger.info(
            f"Devices from text output: {list(dev_files.keys())}. "
            f"Supplementing with _raw: {list(raw_devices_dict.keys())}"
        )

        # Convert the raw devices dictionary format to match our expected format
        for device_id, paths_dict in raw_devices_dict.items():
            if device_id in ["*", "i", "w", "p"]:  # Skip generic device types
                continue
            if device_id not in dev_files:
                dev_files[device_id] = []
            for dir_path, file_paths in paths_dict.items():
                for file_path in file_paths:
                    # Store as tuple (Path to parent directory/archive, PurePosixPath of relative file path)
                    dev_files[device_id].append((Path(dir_path), file_path))

    # If still no devices found and HDF5 fallback is enabled, try HDF5 files
    # Also try HDF5 if we have devices from _raw to get additional metadata
    if not dev_files or (dev_files and config.extract_hdf5_times):
        if not dev_files:
            logger.info("No devices found in text_output or _raw, trying HDF5 files")
        else:
            logger.info("Also checking HDF5 files for additional device metadata")
        h5_files = find_hdf5_files(device_dir)
        logger.debug(f"Found HDF5 files: {h5_files}")

        # Try HDF5 files in priority order: proc_noAvg, proc, raw
        for h5_type in ["proc_noAvg", "proc", "raw"]:
            for h5_file_path in h5_files[h5_type]:
                logger.debug(f"Trying HDF5 file: {h5_file_path} for type: {h5_type}")
                h5_devices = extract_devices_from_hdf5_groups(h5_file_path)
                if h5_devices:
                    # If we already have devices from _raw, merge them with HDF5 devices
                    # Otherwise, create device mapping with HDF5 file as the associated file
                    if dev_files:
                        # Merge HDF5 devices with existing dev_files
                        for device, groups in h5_devices.items():
                            if device not in [
                                "*",
                                "i",
                                "w",
                                "p",
                            ]:  # Skip generic device types
                                if device not in dev_files:
                                    # New device from HDF5, add it with HDF5 file as the source
                                    dev_files[device] = [
                                        (
                                            Path(h5_file_path.parent),
                                            PurePosixPath(h5_file_path.name),
                                        )
                                    ]
                                else:
                                    # Device exists from _raw, keep _raw files (they have more detailed data)
                                    logger.debug(
                                        f"Device {device} found in both _raw and HDF5, using _raw files"
                                    )
                        logger.info(
                            f"Found devices from HDF5 file {h5_file_path}: {list(h5_devices.keys())}"
                        )
                    else:
                        # Create device mapping with HDF5 file as the associated file
                        dev_files = {}
                        for device, groups in h5_devices.items():
                            if device not in [
                                "*",
                                "i",
                                "w",
                                "p",
                            ]:  # Skip generic device types
                                # For HDF5 files, we store the file path as the directory path and use the file name as the "relative" path
                                dev_files[device] = [
                                    (
                                        Path(h5_file_path.parent),
                                        PurePosixPath(h5_file_path.name),
                                    )
                                ]
                        logger.info(
                            f"Found devices from HDF5 file {h5_file_path}: {list(dev_files.keys())}"
                        )
                    break
            if dev_files:
                break

    # Log the number of files for each device
    if dev_files:
        device_info = ", ".join(
            [f"{device}: {len(files)}" for device, files in dev_files.items()]
        )
        logger.info(
            f"  Total data files for devices (of all types sufficient for next proc.): {device_info}"
        )
        device_info = ",\n".join(
            [f"{device}: {str(files)}" for device, files in dev_files.items()]
        )
        logger.debug(f"  {device_info}")
    else:
        # Warn if no data was found in this device directory
        logger.warning(
            f"No data found in device directory '{device_dir.name}'. "
            f"Expected to find data in text_output, _raw, or HDF5 files."
        )

    return dev_files


def discover_device_dirs(
    top_search_dirs: List[Path],
    input_dirs: List[Path] = None,
) -> Mapping[Path, List[Path]]:
    """Scan cruise directories for inclinometer/wavegauge subdirectories.

    Automatically detects if input_dirs are device directories (not cruise directories).
    If no device directories are found using standard cruise search, treats input_dirs
    as device directories directly.

    Args:
        top_search_dirs: List of directories to search in
        input_dirs: Optional list of specific directories to process (overrides top_search_dirs)

    Returns:
        Mapping of cruise directories to their device directories
    """
    # Use provided cruise_dirs if available, otherwise search for cruise directories
    if input_dirs:
        logger.debug(f"Using provided input directories: {input_dirs}")
        found_cruise_dirs = [Path(in_dir) for in_dir in input_dirs]
    else:
        logger.debug(f"Searching for cruise directories in: {top_search_dirs}")
        found_cruise_dirs = find_cruise_directories(top_search_dirs)
    logger.debug(f"Found {len(found_cruise_dirs)} cruise directories")

    cruise_devices = {}
    for cruise_dir in found_cruise_dirs:
        if device_dirs := find_device_dirs(cruise_dir):
            logger.info(
                f"Found {len(device_dirs)} device directories in {cruise_dir.name}"
            )
            cruise_devices[cruise_dir] = device_dirs
        else:
            logger.debug(f"No device directories found in {cruise_dir.name}")

    # If no device directories found using standard cruise search and input_dirs was provided,
    # try treating input_dirs as device directories directly
    if not cruise_devices and input_dirs:
        logger.info(
            "No device directories found using standard cruise search. "
            "Attempting to use input directories as device directories."
        )
        # Use find_device_dirs with device_dirs_top parameter to check if input_dirs
        # are actually device directories
        for input_dir in input_dirs:
            input_path = Path(input_dir)
            # Pass input_dir as both cruise_dir (for logging) and device_dirs_top
            # find_device_dirs will validate if it has _raw or text_output subdirectories
            if device_dirs := find_device_dirs(device_dirs_top=[input_path]):
                logger.info(
                    f"Input directory '{input_path.name}' contains device data "
                    f"(_raw or text_output subdirectories found)"
                )
                # Search cruise directory among parents, or the directory itself if at root
                cruise_parent = (
                    "9999-01-01"  # use some fake if valid cruise dir will not be found
                )
                for cruise_parent in [input_path] + list(input_path.parents):
                    try:
                        _ = parse_dated_dir(cruise_parent.name)
                        break
                    except ValueError:
                        continue
                if cruise_parent not in cruise_devices:
                    cruise_devices[cruise_parent] = []
                cruise_devices[cruise_parent].extend(device_dirs)
            else:
                logger.debug(
                    f"Input directory '{input_path.name}' does not contain device data "
                    f"(no _raw or text_output subdirectories found)"
                )

    logger.debug(
        f"Scan complete. Found {len(cruise_devices)} cruises with device directories"
    )
    return cruise_devices


def _gpx_filename_contains_device_identifiers(file_name_stem: str) -> bool:
    """
    Check if GPX filename stem matches the configured search pattern.

    If ptn_search_gpx is empty, all GPX files are included.
    If ptn_search_gpx is set, only files matching the pattern are included.

    Args:
        file_name_stem: Filename without extension (e.g., "track_i01" or "navigation")

    Returns:
        True if filename should be included, False otherwise
    """
    # If pattern is empty, include all GPX files
    if not config.ptn_search_gpx:
        return True

    # Check if filename contains any separators (digits, +, -, @, _, #)
    if not re.search(rf"\d*{config.ptn_device_dir_sep}", file_name_stem):
        # No separators found - include this file
        return True

    # Remove any leading digits before checking for device identifiers
    # This handles cases like "230507_track_i01" where "230507" is a date prefix
    filename_without_digits = re.sub(r"^\d+", "", file_name_stem)

    # Check if filename matches the configured pattern
    if re.search(config.ptn_search_gpx, filename_without_digits, re.IGNORECASE):
        return True

    # Has separators but doesn't match pattern - exclude this file
    logger.debug(
        f"Excluding GPX file '{file_name_stem}': doesn't match GPX search pattern"
    )
    return False


def find_navigation_files(search_path: Path) -> List[Path]:
    """Finds all .gpx navigation files within a given search path.

    Filters out GPX files that have separators in their filename stem but don't
    contain matching device identifiers between those separators.
    """
    logger.debug(f"Finding navigation files in {search_path}")

    files = []
    for dir_path in chain(search_path.glob("*navigation*"), search_path.glob("*map*")):
        if dir_path.is_dir():
            for file_path in dir_path.glob(f"*.gpx"):
                # Check if file should be included based on device identifier check
                if _gpx_filename_contains_device_identifiers(file_path.stem):
                    files.append(file_path)

    # Use set to get unique file paths, then sort for consistent order
    unique_files = sorted(list(set(files)))
    logger.debug(f"Found {len(unique_files)} unique navigation files in {search_path}")
    return unique_files
