"""
System utility functions for working with directories and files including archives in Windows
"""

import zipfile
from io import TextIOWrapper
import subprocess
import tempfile
import shutil
from typing import Optional, List, Dict, Any, Generator, Tuple
from pathlib import Path, PurePosixPath

from . import config
from .logging_config import setup_logging
logger = setup_logging()

# Try to import libarchive for better archive handling
try:
    import libarchive as la
    HAS_LIBARCHIVE = True
    logger.debug("libarchive-c is available for archive handling")
except ImportError:
    HAS_LIBARCHIVE = False
    logger.debug("libarchive-c not available, will use fallback methods")

try:  # Optional library for 7z archives
    import py7zr
except ImportError:
    py7zr = None


def _read_from_libarchive(
    archive_path: Path,
    target: str,
    n_head: Optional[int] = None,
    skip_header: int = 0
) -> Tuple[List[str], Optional[str]]:
    """
    Read lines from a file inside an archive using libarchive.

    Args:
        archive_path: Path to archive file
        target: Internal file path within the archive
        n_head: Number of lines to read from the start (None for all)
        skip_header: Number of header lines to skip

    Returns:
        Tuple of (list of lines, last line)
    """
    head_lines = []
    last_line = None
    buf = b""
    lines_read = 0

    with la.file_reader(str(archive_path)) as entries:
        for entry in entries:
            if entry.pathname != target:
                continue

            for block in entry.get_blocks():
                buf += block
                *lines, buf = buf.split(b"\n")

                for line in lines:
                    line_s = line.rstrip(b"\r").decode("utf-8", errors="ignore")

                    if skip_header > 0:
                        skip_header -= 1
                        continue

                    if n_head is None or len(head_lines) < n_head:
                        head_lines.append(line_s)

                    last_line = line_s
                    lines_read += 1

            if buf:
                line_s = buf.rstrip(b"\r").decode("utf-8", errors="ignore")
                if skip_header <= 0:
                    if n_head is None or len(head_lines) < n_head:
                        head_lines.append(line_s)
                    last_line = line_s
                    lines_read += 1

            break

    return head_lines, last_line


def _list_libarchive_contents(archive_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    List contents of an archive using libarchive.

    Args:
        archive_path: Path to the archive file

    Yields:
        Dictionary with 'rel_path' and 'is_folder' keys
    """
    with la.file_reader(str(archive_path)) as entries:
        for entry in entries:
            yield {
                "rel_path": PurePosixPath(entry.pathname),
                "is_folder": bool(entry.pathname.endswith("/"))
            }


def _read_sample_lines_from_file(
    dir_archive: Path,
    rel_path: PurePosixPath,
    num_lines: int,
    skip_header: int = 0
) -> List[str]:
    """
    Read a sample of lines from a file (directory or archive).

    Args:
        dir_archive: Path to archive or directory containing the file
        rel_path: Relative path to the file within archive/directory
        num_lines: Number of lines to read
        skip_header: Number of header lines to skip (default: 0)

    Returns:
        List of lines read from the file
    """
    lines = []

    if Path(dir_archive).is_dir():
        # Regular directory - read file directly
        file_path = dir_archive / rel_path
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            # Skip header lines
            for _ in range(skip_header):
                f.readline()

            # Read requested number of lines
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
    else:
        # Archive file - use read_archive_file_lines
        lines, _ = read_archive_file_lines(
            dir_archive, rel_path,
            max_lines=num_lines + skip_header
        )
        # Remove header lines if skip_header > 0
        if skip_header > 0 and len(lines) > skip_header:
            lines = lines[skip_header:]

    return lines


def get_sampling_dt(
        lines: List[str], parse_datetime_func, is_raw, sample_shift: int, msg_add=""
    ) -> float|None:
    """
    Calculate seconds per line
    """
    if not lines or len(lines) < 2:
        logger.debug(f"Not enough lines to calculate time interval{msg_add}")
        return None

    # Get first and second sample lines
    line_sample1 = lines[0]
    idx_line_sample2 = min(sample_shift, len(lines) - 1)
    line_sample2 = lines[idx_line_sample2]

    # Parse timestamps from both lines
    sep = "\t" if "\t" in line_sample1 else "," if "," in line_sample1 else " "
    time_sample1 = parse_datetime_func(line_sample1, is_raw=is_raw, sep=sep)
    time_sample2 = parse_datetime_func(line_sample2, is_raw=is_raw, sep=sep)

    if not time_sample1 and time_sample2 and idx_line_sample2 < len(lines) - 1:
        # Previous 1st sample was bad, use 2nd sample instead and sample more
        time_sample1 = time_sample2
        idx_line_sample1 = idx_line_sample2
        idx_line_sample2 = min(idx_line_sample1 + sample_shift, len(lines) - 1)
        line_sample2 = lines[idx_line_sample2]
        time_sample2 = parse_datetime_func(line_sample2, is_raw=is_raw, sep=sep)
    else:
        idx_line_sample1 = 0
    if not time_sample1 or not time_sample2:
        logger.debug(
            f"Failed to parse timestamps for time interval calculation{msg_add}"
        )
        return None

    # Calculate time difference in seconds between line 1 and line 20
    time_diff_seconds = (time_sample2 - time_sample1).total_seconds()

    if time_diff_seconds <= 0:
        logger.debug(f"Invalid time difference ({time_diff_seconds}s){msg_add}")
        return None

    lines_between = idx_line_sample2 - idx_line_sample1
    if lines_between <= 0:
        lines_between = 1  # Avoid division by zero

    return time_diff_seconds / lines_between


def calculate_lines_for_burst_time(
    dir_archive: Path,
    rel_path: PurePosixPath,
    max_burst_time_detection: int,
    parse_datetime_func,
    skip_header: int = 1,
    is_raw: bool = False,
    seconds_per_line: Optional[int|float] = None,
) -> Optional[int]:
    """
    Calculate the number of lines needed to read to cover max_burst_time_detection seconds.

    Reads lines 1 and 20 (or fewer if file is shorter) to extract timestamps and
    calculate the time interval between lines. Then calculates how many lines are
    needed to cover the configured max_burst_time_detection time.

    Args:
        dir_archive: Path to archive or directory containing the file
        rel_path: Relative path to the file within archive/directory
        max_burst_time_detection: Maximum time in seconds for burst detection
        parse_datetime_func: Function to parse datetime from a line (parse_datetime_from_row)
        skip_header: Number of header lines to skip
        is_raw: Whether the file is in raw inclinometer format (default: False)

    Returns:
        Number of lines to read for burst detection, or None if calculation fails
    """
    sample_shift = 20
    try:
        if not seconds_per_line:
            # Use common helper to read sample lines
            lines = _read_sample_lines_from_file(
                dir_archive, rel_path,
                num_lines=sample_shift * 2 + skip_header,
                skip_header=skip_header
            )
            if not (
                seconds_per_line := get_sampling_dt(
                    lines, parse_datetime_func, is_raw, sample_shift,
                    msg_add=f" in {dir_archive.name}/{rel_path}"
                )
            ):
                return None

        # Calculate how many lines needed to cover max_burst_time_detection
        lines_needed = int(max_burst_time_detection / seconds_per_line)

        # Ensure we read at least sample_shift*2 lines for accurate burst detection
        lines_needed = max(lines_needed, sample_shift*2)

        logger.debug(
            f"Time-based burst calculation for {dir_archive.name}/{rel_path}: "
            f"interval={seconds_per_line:.2f}s/line, need {lines_needed} lines for "
            f"{max_burst_time_detection}s burst detection"
        )

        return lines_needed

    except Exception as e:
        logger.debug(
            f"Error calculating lines for burst time in {dir_archive.name}/{rel_path}: {e}"
        )
        return None


def read_first_last_lines(archive_path: Path, inner_file: str | PurePosixPath, skip_header: int = 0):
    """
    Reads the first and last lines of a file inside a ZIP or 7z archive.
    - Tries libarchive first for better performance
    - Falls back to zipfile/py7zr if libarchive unavailable
    Parameters:
        archive_path: Path - path to archive (.zip or .7z)
        inner_file: name of file inside archive
        skip_header: int - number of lines to skip from start before returning first line

    Returns (first_line, last_line)

    # Example
    # --- \u043f\u0440\u0438\u043c\u0435\u0440 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u044f ---
    fline, lline = read_first_last_lines(Path("archive.zip"), "file.txt", skip_header=2)
    print("first (skipped 2 lines):", fline)
    print("last:", lline)

    fline7, lline7 = read_first_last_lines(Path("archive.7z"), "file.txt", skip_header=2)
    print("first (skipped 2 lines):", fline7)
    print("last:", lline7)
    """
    target = str(inner_file)

    if HAS_LIBARCHIVE:
        try:
            lines, last_line = _read_from_libarchive(archive_path, target, n_head=1, skip_header=skip_header)
            first = lines[0] if lines else None
            return first, last_line
        except Exception as e:
            logger.debug(f"libarchive reading failed for {archive_path}: {e}, falling back to standard method")

    chunk_size = 10 * 1024 * 1024  # 10 MB chunks

    # --- ZIP case ---
    if archive_path.suffix.lower() in config.extensions_archive:
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path) as zf:
                with zf.open(target) as f:
                    reader = TextIOWrapper(f, encoding="utf-8", errors="ignore")
                    first = None
                    for _ in range(skip_header + 1):
                        first = reader.readline().rstrip("\n")
                    f.seek(0, 2)  # \u043a\u043e\u043d\u0435\u0446 \u0444\u0430\u0439\u043b\u0430
                    # \u0447\u0438\u0442\u0430\u0435\u043c \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 10 MB (\u0438\u043b\u0438 \u043c\u0435\u043d\u044c\u0448\u0435)
                    size = f.tell()
                    f.seek(max(0, size - chunk_size), 0)
                    chunk = f.read().decode("utf-8", errors="ignore")
                    last = chunk.strip().splitlines()[-1] if chunk else first
            return first, last

        # --- 7z case ---
        elif archive_path.suffix.lower() == ".7z":
            if py7zr is None:
                raise ImportError(f"py7zr was not found to extract {archive_path}")

            import tempfile
            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                    # Extract only the specific file to the temporary directory
                    archive.extract(path=temp_dir, targets=[target])

                    # Read the extracted file
                    extracted_file_path = temp_dir_path / inner_file
                    if not extracted_file_path.exists():
                        raise FileNotFoundError(f"File {inner_file} not found in archive")

                    with open(extracted_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()

                    # Split into lines
                    lines = file_content.splitlines()

                    # Get first line (with skip_header)
                    first = None
                    if len(lines) > skip_header:
                        first = lines[skip_header]

                    # Get last line
                    last = lines[-1] if lines else ""

                    return first, last

    else:
        raise ValueError(f"Unsupported archive format, only {config.extensions_archive} supported")


def gen_from_archive(archive_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Recursively lists files/folders in ZIP or 7z archive.
    Returns a flat list of all items in the archive:
    {
        'rel_path': PurePosixPath path within archive
        'is_folder': bool,
    }
    """
    if HAS_LIBARCHIVE:
        try:
            yield from _list_libarchive_contents(archive_path)
            return
        except Exception as e:
            logger.debug(f"libarchive listing failed for {archive_path}: {e}, falling back to standard method")

    archive_suffix = archive_path.suffix.lower()
    if archive_suffix in config.extensions_archive:
        try:
            if archive_suffix == ".zip":
                with zipfile.ZipFile(archive_path) as zf:
                    for item in zf.namelist():
                        yield {
                            "rel_path": PurePosixPath(item),
                            "is_folder": bool(item.endswith("/"))
                        }
            elif archive_suffix == ".7z":
                if py7zr is None:
                    raise ImportError(f"py7zr was not found to extract {archive_path}")
                with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                    for item in archive.getnames():
                        yield {"rel_path": PurePosixPath(item), "is_folder": bool(item.endswith("/"))}

        except Exception as e:
            logger.error(f"Error listing archive contents {archive_path}: {e}", exc_info=True)
            return []
    else:
        raise ValueError(f"Unsupported archive format, only {config.extensions_archive} supported")


def list_archive_recursive(archive_path: Path) -> List[Dict[str, Any]]:
    """
    Recursively lists files/folders in ZIP or 7z archive.
    Returns a flat list of all items in the archive:
    {
        'name': str,  # name of file/dir within archive
        'rel_path': PurePosixPath path within archive
        'is_folder': bool,
        'children': []  # only for folders (but we flatten the structure)
    }
    """
    def build(node, path="", result=None):
        if result is None:
            result = []
        for k, v in node.items():
            entry_path = f"{path}/{k}" if path else k
            entry = {"rel_path": PurePosixPath(entry_path), "name": k, "is_folder": bool(v), "children": []}
            result.append(entry)
            if v:
                build(v, entry_path, result)
        return result

    def zip_walk(zf: zipfile.ZipFile):
        tree = {}
        for name in zf.namelist():
            parts = name.rstrip("/").split("/")
            node = tree
            for part in parts[:-1]:
                # Get or create directory node (don't overwrite if it already has content)
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]

            # Handle the final part - if it's a directory (ends with /) or if this path already exists as a directory
            if name.endswith("/") or (parts[-1] in node and isinstance(node[parts[-1]], dict)):
                # This is a directory, preserve any existing content
                if parts[-1] not in node:
                    node[parts[-1]] = {}
                # If it was already a file, we keep it as a file (this should be rare)
            else:
                # This is a file, set it as empty dict
                node[parts[-1]] = {}

        def build(node, path="", result=None):
            if result is None:
                result = []
            for k, v in node.items():
                entry_path = f"{path}/{k}" if path else k
                entry = {"rel_path": PurePosixPath(entry_path), "name": k, "is_folder": bool(v), "children": []}
                result.append(entry)
                if v:
                    build(v, entry_path, result)
            return result
        return build(tree)


    if HAS_LIBARCHIVE:
        try:
            return list(_list_libarchive_contents(archive_path))
        except Exception as e:
            logger.debug(f"libarchive listing failed for {archive_path}: {e}, falling back to standard method")

    try:
        archive_suffix = archive_path.suffix.lower()
        if archive_suffix in config.extensions_archive:
            if archive_suffix == ".zip":
                with zipfile.ZipFile(archive_path) as zf:
                    return zip_walk(zf)
            elif archive_suffix == ".7z":

                if py7zr is None:
                    raise ImportError(f"py7zr was not found to extract {archive_path}")

                def py7zr_walk(archive: py7zr.SevenZipFile):
                    tree = {}
                    for f in archive.getnames():
                        parts = f.rstrip("/").split("/")
                        node = tree
                        for part in parts[:-1]:
                            # Get or create directory node (don't overwrite if it already has content)
                            if part not in node or not isinstance(node[part], dict):
                                node[part] = {}
                            node = node[part]

                        # Handle the final part - if it's a directory (ends with /) or if this path already exists as a directory
                        if f.endswith("/") or (parts[-1] in node and isinstance(node[parts[-1]], dict)):
                            # This is a directory, preserve any existing content
                            if parts[-1] not in node:
                                node[parts[-1]] = {}
                            # If it was already a file, we keep it as a file (this should be rare)
                        else:
                            # This is a file, set it as empty dict
                            node[parts[-1]] = {}
                    return build(tree)


                with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                    return py7zr_walk(archive)
        raise ValueError(f"Unsupported archive format, only {config.extensions_archive} supported")
    except Exception as e:
        logger.error(f"Error listing archive contents {archive_path}: {e}", exc_info=True)
        return []  # Return empty list instead of None when there's an error


def read_archive_file_lines(archive_path: Path, inner_file: str | PurePosixPath, max_lines: Optional[int] = None):
    """
    Reads lines from a file inside a ZIP or 7z archive.

    Parameters:
        archive_path: Path - path to archive (.zip or .7z)
        inner_file: name of file inside archive
        max_lines: Optional[int] - maximum number of lines to read (None for all lines)

    Returns:
        Tuple of (list of lines, last line) from the file. Last line is None if file is empty.
    """
    target = str(inner_file)
    lines = []
    last_line = None

    if HAS_LIBARCHIVE:
        try:
            return _read_from_libarchive(archive_path, target, n_head=max_lines, skip_header=0)
        except Exception as e:
            logger.debug(f"libarchive reading failed for {archive_path}: {e}, falling back to standard method")

    # --- ZIP case ---
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open(str(inner_file)) as f:
                if max_lines is not None:
                    # Read first max_lines lines
                    for i in range(max_lines):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.decode("utf-8", errors="ignore").rstrip("\n"))

                    # Read the rest to get the last line
                    remaining_lines = []
                    for line in f:
                        remaining_lines.append(line.decode("utf-8", errors="ignore").rstrip("\n"))
                    if remaining_lines:
                        last_line = remaining_lines[-1]
                    elif lines:
                        last_line = lines[-1]
                else:
                    # Read all lines
                    for line in f:
                        lines.append(line.decode("utf-8", errors="ignore").rstrip("\n"))
                    if lines:
                        last_line = lines[-1]

    # --- 7z case ---
    elif archive_path.suffix.lower() == ".7z":
        import tempfile

        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            if py7zr is None:
                raise ImportError(f"py7zr was not found to extract {archive_path}")

            temp_dir_path = Path(temp_dir)
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                # Extract only the specific file to the temporary directory
                archive.extract(path=temp_dir, targets=[str(inner_file)])

                # Read the extracted file
                extracted_file_path = temp_dir_path / inner_file
                if not extracted_file_path.exists():
                    raise FileNotFoundError(f"File {inner_file} not found in archive")

                with open(extracted_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    if max_lines is not None:
                        # Read first max_lines lines
                        for i in range(max_lines):
                            line = f.readline()
                            if not line:
                                break
                            lines.append(line.rstrip('\n'))

                        # Read the rest to get the last line
                        remaining_lines = f.readlines()
                        # Strip newlines
                        remaining_lines = [line.rstrip('\n') for line in remaining_lines]
                        if remaining_lines:
                            last_line = remaining_lines[-1]
                        elif lines:
                            last_line = lines[-1]
                    else:
                        # Read all lines
                        lines = f.readlines()
                        # Strip newlines
                        lines = [line.rstrip('\n') for line in lines]
                        if lines:
                            last_line = lines[-1]

    else:
        raise ValueError(f"Unsupported archive format, only {config.extensions_archive} supported")

    return lines, last_line


def extract_archive_with_command_line(src: Path, dst: Path, command: str) -> bool:
    """Extract an archive using command line tool.

    Args:
        src: Path - path to the archive file
        dst: Path - path to the destination directory
        command: Command to use for extraction (e.g., '7z', 'unzip')

    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        # Make sure destination directory exists
        dst.mkdir(parents=True, exist_ok=True)

        if command == '7z':
            result = subprocess.run(['7z', 'x', str(src), f'-o{dst}', '-y'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                print(f"Error extracting archive {src} with 7z command: {result.stderr}")
                return False
        elif command == 'unzip':
            result = subprocess.run(['unzip', '-o', str(src), '-d', str(dst)],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                print(f"Error extracting archive {src} with unzip command: {result.stderr}")
                return False
        else:
            print(f"Unsupported command: {command}")
            return False
    except Exception as e:
        print(f"Error extracting archive {src} with {command} command: {e}")
        return False


def extract_zip_archive(src: Path, dst: Path) -> bool:
    """Extract a ZIP archive using various methods.

    Args:
        src: Path - path to the ZIP file
        dst: Path - path to the destination directory

    Returns:
        True if extraction was successful, False otherwise
    """
    # Try command line tool as fallback
    return extract_archive_with_command_line(src, dst, 'unzip')


def extract_7z_archive(src: Path, dst: Path) -> bool:
    """Extract a 7z archive using various methods.

    Args:
        src: Path - path to the 7z file
        dst: Path - path to the destination directory

    Returns:
        True if extraction was successful, False otherwise
    """
    if py7zr is not None:
        try:  # Pure Python extraction
            with py7zr.SevenZipFile(src, mode='r') as archive:
                archive.extractall(path=dst)
            return True
        except Exception as e:
            logger.debug(f"Pure Python 7z extraction failed: {e}")

    # Fall back to command line tool
    return extract_archive_with_command_line(src, dst, '7z')


def create_temp_directory(base_dir: Optional[Path] = None) -> Path:
    """Create a temporary directory.

    Args:
        base_dir: Base directory for temporary directory (optional)

    Returns:
        Path to the temporary directory
    """
    if base_dir:
        temp_dir = Path(tempfile.mkdtemp(dir=base_dir))
    else:
        temp_dir = Path(tempfile.mkdtemp())

    return temp_dir


def remove_directory(path: Path) -> bool:
    """Remove a directory and all its contents.

    Args:
        path: Path to the directory to remove

    Returns:
        True if removal was successful, False otherwise
    """
    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"Error removing directory {path}: {e}")
        return False