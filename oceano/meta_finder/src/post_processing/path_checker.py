"""Path availability checker with renamed folder detection.

This module provides functionality to:
1. Read paths from TSV files (meta_TCM.tsv or files_TCM.tsv)
2. Check if paths exist on the filesystem
3. Detect renamed folders using similarity matching
4. Generate mapping files with old to new path mappings
"""

import logging
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from meta_finder.logging_config import setup_logging


# Add match_dirs to path to import matcher functions
MATCH_DIRS_PATH = Path("C:/Work/Python/AB_SIO_RAS/cruises_organizer/match_dirs/src")
if str(MATCH_DIRS_PATH) not in sys.path:
    sys.path.insert(0, str(MATCH_DIRS_PATH))
import matcher  # noqa: E402

# Constants for similarity matching
MATCH_CUTOFF: float = 0.3

# Constants for file operations
TSV_DELIMITER: str = "\t"
ENCODING: str = "utf-8"

# Constants for output format
UNCERTAIN_MARKER: str="/?"


logger = logging.getLogger(__name__)


def find_best_match(
    old_path: str,
    candidate_paths: List[str],
    cutoff: float = MATCH_CUTOFF
) -> Tuple[Optional[str], float]:
    """Find the best matching path from candidates.

    Args:
        old_path: Original path to find match for
        candidate_paths: List of candidate paths to search
        cutoff: Minimum similarity threshold for matches

    Returns:
        Tuple of (best_match_path, similarity_score) or (None, 0.0) if no match
    """
    if not candidate_paths:
        return None, 0.0

    old_name = Path(old_path).name

    scores = []
    for candidate in candidate_paths:
        candidate_name = Path(candidate).name
        sim = matcher.hierarchical_weighed_similarity(old_name, candidate_name)
        if sim >= cutoff:
            scores.append((candidate, sim))

    if not scores:
        return None, 0.0

    best_match, best_sim = max(scores, key=lambda x: x[1])
    return best_match, best_sim


def find_similar_path_in_parent(
    old_path: str,
    base_search_dir: Path,
    cutoff: float = MATCH_CUTOFF
) -> Tuple[Optional[str], float]:
    """Search for similar paths in parent directories.

    When a path doesn't exist, search for similar folders in parent directories
    to detect renamed folders.

    Args:
        old_path: Original path that doesn't exist
        base_search_dir: Base directory to search in
        cutoff: Minimum similarity threshold

    Returns:
        Tuple of (similar_path, similarity_score) or (None, 0.0) if no match
    """
    old_path_obj = Path(old_path)
    old_name = old_path_obj.name

    # Search in the parent directory of where the old path should be
    expected_parent = old_path_obj.parent

    # If expected parent is empty or doesn't exist, search in base_search_dir
    if str(expected_parent) == "." or not (base_search_dir / expected_parent).exists():
        search_dir = base_search_dir
    else:
        search_dir = base_search_dir / expected_parent

    if not search_dir.exists():
        return None, 0.0

    # Get all directories in search directory
    candidate_dirs = []
    try:
        for item in search_dir.iterdir():
            if item.is_dir():
                candidate_dirs.append(str(item))
    except (PermissionError, OSError) as e:
        logger.warning(f"Cannot access directory {search_dir}: {e}")
        return None, 0.0

    return find_best_match(old_name, candidate_dirs, cutoff)


def read_tsv_paths(tsv_path: Path) -> List[str]:
    """Read paths from a TSV file.

    Args:
        tsv_path: Path to the TSV file

    Returns:
        List of paths read from the file

    Raises:
        FileNotFoundError: If TSV file doesn't exist
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    paths = []
    with open(tsv_path, "r", encoding=ENCODING) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Assume first column contains the path
                parts = line.split(TSV_DELIMITER)
                if parts:
                    paths.append(parts[0])

    logger.info(f"Read {len(paths)} paths from {tsv_path}")
    return paths


def check_path_availability(
    path: str,
    base_search_dir: Path,
    cutoff: float = MATCH_CUTOFF
) -> Tuple[bool, Optional[str], float]:
    """Check if a path exists and find similar path if not.

    Args:
        path: Path to check
        base_search_dir: Base directory for similarity search
        cutoff: Minimum similarity threshold

    Returns:
        Tuple of (exists, similar_path, similarity_score)
        - exists: True if path exists, False otherwise
        - similar_path: Best matching path if exists is False, None otherwise
        - similarity_score: Similarity score of the match
    """
    path_obj = Path(path)

    # Check if path exists relative to base_search_dir
    full_path = base_search_dir / path_obj if not path_obj.is_absolute() else path_obj

    if full_path.exists():
        return True, None, 0.0

    # Path doesn't exist, try to find similar
    similar_path, sim_score = find_similar_path_in_parent(path, base_search_dir, cutoff)

    return False, similar_path, sim_score


def generate_path_mapping(
    old_paths: List[str],
    base_search_dir: Path,
    cutoff: float = MATCH_CUTOFF
) -> Iterator[Tuple[str, str]]:
    """Generate mapping from old paths to new paths.

    For each old path:
    - If it exists, map to itself
    - If it doesn't exist and a good match is found, map to the match
    - If it doesn't exist and no good match, map to partial path with "/?"

    Args:
        old_paths: List of old paths to check
        base_search_dir: Base directory for similarity search
        cutoff: Minimum similarity threshold for good matches

    Yields:
        Tuples of (old_path, new_path)
    """
    for old_path in old_paths:
        exists, similar_path, sim_score = check_path_availability(old_path, base_search_dir, cutoff)

        if exists:
            # Path exists, map to the full resolved path
            full_path = base_search_dir / old_path if not Path(old_path).is_absolute() else Path(old_path)
            yield (old_path, str(full_path))
        elif similar_path and sim_score >= matcher.HIGH_CONFIDENCE_THRESHOLD:
            yield (old_path, similar_path)
        elif similar_path:
            # Similarity is below high confidence but above cutoff
            # Use the similar path but mark as uncertain
            yield (old_path, f"{similar_path}{UNCERTAIN_MARKER}")
        else:
            # No good match found - use partial path with marker
            path_obj = Path(old_path)
            if path_obj.parent != Path("."):
                yield (old_path, f"{path_obj.parent}{UNCERTAIN_MARKER}")
            else:
                yield (old_path, f"{UNCERTAIN_MARKER}")


def write_mapping_tsv(
    mappings: Iterator[Tuple[str, str]],
    output_path: Path,
    delimiter: str = TSV_DELIMITER
) -> None:
    """Write path mappings to a TSV file.

    Args:
        mappings: Iterator of (old_path, new_path) tuples
        output_path: Path to output TSV file
        delimiter: Delimiter to use in TSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding=ENCODING) as f:
        for old_path, new_path in mappings:
            f.write(f"{old_path}{delimiter}{new_path}\n")

    logger.info(f"Wrote path mappings to {output_path}")


def check_paths_from_tsv(
    tsv_path: Path,
    output_path: Path,
    base_search_dir: Optional[Path] = None,
    cutoff: float = MATCH_CUTOFF
) -> None:
    """Main function to check paths from TSV and generate mapping.

    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output mapping TSV file
        base_search_dir: Base directory for similarity search (defaults to TSV's parent)
        cutoff: Minimum similarity threshold for good matches
    """
    if base_search_dir is None:
        base_search_dir = tsv_path.parent

    logger.info(f"Processing TSV file: {tsv_path}")
    logger.info(f"Base search directory: {base_search_dir}")
    logger.info(f"Similarity cutoff: {cutoff}")

    old_paths = read_tsv_paths(tsv_path)
    mappings = generate_path_mapping(old_paths, base_search_dir, cutoff)
    write_mapping_tsv(mappings, output_path)

    logger.info(f"Path checking complete. Output written to {output_path}")


def find_tsv_files(directory: Path, pattern: str = "*_TCM.tsv") -> List[Path]:
    """Find TSV files matching a pattern in a directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern for TSV files

    Returns:
        List of matching TSV file paths
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    tsv_files = list(directory.glob(pattern))
    logger.info(f"Found {len(tsv_files)} TSV files matching '{pattern}' in {directory}")
    return tsv_files


def process_all_tsv_files(
    directory: Path,
    output_suffix: str = "_path_mapping.tsv",
    base_search_dir: Optional[Path] = None,
    cutoff: float = MATCH_CUTOFF
) -> None:
    """Process all TSV files in a directory.

    Args:
        directory: Directory containing TSV files
        output_suffix: Suffix to add to output files
        base_search_dir: Base directory for similarity search (defaults to TSV's parent)
        cutoff: Minimum similarity threshold for good matches
    """
    tsv_files = find_tsv_files(directory)

    for tsv_path in tsv_files:
        output_path = tsv_path.with_suffix("").with_suffix("")
        output_path = output_path.with_name(f"{output_path.name}{output_suffix}")

        try:
            check_paths_from_tsv(tsv_path, output_path, base_search_dir, cutoff)
        except Exception as e:
            logger.error(f"Error processing {tsv_path}: {e}")


def main() -> None:
    """Main entry point for path checking functionality."""
    setup_logging()
    logger.info("Path availability checker initialized")
