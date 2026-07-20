"""CLI entry point for path availability checker.

This module provides command-line interface for checking path availability
and detecting renamed folders in cruise data directories.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from meta_finder.logging_config import setup_logging
from post_processing.path_checker import (
    check_paths_from_tsv,
    find_tsv_files,
    MATCH_CUTOFF,
)


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Check path availability and detect renamed folders in cruise data. "
            "Reads paths from TSV files (meta_TCM.tsv or files_TCM.tsv) and "
            "generates mapping files with old to new path mappings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Check a single TSV file\n"
            "  python -m meta_finder.path_checker_main meta_TCM.tsv\n\n"
            "  # Check all TSV files in a directory\n"
            "  python -m meta_finder.path_checker_main --directory ./cruises\n\n"
            "  # Specify custom output path\n"
            "  python -m meta_finder.path_checker_main meta_TCM.tsv "
            "--output path_mapping.tsv\n\n"
            "  # Specify custom similarity cutoff\n"
            "  python -m meta_finder.path_checker_main meta_TCM.tsv "
            "--cutoff 0.5\n\n"
            "  # Specify base search directory\n"
            "  python -m meta_finder.path_checker_main meta_TCM.tsv "
            "--search-dir ./Cruises"
        ),
    )

    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Path to TSV file to process or directory containing TSV files. "
            "If --directory is specified, this must be a directory path."
        ),
    )

    parser.add_argument(
        "--directory", "-d",
        action="store_true",
        help="Process all TSV files in the specified directory",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help=(
            "Output path for mapping file. For single file mode, specifies the "
            "output file. For directory mode, specifies output suffix."
        ),
    )

    parser.add_argument(
        "--search-dir", "-s",
        type=Path,
        help=(
            "Base directory for similarity search when paths don't exist. "
            "Defaults to the input file's parent directory."
        ),
    )

    parser.add_argument(
        "--cutoff", "-c",
        type=float,
        default=MATCH_CUTOFF,
        help=(
            f"Minimum similarity threshold for considering a match (default: {MATCH_CUTOFF}). "
            "Values closer to 1.0 require higher similarity."
        ),
    )

    parser.add_argument(
        "--pattern", "-p",
        default="*_TCM.tsv",
        help=(
            "Glob pattern for finding TSV files in directory mode "
            "(default: *_TCM.tsv)"
        ),
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error logging output",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> Optional[str]:
    """Validate command-line arguments.

    Args:
        args: Parsed arguments namespace

    Returns:
        Error message if validation fails, None otherwise
    """
    if not args.input:
        return "Input path is required. Specify a TSV file or directory."

    input_path = Path(args.input)

    if not input_path.exists():
        return f"Input path does not exist: {input_path}"

    if args.directory:
        if not input_path.is_dir():
            return f"Input must be a directory when --directory is used: {input_path}"
    else:
        if not input_path.is_file():
            return f"Input must be a file when --directory is not used: {input_path}"
        if not input_path.suffix == ".tsv":
            return f"Input file must have .tsv extension: {input_path}"

    if hasattr(args, "search_dir") and args.search_dir and not args.search_dir.exists():
        return f"Search directory does not exist: {args.search_dir}"

    if hasattr(args, "cutoff") and (args.cutoff < 0.0 or args.cutoff > 1.0):
        return f"Cutoff must be between 0.0 and 1.0: {args.cutoff}"

    return None


def process_single_file(
    tsv_path: Path,
    output_path: Path,
    search_dir: Optional[Path],
    cutoff: float
) -> int:
    """Process a single TSV file.

    Args:
        tsv_path: Path to input TSV file
        output_path: Path to output mapping file
        search_dir: Base directory for similarity search
        cutoff: Minimum similarity threshold

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        check_paths_from_tsv(tsv_path, output_path, search_dir, cutoff)
        logger.info(f"Successfully processed: {tsv_path}")
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error processing {tsv_path}: {e}", exc_info=True)
        return 1


def process_directory(
    directory: Path,
    output_suffix: str,
    search_dir: Optional[Path],
    cutoff: float,
    pattern: str
) -> int:
    """Process all TSV files in a directory.

    Args:
        directory: Directory containing TSV files
        output_suffix: Suffix for output files
        search_dir: Base directory for similarity search
        cutoff: Minimum similarity threshold
        pattern: Glob pattern for TSV files

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        tsv_files = find_tsv_files(directory, pattern)

        if not tsv_files:
            logger.warning(f"No TSV files found matching pattern '{pattern}' in {directory}")
            return 0

        success_count = 0
        error_count = 0

        for tsv_path in tsv_files:
            output_path = tsv_path.with_suffix("").with_suffix("")
            output_path = output_path.with_name(f"{output_path.name}{output_suffix}")

            try:
                check_paths_from_tsv(tsv_path, output_path, search_dir, cutoff)
                success_count += 1
                logger.info(f"Successfully processed: {tsv_path}")
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {tsv_path}: {e}", exc_info=True)

        logger.info(
            f"Processing complete: {success_count} succeeded, {error_count} failed"
        )

        return 0 if error_count == 0 else 1

    except Exception as e:
        logger.error(f"Error processing directory {directory}: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else (
        logging.WARNING if args.quiet else logging.INFO
    )
    setup_logging(log_level=log_level)

    # Validate arguments
    error = validate_arguments(args)
    if error:
        logger.error(error)
        return 1

    input_path = Path(args.input)

    try:
        if args.directory:
            # Directory mode
            output_suffix = "_path_mapping.tsv"
            if args.output:
                output_suffix = args.output.name

            return process_directory(
                input_path,
                output_suffix,
                args.search_dir,
                args.cutoff,
                args.pattern
            )
        else:
            # Single file mode
            output_path = args.output
            if not output_path:
                # Generate default output path
                output_path = input_path.with_suffix("").with_suffix("")
                output_path = output_path.with_name(f"{output_path.name}_path_mapping.tsv")

            return process_single_file(
                input_path,
                output_path,
                args.search_dir,
                args.cutoff
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
