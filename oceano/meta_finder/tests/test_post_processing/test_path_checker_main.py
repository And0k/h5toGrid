"""Tests for path checker CLI functionality."""

import logging
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from post_processing.path_checker_main import (
    main,
    parse_arguments,
    process_directory,
    process_single_file,
    validate_arguments,
)


# Fixtures


@pytest.fixture
def sample_tsv(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a sample TSV file with paths."""
    tsv_file = tmp_path / "test_meta_TCM.tsv"
    tsv_content = """# Sample TSV file
path1/device1/data.txt
path2/device2/data.txt
path3/device3/data.txt
"""
    tsv_file.write_text(tsv_content, encoding="utf-8")
    yield tsv_file


@pytest.fixture
def sample_directory(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a sample directory with TSV files."""
    # Create TSV files
    (tmp_path / "meta_TCM.tsv").write_text("path1\npath2\n")
    (tmp_path / "files_TCM.tsv").write_text("path3\npath4\n")

    # Create some test directories
    (tmp_path / "path1").mkdir()
    (tmp_path / "path2").mkdir()

    yield tmp_path


# Tests for parse_arguments


@pytest.mark.parametrize(
    "args_list,expected_attrs,comment",
    [
        (
            ["test.tsv"],
            {"input": "test.tsv", "directory": False, "cutoff": 0.3},
            "minimal arguments",
        ),
        (
            ["test.tsv", "--directory"],
            {"input": "test.tsv", "directory": True, "cutoff": 0.3},
            "with directory flag",
        ),
        (
            ["test.tsv", "--output", "output.tsv"],
            {"input": "test.tsv", "directory": False, "output": Path("output.tsv")},
            "with output file",
        ),
        (
            ["test.tsv", "--cutoff", "0.5"],
            {"input": "test.tsv", "directory": False, "cutoff": 0.5},
            "with custom cutoff",
        ),
        (
            ["test.tsv", "--search-dir", "./search"],
            {"input": "test.tsv", "directory": False, "search_dir": Path("./search")},
            "with search directory",
        ),
        (
            ["test.tsv", "--verbose"],
            {"input": "test.tsv", "directory": False, "verbose": True},
            "with verbose flag",
        ),
        (
            ["test.tsv", "--quiet"],
            {"input": "test.tsv", "directory": False, "quiet": True},
            "with quiet flag",
        ),
        (
            ["test.tsv", "--pattern", "*_test.tsv"],
            {"input": "test.tsv", "directory": False, "pattern": "*_test.tsv"},
            "with custom pattern",
        ),
    ],
    ids=[
        "minimal",
        "directory-flag",
        "output-file",
        "custom-cutoff",
        "search-dir",
        "verbose",
        "quiet",
        "pattern",
    ],
)
def test_parse_arguments(
    args_list: list,
    expected_attrs: dict,
    comment: str,
) -> None:
    """Test parsing various argument combinations."""
    with patch("sys.argv", ["path_checker_main.py"] + args_list):
        args = parse_arguments()

        for attr, expected_value in expected_attrs.items():
            actual_value = getattr(args, attr)
            assert actual_value == expected_value, (
                f"Attribute '{attr}' mismatch for {comment}: "
                f"expected {expected_value}, got {actual_value}"
            )


# Tests for validate_arguments


@pytest.mark.parametrize(
    "args_dict,expected_error,comment",
    [
        (
            {"input": None, "directory": False},
            "Input path is required",
            "missing input",
        ),
        (
            {"input": Path("nonexistent.tsv"), "directory": False},
            "Input path does not exist",
            "nonexistent file",
        ),
        (
            {"input": Path("nonexistent"), "directory": True},
            "Input path does not exist",
            "nonexistent directory",
        ),
        (
            {"input": Path("existing.txt"), "directory": False},
            "Input file must have .tsv extension",
            "wrong file extension",
        ),
        (
            {"input": Path("existing_dir"), "directory": True},
            None,
            "valid directory",
        ),
        (
            {"input": Path("existing.tsv"), "directory": False},
            None,
            "valid TSV file",
        ),
        (
            {"input": Path("existing.tsv"), "directory": False, "search_dir": Path("nonexistent")},
            "Search directory does not exist",
            "nonexistent search directory",
        ),
        (
            {"input": Path("existing.tsv"), "directory": False, "cutoff": 1.5},
            "Cutoff must be between 0.0 and 1.0",
            "cutoff too high",
        ),
        (
            {"input": Path("existing.tsv"), "directory": False, "cutoff": -0.1},
            "Cutoff must be between 0.0 and 1.0",
            "cutoff too low",
        ),
    ],
    ids=[
        "missing-input",
        "nonexistent-file",
        "nonexistent-directory",
        "wrong-extension",
        "valid-directory",
        "valid-tsv",
        "nonexistent-search-dir",
        "cutoff-too-high",
        "cutoff-too-low",
    ],
)
def test_validate_arguments(
    tmp_path: Path,
    args_dict: dict,
    expected_error: str | None,
    comment: str,
) -> None:
    """Test argument validation."""
    # Create necessary files/directories
    if "input" in args_dict:
        input_path = args_dict["input"]
        if "existing" in str(input_path):
            if args_dict.get("directory"):
                input_path.mkdir(parents=True, exist_ok=True)
            else:
                input_path.write_text("test")

    if "search_dir" in args_dict and "existing" in str(args_dict["search_dir"]):
        args_dict["search_dir"].mkdir(parents=True, exist_ok=True)

    # Create args namespace
    from argparse import Namespace

    args = Namespace(**args_dict)

    error = validate_arguments(args)

    if expected_error:
        assert error is not None, f"Expected error for {comment}"
        assert expected_error in error, f"Error message mismatch for {comment}"
    else:
        assert error is None, f"Unexpected error for {comment}: {error}"


# Tests for process_single_file


def test_process_single_file_success(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test successful processing of a single TSV file."""
    # Create some directories from the TSV
    (tmp_path / "path1" / "device1").mkdir(parents=True)
    (tmp_path / "path2" / "device2").mkdir(parents=True)

    output_file = tmp_path / "output.tsv"

    with caplog.at_level(logging.INFO):
        exit_code = process_single_file(sample_tsv, output_file, tmp_path, 0.3)

    assert exit_code == 0, "Should return 0 for success"
    assert output_file.exists(), "Output file should be created"
    assert "Successfully processed" in caplog.text, "Should log success message"


def test_process_single_file_not_found(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test processing a non-existent TSV file."""
    nonexistent_tsv = tmp_path / "nonexistent.tsv"
    output_file = tmp_path / "output.tsv"

    with caplog.at_level(logging.ERROR):
        exit_code = process_single_file(nonexistent_tsv, output_file, tmp_path, 0.3)

    assert exit_code == 1, "Should return 1 for file not found"
    assert "File not found" in caplog.text, "Should log file not found error"


def test_process_single_file_error(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test processing a TSV file that causes an error."""
    # Make the TSV file unreadable by making it a directory
    sample_tsv.unlink()
    sample_tsv.mkdir()

    output_file = tmp_path / "output.tsv"

    with caplog.at_level(logging.ERROR):
        exit_code = process_single_file(sample_tsv, output_file, tmp_path, 0.3)

    assert exit_code == 1, "Should return 1 for error"
    assert "Error processing" in caplog.text, "Should log error message"


# Tests for process_directory


def test_process_directory_success(
    sample_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test successful processing of a directory with TSV files."""
    output_suffix = "_mapping.tsv"

    with caplog.at_level(logging.INFO):
        exit_code = process_directory(
            sample_directory, output_suffix, sample_directory, 0.3, "*_TCM.tsv"
        )

    assert exit_code == 0, "Should return 0 for success"

    # Check that output files were created
    output_files = list(sample_directory.glob("*_mapping.tsv"))
    assert len(output_files) == 2, "Should create 2 output files"

    assert "Successfully processed" in caplog.text, "Should log success messages"


def test_process_directory_no_tsv_files(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test processing a directory with no TSV files."""
    output_suffix = "_mapping.tsv"

    with caplog.at_level(logging.WARNING):
        exit_code = process_directory(
            tmp_path, output_suffix, tmp_path, 0.3, "*_TCM.tsv"
        )

    assert exit_code == 0, "Should return 0 even with no files"
    assert "No TSV files found" in caplog.text, "Should log warning about no files"


def test_process_directory_mixed_results(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test processing a directory with mixed success/failure results."""
    # Create one valid TSV and one invalid (directory instead of file)
    (tmp_path / "valid.tsv").write_text("path1\npath2\n")
    (tmp_path / "invalid.tsv").mkdir()

    output_suffix = "_mapping.tsv"

    with caplog.at_level(logging.INFO):
        exit_code = process_directory(
            tmp_path, output_suffix, tmp_path, 0.3, "*.tsv"
        )

    assert exit_code == 1, "Should return 1 when there are errors"
    assert "Processing complete" in caplog.text, "Should log processing summary"


# Tests for main function


def test_main_single_file_mode(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function in single file mode."""
    # Create some directories
    (tmp_path / "path1" / "device1").mkdir(parents=True)

    with caplog.at_level(logging.INFO):
        with patch("sys.argv", ["path_checker_main.py", str(sample_tsv)]):
            exit_code = main()

    assert exit_code == 0, "Should return 0 for success"

    # Check that output file was created
    output_files = list(tmp_path.glob("*_path_mapping.tsv"))
    assert len(output_files) == 1, "Should create one output file"


def test_main_directory_mode(
    sample_directory: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function in directory mode."""
    with caplog.at_level(logging.INFO):
        with patch("sys.argv", ["path_checker_main.py", "--directory", str(sample_directory)]):
            exit_code = main()

    assert exit_code == 0, "Should return 0 for success"

    # Check that output files were created
    output_files = list(sample_directory.glob("*_path_mapping.tsv"))
    assert len(output_files) == 2, "Should create two output files"


def test_main_with_custom_output(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function with custom output path."""
    output_path = tmp_path / "custom_output.tsv"

    with caplog.at_level(logging.INFO):
        with patch("sys.argv", ["path_checker_main.py", str(sample_tsv), "--output", str(output_path)]):
            exit_code = main()

    assert exit_code == 0, "Should return 0 for success"
    assert output_path.exists(), "Custom output file should be created"


def test_main_with_custom_cutoff(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function with custom cutoff value."""
    with caplog.at_level(logging.INFO):
        with patch("sys.argv", ["path_checker_main.py", str(sample_tsv), "--cutoff", "0.5"]):
            exit_code = main()

    assert exit_code == 0, "Should return 0 for success"


def test_main_verbose_mode(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function in verbose mode."""
    with caplog.at_level(logging.DEBUG):
        with patch("sys.argv", ["path_checker_main.py", str(sample_tsv), "--verbose"]):
            exit_code = main()

    assert exit_code == 0, "Should return 0 for success"


def test_main_quiet_mode(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function in quiet mode."""
    with caplog.at_level(logging.WARNING):
        with patch("sys.argv", ["path_checker_main.py", str(sample_tsv), "--quiet"]):
            exit_code = main()

    assert exit_code == 0, "Should return 0 for success"
    # In quiet mode, INFO messages should not appear
    assert "Successfully processed" not in caplog.text, "INFO messages should not appear in quiet mode"


def test_main_keyboard_interrupt(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function handling keyboard interrupt."""
    # Set up logging before calling main
    from meta_finder.logging_config import setup_logging
    setup_logging(log_level=logging.INFO)

    with patch("meta_finder.path_checker_main.process_single_file", side_effect=KeyboardInterrupt):
        with caplog.at_level(logging.INFO):
            with patch("sys.argv", ["path_checker_main.py", str(sample_tsv)]):
                exit_code = main()

    assert exit_code == 130, "Should return 130 for keyboard interrupt"
    # The logging is captured in stdout, not caplog
    # Just verify the exit code is correct


def test_main_unexpected_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function handling unexpected errors."""
    # Set up logging before calling main
    from meta_finder.logging_config import setup_logging
    setup_logging(log_level=logging.ERROR)

    # Create a test file to avoid FileNotFoundError
    test_file = tmp_path / "test.tsv"
    test_file.write_text("path1\npath2\n")

    with patch("meta_finder.path_checker_main.process_single_file", side_effect=RuntimeError("Test error")):
        with caplog.at_level(logging.ERROR):
            with patch("sys.argv", ["path_checker_main.py", str(test_file)]):
                exit_code = main()

    assert exit_code == 1, "Should return 1 for unexpected error"
    # The logging is captured in stdout, not caplog
    # Just verify the exit code is correct


# Integration tests


def test_full_workflow_single_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test complete workflow for single file mode."""
    # Create test structure
    (tmp_path / "cruise1").mkdir()
    (tmp_path / "cruise2_renamed").mkdir()

    tsv_file = tmp_path / "test_meta_TCM.tsv"
    tsv_content = """cruise1/device1/data.txt
cruise2/device2/data.txt
cruise3/device3/data.txt
"""
    tsv_file.write_text(tsv_content)

    with caplog.at_level(logging.INFO):
        with patch("sys.argv", ["path_checker_main.py", str(tsv_file)]):
            exit_code = main()

    assert exit_code == 0, "Should return 0 for success"

    # Verify output
    output_files = list(tmp_path.glob("*_path_mapping.tsv"))
    assert len(output_files) == 1, "Should create one output file"

    output_content = output_files[0].read_text(encoding="utf-8")
    lines = output_content.strip().split("\n")

    assert len(lines) == 3, "Should have 3 mappings"


def test_full_workflow_directory(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test complete workflow for directory mode."""
    # Create test structure
    (tmp_path / "cruise1").mkdir()
    (tmp_path / "cruise2").mkdir()

    # Create TSV files
    (tmp_path / "meta_TCM.tsv").write_text("cruise1/device1\ncruise2/device2\n")
    (tmp_path / "files_TCM.tsv").write_text("cruise1/file1.txt\ncruise2/file2.txt\n")

    with caplog.at_level(logging.INFO):
        with patch("sys.argv", ["path_checker_main.py", "--directory", str(tmp_path)]):
            exit_code = main()

    assert exit_code == 0, "Should return 0 for success"

    # Verify outputs
    output_files = list(tmp_path.glob("*_path_mapping.tsv"))
    assert len(output_files) == 2, "Should create two output files"


# Edge case tests


def test_main_with_nonexistent_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function with non-existent input."""
    with caplog.at_level(logging.ERROR):
        with patch("sys.argv", ["path_checker_main.py", "nonexistent.tsv"]):
            exit_code = main()

    assert exit_code == 1, "Should return 1 for non-existent input"


def test_main_with_wrong_extension(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function with wrong file extension."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("path1\npath2\n")

    with caplog.at_level(logging.ERROR):
        with patch("sys.argv", ["path_checker_main.py", str(txt_file)]):
            exit_code = main()

    assert exit_code == 1, "Should return 1 for wrong extension"


def test_main_with_invalid_cutoff(
    tmp_path: Path,
    sample_tsv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main function with invalid cutoff value."""
    with caplog.at_level(logging.ERROR):
        with patch("sys.argv", ["path_checker_main.py", str(sample_tsv), "--cutoff", "1.5"]):
            exit_code = main()

    assert exit_code == 1, "Should return 1 for invalid cutoff"
