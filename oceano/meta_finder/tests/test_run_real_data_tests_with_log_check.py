#!/usr/bin/env python
"""
Test script that discovers and runs all existing tests that use real data,
then reads and reports any errors found in meta/*.log files.
"""
import os
import sys
import subprocess
import tempfile
import re
from pathlib import Path
import logging
from datetime import datetime


def discover_real_data_tests():
    """
    Discover test files that use real data by searching for patterns in test files.
    """
    test_dir = Path("tests")
    real_data_tests = []

    if not test_dir.exists():
        print(f"Test directory {test_dir} does not exist")
        return real_data_tests

    # Look for test files that contain patterns indicating real data usage
    for test_file in test_dir.glob("*.py"):
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

            # Check for patterns that indicate real data usage
            if any(pattern in content.lower() for pattern in [
                'real_data',
                ':/workdata',  # Path for real data (using forward slashes)
                'real file',
                'real data',
                'directory.exists()',
                'path.exists()',
                'os.path.exists',
                'pathlib.path',
                'discover_datafiles_for_all_dev_in_dev_dir',
                'find_navigation_files'
            ]):
                real_data_tests.append(test_file)

    return real_data_tests


def run_test_and_capture_logs(test_file, log_prefix="test_run"):
    """
    Run a single test file and capture log output.

    Args:
        test_file: Path to the test file to run
        log_prefix: Prefix for log file naming

    Returns:
        tuple: (exit_code, output, log_files)
    """
    print(f"Running test: {test_file}")

    # Capture timestamp to identify logs created during this test run
    timestamp = datetime.now()

    # Run the test using subprocess to capture output
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5-minute timeout
        )

        # Find log files created after the test started
        log_files = []
        meta_dir = Path("meta")
        if meta_dir.exists():
            for log_file in meta_dir.glob("*.log"):
                # Check if log file was modified recently (during our test run)
                if log_file.stat().st_mtime >= timestamp.timestamp():
                    log_files.append(log_file)

        return result.returncode, result.stdout, result.stderr, log_files

    except subprocess.TimeoutExpired:
        print(f"Test {test_file} timed out")
        return -1, "", "Test timed out", []
    except Exception as e:
        print(f"Error running test {test_file}: {e}")
        return -1, "", str(e), []


def extract_errors_from_logs(log_files):
    """
    Extract error entries from the specified log files.

    Args:
        log_files: List of log file paths to examine

    Returns:
        dict: Dictionary mapping log file to list of error entries
    """
    errors = {}

    error_patterns = [
        r'ERROR.*',
        r'CRITICAL.*',
        r'Traceback.*',
        r'Exception.*',
        r'Error.*'
    ]

    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Find all lines that match error patterns
        error_lines = []
        for line_num, line in enumerate(content.split('\n'), 1):
            for pattern in error_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    error_lines.append((line_num, line.strip()))
                    break  # Don't add the same line multiple times

        if error_lines:
            errors[log_file] = error_lines

    return errors


def run_pytest_tests():
    """
    Run pytest tests with real data patterns.

    Returns:
        tuple: (exit_code, output, log_files)
    """
    print("Running pytest tests with real data...")

    # Capture timestamp to identify logs created during this test run
    timestamp = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-k", "real", "-v"],
            capture_output=True,
            text=True,
            timeout=600  # 10-minute timeout for pytest
        )

        # Find log files created after the test started
        log_files = []
        meta_dir = Path("meta")
        if meta_dir.exists():
            for log_file in meta_dir.glob("*.log"):
                # Check if log file was modified recently (during our test run)
                if log_file.stat().st_mtime >= timestamp.timestamp():
                    log_files.append(log_file)

        return result.returncode, result.stdout, result.stderr, log_files

    except subprocess.TimeoutExpired:
        print("Pytest run timed out")
        return -1, "", "Pytest timed out", []
    except Exception as e:
        print(f"Error running pytest: {e}")
        return -1, "", str(e), []


def run_all_pytest_tests():
    """
    Run all pytest tests (not just real data related) to identify comprehensive errors.

    Returns:
        tuple: (exit_code, output, log_files)
    """
    print("Running all pytest tests to find any errors...")

    # Capture timestamp to identify logs created during this test run
    timestamp = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "--tb=short"],  # -x to stop on first failure
            capture_output=True,
            text=True,
            timeout=1200  # 20-minute timeout for all tests
        )

        # Find log files created after the test started
        log_files = []
        meta_dir = Path("meta")
        if meta_dir.exists():
            for log_file in meta_dir.glob("*.log"):
                # Check if log file was modified recently (during our test run)
                if log_file.stat().st_mtime >= timestamp.timestamp():
                    log_files.append(log_file)

        return result.returncode, result.stdout, result.stderr, log_files

    except subprocess.TimeoutExpired:
        print("Pytest run timed out")
        return -1, "", "Pytest timed out", []
    except Exception as e:
        print(f"Error running pytest: {e}")
        return -1, "", str(e), []


def main():
    """
    Main function to run all real data tests and check for errors in logs.
    """
    print("Discovering real data tests...")
    real_data_tests = discover_real_data_tests()

    print(f"Found {len(real_data_tests)} tests that use real data:")
    for test in real_data_tests:
        print(f"  - {test}")

    all_failed_tests = []
    all_error_logs = {}

    # Run each discovered real data test
    for test_file in real_data_tests:
        exit_code, stdout, stderr, log_files = run_test_and_capture_logs(test_file)

        if exit_code != 0:
            all_failed_tests.append((test_file, exit_code, stdout, stderr))

        # Extract errors from any log files generated during this test run
        errors = extract_errors_from_logs(log_files)
        for log_file, error_entries in errors.items():
            if log_file in all_error_logs:
                all_error_logs[log_file].extend(error_entries)
            else:
                all_error_logs[log_file] = error_entries

    # Run all pytest tests to catch any general errors
    print("\nRunning all pytest tests to find any errors...")
    exit_code, pytest_stdout, pytest_stderr, log_files = run_all_pytest_tests()

    if exit_code != 0:
        all_failed_tests.append(("all_pytest_tests", exit_code, pytest_stdout, pytest_stderr))

    # Extract errors from pytest-generated log files
    errors = extract_errors_from_logs(log_files)
    for log_file, error_entries in errors.items():
        if log_file in all_error_logs:
            all_error_logs[log_file].extend(error_entries)
        else:
            all_error_logs[log_file] = error_entries

    # Report failed tests
    if all_failed_tests:
        print(f"\n{len(all_failed_tests)} tests failed:")
        for test, exit_code, stdout, stderr in all_failed_tests:
            print(f"\nTest: {test} (exit code: {exit_code})")
            if stdout:
                print(f"  STDOUT:\n{stdout}")
            if stderr:
                print(f"  STDERR:\n{stderr}")
    else:
        print("\nAll tests passed successfully!")

    # Report errors found in log files generated during test runs
    if all_error_logs:
        print(f"\nFound errors in {len(all_error_logs)} log files generated during test runs:")
        for log_file, error_entries in all_error_logs.items():
            print(f"\nLog file: {log_file}")
            for line_num, error_line in error_entries:
                print(f"  Line {line_num}: {error_line}")
    else:
        print("\nNo errors found in log files generated during test runs.")

    # Also scan for any ERROR/Critical entries in all log files in the meta directory
    print("\nScanning all log files in meta directory for errors...")
    meta_dir = Path("meta")
    all_logs_with_errors = {}

    if meta_dir.exists():
        for log_file in meta_dir.glob("*.log"):
            errors = extract_errors_from_logs([log_file])
            if log_file in errors:
                all_logs_with_errors[log_file] = errors[log_file]

    if all_logs_with_errors:
        print(f"\nFound errors in {len(all_logs_with_errors)} total log files:")
        for log_file, error_entries in all_logs_with_errors.items():
            print(f"\nLog file: {log_file}")
            for line_num, error_line in error_entries:
                print(f"  Line {line_num}: {error_line}")

    # Summary
    total_failed = len(all_failed_tests)
    total_log_errors = sum(len(errors) for errors in all_logs_with_errors.values())

    print(f"\nSUMMARY:")
    print(f"  Failed tests: {total_failed}")
    print(f"  Total error entries in logs: {total_log_errors}")

    if total_failed > 0 or total_log_errors > 0:
        print("  Status: ISSUES FOUND")
        return 1
    else:
        print("  Status: ALL GOOD")
        return 0
