"""
Test that the --help command works properly for the collect script.
"""
import subprocess
import sys
from pathlib import Path


def test_collect_help_command():
    """Test that the collect script responds to --help command."""
    # Run the collect script with --help argument as a module
    result = subprocess.run(
        [sys.executable, "-m", "meta_finder.collect", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd="."
    )

    # Check that the command executed successfully (exit code 0 means help was shown)
    assert result.returncode == 0, f"Help command failed with return code {result.returncode}, stderr: {result.stderr}"

    # Check that help output contains expected content
    help_output = result.stdout

    # Verify that the help output contains key elements
    assert "usage: collect.py" in help_output, "Help output should contain usage information"
    assert "--overwrite-bad-devs-in-info-files" in help_output, "Help should contain the correct overwrite argument name"
    assert "--search-dirs" in help_output, "Help should contain search-dirs argument"
    assert "--create-info-files" in help_output, "Help should contain create-info-files argument"
    assert "--process-metadata" in help_output, "Help should contain process-metadata argument"

    # Verify that no data processing messages appear in help output
    assert "Starting info_devices.json creation process" not in help_output, "Help should not start data processing"
    assert "Processing" not in help_output or "processing" not in help_output.lower(), "Help output should not contain processing messages"

    print("Help command test passed successfully!")
    print(f"Help output contains {len(help_output)} characters")


def test_collect_help_command_alternative():
    """Test that the collect script responds to -h command as well."""
    # Run the collect script with -h argument as a module
    result = subprocess.run(
        [sys.executable, "-m", "meta_finder.collect", "-h"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd="."
    )

    # Check that the command executed successfully
    assert result.returncode == 0, f"Help command (-h) failed with return code {result.returncode}, stderr: {result.stderr}"

    # Check that help output contains expected content
    help_output = result.stdout

    # Verify that the help output contains key elements
    assert "usage: collect.py" in help_output, "Help output should contain usage information"
    assert "--overwrite-bad-devs-in-info-files" in help_output, "Help should contain the correct overwrite argument name"

    print("Help command (-h) test passed successfully!")


if __name__ == "__main__":
    test_collect_help_command()
    test_collect_help_command_alternative()
    print("All help command tests passed!")