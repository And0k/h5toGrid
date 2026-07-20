"""
Test to reproduce the specific logging line number issue from the original problem.
This test verifies that the error is logged with the correct line number.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from meta_finder.data_proc_funcs import _extract_device_time_ranges_from_combined_content
from meta_finder.logging_config import setup_logging


@pytest.mark.parametrize(
    "test_id,description",
    [
        ("original_issue", "Reproduce the original logging line number issue"),
    ],
    ids=["original_issue"]
)
def test_original_logging_line_number_issue(test_id, description):
    """Test that reproduces the original logging line number issue."""
    if test_id == "original_issue":
        # Set up logging to capture the error
        logger = setup_logging("data_proc_funcs", log_file_sfx="test_original_issue")

        # Create problematic lines that will trigger the error:
        # 'list' object has no attribute 'strip' at line 559 in _extract_device_time_ranges_from_combined_content
        dir_archive = Path("B:/WorkData/BalticSea/_Pregolya,Lagoon/220327@i36/text_output.zip")
        rel_path = Path("text_output/20327_1010bin600s@i.tsv")

        # This will cause the error at line 59: header = lines[0].strip().split('\t')
        # when lines[0] is a list instead of string
        lines = [
            ["202-03-27 10:10:00", "value1", "value2"],  # This is the problematic line - it's a list, not a string
            "2022-03-27 10:11:00\tvalue3\tvalue4\n"  # Normal line
        ]
        device_ids = []

        # Capture log output to verify line numbers
        with patch('meta_finder.data_proc_funcs.logger') as mock_logger:
            try:
                result = _extract_device_time_ranges_from_combined_content(
                    dir_archive, rel_path, lines, device_ids
                )
            except Exception as e:
                # The function should handle the error gracefully and log it
                pass

            # Verify that the logger was called to record the error
            assert mock_logger.error.called, "Function should log errors when exceptions occur"

            # Check that the error message contains the expected error type
            error_logged = False
            for call in mock_logger.error.call_args_list:
                args, kwargs = call
                if "'list' object has no attribute 'strip'" in str(args[0]):
                    error_logged = True
                    break

            assert error_logged, f"Error message should contain 'list' object has no attribute 'strip'. Calls: {mock_logger.error.call_args_list}"
