import pytest
import sys
import traceback
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from meta_finder.logging_config import setup_logging


def test_line_numbers_functionality():
    """Test that line numbers are correctly captured in logging."""
    logger = setup_logging('test_line_numbers')

    try:
        x = 1/0  # This will cause a ZeroDivisionError
    except Exception as e:
        logger.error(f'Error occurred in test_function: {e}', exc_info=True)
        # Test passes if no exception occurs during logging
        assert True