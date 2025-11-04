"""
Centralized logging configuration for the meta_finder project.
"""
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional


class SafeStringFormatter(logging.Formatter):
    """Custom formatter that sanitizes log messages to handle problematic characters safely."""

    def sanitize_message(self, msg):
        """Safely convert any message to a string that can be logged safely."""
        if isinstance(msg, str):
            # Replace problematic characters that can cause encoding issues
            # This preserves the information but makes it safe for logging
            try:
                # Try to encode as UTF-8 to test if it's safe
                msg.encode('utf-8')
                return msg
            except UnicodeEncodeError:
                # If there's an encoding error, replace problematic characters
                return msg.encode('utf-8', errors='replace').decode('utf-8')
        else:
            # For non-string objects, convert to string safely
            try:
                str_repr = str(msg)
                str_repr.encode('utf-8')
                return str_repr
            except UnicodeEncodeError:
                return repr(msg).encode('utf-8', errors='replace').decode('utf-8')

    def format(self, record):
        # Sanitize the message to handle problematic characters
        record.msg = self.sanitize_message(record.msg)

        # Also sanitize any arguments
        if record.args:
            sanitized_args = []
            for arg in record.args:
                sanitized_args.append(self.sanitize_message(arg))
            record.args = tuple(sanitized_args)

        # Store original name
        original_name = record.name
        # Clean up the name
        record.name = original_name.replace('meta_finder.', '')

        # If we have an original line number from an exception, use it instead of the current line number
        if hasattr(record, '_original_lineno'):
            # Store the original lineno
            original_lineno = record.lineno
            # Replace with the original line number from the exception
            record.lineno = record._original_lineno

        # Format the record with cleaned name
        formatted = super().format(record)

        # Restore original values
        record.name = original_name
        if hasattr(record, '_original_lineno'):
            record.lineno = original_lineno

        return formatted


class CustomLogger(logging.Logger):
    """Custom logger that provides accurate line number reporting for both regular logging and exception logging.

    For regular log messages (info, warning, etc.), defaults stacklevel to 2 to show the calling
    function's line number instead of the logging function's line.

    For exception logging (when exc_info is provided), extracts the original line number from
    the exception traceback and reports that instead of the line where logger.error was called,
    making debugging significantly easier by showing where errors actually occurred.
    """

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        # For exception logging, we want to preserve the original exception line number
        # rather than adjusting the stacklevel which can cause incorrect line reporting
        if exc_info:
            # When logging exceptions, we want to show the original error location
            # Extract line number from the exception traceback and use it
            try:
                # Extract line number from the exception traceback
                import sys
                if exc_info is True:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                else:
                    exc_type, exc_value, exc_tb = exc_info

                if exc_tb:
                    # Walk to the end of the traceback to get the original error location
                    tb = exc_tb
                    while tb.tb_next:
                        tb = tb.tb_next
                    original_line_number = tb.tb_lineno

                    # Store the original line number in extra to be used by the formatter
                    if extra is None:
                        extra = {}
                    extra['_original_lineno'] = original_line_number
                    # Use stacklevel=2 to show the calling function (not the _log method itself)
                    stacklevel = 2
            except Exception:
                # If we can't extract the line number, use the default behavior
                if stacklevel == 1:
                    stacklevel = 2
        else:
            # Use stacklevel=2 by default for regular logging to show the calling function's line
            if stacklevel == 1:
                stacklevel = 2

        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)


def get_formatter(name=True, funcName=True, datefmt=None, msecs=False):
    """Create formatter with or without function name, ..."""
    formatter = SafeStringFormatter(
        "".join([
            "%(asctime)s",
            ".%(msecs)03d" if msecs else "",
            " %(name)s" if name else "",
            ".%(funcName)s:%(lineno)d" if funcName else "",
            " %(levelname)s: %(message)s",
        ]),
        datefmt=datefmt,
    )
    return formatter


def setup_logging(
    name: str = __name__,
    log_level: int = logging.INFO,
    log_file_dir: Optional[Path] = None,
    log_file_sfx: str = "meta_finder",
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
    console_format_args={"datefmt": "%H:%M:%S"},
    file_format_args={}
) -> logging.Logger:
    """
    Set up centralized logging with consistent formatting and handlers.

    Args:
        name: Name for the logger (defaults to __name__)
        log_level: Root logger level

        log_file_dir: Directory for log files (defaults to current working directory / "meta")
        log_file_sfx: log file name suffix
        console_level: Log level for console output
        file_level: Log level for file output (defaults to INFO)
        console_format_args: arguments of get_formatter() to specify whether to include function name in
        conslole log format, ...
        file_format_args: same for file log format
    Returns:
        Configured logger instance
    """
    # Set the custom logger class as the default
    logging.setLoggerClass(CustomLogger)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(get_formatter(**console_format_args))

    # Create file handler
    if log_file_dir is None:
        # Check if we're in test mode
        import os
        if os.environ.get('AB_SIO_RAS_TEST_MODE') == '1':
            log_file_dir = Path("test_data") / "meta_temp" / "logs"
        else:
            log_file_dir = Path.cwd() / "meta"
    log_file_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%y%m%d_%H%M")
    log_file = log_file_dir / f"{timestamp}_{log_file_sfx}.log"

    # Create file handler with UTF-8 encoding to handle special characters properly
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(get_formatter(**file_format_args))

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Get the named logger
    logger = logging.getLogger(name)

    # Reset to default logger class to avoid affecting other loggers
    logging.setLoggerClass(logging.Logger)

    return logger
