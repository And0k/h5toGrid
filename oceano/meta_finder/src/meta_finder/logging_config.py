"""
Centralized logging configuration for the meta_finder project.
"""
import logging
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Optional


# Module-level cache for package directory and name (shared between CustomLogger and formatter)
_package_dir: Optional[Path] = None
_package_name: Optional[str] = None


def _get_package_info() -> tuple[Path, str]:
    """Get the package directory and name, cached for performance."""
    global _package_dir, _package_name
    if _package_dir is None or _package_name is None:
        # Get the path of this file (logging_config.py)
        my_file = Path(sys._getframe().f_code.co_filename)
        _package_dir = my_file.parent
        # Extract package name from directory (last part of path)
        _package_name = _package_dir.name
    return _package_dir, _package_name


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

        # Store original name and funcName
        original_name = record.name
        original_funcName = record.funcName

        # Get package name from module-level cache
        _, package_name = _get_package_info()
        package_prefix = f"{package_name}."

        # Clean up the name by removing package prefix
        cleaned_name = original_name.replace(package_prefix, '')

        # Check for same-module call (caller_function added by CustomLogger)
        if hasattr(record, 'caller_function') and record.caller_function:
            # Same-module call: show "caller>callee" instead of "module.callee"
            record.name = f"{record.caller_function}>{record.funcName}"
            # Clear funcName to avoid duplication in format string
            record.funcName = ""
        else:
            # Cross-module call or top-level: show module name
            record.name = cleaned_name

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
        record.funcName = original_funcName
        if hasattr(record, '_original_lineno'):
            record.lineno = original_lineno

        return formatted


    def formatException(self, exc_info):
        """
        Format exception with modified traceback style.
        Replaces 'File "file", line 123' with 'File "file:123"' which are treated as links to line in vscode terminal
        """
        # Get standard formatted exception
        tb_lines = traceback.format_exception(*exc_info)

        # Modify each line
        modified_lines = []
        for line in tb_lines:
            # Replace the pattern: File "filename", line lineno
            # With: File "filename:lineno"
            modified = re.sub(
                r'File "([^"]+)", line (\d+)',
                r'File "\1:\2"',
                line
            )
            modified_lines.append(modified)

        return ''.join(modified_lines)


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

    def findCaller(self, stack_info=False, stacklevel=1):
        """
        Override findCaller to skip importlib frames and find the actual caller in user code.

        This prevents the logger from showing 'importlib._bootstrap' as the calling module
        instead of the actual module that made the logging call.

        We positively identify frames from the meta_finder project by checking if the file
        is in the meta_finder source directory, excluding logging_config.py itself.
        """
        # Start from the current frame (findCaller itself) and walk up
        frame = sys._getframe()

        # Get cached package directory and name from module-level cache
        package_dir, package_name = _get_package_info()

        # Walk up the stack to find the first frame from package (excluding logging_config.py)
        while frame:
            filename = Path(frame.f_code.co_filename)
            abs_filename = filename.resolve()
            module_name = frame.f_globals.get('__name__', '')

            # Check if this file is from our package using multiple indicators:
            # 1. File path is within package directory (most reliable)
            # 2. Module name starts with package name (covers both 'package' and 'package.*')
            is_package_file = (
                self._is_path_within_directory(abs_filename, package_dir) or
                module_name.startswith(package_name)
            )

            if is_package_file:
                # Skip logging_config.py - we want user code, not our logging internals
                if filename.name == 'logging_config.py':
                    frame = frame.f_back
                    continue

                # Found a package frame - this is our callee (user code)
                callee_frame = frame
                co = callee_frame.f_code
                return (co.co_filename, callee_frame.f_lineno, co.co_name, None)

            # Move up to the caller's frame
            frame = frame.f_back

        # Fallback to parent implementation
        return super().findCaller(stack_info, stacklevel)

    @staticmethod
    def _is_path_within_directory(path: Path, directory: Path) -> bool:
        """Check if path is within directory (handles symlinks, relative paths, edge cases)."""
        try:
            return path.resolve().is_relative_to(directory.resolve())
        except (ValueError, OSError):
            return False

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, stack_info=False):
        """
        Override makeRecord to add caller_function to the LogRecord for same-module calls.
        """
        # Get cached package directory and name from module-level cache
        package_dir, package_name = _get_package_info()

        # Walk up the stack from the logging call to find same-module caller
        frame = sys._getframe(2)  # Skip makeRecord and the calling frame
        caller_func = None

        while frame:
            filename = Path(frame.f_code.co_filename)
            abs_filename = filename.resolve()
            module_name = frame.f_globals.get('__name__', '')

            # Check if this file is from our package using multiple indicators
            is_package_file = (
                self._is_path_within_directory(abs_filename, package_dir) or
                module_name.startswith(package_name)
            )

            if is_package_file:
                # Skip logging_config.py - we want user code
                if filename.name == 'logging_config.py':
                    frame = frame.f_back
                    continue

                # This is the callee (where logger.info/debug/etc was called)
                callee_filename = abs_filename

                # Check the caller of this frame
                caller_frame = frame.f_back
                if caller_frame:
                    caller_filename = Path(caller_frame.f_code.co_filename).resolve()
                    # Same file check
                    if caller_filename == callee_filename:
                        caller_func = caller_frame.f_code.co_name
                        break

                break

            frame = frame.f_back

        # Add caller_function to extra if we detected a same-module call
        if extra is None:
            extra = {}
        if caller_func:
            extra['caller_function'] = caller_func

        # Call parent's makeRecord
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, stack_info)

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


def setup_logging(
    name: Optional[str] = None,
    log_level: int = None,
    include_function_name: bool = True,
    log_file_dir: Optional[Path] = None,
    log_file_sfx: str = "meta_finder",
    console_level: int = None,
    file_level: int = None
) -> logging.Logger:
    """
    Set up centralized logging with consistent formatting and handlers.
    (with function name / line number in logs)
    Args:
        name: Name for the logger (defaults to __name__)
        log_level: Root logger level (defaults to config.logging_level)
        include_function_name: Whether to include function name in log format
        log_file_dir: Directory for log files (defaults to current working directory / "meta")
        log_file_sfx: log file name suffix
        console_level: Log level for console output (defaults to config.logging_level)
        file_level: Log level for file output (defaults to DEBUG)

    Returns:
        Configured logger instance
    """
    # Import config to get the global logging level setting
    from . import config

    if name is None:
        # Get the caller's module name - sys._getframe(1) is the caller of setup_logging
        caller_frame = sys._getframe(1)
        if caller_frame is not None:
            name = caller_frame.f_globals.get("__name__", "")
        else:
            name = ""
    elif isinstance(name, str) and __name__ == "__main__":
        name = ""

    # Set default values from config if not provided
    if log_level is None:
        log_level = config.logging_level
    if console_level is None:
        console_level = config.logging_level
    if file_level is None:
        file_level = logging.DEBUG

    # Set the custom logger class as the default
    logging.setLoggerClass(CustomLogger)

    # Create formatter with or without function name
    if include_function_name:
        formatter = SafeStringFormatter(
            "%(asctime)s\t%(name)s.%(funcName)s:%(lineno)d\t%(levelname)s:\t%(message)s",
            datefmt="%H:%M:%S"
        )
    else:
        formatter = SafeStringFormatter(
            "%(asctime)s\t%(name)s\t%(levelname)s:\t%(message)s",
            datefmt="%H:%M:%S"
        )

    # Set up root logger
    # Use the most permissive level so handlers can do their own filtering
    # This ensures DEBUG messages can reach the file handler even when console is INFO
    most_permissive_level = min(log_level, console_level, file_level)
    root_logger = logging.getLogger()
    root_logger.setLevel(most_permissive_level)

    # Clear any existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # Create file handler
    if log_file_dir is None:
        # Check if we're in test mode
        import os
        log_file_dir = (
            Path(__file__).parent.parent.parent / "test_data" / "meta_temp" / "logs"
            if os.environ.get("META_FINDER_TEST_MODE") == "1"
            else Path.cwd() / "meta"
        )
    log_file_dir.mkdir(exist_ok=True, parents=True)

    timestamp = time.strftime("%y%m%d_%H%M")
    log_file = log_file_dir / f"{timestamp}_{log_file_sfx}.log"

    # Create file handler with UTF-8 encoding to handle special characters properly
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Get the named logger
    logger = logging.getLogger(name)

    # Reset to default logger class to avoid affecting other loggers
    logging.setLoggerClass(logging.Logger)

    return logger


# def log_error_with_traceback(
#     logger: logging.Logger,
#     message: str,
#     exception: Optional[Exception] = None,
#     print_to_console: bool = True
# ) -> None:
#     """
#     Log an error with full traceback information.

#     Args:
#         logger: Logger instance to use
#         message: Error message to log
#         exception: Exception object (if available)
#         print_to_console: Whether to also print to console
#     """
#     if exception:
#         # Format the full traceback including exception and line number
#         tb_str = traceback.format_exc()
#         full_message = f"{message}\nException: {str(exception)}\nTraceback:\n{tb_str}"
#     else:
#         full_message = message
#         tb_str = traceback.format_stack()
#         full_message += f"\nTraceback:\n{''.join(tb_str)}"

#     # Always log at ERROR level
#     logger.error(full_message)

#     # Optionally print to console as well
#     if print_to_console:
#         print(full_message, file=sys.stderr)


# def get_error_details(exception: Exception) -> dict:
#     """
#     Extract detailed error information including line number and context.

#     Args:
#         exception: Exception object to analyze

#     Returns:
#         Dictionary with error details
#     """
#     exc_type, exc_value, exc_traceback = type(exception), exception, exception.__traceback__

#     # Get the last frame in the traceback
#     tb = exc_traceback
#     while tb.tb_next:
#         tb = tb.tb_next

#     frame = tb.tb_frame
#     filename = tb.tb_frame.f_code.co_filename
#     line_number = tb.tb_lineno
#     function_name = tb.tb_frame.f_code.co_name

#     return {
#         'exception_type': exc_type.__name__,
#         'exception_message': str(exc_value),
#         'filename': filename,
#         'line_number': line_number,
#         'function_name': function_name,
#         'locals': {k: repr(v) for k, v in frame.f_locals.items()}
#     }