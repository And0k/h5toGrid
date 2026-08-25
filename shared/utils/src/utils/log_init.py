"""
Message-style layer of unified monorepo logging: str.format()-style records,
caller-context filters and duplicate-suppression options used at call sites
(``lf = LoggingStyleAdapter(__name__)``).  Handler/formatter/root-logger setup
lives in :mod:`utils.logging_config`.
"""

from dataclasses import dataclass
import logging
from inspect import currentframe

# Setup API implementation lives in logging_config; re-exported for legacy call sites
from .logging_config import init_logging as init_logging


@dataclass(repr=False)  # , slots=True
class Message:
    fmt: str
    args: tuple

    def __str__(self):
        try:
            return self.fmt.format(*self.args) if self.args else self.fmt
        except KeyError:  # allow dict argument
            try:
                return self.fmt.format_map(self.args[0])
            except IndexError:
                return (
                    self.fmt
                    + "\n- Bad format string!\n"
                    + (f"Logging arguments: {self.args}" if len(self.args) else "")
                )
        except (TypeError, IndexError):
            print("Logging error due to wrong format string:", self.fmt, "for arguments:", self.args)
            raise


class LoggingContextFilter(logging.Filter):
    # https://docs.python.org/3/howto/logging-cookbook.html

    def filter(self, record):
        # First frame is the file in which this class is defined.
        frame = currentframe().f_back
        try:
            # Walk back through multiple levels of logging.
            while "logging" in frame.f_code.co_filename or frame.f_code.co_name.startswith(
                (
                    "log",
                    "<module>",
                )
            ):
                # print(frame.f_code.co_filename, frame.f_code.co_name)
                frame = frame.f_back
        except Exception:  # any frame-walk failure → keep last valid frame
            pass
        # Create the overrides
        # record.filename = full_name
        record.funcName = frame.f_code.co_name
        record.lineno = frame.f_lineno
        return True


class LoggingStyleAdapter(logging.LoggerAdapter):
    """
    Switch stdlib %-style to loguru {} format (str.format() style) in logging messages. Usage:
    logger = LoggingStyleAdapter(__name__)
    also prepends message with [self.extra['id']]
    """

    def __init__(self, logger, extra=None):
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        f = LoggingContextFilter()
        logger.addFilter(f)

        self.message = Message("", ())
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        try:
            extra_id = self.extra["id"]
        except KeyError:
            return msg, kwargs
        else:
            return f"[{extra_id}] {msg}", kwargs

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            self.message.fmt, kwargs = self.process(msg, kwargs)
            self.message.args = args
            self.logger._log(level, self.message, (), **kwargs)


class LoggingFilter_DuplicatesOption(logging.Filter):
    """
    A logging filter that provides options to handle duplicate log messages.

    This filter can either suppress duplicate messages at regular intervals or
    append a counter showing how many times a message has been repeated.

    Usage:
        - To suppress duplicates: logger.info("message", {"filter_same": interval})
        - To add counter: logger.info("message", {"add_same_counter": interval})
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lookup_lockup = {}

    def filter(self, record):
        """
        Filter log records based on duplicate handling configuration.

        Args:
            record: The log record to filter

        Returns:
            bool: True if the record should be logged, False otherwise
        """
        try:
            msg_args0 = (msg := record.msg).args[0]
            try:
                # Check for 'filter_same' option - suppresses duplicates
                log_interval = msg_args0.pop("filter_same")
                b_change_msg = False
            except KeyError:
                try:
                    # Check for 'add_same_counter' option - adds repetition counter
                    log_interval = msg_args0.pop("add_same_counter")
                    b_change_msg = True
                except KeyError:
                    # No duplicate handling options found - allow the message
                    b_change_msg = False
                    return True

                # If interval is None or 1 (and not changing message), allow the message
                if log_interval is None or (log_interval == 1 and not b_change_msg):
                    return True
            except TypeError:  # argument is not a dict => no filter options inside
                return True  # continue with general logging argument
        except (AttributeError, IndexError):
            return True  # Invalid message format - allow the message

        # Create a unique key for the message based on format and arguments
        current_log = (msg.fmt, tuple(msg_args0.items()))

        try:
            # Get the current count for this message
            cnt = self._lookup_lockup[current_log]
        except KeyError:
            # First occurrence of this message - initialize counter and allow
            self._lookup_lockup[current_log] = 0
            return True

        # Increment the counter for this message
        cnt += 1
        self._lookup_lockup[current_log] = cnt

        # If adding counter to message, modify the format string
        if b_change_msg:
            record.msg.fmt += f" (repeated {cnt})"

        # Return True only if we should log at this interval
        return not cnt % log_interval


def my_logging(name, logger=None):
    logger = logging.getLogger(name)
    logger.addFilter(LoggingFilter_DuplicatesOption())
    return LoggingStyleAdapter(logger)
