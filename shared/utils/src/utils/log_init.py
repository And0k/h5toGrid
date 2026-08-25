from dataclasses import dataclass
import logging
from inspect import currentframe

from utils.init import dir_create_if_need, l, this_prog_basename
import os
import sys
from pathlib import Path


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
                return self.fmt + '\n- Bad format string!\n' + (
                    f'Logging arguments: {self.args}' if len(self.args) else ''
                    )
        except (TypeError, IndexError):
            print('Logging error due to wrong format string:', self.fmt, 'for arguments:', self.args)
            raise


class LoggingContextFilter(logging.Filter):
    # https://docs.python.org/3/howto/logging-cookbook.html

    def filter(self, record):
        # First frame is the file in which this class is defined.
        frame = currentframe().f_back
        try:
            # Walk back through multiple levels of logging.
            while "logging" in frame.f_code.co_filename or frame.f_code.co_name.startswith((
                "log",
                "<module>",
            )):
                # print(frame.f_code.co_filename, frame.f_code.co_name)
                frame = frame.f_back
        except:
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


def init_logging(logger="", log_file=None, level_file="INFO", level_console=None):
    """
    Logging to file flogD/flogN.log and console with piorities level_file and levelConsole
    :param logger: name of logger or logger. Default: '' - name of root logger.
    :param log_file: name of log file. Default: & + "this program file name"
    :param level_file: 'INFO'
    :param level_console: 'WARN'
    :return: logging Logger

    Call example:
    l= init_logging('', None, args.verbose)
    l.warning(msgFile)
    """
    global l
    if log_file:
        if not os.path.isabs(log_file):
            # if flogD is None:
            flogD = os.path.dirname(sys.argv[0])
            log_file = os.path.join(flogD, log_file)
    else:
        # if flogD is None:
        flogD = os.path.join(os.path.dirname(sys.argv[0]), "log")
        dir_create_if_need(flogD)
        log_file = os.path.join(flogD, f"&{this_prog_basename()}.log")  # '&' is for autoname indication

    if logger is None:
        logger = sys._getframe(1).f_back.f_globals["__name__"]  # replace with name of caller
    elif isinstance(logger, str) and __name__ == "__main__":
        logger = ""

    was_l = bool(l)
    if was_l:
        try:  # a bit more check that we already have logger
            l = logging.getLogger(logger) if isinstance(logger, str) else logger
        except Exception:
            pass
        if l and l.hasHandlers():
            l.handlers.clear()  # or if we have good handlers return l?
    else:
        l = logging.getLogger(logger) if isinstance(logger, str) else logger

    try:
        filename = Path(log_file)
        b_default_path = not filename.parent.exists()
    except FileNotFoundError:
        b_default_path = True
    if b_default_path:
        filename = Path(__file__).parent / "logs" / filename.name

    # Create handlers if there no them in root
    if not l.root.hasHandlers():
        # Force UTF-8 for file handler — system encoding (e.g. cp1251) can't encode e.g. ×
        logging.basicConfig(
            filename=filename,
            format="%(asctime)s %(message)s",
            level=level_file,
            encoding="utf-8",
        )

        # set up logging to console — reconfigure stderr to UTF-8 for PyInstaller frozen builds
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
        console = logging.StreamHandler()
        console.setLevel(level_console or "INFO")  # default INFO regardless of file level
        # set a format which is simpler for console use
        formatter = logging.Formatter("%(message)s")  # %(name)-12s: %(levelname)-8s ...
        console.setFormatter(formatter)
        l.addHandler(console)

    # Or do not use root handlers:
    # l.propagate = not l.root.hasHandlers()  # to default

    if b_default_path:
        l.warning("Bad log path: %s! Using new path with default dir: %s", log_file, filename)

    return l
