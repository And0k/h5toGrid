import logging
from meta_finder.logging_config import setup_logging


def debug_logger():
    """Debug the logger configuration."""

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Create a logger using setup_logging
    logger = setup_logging(__name__)

    print("Testing logger...")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    print("Logger test completed!")


if __name__ == "__main__":
    debug_logger()