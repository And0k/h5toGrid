import logging
import tempfile
import json
from pathlib import Path
from meta_finder.metadata_extractor import read_metadata_files_to_dict


class MessageCaptureHandler(logging.Handler):
    """Custom logging handler to capture messages."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        msg = self.format(record)
        self.messages.append(msg)


def verify_enhanced_logging():
    """Verify that enhanced logging messages are appearing."""

    # Create a custom handler to capture messages
    capture_handler = MessageCaptureHandler()
    capture_handler.setLevel(logging.DEBUG)

    # Set up logging with our custom handler
    logger = logging.getLogger('meta_finder.metadata_extractor')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(capture_handler)

    # Also add a console handler to see messages in real-time
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a temporary JSON file with sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_data = {
            "i7": ["5", None, 1.5, "⭡", 54.858967, 21.146680, "2019-12-10T14:22", "2019-12-26T12:40"],
            "i23": ["4", None, 1.5, "⭡", 54.68898, 21.103796, "2019-12-10T13:15", "2019-12-26T11:5"]
        }
        json.dump(json_data, f)
        temp_file_path = Path(f.name)

    try:
        print("=" * 80)
        print("VERIFYING ENHANCED LOGGING MESSAGES")
        print("=" * 80)

        print("\nCalling read_metadata_files_to_dict()...")
        print("Capturing all log messages...")
        print("-" * 80)

        # Call the function - this should generate enhanced logging messages
        result = read_metadata_files_to_dict(temp_file_path)

        print("-" * 80)
        print("Function completed successfully!")
        print(f"Processed {len(result)} devices")

        # Check if enhanced logging messages appeared
        enhanced_messages_found = []
        expected_messages = [
            "and converting to dict format",
            "Raw JSON data extracted from",
            "Converted metadata for device",
            "Final converted metadata result"
        ]

        print("\nChecking for enhanced logging messages:")
        for expected_msg in expected_messages:
            found = any(expected_msg in msg for msg in capture_handler.messages)
            status = "✓ FOUND" if found else "✗ NOT FOUND"
            print(f"  {status}: {expected_msg}")
            if found:
                enhanced_messages_found.append(expected_msg)

        print(f"\nTotal enhanced messages found: {len(enhanced_messages_found)}/{len(expected_messages)}")

        if len(enhanced_messages_found) == len(expected_messages):
            print("SUCCESS: All enhanced logging messages are working correctly!")
        else:
            print("ISSUE: Some enhanced logging messages are missing.")
            print("\nAll captured messages:")
            for i, msg in enumerate(capture_handler.messages, 1):
                print(f"  {i:2d}. {msg}")

        print("=" * 80)

    finally:
        # Clean up the temporary file
        temp_file_path.unlink()

        # Clean up handlers
        logger.removeHandler(capture_handler)
        logger.removeHandler(console_handler)


if __name__ == "__main__":
    verify_enhanced_logging()