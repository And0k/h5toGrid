import json
import logging
from pathlib import Path
from meta_finder.metadata_extractor import read_metadata_files_to_dict


def test_original_content_logging():
    """Test that enhanced logging works with the original content provided by the user."""

    # Path to the original content file
    json_path = Path("test_data/info_devices.json")

    if not json_path.exists():
        print(f"Test file {json_path} does not exist")
        return

    print(f"Testing enhanced logging with original content from {json_path}")

    # Enable debug logging to see the enhanced messages
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Test the function - this should generate the enhanced logging messages
    result = read_metadata_files_to_dict(json_path)

    print("Function executed successfully")
    print(f"Number of devices processed: {len(result)}")

    # Show basic info about the result without printing Unicode characters
    for device_id, metadata in list(result.items())[:2]:  # Show first 2 devices
        print(f"Device {device_id}: {len(metadata)} metadata fields processed")

    print("Test completed!")


if __name__ == "__main__":
    test_original_content_logging()


if __name__ == "__main__":
    test_original_content_logging()