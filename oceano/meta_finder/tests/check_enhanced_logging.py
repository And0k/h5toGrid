import logging
import tempfile
import json
from pathlib import Path
from meta_finder.metadata_extractor import read_metadata_files_to_dict


def check_enhanced_logging():
    """Check if enhanced logging messages are appearing."""

    # Set up logging to show debug messages
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

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
        print("CHECKING IF ENHANCED LOGGING MESSAGES APPEAR")
        print("=" * 80)

        print("\nCalling read_metadata_files_to_dict()...")
        print("Expected enhanced logging messages:")
        print("1. 'Extracting metadata from ... and converting to dict format'")
        print("2. 'Raw JSON data extracted from ...'")
        print("3. 'Converted metadata for device ...'")
        print("4. 'Final converted metadata result: ...'")
        print("-" * 80)

        # Call the function - this should generate enhanced logging messages
        result = read_metadata_files_to_dict(temp_file_path)

        print("-" * 80)
        print("Function completed successfully!")
        print(f"Processed {len(result)} devices")
        print("=" * 80)

    finally:
        # Clean up the temporary file
        temp_file_path.unlink()


if __name__ == "__main__":
    check_enhanced_logging()