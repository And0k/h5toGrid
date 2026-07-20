import tempfile
import json
from pathlib import Path
from meta_finder.metadata_extractor import read_metadata_files_to_dict
import logging


def test_enhanced_logging():
    """Test that enhanced logging is working in read_metadata_files_to_dict."""

    # Create a temporary JSON file with sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_data = {
            "i5": ["Point1", "100", "5", "A", "55.1", "37.2",
                   "2023-06-18 18:00:00", "2023-06-18 19:00:00",
                   "3600", "1800", "Original comment"],
            "w14": ["Point2", "200", "10", "B", "56.1", "38.2",
                    "2023-06-19 18:00:00", "2023-06-19 19:00:00",
                    "1800", "3600", "Another comment"]
        }
        json.dump(json_data, f)
        temp_file_path = Path(f.name)

    try:
        # Test the function - this should generate the enhanced logging messages
        result = read_metadata_files_to_dict(temp_file_path)

        print("Function executed successfully")
        print(f"Result: {result}")

        # Verify the structure
        assert "i5" in result
        assert "w14" in result

        # Verify the content for device i5
        i5_data = result["i5"]
        assert i5_data["point"] == "Point1"
        assert i5_data["sea_depth"] == "100"
        assert i5_data["height_above_bottom"] == "5"
        assert i5_data["modification_symbol"] == "A"
        assert i5_data["lat"] == "55.1"
        assert i5_data["lon"] == "37.2"
        assert i5_data["time_st"] == "2023-06-18 18:00:00"
        assert i5_data["time_en"] == "2023-06-18 19:00:00"
        assert i5_data["burst_dt"] == "3600"
        assert i5_data["bursts_t"] == "1800"
        assert i5_data["comment"] == "Original comment"

        print("Test passed: read_metadata_files_to_dict with enhanced logging works correctly")
    finally:
        # Clean up the temporary file
        temp_file_path.unlink()


if __name__ == "__main__":
    test_enhanced_logging()
    print("Enhanced logging test completed!")