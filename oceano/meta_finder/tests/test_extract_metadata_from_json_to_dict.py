import tempfile
import json
from pathlib import Path
from meta_finder.metadata_extractor import read_metadata_files_to_dict


def test_read_metadata_files_to_dict():
    """Test that the new function correctly converts JSON metadata to dict format."""

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
        # Test the function
        result = read_metadata_files_to_dict(temp_file_path)

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

        # Verify the content for device w14
        w14_data = result["w14"]
        assert w14_data["point"] == "Point2"
        assert w14_data["sea_depth"] == "200"
        assert w14_data["height_above_bottom"] == "10"
        assert w14_data["modification_symbol"] == "B"
        assert w14_data["lat"] == "56.1"
        assert w14_data["lon"] == "38.2"
        assert w14_data["time_st"] == "2023-06-19 18:00:00"
        assert w14_data["time_en"] == "2023-06-19 19:00:00"
        assert w14_data["burst_dt"] == "1800"
        assert w14_data["bursts_t"] == "3600"
        assert w14_data["comment"] == "Another comment"

        print("Test passed: read_metadata_files_to_dict works correctly")
    finally:
        # Clean up the temporary file
        temp_file_path.unlink()


def test_read_metadata_files_to_dict_with_shorter_list():
    """Test that the function handles shorter metadata lists correctly."""

    # Create a temporary JSON file with sample data (shorter than 11 elements)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_data = {
            "i5": ["Point1", "100", "5", "A", "55.1", "37.2"]  # Only 6 elements
        }
        json.dump(json_data, f)
        temp_file_path = Path(f.name)

    try:
        # Test the function
        result = read_metadata_files_to_dict(temp_file_path)

        # Verify the structure
        assert "i5" in result

        # Verify the content - missing fields should have default values
        i5_data = result["i5"]
        assert i5_data["point"] == "Point1"
        assert i5_data["sea_depth"] == "100"
        assert i5_data["height_above_bottom"] == "5"
        assert i5_data["modification_symbol"] == "A"
        assert i5_data["lat"] == "55.1"
        assert i5_data["lon"] == "37.2"
        assert i5_data["time_st"] == "?"  # Default value
        assert i5_data["time_en"] == "?"  # Default value
        assert i5_data["burst_dt"] == ""  # Default value for burst_dt
        assert i5_data["bursts_t"] == ""  # Default value for bursts_t
        assert i5_data["comment"] == "?"  # Default value

        print("Test passed: read_metadata_files_to_dict handles shorter lists correctly")
    finally:
        # Clean up the temporary file
        temp_file_path.unlink()


if __name__ == "__main__":
    test_read_metadata_files_to_dict()
    test_read_metadata_files_to_dict_with_shorter_list()
    print("All tests passed!")