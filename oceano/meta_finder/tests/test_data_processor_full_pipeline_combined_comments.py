import pytest
import os
import json
from pathlib import Path
from meta_finder.collect import process_all_metadata
from meta_finder.file_writer import write_metadata_table
from meta_finder.config import get_config


def test_full_pipeline_with_combined_device_comments(create_test_device_structure, test_output_dir):
    """Test the full pipeline with combined device comments."""
    # Create a mock device directory structure using the fixture
    device_dir = create_test_device_structure(
        device_name="inclinometer_test",
        has_text_output=True,
        has_info_file=False  # We'll create our own info file
    )

    # Get the cruise directory (parent of device_dir)
    cruise_dir = device_dir.parent

    # Create our own info_devices.json file with test content
    info_json_path = device_dir / "info_devices.json"
    info_json_content = {
        "i5": ["Point1", "100", "5", "A", "?", "?", "2023-06-18 18:00:00", "2023-06-18 19:0:00"],
        "i14": ["Point1", "100", "5", "A", "?", "?", "2023-06-18 18:00:00", "2023-06-18 19:00:00"]
    }
    with open(info_json_path, 'w') as f:
        json.dump(info_json_content, f)

    # Create a combined file that contains data for both i5 and i14 devices
    # This will trigger the combined comment generation
    text_output_dir = device_dir / "text_output"
    text_output_dir.mkdir(exist_ok=True)

    # Create a combined file with both i5 and i14 data
    # Use comma-separated device names to ensure both devices are extracted
    combined_file = text_output_dir / "230618_1800bin10s_i5,i14.tsv"
    combined_content = (
        "Time\tVabs_i5\tVdir_i5\tv_i5\tu_i5\tInclination_i5\tTemp_i5\t"
        "Vabs_i14\tVdir_i14\tv_i14\tu_i14\tInclination_i14\tTemp_i14\n"
        "2023-06-18 18:00:00.000000\t1.0\t45.0\t0.5\t0.3\t5.0\t25.0\t"
        "1.1\t46.0\t0.6\t0.4\t5.1\t25.5\n"
        "2023-06-18 18:10:00.000000\t1.1\t46.0\t0.6\t0.4\t5.1\t25.5\t"
        "1.2\t47.0\t0.7\t0.5\t5.2\t26.0\n"
    )
    combined_file.write_text(combined_content)

    # Temporarily add our temp directory to search_dirs
    config = get_config()
    original_search_dirs = config.search_dirs
    config.search_dirs = tuple(list(config.search_dirs) + [Path(cruise_dir.parent)])  # Add parent directory to search in

    try:
        # Discover cruise directories and their device directories
        from meta_finder.file_finder import discover_device_dirs
        cruise_dir_list = [cruise_dir]  # Use the created cruise directory
        cruise_to_dev_dirs = {cruise_dir: [device_dir]}  # Manually create the mapping

        # Process the metadata
        json_metadata = process_all_metadata(cruise_to_dev_dirs)

        # Create a temporary output file in the test output directory
        temp_output = test_output_dir / "test_meta_TCM_combined.tsv"

        # Write the metadata table
        write_metadata_table(json_metadata, temp_output)

        # Read and check the output
        with open(temp_output, 'r') as f:
            content = f.read()

            # Check if combined comments are present
            assert "i5+i14 output" in content, "Combined comments should be found in output"

    finally:
        # Restore original search_dirs
        config.search_dirs = original_search_dirs