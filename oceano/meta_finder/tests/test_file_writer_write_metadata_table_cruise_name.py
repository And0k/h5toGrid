import pytest
from pathlib import Path
from meta_finder.file_writer import write_metadata_table


def test_cruise_name_mapping(test_output_dir):
    """Test that cruise names are correctly mapped in output"""
    # Create test data with 'cruise' key
    test_metadata = {
        Path("test.json"): {
            "device1": {
                "cruise": "ABP53",
                "device_id": "i3",
                "point": "T3",
                "sea_depth": "115.8",
                "height_above_bottom": "0",
                "lat": "5.9129516",
                "lon": "19.072616",
                "time_st": "2023-05-08 23:24:04",
                "time_en": "2023-05-23 15:41:30",
                "burst_dt": "-",
                "bursts_t": "-",
                "data_paths": []
            }
        }
    }

    # Write to temporary file in test output directory
    output_file = test_output_dir / "test_cruise_name_output.tsv"
    write_metadata_table(test_metadata, output_file)

    # Read the output and check that cruise_name column has the correct value
    with open(output_file, 'r') as f:
        lines = f.readlines()

    # Check header
    assert 'cruise_name' in lines[0]

    # Check data line
    assert 'ABP53' in lines[1]