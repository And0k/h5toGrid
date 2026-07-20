import os
import tempfile
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from meta_finder import utils_sys
import re

def test_archive_split_files_detailed():
    """Detailed debug script to test archive split files functionality."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Create test files with the same pattern but different timestamps
        # First file (earlier timestamp)
        first_file_content = (
            "Time\tVabs_i03\tVdir_i03\n"
            "2023-05-08 15:51:0.000000\t1.0\t45.0\n"
            "2023-05-08 15:52:00.000000\t1.1\t46.0\n"
            "2023-05-08 15:53:00.0000\t1.2\t47.0\n"
        )

        # Second file (later timestamp) - continuation of data
        second_file_content = (
            "Time\tVabs_i03\tVdir_i03\n"
            "2023-05-08 16:51:0.000000\t2.0\t55.0\n"
            "2023-05-08 16:52:00.000000\t2.1\t56.0\n"
            "2023-05-08 16:53:00.000\t2.2\t57.0\n"
        )

        # Create a zip archive with the split files
        archive_path = os.path.join(temp_dir, "text_output.zip")
        with zipfile.ZipFile(archive_path, 'w') as zipf:
            zipf.writestr("230508_1551bin2s@i03.tsv", first_file_content)
            zipf.writestr("230508_1651bin2s@i03.tsv", second_file_content)

        print(f"Created archive: {archive_path}")

        # Test the pattern matching logic directly
        rel_path = Path("230508_1551bin2s@i03.tsv")
        base_name = rel_path.name
        parent_path = rel_path.parent

        print(f"Base name: {base_name}")
        print(f"Parent path: {parent_path}")

        # List all files in the archive
        archive_contents = utils_sys.list_archive_recursive(Path(archive_path))
        print(f"Archive contents: {archive_contents}")

        # Extract the pattern from the filename (everything except the timestamp part)
        # Pattern: {yymmdd_HHMM}{rest of filename}
        pattern_match = re.match(r'(\d{6}_\d{4})(.*)', base_name)
        if pattern_match:
            timestamp_part = pattern_match.group(1)
            rest_part = pattern_match.group(2)
            print(f"Pattern match: timestamp_part={timestamp_part}, rest_part={rest_part}")

            # Look for files with the same rest part in the same directory
            matching_files = []
            for item in archive_contents:
                if not item["is_folder"]:
                    file_path = item["rel_path"]
                    print(f"Checking file: {file_path}, parent: {file_path.parent}")
                    # Check if file is in the same directory
                    if file_path.parent == parent_path:
                        file_name = file_path.name
                        print(f"File name: {file_name}")
                        file_match = re.match(r'(\d{6}_\d{4})(' + re.escape(rest_part) + r')', file_name)
                        if file_match:
                            file_timestamp = file_match.group(1)
                            print(f"Found matching file: {file_path} with timestamp {file_timestamp}")
                            matching_files.append((file_timestamp, file_path))
            # Sort by timestamp
            matching_files.sort(key=lambda x: x[0])
            print(f"Found {len(matching_files)} matching files: {matching_files}")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_archive_split_files_detailed()