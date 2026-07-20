import os
import tempfile
import shutil
import zipfile
from pathlib import Path
from meta_finder.data_proc_funcs import read_file_lines_universal

def test_archive_split_files():
    """Debug script to test archive split files functionality."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Create test files with the same pattern but different timestamps
        # First file (earlier timestamp)
        first_file_content = (
            "Time\tVabs_i03\tVdir_i03\n"
            "2023-05-08 15:51:00.000000\t1.0\t45.0\n"
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

        # Test without max_lines (read all lines)
        lines, last_line = read_file_lines_universal(Path(archive_path), Path("230508_1551bin2s@i03.tsv"))

        print(f"Lines read: {len(lines)}")
        for i, line in enumerate(lines):
            print(f"  Line {i}: {repr(line)}")
        print(f"Last line: {repr(last_line)}")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_archive_split_files()