import os
import tempfile
import shutil
from pathlib import Path
from meta_finder.data_proc_funcs import read_file_lines_universal, _find_matching_files_in_directory

def test_split_files_issue():
    """Debug script to understand why split files are not being detected."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Create test files with the same pattern but different timestamps
        # First file (earlier timestamp)
        first_file_path = os.path.join(temp_dir, "230508_1551bin2s@i03.tsv")
        with open(first_file_path, "w") as f:
            f.write("Time\tVabs_i03\tVdir_i03\n")
            f.write("2023-05-08 15:51:00.000000\t1.0\t45.0\n")
            f.write("2023-05-08 15:52:00.000000\t1.1\t46.0\n")
            f.write("2023-05-08 15:53:00.000000\t1.2\t47.0\n")

        # Second file (later timestamp) - continuation of data
        second_file_path = os.path.join(temp_dir, "230508_1651bin2s@i03.tsv")
        with open(second_file_path, "w") as f:
            f.write("Time\tVabs_i03\tVdir_i03\n")
            f.write("2023-05-08 16:51:00.000000\t2.0\t55.0\n")
            f.write("2023-05-08 16:52:00.000000\t2.1\t56.0\n")
            f.write("2023-05-08 16:53:00.000000\t2.2\t57.0\n")

        print(f"Created files:")
        print(f"  {first_file_path}")
        print(f"  {second_file_path}")

        # Test the split file detection
        base_name = "230508_1551bin2s@i03.tsv"
        parent_dir = Path(temp_dir)
        print(f"Base name: {base_name}")
        print(f"Parent dir: {parent_dir}")

        matching_files = _find_matching_files_in_directory(parent_dir, base_name)
        print(f"Matching files: {matching_files}")
        print(f"Number of matching files: {len(matching_files)}")

        # Test the main function
        lines, last_line = read_file_lines_universal(Path(temp_dir), Path("230508_1551bin2s@i03.tsv"))
        print(f"Lines read: {len(lines)}")
        print(f"Last line: {last_line}")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_split_files_issue()