"""Test HDF5 processor directory and file exclusion via ptn_dir_exclude patterns."""
import tempfile
from pathlib import Path

# from meta_finder import config
from meta_finder.hdf5_processor import _iter_hdf5_search_directories, find_hdf5_files, _is_valid_hdf5_file


def test_hdf5_search_directories_exclusion():
    """Test that _iter_hdf5_search_directories excludes directories matching patterns.

    Note: \bh5\b requires h5 as a whole word — underscores are word characters,
    so 'h5_data' does NOT match. Use names like 'my-h5' or 'h5.data' instead.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        (base_dir / "_raw").mkdir()
        # "h5.files" — h5 followed by '.' (non-word char) → word boundary match
        (base_dir / "_raw" / "h5.files").mkdir()
        (base_dir / "_raw" / "excluded-").mkdir()
        # "my-h5" — h5 preceded by '-' → word boundary match
        (base_dir / "my-h5").mkdir()
        (base_dir / "old_h5-").mkdir()
        (base_dir / "regular_dir").mkdir()

        search_dirs = list(_iter_hdf5_search_directories(base_dir))

        all_paths = [str(d[0]) for d in search_dirs]
        assert not any("excluded-" in p for p in all_paths), "excluded- directory should be excluded"
        assert not any("old_h5-" in p for p in all_paths), "old_h5- directory should be excluded"
        assert any("my-h5" in str(d[0]) for d in search_dirs), "my-h5 should be included (\\bh5\\b matches)"
        assert any("h5.files" in str(d[0]) for d in search_dirs), "h5.files inside _raw should be included"


def test_hdf5_raw_subdir_exclusion():
    """Test that subdirectories inside _raw are properly excluded.

    Note: \bh5\b requires h5 as a whole word — underscores are word characters,
    so 'h5_valid' does NOT match. Use names like 'valid-h5' instead.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        raw_dir = base_dir / "_raw"
        raw_dir.mkdir()

        (raw_dir / "valid-h5").mkdir()
        (raw_dir / "h5_excluded-").mkdir()
        (raw_dir / "data.h5").mkdir()
        (raw_dir / "backup-").mkdir()

        search_dirs = list(_iter_hdf5_search_directories(base_dir))
        all_paths = [str(d[0]) for d in search_dirs]

        assert not any("h5_excluded-" in p for p in all_paths), "h5_excluded- should be excluded"
        assert any("valid-h5" in p for p in all_paths), "valid-h5 should be included (\\bh5\\b matches)"
        assert any("data.h5" in p for p in all_paths), "data.h5 should be included (\\bh5\\b matches)"


def test_hdf5_files_excluded_by_ptn_dir_exclude():
    """HDF5 files whose names match ptn_dir_exclude patterns are filtered out by find_hdf5_files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device_dir = Path(tmpdir)

        # Create valid and excluded HDF5 files in device_dir
        (device_dir / "good.h5").touch()
        # matches default pattern r".*-(?:\.|$)" — stem ends with '-'
        (device_dir / "old_data-.h5").touch()
        # matches default pattern "^bad$" — name is exactly "bad" (no extension)
        (device_dir / "bad").touch()
        (device_dir / "data.proc_noAvg.h5").touch()
        # excluded by pattern (stem ends with '-')
        (device_dir / "excluded-.proc_noAvg.h5").touch()
        (device_dir / "regular.txt").touch()

        h5_files = find_hdf5_files(device_dir)
        all_names = [f.name for files in h5_files.values() for f in files]

        assert "good.h5" in all_names, "good.h5 should be found"
        assert "data.proc_noAvg.h5" in all_names, "data.proc_noAvg.h5 should be found"
        assert "old_data-.h5" not in all_names, (
            "old_data-.h5 should be excluded by ptn_dir_exclude pattern"
        )
        assert "excluded-.proc_noAvg.h5" not in all_names, (
            "excluded-.proc_noAvg.h5 should be excluded by ptn_dir_exclude pattern"
        )
        # "bad" has no HDF5 extension so it's not collected regardless
        assert "bad" not in all_names, "bad (no .h5 extension) should not appear"


def test_is_valid_hdf5_file_respects_ptn_dir_exclude():
    """_is_valid_hdf5_file rejects HDF5 files matching ptn_dir_exclude patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)

        good = d / "data.h5"
        good.touch()
        excluded = d / "old-.h5"
        excluded.touch()

        assert _is_valid_hdf5_file(good), "data.h5 should pass validation"
        assert not _is_valid_hdf5_file(excluded), (
            "old-.h5 should be rejected by ptn_dir_exclude"
        )


def test_hdf5_files_in_raw_dir_excluded_by_ptn_dir_exclude():
    """HDF5 files in _raw directory are also filtered by ptn_dir_exclude."""
    with tempfile.TemporaryDirectory() as tmpdir:
        device_dir = Path(tmpdir)
        raw_dir = device_dir / "_raw"
        raw_dir.mkdir()

        (raw_dir / "raw_good.h5").touch()
        (raw_dir / "raw_bad-.h5").touch()

        h5_files = find_hdf5_files(device_dir)
        raw_names = [f.name for f in h5_files["raw"]]

        assert "raw_good.h5" in raw_names, "raw_good.h5 should be found in raw"
        assert "raw_bad-.h5" not in raw_names, (
            "raw_bad-.h5 should be excluded by ptn_dir_exclude pattern"
        )


if __name__ == "__main__":
    test_hdf5_search_directories_exclusion()
    test_hdf5_raw_subdir_exclusion()
    test_hdf5_files_excluded_by_ptn_dir_exclude()
    test_is_valid_hdf5_file_respects_ptn_dir_exclude()
    test_hdf5_files_in_raw_dir_excluded_by_ptn_dir_exclude()
    print("All HDF5 exclusion tests passed!")
