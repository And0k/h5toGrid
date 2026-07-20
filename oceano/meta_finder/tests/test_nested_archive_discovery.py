"""Test that archives inside nested subdirectories of text_output are discovered.

Verifies the fix for the issue where ZIP archives inside nested subdirectories
(e.g. text_output/V,P_txt/1D(Time,Pressure,inclination,Vdir).ZIP) were not found
because archive collection used iterdir() (single-level) instead of rglob (recursive).

Real-world scenario that triggered this bug:
    text_output/
    ├── 180418_0710bin600s@i09.csv              # found (direct text file)
    └── V,P_txt/
        ├── 1D(Time,Pressure,inclination,Vdir).ZIP/
        │   └── 180418_1000@i11.txt              # was SKIPPED (nested archive)
        └── txt,vsz.zip/
            └── 180418@ip11.txt                   # was SKIPPED (nested archive)
"""

import zipfile

import pytest
from pathlib import Path, PurePosixPath

from meta_finder.file_finder import extract_devices_from_text_output
from meta_finder.parse_data_file_name import parse_filename_for_metadata


@pytest.fixture
def device_dir_with_nested_archives(tmp_path):
    """Create a device directory with text_output containing nested archives.

    Mirrors the real structure from the bug report, using filenames with
    valid device separators (@, #, _) that parse_filename_for_metadata()
    can extract device IDs from.
    """
    device_dir = tmp_path / "180418_Svetlogorsk"
    text_output = device_dir / "text_output"
    nested_dir = text_output / "V,P_txt"
    nested_dir.mkdir(parents=True)

    # Direct text file in text_output (was already found before the fix)
    (text_output / "180418_0710bin600s@i09.csv").write_text(
        "Time\tInclination_i9\n2018-04-18 07:10:00\t1.0\n"
    )

    # Nested ZIP archive 1 with a file containing device i11
    archive1 = nested_dir / "1D(Time,Pressure,inclination,Vdir).ZIP"
    with zipfile.ZipFile(archive1, "w") as zf:
        zf.writestr(
            "180418_1000@i11.txt",
            "Time\tPressure_i11\n2018-04-18 10:00:00\t1013.25\n",
        )

    # Nested ZIP archive 2 with a file containing device ip11
    archive2 = nested_dir / "txt,vsz.zip"
    with zipfile.ZipFile(archive2, "w") as zf:
        zf.writestr(
            "180418@ip11.txt",
            "Time\tInclination_ip11\n2018-04-18 10:00:00\t0.5\n",
        )

    return device_dir


@pytest.fixture
def device_dir_with_deeply_nested_archives(tmp_path):
    """Create a device directory with archives nested 2+ levels deep."""
    device_dir = tmp_path / "180418_Svetlogorsk"
    text_output = device_dir / "text_output"
    deep_dir = text_output / "V,P_txt" / "pressure_data"
    deep_dir.mkdir(parents=True)

    archive = deep_dir / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "180418_1200@i15.csv",
            "Time\tInclination_i15\n2018-04-18 12:00:00\t2.0\n",
        )

    return device_dir


class TestNestedArchiveDiscovery:
    """Verify archives in nested text_output subdirectories are discovered."""

    def test_all_nested_archive_files_found(self, device_dir_with_nested_archives):
        """All text files from nested ZIPs must be discovered alongside direct text files."""
        result = extract_devices_from_text_output(device_dir_with_nested_archives)

        # Device i9 from the direct CSV file
        assert "i9" in result, "Device i9 from direct text_output CSV should be found"
        assert len(result["i9"]) == 1, (
            f"Device i9 should have exactly 1 file, got {len(result['i9'])}"
        )

        # Device i11 from the nested ZIP archive
        assert "i11" in result, (
            "Device i11 from nested ZIP (1D(Time,Pressure,inclination,Vdir).ZIP) should be found"
        )
        assert len(result["i11"]) == 1, (
            f"Device i11 should have exactly 1 file, got {len(result['i11'])}"
        )

        # Device ip11 from the second nested ZIP archive
        assert "ip11" in result, (
            "Device ip11 from nested ZIP (txt,vsz.zip) should be found"
        )
        assert len(result["ip11"]) == 1, (
            f"Device ip11 should have exactly 1 file, got {len(result['ip11'])}"
        )

    def test_no_duplicate_files_from_nested_and_direct_scan(
        self, device_dir_with_nested_archives
    ):
        """Files must not appear twice even though rglob covers subdirectories."""
        result = extract_devices_from_text_output(device_dir_with_nested_archives)

        for device, files in result.items():
            # Deduplicate by (dir_path, rel_path) tuple
            unique = set((str(dp), str(rp)) for dp, rp in files)
            assert len(unique) == len(files), (
                f"Device {device} has duplicate entries: "
                f"{len(files)} total but only {len(unique)} unique"
            )

    def test_archive_rel_paths_are_relative_to_archive(self, device_dir_with_nested_archives):
        """Relative paths for files inside archives should be relative to the archive."""
        result = extract_devices_from_text_output(device_dir_with_nested_archives)

        # i11 file is inside the nested ZIP
        _, i11_rel = result["i11"][0]
        assert i11_rel == PurePosixPath("180418_1000@i11.txt"), (
            f"Expected rel_path '180418_1000@i11.txt', got {i11_rel}"
        )

        # ip11 file is inside the second nested ZIP
        _, ip11_rel = result["ip11"][0]
        assert ip11_rel == PurePosixPath("180418@ip11.txt"), (
            f"Expected rel_path '180418@ip11.txt', got {ip11_rel}"
        )

    def test_direct_file_rel_path_includes_no_subdir(self, device_dir_with_nested_archives):
        """Direct text file rel_path should not include text_output prefix."""
        result = extract_devices_from_text_output(device_dir_with_nested_archives)

        _, i9_rel = result["i9"][0]
        assert i9_rel == PurePosixPath("180418_0710bin600s@i09.csv"), (
            f"Expected rel_path '180418_0710bin600s@i09.csv', got {i9_rel}"
        )

    def test_deeply_nested_archive_discovered(self, device_dir_with_deeply_nested_archives):
        """Archives nested 2+ levels deep must also be discovered."""
        result = extract_devices_from_text_output(device_dir_with_deeply_nested_archives)

        assert "i15" in result, (
            "Device i15 from archive 2 levels deep (V,P_txt/pressure_data/data.zip) should be found"
        )
        # dir_path should point to the archive, rel_path to the file within it
        dir_path, rel_path = result["i15"][0]
        assert dir_path.suffix.lower() == ".zip", (
            f"dir_path should be the archive file, got {dir_path}"
        )
        assert rel_path == PurePosixPath("180418_1200@i15.csv"), (
            f"Expected rel_path '180418_1200@i15.csv', got {rel_path}"
        )


class TestOptionalSeparatorInFilenameParsing:
    """Verify device separator is optional when valid type/model + number exist."""

    @pytest.mark.parametrize(
        "filename, expected_device",
        [
            # Separator present — baseline cases that always worked
            pytest.param("180418_1000@i11.txt", "i11", id="at_separator_i11"),
            pytest.param("180418_1000#i11.txt", "i11", id="hash_separator_i11"),
            pytest.param("180418_1000_i11.txt", "i11", id="underscore_separator_i11"),
            pytest.param("180418_1000@ip11.txt", "ip11", id="at_separator_ip11"),
            pytest.param("180418_1000@inclPres11.txt", "ip11", id="at_separator_inclPres11"),
            # No separator — device type directly follows time range
            pytest.param("180418_1000i11.txt", "i11", id="no_separator_i11"),
            pytest.param("180418_1000w05.txt", "w5", id="no_separator_w05"),
            pytest.param("180418_1000incl11.txt", "i11", id="no_separator_incl11"),
            # No separator with ip type (inkl prefix variant)
            pytest.param("180418_1000ip11.txt", "ip11", id="no_separator_ip11"),
            pytest.param("180418_1000inkl11.txt", "i11", id="no_separator_inkl11"),
            # No separator with model — Pres model yields 'p' prefix making ip11
            pytest.param("180418_1000inclPres11.txt", "ip11", id="no_separator_inclPres11"),
            pytest.param("180418_1000iPres11.txt", "ip11", id="no_separator_iPres11"),
            # No separator with bin interval before device
            pytest.param("180418_1000bin600s@i11.txt", "i11", id="at_separator_bin_then_i11"),
            pytest.param("180418_1000bin600si11.txt", "i11", id="no_separator_bin_then_i11"),
            pytest.param("180418_1000bin600sip11.txt", "ip11", id="no_separator_bin_then_ip11"),
        ],
    )
    def test_filename_parsed_with_expected_device(self, filename, expected_device):
        """Filename must be parsed with the expected device ID regardless of separator."""
        result = parse_filename_for_metadata(filename)
        assert result, f"Filename '{filename}' should be parsed successfully"
        assert result["devices"] == [expected_device], (
            f"Filename '{filename}': "
            f"expected devices=['{expected_device}'], got devices={result['devices']}"
        )

    @pytest.mark.parametrize(
        "filename",
        [
            pytest.param("180418_1000.txt", id="no_device_only_datetime_and_time"),
            pytest.param("random_file.txt", id="non_matching_filename"),
        ],
    )
    def test_filename_without_device_gets_star(self, filename):
        """Filename without device info must yield devices=['*'] or be unparseable."""
        result = parse_filename_for_metadata(filename)
        if result:  # only check if filename is parseable at all
            assert result["devices"] == ["*"], (
                f"Filename '{filename}': expected devices=['*'], got devices={result['devices']}"
            )

    @pytest.mark.parametrize(
        "filename, expected_device",
        [
            pytest.param("180418_1000i11.txt", "i11", id="no_separator_nested_archive_i11"),
            pytest.param("180418_1000w03.txt", "w3", id="no_separator_nested_archive_w03"),
            pytest.param("180418_1000ip11.txt", "ip11", id="no_separator_nested_archive_ip11"),
            pytest.param("180418_1000inclPres11.txt", "ip11", id="no_separator_nested_archive_inclPres11"),
        ],
    )
    def test_no_separator_files_in_nested_archive_discovered(
        self, tmp_path, filename, expected_device
    ):
        """Files without separator inside nested archives must be discovered as correct device."""
        device_dir = tmp_path / "180418_Svetlogorsk"
        text_output = device_dir / "text_output"
        nested_dir = text_output / "subdir"
        nested_dir.mkdir(parents=True)

        archive = nested_dir / "data.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(filename, f"Time\tInclination_{expected_device}\n")

        result = extract_devices_from_text_output(device_dir)

        assert expected_device in result, (
            f"Device {expected_device} from nested archive should be found"
        )
        assert len(result[expected_device]) == 1, (
            f"Device {expected_device} should have exactly 1 file"
        )
        _, rel_path = result[expected_device][0]
        assert rel_path == PurePosixPath(filename), (
            f"Expected rel_path '{filename}', got {rel_path}"
        )
