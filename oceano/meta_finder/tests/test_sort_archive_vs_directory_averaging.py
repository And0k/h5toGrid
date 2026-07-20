"""
Test that averaging interval takes priority over source type (archive vs directory)
in sort_data_paths, reproducing the bug where bin300s from directory was selected
over bin2s from archive.
"""

import pytest
from pathlib import Path, PurePosixPath

from meta_finder.data_processor import sort_data_paths


@pytest.mark.parametrize(
    ("text_outputs", "json_devices", "expected_order", "description"),
    [
        (
            {
                ("text_output", "210922_1100bin300s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 300,
                },
                ("text_output.zip", "text_output/210922_1100bin2s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 2,
                },
            },
            {"w2"},
            [
                (Path("text_output.zip"), PurePosixPath("text_output/210922_1100bin2s@w02.tsv")),
                (Path("text_output"), PurePosixPath("210922_1100bin300s@w02.tsv")),
            ],
            "bin2s from archive must beat bin300s from directory",
        ),
        (
            {
                ("text_output", "210922_1100bin2s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 2,
                },
                ("text_output.zip", "text_output/210922_1100bin300s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 300,
                },
            },
            {"w2"},
            [
                (Path("text_output"), PurePosixPath("210922_1100bin2s@w02.tsv")),
                (Path("text_output.zip"), PurePosixPath("text_output/210922_1100bin300s@w02.tsv")),
            ],
            "bin2s dir beats bin300s archive: same averaging tier, directory wins tiebreak",
        ),
        (
            {
                ("text_output", "210922_1100bin2s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 2,
                },
                ("text_output.zip", "text_output/210922_1100bin2s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 2,
                },
            },
            {"w2"},
            [
                (Path("text_output"), PurePosixPath("210922_1100bin2s@w02.tsv")),
                (Path("text_output.zip"), PurePosixPath("text_output/210922_1100bin2s@w02.tsv")),
            ],
            "same averaging: directory beats archive on source type tiebreak",
        ),
        (
            {
                ("text_output", "210922_1100bin600s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 600,
                },
                ("text_output.zip", "text_output/210922_1100bin2s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 2,
                },
                ("text_output", "210922_1100bin7200s@w02.tsv"): {
                    "devices": ["w2"], "averaging_interval": 7200,
                },
            },
            {"w2"},
            [
                (Path("text_output.zip"), PurePosixPath("text_output/210922_1100bin2s@w02.tsv")),
                (Path("text_output"), PurePosixPath("210922_1100bin600s@w02.tsv")),
                (Path("text_output"), PurePosixPath("210922_1100bin7200s@w02.tsv")),
            ],
            "bin2s archive first, then bin600s dir, then bin7200s dir",
        ),
    ],
    ids=[
        "archive_bin2s_over_dir_bin300s",
        "dir_bin2s_over_archive_bin300s",
        "same_avg_dir_over_archive",
        "three_way_mixed_sources",
    ],
)
def test_averaging_beats_source_type(text_outputs, json_devices, expected_order, description):
    """Averaging interval must override source type (directory vs archive) in priority."""
    paths_meta = {
        (Path(p), PurePosixPath(rp)): meta for (p, rp), meta in text_outputs.items()
    }
    sorted_items = sort_data_paths(paths_meta, json_devices)
    actual_order = [item[0] for item in sorted_items]
    assert actual_order == expected_order, (
        f"{description}: expected {[tuple(str(x) for x in p) for p in expected_order]}, "
        f"got {[tuple(str(x) for x in p) for p in actual_order]}"
    )


@pytest.mark.parametrize(
    ("avg_values", "expected_order", "description"),
    [
        (
            {"a": 2.0, "b": 300.0, "c": 7200.0},
            ["a", "b", "c"],
            "2s best, then 300s, then 7200s: distance from 2s",
        ),
        (
            {"a": 2.0, "b": 1.0, "c": 300.0},
            ["a", "b", "c"],
            "2s best, 1s second (distance=1), 300s last (distance=298)",
        ),
        (
            {"a": 2.0, "b": 0.5, "c": 300.0, "d": 7200.0},
            ["a", "b", "c", "d"],
            "2s best, 0.5s second (distance=1.5), 300s third (298), 7200s last",
        ),
        (
            {"a": 2.0, "b": 2.0001},
            ["a", "b"],
            "2s beats default 2.0001 (distance 0 vs 0.0001)",
        ),
    ],
    ids=[
        "basic_coarse_ascending",
        "finer_than_2s_lower_than_2s",
        "full_range_ordering",
        "default_vs_explicit_2s",
    ],
)
def test_averaging_distance_from_2s(avg_values, expected_order, description):
    """Averaging priority uses distance from 2s: 2s is optimal."""
    from meta_finder import config
    paths_meta = {
        (Path("dir"), PurePosixPath(f"{name}.tsv")): {
            "devices": ["i1"],
            "averaging_interval": (
                avg if avg != 2.0001 else config.default_text_file_averaging
            ),
        }
        for name, avg in avg_values.items()
    }
    sorted_items = sort_data_paths(paths_meta, {"i1"})
    actual_order = [item[0][1].stem for item in sorted_items]
    assert actual_order == expected_order, (
        f"{description}: expected {expected_order}, got {actual_order}"
    )
