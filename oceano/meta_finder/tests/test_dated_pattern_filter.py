"""Unit tests for dated-part pattern mask building and first/last file filtering.

Tests cover:
- _build_dated_mask_pattern: regex construction from filenames with dated prefixes
- _find_dated_files_with_same_pattern_generic: min/max file selection
- _PatternMinMaxTracker: single-pass inline min/max tracking per pattern group
"""

import re
from pathlib import Path, PurePosixPath

import pytest

from meta_finder.data_proc_funcs import (
    _build_dated_mask_pattern,
    _find_dated_files_with_same_pattern_generic,
)
from meta_finder.file_finder import _PatternMinMaxTracker


# ---------------------------------------------------------------------------
# _build_dated_mask_pattern
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("filename", "should_match", "should_reject", "test_id"),
    [
        (
            "180418_1000inclPres11.txt",
            ["180418_1000inclPres11.txt", "180530_0000inclPres11.txt", "180418_0710inclPres11.txt"],
            ["180418_1000inclPres14.txt", "180418@ip11.txt", "180418_1000inclPres11.csv"],
            "inclPres11_pattern",
        ),
        (
            "180418@ip11.txt",
            ["180418@ip11.txt", "180530@ip11.txt", "180419@ip11.txt"],
            ["180418@ip14.txt", "180418_1000inclPres11.txt"],
            "at_ip11_pattern",
        ),
        (
            "180418_0710bin600s@i09.csv",
            ["180418_0710bin600s@i09.csv", "180520_1200bin600s@i09.csv"],
            ["180418_0710bin600s@i10.csv", "180418_0710bin600s@i09.txt"],
            "bin600s_i09_pattern",
        ),
        (
            "180510_1545inclPres14.txt",
            ["180510_1545inclPres14.txt", "180511_0000inclPres14.txt", "180530_0000inclPres14.txt"],
            ["180510_1545inclPres11.txt"],
            "inclPres14_pattern",
        ),
        (
            "180418_0710-18_2304inclPres11.txt",
            ["180418_0710-18_2304inclPres11.txt", "180520_0000-20_1200inclPres11.txt"],
            ["180418_0710inclPres11.txt"],
            "time_range_pattern",
        ),
        (
            "config.txt",
            ["config.txt"],
            ["config.csv", "other.txt"],
            "no_dated_prefix",
        ),
    ],
    ids=lambda p: p if isinstance(p, str) else "",
)
def test_build_dated_mask_pattern(filename, should_match, should_reject, test_id):
    """Verify mask pattern matches expected filenames and rejects others."""
    pattern = _build_dated_mask_pattern(filename)
    for name in should_match:
        assert pattern.match(name), (
            f"{test_id}: pattern should match {name!r}, built from {filename!r}"
        )
    for name in should_reject:
        assert not pattern.match(name), (
            f"{test_id}: pattern should reject {name!r}, built from {filename!r}"
        )


# ---------------------------------------------------------------------------
# _find_dated_files_with_same_pattern_generic
# ---------------------------------------------------------------------------

def _pp(name: str) -> PurePosixPath:
    """Shorthand to create PurePosixPath from filename only."""
    return PurePosixPath(name)


@pytest.mark.parametrize(
    ("base_name", "paths", "expected_names", "test_id"),
    [
        (
            "180418_1000inclPres11.txt",
            [_pp(f"1804{n:02d}_0000inclPres11.txt") for n in range(18, 31)],
            ["180418_0000inclPres11.txt", "180430_0000inclPres11.txt"],
            "ip11_21_files_to_first_last",
        ),
        (
            "180510_1545inclPres14.txt",
            [_pp(f"1805{n:02d}_0000inclPres14.txt") for n in range(10, 31)],
            ["180510_0000inclPres14.txt", "180530_0000inclPres14.txt"],
            "ip14_21_files_to_first_last",
        ),
        (
            "180418@ip11.txt",
            [_pp("180418@ip11.txt")],
            ["180418@ip11.txt"],
            "single_file_returns_itself",
        ),
        (
            "180418_1000inclPres11.txt",
            [_pp("180418_1000inclPres11.txt"), _pp("180419_0000inclPres11.txt")],
            ["180418_1000inclPres11.txt", "180419_0000inclPres11.txt"],
            "two_files_returns_both",
        ),
        (
            "180418_1000inclPres11.txt",
            [_pp("180418_1000inclPres14.txt"), _pp("180419_1000inclPres14.txt")],
            [],
            "no_matching_files_returns_empty",
        ),
        (
            "nonexistent.txt",
            [_pp("nonexistent.txt")],
            ["nonexistent.txt"],
            "fallback_exact_match",
        ),
    ],
    ids=lambda p: p if isinstance(p, str) else "",
)
def test_find_dated_files_generic(base_name, paths, expected_names, test_id):
    """Verify first/last selection returns expected filenames."""
    result = _find_dated_files_with_same_pattern_generic(base_name, paths)
    result_names = [p.name for p in result]
    assert result_names == expected_names, (
        f"{test_id}: expected {expected_names}, got {result_names}"
    )


# ---------------------------------------------------------------------------
# _PatternMinMaxTracker
# ---------------------------------------------------------------------------

# Shared dummy base_dir for tracker tests
_DUMMY_DIR = Path("/tmp/test_text_output")


class TestPatternMinMaxTracker:
    """Tests for the single-pass inline min/max tracker."""

    def test_single_file(self):
        """Single file is kept as both min and max (returned once)."""
        tracker = _PatternMinMaxTracker()
        tracker.add(_DUMMY_DIR, PurePosixPath("180418_0000inclPres11.txt"), {"devices": ["ip11"]})
        result = tracker.result()
        assert len(result) == 1, f"Single file should produce 1 result, got {len(result)}"
        assert result[0] == (_DUMMY_DIR, PurePosixPath("180418_0000inclPres11.txt"))

    def test_two_files_same_pattern(self):
        """Two files with same pattern: both kept."""
        tracker = _PatternMinMaxTracker()
        tracker.add(_DUMMY_DIR, PurePosixPath("180418_0000inclPres11.txt"), {"devices": ["ip11"]})
        tracker.add(_DUMMY_DIR, PurePosixPath("180430_0000inclPres11.txt"), {"devices": ["ip11"]})
        result = tracker.result()
        assert len(result) == 2, f"Two files should produce 2 results, got {len(result)}"
        names = [r[1].name for r in result]
        assert names == ["180418_0000inclPres11.txt", "180430_0000inclPres11.txt"]

    def test_many_files_keeps_first_last(self):
        """21 files with same pattern: only first and last kept."""
        tracker = _PatternMinMaxTracker()
        for n in range(18, 39):
            tracker.add(
                _DUMMY_DIR,
                PurePosixPath(f"1804{n:02d}_0000inclPres11.txt"),
                {"devices": ["ip11"]},
            )
        result = tracker.result()
        assert len(result) == 2, f"21 same-pattern files should produce 2 results, got {len(result)}"
        names = [r[1].name for r in result]
        assert names[0] == "180418_0000inclPres11.txt", f"First should be earliest, got {names[0]}"
        assert names[1] == "180438_0000inclPres11.txt", f"Last should be latest, got {names[1]}"
        assert tracker.total_seen == 21

    def test_multiple_pattern_groups(self):
        """Files with different suffixes form separate pattern groups."""
        tracker = _PatternMinMaxTracker()
        # Group 1: inclPres11
        tracker.add(_DUMMY_DIR, PurePosixPath("180418_0000inclPres11.txt"), {"devices": ["ip11"]})
        tracker.add(_DUMMY_DIR, PurePosixPath("180425_0000inclPres11.txt"), {"devices": ["ip11"]})
        tracker.add(_DUMMY_DIR, PurePosixPath("180430_0000inclPres11.txt"), {"devices": ["ip11"]})
        # Group 2: @ip11
        tracker.add(_DUMMY_DIR, PurePosixPath("180418@ip11.txt"), {"devices": ["ip11"]})
        tracker.add(_DUMMY_DIR, PurePosixPath("180510@ip11.txt"), {"devices": ["ip11"]})

        result = tracker.result()
        assert len(result) == 4, (
            f"2 pattern groups × 2 each = 4 results, got {len(result)}"
        )
        names = [r[1].name for r in result]
        assert "180418_0000inclPres11.txt" in names
        assert "180430_0000inclPres11.txt" in names
        assert "180418@ip11.txt" in names
        assert "180510@ip11.txt" in names
        assert tracker.total_seen == 5

    def test_out_of_order_insertion(self):
        """Files added in non-chronological order still produce correct min/max."""
        tracker = _PatternMinMaxTracker()
        tracker.add(_DUMMY_DIR, PurePosixPath("180425_0000inclPres11.txt"), {"devices": ["ip11"]})
        tracker.add(_DUMMY_DIR, PurePosixPath("180418_0000inclPres11.txt"), {"devices": ["ip11"]})
        tracker.add(_DUMMY_DIR, PurePosixPath("180430_0000inclPres11.txt"), {"devices": ["ip11"]})
        tracker.add(_DUMMY_DIR, PurePosixPath("180420_0000inclPres11.txt"), {"devices": ["ip11"]})

        result = tracker.result()
        assert len(result) == 2
        names = [r[1].name for r in result]
        assert names[0] == "180418_0000inclPres11.txt", f"Min should be earliest, got {names[0]}"
        assert names[1] == "180430_0000inclPres11.txt", f"Max should be latest, got {names[1]}"

    def test_total_seen_counter(self):
        """total_seen counts all add() calls, not just kept entries."""
        tracker = _PatternMinMaxTracker()
        for n in range(18, 31):
            tracker.add(_DUMMY_DIR, PurePosixPath(f"1804{n:02d}_0000inclPres11.txt"), {})
        assert tracker.total_seen == 13
        assert len(tracker.result()) == 2

    def test_mixed_device_patterns(self):
        """Files for different devices (ip11 vs ip14) tracked independently."""
        tracker_ip11 = _PatternMinMaxTracker()
        tracker_ip14 = _PatternMinMaxTracker()

        for n in range(18, 31):
            tracker_ip11.add(_DUMMY_DIR, PurePosixPath(f"1804{n:02d}_0000inclPres11.txt"), {})
        for n in range(10, 31):
            tracker_ip14.add(_DUMMY_DIR, PurePosixPath(f"1805{n:02d}_0000inclPres14.txt"), {})

        r11 = tracker_ip11.result()
        r14 = tracker_ip14.result()
        assert len(r11) == 2, f"ip11 tracker should have 2 results, got {len(r11)}"
        assert len(r14) == 2, f"ip14 tracker should have 2 results, got {len(r14)}"
        assert r11[0][1].name == "180418_0000inclPres11.txt"
        assert r11[1][1].name == "180430_0000inclPres11.txt"
        assert r14[0][1].name == "180510_0000inclPres14.txt"
        assert r14[1][1].name == "180530_0000inclPres14.txt"

    def test_base_dir_preserved_per_entry(self):
        """Each result entry carries its own base_dir, not overwritten by later entries."""
        dir_a = Path("/tmp/text_output")
        dir_b = Path("/tmp/archive.zip")

        tracker = _PatternMinMaxTracker()
        tracker.add(dir_a, PurePosixPath("180418_0000inclPres11.txt"), {})
        tracker.add(dir_b, PurePosixPath("180430_0000inclPres11.txt"), {})

        result = tracker.result()
        assert len(result) == 2
        assert result[0] == (dir_a, PurePosixPath("180418_0000inclPres11.txt"))
        assert result[1] == (dir_b, PurePosixPath("180430_0000inclPres11.txt"))
