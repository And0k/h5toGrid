"""
Test burst detection edge cases to verify fix for zero burst_dt issue.
"""
import pytest
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meta_finder.data_proc_funcs import _extract_burst_info_from_lines


@pytest.mark.parametrize(
    "lines, averaging_interval, expected_bursts_t, expected_burst_dt, description",
    [
        # Test case 1: Continuous data (no gaps)
        (
            ["Time\tValue\n", "2016-02-19 14:02:30\t1\n", "2016-02-19 14:02:32\t2\n", "2016-02-19 14:02:34\t3\n"],
            2,
            "-",
            "-",
            "continuous data with no gaps should return dash for burst values"
        ),
        # Test case 2: Single gap (not enough for burst detection)
        (
            ["Time\tValue\n", "2016-02-19 14:02:30\t1\n", "2016-02-19 14:02:32\t2\n",
             "2016-02-19 14:10:30\t3\n", "2016-02-19 14:10:32\t4\n"],
            2,
            "-",
            "-",
            "single gap should return dash for burst values"
        ),
        # Test case 3: Two gaps with valid burst (normal case)
        (
            ["Time\tValue\n",
             "2016-02-19 14:02:30\t1\n",  # Start of first burst
             "2016-02-19 14:02:32\t2\n",  # Within first burst (diff: 2s < 4s threshold)
             "2016-02-19 14:02:34\t3\n",  # Within first burst (diff: 2s < 4s threshold)
             "2016-02-19 14:05:30\t4\n",  # End of first burst (diff: 176s > 4s threshold - GAP!)
             "2016-02-19 14:05:32\t5\n",  # Start of second burst (diff: 2s < 4s threshold)
             "2016-02-19 14:05:34\t6\n",  # Within second burst (diff: 2s < 4s threshold)
             "2016-02-19 14:10:30\t7\n",  # End of second burst (diff: 296s > 4s threshold - GAP!)
             "2016-02-19 14:10:32\t8\n",  # Start of third burst (diff: 2s < 4s threshold)
            ],
            2,
            180,  # bursts_t = 14:05:30 - 14:02:30 = 180s (start of burst 2 - start of burst 1)
            4,  # burst_dt = 14:02:34 - 14:02:30 = 4s (duration of first burst)
            "two valid gaps should calculate burst_t and burst_dt correctly"
        ),
        # Test case 4: Adjacent gaps (burst_dt = 0) - the bug case
        (
            ["Time\tValue\n",
             "2016-02-19 14:02:30\t1\n",  # Start of first burst
             "2016-02-19 14:05:30\t2\n",  # End of first burst (gap)
             "2016-02-19 14:05:30\t3\n",  # Start of second burst (SAME timestamp as end of first!)
             "2016-02-19 14:10:30\t4\n",  # End of second burst
            ],
            2,
            "-",  # bursts_t should be "-" because burst_dt is 0 (adjacent gaps)
            "-",  # burst_dt should be "-" because gaps are adjacent (0 duration)
            "adjacent gaps with zero burst duration should return dash for both values"
        ),
        # Test case 5: Overlapping gaps (burst_dt negative)
        (
            ["Time\tValue\n",
             "2016-02-19 14:02:30\t1\n",  # Start of first burst
             "2016-02-19 14:05:30\t2\n",  # End of first burst
             "2016-02-19 14:05:20\t3\n",  # Start of second burst (BEFORE end of first burst!)
             "2016-02-19 14:10:30\t4\n",  # End of second burst
            ],
            2,
            "-",  # bursts_t should be "-" because burst_dt is negative (overlapping gaps)
            "-",  # burst_dt should be "-" because gaps overlap (negative duration)
            "overlapping gaps should return dash for both values"
        ),
    ],
    ids=[
        "continuous_no_gaps",
        "single_gap_only",
        "valid_two_gaps",
        "adjacent_gaps_zero_duration",
        "overlapping_gaps_negative_duration",
    ]
)
def test_burst_detection_edge_cases(
    lines, averaging_interval, expected_bursts_t, expected_burst_dt, description
):
    """Test burst detection with various edge cases."""
    bursts_t, burst_dt = _extract_burst_info_from_lines(lines, averaging_interval)

    # Verify burst_t
    if expected_bursts_t == "-":
        assert bursts_t == "-", (
            f"{description}: expected bursts_t='-', got bursts_t={bursts_t}"
        )
    else:
        assert bursts_t == expected_bursts_t, (
            f"{description}: expected bursts_t={expected_bursts_t}, got bursts_t={bursts_t}"
        )

    # Verify burst_dt
    if expected_burst_dt == "-":
        assert burst_dt == "-", (
            f"{description}: expected burst_dt='-', got burst_dt={burst_dt}"
        )
    else:
        assert burst_dt == expected_burst_dt, (
            f"{description}: expected burst_dt={expected_burst_dt}, got burst_dt={burst_dt}"
        )


def test_insufficient_lines():
    """Test that insufficient lines returns default values."""
    lines = ["Time\tValue\n", "2016-02-19 14:02:30\t1\n"]
    bursts_t, burst_dt = _extract_burst_info_from_lines(lines, 2)

    assert bursts_t == "-", "Insufficient lines should return '-' for bursts_t"
    assert burst_dt == "-", "Insufficient lines should return '-' for burst_dt"


def test_only_header():
    """Test that only header returns default values."""
    lines = ["Time\tValue\n"]
    bursts_t, burst_dt = _extract_burst_info_from_lines(lines, 2)

    assert bursts_t == "-", "Only header should return '-' for bursts_t"
    assert burst_dt == "-", "Only header should return '-' for burst_dt"
