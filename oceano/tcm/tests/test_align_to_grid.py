"""
Test for align_to_grid function to ensure min_date is correctly aligned to the grid.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from tcm.spectr_clc import align_to_grid


@pytest.mark.parametrize(
    "grid_origin,min_date_candidate,dt_interval,expected_min_date",
    [
        # Test case 1: grid_origin at midnight, dt_interval=1 hour, min_date_candidate on grid
        (
            pd.Timestamp("2019-01-01T00:00:00"),
            pd.Timestamp("2019-01-01T05:00:00"),
            np.timedelta64(1, 'h'),
            pd.Timestamp("2019-01-01T05:00:00"),
        ),
        # Test case 2: grid_origin at midnight, dt_interval=1 hour, min_date_candidate off grid
        (
            pd.Timestamp("2019-01-01T00:00:00"),
            pd.Timestamp("2019-01-01T05:30:00"),
            np.timedelta64(1, 'h'),
            pd.Timestamp("2019-01-01T06:00:00"),
        ),
        # Test case 3: grid_origin at midnight, dt_interval=2 hours, min_date_candidate off grid
        (
            pd.Timestamp("2019-01-01T00:00:00"),
            pd.Timestamp("2019-01-01T05:30:00"),
            np.timedelta64(2, 'h'),
            pd.Timestamp("2019-01-01T06:00:00"),
        ),
        # Test case 4: grid_origin at midnight, dt_interval=30 minutes, min_date_candidate off grid
        # 5:37 is between 5:30 and 6:00, so should round to 6:00 (next 30-min interval)
        (
            pd.Timestamp("2019-01-01T00:00:00"),
            pd.Timestamp("2019-01-01T05:37:00"),
            np.timedelta64(30, 'm'),
            pd.Timestamp("2019-01-01T06:00:00"),
        ),
        # Test case 5: grid_origin at midnight, dt_interval=1 day, min_date_candidate off grid
        (
            pd.Timestamp("2019-01-01T00:00:00"),
            pd.Timestamp("2019-01-15T12:00:00"),
            np.timedelta64(1, 'D'),
            pd.Timestamp("2019-01-16T00:00:00"),
        ),
        # Test case 6: grid_origin at 3am, dt_interval=1 hour, min_date_candidate off grid
        (
            pd.Timestamp("2019-01-01T03:00:00"),
            pd.Timestamp("2019-01-01T08:30:00"),
            np.timedelta64(1, 'h'),
            pd.Timestamp("2019-01-01T09:00:00"),
        ),
        # Test case 7: Large interval count (from user's example)
        (
            pd.Timestamp("2019-01-01T00:00:00"),
            pd.Timestamp("2020-11-15T00:00:00"),
            np.timedelta64(1, 'h'),
            pd.Timestamp("2020-11-15T00:00:00"),
        ),
    ],
    ids=[
        "on_grid_1h",
        "off_grid_1h",
        "off_grid_2h",
        "off_grid_30m",
        "off_grid_1d",
        "off_grid_1h_custom_origin",
        "large_interval_count",
    ],
)
def test_align_to_grid(
    grid_origin, min_date_candidate, dt_interval, expected_min_date
):
    """Test that align_to_grid correctly aligns min_date to the grid."""
    result = align_to_grid(grid_origin, min_date_candidate, dt_interval)
    assert (
        result == expected_min_date
    ), f"Expected {expected_min_date}, but got {result} for grid_origin={grid_origin}, "
    f"min_date_candidate={min_date_candidate}, dt_interval={dt_interval}"


def test_align_to_grid_result_on_grid():
    """Test that the result is always on the grid (multiple of dt_interval from origin)."""
    grid_origin = pd.Timestamp("2019-01-01T00:00:00")
    dt_interval = np.timedelta64(1, 'h')

    # Test with various min_date candidates
    for hours in [0, 1, 5, 12, 23, 24, 100, 1000]:
        min_date_candidate = grid_origin + pd.Timedelta(hours=hours) + pd.Timedelta(minutes=30)
        result = align_to_grid(grid_origin, min_date_candidate, dt_interval)

        # Verify result is on the grid
        time_diff = result - grid_origin
        intervals = time_diff / pd.Timedelta(dt_interval)
        assert intervals == int(intervals), (
            f"Result {result} is not on the grid: "
            f"{intervals} intervals from origin is not an integer"
        )

        # Verify result >= min_date_candidate
        assert result >= min_date_candidate, (
            f"Result {result} is before min_date_candidate {min_date_candidate}"
        )


def test_user_example_large_interval():
    """Test the specific user example with large interval count."""
    grid_origin = pd.Timestamp("2019-01-01T00:00:00")
    min_date_candidate = pd.Timestamp("2020-11-15T00:00:00")
    dt_interval = np.timedelta64(1, 'h')

    result = align_to_grid(grid_origin, min_date_candidate, dt_interval)

    # Verify result is exactly the candidate (since it's on the grid)
    assert result == min_date_candidate, (
        f"Expected {min_date_candidate}, but got {result}"
    )

    # Verify result is on the grid
    time_diff = result - grid_origin
    intervals = time_diff / pd.Timedelta(dt_interval)
    assert intervals == int(intervals), (
        f"Result {result} is not on the grid: "
        f"{intervals} intervals from origin is not an integer"
    )

    # Verify the large interval count (16416 hours from 2019-01-01 to 2020-11-15)
    assert intervals == 16416, (
        f"Expected 16416 intervals, but got {intervals}"
    )


def test_overlap_shifts_remain_on_grid():
    """
    Test that overlap-shifted intervals remain on the grid.

    When overlap is used (e.g., overlap=0.5), the effective grid step becomes
    (1 - overlap) * dt_interval = 0.5 * dt_interval. This ensures that both
    base intervals and shifted intervals are on the grid.
    """
    grid_origin = pd.Timestamp("2019-01-01T00:00:00")
    min_date_candidate = pd.Timestamp("2019-01-01T05:30:00")
    dt_interval = np.timedelta64(1, 'h')
    overlap = 0.5

    # Align min_date to grid with overlap=0.5
    # Effective grid step is 0.5 * dt_interval = 30 minutes
    # 05:30 is already on the 30-minute grid (11 intervals of 30 min from 00:00)
    min_date = align_to_grid(grid_origin, min_date_candidate, dt_interval, overlap)
    assert min_date == pd.Timestamp("2019-01-01T05:30:00"), (
        f"Expected 05:30:00, got {min_date}"
    )

    # Verify that min_date is on the finer grid (30-minute intervals)
    effective_step = pd.Timedelta(dt_interval) * (1 - overlap)
    time_diff = min_date - grid_origin
    intervals = time_diff / effective_step
    assert intervals == int(intervals), (
        f"min_date {min_date} is not on the finer grid: "
        f"{intervals} intervals of {effective_step} from origin is not an integer"
    )
    assert intervals == 11, f"Expected 11 intervals, got {intervals}"


def test_overlap_with_candidate_off_grid():
    """
    Test overlap behavior when min_date_candidate is not on the finer grid.

    Ensures that alignment works correctly even when the candidate is off-grid.
    """
    grid_origin = pd.Timestamp("2019-01-01T00:00:00")
    min_date_candidate = pd.Timestamp("2019-01-01T05:45:00")
    dt_interval = np.timedelta64(1, 'h')
    overlap = 0.5

    # Align min_date to grid with overlap=0.5
    # Effective grid step is 30 minutes
    # 05:45 should align to 06:00 (12 intervals of 30 min from 00:00)
    min_date = align_to_grid(grid_origin, min_date_candidate, dt_interval, overlap)
    assert min_date == pd.Timestamp("2019-01-01T06:00:00"), (
        f"Expected 06:00:00, got {min_date}"
    )

    # Verify that min_date is on the finer grid
    effective_step = pd.Timedelta(dt_interval) * (1 - overlap)
    time_diff = min_date - grid_origin
    intervals = time_diff / effective_step
    assert intervals == int(intervals), (
        f"min_date {min_date} is not on the finer grid: "
        f"{intervals} intervals of {effective_step} from origin is not an integer"
    )
    assert intervals == 12, f"Expected 12 intervals, got {intervals}"


def test_overlap_zero_same_as_no_overlap():
    """
    Test that overlap=0 produces the same result as no overlap.
    """
    grid_origin = pd.Timestamp("2019-01-01T00:00:00")
    min_date_candidate = pd.Timestamp("2019-01-01T05:30:00")
    dt_interval = np.timedelta64(1, 'h')

    # Without overlap
    min_date_no_overlap = align_to_grid(grid_origin, min_date_candidate, dt_interval)

    # With overlap=0
    min_date_overlap_zero = align_to_grid(
        grid_origin, min_date_candidate, dt_interval, overlap=0
    )

    assert min_date_no_overlap == min_date_overlap_zero, (
        f"overlap=0 should produce same result as no overlap: "
        f"{min_date_no_overlap} vs {min_date_overlap_zero}"
    )


def test_overlap_grid_alignment_with_candidate_on_grid():
    """
    Test overlap behavior when min_date_candidate is already on the grid.

    This ensures that when the candidate is already aligned, the overlap
    shifts work correctly without introducing misalignment.
    """
    grid_origin = pd.Timestamp("2019-01-01T00:00:00")
    min_date_candidate = pd.Timestamp("2019-01-01T05:00:00")  # Already on grid
    dt_interval = np.timedelta64(1, 'h')

    # Align min_date to grid (should remain 05:00)
    min_date = align_to_grid(grid_origin, min_date_candidate, dt_interval)
    assert min_date == min_date_candidate

    # Verify that min_date is on the grid
    time_diff = min_date - grid_origin
    intervals = time_diff / pd.Timedelta(dt_interval)
    assert intervals == 5, f"Expected 5 intervals, got {intervals}"


@pytest.mark.parametrize(
    "grid_origin,min_date_candidate,dt_interval,expected_min_date",
    [
        # Test case 1: Both inputs are datetime.datetime objects (original bug case)
        (
            datetime(2019, 1, 1, 0, 0),
            datetime(2020, 12, 2, 12, 14),
            np.timedelta64(60, 'm'),
            pd.Timestamp("2020-12-02T13:00:00"),
        ),
        # Test case 2: grid_origin is datetime.datetime, min_date_candidate is pd.Timestamp
        (
            datetime(2019, 1, 1, 0, 0),
            pd.Timestamp("2020-12-02T12:14:00"),
            np.timedelta64(60, 'm'),
            pd.Timestamp("2020-12-02T13:00:00"),
        ),
        # Test case 3: grid_origin is pd.Timestamp, min_date_candidate is datetime.datetime
        (
            pd.Timestamp("2019-01-01T00:00:00"),
            datetime(2020, 12, 2, 12, 14),
            np.timedelta64(60, 'm'),
            pd.Timestamp("2020-12-02T13:00:00"),
        ),
        # Test case 4: Both datetime.datetime with smaller interval
        (
            datetime(2019, 1, 1, 0, 0),
            datetime(2019, 1, 1, 5, 30),
            np.timedelta64(1, 'h'),
            pd.Timestamp("2019-01-01T06:00:00"),
        ),
    ],
    ids=[
        "both_datetime_large_interval",
        "mixed_datetime_ts_large_interval",
        "mixed_ts_datetime_large_interval",
        "both_datetime_small_interval",
    ],
)
def test_align_to_grid_datetime_inputs(
    grid_origin, min_date_candidate, dt_interval, expected_min_date
):
    """
    Test that align_to_grid correctly handles datetime.datetime inputs.

    This test covers the bug where datetime.datetime inputs caused the function
    to return grid_origin instead of the correctly aligned timestamp.
    """
    result = align_to_grid(grid_origin, min_date_candidate, dt_interval)
    assert (
        result == expected_min_date
    ), f"Expected {expected_min_date}, but got {result} for grid_origin={grid_origin}, "
    f"min_date_candidate={min_date_candidate}, dt_interval={dt_interval}"

    # Verify result >= min_date_candidate (critical invariant)
    assert result >= pd.Timestamp(min_date_candidate), (
        f"Result {result} is before min_date_candidate {min_date_candidate}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
