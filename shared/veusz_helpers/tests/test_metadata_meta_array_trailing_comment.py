"""Regression tests for 11-element ``info_devices`` arrays (trailing comment).

Covers :func:`veusz_helpers.common.metadata._meta_array_to_dict` and
:func:`veusz_helpers.common.metadata.extract_devices_info` with the documented
11-element layout ``[p, b, bd, s, lat, lon, time_st, time_en, burst_dt, bursts_t, comment]``.
"""

import pytest

from veusz_helpers.common import metadata


@pytest.mark.parametrize(
    "array, expected_keys, expected_r, comment",
    [
        (
            ["6", 15.8, 0, "⭡", 54.96487, 20.31636, "2023-06-16T13:20:59", "2023-07-24T09:55:50"],
            {"p", "b", "d", "s", "c", "r"},
            ("2023-06-16T13:20:59", "2023-07-24T09:55:50"),
            "8-element array without bursts/comment keeps p,b,d,s,c,r",
        ),
        (
            ["6", 15.8, 0, "⭡", 54.96487, 20.31636, "2023-06-16T13:20:59", "2023-07-24T09:55:50", 3600.0, 5],
            {"p", "b", "d", "s", "c", "r", "t", "T"},
            ("2023-06-16T13:20:59", "2023-07-24T09:55:50"),
            "10-element array with bursts adds t,T",
        ),
        (
            [
                "7",
                18.0,
                0,
                "⭡",
                54.97144,
                20.32114,
                "2023-06-16T13:30:23",
                "2023-07-24T09:43:31",
                None,
                None,
                "Т.к. часы инклинометра неисправны, время восстанавливалось по запуску",
            ],
            {"p", "b", "d", "s", "c", "r"},
            ("2023-06-16T13:30:23", "2023-07-24T09:43:31"),
            "11-element Kulikovo i03 array: trailing comment is ignored, r intact",
        ),
        (
            [
                "1",
                7.2,
                0,
                "⤉",
                54.94679,
                20.30924,
                "2023-06-16T12:11:20",
                "2023-07-14T18:08:22",
                None,
                None,
                "note",
            ],
            {"p", "b", "d", "s", "c", "r"},
            ("2023-06-16T12:11:20", "2023-07-14T18:08:22"),
            "11-element array with falsy bursts keeps r, drops comment",
        ),
        (
            ["?", "?", "?", "?", "?", "?", "2023-06-16 12:56:40", "2023-06-20 04:15:02"],
            {"p", "b", "d", "s", "c", "r"},
            ("2023-06-16 12:56:40", "2023-06-20 04:15:02"),
            "Kulikovo ip1 unknown depths/coords: NaN-like ? maps to None, no crash",
        ),
    ],
    ids=["8_elem", "10_elem_bursts", "11_elem_comment", "11_elem_comment_no_bursts", "unknown_qmarks"],
)
def test_meta_array_to_dict_trailing_comment(array, expected_keys, expected_r, comment):
    result = metadata._meta_array_to_dict(*array)
    assert set(result) == expected_keys, f"{comment}: keys mismatch - {set(result)!r} != {expected_keys!r}"
    assert result["r"] == expected_r, f"{comment}: r mismatch - {result['r']!r} != {expected_r!r}"
    if "unknown" in comment:
        assert result["d"] is None, f"{comment}: d must be None - got {result['d']!r}"
        assert result["c"] is None, f"{comment}: c must be None - got {result['c']!r}"


@pytest.mark.parametrize(
    "meta, devices, expected_r, comment",
    [
        (
            {
                "i03": [
                    "7",
                    18.0,
                    0,
                    "⭡",
                    54.97144,
                    20.32114,
                    "2023-06-16T13:30:23",
                    "2023-07-24T09:43:31",
                    None,
                    None,
                    "Т.к. часы инклинометра неисправны",
                ]
            },
            ["i03"],
            ("2023-06-16T13:30:23", "2023-07-24T09:43:31"),
            "extract_devices_info maps 11-element i03 array to dict with r",
        ),
    ],
    ids=["extract_11_elem"],
)
def test_extract_devices_info_trailing_comment(meta, devices, expected_r, comment):
    result = metadata.extract_devices_info(meta, devices)
    assert "i03" in result, f"{comment}: i03 missing - got {sorted(result)!r}"
    assert result["i03"]["r"] == expected_r, (
        f"{comment}: r mismatch - {result['i03']['r']!r} != {expected_r!r}"
    )
