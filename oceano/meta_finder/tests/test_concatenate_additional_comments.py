"""Test _concatenate_additional_comments function preserves bursts_t field.

This test verifies that the fix for preserving the bursts_t field (index 9)
when concatenating additional comments works correctly, especially when the
existing metadata list has fewer elements than expected.
"""

import pytest

from meta_finder.io_info_files import _concatenate_additional_comments


class TestConcatenateAdditionalComments:
    """Test cases for _concatenate_additional_comments function."""

    @pytest.mark.parametrize(
        "metadata_list, comment_index, expected_result, test_description",
        [
            # Test case 1: Existing list with 9 elements (missing bursts_t and comment)
            # This is the bug scenario - bursts_t should be preserved after merge
            (
                ["?", "?", "?", "?", "?", "?", "2016-02-19 14:01:15", "2016-03-18 09:03:45", 150],
                10,  # comment_index
                ["?", "?", "?", "?", "?", "?", "2016-02-19 14:01:15", "2016-03-18 09:03:45", 150, "", ""],
                "9-element list padded to 11 elements with empty bursts_t and comment",
            ),
            # Test case 2: List with 10 elements (missing comment only)
            (
                ["?", "?", "?", "?", "?", "?", "2016-02-19 14:01:15", "2016-03-18 09:03:45", 150, 3600],
                10,  # comment_index
                ["?", "?", "?", "?", "?", "?", "2016-02-19 14:01:15", "2016-03-18 09:03:45", 150, 3600, ""],
                "10-element list padded to 11 elements with empty comment",
            ),
            # Test case 3: Full 11-element list with empty comment
            (
                ["?", "?", "?", "?", "?", "?", "2016-02-19 14:01:15", "2016-03-18 09:03:45", 150, 3600, ""],
                10,  # comment_index
                ["?", "?", "?", "?", "?", "?", "2016-02-19 14:01:15", "2016-03-18 09:03:45", 150, 3600, ""],
                "11-element list with empty comment unchanged",
            ),
            # Test case 4: Full 11-element list with comment and additional fields
            (
                [
                    "?", "?", "?", "?", "?", "?",
                    "2016-02-19 14:01:15", "2016-03-18 09:03:45",
                    150, 3600, "Sample comment",
                    "Additional info 1", "Additional info 2"
                ],
                10,  # comment_index
                [
                    "?", "?", "?", "?", "?", "?",
                    "2016-02-19 14:01:15", "2016-03-18 09:03:45",
                    150, 3600, "Sample comment. Additional info 1. Additional info 2"
                ],
                "11-element list with comment and additional fields concatenated",
            ),
            # Test case 5: List with comment and placeholder additional fields
            (
                [
                    "?", "?", "?", "?", "?", "?",
                    "2016-02-19 14:01:15", "2016-03-18 09:03:45",
                    150, 3600, "Sample comment",
                    "?", "-", ""
                ],
                10,  # comment_index
                [
                    "?", "?", "?", "?", "?", "?",
                    "2016-02-19 14:01:15", "2016-03-18 09:03:45",
                    150, 3600, "Sample comment"
                ],
                "11-element list with comment and placeholder additional fields filtered",
            ),
            # Test case 6: List with None values in additional fields
            (
                [
                    "?", "?", "?", "?", "?", "?",
                    "2016-02-19 14:01:15", "2016-03-18 09:03:45",
                    150, 3600, "Sample comment",
                    None, "Additional info", None
                ],
                10,  # comment_index
                [
                    "?", "?", "?", "?", "?", "?",
                    "2016-02-19 14:01:15", "2016-03-18 09:03:45",
                    150, 3600, "Sample comment. Additional info"
                ],
                "11-element list with None values filtered out",
            ),
            # Test case 7: Empty list
            (
                [],
                10,  # comment_index
                ["", "", "", "", "", "", "", "", "", "", ""],
                "Empty list padded to 11 elements with empty strings",
            ),
            # Test case 8: List shorter than comment_index
            (
                ["?", "?", "?"],
                10,  # comment_index
                ["?", "?", "?", "", "", "", "", "", "", "", ""],
                "3-element list padded to 11 elements with empty strings",
            ),
        ],
    )
    def test_concatenate_additional_comments_preserves_fields(
        self, metadata_list, comment_index, expected_result, test_description
    ):
        """Test that _concatenate_additional_comments preserves all fields including bursts_t.

        Args:
            metadata_list: Input metadata list
            comment_index: Index of comment field
            expected_result: Expected output list
            test_description: Description of the test case
        """
        result = _concatenate_additional_comments(list(metadata_list))

        # Verify the result matches expected
        assert len(result) == len(expected_result), (
            f"{test_description}: Expected length {len(expected_result)}, got {len(result)}")


        for i, (expected, actual) in enumerate(zip(expected_result, result)):
            assert actual == expected, (
                f"{test_description}: Field {i} mismatch - expected {expected!r}, got {actual!r}"
            )

    def test_concatenate_additional_comments_does_not_modify_original(self):
        """Test that the function does not modify the original metadata_list."""
        original_list = ["?", "?", "?", "?", "?", "?", "time1", "time2", 150, 3600, "comment"]
        original_copy = list(original_list)

        _concatenate_additional_comments(original_list)

        # Verify original list was not modified
        assert (
            original_list == original_copy,
            "Original metadata_list should not be modified by the function"
        )

    @pytest.mark.parametrize(
        "metadata_list, comment_index, expected_bursts_t, test_description",
        [
            # Test that bursts_t (index 9) is preserved in various scenarios
            (["?"] * 9, 10, "", "9-element list should have empty bursts_t after padding"),
            (["?"] * 10, 10, "?", "10-element list preserves original bursts_t at index 9"),
            (["?"] * 9 + [3600], 10, 3600, "10-element list with bursts_t value"),
            (
                ["?"] * 8 + [150, 3600],
                10,
                3600,
                "10-element list with burst_dt and bursts_t values",
            ),
        ],
    )
    def test_bursts_t_field_preserved(
        self, metadata_list, comment_index, expected_bursts_t, test_description
    ):
        """Test that the bursts_t field (index 9) is preserved correctly.

        Args:
            metadata_list: Input metadata list
            comment_index: Index of comment field
            expected_bursts_t: Expected value of bursts_t field (index 9)
        """
        result = _concatenate_additional_comments(list(metadata_list))

        # Verify bursts_t field (index 9) is preserved
        assert len(result) > 9, (
            f"Result should have at least 10 elements to include bursts_t field, got {len(result)}"
        )

        actual_bursts_t = result[9]
        assert actual_bursts_t == expected_bursts_t, (
            f"bursts_t field (index 9) mismatch - expected {expected_bursts_t!r}, got {actual_bursts_t!r}"
        )
