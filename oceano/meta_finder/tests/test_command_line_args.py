import pytest
import sys
from unittest.mock import patch
from meta_finder.config import Config


class TestCommandLineArgs:
    """Test command line argument parsing for the metadata processor."""

    @pytest.mark.parametrize(
        "args, expected_create_info_files, expected_process_metadata, comment",
        [
            (["--create-info-files", "--no-process-metadata"], True, False,
             "create-info-files with no-process-metadata should disable processing"),
            (["--no-create-info-files", "--process-metadata"], False, True,
             "no-create-info-files with process-metadata should enable processing only"),
            (["--create-info-files", "--process-metadata"], True, True,
             "both flags enabled should work normally"),
            ([], False, True,  # defaults are create_info_files=False, process_metadata=True
             "no flags should use defaults - no create files, process metadata"),
            (["--create-info-files"], True, True,
             "only create-info-files flag should enable creation, keep processing enabled by default"),
            (["--no-process-metadata"], False, False,
             "only no-process-metadata flag should disable processing, keep creation disabled by default"),
        ],
        ids=[
            "create_info_no_process",
            "no_create_process",
            "both_enabled",
            "no_flags_defaults",
            "create_only",
            "process_disabled"
        ]
    )
    def test_process_metadata_flag_parsing(self, args, expected_create_info_files, expected_process_metadata, comment):
        """Test that process_metadata flag is correctly parsed with --no- prefix."""
        # Add dummy arguments required by argparse but not relevant to our test
        test_args = ["script_name"] + args

        with patch.object(sys, 'argv', test_args):
            config = Config.from_args()

        assert config.create_info_files == expected_create_info_files, (
            f"create_info_files flag test failed: expected {expected_create_info_files}, "
            f"got {config.create_info_files} when args were {args}. {comment}"
        )
        assert config.process_metadata == expected_process_metadata, (
            f"process_metadata flag test failed: expected {expected_process_metadata}, "
            f"got {config.process_metadata} when args were {args}. {comment}"
        )

    def test_no_process_metadata_flag_exists(self):
        """Test that the --no-process-metadata flag can be used without error."""
        test_args = ["script_name", "--no-process-metadata"]

        with patch.object(sys, 'argv', test_args):
            config = Config.from_args()

        assert config.process_metadata is False, (
            "process_metadata should be False when --no-process-metadata flag is used"
        )

    def test_no_create_info_files_flag_exists(self):
        """Test that the --no-create-info-files flag can be used without error."""
        test_args = ["script_name", "--no-create-info-files"]

        with patch.object(sys, 'argv', test_args):
            config = Config.from_args()

        assert config.create_info_files is False, (
            "create_info_files should be False when --no-create-info-files flag is used"
        )