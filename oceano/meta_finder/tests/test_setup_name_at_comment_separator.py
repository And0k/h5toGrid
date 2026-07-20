"""Test add_dataset_name with @comment separator after device list.

Covers the case where a directory name like ``190801@i@Schuka`` contains a second
``@`` that introduces a comment/station name.  The device suffix should be
recognised as ``@i``, the comment part ``@Schuka`` discarded, and the setup
name contain only ``{date}@{device_type}``.
"""
import pytest
from pathlib import PurePath
from meta_finder.parse_cruise_dir_name import add_dataset_name


@pytest.mark.parametrize(
    "cruise_dir,device_dir,expected_name,expected_date",
    [
        (
            "B:/Cruises/BalticSea/190801@i@Schuka",
            "B:/Cruises/BalticSea/190801@i@Schuka",
            "190801@i",
            "190801",
        ),
        (
            "B:/Cruises/BalticSea/221103@ib26,28-30",
            "B:/Cruises/BalticSea/221103@ib26,28-30",
            "221103@i",
            "221103",
        ),
        (
            "B:/Cruises/BalticSea/230507_ABP53_inclinometer@i3,4",
            "B:/Cruises/BalticSea/230507_ABP53_inclinometer@i3,4",
            "ABP53",
            "230507",
        ),
    ],
    ids=[
        "190801@i@Schuka should ignore comment after second @",
        "221103@ib26,28-30 reference case no cruise name",
        "230507_ABP53_inclinometer@i3,4 with cruise prefix",
    ],
)
def test_add_dataset_name_at_comment_separator(
    cruise_dir, device_dir, expected_name, expected_date
) -> None:
    """Verify setup name extraction strips trailing ``@comment`` correctly."""
    dataset_name, date_str = add_dataset_name(
        PurePath(device_dir), PurePath(cruise_dir)
    )
    assert dataset_name == expected_name, (
        f"Expected setup_name={expected_name!r} for {device_dir!r}, "
        f"got {dataset_name!r}"
    )
    assert date_str == expected_date, (
        f"Expected date={expected_date!r} for {device_dir!r}, got {date_str!r}"
    )
