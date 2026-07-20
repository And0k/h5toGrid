"""Test add_dataset_name for the real AI55 production scenario with 3 device dirs."""
import pytest
from pathlib import PurePosixPath
from meta_finder.parse_cruise_dir_name import add_dataset_name


@pytest.mark.parametrize(
    "device_dirs_in_order, expected_names",
    [
        (
            [
                "B:/Cruises/BalticSea/200630_AI55/inclinometer",
                "B:/Cruises/BalticSea/200630_AI55/inclinometer/200630@i11,19",
                "B:/Cruises/BalticSea/200630_AI55/inclinometer/200701@i04,14",
            ],
            {"inclinometer": "AI55", "200630@i11,19": "2006_AI55", "200701@i04,14": "2007_AI55"},
        ),
    ],
    ids=["ai55_production_order"],
)
def test_ai55_production_scenario(device_dirs_in_order, expected_names):
    """Simulate the real production scenario where discover_device_dirs finds 3 dirs
    under cruise 200630_AI55: inclinometer, inclinometer/200630@i11,19, inclinometer/200701@i04,14.
    All should get unique dataset names in used_datasets_paths."""
    cruise_dir = PurePosixPath("B:/Cruises/BalticSea/200630_AI55")
    used_datasets_paths = {}
    datasets_dates = {}

    for device_dir_str in device_dirs_in_order:
        device_dir = PurePosixPath(device_dir_str)
        dataset_name, date_str = add_dataset_name(device_dir, cruise_dir, used_datasets_paths)
        datasets_dates[device_dir] = date_str

    # All 3 device dirs should be in used_datasets_paths (values)
    assert len(used_datasets_paths) == len(device_dirs_in_order), (
        f"Expected {len(device_dirs_in_order)} entries in used_datasets_paths, "
        f"got {len(used_datasets_paths)}: {used_datasets_paths}"
    )

    # Check each device dir got a name
    paths_to_names = {str(v): k for k, v in used_datasets_paths.items()}
    for device_dir_str in device_dirs_in_order:
        device_dir = PurePosixPath(device_dir_str)
        assert device_dir in used_datasets_paths.values(), (
            f"Device dir {device_dir} not found in used_datasets_paths values. "
            f"Present: {list(used_datasets_paths.values())}"
        )
