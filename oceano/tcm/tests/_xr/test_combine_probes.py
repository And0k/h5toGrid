"""Tests for _merge_groups_to_combined / _combined_group_is_current idempotency.

Re-running _merge_groups_to_combined with unchanged per-probe data must not
rewrite the combined output.  The guard is ``_combined_group_is_current``,
which reads the combined time-range from the combined file and the per-probe
time-ranges from the per-probe source file — they may live in different files
(e.g. ``.proc_Avg.nc`` vs ``.proc.nc``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tcm import _constants, policy
from tcm._xr import storage
from tcm.processing import _combined_group_is_current, _merge_groups_to_combined


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# _merge_groups_to_combined requires ≥ 2 groups to merge
_PCIDS_2 = ["i90", "i67"]
_BIN_S = 600


def _make_binned_ds(
    n: int = 50,
    *,
    start: str = "2024-01-01",
    freq: str = "600s",
    seed: int = 42,
) -> xr.Dataset:
    """Build a minimal binned Dataset with velocity/inclination data."""
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "u": ("time", rng.normal(0, 1, n)),
            "v": ("time", rng.normal(0, 1, n)),
            "inclination": ("time", rng.normal(0, 5, n)),
            "direction": ("time", rng.normal(180, 30, n)),
        },
        coords={"time": pd.date_range(start, periods=n, freq=freq)},
    )


def _write_per_probes(
    avg_path: Path,
    pcids: list[str],
    bin_s: int,
    n: int = 50,
    *,
    offset_minutes: int = 0,
    seed: int = 42,
) -> None:
    """Write per-probe binned groups into *avg_path* via store_processed_incremental."""
    for i, pcid in enumerate(pcids):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=offset_minutes + i * 10)
        ds = _make_binned_ds(n, start=str(start), seed=seed)
        storage.store_processed_incremental(
            ds,
            avg_path,
            group=f"{pcid}bin{bin_s}s",
            filter_params="test params",
        )


def _merge(
    avg_path: Path,
    combined_path: Path,
    pcids: list[str],
    bin_s: int,
) -> None:
    """Call _merge_groups_to_combined with standard args."""
    _merge_groups_to_combined(
        avg_path,
        pcids,
        combined_group=f"i_bin{bin_s}s",
        label=f"bin{bin_s}s",
        bin_s=bin_s,
        combined_nc_path=combined_path,
    )


def _mtime_ns(path: Path) -> int:
    """Return file modification time in nanoseconds."""
    return path.stat().st_mtime_ns


@pytest.fixture(autouse=True)
def _init_policy():
    """Initialize I/O policy for tests."""

    class _P:
        def allow_nc(self, _):
            return True

        def require_nc(self, _):
            pass

        h5 = True

    policy._io.set(_P())


# ---------------------------------------------------------------------------
# _combined_group_is_current
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestCombinedGroupIsCurrent:
    """Verify _combined_group_is_current reads per-probe data from the correct file.

    After the fix, the function signature is
    ``(combined_path, per_probe_path, combined_grp, pcids, bin_s)``.
    Before the fix it only takes ``(nc_path, combined_grp, pcids, bin_s)``
    and reads per-probe groups from the combined file (wrong).
    """

    @pytest.mark.parametrize(
        "pcids",
        [pytest.param(_PCIDS_2, id="2probes")],
    )
    def test_true_when_combined_covers_per_probes(self, tmp_path, pcids):
        """Combined group time-range covers all per-probe ranges → True."""
        avg, comb = tmp_path / "t.proc_Avg.nc", tmp_path / "t.proc.nc"
        _write_per_probes(avg, pcids, _BIN_S)
        _merge(avg, comb, pcids, _BIN_S)

        assert _combined_group_is_current(comb, avg, f"i_bin{_BIN_S}s", pcids, _BIN_S) is True, (
            f"pcids={pcids}: combined covers all per-probe ranges"
        )

    @pytest.mark.parametrize(
        "pcids",
        [pytest.param(_PCIDS_2, id="2probes")],
    )
    def test_false_when_per_probe_extended(self, tmp_path, pcids):
        """Per-probe data extends beyond combined → stale → False."""
        avg, comb = tmp_path / "t.proc_Avg.nc", tmp_path / "t.proc.nc"
        _write_per_probes(avg, pcids, _BIN_S)
        _merge(avg, comb, pcids, _BIN_S)

        # Extend per-probe with later timestamps
        _write_per_probes(avg, pcids, _BIN_S, n=10, offset_minutes=100_000, seed=99)

        assert _combined_group_is_current(comb, avg, f"i_bin{_BIN_S}s", pcids, _BIN_S) is False, (
            f"pcids={pcids}: per-probe data extends beyond combined"
        )

    def test_false_when_combined_missing(self, tmp_path):
        """Combined file doesn't exist → False."""
        avg = tmp_path / "t.proc_Avg.nc"
        _write_per_probes(avg, _PCIDS_2, _BIN_S)

        assert (
            _combined_group_is_current(
                tmp_path / "nonexistent.nc",
                avg,
                "i_bin600s",
                _PCIDS_2,
                _BIN_S,
            )
            is False
        ), "Combined file missing"

    def test_false_when_combined_group_missing(self, tmp_path):
        """Combined file exists but no matching group → False."""
        avg, comb = tmp_path / "t.proc_Avg.nc", tmp_path / "t.proc.nc"
        _write_per_probes(avg, _PCIDS_2, _BIN_S)
        # Write combined under a different group name
        _merge_groups_to_combined(
            avg,
            _PCIDS_2,
            combined_group="other_group",
            label="other",
            bin_s=_BIN_S,
            combined_nc_path=comb,
        )

        assert (
            _combined_group_is_current(
                comb,
                avg,
                f"i_bin{_BIN_S}s",
                _PCIDS_2,
                _BIN_S,
            )
            is False
        ), "Combined group absent from file"


# ---------------------------------------------------------------------------
# _merge_groups_to_combined idempotency
# ---------------------------------------------------------------------------


@pytest.mark.xr
class TestMergeGroupsIdempotent:
    """Second _merge_groups_to_combined call must not rewrite when data is unchanged."""

    @pytest.mark.parametrize(
        "pcids",
        [pytest.param(_PCIDS_2, id="2probes")],
    )
    def test_no_rewrite_on_same_data(self, tmp_path, pcids):
        """File mtime + content must not change on re-merge with same per-probe data."""
        avg, comb = tmp_path / "t.proc_Avg.nc", tmp_path / "t.proc.nc"
        grp = f"i_bin{_BIN_S}s"

        _write_per_probes(avg, pcids, _BIN_S)
        _merge(avg, comb, pcids, _BIN_S)

        # Capture state after first merge
        mtime_before = _mtime_ns(comb)
        with xr.open_dataset(comb, group=grp, engine=_constants.nc_engine) as ds1:
            time_before = ds1["time"].values.copy()
            probe_before = ds1["probe"].values.copy()

        # Second merge — must be a no-op
        _merge(avg, comb, pcids, _BIN_S)

        assert _mtime_ns(comb) == mtime_before, f"pcids={pcids}: file mtime changed on re-merge"
        with xr.open_dataset(comb, group=grp, engine=_constants.nc_engine) as ds2:
            np.testing.assert_array_equal(ds2["time"].values, time_before)
            np.testing.assert_array_equal(ds2["probe"].values, probe_before)

    @pytest.mark.parametrize(
        "pcids",
        [pytest.param(_PCIDS_2, id="2probes")],
    )
    def test_rewrite_when_per_probe_extended(self, tmp_path, pcids):
        """Combined must grow when per-probe data extends beyond existing combined."""
        avg, comb = tmp_path / "t.proc_Avg.nc", tmp_path / "t.proc.nc"
        grp = f"i_bin{_BIN_S}s"

        _write_per_probes(avg, pcids, _BIN_S, n=50)
        _merge(avg, comb, pcids, _BIN_S)

        with xr.open_dataset(comb, group=grp, engine=_constants.nc_engine) as ds1:
            n_time_before = ds1.sizes["time"]

        # Extend per-probe data
        _write_per_probes(avg, pcids, _BIN_S, n=10, offset_minutes=100_000, seed=99)
        _merge(avg, comb, pcids, _BIN_S)

        with xr.open_dataset(comb, group=grp, engine=_constants.nc_engine) as ds2:
            assert ds2.sizes["time"] > n_time_before, (
                f"pcids={pcids}: combined not updated after per-probe extension"
            )

    def test_avg_nc_not_modified_on_re_run(self, tmp_path):
        """Per-probe .proc_Avg.nc must not be touched by re-merge."""
        avg, comb = tmp_path / "t.proc_Avg.nc", tmp_path / "t.proc.nc"

        _write_per_probes(avg, _PCIDS_2, _BIN_S)
        _merge(avg, comb, _PCIDS_2, _BIN_S)

        mtime_before = _mtime_ns(avg)
        _merge(avg, comb, _PCIDS_2, _BIN_S)

        assert _mtime_ns(avg) == mtime_before, ".proc_Avg.nc modified on re-merge"
