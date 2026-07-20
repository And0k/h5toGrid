"""
test_time_corr_new.py — pytest suite for time_corr_new.py

Covers: rms_quantization_theory, _make_range_mask, _estimate_freq_np,
        _trim_overlong_runs, _bilateral_check / _hwm_check,
        _snap_segment_np / _snap_to_grid, _correct_time (delete_inversions,
        time_ranges HWM-isolation regression, whole-segment ALARM flagging),
        save_time_corr_diagnostics, build_show_diag / build_diag_cmap.

Run:  pytest test_time_corr_new.py -v
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# import importlib.util, logging, sys, types
# # ---------------------------------------------------------------------------
# # Bootstrap: stub package, load module directly from sibling file
# # ---------------------------------------------------------------------------
# _pkg   = types.ModuleType("tcm")
# _utils = types.ModuleType("tcm.utils2init")
# # Null adapter: avoids %-vs-{} format conflict between stdlib logging and LoggingStyleAdapter
# _NullLog = type("L", (), {m: (lambda *a, **k: None) for m in ("debug","info","warning","error","exception")})
# _utils.LoggingStyleAdapter  = lambda n: _NullLog()                    # type: ignore[attr-defined]
# _utils.dir_create_if_need   = lambda p: (p.mkdir(parents=True, exist_ok=True), p)[-1]  # type: ignore
# sys.modules.update({"tcm": _pkg, "tcm.utils2init": _utils})

# _spec = importlib.util.spec_from_file_location(
#     "tcm.utils_time_corr", Path(__file__).with_name("utils_time_corr.py"))
# _m = importlib.util.module_from_spec(_spec)      # type: ignore[arg-type]
# _m.__package__ = "tcm"
# sys.modules["tcm.utils_time_corr"] = _m
# _spec.loader.exec_module(_m)                      # type: ignore[union-attr]

from tcm.utils_time_corr import (
    DiagBit, _ROW_MASKS, NS,
    _bilateral_check, _correct_time, _estimate_freq_np, _find_hole_edges,
    _hwm_check, make_range_mask, _remove_outliers_combined,
    _snap_segment_np, _snap_to_grid, _trim_overlong_runs,
    rms_quantization_theory, save_time_corr_diagnostics,
)
from tcm.plot_time_corr_diagnostics import build_diag_cmap, build_show_diag

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------
_CFG: dict = {"corr_time_outlier_threshold_s": 0.6}


def _floor(t: np.ndarray) -> np.ndarray:
    """Floor int64 ns timestamps to second boundary."""
    return (t // NS) * NS


def _seq(freq: float = 5.0, n: int = 200, t0: int = 0) -> np.ndarray:
    """Floored N-Hz sequence of n samples."""
    return _floor(np.arange(t0, t0 + n * int(round(1e9 / freq)), int(round(1e9 / freq)), np.int64))


def _fractional(freq: float, n_secs: int) -> np.ndarray:
    """floor(freq) or ceil(freq) samples/s, proportion matching frac(freq)."""
    N = int(freq); frac = freq - N
    hi = set(np.round(np.linspace(0, n_secs - 1, int(n_secs * frac))).astype(int))
    return np.array([m * NS for m in range(n_secs) for _ in range(N + (1 if m in hi else 0))],
                    np.int64)


def _burst(freq: float = 5.0, n: int = 20, blen: int = 10,
           gap_s: float = 5.0, aligned: bool = False) -> np.ndarray:
    """Floored burst: blen pts at freq, gap_s silence, repeat n times."""
    dt = int(round(1e9 / freq)); parts: list[np.ndarray] = []; t0 = 0
    for _ in range(n):
        if aligned: t0 = ((t0 + NS - 1) // NS) * NS
        parts.append(np.arange(t0, t0 + blen * dt, dt, np.int64))
        t0 = parts[-1][-1] + int(gap_s * NS)
    return _floor(np.concatenate(parts))


# ===========================================================================
class TestRmsQuantization:
# ===========================================================================

    @pytest.mark.parametrize("N, exp", [(1, 0.0), (2, 0.354), (4, 0.468), (5, 0.490),
                                         (6, 0.505), (10, 0.534)])
    def test_known_values(self, N: int, exp: float) -> None:
        assert abs(rms_quantization_theory(float(N)) - exp) < 0.002

    def test_below_2_is_zero(self) -> None:
        assert rms_quantization_theory(1.4) == 0.0   # round(1.4)=1 < 2

    def test_noninteger_uses_round(self) -> None:
        # round(5.6) = 6, so result should match N=6
        assert rms_quantization_theory(5.6) == pytest.approx(rms_quantization_theory(6.0))


# ===========================================================================
class TestMakeRangeMask:
# ===========================================================================
    _BASE = pd.Timestamp("2000-01-01 10:00", tz="UTC").value

    def _t(self, *offsets_s: int) -> np.ndarray:
        return np.array([self._BASE + o * NS for o in offsets_s], np.int64)

    def test_empty_all_true(self) -> None:
        assert make_range_mask(self._t(0, 60, 120), []).all()

    def test_inside_interval(self) -> None:
        t = self._t(0, 300, 600)
        tr = [pd.Timestamp("2000-01-01 10:00", tz="UTC"),
              pd.Timestamp("2000-01-01 10:10", tz="UTC")]
        assert make_range_mask(t, tr).all()

    def test_outside_interval(self) -> None:
        t = self._t(0, 300)
        tr = [pd.Timestamp("2000-01-01 11:00", tz="UTC"),
              pd.Timestamp("2000-01-01 12:00", tz="UTC")]
        assert not make_range_mask(t, tr).any()

    def test_nat_start_open_lower(self) -> None:
        t   = self._t(-3600, 0, 3601)
        tr  = [pd.NaT, pd.Timestamp("2000-01-01 10:30", tz="UTC")]
        m   = make_range_mask(t, tr)
        assert m[0] and m[1] and not m[2]

    def test_nat_end_open_upper(self) -> None:
        t   = self._t(0, 3600, 7200)
        tr  = [pd.Timestamp("2000-01-01 10:30", tz="UTC"), pd.NaT]
        m   = make_range_mask(t, tr)
        assert not m[0] and m[1] and m[2]

    def test_multi_range(self) -> None:
        t  = self._t(0, 300, 3600, 3900, 7200)
        tr = [pd.Timestamp("2000-01-01 10:00", tz="UTC"),
              pd.Timestamp("2000-01-01 10:10", tz="UTC"),
              pd.Timestamp("2000-01-01 11:00", tz="UTC"),
              pd.Timestamp("2000-01-01 11:10", tz="UTC")]
        np.testing.assert_array_equal(make_range_mask(t, tr), [True, True, True, True, False])

    def test_odd_length_padded_nat_end(self) -> None:
        """Odd-length time_ranges → NaT appended = open upper bound."""
        t  = self._t(0, 300)
        tr = [pd.Timestamp("2000-01-01 10:00", tz="UTC")]   # single start → open end
        assert make_range_mask(t, tr).all()

    def test_pandas2_us_unit(self) -> None:
        """pandas ≥2.0 infers datetime64[us] from Python datetime; .as_unit('ns') required."""
        t  = self._t(3600)                                   # 11:00 UTC
        tr = [datetime(2000, 1, 1, 10, 0), datetime(2000, 1, 1, 12, 0)]
        assert make_range_mask(t, tr).all(), (
            ".as_unit('ns') missing: us-precision mismatch vs ns timestamps")


# ===========================================================================
class TestEstimateFreq:
# ===========================================================================

    @pytest.mark.parametrize("t, exp, tol", [
        (_seq(5.0, 2000),          5.0,   0.01),   # exact 5 Hz continuous
        (_burst(5.0),              5.0,   0.01),   # burst non-aligned
        (_burst(5.0, aligned=True), 5.0,  0.01),   # burst second-aligned
        (_fractional(5.3, 1000),   5.3,   0.05),   # genuine 5.3 Hz
        (_fractional(5.5, 1000),   5.5,   0.02),   # genuine 5.5 Hz
        (_fractional(5.8, 1000),   5.8,   0.05),   # genuine 5.8 Hz
        # 0.36 % extra seconds → mean of {5,6} gives 5.003, not rounded to 5.0
        (_fractional(5.003, 30000), 5.003, 0.005),
    ], ids=["5Hz-cont", "5Hz-burst", "5Hz-burst-align", "5.3Hz", "5.5Hz", "5.8Hz", "5.003Hz"])
    def test_regime_b(self, t: np.ndarray, exp: float, tol: float) -> None:
        assert abs(_estimate_freq_np(t) - exp) <= tol, f"got {_estimate_freq_np(t):.4f}"

    @pytest.mark.parametrize("freq, tol", [
        (10.0,  0.01),   # exact sub-second
        (4.975, 0.01),   # was rounded to 5.0 with old round(x*2)/2.0
    ], ids=["10Hz", "4.975Hz"])
    def test_regime_a(self, freq: float, tol: float) -> None:
        dt = int(round(1e9 / freq))
        t  = np.arange(0, 5000 * dt, dt, np.int64)
        assert abs(_estimate_freq_np(t) - freq) <= tol

    def test_regime_a_burst_sub_second(self) -> None:
        dt5  = int(1e9 / 5)
        t    = np.concatenate([np.arange(t0, t0 + 10 * dt5, dt5, np.int64)
                                for t0 in range(0, 30 * (10 * dt5 + 5 * NS), 10 * dt5 + 5 * NS)])
        assert abs(_estimate_freq_np(t) - 5.0) < 0.02

    def test_empty_returns_zero(self) -> None:
        assert _estimate_freq_np(np.array([], np.int64)) == 0.0

    def test_single_point_returns_zero(self) -> None:
        assert _estimate_freq_np(np.array([0], np.int64)) == 0.0


# ===========================================================================
class TestTrimOverlong:
# ===========================================================================

    def test_clean_5hz_untouched(self) -> None:
        assert _trim_overlong_runs(_seq(5.0, 200), 5.0).all()

    def test_near_integer_trims_6th_sample(self) -> None:
        """freq=5.0 (frac=0 < 1e-3): 6th sample in overlong seconds is trimmed."""
        # Inject 20 six-sample seconds into otherwise clean 5 Hz data
        rng   = np.random.default_rng(1)
        n     = 2000
        extra = set(rng.choice(n, 20, replace=False))
        t     = np.array([m * NS for m in range(n)
                           for _ in range(6 if m in extra else 5)], np.int64)
        b     = _trim_overlong_runs(t, 5.0)          # freq is integer → trim gate fires
        assert not b.all()
        assert (~b).sum().item() == 20               # exactly one extra per 6-sample second

    def test_genuine_fractional_untouched(self) -> None:
        """5.3 Hz: frac=0.3 ≥ 1e-3 → snap handles 5/6-sample seconds via ±dt/2; no trim."""
        assert _trim_overlong_runs(_fractional(5.3, 500), 5.3).all()

    def test_prevents_parking_scan_drift(self) -> None:
        """After trim, RMS of snap should equal RMS_q; without trim it would be >> 1 s."""
        rng   = np.random.default_rng(42); n = 5_000
        extra = set(rng.choice(n, 30, replace=False))
        t     = np.array([m * NS for m in range(n) for _ in range(6 if m in extra else 5)], np.int64)
        t2    = t[_trim_overlong_runs(t, 5.0)]
        dt    = np.int64(NS // 5)
        rms   = (np.sqrt(np.mean((t2 - _snap_segment_np(t2, np.int64(t2[0]), dt)).astype(float)**2))
                 / NS).item()
        assert rms < rms_quantization_theory(5.0) * 1.1, f"RMS {rms:.3f}s >> theory after trim"


# ===========================================================================
class TestBilateral:
# ===========================================================================

    def test_forward_spike_detected(self) -> None:
        t = _seq(); t = t.copy(); t[50] += 3 * NS
        b = _bilateral_check(t, NS // 5, int(0.6 * NS))
        assert not b[50] and b[:50].all() and b[51:].all()

    def test_backward_spike_detected(self) -> None:
        t = _seq(); t = t.copy(); t[50] -= 3 * NS
        assert not _bilateral_check(t, NS // 5, int(0.6 * NS))[50]

    def test_misses_backward_section_interior(self) -> None:
        """Sustained backward: internal diffs ≈ +dt → bilateral silent inside the section."""
        t = _seq(n=150); t = t.copy(); t[30:80] -= 20 * NS
        b = _bilateral_check(t, NS // 5, int(0.6 * NS))
        assert b[35:75].any(), "bilateral must miss interior of backward section"

    def test_clean_all_true(self) -> None:
        assert _bilateral_check(_seq(n=300), NS // 5, int(0.6 * NS)).all()


# ===========================================================================
class TestHwm:
# ===========================================================================

    def test_detects_backward_section(self) -> None:
        t = _seq(n=200); t = t.copy(); t[80:120] -= 30 * NS
        b = _hwm_check(t, np.ones(t.size, bool), int(0.6 * NS))
        assert not b[80:120].any() and b[:80].all() and b[120:].all()

    def test_forward_spike_not_contaminating_hwm(self) -> None:
        """Spike already in b_ok=False must not elevate HWM and flag downstream data."""
        t = _seq(n=200); t = t.copy(); t[50] += 50 * NS
        b_bil = np.ones(t.size, bool); b_bil[50] = False
        b_hwm = _hwm_check(t, b_bil, int(0.6 * NS))
        assert b_hwm[51:].all(), "data after forward spike falsely flagged by HWM"

    def test_clean_all_true(self) -> None:
        t = _seq(n=200)
        assert _hwm_check(t, np.ones(t.size, bool), int(0.6 * NS)).all()


# ===========================================================================
class TestSnapSegment:
# ===========================================================================

    def test_5hz_rms_matches_theory(self) -> None:
        t   = _seq(5.0, 1000)
        s   = _snap_segment_np(t, np.int64(t[0]), np.int64(NS // 5))
        rms = (np.sqrt(np.mean((t - s).astype(float)**2)) / NS).item()
        assert abs(rms - rms_quantization_theory(5.0)) < 0.001

    def test_fractional_5_3hz_rms_matches_theory(self) -> None:
        """5.3 Hz floored: snap RMS ≈ RMS_q(5.3); no parking-scan drift."""
        t   = _fractional(5.3, 500)
        dt  = np.int64(int(round(1e9 / 5.3)))
        s   = _snap_segment_np(t, np.int64(t[0]), dt)
        rms = (np.sqrt(np.mean((t - s).astype(float) ** 2)) / NS).item()
        assert rms < rms_quantization_theory(5.3) * 1.5, f"RMS {rms:.4f}s >> theory (drift?)"
        assert (np.diff(s) > 0).all(), "snap must be strictly monotone"

    def test_strictly_monotone(self) -> None:
        assert (np.diff(_snap_segment_np(_seq(5.0, 500), np.int64(0), np.int64(NS // 5))) > 0).all()


# ===========================================================================
class TestCorrectTime:
# ===========================================================================

    # ── delete_inversions ────────────────────────────────────────────────────

    def test_delete_inv_timestamps_unchanged(self) -> None:
        t = _seq(n=200); t_bad = t.copy(); t_bad[80:120] -= 10 * NS
        t_out, *_ = _correct_time(t_bad, dict(_CFG), "delete_inversions")
        np.testing.assert_array_equal(t_out, t_bad)

    def test_delete_inv_backward_masked(self) -> None:
        t = _seq(n=200); t_bad = t.copy(); t_bad[80:120] -= 10 * NS
        _, b_ok, stats, act = _correct_time(t_bad, dict(_CFG), "delete_inversions")
        assert (~b_ok[80:120]).any()       # backward section masked
        assert stats["rms_max"] == 0.0    # no snap → zero RMS
        assert (act & DiagBit.BACKWARD).any()
        assert (act & DiagBit.NOT_MONO).any()

    def test_delete_inv_vs_snap_timestamp_difference(self) -> None:
        t = _seq()
        t_di, b_di, *_ = _correct_time(t, dict(_CFG), "delete_inversions")
        t_sn, b_sn, *_ = _correct_time(t, dict(_CFG), True)
        np.testing.assert_array_equal(t_di[b_di], t[b_di])   # d_inv: unchanged ✓
        assert not np.array_equal(t_sn[b_sn], t[b_sn])       # snap: modified  ✓

    # ── time_ranges HWM isolation (regression) ───────────────────────────────

    def test_oor_cannot_elevate_hwm(self) -> None:
        """
        Regression: OOR 12:xx data formerly elevated max_accumulate(t_clean),
        causing subsequent in-range 10:xx data to receive DiagBit.BACKWARD.
        b_in_range subsetting must fully isolate the HWM from OOR data.
        """
        base = pd.Timestamp("2000-01-01 10:00", tz="UTC").value
        t    = np.array(
            [base + i * NS for i in range(5)]            # 10:00–10:04  in-range ✓
          + [base + 120 * NS + i * NS for i in range(5)] # 12:00–12:04  OOR    ✗
          + [base + 5 * NS + i * NS  for i in range(5)], # 10:05–10:09  in-range ✓
            np.int64)
        b_range = np.array([True]*5 + [False]*5 + [True]*5)
        _, b_ok, _, act = _correct_time(t, dict(_CFG), True, b_range)

        assert not (act[:5]  & DiagBit.BACKWARD).any(), "pre-OOR block falsely backward-flagged"
        assert not (act[10:] & DiagBit.BACKWARD).any(), "post-OOR block falsely backward-flagged"
        assert     (act[5:10] & DiagBit.OUT_OF_RANGE).all(), "OOR positions not DiagBit.OUT_OF_RANGE"
        assert     b_ok[10:].all(), "post-OOR in-range points must be valid"

    def test_without_b_in_range_hwm_is_corrupted(self) -> None:
        """Without b_in_range, OOR data DOES corrupt HWM — demonstrates the regression."""
        base = pd.Timestamp("2000-01-01 10:00", tz="UTC").value
        t    = np.array(
            [base + i * NS for i in range(5)]
          + [base + 120 * NS + i * NS for i in range(5)]
          + [base + 5 * NS + i * NS  for i in range(5)], np.int64)
        _, _, _, act = _correct_time(t, dict(_CFG), True, None)  # no range mask
        assert (act[10:] & DiagBit.BACKWARD).any(), (
            "Without b_in_range, post-OOR block should be falsely backward-flagged")

    # ── snap mode ────────────────────────────────────────────────────────────

    def test_snap_rms_matches_theory(self) -> None:
        _, _, stats, _ = _correct_time(_seq(5.0, 1000), dict(_CFG), True)
        assert abs(stats["rms_max"] - rms_quantization_theory(5.0)) < 0.01

    def test_trim_count_in_stats(self) -> None:
        # 1 extra out of 2000 secs → frac = 0.0005 < 1e-3 → trim gate activates
        extra = {1000}
        t = np.array([m * NS for m in range(2000) for _ in range(6 if m in extra else 5)], np.int64)
        _, _, stats, _ = _correct_time(t, dict(_CFG), True)
        assert stats["n_trim"] == 1

    def test_holes_segmented_and_flagged(self) -> None:
        dt5 = int(1e9 / 5)
        t1  = _seq(5.0, 100)
        t2  = _seq(5.0, 100, t0=t1[-1] + 10 * NS)  # gap > dt_hole
        _, _, stats, act = _correct_time(np.concatenate([t1, t2]), dict(_CFG), True)
        assert stats["n_holes"] == 1
        assert ((act & DiagBit.HOLE) != 0).sum().item() == 1

    # ── process=None uses corr_time_mode ─────────────────────────────────────

    def test_none_process_with_corr_time_mode_snaps(self) -> None:
        t   = _seq()
        cfg = {**_CFG, "corr_time_mode": "increase"}
        _, _, stats, _ = _correct_time(t, cfg, None)
        assert stats["rms_max"] > 0, "corr_time_mode=increase → should snap"

    # ── unknown process ───────────────────────────────────────────────────────

    def test_unknown_process_mask_only(self) -> None:
        t = _seq()
        t_out, _, stats, _ = _correct_time(t, dict(_CFG), "unknown_mode")
        np.testing.assert_array_equal(t_out, t)
        assert stats["rms_max"] == 0.0

    # ── action bitmask coverage ───────────────────────────────────────────────

    def test_action_spike(self) -> None:
        t = _seq(); t[50] += 5 * NS
        _, _, _, act = _correct_time(t, dict(_CFG), True)
        assert act[50] & DiagBit.SPIKE

    def test_action_oor(self) -> None:
        t = _seq(n=10)
        b = np.array([True, True, False, False, True, True, True, True, True, True])
        _, _, _, act = _correct_time(t, dict(_CFG), True, b)
        assert (act[2:4] & DiagBit.OUT_OF_RANGE).all()
        assert not (act[[0, 1, 4]] & DiagBit.OUT_OF_RANGE).any()


# ===========================================================================
class TestSaveDiagnostics:
# ===========================================================================

    @staticmethod
    def _data_with_backward() -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
        t = _seq(n=200); t_bad = t.copy(); t_bad[80:120] -= 10 * NS
        t_out, _, stats, action = _correct_time(t_bad, dict(_CFG), "delete_inversions")
        return t_bad, t_out, stats, action

    def test_saves_npz(self, tmp_path: Path) -> None:
        t, t_out, stats, action = self._data_with_backward()
        p = save_time_corr_diagnostics(t, t_out, action, stats, {}, str(tmp_path))
        assert p is not None and p.suffix == ".npz" and p.exists()

    def test_required_keys_present(self, tmp_path: Path) -> None:
        t, t_out, stats, action = self._data_with_backward()
        p = save_time_corr_diagnostics(t, t_out, action, stats, {}, str(tmp_path))
        assert {"index", "action", "dt_s", "freq", "rms_theory"} <= set(np.load(p).keys())

    def test_dropped_pts_have_nan_dt_s(self, tmp_path: Path) -> None:
        t, t_out, stats, action = self._data_with_backward()
        p = save_time_corr_diagnostics(t, t_out, action, stats, {}, str(tmp_path))
        d = np.load(p)
        dropped = (d["action"] & _ROW_MASKS[1]) != 0   # any row-1 fate bit ⟹ dropped+interpolated
        assert np.isnan(d["dt_s"][dropped]).all()

    def test_dt_obs_s_not_stored(self, tmp_path: Path) -> None:
        """dt_obs_s = np.diff(t_in)/1e9 is a one-liner → must NOT be in NPZ."""
        t, t_out, stats, action = self._data_with_backward()
        p = save_time_corr_diagnostics(t, t_out, action, stats, {}, str(tmp_path))
        assert "dt_obs_s" not in np.load(p)

    def test_no_events_returns_none(self, tmp_path: Path) -> None:
        # snap of clean data → no trim/spike/backward/alarm/not_mono → action all-zero → None
        t = _seq()
        t_out, _, stats, action = _correct_time(t, dict(_CFG), True)
        assert save_time_corr_diagnostics(t, t_out, action, stats, {}, str(tmp_path)) is None

    def test_index_sorted_and_subset(self, tmp_path: Path) -> None:
        t, t_out, stats, action = self._data_with_backward()
        p  = save_time_corr_diagnostics(t, t_out, action, stats, {}, str(tmp_path))
        d  = np.load(p)
        assert (np.diff(d["index"]) > 0).all(), "index must be sorted"
        assert d["index"].size < t.size,         "only significant positions stored"

    def test_n_stored_equals_full_series_length(self, tmp_path: Path) -> None:
        """n is genuinely non-derivable from sparse index — must be stored exactly."""
        t, t_out, stats, action = self._data_with_backward()
        p = save_time_corr_diagnostics(t, t_out, action, stats, {}, str(tmp_path))
        assert np.load(p)["n"].item() == t.size


# ===========================================================================
class TestDisplayMatrix:
# ===========================================================================

    def test_show_diag_shape(self) -> None:
        idx = np.array([2, 5, 9])
        act = np.array([DiagBit.TRIM, DiagBit.HOLE, DiagBit.ALARM], np.uint8)
        assert build_show_diag(10, idx, act).shape == (2, 10)

    def test_show_diag_zero_outside_events(self) -> None:
        idx = np.array([3])
        act = np.array([DiagBit.SPIKE], np.uint8)
        m   = build_show_diag(10, idx, act)
        assert not m[:, [i for i in range(10) if i != 3]].any()

    def test_show_diag_row_split(self) -> None:
        """Row 0 gets property bits only; row 1 gets fate bits only (masked, not shifted)."""
        idx = np.array([0])
        act = np.array([DiagBit.HOLE | DiagBit.TRIM], np.uint8)   # one property + one fate bit
        m   = build_show_diag(1, idx, act)
        assert m[0, 0] == DiagBit.HOLE   # row 0 = act & _ROW_MASKS[0]
        assert m[1, 0] == DiagBit.TRIM   # row 1 = act & _ROW_MASKS[1] (unshifted)

    def test_diag_cmap_has_128_colors(self) -> None:
        pytest.importorskip("matplotlib")
        assert build_diag_cmap().N == 128

    def test_diag_cmap_zero_is_transparent(self) -> None:
        pytest.importorskip("matplotlib")
        assert tuple(build_diag_cmap().colors[0]) == (0.0, 0.0, 0.0, 0.0)

    def test_diag_cmap_every_bit_distinct_from_zero(self) -> None:
        """Every DiagBit member must map to a non-transparent, distinguishable colour."""
        pytest.importorskip("matplotlib")
        colors = build_diag_cmap().colors
        for bit in DiagBit:
            assert tuple(colors[bit.value]) != (0.0, 0.0, 0.0, 0.0), f"{bit.name} is transparent"

    def test_diag_cmap_row1_never_blends(self) -> None:
        """Row-1 (fate) colours must be pure per-bit hues, unaffected by row-0 bits present."""
        pytest.importorskip("matplotlib")
        colors = build_diag_cmap().colors
        pure_trim       = colors[DiagBit.TRIM.value]
        trim_plus_hole  = colors[(DiagBit.TRIM | DiagBit.HOLE).value]   # row-1 dominates by design
        np.testing.assert_array_equal(pure_trim, trim_plus_hole)
