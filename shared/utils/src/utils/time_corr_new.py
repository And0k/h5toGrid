"""
Time-correction pipeline for low-resolution sensor timestamps.
Recovers monotone increased, grid alined time for 1s precisio input (N identical values per second at ~N Hz).

Anomalies, ordered by RMS impact on snap-to-grid:
  ① Out-of-range data (time_ranges)  elevates HWM → false backward detection  [DiagBit.OUT_OF_RANGE]
  ② Sustained backward sections       entire segment goes backward              [DiagBit.BACKWARD]
  ③ Isolated spikes                   single corrupt value                      [DiagBit.SPIKE]
  ④ Normal 1 s quantisation           RMS_q(N) = (1/N)√[(N-1)(2N-1)/6]

Pipeline (_correct_time):
  1. Subset to b_in_range  — out-of-range data never enters HWM or bilateral
  2. _trim_overlong_runs   — prevents parking-scan drift for near-integer freq
  3. _bilateral_check      — isolated spikes (O(n) per iteration)
  4. _hwm_check            — sustained backward sections (O(n))
  5. _find_hole_edges      — segment boundaries on clean subset
  6. _snap_to_grid         — g(k) = origin + k·dt_step per segment

All ops receive only b_in_range subset of t_ns → out-of-range can never corrupt the HWM.

Diagnostics (save_time_corr_diagnostics, plot_time_corr_diagnostics):
  NPZ arrays: index int32, action uint8, dt_s float32, freq float64, rms_theory float64
  action bitmask: see DiagBit; two-row (2, n) display via build_show_diag + build_diag_cmap.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from enum import IntFlag
from pathlib import Path
from typing import Any, Final, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from .utils2init import LoggingStyleAdapter, dir_create_if_need

lf = LoggingStyleAdapter(__name__)

NS = 1_000_000_000  # 1 s in ns (int64)
NS_F = np.float64(NS)

# Action bitmask: OR-able uint8 per sample, stored sparse (non-zero positions only)
class DiagBit(IntFlag):
    """Sparse diagnostic action bitmask (uint8), packed as two 4-bit rows for display.

    Row 0 (mask 0x0F) — output *properties*, may combine freely per sample:
        HOLE:  gap > dt_hole boundary; set on first sample after silence
        ALARM:        |t_corrected−t_obs| > (N−1)·dt_step for THIS sample, point-wise; N=round(freq)
                      is the largest offset any non-drifting run can produce (see _correct_time)
                      — deliberately not an RMS-based statistic, which for small N sits below
                      this ceiling and would falsely flag every run's last-in-run point
        OUT_OF_RANGE: excluded by time_ranges; t_out = t_obs unchanged, b_ok = False
        NOT_MONO:     final value non-monotone after snap; b_ok = False

    Row 1 (mask 0x70) — sample *fate*, mutually exclusive by pipeline construction, resulted in linear
    interpolated `t_out` values from neighbours.
    (trim precedes bilateral/HWM on the same subset ⟹ ≤1 removal reason per sample):
        TRIM      overlong-run sample dropped;
        SPIKE     bilateral outlier dropped;
        BACKWARD  HWM backward section dropped;

    See build_show_diag / build_diag_cmap for the (2, n) matrix + colormap this drives.
    """
    HOLE         = 0x01
    ALARM        = 0x02
    OUT_OF_RANGE = 0x04
    NOT_MONO     = 0x08
    TRIM         = 0x10
    SPIKE        = 0x20
    BACKWARD     = 0x40

_ROW_MASKS: Final[NDArray[np.uint8]] = np.array([0x0F, 0x70], dtype=np.uint8)  # [properties, fate]

# =============================================================================
# Utilities
# =============================================================================

def rms_quantization_theory(freq: float) -> float:
    """RMS_q(N) = (1/N)·√[(N−1)(2N−1)/6], N = round(freq). Zero for N<2."""
    N = round(freq)
    return 0.0 if N < 2 else (1.0 / N) * ((N - 1) * (2 * N - 1) / 6.0) ** 0.5


I64MIN = np.iinfo(np.int64).min
I64MAX = np.iinfo(np.int64).max

def make_range_mask(
    t_ns: NDArray[np.int64], time_ranges: Sequence[str | pd.Timestamp | None]
) -> NDArray[np.bool_]:
    """Vectorized inclusion mask: t_ns ∈ ∪[sᵢ, eᵢ]. NaT/None ≡ open bound.
    t_ns: input time (int64 ns)
    time_ranges: flat alternating limits [start1, end1, start2, end2, …]
    Broadcasting:
        (N, 1) >= (1, M) -> (N, M)
        (N, 1) <= (1, M) -> (N, M)
    """
    if not time_ranges:
        return np.ones(t_ns.size, dtype=bool)

    # Parse to UTC; handles mixed strings/Timestamps/None
    dti = pd.to_datetime(time_ranges, utc=True)
    if len(dti) & 1:
        dti = dti.insert(len(dti), pd.NaT)  # C-level odd-length padding

    # Int64 view (NaT -> iNaT = I64MIN)
    iv = dti.as_unit("ns").view(np.int64)

    # Split & branchless bound expansion
    s = iv[::2]  # s=I64MIN is naturally <= any valid t_ns
    e = iv[1::2]
    e[e == I64MIN] = I64MAX  # e=I64MIN must be mapped to I64MAX to satisfy t_ns <= e.
    return ((t_ns[:, None] >= s) & (t_ns[:, None] <= e)).any(axis=1)

    # previous version:
    # b = np.zeros(t_ns.size, bool)
    # for s, e in zip(iv[::2], iv[1::2]):
    #     b |= (t_ns >= (iNaT if s == iNaT else s)) & (t_ns <= (I64MAX if e == iNaT else e))
    # return b



# =============================================================================
# Step 1 – UTC conversion
# =============================================================================

def _to_utc(
    date: Union[pd.Series, pd.Index, np.ndarray],
    dt_from_utc: Optional[timedelta],
) -> np.ndarray:
    """Convert date to tz-naive ``datetime64[ns]`` values in UTC."""
    if not date.size:
        return np.empty(0, dtype="M8[ns]")

    if dt_from_utc:
        delta = pd.Timedelta(dt_from_utc)

        match next(iter(date)):
            case str():
                h = int(delta.total_seconds() / 3600)
                lf.debug("String dates: UTC offset {:+d}h embedded in zone label", h)
                t = pd.to_datetime(
                    (date.astype(object) + f"{h:+03d}").astype("M8[ns]"),
                    utc=True,
                )
            case _ if isinstance(date, pd.Index):
                t = date - delta
                try:
                    t = t.tz_localize("UTC")
                except TypeError:
                    lf.warning("Input already UTC – subtracted {}", dt_from_utc)
            case _:
                arr = date.to_numpy() if isinstance(date, pd.Series) else np.asarray(date, dtype="M8[ns]")
                td64 = delta.to_timedelta64()

                try:
                    t = pd.to_datetime(arr - td64, utc=True)
                except OverflowError:
                    ms = np.asarray(arr, dtype="M8[ms]").view(np.int64)
                    ms -= td64.astype("m8[ms]").astype(np.int64)
                    t = pd.to_datetime(ms.view("M8[ms]"), utc=True)

        lf.info(
            "UTC offset |{}| {}",
            abs(dt_from_utc),
            "subtracted" if dt_from_utc > timedelta(0) else "added",
        )
    else:
        if isinstance(dtype := getattr(date, "dtype", None), np.dtype) and np.issubdtype(
            dtype, np.datetime64
        ):
            return np.asarray(date, dtype="M8[ns]")

        t = pd.to_datetime(date, utc=True)

    match t:
        case pd.Series() as s:
            s = s.dt.as_unit("ns")
            if getattr(s.dtype, "tz", None):
                s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            return s.to_numpy()

        case pd.DatetimeIndex() as idx:
            idx = idx.as_unit("ns")
            if idx.tz:
                idx = idx.tz_convert("UTC").tz_localize(None)
            return idx.to_numpy()

        case _:
            return np.asarray(t, dtype="M8[ns]")



# =============================================================================
# Step 2 – Frequency estimation
# =============================================================================

def _ceil_to_decimals_or_to_fractions(freq: float, round_snapping: int | float) -> float:
    """Round_snapping: >1 → decimal-place ceiling; else → ceil to next multiple of round_snapping Hz."""
    scale = 10**round_snapping if round_snapping > 1 else 1 / round_snapping
    return (np.ceil(freq * scale) / scale).item()


def _estimate_freq_np(t_ns: NDArray[np.int64]) -> float:
    """Robust O(n log n) frequency estimation; two regimes auto-detected:

    B – floored 1 s (min_pos_diff ≥ 0.5 s ∧ n_zeros ≥ n_pos):
        Run-lengths of consecutive equal values → {N_base, N_base+1}.
        N_base = mode(runs); adjusted to mode−1 iff {mode−1, mode} jointly ≥ 80 % of runs
        (gate separates genuine fractional Hz from burst-boundary scatter).
        mean({N_base, N_base+1}) gives exact fractional freq:
            5.3 Hz ⟺ 70 %×{5} + 30 %×{6} → 5.30.

    A – sub-second or mixed:
        freq_j = di_inc_j / dt_f_j · 1e9 per interval between consecutive increases.
        di_inc/dt_f is burst-structure-independent; 3 σ filter removes inter-burst outliers.
    """
    if t_ns.size < 2:
        return 0.0
    dt = np.ediff1d(t_ns)
    pos = dt[dt > 0]
    if not pos.size:
        return 0.0
    n_zero = int((dt == 0).sum())

    # ── Regime B: floored 1s ─────────────────────────────────────────────────────────────
    if pos.min().item() >= 5e8 and n_zero >= pos.size:
        # Each group of identical consecutive timestamps = one second of data.
        # Run-length of group m = number of samples recorded in second m.
        bounds = np.flatnonzero(np.ediff1d(t_ns, to_begin=1) != 0)  # second-boundary positions
        runs   = np.ediff1d(bounds, to_end=t_ns.size - bounds[-1])  # samples/second for every second
        total  = runs.size                             # total seconds in recording

        vals, cnts = np.unique(runs, return_counts=True)   # sorted run-lengths + occurrence counts
        i_mode     = np.argmax(cnts).item()
        N_mode     = vals[i_mode].item()                 # most common samples/second
        frac_mode  = (cnts[i_mode] / total).item()       # its fraction of all seconds

        # i_lo = index of (N_mode−1) in vals if present; reused below for whichever
        i_lo = np.searchsorted(vals, N_mode - 1)
        has_lo   = i_lo < vals.size and vals[i_lo] == N_mode - 1
        frac_lo  = (cnts[i_lo] / total).item() if has_lo else 0.0

        # Determine N_base = floor(true_freq):
        #   5.3 Hz: 70% of seconds have 5 samples (mode=5=floor)         → N_base = N_mode
        #   5.8 Hz: 80% of seconds have 6 samples (mode=6=ceil!)         → N_base = N_mode−1
        # 80%-gate: {mode−1, mode} together ≥80% of seconds ⟹ genuine {floor,ceil} pair.
        # Below 80%: sub-mode values are burst-boundary artifacts, not a valid floor.
        if frac_lo > 0.05 and frac_mode + frac_lo >= 0.80:
            N_base, frac_N_base, frac_N_base_p1 = N_mode - 1, frac_lo, frac_mode
            gate_note = f"  [mode={N_mode} is ceil; {frac_lo+frac_mode:.0%} in 80%-gate]"
        else:
            N_base, frac_N_base, gate_note = N_mode, frac_mode, ""
            # ceil fraction not yet known — reuses vals/cnts already computed above,
            # no new searchsorted vs N_base since N_base+1 = N_mode+1 here
            i_hi = np.searchsorted(vals, N_mode + 1)
            frac_N_base_p1 = (cnts[i_hi] / total).item() if (
                i_hi < vals.size and vals[i_hi] == N_mode + 1) else 0.0

        # freq = frac_N_base·N_base + frac_N_base_p1·(N_base+1) = mean of the two valid run-lengths
        kept = runs[(runs == N_base) | (runs == N_base + 1)]
        freq = kept.mean().item() if kept.size >= 2 else float(N_base or N_mode)

        lf.info(
            "freq={:.6g}Hz found from {} floored 1s-runs: {}smp/s at {:.1%} + {}smp/s at {:.1%}{}",
            freq,
            total,
            N_base,
            frac_N_base,
            N_base + 1,
            frac_N_base_p1,
            gate_note,
        )
        return freq

    # ── Regime A: sub-second or mixed ─────────────────────────────────────────────────────────────
    i_inc = np.flatnonzero(dt > 0)
    if i_inc.size < 2:
        return 0.0
    di_inc = np.ediff1d(i_inc)
    dt_f = dt[i_inc[:-1]].astype(np.float64)
    if dt_f.size < 2:
        return 0.0
    freqs = NS_F * di_inc / dt_f
    med, std = np.median(freqs).item(), freqs.std().item()
    b_good = (np.abs(freqs - med) <= 3 * std) if std > 0 else np.ones(len(freqs), bool)
    freq = float(freqs[b_good].mean()) if (n_good := b_good.sum().item()) >= 2 else med
    lf.info(
        "freq={:.4f}Hz found as mean over kept {}/{} intervals in 3σ from median={:.4f}",
        freq,
        freqs.size,
        n_good,
        med,
    )
    return freq


# Minimum time step (ns) resolvable by the CF float64-seconds encoding.
# For current timestamps (~1.76e9 s), float64 has ~100 ns precision.
# 200 ns provides a safety margin.  Used by:
# - store_processed_incremental: monotonicity dedup before NC write
# - utils_time_corr._resolve_freq: frequency floor (freq ≤ NS / DT_CF_NS)
DT_CF_NS: int = 200


def _resolve_freq(t_ns: NDArray[np.int64], cfg_in: MutableMapping[str, Any]) -> float:
    """
    _estimate_freq_np + cfg fallback chain + optional ceil rounding.
    Updates cfg_in['fs_last'] with raw estimate. Logs once with final freq.
    Caps at NS / DT_CF_NS to prevent snap_to_grid producing timestamps that
    collapse under the CF float64-seconds encoding (see storage.DT_CF_NS).
    """
    freq_max = NS_F / DT_CF_NS
    if (freq_raw := _estimate_freq_np(t_ns)) > 0:
        cfg_in["fs_last"] = freq_raw
        freq = _ceil_to_decimals_or_to_fractions(freq_raw, cfg_in.get("fs_rounding", 100))
        if freq != freq_raw:
            lf.debug("freq={:.6g}Hz after fs_rounding={} applied", freq, cfg_in.get("fs_rounding"))
    else:
        for key, label in (("fs_last", "last-file"), ("fs", "configured"), ("fs_old_method", "old-method")):
            if v := cfg_in.get(key):
                lf.warning("freq estimation failed => using {} value: {:.6g}Hz", label, v)
                freq = float(v)
                break
        else:
            lf.warning("freq unknown => defaulting to 1Hz")
            freq = 1.0
    if freq > freq_max:
        lf.warning("freq={:.6g}Hz capped to {:.6g}Hz (CF float64 resolution floor)", freq, freq_max)
        freq = freq_max
    return freq


# =============================================================================
# Step 3 – Trim overlong runs
# =============================================================================

def _trim_overlong_runs(t_ns: NDArray[np.int64], freq: float) -> NDArray[np.bool_]:
    """
    Cap each equal-value run at N = round(freq) samples to prevent parking-scan drift.

    Root cause: N+1-sample second → snap parking scan shifts all subsequent seconds
    by +dt permanently → cumulative drift O(n_extra·dt) over whole file.

    Activated only when freq is within 1e-3 of an integer (near-exact N Hz device).
    For genuine fractional rates (e.g. 5.3 Hz), snap's ±dt/2 rounding handles
    5/6-sample seconds correctly without trimming — verified by test_snap_fractional_hz.

    Algorithm O(n): pos_in_run[i] = i − max_accum(where(is_new, i, 0)).
    """
    N = int(round(freq))
    if N < 1 or (freq - int(freq)) >= 1e-3:  # genuine fractional → snap handles it
        return np.ones(t_ns.size, dtype=bool)
    idx = np.arange(t_ns.size, dtype=np.int64)
    run_start = np.maximum.accumulate(np.where(np.ediff1d(t_ns, to_begin=1) != 0, idx, np.int64(0)))
    return (idx - run_start) < N


# =============================================================================
# Step 4 – Hole detection
# =============================================================================

def _find_hole_edges(t_ns: NDArray[np.int64], dt_hole_ns: int) -> NDArray[np.int64]:
    """gap > dt_hole_ns → [0, h₁, …, n] segment boundaries. O(n)."""
    return np.r_[np.int64(0), np.flatnonzero(np.ediff1d(t_ns) > dt_hole_ns) + 1, np.int64(t_ns.size)]


# =============================================================================
# Step 5 – Outlier removal
# =============================================================================

def _bilateral_check(
    t_ns: NDArray[np.int64],
    dt_step_ns: int,
    threshold_ns: int,
    max_iter: int = 3,
) -> NDArray[np.bool_]:
    """
    Isolated spike i: |Δfwd[i]−dt|>thresh ∧ |Δbwd[i]−dt|>thresh. O(n)/iter.
    Real gap: one side anomalous, other side normal → NOT flagged (see module docstring §table).
    Sustained backward: all internal diffs = +dt → NOT detected here; use _hwm_check.
    """
    b_ok = np.ones(t_ns.size, dtype=bool)
    for _ in range(max_iter):
        good = np.flatnonzero(b_ok)
        t_cur = t_ns.copy()
        if not b_ok.all() and len(good) >= 2:
            t_cur[~b_ok] = np.interp(
                np.flatnonzero(~b_ok).astype(float), good.astype(float), t_ns[good].astype(float)
            ).astype(np.int64)
        dt_f = np.ediff1d(t_cur, to_begin=dt_step_ns)
        dt_b = np.ediff1d(t_cur, to_end=dt_step_ns)
        b_bad = (np.abs(dt_f - dt_step_ns) > threshold_ns) & (np.abs(dt_b - dt_step_ns) > threshold_ns)
        if not b_bad.any() or (b_bad == ~b_ok).all():
            break
        b_ok &= ~b_bad
    return b_ok


def _hwm_check(
    t_ns: NDArray[np.int64],
    b_ok_bilateral: NDArray[np.bool_],
    threshold_ns: int,
) -> NDArray[np.bool_]:
    """
    Backward section: t_interp[i] < max_accum(t_interp)[i] − thresh. O(n).
    Operates on spike-interpolated signal so forward spikes don't falsely elevate HWM.
    """
    t_clean = t_ns.copy()
    if not b_ok_bilateral.all() and (good := np.flatnonzero(b_ok_bilateral)).size >= 2:
        t_clean[~b_ok_bilateral] = np.interp(
            np.flatnonzero(~b_ok_bilateral).astype(float), good.astype(float), t_ns[good].astype(float)
        ).astype(np.int64)
    return t_clean >= np.maximum.accumulate(t_clean) - threshold_ns


def _remove_outliers_combined(
    t_ns: NDArray[np.int64],
    dt_step_ns: int,
    threshold_ns: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], dict[str, int]]:
    """
    Bilateral (spikes) + HWM (backward sections).  Details at DEBUG; summary in caller.
    Returns (b_ok, b_bilateral, b_hwm, {"n_spikes", "n_backward"}).
    """
    b_bilateral = _bilateral_check(t_ns, dt_step_ns, threshold_ns)
    b_hwm = _hwm_check(t_ns, b_bilateral, threshold_ns)
    n_spikes, n_back = (~b_bilateral).sum().item(), (~b_hwm).sum().item()
    if n_spikes:
        lf.info(
            "bilateral spikes: {} at {}",
            n_spikes,
            [
                datetime.fromtimestamp(int(t) / NS_F).strftime("%Y-%m-%dT%H:%M:%S")
                for t in t_ns[~b_bilateral][:10]
            ],
        )
    if n_back:
        lf.info(
            "HWM backward: {} pts [{} … {}]",
            n_back,
            *[
                datetime.fromtimestamp(int(t) / NS_F).strftime("%Y-%m-%dT%H:%M:%S")
                for t in t_ns[~b_hwm][[0, -1]]
            ],
        )
    return b_bilateral & b_hwm, b_bilateral, b_hwm, {"n_spikes": n_spikes, "n_backward": n_back}


# =============================================================================
# Step 6 – Snap to grid
# =============================================================================

def _snap_segment_np(
    t_seg: NDArray[np.int64], seg_origin: np.int64, dt_step_ns: np.int64
) -> NDArray[np.int64]:
    """
    Snap t_seg → g(k) = seg_origin + k·dt_step, strictly monotone.

    k = (t − origin + dt/2) // dt          ← nearest-node rounding
    k = max_accum(k)                        ← monotone
    Parking scan (overflow across equal-k runs):
        excl_cum = prefix_sum(run_lens, exclusive)
        run_k_adj = max_accum(run_k − excl_cum) + excl_cum
    Reconstruct: repeat(run_k_adj, run_lens) + within_run_pos
    After _trim_overlong_runs: ∀ runs len ≤ N → no overflow, all diffs = dt_step.
    """
    k = (t_seg - seg_origin + dt_step_ns // 2) // dt_step_ns
    k = np.maximum.accumulate(k)
    is_new = np.ediff1d(k, to_begin=1) != 0
    rs = np.flatnonzero(is_new)
    rl = np.ediff1d(rs, to_end=k.size - rs[-1]).astype(np.int64)
    run_k = k[rs].copy()
    excl_cum = np.r_[np.int64(0), np.cumsum(rl[:-1])]
    run_k = np.maximum.accumulate(run_k - excl_cum) + excl_cum
    ar = np.arange(k.size, dtype=np.int64)
    return (
        seg_origin
        + (np.repeat(run_k, rl) + ar - np.maximum.accumulate(np.where(is_new, ar, np.int64(0)))) * dt_step_ns
    )


def _snap_to_grid(
    t_ns: NDArray[np.int64], b_ok: NDArray[np.bool_], freq: float, seg_edges: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.bool_], dict]:
    """
    Snap per segment; origin = t_good[0] (1 s anchor, robust to variable rate).
    Returns (t_corrected_ns, b_monotone, snap_stats).

    snap_stats keys: rms_max rms_mean rms_theory n_segs seg_rms — aggregate quality only.
    alarm_segs: list[(i0, i1)] segment ranges (into input t_ns) where RMS > alarm threshold —
        caller flags the FULL [i0, i1) range with DiagBit.ALARM
    """
    dt_step_ns = np.int64(round(NS_F / freq))
    rms_q = rms_quantization_theory(freq)
    t_out = t_ns.copy()
    seg_rms: list[float] = []
    n_segs = seg_edges.size - 1

    for i_s, (i0, i1) in enumerate(zip(seg_edges[:-1], seg_edges[1:])):
        seg_mask = b_ok[i0:i1]
        t_good = t_ns[i0:i1][seg_mask]
        if t_good.size < 2:
            continue
        t_snapped = _snap_segment_np(t_good, np.int64(t_good[0]), dt_step_ns)
        rms_s = (np.sqrt(np.mean((t_good - t_snapped).astype(float) ** 2)) / NS_F).item()
        seg_rms.append(rms_s)
        if i_s < 3:
            lf.debug(
                "seg {}/{} n={}: RMS={:.4f}s (q_theory={:.4f}s ×{:.2f}) nu={}",
                i_s + 1,
                n_segs,
                t_good.size,
                rms_s,
                rms_q,
                rms_s / max(rms_q, 1e-12),
                int((np.diff(t_snapped) != dt_step_ns).sum()),
            )
        good_idx = np.flatnonzero(seg_mask)
        t_out[i0 + good_idx] = t_snapped
        if (bad_idx := np.flatnonzero(~seg_mask)).size and good_idx.size >= 2:
            # Snap interpolated values to the grid — they're estimates anyway,
            # and landing on-grid guarantees minimum spacing = dt_step (>> float64 resolution).
            origin = t_snapped[0]
            t_out[i0 + bad_idx] = origin + np.rint(
                (np.interp(
                    bad_idx.astype(np.float64), good_idx.astype(np.float64), t_snapped.astype(np.float64)
                ) - origin) / dt_step_ns
            ).astype(np.int64) * dt_step_ns

    b_monotone = np.ediff1d(t_out, to_begin=1) > 0
    rms_max = max(seg_rms) if seg_rms else 0.0
    rms_mean = sum(seg_rms) / len(seg_rms) if seg_rms else 0.0

    if n_non := (~b_monotone).sum().item():
        lf.warning("{} interpolated outlier position(s) non-monotone after snap (masked)", n_non)
    lf.debug("snap {}/{} segs RMS: mean={:.4f}s max={:.4f}s q_theory={:.4f}s",
            n_segs - seg_rms.count(0), n_segs, rms_mean, rms_max, rms_q)

    return t_out, b_monotone, {
        "rms_max": rms_max, "rms_mean": rms_mean, "rms_theory": rms_q,
        "n_segs": n_segs, "seg_rms": seg_rms,
    }


# =============================================================================
# Step 7 – Orchestrator
# =============================================================================


def _null_snap_stats(freq: float, n_segs: int) -> dict:
    """Placeholder snap_stats for non-snap modes; seg_rms=[]."""
    return {"rms_max": 0.0, "rms_mean": 0.0, "rms_theory": rms_quantization_theory(freq),
            "n_segs": n_segs, "seg_rms": []}


def _correct_time(
    t_ns: NDArray[np.int64],
    cfg_in: Mapping[str, Any],
    process: Union[bool, str, None],
    b_in_range: Optional[NDArray[np.bool_]] = None,
) -> tuple[NDArray[np.int64], NDArray[np.bool_], dict, NDArray[np.uint8]]:
    """
    Full pipeline on int64 ns array; returns (t_out_ns, b_monotone, stats, action).

    b_in_range: bool mask — ALL outlier/snap ops restricted to True positions so that
    out-of-range data (time_ranges config) can never elevate the HWM or set drift.
    None → all True (process whole array).

    action: uint8[n] bitmask per sample (DiagBit members); stored sparse by caller.
    stats keys: n_total n_trim n_spikes n_backward n_holes freq + snap_stats keys.
    """
    if t_ns.size < 2:
        return t_ns.copy(), np.ones(t_ns.size, bool), {}, np.zeros(t_ns.size, np.uint8)

    if b_in_range is None:
        b_in_range = np.ones(t_ns.size, bool)
    in_idx = np.flatnonzero(b_in_range)  # positions trusted by time_ranges
    t_sub = t_ns[in_idx]  # ALL subsequent ops on this subset

    freq = _resolve_freq(t_sub, cfg_in)
    dt_step_ns = np.int64(round(NS_F / freq))
    thresh_s = float(cfg_in.get("corr_time_outlier_threshold_s", 0.6))
    min_hole = cfg_in.get("dt_interp_between", timedelta(seconds=1.5))
    dt_hole_ns = np.int64(
        max(
            (min_hole.total_seconds() if isinstance(min_hole, timedelta) else min_hole),
            2.0 / freq,
        )
        * NS_F
    )
    lf.debug(
        "n={} in-range={} freq={:.6g}Hz dt_step={:.3f}s dt_hole={:.2f}s thresh={}s",
        t_ns.size,
        in_idx.size,
        freq,
        float(dt_step_ns) / NS_F,
        float(dt_hole_ns) / NS_F,
        thresh_s,
    )

    # Trim anomalous overlong runs (near-integer freq only; see _trim_overlong_runs)
    b_trim_s = _trim_overlong_runs(t_sub, freq)
    if n_trim := (~b_trim_s).sum().item():
        lf.warning(
            "Trimmed {} overlong-run sample(s): prevented ≈{:.1f}s drift (rate > {:.6g}Hz)",
            n_trim,
            n_trim * float(dt_step_ns) / NS_F,
            freq,
        )

    # Outlier removal (HWM never sees out-of-range data → HWM safe)
    b_ok_s, b_bil_s, b_hwm_s, out_stats = _remove_outliers_combined(
        t_sub, int(dt_step_ns), int(thresh_s * NS_F)
    )
    b_ok_s &= b_trim_s
    if (n_total := in_idx.size - b_ok_s.sum().item()):  # unique removed (overlap not double-counted)
        lf.info(
            "Removed {:.1f}% = ({} overlong + {} spike + {} backward{})/{}",
            100 * n_total / max(in_idx.size, 1),
            n_trim,
            out_stats["n_spikes"],
            (n_back := out_stats["n_backward"]),
            " (clock correction?)" if n_back else "",
            in_idx.size,
        )
    else:
        lf.debug("No outliers / overlong runs in in-range data")

    # Hole segmentation on clean subset; edges are indices into t_sub
    clean_s = np.flatnonzero(b_ok_s)
    ec = _find_hole_edges(t_sub[b_ok_s], dt_hole_ns)
    seg_edges_s = np.r_[np.int64(0), clean_s[ec[1:-1]], np.int64(t_sub.size)]
    if (n_holes := seg_edges_s.size - 2):
        lf.debug(
            "{} segments: {} hole{} > {:.2f}s",
            n_holes + 1, n_holes, "s" if n_holes > 1 else "", dt_hole_ns.item() / NS
        )

    # Snap → delete_inversions → mask-only + warns if `process` is unknown
    if process in (True, "increase", "True") or (process is None and cfg_in.get("corr_time_mode")):
        t_sub_out, b_mono_s, snap_stats = _snap_to_grid(t_sub, b_ok_s, freq, seg_edges_s)
    else:
        t_sub_out = t_sub.copy()
        b_mono_s = (np.ediff1d(t_sub_out, to_begin=1) > 0) & b_ok_s
        snap_stats = _null_snap_stats(freq, n_holes + 1)
        if process == "delete_inversions":
            # Timestamps unchanged; non-monotone positions (after bilateral+HWM) masked in b_ok.
            # Differs from False (skips _correct_time) and None (no config): still runs
            # trim + outlier removal so that backward sections are properly excised first.
            lf.info(
                "delete_inversions: {}/{} non-monotone masked, timestamps unchanged",
                (~b_mono_s).sum().item(),
                t_sub.size,
            )
        else:
            if process is not None:
                lf.warning("unknown process={!r} — falling back to mask-only", process)

            lf.info("process={}: mask-only, timestamps unchanged", process)
    b_one_fail = True
    t_sub_tst = t_sub_out.copy()
    while not (b_mono_tst := (np.ediff1d(t_sub_tst := t_sub_tst[b_mono_s], to_begin=b_one_fail) > 0)).all():
        lf.warning(f"Monotonicity violated in {(~b_mono_tst).sum().item()} samples => del. each at right")
        b_mono_s[b_mono_s] = b_mono_tst
        b_one_fail = False  # the correction is fully failed, del 1st too to definitely get out of the loop

    # Map results back to full-length arrays
    t_out = t_ns.copy()
    t_out[in_idx] = t_sub_out

    b_monotone = np.zeros(t_ns.size, bool)
    b_monotone[in_idx] = b_mono_s  # out-of-range positions stay False

    # Point-wise correction magnitude for DiagBit.ALARM (and the log below)
    # mono_full_idx aligns 1:1 with dc: full-array positions of the monotone t_sub subset.
    # Threshold = (N−1)·dt_step: the largest offset ANY non-drifting N-sample run can produce
    # (k∈[0,N−1] within-run position × dt_step)
    mono_full_idx = in_idx[b_mono_s]
    dt_corr       = (t_sub_out[b_mono_s].astype(float) - t_sub[b_mono_s].astype(float)) / NS_F
    alarm_thr     = (round(freq) - 1) * (float(dt_step_ns) / NS_F) + 1e-9
    alarm_mask    = np.abs(dt_corr) > alarm_thr
    # Build action array — every algorithmic decision recorded here for display on two rows
    action = np.zeros(t_ns.size, np.uint8)
    action[~b_in_range]                   |= DiagBit.OUT_OF_RANGE.value   # row 0: property
    action[in_idx[~b_trim_s]]             |= DiagBit.TRIM.value           # row 1: fate
    action[in_idx[~b_bil_s]]              |= DiagBit.SPIKE.value          # row 1: fate
    action[in_idx[~b_hwm_s]]              |= DiagBit.BACKWARD.value       # row 1: fate
    if n_holes:
        action[in_idx[clean_s[ec[1:-1]]]] |= DiagBit.HOLE.value           # row 0: property
    action[mono_full_idx[alarm_mask]]     |= DiagBit.ALARM.value          # row 0: property, point-wise
    action[in_idx[~b_mono_s]]             |= DiagBit.NOT_MONO.value       # row 0: property

    n_alarm = alarm_mask.sum().item()
    n_mono = b_monotone.sum().item()
    dt_min = dt_corr.min().item() if dt_corr.size else 0.0
    dt_max = dt_corr.max().item() if dt_corr.size else 0.0
    pct_removed = 100 * n_total / max(in_idx.size, 1)
    if n_alarm or (t_ns.size - n_mono) > in_idx.size * 0.001:
        lf.warning(
            "time correction: {}/{} monotone (in-range={}); {:.1f}% removed "
            "(spikes={}, backward={}); correction [{:.3f}, {:.3f}]s; {} pts > alarm {:.2f}s",
            n_mono, t_ns.size, b_in_range.sum().item(),
            pct_removed, out_stats["n_spikes"], out_stats["n_backward"],
            dt_min, dt_max, n_alarm, alarm_thr,
        )
    else:
        lf.info(
            "time correction: {}/{} monotone; {:.1f}% removed; correction [{:.3f}, {:.3f}]s",
            n_mono, t_ns.size, pct_removed, dt_min, dt_max,
        )

    return (
        t_out,
        b_monotone,
        {
            "n_total": n_total,
            "n_trim": n_trim,
            **out_stats,
            "n_holes": n_holes,
            "freq": freq,
            **snap_stats,
        },
        action,
    )


# =============================================================================
# Public API
# =============================================================================

def time_corr(
    date: Union[pd.Series, pd.Index, np.ndarray],
    cfg_in: Mapping[str, Any],
    process: Union[str, bool, None] = None,
    path_save_image: str = "diagnostics",
) -> tuple[np.ndarray, NDArray[np.bool_]]:
    """Correct timestamps from low-resolution sensor data.

    cfg_in keys:
        dt_from_utc           timedelta    UTC offset to subtract
        time_ranges           Sequence     flat [s1,e1,s2,e2,…] of Timestamp/None (NaT=open)
        fs / fs_last          float        frequency fallback
        fs_rounding           int|float    ceil-snap (100=6 decimal default; <1 or ≥100=decimals; else Hz step)
        corr_time_mode        str          as process when process is None
        b_keep_not_a_time     bool         preserve NaT positions in output
        dt_interp_between     timedelta    min gap = real hole (default 1.5 s)
        corr_time_outlier_threshold_s   float        spike/backward threshold (default 0.6 s)
    process: None/False | True/'increase' | 'delete_inversions'

    Returns (tim_utc, b_ok): tz-naive datetime64[ns] (values UTC) + bool mask, same length as date.
    """
    if not len(date):
        return np.array([], dtype="datetime64[ns]"),  np.bool_([])
    if process == "False":
        process = False
    elif process in ("True", "increase"):
        process = True
    if process is None:
        process = cfg_in.get("corr_time_mode")

    t_in = _to_utc(date, cfg_in.get("dt_from_utc"))  # datetime64[ns] numpy
    b_nat = np.isnat(t_in)
    n_nat = int(b_nat.sum())
    t_use = t_in[~b_nat] if n_nat else t_in
    if n_nat:
        lf.info("{} NaT: {}", n_nat, "kept" if cfg_in.get("b_keep_not_a_time") else "will interpolate")

    # Edge-row data (config generation: only first/last CSV rows ± t_prev overlap)
    # — too few points for frequency estimation; skip _correct_time entirely.
    if t_use.size <= 4:
        if process not in (False, None):
            lf.warning(
                "time correction skipped: only {} value(s) — too few for freq estimation",
                t_use.size,
            )
        lf.debug("{} edge-row value(s), correction skipped", t_use.size)
        b_out = np.ones(len(t_in), bool)
        if n_nat:
            b_out[b_nat] = False
        return t_in, b_out

    # Build range mask BEFORE correction — prevents out-of-range data from entering HWM
    t_ns = t_use.view(np.int64).copy()
    b_in_range = make_range_mask(t_ns, cfg_in.get("time_ranges") or [])
    if (n_out := (~b_in_range).sum().item()):
        lf.info("{}/{} pts outside time_ranges (excluded from correction)", n_out, t_ns.size)

    if (process is not False) and b_in_range.any():
        t_c, b_mono, stats, action = _correct_time(t_ns, cfg_in, process, b_in_range)
    else:
        t_c = t_ns
        b_mono = (np.ediff1d(t_c, to_begin=1) > 0) & b_in_range
        action = np.zeros(t_ns.size, np.uint8)
        action[~b_in_range] |= DiagBit.OUT_OF_RANGE.value
        stats = {"n_total": 0, "rms_max": 0.0, "alarm": False, "freq": 0.0, "rms_theory": 0.0, "seg_rms": []}
        lf.debug("process=False: UTC + range mask only")

    # Re-insert NaTs; always return len(t_in)
    if n_nat:
        t_out_ns = np.full(len(t_in), np.iinfo(np.int64).min, np.int64)
        b_out = np.zeros(len(t_in), bool)
        t_out_ns[~b_nat] = t_c
        b_out[~b_nat] = b_mono
    else:
        t_out_ns = t_c
        b_out = b_mono
    if path_save_image and action.any():
        if (p := save_time_corr_diagnostics(t_ns, t_c, action, stats, cfg_in, path_save_image)):
            try:
                from tcm.plot_time_corr_diagnostics import plot_time_corr_diagnostics
                plot_time_corr_diagnostics(p, t_obs_ns=t_ns, path_save=None)
            except Exception as e:
                lf.debug("Plot not saved: {}", str(e))

    t_out = t_out_ns.view("datetime64[ns]")
    t_good = t_out[b_out]
    if t_good.size and (n_dup := t_good.size - np.unique(t_good).size):
        # Find first duplicate via sorted adjacent comparison
        sorted_idx = np.argsort(t_good.view(np.int64))
        t_sorted = t_good[sorted_idx]
        dup_at = np.flatnonzero(np.diff(t_sorted.view(np.int64)) == 0)[0]
        raise ValueError(
            f"time_corr produced {n_dup} duplicate timestamp(s). "
            f"First duplicate: {t_sorted[dup_at]} (n_good={t_good.size})"
        )
    return t_out, b_out


# =============================================================================
# Diagnostics – save
# =============================================================================

def save_time_corr_diagnostics(
    t_obs_ns: NDArray[np.int64],
    t_corr_ns: NDArray[np.int64],
    action: NDArray[np.uint8],
    stats: dict,
    cfg_in: Mapping[str, Any],
    path_save: str = "corr_time_mode",
) -> Optional[Path]:
    """
    Save sparse NPZ at positions where action ≠ 0 (algorithmic decisions only).

    Sparsity relies on flags, not |dt_s| magnitude: floor-reconstruction inherently gives
    most corrected points in 1 s-floored data a large |dt_s| (up to ~1 s, RMS≈rms_theory)
    even when nothing is wrong — that's the algorithm's normal output, not an anomaly.
    A magnitude threshold therefore cannot separate "expected" from "flagged" here; DiagBit
    is the only reliable criterion. DiagBit.ALARM is assigned point-wise (|dt_s| > threshold
    per sample, see DiagBit docstring) rather than by segment-average, so every individually
    anomalous position is captured — a segment's mean RMS can hide such points entirely.

    Stored arrays:
        index      int32[k]    flagged positions (sorted; see DiagBit for what sets a flag)
        action     uint8[k]    DiagBit bitmask (see class docstring for row 0/1 semantics)
        dt_s       float32[k]  t_corr−t_obs [s]
        freq       float64[]   scalar — estimated frequency
        rms_theory float64[]   scalar — RMS_q(freq), unavoidable quantisation floor
        n          int64[]     scalar — full series length (needed for build_show_diag column
                                alignment; NOT derivable from sparse index if trailing samples
                                carry no event)
    NOT stored (trivially re-derivable):
        dt_obs_s = np.ediff1d(t_obs_ns) / 1e9   (one-liner from source)

    Load: d=np.load(p); dropped = (d['action'] & _ROW_MASKS[1]) != 0; dt_s = d['dt_s']
    """
    try:
        if not (sig := np.flatnonzero(action)).size:
            lf.debug("no significant events — skip diagnostics")
            return None
        dt_s = ((t_corr_ns[sig] - t_obs_ns[sig]).astype(np.float64) / NS_F).astype(np.float32)
        # Fate-bit points (TRIM|SPIKE|BACKWARD) were dropped+interpolated — correction undefined
        dt_s[(action[sig] & _ROW_MASKS[1]) != 0] = np.float32(np.nan)

        if not (p := Path(path_save)).is_absolute():
            base, stem = Path("."), None
            for field in ("file_cur", "text_path", "path"):
                if bv := cfg_in.get(field):
                    base = Path(bv)
                    stem = base.stem if field == "file_cur" else None
                    break
            while not base.is_dir():
                base = base.parent
            p = dir_create_if_need(base / str(path_save))
            if stem:
                if len(parts := stem.split("@", 1)) == 2:  # put `@{pid}` after "dt"
                    stem_npz = "_dt@".join(parts) if parts[0] else f"dt@{parts[-1]}"
                else:  # no `@{pid}` detected
                    stem_npz = f"{stem}_dt"
                p = p / stem_npz
        if p.is_dir():
            ts0 = datetime.fromtimestamp(t_obs_ns[0] // 1e9, tz=timezone.utc)
            ts1 = datetime.fromtimestamp(t_obs_ns[-1] // 1e9, tz=timezone.utc)
            p = p / f"{ts0:%y%m%d_%H%M}-{ts1:%H%M}_dt.npz"
        elif p.suffix != ".npz":
            p = p.with_suffix(".npz")

        np.savez_compressed(
            p,
            index=sig.astype(np.int32),
            action=action[sig],
            dt_s=dt_s,
            freq=np.float64(stats.get("freq", 0)),
            rms_theory=np.float64(stats.get("rms_theory", 0)),
            n          = np.int64(t_obs_ns.size)
        )
        # Per-flag event counts via direct DiagBit iteration — names double as legend labels
        counts = {bit.name: ((action[sig] & bit) != 0).sum().item() for bit in DiagBit}
        lf.info(
            "diagnostics {} saved ({} events): {}",
            p,
            len(sig),
            ", ".join(f"{k}={v}" for k, v in counts.items() if v),
        )
        return p
    except Exception:
        lf.warning("Could not save diagnostics", exc_info=True)
        return None
