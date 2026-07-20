"""
Diagnostics – plot  (3-panel: correction magnitude, event timeline, cumsum)
"""

from __future__ import annotations
from pathlib import Path
from typing import Final, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from .utils2init import LoggingStyleAdapter
from tcm.utils_time_corr import (
    NS_F,
    _ROW_MASKS,
    DiagBit
)
lf = LoggingStyleAdapter(__name__)
# BACKWARD (clock jump) and OUT_OF_RANGE (excluded window) span arbitrary, unbounded gaps —
# unlike TRIM/SPIKE (single-sample, bounded local interpolation, safe to accumulate). Used by
# plot_time_corr_diagnostics's Panel 3 to reset the cumulative-correction sum at each boundary.
_RESET_BITS: Final[int] = int(DiagBit.BACKWARD | DiagBit.OUT_OF_RANGE)

# =============================================================================
# Diagnostics – two-row event matrix + colormap  (feeds Panel 2 of the plot below)
# =============================================================================

def build_show_diag(n: int, idx: NDArray[np.intp], act: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Scatter sparse (idx, act) events into a dense (2, n) row-split matrix for imshow.

    Column i ↔ sample index i exactly (uniform grid ⟹ imshow extent = [0, n-1] is exact;
    a wall-clock extent is only an approximation under near-constant sample rate).

    Row 0 = act & _ROW_MASKS[0]  (properties, may combine — additive colour in build_diag_cmap)
    Row 1 = act & _ROW_MASKS[1]  (fate, mutually exclusive — hue-selected colour)

    Advanced indexing (show_diag[:, idx] = ...) avoids np.put's raveling/offset arithmetic
    for 2D targets and broadcasts _ROW_MASKS[:, None] natively over the idx columns.
    """
    show_diag = np.zeros((2, n), dtype=np.uint8)
    show_diag[:, idx] = act & _ROW_MASKS[:, None]
    return show_diag


def build_diag_cmap() -> "ListedColormap":  # noqa: F821  (matplotlib imported lazily below)
    """128-colour map (index = raw action byte, 0..127) matching DiagBit's two-row split.

    Row 0 (bits 0-3, values 0-15): RGBA basis vectors combined by np.einsum — additive
        mixing, so e.g. HOLE+ALARM together read as a distinct warm red-orange blend,
        visually communicating "these properties co-occur" without a separate legend entry.
    Row 1 (bits 4-6, values 16/32/64 ⟹ shifted index 1-3 of 8): HSV-distinct hues via
        colorsys — never blends since row-1 bits are mutually exclusive by construction.

    Combination rule cmap[i]: row-1 colour dominates when any row-1 bit is set (fate is
    the primary signal), else falls back to the row-0 additive blend (context only).
    """
    import colorsys
    from matplotlib.colors import ListedColormap   # lazy: mirrors plot fn's optional-dep guard

    row0_basis = np.array([                    # aligned to DiagBit bits 0..3 (row 0)
        [0.80, 0.10, 0.10, 1.0],   # HOLE          — warm red
        [1.00, 0.50, 0.00, 1.0],   # ALARM         — orange
        [0.20, 0.20, 0.80, 1.0],   # OUT_OF_RANGE  — cool blue
        [0.50, 0.50, 0.50, 1.0],   # NOT_MONO      — grey
    ], dtype=np.float32)
    row0_bits   = (1 << np.arange(4)).astype(np.uint8)
    row0_active = (np.arange(16, dtype=np.uint8)[:, None] & row0_bits).astype(bool)   # (16, 4)
    row0_colors = np.einsum("ij,jk->ik", row0_active, row0_basis)                      # (16, 4)
    row0_colors[:, :3] = np.clip(row0_colors[:, :3], 0.0, 1.0)     # additive RGB can exceed 1.0
    row0_colors[:, 3]  = row0_active.any(axis=1).astype(np.float32)  # opaque iff any bit set

    # Row-1 has 3 possible bits (TRIM, SPIKE, BACKWARD) ⟹ 8 states after shifting by 4;
    # state 0 (no fate) stays transparent, states 1..7 get distinct hues.
    row1_colors = np.array([
        (0.0, 0.0, 0.0, 0.0) if state == 0 else (*colorsys.hsv_to_rgb(state / 8, 0.9, 0.85), 1.0)
        for state in range(8)
    ], dtype=np.float32)

    cmap = np.empty((128, 4), dtype=np.float32)
    for i in range(128):
        row1_state = (i >> 4) & 0x07
        cmap[i] = row1_colors[row1_state] if row1_state else row0_colors[i & 0x0F]
    return ListedColormap(cmap)


def segmented_cumsum(dt_s: NDArray[np.float64], act: NDArray[np.uint8]) -> NDArray[np.float64]:
    """
    Cumulative sum of dt_s, reset to 0 at each _RESET_BITS (BACKWARD/OUT_OF_RANGE) boundary.

    Rationale: TRIM/SPIKE are single-sample, bounded local interpolations — safe to accumulate.
    BACKWARD/OUT_OF_RANGE span arbitrary, unbounded gaps (a clock jump; an excluded time window)
    — a genuine timeline discontinuity that shouldn't share a running baseline with what preceded
    it. Fed by Panel 3 of plot_time_corr_diagnostics; NaN at each reset point breaks the plotted
    line there (matplotlib skips NaN in line plots — a visual discontinuity, not an artifact).
    """
    reset  = (act & _RESET_BITS) != 0
    seg_id = np.cumsum(reset & ~np.r_[False, reset[:-1]])       # +1 at each reset run's first point

    # the idiomatic vectorised tool for grouped-cumsum — no manual loop or offset arithmetic):
    cum    = pd.Series(np.where(reset, 0.0, dt_s)).groupby(seg_id).cumsum().to_numpy(copy=True)
    cum[reset] = np.nan
    return cum


# =============================================================================
# Diagnostics – plot  (3-panel: correction magnitude, event timeline, cumsum)
# =============================================================================

def plot_time_corr_diagnostics(
    npz_path: Union[Path, str],
    t_obs_ns: Optional[NDArray[np.int64]] = None,
    path_save: Optional[Union[Path, str]] = None,
) -> Optional[Path]:
    """
    3-panel figure from NPZ saved by save_time_corr_diagnostics.

    x-axis: sample index, or wall-clock hours if t_obs_ns provided (uniform-column
    approximation under near-constant sample rate — see build_show_diag docstring).

    Panel 1 — dt_s scatter at event positions (non-NaN, colour=|dt_s| LogNorm):
        dotted lines at ±rms_theory (population spread) and ±alarm_thr=(N−1)·dt_step (per-point ceiling).
        Flat ≈ 0 = good; monotone ramp = surviving drift.

    Panel 2 — build_show_diag(n, index, action) rendered via imshow + build_diag_cmap:
        row 0 (properties, additive blend): HOLE, ALARM, OUT_OF_RANGE, NOT_MONO may combine —
            e.g. ALARM+NOT_MONO together at the same sample read as a distinct blend ⟹ snap
            could not resolve that point, visible directly from colour, no cross-referencing.
        row 1 (fate, exclusive hue): TRIM, SPIKE, BACKWARD — never blends by construction.
        Legend built from DiagBit member names directly (see build_diag_cmap).

    Panel 3 — segmented_cumsum(dt_s, action): grouped by run-id of _RESET_BITS boundaries.
        TRIM/SPIKE included (bounded, safe to accumulate); sum resets to 0 and the plotted line
        breaks (NaN) at each BACKWARD/OUT_OF_RANGE — a real timeline discontinuity shouldn't
        share a baseline with what came before it. See segmented_cumsum's docstring for why.
        flat = no net drift; ramp slope/dt_step ≈ extra samples/s.
        Caption: net drift (excluding reset points), implied extra/missing sample count, rate.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        lf.debug("matplotlib not available – skip plot")
        return None

    d = np.load(npz_path, allow_pickle=False)
    index = d["index"].astype(np.int64)
    act = d["action"]
    dt_s = d["dt_s"].astype(np.float64)
    freq      = d["freq"].item()
    rms_q     = d["rms_theory"].item()
    n         = d["n"].item()          # full series length — exact column↔index map in Panel 2
    # Must match _correct_time's DiagBit.ALARM formula exactly (see that function's comment)
    alarm_thr = (round(freq) - 1) * (1.0 / freq) if freq else 0.0

    # Event-only x positions (panels 1 & 3, scatter/line — irregular spacing handled natively)
    x_ev = ((t_obs_ns[index] - t_obs_ns[0]).astype(float) / (NS_F * 3600) if t_obs_ns is not None
            else index.astype(float))
    # Full-series x domain (panel 2 imshow extent — must span all n columns, not just events)
    x_full = ((0.0, (t_obs_ns[-1] - t_obs_ns[0]).astype(float) / (NS_F * 3600)) if t_obs_ns is not None
              else (0.0, float(n - 1)))
    xlabel = "Wall-clock (h from start)" if t_obs_ns is not None else "Sample index"

    corr = (act & _ROW_MASKS[1]) == 0   # row-1 unbounded-gap points, corrected

    fig, axes  = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    alarm_flag = (act & DiagBit.ALARM).any()
    fig.suptitle(
        f"Time-correction diagnostics | freq={freq:.4g}Hz | "
        f"RMS_max={np.nanmax(np.abs(dt_s)):.3f}s (×{np.nanmax(np.abs(dt_s))/max(rms_q,1e-9):.1f}·RMS_q)"
        + (" | ALARM" if alarm_flag else ""),
        fontsize=11, fontweight="bold", color="red" if alarm_flag else "black")

    # Panel 1: correction magnitude
    ax1 = axes[0]
    if corr.any():
        vmax = max(np.nanmax(np.abs(dt_s[corr])), alarm_thr * 2)
        sc   = ax1.scatter(x_ev[corr], dt_s[corr], c=np.abs(dt_s[corr]), cmap="RdYlGn_r", s=1.5,
                           norm=mcolors.LogNorm(vmin=max(1e-4, rms_q * 0.1), vmax=vmax))
        plt.colorbar(sc, ax=ax1, label="|dt|, s")
    for val, col, ls, lbl in ((rms_q, "green", "--", f"RMS_q={rms_q:.4f}s"),
                               (alarm_thr, "red", ":", f"alarm={alarm_thr:.4f}s")):
        ax1.axhline(val, color=col, lw=0.9, ls=ls, label=lbl)
        ax1.axhline(-val, color=col, lw=0.9, ls=ls)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_ylabel("t_corr − t_obs, s")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Per-sample correction (flat≈0=good; ramp=drift; gap=backward removed)", fontsize=9)

    # Panel 2: two-row diagnostic matrix — one imshow call, colour encodes full action byte
    ax2       = axes[1]
    show_diag = build_show_diag(n, index, act)
    cmap      = build_diag_cmap()
    ax2.imshow(show_diag, aspect="auto", cmap=cmap, vmin=0, vmax=127, interpolation="none",
               extent=[*x_full, -0.5, 1.5])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["properties", "fate"])
    handles = [mpatches.Patch(color=cmap.colors[bit.value], label=bit.name)
               for bit in DiagBit if (act & bit).any()]         # legend limited to bits present
    ax2.legend(handles=handles, fontsize=7, loc="upper right", ncol=len(handles) or 1)
    ax2.set_title("Row 0 properties (blend on co-occurrence) · Row 1 fate (exclusive hue)",
                  fontsize=9)

    # Panel 3: segmented cumulative correction — resets at each BACKWARD/OUT_OF_RANGE boundary
    ax3 = axes[2]
    cum = segmented_cumsum(dt_s, act)
    ax3.plot(x_ev, cum, "r-", lw=0.8)
    ax3.axhline(0, color="gray", lw=0.5)
    net     = np.where(act & _RESET_BITS, 0.0, dt_s).sum().item()   # same exclusion as segmented_cumsum
    dt_step = 1.0 / freq if freq else None
    caption = f"Net={net:.2f}s"
    if dt_step and abs(net) > dt_step:
        caption += (f"  ≈{abs(net)/dt_step:.0f} extra/missing samples"
                    f"  implied_rate≈{freq + net / (index.size / max(freq, 1e-9)):.5g}Hz")
    ax3.annotate(caption, xy=(0.02, 0.07), xycoords="axes fraction", fontsize=8)
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("Σ(correction), s")
    ax3.set_title("Segmented cumulative correction (flat=OK; ramp=drift; break=timeline reset)", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = (Path(path_save) if path_save else Path(npz_path)).with_suffix(".png")
    if path_save and Path(path_save).is_dir():
        out = Path(path_save) / out.name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    lf.info("diagnostics plot {} saved", out)
    return out


if __name__ == "__main__":
    npz_path = r"D:\WorkData\experiment\inclinometer\260604_test_format\_raw\diagnostics\@i_p1_dt.npz"
    t_obs_ns = None
    path_save = None
    plot_time_corr_diagnostics(npz_path, t_obs_ns, path_save)
