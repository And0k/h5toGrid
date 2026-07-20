"""
Calibration visualisation — 3-D ellipsoid plots and channel diagnostics.

All matplotlib code is isolated here so that core math modules
(:mod:`tcm.calibration.calibrate`, :mod:`tcm.calibration.spatial_binning`)
stay dependency-free.

Ported from ``tcm._dask_legacy.incl_calibr_hy`` (``plotEllipsoid``,
``calibrate_plot``, ``axes_connect_on_move``, ``filter_channes`` plot parts).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from tcm import utils2init
from tcm.calibration.vis_common import residual_colors, to_lonlat, _attach_external_colorbar

# Lazy matplotlib — never import at module level to keep core math fast
try:
    import matplotlib
    import matplotlib.axes
    import matplotlib.figure
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
else:
    from tcm.calibration.vis_coverage import render_spherical_voronoi
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['figure.figsize'] = (16, 7)
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        pass
    matplotlib.interactive(True)
    plt.style.use('bmh')

lf = utils2init.LoggingStyleAdapter(__name__)

# --------------------------------------------------------------------------- #
# 3-D ellipsoid rendering
# --------------------------------------------------------------------------- #

def plot_ellipsoid(
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    ax: "Optional[matplotlib.axes.Axes]" = None,
    *,
    plot_axes: bool = False,
    cage_color: str = "b",
    cage_alpha: float = 0.2,
    n_u: int = 100,
    n_v: int = 100,
) -> "matplotlib.axes.Axes":
    """Render an ellipsoid wireframe on a 3-D axes.

    Parameters
    ----------
    center
        ``(3,)`` — ellipsoid centre.
    radii
        ``(3,)`` — semi-axis lengths.
    rotation
        ``(3, 3)`` — rotation matrix.
    ax
        Target 3-D axes.  Created if ``None``.
    plot_axes
        Draw semi-axis lines.
    cage_color / cage_alpha
        Wireframe colour and transparency.
    n_u / n_v
        Angular resolution.

    Returns
    -------
    matplotlib.axes.Axes
    """
    make_ax = ax is None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)

    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            xyz = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
            x[i, j], y[i, j], z[i, j] = xyz

    if plot_axes:
        axes_vecs = np.diag(radii) @ rotation
        for p in axes_vecs:
            t = np.linspace(-1, 1, n_u)
            ax.plot(p[0] * t + center[0], p[1] * t + center[1], p[2] * t + center[2], color=cage_color)

    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=cage_color, alpha=cage_alpha)

    if make_ax:
        plt.show()
    return ax


# --------------------------------------------------------------------------- #
# Connected 3-D axes (sync camera rotation between two panels)
# --------------------------------------------------------------------------- #

def axes_connect_on_move(
    ax1: "matplotlib.axes.Axes",
    ax2: "matplotlib.axes.Axes",
) -> int:
    """Link camera rotation of two 3-D axes so rotating one rotates the other.

    Returns the ``mpl_connect`` event id.
    """
    canvas = ax1.figure.canvas

    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
        else:
            return
        canvas.draw_idle()

    return canvas.mpl_connect('motion_notify_event', on_move)


def _add_cbar(fig, ax, mappable):
    """Colorbar: tight via make_axes_locatable for standard 2-D axes only.

    ``make_axes_locatable`` does not work with 3-D axes (creates
    ``LineCollection`` without ``do_3d_projection``) or geo-projected
    axes (``set_xlim`` not supported).  For those, use ``fig.colorbar``
    which handles any axes type.
    """
    if isinstance(ax, Axes3D) or getattr(ax, "name", "") not in ("rect", ""):
        fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
    else:
        _attach_external_colorbar(ax, mappable)


# --------------------------------------------------------------------------- #
# Calibration result — 3-D source / calibrated panels
# --------------------------------------------------------------------------- #

def calibrate_plot(
    raw3d: np.ndarray,
    gain: np.ndarray,
    bias: np.ndarray,
    fig: "Optional[matplotlib.figure.Figure]" = None,
    *,
    window_title: Optional[str] = None,
    clear: bool = True,
    raw3d_other: Optional[np.ndarray] = None,
    raw3d_other_color: str = "r",
    marker_size: float = 5.0,
    projection: str = "sphere",
    field_magnitude: float = 1.0,
) -> "matplotlib.figure.Figure":
    """Two-panel plot: source data with fitted ellipsoid + calibrated on unit sphere.

    Convention: ``gain @ (raw3d - bias)`` maps samples onto the sphere
    of radius *field_magnitude* (see :func:`calibrate.calibrate`).

    Both panels use signed residuals with :class:`TwoSlopeNorm` so that
    the diverging colormap (``RdBu_r``) is centred at zero — inside vs
    outside the expected sphere is immediately visible.

    Parameters
    ----------
    raw3d
        Shape ``(3, N)`` — raw sensor data.
    gain
        ``(3, 3)`` — calibration gain matrix.
    bias
        ``(3, 1)`` — calibration bias vector.
    fig
        Reuse existing figure.  Created if ``None``.
    window_title
        Window title string.
    clear
        Clear figure before drawing.
    raw3d_other
        Shape ``(3, K)`` — points replaced by bin averaging.
    raw3d_other_color
        Colour for *raw3d_other* scatter.
    marker_size
        Scatter marker size.
    projection
        ``"sphere"`` (default) for 3-D wireframe with ellipsoid, or
        ``"mollweide"`` for 2-D map projection.
    field_magnitude
        Known reference magnitude (IGRF for M, gravity for A).
        ``1.0`` = unit sphere (default).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        lf.warning("matplotlib not available — skipping calibrate_plot")
        return None

    # ── Create or reuse figure (fig.clear removes stale colorbar axes) ──
    if fig is None:
        fig = plt.figure()
    elif clear:
        fig.clear()

    proj_kw = "mollweide" if projection == "mollweide" else "3d"
    ax1 = fig.add_subplot(121, projection=proj_kw)
    ax2 = fig.add_subplot(122, projection=proj_kw)
    fig.subplots_adjust(wspace=0.15)

    if window_title:
        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

    # DRY: signed residuals + TwoSlopeNorm via vis_common
    calibrated = gain @ (raw3d - bias)
    raw_c, cal_c, raw_n, cal_n, cmap_name = residual_colors(
        raw3d, calibrated, reference_magnitude=field_magnitude,
    )

    if projection == "mollweide":
        # ── Left panel: raw data (normalized to unit vectors) ───────────
        raw_unit = raw3d / np.maximum(np.linalg.norm(raw3d, axis=0), 1e-30)
        sc1 = ax1.scatter(
            *to_lonlat(raw_unit), c=raw_c, norm=raw_n, cmap=cmap_name,
            marker='.', s=marker_size, alpha=0.7, edgecolors="none",
        )
        _add_cbar(fig, ax1, sc1)
        ax1.set_title('Source (mean(‖r‖) − ‖r‖)')

        if raw3d_other is not None:
            other_unit = raw3d_other / np.maximum(np.linalg.norm(raw3d_other, axis=0), 1e-30)
            ax1.scatter(
                *to_lonlat(other_unit), c=raw3d_other_color, s=4, marker='.',
                alpha=0.5, edgecolors="none",
            )

        # ── Right panel: calibrated data ────────────────────────────────
        sc2 = ax2.scatter(
            *to_lonlat(calibrated / field_magnitude), c=cal_c, norm=cal_n, cmap=cmap_name,
            marker='.', s=marker_size, alpha=0.7, edgecolors="none",
        )
        _add_cbar(fig, ax2, sc2)
        ax2.set_title(f'Calibrated ({field_magnitude} − ‖cal‖)')

    else:  # sphere (3D)
        # ── Left panel: source + ellipsoid fit ───────────────────────────
        ax1.set_title('Source (mean(‖r‖) − ‖r‖)')
        sc1 = ax1.scatter(*raw3d, c=raw_c, norm=raw_n, cmap=cmap_name, marker='.', s=marker_size)
        _add_cbar(fig, ax1, sc1)

        if raw3d_other is not None:
            ax1.scatter(*raw3d_other, c=raw3d_other_color, s=4, marker='.')

        # Draw fitted ellipsoid (semi-axes = field_magnitude * singular values)
        try:
            U, s, rotation = np.linalg.svd(np.linalg.inv(gain))
            plot_ellipsoid(
                bias.flatten(), field_magnitude * s, rotation,
                ax=ax1, plot_axes=True, cage_color='r', cage_alpha=0.1,
            )
        except np.linalg.LinAlgError:
            lf.debug("Singular gain matrix — cannot plot ellipsoid")

        # ── Right panel: calibrated on sphere of radius field_magnitude ─
        ax2.set_title(f'Calibrated ({field_magnitude} − ‖cal‖)')
        sc2 = ax2.scatter(*calibrated, c=cal_c, norm=cal_n, cmap=cmap_name, marker='.', s=marker_size)
        _add_cbar(fig, ax2, sc2)

        # Reference sphere of radius field_magnitude
        plot_ellipsoid(
            np.zeros(3), field_magnitude * np.ones(3), np.eye(3),
            ax=ax2, plot_axes=True,
        )

        axes_connect_on_move(ax1, ax2)

    return fig


# --------------------------------------------------------------------------- #
# Coverage visualisation (delegates to vis_coverage.py for surface patches)
# --------------------------------------------------------------------------- #

def coverage_heatmap(
    directions: np.ndarray,
    density: np.ndarray,
    fig: "Optional[matplotlib.figure.Figure]" = None,
    *,
    projection: str = "mollweide",
    window_title: Optional[str] = None,
    sample_directions: "np.ndarray | None" = None,
    uncertainty: "dict | None" = None,
) -> "Optional[matplotlib.figure.Figure]":
    """Two-subplot figure: coverage density (left) + calibration uncertainty (right).

    Left panel shows Voronoi-patch density from :func:`robust.coverage_at`;
    right panel shows ``systematic_z_score`` from :func:`robust.uncertainty_at`
    (signed, TwoSlopeNorm centered at 0).  Both share the same query grid.

    Parameters
    ----------
    directions : ``(3, M)`` unit vectors (from :func:`robust.coverage_at`).
    density : ``(M,)`` density values at each direction.
    fig : Reuse existing figure.  Created if ``None``.
    projection : ``"mollweide"`` (default) or ``"sphere"`` (3-D).
    window_title : Figure window title.
    sample_directions : ``(3, N)`` actual calibrated sample directions
        to overlay as scatter dots.  ``None`` = no overlay.
    uncertainty : dict from :func:`robust.uncertainty_at` with keys
        ``jackknife_spread_rad``, ``noise_floor``, ``systematic_z_score``.
        ``None`` → right panel shows placeholder.

    Returns
    -------
    matplotlib.figure.Figure or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        lf.warning("matplotlib not available -> skipping coverage_heatmap")
        return None

    # ── Create two-subplot figure ───────────────────────────────────────
    if fig is None:
        fig = plt.figure()
    else:
        fig.clear()

    proj_kw = "mollweide" if projection == "mollweide" else "3d"
    ax_cov = fig.add_subplot(121, projection=proj_kw)
    ax_unc = fig.add_subplot(122, projection=proj_kw)
    fig.subplots_adjust(wspace=0.15)

    if window_title:
        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

    # ── Left: coverage density ──────────────────────────────────────────
    density_plot = np.where(density > 0, density, np.nan)
    finite = density_plot[np.isfinite(density_plot)]
    d_norm = LogNorm(vmin=max(finite.min(), 1e-10), vmax=finite.max()) if finite.size else None

    render_spherical_voronoi(
        directions, density_plot, projection=projection, cmap="viridis",
        edgecolor="none", norm=d_norm, ax=ax_cov,
    )

    if sample_directions is not None:
        if projection == "mollweide":
            ax_cov.scatter(*to_lonlat(sample_directions), c="k", s=1, alpha=0.3, edgecolors="none", zorder=5)
        elif projection == "sphere":
            ax_cov.scatter(*sample_directions, c="k", s=1, alpha=0.3, edgecolors="none", zorder=5)

    ax_cov.set_title("Coverage (density)")
    if d_norm is not None:
        _add_cbar(fig, ax_cov, plt.cm.ScalarMappable(norm=d_norm, cmap="viridis"))

    # ── Right: uncertainty (systematic z-score) ─────────────────────────
    if uncertainty is not None and "systematic_z_score" in uncertainty:
        z_score = uncertainty["systematic_z_score"]
        z_plot = np.where(np.isnan(z_score), np.nan, z_score)
        z_finite = z_plot[np.isfinite(z_plot)]
        z_norm = None
        if z_finite.size:
            z_emax = max(float(np.abs(z_finite).max()), 0.1)
            z_norm = TwoSlopeNorm(vmin=-z_emax, vcenter=0, vmax=z_emax)

        render_spherical_voronoi(
            directions, z_plot, projection=projection, cmap="RdBu_r",
            edgecolor="none", norm=z_norm, ax=ax_unc,
        )
        ax_unc.set_title("Uncertainty (systematic z-score)")
        if z_norm is not None:
            _add_cbar(fig, ax_unc, plt.cm.ScalarMappable(norm=z_norm, cmap="RdBu_r"))
    else:
        ax_unc.set_title("Uncertainty (not computed)")

    return fig


# --------------------------------------------------------------------------- #
# Channel despiking diagnostics
# --------------------------------------------------------------------------- #

def plot_despiked_channels(
    time_index: np.ndarray,
    data_3d: np.ndarray,
    *,
    mask_good: np.ndarray,
    fig: "Optional[matplotlib.figure.Figure]" = None,
    fig_save_prefix: Optional[str] = None,
    window_title: Optional[str] = None,
    labels: Sequence[str] = ("x", "y", "z"),
) -> Tuple["matplotlib.figure.Figure", np.ndarray]:
    """Plot each channel with despiked points overlaid.

    Parameters
    ----------
    time_index
        X-axis values (time or sample index).
    data_3d
        Shape ``(3, N)`` — source data.
    mask_good
        Shape ``(N,)`` boolean — ``True`` for kept points.
    fig
        Reuse existing figure.
    fig_save_prefix
        If given, save each channel plot as ``{prefix}despike({label}).png``.
    window_title
        Figure window title.
    labels
        Per-channel labels (default ``("x", "y", "z")``).

    Returns
    -------
    fig, axes
        Figure and list of 3 axes.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None, None

    axes = []
    if fig is None:
        fig, axes_list = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    else:
        axes_list = fig.axes[:3]
        for ax in axes_list:
            ax.clear()

    for i, (label, color_source, color_kept) in enumerate(
        zip(labels, ['r', 'c', 'm'], ['g', 'b', 'k'])
    ):
        ax = axes_list[i]
        ax.set_title(f'despike({label})')
        ax.plot(time_index, data_3d[i], color=color_source, alpha=0.4, label='source', linewidth=0.5)
        kept = np.ma.masked_where(~mask_good, data_3d[i].copy())
        ax.plot(time_index, kept, color=color_kept, alpha=0.7, label='kept', linewidth=0.5)
        ax.legend(prop={'size': 10}, loc='upper right')
        axes.append(ax)

        if fig_save_prefix:
            try:
                fig.savefig(f'{fig_save_prefix}despike({label}).png', dpi=300, bbox_inches="tight")
            except Exception as e:
                lf.warning("Cannot save fig: {}", utils2init.standard_error_info(e))

    return fig, np.array(axes)
