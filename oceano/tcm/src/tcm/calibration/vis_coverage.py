"""
Draws "how well did the calibration rotation cover the sphere" plots — a colored map (flat Mollweide
or 3-D globe) showing a value (point density, calibration uncertainty, etc.) at every point on the
unit sphere. Called from :mod:`visualization` (`coverage_heatmap`) and :mod:`run`; `plot_coverage` is
also usable standalone for a quick look at any per-direction quantity.

Two unrelated things a caller needs to pick between:

Sampling grid — where the plotted values live *before* rendering: `"fibonacci"` (near-uniform,
low-discrepancy, no dependency — see `fibonacci_sphere_vectors`, same construction as
`moments.fibonacci_sphere`, duplicated here to keep this module's plotting code independent of the
core math modules) or `"healpix"` (equal-area pixels, requires the optional `hpgeom` package —
`HPGEOM_AVAILABLE` is checked before every use, functions raise plainly if it's missing).

Rendering method — how sample values become a picture: `render_spherical_voronoi` computes the exact
Voronoi tessellation of the sphere and paints each site's own cell (every point on the sphere is
covered by exactly one site's color, boundaries are exact) — best for genuinely showing *ownership*
of the whole sphere by a modest number of samples (thousands, not millions: `SphericalVoronoi` is not
cheap). `make_map_from_samples` instead assigns each cell of a regular lon/lat (or HEALPix) grid the
value of its single nearest sample by dot product — a *nearest-neighbor* fill, not an interpolation
(no smoothing between samples, no output for a query point that has no samples anywhere near it) —
cheaper and fine for a dense/large sample set where visual smoothness is not the point.

`plot_coverage` is the one function most callers want: given a sampling grid + optional per-point
`values`, it picks a sensible rendering path and returns a ready-to-show ``(fig, ax)``.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:  # lightweight HEALPix geometry library
    import hpgeom as hg
    HPGEOM_AVAILABLE = True
except Exception:
    hg = None
    HPGEOM_AVAILABLE = False

# Spherical Voronoi (fallback for patch-based rendering)
from scipy.spatial import SphericalVoronoi

# -------------------------
# Adapter functions for HEALPix geometry (hpgeom)
# -------------------------
def pix2ang_hpgeom(nside, ipix):
    """
    HEALPix pixel index(es) -> sky coordinates. Thin wrapper so the rest of this module calls one
    stable name regardless of which `hpgeom` version's argument spelling is installed.

    :param nside: HEALPix resolution parameter (higher = finer pixels; must be a power of 2).
    :param ipix: int or array-like of pixel indices.
    :return: (theta, phi) each same shape as `ipix` — theta = colatitude from +Z in [0, pi], phi =
        longitude in [0, 2*pi), both radians.
    """
    if not HPGEOM_AVAILABLE:
        raise RuntimeError("hpgeom not available")
    # hpgeom API: hg.pix2ang(nside, ipix, nest=False) -> (theta, phi)
    # (If your hpgeom version uses different names, adapt here.)
    theta, phi = hg.pix2ang(nside, np.asarray(ipix), nest=False)
    return np.asarray(theta), np.asarray(phi)

def boundaries_hpgeom(nside, ipix, step=1):
    """
    3-D corner points of each HEALPix pixel's boundary, for drawing pixels as filled polygons (see
    `plot_healpix_on_sphere_hpgeom`).

    :param nside: HEALPix resolution parameter.
    :param ipix: array-like of pixel indices to get boundaries for.
    :param step: vertices per pixel edge (1 = corners only, higher = denser polygon for pixels drawn
        large enough that a straight-edge approximation of their curved boundary would look faceted).
    :return: (n_vertices, len(ipix), 3) unit-vector corner points, `n_vertices = 4 * step`.
    """
    if not HPGEOM_AVAILABLE:
        raise RuntimeError("hpgeom not available")
    # hpgeom API: hg.boundaries(nside, ipix, step=1, nest=False) -> array (n_vertices, npix, 3)
    b = hg.boundaries(nside, np.asarray(ipix), step=step, nest=False)
    return b  # (n_vertices, npix, 3)

# -------------------------
# Existing utilities (Fibonacci, grid, Voronoi) reused
# -------------------------
def fibonacci_sphere_vectors(n: int) -> np.ndarray:
    """
    `n` points spread near-uniformly over the unit sphere (golden-angle spiral construction) — the
    default `"fibonacci"` sampling grid for `plot_coverage`.

    Same construction as `moments.fibonacci_sphere`, kept as a separate copy so this plotting module
    has no import dependency on the core math package (see module docstring) — if the two ever need
    to change, change them together.

    :param n: number of points.
    :return: (3, n) unit vectors.
    """
    i = np.arange(n) + 0.5
    incl = np.arccos(1 - 2 * i / n)        # theta (0..pi)
    az = np.pi * (1 + 5 ** 0.5) * i       # golden angle times i
    x = np.sin(incl) * np.cos(az)
    y = np.sin(incl) * np.sin(az)
    z = np.cos(incl)
    return np.vstack([x, y, z])           # shape (3, n)

def choose_lonlat_res_for_fibonacci(N, alpha=4, lon_max=4096, lat_max=2048):
    """
    Auto-pick a lon/lat grid size that roughly matches `N` Fibonacci samples' own resolution — enough
    grid cells that `make_map_from_samples` doesn't waste most samples on duplicate cells (grid too
    coarse) or leave most cells empty/aliased (grid far finer than the data can actually resolve),
    without the caller having to work out a cell count by hand for every `N`.

    :param N: number of samples the grid will receive values from (`fibonacci_sphere_vectors`'s `n`).
    :param alpha: target grid cells per sample; higher = finer grid for the same `N`.
    :param lon_max, lat_max: hard caps so a very large `N` doesn't request an unreasonably huge grid.
    :return: (n_lon, n_lat) grid resolution, each clamped to at least a usable minimum (64, 32).
    """
    M = int(np.ceil(alpha * N))
    n_lon = int(min(lon_max, max(64, int(np.round(np.sqrt(M * 2))))))
    n_lat = int(min(lat_max, max(32, int(np.ceil(M / n_lon)))))
    return n_lon, n_lat

def make_map_from_samples(vecs, values, target="grid", grid_lon_res=720, grid_lat_res=360, healpix_nside=None):
    """
    Fill a regular lon/lat grid or HEALPix map from scattered `(vecs, values)` samples by
    nearest-neighbor assignment: each output cell gets the `values` entry of whichever input `vecs`
    is closest by dot product (equivalently, smallest angular distance) — cheap (one matrix multiply
    plus argmax, no tree), but a real caveat to know before reading the picture: this is *not*
    interpolation. There is no smoothing between samples, and a cell far from every sample still gets
    some (nearest, however distant) sample's value with no indication of that distance — coverage
    gaps read as a same-colored region, not as missing data, unless the caller checks separately.

    :param vecs: (3, N) unit vectors — the samples' own directions.
    :param values: (N,) value at each sample.
    :param target: `"grid"` for a regular lon/lat grid, `"healpix"` for a HEALPix map (requires
        `hpgeom`, see module docstring).
    :param grid_lon_res, grid_lat_res: grid resolution, `target="grid"` only.
    :param healpix_nside: HEALPix resolution, `target="healpix"` only.
    :return: `target="grid"`: (lon, lat, map_vals) with `map_vals` shape (grid_lat_res, grid_lon_res).
        `target="healpix"`: `healpix_map`, shape (12 * healpix_nside**2,).
    """
    vecs = np.asarray(vecs)
    values = np.asarray(values)
    if target == "grid":
        lon = np.linspace(-np.pi, np.pi, grid_lon_res, endpoint=False)
        lat = np.linspace(-np.pi/2, np.pi/2, grid_lat_res)
        lon2d, lat2d = np.meshgrid(lon, lat)
        cx = np.cos(lat2d) * np.cos(lon2d)
        cy = np.cos(lat2d) * np.sin(lon2d)
        cz = np.sin(lat2d)
        grid_vecs = np.stack([cx.ravel(), cy.ravel(), cz.ravel()])
        dots = vecs.T @ grid_vecs
        idx = np.argmax(dots, axis=0)
        map_vals = values[idx].reshape(grid_lat_res, grid_lon_res)
        return lon, lat, map_vals
    if target == "healpix":
        # use hpgeom adapter to get pixel centers
        if not HPGEOM_AVAILABLE:
            raise RuntimeError("hpgeom not installed")
        nside = int(healpix_nside)
        npix = hg.nside2npix(nside)
        theta, phi = pix2ang_hpgeom(nside, np.arange(npix))
        px = np.sin(theta) * np.cos(phi)
        py = np.sin(theta) * np.sin(phi)
        pz = np.cos(theta)
        pix_vecs = np.vstack([px, py, pz])
        dots = vecs.T @ pix_vecs
        idx = np.argmax(dots, axis=0)
        healpix_map = values[idx]
        return healpix_map
    raise ValueError("target must be 'grid' or 'healpix'")

# -------------------------
# Voronoi rendering (exact ownership)
# -------------------------
def render_spherical_voronoi(vecs, values, projection="mollweide", cmap="viridis", edgecolor=None,
                              norm=None, ax=None):
    """Render Voronoi patches colored by *values* — exact tessellation of the whole sphere into one
    cell per site, unlike `make_map_from_samples`'s nearest-neighbor grid fill (see module docstring
    for when to reach for which). Best for a modest site count (thousands); `SphericalVoronoi`'s cost
    grows faster than the grid approach's for very large N.

    Parameters
    ----------
    vecs : ``(3, N)`` unit vectors (Voronoi sites).
    values : ``(N,)`` per-site values.  ``NaN`` entries are rendered
        in the colormap's "bad" color (set via ``cmap.set_bad``).
    projection : ``"mollweide"`` or ``"sphere"``.
    cmap : colormap name or object.
    edgecolor : patch edge color (``None`` = no edges).
    norm : optional :class:`matplotlib.colors.Normalize` instance.
        Default ``None`` → linear ``Normalize(vmin, vmax)`` over
        non-NaN values.
    ax : optional existing axes to render onto.  ``None`` → create new
        figure and axes.  Must have matching projection.
    """
    vecs = np.asarray(vecs)
    values = np.asarray(values, dtype=float)
    sites = vecs.T
    sv = SphericalVoronoi(sites, radius=1.0, center=[0, 0, 0])
    sv.sort_vertices_of_regions()

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap).copy()
    else:
        cmap_obj = cmap.copy()
    # Light gray for NaN (zero-coverage) regions
    cmap_obj.set_bad(color=(0.85, 0.85, 0.85, 1.0))

    if norm is None:
        finite = values[np.isfinite(values)]
        vmin = finite.min() if finite.size else 0.0
        vmax = finite.max() if finite.size else 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap_obj(norm(values))

    if projection == "sphere":
        created_fig = ax is None
        if created_fig:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
        polys = []
        facecolors = []
        for region_idx, region in enumerate(sv.regions):
            polys.append(sv.vertices[region])
            facecolors.append(colors[region_idx])
        coll = Poly3DCollection(polys, facecolors=facecolors, linewidths=0.1, edgecolors=edgecolor, alpha=1.0)
        ax.add_collection3d(coll)
        ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
        ax.set_box_aspect([1, 1, 1])
        if created_fig:
            ax.set_axis_off()
        return fig, ax

    if projection == "mollweide":
        import matplotlib.patches as mpatches
        created_fig = ax is None
        if created_fig:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection="mollweide")
        else:
            fig = ax.figure
        for i, region in enumerate(sv.regions):
            verts = sv.vertices[region]
            lon = np.arctan2(verts[:, 1], verts[:, 0])
            lat = np.arcsin(verts[:, 2])
            poly_coords = np.column_stack([lon, lat])
            patch = mpatches.Polygon(poly_coords, closed=True, facecolor=colors[i], edgecolor=edgecolor,
                                     linewidth=0.2)
            ax.add_patch(patch)
        return fig, ax
    raise ValueError("unknown projection")

# -------------------------
# HEALPix plotting via hpgeom boundaries (3D)
# -------------------------
def plot_healpix_on_sphere_hpgeom(healpix_map, nside, cmap="viridis", title=None, elev=30, azim=60):
    """
    Render a HEALPix map as filled pixel polygons on a 3-D globe (`boundaries_hpgeom` for the pixel
    outlines) — the HEALPix-grid counterpart of `render_spherical_voronoi`'s Fibonacci/Voronoi
    rendering; called by `plot_coverage` for `method="healpix", plot_mode` other than `"mollweide"`.

    :param healpix_map: (12 * nside**2,) value per HEALPix pixel (see `make_map_from_samples`).
    :param nside: HEALPix resolution matching `healpix_map`.
    :param cmap: colormap name.
    :param title: optional plot title.
    :param elev, azim: initial 3-D view angle (degrees).
    :return: (fig, ax).
    """
    if not HPGEOM_AVAILABLE:
        raise RuntimeError("hpgeom not installed")
    npix = hg.nside2npix(nside)
    # hpgeom.boundaries returns (n_vertices, npix, 3)
    b = boundaries_hpgeom(nside, np.arange(npix), step=1)
    verts = b.transpose(1,0,2)   # (npix, n_vertices, 3)
    vals = healpix_map
    norm = plt.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(vals))
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    polys = Poly3DCollection(verts, facecolors=colors, linewidths=0, edgecolors=None)
    ax.add_collection3d(polys)
    ax.auto_scale_xyz([-1,1], [-1,1], [-1,1])
    mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    mappable.set_array(vals)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label="coverage")
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return fig, ax

# -------------------------
# High-level API (uses hpgeom if method='healpix')
# -------------------------
def plot_coverage(method="fibonacci", n_or_nside=1024, values=None,
                  plot_mode="mollweide", grid_res=None, alpha_grid=4,
                  patch_based=False, cmap="viridis", edgecolor=None):
    """
    One-call entry point: build a sampling grid, optionally fill it with `values`, and render a
    coverage map. For a quick look at any per-direction quantity, this is normally the only function
    in this module a caller needs; the pieces it wires together (grid choice, rendering method) are
    documented individually below and in the module docstring if finer control is needed.

    :param method: `"fibonacci"` (default, no extra dependency) or `"healpix"` (needs `hpgeom`) — see
        module docstring.
    :param n_or_nside: `method="fibonacci"`: number of sample points. `method="healpix"`: HEALPix
        `nside`.
    :param values: (N,) value per grid point, N matching `method`'s point count; None (default) =
        `(z + 1) / 2` — a smooth placeholder gradient from south to north pole, useful for sanity-
        checking the grid/rendering itself before real data is available.
    :param plot_mode: `"mollweide"` (default, flat all-sky map) or `"sphere"` (3-D globe).
    :param grid_res: `method="fibonacci"` only, `patch_based=False` only: explicit (lon_res, lat_res)
        for `make_map_from_samples`; None (default) = auto-chosen by `choose_lonlat_res_for_fibonacci`.
    :param alpha_grid: `grid_res=None` only: passed through as `choose_lonlat_res_for_fibonacci`'s
        `alpha` (grid density relative to point count).
    :param patch_based: `method="fibonacci"` only: True renders exact Voronoi cells
        (`render_spherical_voronoi`) instead of a nearest-neighbor grid fill — see module docstring
        for the trade-off; ignored (grid fill always used) for `method="healpix"`.
    :param cmap, edgecolor: passed through to whichever rendering function is used.
    :return: (fig, ax) or, for `render_spherical_voronoi`/`plot_healpix_on_sphere_hpgeom`, whatever
        those return (same `(fig, ax)` shape) — None if `method="healpix"` and `hpgeom` is missing.
    """
    vecs = None
    if method == "fibonacci":
        vecs = fibonacci_sphere_vectors(int(n_or_nside))
    elif method == "healpix":
        if not HPGEOM_AVAILABLE:
            raise RuntimeError("hpgeom not installed; install hpgeom or use fibonacci")
        nside = int(n_or_nside)
        npix = hg.nside2npix(nside)
        theta, phi = pix2ang_hpgeom(nside, np.arange(npix))
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vecs = np.vstack([x, y, z])
    else:
        raise ValueError("unknown method")

    N = vecs.shape[1]
    if values is None:
        values = (vecs[2] + 1) / 2.0
    values = np.asarray(values)
    assert values.shape[0] == N

    if method == "fibonacci":
        if patch_based:
            return render_spherical_voronoi(vecs, values, projection=plot_mode, cmap=cmap, edgecolor=edgecolor)
        if grid_res is None:
            lon_res, lat_res = choose_lonlat_res_for_fibonacci(N, alpha=alpha_grid)
        else:
            lon_res, lat_res = grid_res
        lon, lat, map_vals = make_map_from_samples(vecs, values, target="grid",
                                                  grid_lon_res=lon_res, grid_lat_res=lat_res)
        title = f"Fibonacci N={N} grid {lon_res}x{lat_res}"
        if plot_mode == "mollweide":
            return plot_mollweide_from_grid(lon, lat, map_vals, cmap=cmap, title=title)
        else:
            return plot_sphere_from_grid(lon, lat, map_vals, cmap=cmap, title=title)

    if method == "healpix":
        nside = int(n_or_nside)
        healpix_map = make_map_from_samples(vecs, values, target="healpix", healpix_nside=nside)
        if plot_mode == "mollweide":
            # hpgeom does not provide a direct mollview; use pcolormesh on lon/lat grid
            # convert healpix_map to lon/lat grid for Mollweide plotting
            lon_res, lat_res = 720, 360
            lon = np.linspace(-np.pi, np.pi, lon_res, endpoint=False)
            lat = np.linspace(-np.pi/2, np.pi/2, lat_res)
            # map healpix pixels to grid by nearest pixel center
            lon2d, lat2d = np.meshgrid(lon, lat)
            cx = np.cos(lat2d) * np.cos(lon2d)
            cy = np.cos(lat2d) * np.sin(lon2d)
            cz = np.sin(lat2d)
            grid_vecs = np.stack([cx.ravel(), cy.ravel(), cz.ravel()])
            # pixel centers from hpgeom
            theta, phi = pix2ang_hpgeom(nside, np.arange(hg.nside2npix(nside)))
            px = np.sin(theta) * np.cos(phi)
            py = np.sin(theta) * np.sin(phi)
            pz = np.cos(theta)
            pix_vecs = np.vstack([px, py, pz])
            dots = pix_vecs.T @ grid_vecs  # (npix, M)
            idx = np.argmax(dots, axis=0)
            map_vals = healpix_map[idx].reshape(lat_res, lon_res)
            return plot_mollweide_from_grid(lon, lat, map_vals, cmap=cmap, title=f"HEALPix nside={nside}")
        else:
            return plot_healpix_on_sphere_hpgeom(healpix_map, nside, cmap=cmap, title=f"HEALPix nside={nside} Sphere")

# -------------------------
# Helper plotting functions reused (pcolormesh / sphere)
# -------------------------
def plot_mollweide_from_grid(lon, lat, map_vals, cmap="viridis", title=None):
    """
    Flat all-sky map of a regular lon/lat grid (from `make_map_from_samples`) via
    :meth:`~matplotlib.axes.Axes.pcolormesh` on a Mollweide projection — the `plot_mode="mollweide"`
    rendering path for `method="fibonacci", patch_based=False` in `plot_coverage`.

    :param lon: (n_lon,) grid longitudes, radians (cell centers).
    :param lat: (n_lat,) grid latitudes, radians (cell centers).
    :param map_vals: (n_lat, n_lon) value per grid cell.
    :param cmap: colormap name.
    :param title: optional plot title.
    :return: (fig, ax).
    """
    lon_edges = np.concatenate([lon, [lon[0] + 2*np.pi/len(lon)]])
    lat_step = lat[1] - lat[0]
    lat_edges = np.linspace(lat[0] - lat_step/2, lat[-1] + lat_step/2, len(lat)+1)
    LonE, LatE = np.meshgrid(lon_edges, lat_edges)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection="mollweide")
    pcm = ax.pcolormesh(LonE, LatE, map_vals, cmap=cmap, shading="flat")
    fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, label="coverage")
    if title:
        ax.set_title(title)
    return fig, ax

def plot_sphere_from_grid(lon, lat, map_vals, cmap="viridis", title=None, elev=30, azim=60):
    """
    3-D-globe counterpart of `plot_mollweide_from_grid`: same regular lon/lat grid, rendered as a
    colored surface on a sphere instead of a flat projection — the `plot_mode="sphere"` rendering
    path for `method="fibonacci", patch_based=False` in `plot_coverage`.

    :param lon: (n_lon,) grid longitudes, radians.
    :param lat: (n_lat,) grid latitudes, radians.
    :param map_vals: (n_lat, n_lon) value per grid cell.
    :param cmap: colormap name.
    :param title: optional plot title.
    :param elev, azim: initial 3-D view angle (degrees).
    :return: (fig, ax).
    """
    Lon, Lat = np.meshgrid(lon, lat)
    X = np.cos(Lat) * np.cos(Lon)
    Y = np.cos(Lat) * np.sin(Lon)
    Z = np.sin(Lat)
    norm = plt.Normalize(vmin=np.nanmin(map_vals), vmax=np.nanmax(map_vals))
    cmap_obj = plt.get_cmap(cmap)
    facecolors = cmap_obj(norm(map_vals))
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=facecolors,
                           linewidth=0, antialiased=False, shade=False)
    mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    mappable.set_array(map_vals)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label="coverage")
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return fig, ax

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example 1: Fibonacci Voronoi on Mollweide (exact ownership regions)
    fig, ax = plot_coverage(method="fibonacci", n_or_nside=512, patch_based=True, plot_mode="mollweide")
    plt.show()

    # Example 2: Fibonacci grid auto-chosen resolution, sphere plot
    fig, ax = plot_coverage(method="fibonacci", n_or_nside=2000, patch_based=False, plot_mode="sphere")
    plt.show()

    # HEALPix via hpgeom (if available)
    if HPGEOM_AVAILABLE:
        fig, ax = plot_coverage(method="healpix", n_or_nside=32, plot_mode="mollweide")
        plt.show()
    else:
        print("hpgeom not installed — install hpgeom to use HEALPix mode.")
