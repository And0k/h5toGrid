"""
Switchable Fibonacci / HEALPix sampling, automatic grid choice,
Voronoi patch rendering and Mollweide / 3D sphere plotting.
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
    Adapter: return (theta, phi) for pixel indices ipix using hpgeom.
    ipix can be array-like.
    """
    if not HPGEOM_AVAILABLE:
        raise RuntimeError("hpgeom not available")
    # hpgeom API: hg.pix2ang(nside, ipix, nest=False) -> (theta, phi)
    # (If your hpgeom version uses different names, adapt here.)
    theta, phi = hg.pix2ang(nside, np.asarray(ipix), nest=False)
    return np.asarray(theta), np.asarray(phi)

def boundaries_hpgeom(nside, ipix, step=1):
    """
    Adapter: return boundaries for pixels ipix.
    Expected return shape: (n_vertices, len(ipix), 3) or list of vertex arrays.
    We'll normalize to (n_vertices, npix, 3) to match earlier code.
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
    """Return unit vectors shape (3, n) for near-uniform Fibonacci sampling."""
    i = np.arange(n) + 0.5
    incl = np.arccos(1 - 2 * i / n)        # theta (0..pi)
    az = np.pi * (1 + 5 ** 0.5) * i       # golden angle times i
    x = np.sin(incl) * np.cos(az)
    y = np.sin(incl) * np.sin(az)
    z = np.cos(incl)
    return np.vstack([x, y, z])           # shape (3, n)

def choose_lonlat_res_for_fibonacci(N, alpha=4, lon_max=4096, lat_max=2048):
    M = int(np.ceil(alpha * N))
    n_lon = int(min(lon_max, max(64, int(np.round(np.sqrt(M * 2))))))
    n_lat = int(min(lat_max, max(32, int(np.ceil(M / n_lon)))))
    return n_lon, n_lat

def make_map_from_samples(vecs, values, target="grid", grid_lon_res=720, grid_lat_res=360, healpix_nside=None):
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
    """Render Voronoi patches colored by *values*.

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
