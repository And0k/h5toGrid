"""
Robust, self-assessing calibration: outlier rejection tied to the calibration goal itself (not a
generic per-axis filter), an iterative fit that logs quality at every stage, and diagnostics that
separate *where* coverage is thin from *when* something went wrong at an otherwise well-covered
direction.

The central quantity throughout is the radial residual (`radial_residuals`): once fully calibrated, a
good sample lands almost exactly on the field_magnitude-radius sphere, so how far it lands from that
sphere is a direct, calibration-goal-specific measure of "how much do I trust this point" — unlike a
generic sigma-clip on raw X/Y/Z, which knows nothing about the ellipsoid model being fit and would
reject or accept points for the wrong reasons (e.g. accepting a point with "normal-looking" raw values
that is nonetheless far from the fitted ellipsoid in the direction that matters).

This creates the same bootstrap question as direction estimation in `calibration.py`: judging outliers
needs a calibration, but outliers can distort that same calibration. `autocalibrate` resolves it the
same way (IRLS-style): fit on everything -> reject the worst offenders by a robust (MAD, not std)
threshold -> refit on what remains -> repeat. MAD matters here specifically because a few genuine
outliers inflate a std-based threshold enough to hide themselves; the median-based scale estimate does
not have that failure mode.
"""
import numpy as np
from scipy.spatial import cKDTree

from tcm.calibration import calibrate as cal
from tcm.calibration import moments
from tcm.calibration.calibrate import to_unit_vector
from tcm.calibration.moments import fibonacci_sphere
from tcm import utils2init

lf = utils2init.LoggingStyleAdapter(__name__)

MAD_TO_STD = 1.4826                                 # scale making MAD consistent with std, Gaussian data


def moment_condition_sensitivity(directions: np.ndarray, weights: np.ndarray,
                                  test_directions: np.ndarray | None = None,
                                  perturbation: float = 1e-3) -> np.ndarray:
    """
    (55,) sensitivity of the *calibrated direction* at a reference set of test points to each
    half-vectorized moment condition — how much does the predicted unit vector move if this one
    condition is perturbed, everything else held fixed? A condition the fit barely depends on gets a
    small entry, one it depends on heavily gets a large one. Feed into
    `moments.solve_optimal_weights(..., condition_weights=...)` so the fit favors matching the
    conditions the application is actually sensitive to.

    Motivation: fitting the ellipsoid well in an abstract, uniformly-weighted 55-condition sense (the
    default) is a proxy for what an inclinometer application actually needs, which is small error in
    two derived quantities specifically — gravity direction (tilt) and heading. Written as an
    objective this application actually cares about, for samples i with weights w_i, true gravity
    direction g_i and estimated g_hat_i, true/estimated azimuth psi_i/psi_hat_i, and geodesic distance
    d_S2 on the sphere:

        J_application = sum_i w_i [alpha * d_S2(g_hat_i, g_i)^2 + beta * (psi_hat_i - psi_i)^2]
                         + regularization

    Minimizing this directly is not generally possible during calibration: g_i and psi_i are exactly
    what is unknown (that is the whole calibration problem), so a fully faithful version needs joint
    latent-variable estimation over all orientations, world vectors, and sensor parameters at once
    (e.g. Bayesian MAP or a factor graph over SO(3) x S^2) rather than the independent per-sensor
    ellipsoid fit this codebase otherwise uses. This function is a minimal alternative that stays
    within the existing 55-condition fit: it reweights which of the 55 conditions matter, using local
    sensitivity as a proxy for "how much would getting this condition wrong move g_hat", rather than
    replacing the fit with a joint estimator.

    No ground truth is needed (and none exists during calibration) because this asks a *local
    sensitivity* question — how much does condition k's mismatch propagate to direction error — not
    an absolute accuracy one. d_S2(g_hat, g)'s first-order dependence on a perturbation of S is exactly
    this Jacobian, regardless of what the (unknown) true g is; no bootstrap/IRLS is needed either,
    unlike direction estimation itself (see `calibration.py`'s module docstring) — sensitivity is a
    property of the fit's local behavior around the reference S, not of the unknown true geometry.

    Scope: this covers only the tilt/direction term (alpha) of J_application above, not the azimuth
    term (beta) — the simpler of the two, since it needs only one sensor. The azimuth term needs a
    *joint* sensitivity across the accelerometer and magnetometer together (heading depends on both
    simultaneously, via `orientation._bearing`) and is not implemented here.

    Honest empirical finding, not a claimed improvement: feeding this into `condition_weights` (three
    normalizations tried: raw, sqrt-compressed, log-compressed) did not measurably improve direction
    accuracy over uniform condition weights on the severe-clustering test case (differences in the
    4th decimal place, no consistent direction across trials). Plausible reason: `solve_optimal_weights`
    already targets a *reachable* moment matrix (`restricted_moment_matrix`) that the existing uniform
    weighting fits closely, leaving little of the condition-vs-condition trade-off this is meant to
    resolve — condition weighting should matter more when the 55 conditions are in genuine tension
    (coverage too sparse/structured to hit all of them well simultaneously), which was not true of the
    cases tested. Kept as infrastructure — correctly implemented, a direct (if partial) answer to "not
    every condition matters equally for the application" — but not verified to help yet; treat as
    experimental until tested on a case with real condition-level tension.

    :param directions, weights: the calibration data and a reference weighting (e.g. one
        `moments.solve_optimal_weights` call with uniform condition_weights) whose achieved S is
        perturbed around.
    :param test_directions: (3, M) points to evaluate direction sensitivity at; default `directions`
        itself, subsampled to at most 300 for cost — representative of what will actually be measured.
    :param perturbation: relative size of each condition's perturbation (of the achieved S, not of b).
    :return: (55,) nonnegative sensitivity, NOT normalized — pass directly as condition_weights, or
        rescale first for a specific overall balance against the plain moment-matching term.
    """
    if test_directions is None:
        stride = max(1, directions.shape[1] // 300)
        test_directions = directions[:, ::stride]

    phi = moments.monomial(directions)
    reference_S = (phi * weights) @ phi.T
    Q0, n0, _ = cal._extract_quadric_from_S(reference_S)
    baseline_direction = cal._estimate_directions(test_directions, Q0, n0)

    sensitivity = np.zeros(55)
    for k, (row, col) in enumerate(zip(moments._TRIU_ROWS, moments._TRIU_COLS)):
        delta = perturbation * (abs(reference_S[row, col]) + 1e-9)
        perturbed_S = reference_S.copy()
        perturbed_S[row, col] += delta
        perturbed_S[col, row] = perturbed_S[row, col]
        try:
            Qk, nk, _ = cal._extract_quadric_from_S(perturbed_S)
        except np.linalg.LinAlgError:
            continue                                           # leave sensitivity[k] = 0: no usable signal here
        shifted_direction = cal._estimate_directions(test_directions, Qk, nk)
        angular_shift = np.arccos(np.clip((baseline_direction * shifted_direction).sum(0), -1., 1.))
        sensitivity[k] = np.mean(angular_shift ** 2) / delta ** 2

    lf.debug("moment_condition_sensitivity: range=[{:.3g}, {:.3g}] ratio={:.3g} n_test={:d}",
              sensitivity.min(), sensitivity.max(),
              sensitivity.max() / max(sensitivity.min(), 1e-300), test_directions.shape[1])
    return sensitivity


def radial_residuals(raw: np.ndarray, calibration: cal.SensorCalibration, field_magnitude: float) -> np.ndarray:
    """
    Relative deviation of each calibrated sample's radius from field_magnitude — ~0 for a sample that
    truly lies on the calibrated ellipsoid, regardless of which direction it's in.

    :param raw: (3, N) raw samples.
    :param calibration: this sensor's ellipsoid fit.
    :param field_magnitude: known reference magnitude (see `calibration.calibrate`).
    :return: (N,) (radius / field_magnitude) - 1.
    """
    radius = np.linalg.norm(calibration.a2d @ (raw - calibration.bias), axis=0)
    return radius / field_magnitude - 1.


def _mad_outlier_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean keep-mask: True where `values` is within `threshold` MAD-based scales of the median."""
    center = np.median(values)
    scale = MAD_TO_STD * np.median(np.abs(values - center))
    return np.abs(values - center) <= threshold * max(scale, 1e-12)


def reject_outliers(raw: np.ndarray, calibration: cal.SensorCalibration, field_magnitude: float,
                     mad_threshold: float = 4.) -> np.ndarray:
    """
    Keep-mask from a robust threshold on `radial_residuals` — points whose calibrated radius is
    implausibly far from field_magnitude, however "normal" their raw X/Y/Z looked individually.

    :param mad_threshold: samples beyond this many (MAD-based) scales from the median residual are
        rejected; larger = more permissive. 4 is a common, fairly conservative default (roughly
        equivalent to ~2.7 sigma for Gaussian data after the MAD_TO_STD correction).
    :return: (N,) boolean, True = keep.
    """
    keep = _mad_outlier_mask(radial_residuals(raw, calibration, field_magnitude), mad_threshold)
    n = raw.shape[1]
    n_rejected = n - keep.sum()
    lf.debug("reject_outliers: {}/{} rejected ({:.1g}%)", n_rejected, raw.shape[1], 100 * n_rejected / max(n, 1))
    return keep


def autocalibrate(raw: np.ndarray, field_magnitude: float, max_iterations: int = 5,
                   mad_threshold: float = 4., weighted: bool = True) -> tuple[cal.SensorCalibration, list[dict]]:
    """
    Fit -> reject outliers (by the fit's own residuals) -> refit -> repeat until nothing new is
    rejected or `max_iterations` is reached, logging/returning a quality snapshot at every stage.

    Works from limited data: `calibrate`'s weighting step already regularizes toward a data-driven
    baseline rather than assuming a specific N (see `moments.solve_optimal_weights`), and each
    iteration here only ever *removes* points, so it degrades gracefully (worst case, converges after
    one iteration having rejected nothing).

    :param raw: (3, N) raw samples.
    :param field_magnitude: known reference magnitude.
    :param max_iterations: upper bound on fit/reject/refit cycles.
    :param mad_threshold: see `reject_outliers`.
    :param weighted: passed through to `calibration.calibrate`.
    :return: (final calibration, history) — history[i] is a dict with n_used, n_rejected_this_round,
        residual_median, residual_p95 for iteration i, in order; inspect for convergence/quality.
    """
    keep = np.ones(raw.shape[1], dtype=bool)
    history = []
    calibration = cal.calibrate(raw, field_magnitude, weighted=weighted)   # placeholder, overwritten below
    for iteration in range(max_iterations):
        calibration = cal.calibrate(raw[:, keep], field_magnitude, weighted=weighted)
        residual = np.abs(radial_residuals(raw[:, keep], calibration, field_magnitude))
        keep_here = _mad_outlier_mask(residual, mad_threshold)
        record = {
            "iteration": iteration,
            "n_used": keep.sum().item(),
            "n_rejected_this_round": (~keep_here).sum().item(),
            "residual_median": np.median(residual).item(),
            "residual_p95": np.quantile(residual, 0.95).item(),
        }
        history.append(record)
        lf.debug(" ".join(f"{k}={v}" for k, v in record.items()))
        if keep_here.all():
            break
        indices = np.flatnonzero(keep)
        keep[indices[~keep_here]] = False
    return calibration, history


def uncertainty_at(raw: np.ndarray, field_magnitude: float, query_directions: np.ndarray | None = None,
                    n_regions: int = 20, min_region_samples: int = 30,
                    weighted: bool = True) -> tuple[np.ndarray, dict]:
    """
    Local calibration uncertainty at each of `query_directions`, decomposed into three separately
    interpretable pieces rather than a single density number (density alone cannot say whether an
    expected error comes from thin coverage, from ordinary sensor noise, or from the ellipsoid model
    not fitting well there):

    - `jackknife_spread_rad`: leave-one-region-out refit spread. The sphere is split into `n_regions`
      regions; for each, refit with that region's samples held out, then ask what direction the SAME
      raw reading (one synthesized from the full-data fit to correspond to that query direction) would
      be assigned under the region-held-out fit. Low spread means the fit near that direction does not
      depend much on any one region of data — both thin coverage (removing the *only* nearby region
      changes a lot) and unusually influential/discordant data (removing a well-covered but disagreeing
      region also changes a lot) show up here, without needing to tell those two cases apart in advance.
    - `noise_floor`: a single global number, the robust (MAD) scale of `radial_residuals` — what
      irreducible scatter to expect from sensor noise alone, everywhere, regardless of location.
    - `systematic_z_score`: at each query direction, the local mean residual among nearby samples,
      divided by what pure noise predicts for that many samples (noise_floor / sqrt(n_local)). Near 0
      means the local discrepancy looks like ordinary noise averaging out; a large value means it does
      not — a spatially coherent bias the quadric model is not capturing there, not merely imprecision
      from having few samples (which `jackknife_spread_rad` already reflects on its own).

    :param raw: (3, N) raw samples.
    :param field_magnitude: known reference magnitude.
    :param query_directions: (3, M) unit vectors; default a 200-point Fibonacci-sphere grid.
    :param n_regions: number of leave-one-out regions for the jackknife.
    :param min_region_samples: skip a region's jackknife fit if removing it would leave fewer than
        this many samples overall (fit becomes unreliable, not informative about that region).
    :param weighted: passed to each refit; `False` is substantially faster (skips `"auto"` lambda
        search each time) at some cost to how representative the spread is of the actual calibration
        used.
    :return: (query_directions used, dict), where dict fields:
        jackknife_spread_rad (M,), noise_floor (scalar),
        systematic_z_score (M,), n_local (M,, sample count within the local neighborhood used for
        systematic_z_score), local_mean_residual (M,), n_regions_used (scalar, out of n_regions)).
        See `expected_direction_error` for turning these into an actual estimated angular error.

    Honest empirical finding on the two location-resolved pieces, not both claimed with equal
    confidence: `systematic_z_score` clearly separates an injected coherent local distortion from
    ordinary noise (~27 vs ~3 in one test) — trust this one. `jackknife_spread_rad` showed only a weak
    signal in testing so far (a deliberately sparse-but-nonempty region came out ~15% higher than a
    dense one, not the large contrast expected) — a global ellipsoid fit with several hundred to a few
    thousand points is not very sensitive to removing any single region of `n_regions`, so the effect
    may need either far fewer, larger regions, or a different leave-out structure (e.g. by time window
    instead of by direction) to show up clearly; treat it as a secondary, exploratory signal for now,
    not a validated primary one. Unlike `jackknife_spread_rad`, `n_local` correctly reads as "no
    information" (0) for a completely unsampled direction, which jackknife structurally cannot detect
    (removing an already-empty region changes nothing).
    """
    query_directions = fibonacci_sphere(200) if query_directions is None else query_directions
    full_calibration = cal.calibrate(raw, field_magnitude, weighted=weighted)
    synthetic_raw = (np.linalg.solve(full_calibration.a2d, field_magnitude * query_directions)
                      + full_calibration.bias)

    sample_directions = to_unit_vector(raw, full_calibration)
    region_centers = fibonacci_sphere(n_regions)
    region_of_sample = np.argmax(region_centers.T @ sample_directions, axis=0)

    jackknife_directions = []
    region_used = np.zeros(n_regions, dtype=bool)
    for region in range(n_regions):
        keep = region_of_sample != region
        if keep.sum() < min_region_samples:
            continue
        try:
            jackknife_calibration = cal.calibrate(raw[:, keep], field_magnitude, weighted=weighted)
        except np.linalg.LinAlgError:
            continue
        jackknife_directions.append(to_unit_vector(synthetic_raw, jackknife_calibration))
        region_used[region] = True

    jackknife_directions = np.stack(jackknife_directions)                  # (n_used, 3, M)
    mean_direction = jackknife_directions.mean(0)
    mean_direction /= np.linalg.norm(mean_direction, axis=0)
    angles = np.arccos(np.clip(np.einsum("rcm,cm->rm", jackknife_directions, mean_direction), -1., 1.))
    jackknife_spread = angles.std(0)

    residual = radial_residuals(raw, full_calibration, field_magnitude)
    noise_floor = MAD_TO_STD * np.median(np.abs(residual - np.median(residual))).item()
    sample_tree = cKDTree(sample_directions.T)
    nearest_other = sample_tree.query(sample_directions.T, k=2)[0][:, 1]
    local_radius = 4 * np.quantile(nearest_other, 0.9)
    neighbors = cKDTree(query_directions.T).sparse_distance_matrix(sample_tree, max_distance=local_radius,
                                                                     output_type="coo_matrix")
    n_local = np.bincount(neighbors.row, minlength=query_directions.shape[1])
    local_mean_residual = np.bincount(neighbors.row, weights=residual[neighbors.col],
                                       minlength=query_directions.shape[1]) / np.maximum(n_local, 1)
    expected_scale = noise_floor / np.sqrt(np.maximum(n_local, 1))
    safe_scale = np.maximum(expected_scale, 1e-12)
    systematic_z_score = np.where(n_local > 0, local_mean_residual / safe_scale, np.nan)

    lf.debug(
        "uncertainty estimated from {:d} samples ({:d}/{:d} jackknife regions used): noise_floor={:.4g},"
        "median local sample count={:.1f}",
        raw.shape[1],
        (n_regions_used:=region_used.sum().item()),
        n_regions,
        noise_floor,
        np.median(n_local).item(),
    )
    return query_directions, {
        "jackknife_spread_rad": jackknife_spread,
        "noise_floor": noise_floor,
        "systematic_z_score": systematic_z_score,
        "n_local": n_local,
        "local_mean_residual": local_mean_residual,
        "n_regions_used": n_regions_used,
    }


ANGULAR_ERROR_FACTOR = 2. ** .5     # radial residual -> angle, isotropic-noise approximation (see below)


def _direction_to_tilt_azimuth_deg(direction: np.ndarray) -> tuple[float, float]:
    """Human-readable (tilt from +Z, azimuth from +X toward +Y) for one (3,) unit vector, degrees."""
    tilt = np.degrees(np.arccos(np.clip(direction[2], -1., 1.)))
    azimuth = np.degrees(np.arctan2(direction[1], direction[0])) % 360
    return tilt, azimuth


def expected_direction_error(raw: np.ndarray, field_magnitude: float,
                              query_directions: np.ndarray | None = None,
                              target_directions: np.ndarray | None = None, n_regions: int = 20,
                              min_region_samples: int = 30, weighted: bool = True) -> tuple[np.ndarray, dict]:
    """
    Estimated calibrated-direction error, in degrees, at each of `query_directions` — turns
    `uncertainty_at`'s relative diagnostics into an actual, interpretable error estimate, decomposed
    into a *precision* term (shrinks with more nearby samples — a standard error of a local mean) and
    a *systematic* term (does not shrink with more samples — a genuine bias the model is not
    capturing), combined in quadrature:

        precision_deg  ~ angular_noise / sqrt(n_local)
        systematic_deg ~ angular conversion of |local_mean_residual|
        total_deg      = sqrt(precision_deg^2 + systematic_deg^2)

    The radial-residual-to-angle conversion assumes roughly isotropic sensor noise/residual: a
    perturbation of magnitude `delta` in the calibrated reading shifts the estimated direction by about
    `|delta_perp| / field_magnitude` radians, and for isotropic `delta`, `|delta_perp| ~ sqrt(2) *
    |delta_radial|` (2 tangential dimensions against 1 radial one, equal per-dimension variance).
    Verified to within 0.1% on synthetic isotropic Gaussian noise; treat as an approximation where the
    true noise/residual structure is markedly anisotropic (e.g. one axis much noisier than the others).

    Also logs a top-level, human-readable sumomentsary (not per internal sub-call — see `uncertainty_at`
    and `moments.solve_optimal_weights` for the detailed, DEBUG-level internals this is built from):
    the worst-estimated-error direction (as tilt/azimuth, degrees) and its estimated error, separately
    over the whole query grid and, if `target_directions` is given, restricted to that region — the
    two can differ substantially (see `calibration.md`, "Choosing a target region").

    :param raw, field_magnitude, n_regions, min_region_samples, weighted: passed to `uncertainty_at`.
    :param query_directions: (3, M) reference grid; default a 300-point Fibonacci sphere (whole sphere).
    :param target_directions: (3, K) directions representing the region the device is meant to operate
        over, if narrower than the whole sphere; triggers a second, target-region-only log sumomentsary.
    :return: (query_directions, dict with precision_deg, systematic_deg, total_deg (each matching
        query_directions' column count), worst_direction (3,), worst_error_deg (scalar)).
    """
    query_directions = fibonacci_sphere(300) if query_directions is None else query_directions
    evaluate_at = query_directions if target_directions is None \
        else np.hstack([query_directions, target_directions])
    _, unc = uncertainty_at(raw, field_magnitude, evaluate_at, n_regions, min_region_samples, weighted)

    angular_noise = ANGULAR_ERROR_FACTOR * unc["noise_floor"]
    precision_deg = np.degrees(angular_noise / np.sqrt(np.maximum(unc["n_local"], 1e-6)))
    systematic_deg = np.degrees(ANGULAR_ERROR_FACTOR * np.abs(unc["local_mean_residual"]))
    total_deg = np.sqrt(precision_deg ** 2 + systematic_deg ** 2)

    def sumomentsarize(label, indices):
        worst = indices[np.argmax(total_deg[indices])]
        tilt, azimuth = _direction_to_tilt_azimuth_deg(evaluate_at[:, worst])
        dominant = "thin coverage" if precision_deg[worst] >= systematic_deg[worst] else "a local model mismatch"
        (lf.warning if total_deg[worst] > 3 * np.median(total_deg[indices]) else lf.info)(
            "calibration quality ({:s}): typical estimated direction error {:.2f}°; worst is {:.2f}° "
            "near tilt={:.1f}/azimuth={:.1f}°, mainly due to {:s}",
            label,
            np.median(total_deg[indices]),
            total_deg[worst],
            tilt,
            azimuth,
            dominant,
        )

    n_query = query_directions.shape[1]
    sumomentsarize("whole sphere", np.arange(n_query))
    if target_directions is not None:
        sumomentsarize("stated target region", np.arange(n_query, evaluate_at.shape[1]))

    worst_overall = np.argmax(total_deg[:n_query])
    result = {"precision_deg": precision_deg[:n_query], "systematic_deg": systematic_deg[:n_query],
              "total_deg": total_deg[:n_query], "worst_direction": query_directions[:, worst_overall],
              "worst_error_deg": float(total_deg[worst_overall])}
    return query_directions, result

def coverage_at(
    raw: np.ndarray,
    calibration: cal.SensorCalibration,
    query_directions: np.ndarray | int | None = None,
    bandwidth_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Note: the above function is preferred for calibration quality estimation

    Weighted local sample density at each of `query_directions` — where data is thin or absent, not
    how reliable the fit is (see `uncertainty_at` for that: density alone conflates coverage, noise,
    and model-fit quality, and cannot tell them apart — this is the "where" complement to its "how
    reliable", not a replacement for it).

    :param raw: (3, N) raw samples.
    :param calibration: this sensor's ellipsoid fit.
    :param query_directions:
      - (3, M) unit vectors to evaluate coverage at; or
      - int number of points to use with Fibonacci-sphere grid spanning the whole sphere, default: a 200-point
    :param bandwidth_fraction: as in `moments.local_density_baseline`.
    :return: (query_directions used, (M,) density — arbitrary units, compare relatively, not absolutely).
    """
    query_directions = (
        moments.fibonacci_sphere(200)
        if query_directions is None
        else moments.fibonacci_sphere(query_directions)
        if isinstance(query_directions, int)
        else query_directions
    )
    directions = cal.to_unit_vector(raw, calibration)
    sample_tree, query_tree = cKDTree(directions.T), cKDTree(query_directions.T)
    nearest_other = sample_tree.query(directions.T, k=2)[0][:, 1]
    bandwidth = bandwidth_fraction * np.quantile(nearest_other, 0.9)
    neighbors = query_tree.sparse_distance_matrix(sample_tree, max_distance=4 * bandwidth,
                                                    output_type="coo_matrix")
    density = np.bincount(
        neighbors.row,
        weights=np.exp(-neighbors.data**2 / (2 * bandwidth**2)),
        minlength=query_directions.shape[1],
    )
    lf.debug(
        "weighted direction density ({} bins): avg={:.3g} range=[{:.3g}, {:.3g}], zero at {}bins ({}samples)",
        query_directions.shape[1],
        density.mean().item(),
        density.min().item(),
        density.max().item(),
        (density == 0).sum().item(),
        raw.shape[1],
    )
    return query_directions, density


def anomalous_time_windows(raw: np.ndarray, calibration: cal.SensorCalibration, field_magnitude: float,
                            window_size: int = 50, n_direction_bins: int = 100,
                            mad_threshold: float = 4.) -> list[dict]:
    """
    Time windows whose residual is anomalously large *relative to other windows that visited nearly
    the same direction* — distinguishes "this direction is inherently hard to calibrate" (would show
    up as consistently high residual across every visit) from "something went wrong specifically
    during this time window" (transient interference, a knock, a temperature spike — high residual
    only at this visit, while other visits to the same spot were fine).

    :param raw: (3, N) raw samples, in time order (consecutive columns = consecutive in time).
    :param calibration: this sensor's ellipsoid fit.
    :param field_magnitude: known reference magnitude.
    :param window_size: samples per time window (contiguous chunks of the column order).
    :param n_direction_bins: resolution of the direction grid windows are assigned to; more bins =
        finer spatial resolution but fewer windows per bin to compare against.
    :param mad_threshold: how many (MAD-based) scales from its bin's median residual a window must
        exceed to be flagged.
    :return: list of dicts (window_index, sample_slice, residual, bin_residual_median, z_score),
        one per flagged window, sorted by |z_score| descending.
    """
    n = raw.shape[1]
    window_bounds = list(zip(range(0, n, window_size), range(window_size, n + window_size, window_size)))
    direction = cal.to_unit_vector(raw, calibration)
    residual = np.abs(radial_residuals(raw, calibration, field_magnitude))
    window_direction = np.stack([direction[:, a:b].mean(1) for a, b in window_bounds], axis=1)
    window_direction /= np.linalg.norm(window_direction, axis=0)
    window_residual = np.array([np.median(residual[a:b]) for a, b in window_bounds])

    bin_of_window = np.argmax(moments.fibonacci_sphere(n_direction_bins).T @ window_direction, axis=0)
    flagged = []
    for b in np.unique(bin_of_window):
        in_bin = np.flatnonzero(bin_of_window == b)
        if in_bin.size < 3:                              # need enough peers in this bin to judge "anomalous"
            continue
        bin_values = window_residual[in_bin]
        keep = _mad_outlier_mask(bin_values, mad_threshold)
        bin_median = np.median(bin_values)
        bin_scale = max(MAD_TO_STD * np.median(np.abs(bin_values - bin_median)), 1e-12)
        for window_index in in_bin[~keep]:
            flagged.append({
                "window_index": int(window_index), "sample_slice": window_bounds[window_index],
                "residual": float(window_residual[window_index]), "bin_residual_median": float(bin_median),
                "z_score": float((window_residual[window_index] - bin_median) / bin_scale),
            })
    flagged.sort(key=lambda item: -abs(item["z_score"]))
    lf.debug("anomalous time windows: {}/{}", len(flagged), len(window_bounds))
    return flagged
