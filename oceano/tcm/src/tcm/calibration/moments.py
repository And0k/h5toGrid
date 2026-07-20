"""
Sample-weighting scheme for Li-Griffiths ellipsoid fitting under non-uniform angular coverage.

Multi-position rotation calibration (rotate about a fixed axis, reposition, repeat) samples the
sensor's response sphere very unevenly: dense along each rotation's track, sparse between axes,
sparser still near shared poles. `fit_quadric_form`'s S = D @ D.T treats every sample as equally
informative, so this unevenness biases the fit exactly as unweighted least squares biases any
statistic computed under a skewed sampling design.

The correction below does *not* flatten point *density* (w_i ~ 1/local_density — simple, but blind to
what the fit actually needs) — it solves for weights so the discrete moment matrix S = sum(w_i * phi_i
@ phi_i.T) matches the specific moments the Li-Griffiths generalized eigenproblem depends on, i.e. the
continuous-sphere target

    S_inf = integral_{S^2} phi(Omega) @ phi(Omega).T dOmega

available in closed form (`analytic_moment_matrix`) since phi's entries are degree <=4 monomials and
all odd sphere moments vanish by symmetry. Choosing w to solve min || sum(w_i phi_i phi_i.T) - target ||
is then a small (55-condition, see `outer_product`) linear least-squares problem in the — potentially
huge — w; see `solve_optimal_weights` for why that under-determined system needs regularizing.

`target` is *not* simply `analytic_moment_matrix()` (the whole sphere's moments): `build_linear_system`
uses `restricted_moment_matrix` instead, which targets whatever region the data actually reaches. This
is not a refinement, it is load-bearing: matching the whole sphere on data that only covers part of it
(a hemisphere, say, because the rig genuinely cannot be flipped past 90 degrees of tilt) does not just
fail to help — the optimizer chases moments that are analytically unreachable with same-signed data
(e.g. an odd-in-z moment that must be 0 over S^2 but never can be when every sample has z > 0), and the
resulting weights actively degrade the fit. Confirmed empirically at >20x worse calibration bias than
not weighting at all, on genuinely hemisphere-only coverage (see `test_calibration.py`'s regression
test) — fixed by detecting the reached region numerically rather than assuming it is everything.

This module only consumes *directions* (unit vectors); it has no notion of sensor bias/scale/soft-iron
and cannot see the (yet unknown) calibration, so it carries none of the model-circularity risk discussed
where directions are actually estimated from raw data (see `calibration.py`'s module docstring).

Nomenclature is the same problem survey statistics calls *calibration weighting*: adjust design weights
minimally so they reproduce known auxiliary totals [2]; our "totals" are sphere (or region-restricted)
moments instead of census figures. It also connects to the well-documented statistical bias of algebraic
conic/quadric fitting under noisy, non-uniform sampling [3], which is what makes weighting matter at
all — see the integration test in `test_calibration.py` for a direct empirical measurement of that bias
and how much of it the weighting removes.

References
----------
.. [1] Qingde Li, J.G. Griffiths, "Least squares ellipsoid specific fitting", GMP 2004, pp.335-340.
.. [2] J.C. Deville, C.E. Saerndal, "Calibration Estimators in Survey Sampling", JASA 87(418), 1992.
.. [3] K. Kanatani, "Statistical Bias of Conic Fitting and Renormalization", IEEE PAMI 16(3), 1994.
.. [4] Sphere moment closed form: https://en.wikipedia.org/wiki/Solid_harmonics#Moments
"""
import logging

import numpy as np
from scipy import optimize, special
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

SPHERE_AREA = 4 * np.pi                                     # integral_{S^2} dOmega; also target Sum(w_i)
MIN_EFFECTIVE_FRACTION = 0.1                          # solve_optimal_weights("auto"): min accepted n_eff/N

# phi(x) = [x^2, y^2, z^2, 2yz, 2xz, 2xy, 2x, 2y, 2z, 1] — single source of truth for both `monomial`
# (numeric) and `analytic_moment_matrix` (symbolic exponents), so the two can't drift apart.
_PHI_EXPONENTS = np.array([
    [2, 0, 0], [0, 2, 0], [0, 0, 2], [0, 1, 1], [1, 0, 1],
    [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
])                                                         # (10, 3): (px, py, pz) exponents per monomial
_PHI_COEFS = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 1])      # (10,): each monomial's scalar prefactor

_TRIU_ROWS, _TRIU_COLS = np.triu_indices(10)              # 55 unique entries of symmetric 10x10 (w/ diag)
_TRIU_SCALE = np.where(_TRIU_ROWS == _TRIU_COLS, 1., np.sqrt(2))   # off-diag counted twice in full matrix


def monomial(point: np.ndarray) -> np.ndarray:
    """
    phi(x) = [x^2, y^2, z^2, 2yz, 2xz, 2xy, 2x, 2y, 2z, 1] — Li-Griffiths [1] quadric-form basis.

    :param point: (3, N) — x, y, z rows, N samples.
    :return: (10, N) phi, one column per sample.
    """
    powers = point[np.newaxis] ** _PHI_EXPONENTS[..., np.newaxis]     # (10,3,N): coord**exp per monomial
    return _PHI_COEFS[:, np.newaxis] * powers.prod(axis=1)


def outer_product(phi: np.ndarray) -> np.ndarray:
    """
    Half-vectorize phi @ phi.T per sample: unique upper-triangular entries, sqrt(2)-scaled off-diagonal
    so that ``norm(outer_product(phi))**2 == norm(phi @ phi.T, 'fro')**2`` exactly (standard symmetric
    "vech" isometry, e.g. Magnus & Neudecker, *Matrix Differential Calculus*, Ch. 3).

    :param phi: (10, N) monomial vectors (see `monomial`).
    :return: (55, N) half-vectorized phi @ phi.T, one column per sample.
    """
    return phi[_TRIU_ROWS] * phi[_TRIU_COLS] * _TRIU_SCALE[:, np.newaxis]


def analytic_moment_matrix() -> np.ndarray:
    """
    b = vec(S_inf) = integral_{S^2} phi(Omega) @ phi(Omega).T dOmega, half-vectorized to match
    `outer_product` — the exact target a perfectly uniform sample reproduces in expectation;
    independent of any actual data (see module docstring: no model-circularity here).

    Closed form for sphere moments — all odd ones vanish by the sphere's reflection symmetry [4]:
        integral_{S^2} x**p y**q z**r dOmega = 4*pi * (p-1)!!(q-1)!!(r-1)!! / (p+q+r+1)!!
    for even p, q, r (else 0), convention (-1)!! = 1.

    :return: (55,) analytic b.
    """
    exponents = _PHI_EXPONENTS[_TRIU_ROWS] + _PHI_EXPONENTS[_TRIU_COLS]    # (55,3): summed (px,py,pz) pair
    coefs = _PHI_COEFS[_TRIU_ROWS] * _PHI_COEFS[_TRIU_COLS]               # (55,): matching scalar prefactor
    vanishes = (exponents % 2 != 0).any(axis=-1)                            # any odd exponent => zero moment
    shifted = np.where(exponents == 0, 1, exponents - 1)                # (p-1); 0->1 => factorial2(1)=1
    moment = 4 * np.pi * special.factorial2(shifted).prod(axis=-1) / special.factorial2(exponents.sum(-1) + 1)
    return coefs * _TRIU_SCALE * np.where(vanishes, 0., moment)


def restricted_moment_matrix(directions: np.ndarray, n_reference: int = 8000,
                              coverage_bandwidths: float = 4.) -> np.ndarray:
    """
    b = vec(S_R), R being whatever region `directions` spans — generalizes `analytic_moment_matrix`
    (R = S^2, the whole sphere) to an arbitrary R, detected numerically: a dense reference grid is kept
    only where it falls within `coverage_bandwidths` of the typical inter-sample spacing from at least
    one point of `directions`, then phi @ phi.T is averaged *uniformly* (not density-weighted) over
    what survives and scaled by R's own measured solid angle.

    Two distinct uses, not to be conflated (see `build_linear_system`'s docstring): (a) pass a set of
    directions *chosen to represent the region the device is meant to operate over* (which need not be
    the calibration data) as an explicit `target` there; (b) call this directly on the calibration data
    itself only when that data's own coverage *is* the intended operating envelope (e.g. an in-situ
    auto-calibration whose only interest is the region it happens to visit — see `calibration.md`).
    Reduces to `analytic_moment_matrix()` (up to grid density) once `directions` spans the whole sphere.

    :param directions: (3, N) unit vectors spanning the region of interest.
    :param n_reference: reference grid size; larger = finer detection of the region's boundary, at
        proportionally more cost (a single cKDTree query, still cheap next to `solve_optimal_weights`).
    :param coverage_bandwidths: how many local-spacing units from the nearest point of `directions` a
        reference point may be and still count as "covered". Spacing is *per sample*, not one global
        number (each sample's own distance to its 2nd-nearest neighbor): a single global spacing
        estimate breaks as soon as coverage is a mix of dense and sparse regions, in either direction
        — dominated by a dense cluster, it wrongly shrinks the reach credited to genuinely sparse
        areas; dominated by sparse background, it wrongly bridges real gaps next to a dense cluster.
        This also handles near-duplicate points that differ only by measurement noise (not exactly
        identical) without needing a separate noise-floor parameter: their own 2nd-nearest-neighbor
        distance is small either way, so the *local* radius they contribute shrinks correctly without
        distorting the spacing estimate used elsewhere on the sphere.
    :return: (55,) numerically-integrated b, on the same half-vectorized convention as `outer_product`.
    """
    sample_tree = cKDTree(directions.T)
    local_spacing = sample_tree.query(directions.T, k=min(3, directions.shape[1]))[0][:, -1]
    reference = fibonacci_sphere(n_reference)
    nearest_distance, nearest_sample = sample_tree.query(reference.T, k=1)
    covered = reference[:, nearest_distance <= coverage_bandwidths * local_spacing[nearest_sample]]
    solid_angle_per_point = SPHERE_AREA / n_reference
    log.debug("estimated angular coverage from %d samples: %.1f%% of the sphere reached",
               directions.shape[1], 100 * covered.shape[1] / n_reference)
    return solid_angle_per_point * outer_product(monomial(covered)).sum(axis=1)


def achievable_condition_weights(directions: np.ndarray, target: np.ndarray,
                                  discrepancy_threshold: float = 0.5) -> np.ndarray:
    """
    (55,) near-binary weight per moment condition: ~1 where `target` is realistically reachable given
    `directions`' own coverage, ~0 where it structurally is not (e.g. an odd-in-z moment that must be
    exactly 0 over the whole sphere but never can be if every sample has the same-sign z) — feed into
    `solve_optimal_weights`'s `condition_weights` so the optimizer stops distorting weights chasing
    conditions it cannot satisfy, while it still pulls the reachable ones toward `target` properly (see
    `build_linear_system`'s docstring for why this differs from just substituting a smaller target).

    Reachability is measured against `restricted_moment_matrix(directions)` (what a uniform sample of
    directions' own coverage naturally produces), not a per-condition min/max range check over the raw
    samples: that range is nearly always wide enough to technically contain `target` and so is a weak
    signal in practice (confirmed empirically — it leaves the fit essentially as distorted as no
    weighting at all), whereas the region-level comparison correctly separates the two clearly-bimodal
    populations that show up in practice (confirmed empirically — see `test_moments.py`): conditions
    close to naturally achievable, and conditions saturated at maximal discrepancy. A soft power-law
    downweighting was tried first and does not work for the same reason: discrepancy saturates near a
    fixed ceiling for the unreachable population, so raising the power leaves them practically
    unchanged (x close to 1 stays close to 1 under x**n for any n).

    :param directions: (3, N) unit vectors (the calibration data).
    :param target: (55,) target moments (e.g. `analytic_moment_matrix()` for "the whole sphere", or
        `restricted_moment_matrix()` of a *different* set of directions representing the region the
        device is meant to operate over).
    :param discrepancy_threshold: relative-discrepancy cutoff (in [0, 1]) between `target` and what
        `directions`' own coverage naturally produces, above which a condition is treated as
        unreachable; 0.5 separated the two populations cleanly in every case tested so far.
    :return: (55,) weights, `discrepancy_threshold`-thresholded, not smoothly graded.
    """
    achievable = restricted_moment_matrix(directions)
    relative_discrepancy = np.abs(target - achievable) / (np.abs(target) + np.abs(achievable) + 1e-12)
    unreachable = relative_discrepancy > discrepancy_threshold
    if unreachable.any():
        log.debug("%.0f%% of the fit's internal constraints cannot be matched by the current angular "
                   "coverage; calibration accuracy will be reduced in under-sampled directions",
                   100 * unreachable.mean())
    return np.where(unreachable, 1e-6, 1.)


def build_linear_system(directions: np.ndarray,
                         target: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble A w ~ b: A's N columns are each direction's half-vectorized phi @ phi.T, b is `target` —
    what their weighted sum should reproduce.

    `target` defaults to `analytic_moment_matrix()` (the whole sphere) rather than being inferred from
    `directions`' own coverage: the region a device needs to be calibrated *for* is a statement about
    the device's intended use, not something to read off of whatever the calibration data happened to
    cover — silently doing the latter would hide a real coverage shortfall behind a small residual
    instead of surfacing it (see `calibration.md`, "Choosing a target region"). Pass an explicit
    `target` — e.g. `restricted_moment_matrix()` of a *separately chosen* set of directions describing
    where the device actually operates — when the whole sphere genuinely is not the right goal.

    Whatever `target` is, some of its 55 conditions may be unreachable given `directions`' actual
    coverage; that is expected and is `solve_optimal_weights`'s `condition_weights` argument's job to
    handle (see `achievable_condition_weights`), not this function's.

    :param directions: (3, N) unit vectors — see `calibration._estimate_directions` for how these are
        obtained without circularly assuming the (unknown) ellipsoid is already a sphere.
    :param target: (55,) target moments; None (default) = the whole sphere.
    :return: A (55, N), b (55,) — ready for `achievable_condition_weights`/`solve_optimal_weights`.
    """
    target = analytic_moment_matrix() if target is None else target
    return outer_product(monomial(directions)), target


def fibonacci_sphere(n: int) -> np.ndarray:
    """Near-uniform point set on S^2 (Fibonacci-sphere construction) — a standard low-discrepancy
    default, used both as a query grid (`robust_calibration.expected_direction_error`) and in tests."""
    i = np.arange(n) + .5
    inclination = np.arccos(1 - 2 * i / n)
    azimuth = np.pi * (1 + 5 ** .5) * i
    return np.stack([np.sin(inclination) * np.cos(azimuth),
                      np.sin(inclination) * np.sin(azimuth),
                      np.cos(inclination)])


def local_density_baseline(directions: np.ndarray, bandwidth_fraction: float = 0.25,
                            cutoff_bandwidths: float = 4., spacing_quantile: float = 0.9) -> np.ndarray:
    """
    w0_i ~ 1 / (Gaussian-kernel density at direction i), normalized to sum to 4*pi — the standard
    inverse-density weighting heuristic (see calibration_wiki.md, "Local density baseline"), used
    *only* as `solve_optimal_weights`'s ridge baseline, never as the final answer: flattening local
    density is not quite the same target as matching Li-Griffiths' specific moments (see module
    docstring) — e.g. it has no way to know that x^4's moment matters to the fit while x^3*y's doesn't
    (always 0 on a sphere, regardless of sampling). Depends only on the sample's own geometry, so it
    needs no estimate of the (unknown) ellipsoid at all.

    Density uses *all* neighbors within `cutoff_bandwidths` * bandwidth (scipy's sparse distance
    matrix, not a fixed k): a fixed-k version saturates once a duplicated cluster exceeds k (every
    member then sees k identical neighbors regardless of how many more exist), letting a cluster's
    *total* baseline weight grow with its size instead of staying fixed.

    `bandwidth` is `bandwidth_fraction` times the `spacing_quantile`-th quantile of nearest-neighbor
    distance, not the median: with the median, once a dense cluster or near-duplicate group is close
    to (or over) half the sample, it dominates the statistic and collapses the bandwidth toward that
    cluster's own fine internal spacing, which then fails to resolve the cluster from the sparser
    background at all (confirmed empirically: a 500-spread + 4500-clustered mix gave cluster and spread
    points *equal* per-point weight at the median, versus a ~10x ratio — correctly reflecting the
    density difference — at the 90th percentile). A high quantile is not immune to the same failure
    once the dense population's share exceeds it (e.g. >90% concentrated onto one spot with the
    default): raise `spacing_quantile` for known-heavier concentration, or supply the true noise/
    duplication scale directly by pre-filtering `directions` if it is known independently.

    This bandwidth choice also covers near-duplicate points that differ only by measurement noise (not
    exactly identical): their pairwise distances are small but nonzero, so a raw "exclude distance 0"
    filter alone would not catch them, but the quantile is computed over the *whole* nearest-neighbor
    distribution regardless of whether the small values are exactly 0 or merely noise-small.

    :param directions: (3, N) unit vectors.
    :param bandwidth_fraction: kernel bandwidth as a fraction of the reference spacing.
    :param cutoff_bandwidths: neighbor search radius, in bandwidths (Gaussian tail beyond ~4 is negligible).
    :param spacing_quantile: quantile of nearest-neighbor distance defining the reference spacing.
    :return: (N,) positive baseline weights summing to 4*pi.
    """
    tree = cKDTree(directions.T)
    nearest_other = tree.query(directions.T, k=2)[0][:, 1]
    bandwidth = bandwidth_fraction * np.quantile(nearest_other, spacing_quantile)
    neighbors = tree.sparse_distance_matrix(tree, max_distance=cutoff_bandwidths * bandwidth,
                                             output_type="coo_matrix")
    kernel = np.exp(-neighbors.data ** 2 / (2 * bandwidth ** 2))
    density = np.bincount(neighbors.row, weights=kernel, minlength=directions.shape[1])
    inverse_density = 1. / density
    return SPHERE_AREA * inverse_density / inverse_density.sum()


def _solve_for_lambda(A: np.ndarray, b: np.ndarray, baseline: np.ndarray, relative_regularization: float,
                       condition_weights: np.ndarray):
    """Core solve at one fixed lambda — see `solve_optimal_weights` for the objective/method."""
    c = A @ baseline - b

    def objective_and_gradient(u):
        residual = c + A @ (baseline * u)
        weighted_residual = condition_weights * residual
        objective = residual @ weighted_residual + relative_regularization * (u @ u)
        gradient = 2 * baseline * (A.T @ weighted_residual) + 2 * relative_regularization * u
        return objective, gradient

    result = optimize.minimize(
        objective_and_gradient,
        x0=np.zeros(A.shape[1]),
        jac=True,
        method="L-BFGS-B",
        bounds=optimize.Bounds(-1.0, np.inf),
    )
    if not result.success:
        log.warning("sample-weight optimization did not fully converge at regularization=%.2g (%s); "
                     "resulting weights may be unreliable", relative_regularization, result.message)
    return baseline * (1 + result.x), result.nit


def solve_optimal_weights(A: np.ndarray, b: np.ndarray, baseline: np.ndarray,
                           relative_regularization: float | str = "auto",
                           condition_weights: np.ndarray | None = None) -> np.ndarray:
    """
    min_w  ||A w - b||^2 + lambda * ||w/w0 - 1||^2   s.t.  w >= 0
    reparametrized as u = w/w0 - 1 (so bounds become the simple u >= -1), w0 = `baseline`.

    Plain NNLS on this system (55 conditions, N >> 55 unknowns) is under-determined and — like any
    least-norm solver facing a wide, rank-deficient A — collapses onto a *sparse* vertex solution using
    only a handful of points (confirmed empirically in the test suite): the opposite of what we want,
    which is a gentle density correction that keeps using all the data. The ridge term pulls the
    solution back toward w0, allowing only as much deviation as the moment mismatch actually needs —
    survey statistics' "distance from design weights" idea [2] (see module docstring), here with an L2
    distance so the objective is smooth and cheaply solvable at N in the tens of thousands via L-BFGS-B
    (O(55 N) per gradient step, memory O(N): no N x N matrix is ever formed).

    A *global* uniform baseline (w0 = 4*pi/N for everyone) looks tempting but is wrong: it prices
    suppressing K near-duplicate columns at K times the cost of suppressing one, so a duplicated
    cluster's total weight grows with K instead of staying fixed (empirically confirmed while
    developing this — see test suite). `baseline` must instead depend on local sample geometry —
    `local_density_baseline` is the natural default.

    A single fixed lambda is not universal: data concentrated onto few axes (large near-duplicate
    clusters from repeated spins at a handful of settings, say) needs a *weaker* lambda than the same
    N spread continuously, or it under-fits the moments badly for no good reason (confirmed
    empirically — comparable-looking protocols differed by 10x+ in achievable residual at one fixed
    lambda, resolved by tuning lambda down). `"auto"` (default) picks, from a log-spaced sweep, the
    smallest lambda for which the effective sample size stays above `MIN_EFFECTIVE_FRACTION` of N —
    as much moment-fit tightness as the regularization budget allows, without over-concentrating
    weight onto too few points.

    `condition_weights` lets the 55 moment conditions matter unequally — e.g. weighting toward
    whichever ones a downstream application (an inclinometer's tilt/heading accuracy specifically,
    say) is most sensitive to, rather than treating an exact match on every condition as equally
    valuable. See `robust_calibration.moment_condition_sensitivity` for computing a sensible one
    instead of guessing weights by hand; None (default) is uniform (every condition matters equally,
    the original formulation).

    :param A: (55, N) per-sample half-vectorized phi @ phi.T (from `build_linear_system`).
    :param b: (55,) analytic S_inf target.
    :param baseline: (N,) positive reference weights (see `local_density_baseline`); u = w/baseline - 1
        is regularized toward 0, so `baseline` — not the moment fit — carries all information about
        which samples are locally redundant.
    :param relative_regularization: lambda, weight of "stay close to baseline" vs. the moment-matching
        term (larger => smoother/safer weights, smaller => tighter/riskier moment fit), or "auto".
    :param condition_weights: (55,) nonnegative importance of each moment condition; None = uniform.
    :return: (N,) nonnegative weights. Sum(w) ~ 4*pi emerges from the "1"-monomial's own moment
        condition (see module docstring) rather than being imposed as a separate constraint.
    """
    condition_weights = np.ones(55) if condition_weights is None else condition_weights
    if relative_regularization == "auto":
        candidates = np.geomspace(1e-6, 1e-1, 12)               # tightest fit first, relax until n_eff safe
        for relative_regularization in candidates:
            weights, _ = _solve_for_lambda(A, b, baseline, relative_regularization, condition_weights)
            if weights.sum() ** 2 / (weights ** 2).sum() >= MIN_EFFECTIVE_FRACTION * weights.size:
                break
        log.debug("regularization auto-selected: %.2g", relative_regularization)

    weights, n_iterations = _solve_for_lambda(A, b, baseline, relative_regularization, condition_weights)
    effective_sample_size = weights.sum() ** 2 / (weights ** 2).sum()
    log.debug("sample weights solved: n_samples=%d effective_sample_size=%.1f weight_cv=%.3g "
               "solver_iterations=%d", weights.size, effective_sample_size,
               weights.std() / weights.mean(), n_iterations)
    return weights
