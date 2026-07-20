# Calibration Wiki: Theory and Method

Technical reference for the algorithms in `moments.py`, `calibration.py`, `orientation.py`, and
`robust_calibration.py`. Every construction used in the code is derived or cited here.

## 1. Notation

Raw **sensor sample**: a point `h` in R^3 (accelerometer or magnetometer reading). Centered sample:
`s = h - center` for some estimate of the sensor's bias. **Direction**: a unit vector `u = v/|v|` for some
`v` in R^3. `S^2` denotes the unit sphere in R^3, `dOmega` its surface measure (`integral_{S^2} dOmega
= 4*pi`). Matrices are capitalized (`D`, `S`, `Q`) to mirror standard usage in the source material;
vectors and scalars are lowercase.

## 2. The Li-Griffiths ellipsoid fit

A triaxial sensor with hard-iron bias `b`, and combined scale/soft-iron/misalignment matrix `A`,
measures a field of true magnitude `F` as

    h = A^-1 (F u) + b,          u in S^2 unit direction, |A^-1 (F u)| defines an ellipsoid in h.

Centering `s = h - b` and expanding the implicit quadric `s^T M s + s^T n + d = 0` gives, for the design
vector

    phi(s) = [x^2, y^2, z^2, 2yz, 2xz, 2xy, 2x, 2y, 2z, 1]^T   (s = [x, y, z]^T),

a homogeneous linear system `S v = 0` in the 10 quadric coefficients, where `S = sum_i phi(s_i)
phi(s_i)^T`. Li and Griffiths [1] solve this subject to a specific-ellipsoid constraint via a
generalized eigenvalue problem on a 6x6 block `E = C^-1 (S_11 - S_12 S_22^-1 S_21)` (their Eq. 15); the
eigenvector for the largest eigenvalue gives the shape parameters, and back-substitution (their Eq. 13)
gives the linear and constant terms. `calibrate.fit_quadric_form` implements this directly;
`calibrate._extract_quadric_from_S` isolates the S-to-parameters step so it can be called on an
arbitrary (e.g. perturbed) `S`, independent of how `S` was built from data (used by
`robust_calibration.moment_condition_sensitivity`, Section 9.3).

Given `M`, `n`, `d`, the calibration parameters follow from `Q = M`, `Q_inv = Q^-1`:

    b = center - Q_inv n
    A  = (F / sqrt(n^T Q_inv n - d)) * sqrtm(Q)

(`calibrate.calibrate`). `sqrtm` can return a complex dtype for a near-singular `Q`; since `Q` is
symmetric positive (semi-)definite by construction, the physical answer is the real part.

## 3. Non-uniform sampling bias

`S = sum_i phi(s_i) phi(s_i)^T` treats every sample as equally informative. A multi-position rotation
protocol (rotate about a fixed axis, reposition, repeat) does not produce uniform angular coverage:
samples are dense along each rotation's track and sparse between axes. This section derives why that
unevenness biases the fit, and Sections 4-5 construct the correction.

### 3.1 Mechanism

Let `phi_i = phi(bar_s_i) + delta_i`, where `bar_s_i` is the noise-free point and `delta_i` is the
noise-induced perturbation of `phi` (itself nonlinear in the raw measurement noise `epsilon_i`, since
`phi` contains quadratic terms: `E[(bar_x + epsilon)^2] = bar_x^2 + sigma^2` already shows `E[delta]` is
generally nonzero even for zero-mean `epsilon`). Expanding `phi_i phi_i^T` and taking expectations,

    E[S] - S_true = sum_i w_i [ bar_phi_i E[delta_i]^T + E[delta_i] bar_phi_i^T + E[delta_i delta_i^T] ],

a sum of terms that individually depend on where each `bar_phi_i` sits. For uniform coverage, this sum
has the sphere's full symmetry and its leading-order effect on the fitted *shape* (as opposed to
overall scale) is small; for uneven coverage, the sum lacks that symmetry and biases the eigenvector
direction, not just its eigenvalue. This is the same class of effect documented for algebraic conic
fitting generally by Kanatani [3]: direct/algebraic fits of conics and quadrics from noisy points carry
a systematic bias whose magnitude depends on the sampling geometry, correctable by adjusting the
moment matrix's structure before extracting parameters.

### 3.2 Correction strategy

Rather than modeling the bias term above directly, the code corrects its root cause: choose weights `w`
so that the *discrete* moment matrix matches a *continuous, uniform* reference,

    S_w = sum_i w_i phi(u_i) phi(u_i)^T  ->  S_target = integral_R phi(Omega) phi(Omega)^T dOmega,

where `u_i` are directions (Section 6) and `R` is the region of interest (`S^2` or a subset, Section
4.3). This is not a flattening of point *density* (the classical, coarser heuristic is `w_i ~
1/local_density`; Section 5 uses this only as a regularization baseline, not the objective) but a match
of the *specific* moments the Li-Griffiths eigenproblem is built from.

The same idea is standard in survey statistics under the name *calibration estimation* [2]: adjust
design weights minimally so a weighted sample reproduces known auxiliary totals. Here the "totals" are
closed-form (or numerically estimated, Section 4.3) sphere moments instead of census figures, and
"minimally" is operationalized as a ridge penalty toward a density-based baseline (Section 5.2).

## 4. Analytic sphere moments

### 4.1 Closed form

For non-negative integers `p, q, r`,

    integral_{S^2} x^p y^q z^r dOmega = 0                                    if p, q, or r is odd,
                                       = 4*pi * (p-1)!! (q-1)!! (r-1)!! / (p+q+r+1)!!   if all even,

with the convention `(-1)!! = 1`. The odd case follows from the sphere's invariance under negating any
single coordinate. The even case follows from the Gamma-function identity for moments of the uniform
measure on `S^{n-1}` (here `n = 3`):

    integral_{S^2} x^p y^q z^r dOmega = 2 Gamma((p+1)/2) Gamma((q+1)/2) Gamma((r+1)/2) / Gamma((p+q+r+3)/2),

using `Gamma(m + 1/2) = (2m-1)!! sqrt(pi) / 2^m` to convert Gamma ratios to double factorials; the
`2^m`-type factors cancel between numerator and denominator, leaving the double-factorial form above.
Since `phi`'s ten components are degree <=2 monomials, `phi phi^T`'s entries are degree <=4, so only
`(p-1)!!` for `p` in `{0, 2, 4}` is ever needed -- i.e. `(-1)!! = 1`, `1!! = 1`, `3!! = 3` -- and the
denominator only ever needs `(p+q+r+1)!!` for `p+q+r` in `{0, 2, 4}`, i.e. `1!! = 1`, `3!! = 3`,
`5!! = 15`. `moments.analytic_moment_matrix` evaluates this in closed form via `scipy.special.factorial2`,
vectorized over the 55 upper-triangular entries at once (Section 4.2).

`b[9,9]` (the "1" times "1" entry) equals `integral_{S^2} 1 dOmega = 4*pi`, the sphere's area. Since
`phi`'s ninth component is the constant 1, `sum_i w_i * 1 = sum_i w_i` is exactly this entry's discrete
counterpart -- so `sum(w) ~ 4*pi` is not an externally imposed normalization, it is one of the 55
moment-matching conditions, arising for free.

### 4.2 Half-vectorization

`phi phi^T` is symmetric 10x10 (100 entries, 55 independent). Representing it by its upper-triangular
entries, scaled by `sqrt(2)` off the diagonal, makes the half-vectorized inner product equal the
Frobenius inner product of the full matrices exactly (the standard symmetric-matrix "vech" isometry;
Magnus & Neudecker [4], Ch. 3): with `vech(M)_k` denoting entry `(i,j)`, `i <= j`, scaled by `sqrt(2)`
for `i != j`, `vech(M) . vech(N) = sum_{i,j} M_ij N_ij = trace(M^T N)`. `moments.outer_product` computes
this for data (`phi phi^T` per sample); `moments._TRIU_ROWS`, `_TRIU_COLS`, `_TRIU_SCALE` hold the
shared index/scale arrays used consistently by every function operating in this representation. `phi`
itself is built from the same exponent/coefficient table (`moments._PHI_EXPONENTS`, `_PHI_COEFS`) used
by both `monomial` (numeric evaluation) and `analytic_moment_matrix` (symbolic exponent arithmetic), so
the two cannot drift apart under maintenance.

### 4.3 Region-restricted targets

`analytic_moment_matrix` assumes `R = S^2`. If a calibration protocol structurally cannot reach the
whole sphere (Section 4.4 discusses when this matters), the correct target is `integral_R phi phi^T
dOmega` for the actual `R`, not the whole sphere. `moments.restricted_moment_matrix` estimates this
numerically: sample a dense reference grid (`fibonacci_sphere`, Section 7), keep only the reference
points within a coverage radius of at least one real sample, and average `phi phi^T` *uniformly* (not
density-weighted) over what survives, scaled by the surviving points' measured solid angle. It reduces
to `analytic_moment_matrix()` (up to grid resolution) once the data spans the whole sphere.

The coverage radius must be a *per-sample*, not a single global, quantity: each sample's own distance
to its second-nearest neighbor. A single global spacing estimate (e.g. the median nearest-neighbor
distance over the whole sample) fails whenever coverage mixes dense and sparse regions in either
direction -- dominated by a dense cluster it wrongly shrinks the reach credited to sparse areas
elsewhere; dominated by sparse background it wrongly bridges real gaps next to a dense cluster. The
per-sample version assigns the locally appropriate radius to each region independently.

### 4.4 When the whole-sphere target is (and is not) the right goal

A device intended to operate over the whole sphere should be calibrated to fit the whole sphere well,
independent of which subset the calibration protocol happened to sample -- the ellipsoid is a single
global object, and, in the noise-free case, any sufficiently non-degenerate subset of points on it
determines all of its parameters exactly (Section 3, noiseless limit). Automatically substituting
`restricted_moment_matrix`'s region for the target would silently accept whatever the calibration
protocol covered as the definition of "good enough", which is a different question from what region the
device actually needs. For this reason `moments.build_linear_system` defaults `target` to
`analytic_moment_matrix()` (the whole sphere) and does not infer a target from `directions`'s own
coverage; a caller who deliberately wants a restricted goal (e.g. an in-situ calibration whose only
interest is the region the instrument will actually revisit, such as a current meter recording mostly
near one heading) must construct that target explicitly, typically from a *separately chosen* set of
directions representing the intended region, not from the calibration data itself. See
`calibration.md`, "Choosing a target region", for the corresponding user-facing guidance including that
extreme case.

Targeting the whole sphere from partial coverage introduces a distinct numerical hazard, addressed
next.

### 4.5 Achievable condition weighting

Some of the 55 conditions in a whole-sphere target may be unreachable given the actual sample coverage.
For example, if every sample has `z > 0` (hemisphere-only coverage), any moment odd in `z` -- zero on
the whole sphere by symmetry (Section 4.1) -- integrates to a strictly positive value over the covered
region and can never be driven to zero by any choice of non-negative weights. Minimizing the plain
weighted least-squares residual `||A w - b||^2` over all 55 conditions equally does not fail
gracefully in this situation: because the conditions share the same weight vector, the optimizer's
attempt to reduce the irreducible residual on the unreachable conditions distorts the weights enough to
measurably harm the achievable ones too. This was confirmed directly: on synthetic hemisphere-only data
with a mild anisotropic scale/soft-iron matrix and Gaussian noise (`sigma = 0.3`, matching the scale
used throughout the test suite), whole-sphere-targeted weighting increased mean calibration error by a
factor exceeding 20 relative to no weighting at all; a sweep of noise level (`0.01` to `3.0`) and
anisotropy severity (axis ratio `1.01` to `3.0`) at the fix described below kept this ratio within
roughly `0.5` to `3.6` (i.e. mildly better to mildly worse than unweighted, never catastrophic) across
that whole range -- see `test_calibration.py` for the regression test and this document's Section 4.5.1
for what was tried and ruled out before arriving at the working fix.

`moments.achievable_condition_weights(directions, target)` computes a per-condition weight `kappa_k` in
`{~0, 1}`: compare `target` against `restricted_moment_matrix(directions)` (what the data's own
coverage can actually deliver) and flag a condition unreachable if the relative discrepancy exceeds a
threshold (default `0.5`). Passed as `condition_weights` to `solve_optimal_weights`, this lets the
optimizer pull the achievable conditions toward the *whole-sphere* target (addressing Section 4.4's
concern directly, rather than substituting a smaller target) while no longer paying a distortion cost
for the unreachable ones.

#### 4.5.1 Why a hard threshold, not a smooth downweighting

A softer, continuously-graded downweighting was tried first: `kappa_k = 1 / (1 + r_k^p)` for relative
discrepancy `r_k` and a power `p`, increasing `p` to penalize large discrepancies more steeply. This
did not work, even at `p = 20`: on the hemisphere test case, the discrepancy distribution is clearly
bimodal (roughly a quarter of conditions near `r_k ~ 0.2-0.3`, the remainder saturated at `r_k ~ 1`,
its analytic ceiling whenever the whole-sphere target is exactly zero and the achievable region's value
is not). A value at or near this ceiling stays near the ceiling under `x^p` for any `p` (`1^p = 1`
identically), so no power made the unreachable population's weight meaningfully smaller. A separate,
more principled-looking attempt -- checking whether `target` falls within the per-condition min/max
range achievable by *some* single-sample weighting (`[sum(w) * min_i A_ki, sum(w) * max_i A_ki]`) --
also failed: that range is a relaxation of the true, jointly-constrained feasible set (all 55 conditions
simultaneously, with weights additionally regularized toward the density baseline) and in practice is
almost always wide enough to contain the target regardless, so it does not discriminate the two
populations either. A threshold directly on `restricted_moment_matrix`'s own region-level comparison
does discriminate them, because it reflects the same joint, region-level structure that produces the
bimodal split in the first place.

## 5. Sample weight solver

### 5.1 The under-determined system and its failure mode

`solve_optimal_weights` solves `min_w ||A w - b||^2` (using `achievable_condition_weights` when
supplied, see Section 4.5) for `w >= 0`, with `A` shape `(55, N)` and `N` frequently in the thousands to
tens of thousands. This system is severely under-determined, and plain non-negative least squares
(`scipy.optimize.nnls`) on it does not fail by giving a poor fit -- it fails by collapsing onto a
*sparse* vertex solution using a handful of points (confirmed on 1500 near-uniform directions: `25`
nonzero weights survived), for the same structural reason an under-determined linear program's vertex
solutions are sparse. This is the opposite of the intended gentle density correction, which should use
all the data.

### 5.2 Ridge regularization toward a density baseline

The fix is a ridge penalty toward a baseline `w0` (Section 5.3):

    min_u  ||A (w0 * (1+u)) - b||^2 + lambda ||u||^2,   u >= -1   (so w = w0 (1+u) >= 0),

solved via L-BFGS-B with an explicit gradient (both `O(55 N)` per evaluation; no `N x N` matrix is ever
formed, so this scales to `N` in the tens of thousands in well under a second). The reparametrization
`u = w/w0 - 1` makes the bound simple (`u >= -1`) and makes `lambda` scale-free (expressed in units of
"multiples of the baseline" rather than absolute weight, so one default is sensible across a wide range
of `N`). A *global uniform* baseline (`w0 = 4*pi/N` for every sample) was tried and rejected: it prices
suppressing `K` near-duplicate samples at `K` times the cost of suppressing one, so a duplicated
cluster's total weight grows with `K` instead of staying fixed at roughly what a single representative
sample there would receive (confirmed for `K` from `5` to `500`).

`relative_regularization` (`lambda`) is not universal across datasets of the same size: data
concentrated onto a few discrete axes (e.g. a protocol that spins the instrument at only 3-4 azimuths
per tilt, producing large near-duplicate clusters at each shared pole) needs a substantially weaker
`lambda` than the same `N` spread continuously, or the achievable moment residual is inflated by an
order of magnitude or more for no good reason. The default `"auto"` sweeps a log-spaced range of
`lambda` from tightest to loosest and keeps the smallest value for which the effective sample size,

    n_eff = (sum w)^2 / sum(w^2)                              (Kish's design effect [5]),

stays at or above `MIN_EFFECTIVE_FRACTION` (default `0.1`) of `N`: as tight a moment fit as the
regularization budget allows, without concentrating weight onto too few points. This threshold was
tuned on the adversarial 500-uniform-plus-4500-clustered case (Section 5.4): `0.3` left the achievable
residual an order of magnitude worse than necessary (the clustered majority genuinely does carry little
independent information, so demanding `n_eff` stay above 30% of `N` over-constrains the fit); `0.1`
recovered a near-optimal residual while keeping `n_eff` at a physically sensible value (roughly the
number of genuinely distinct sampled locations).

### 5.3 Local density baseline

`local_density_baseline` computes `w0_i ~ 1 / density(u_i)` via a Gaussian kernel density estimate,
using all neighbors within a bandwidth-scaled radius (`scipy.spatial.cKDTree.sparse_distance_matrix`,
not a fixed-`k` neighbor count): a fixed-`k` version saturates once a duplicated cluster's size exceeds
`k` (every member then sees exactly `k` identical neighbors regardless of how many more actually
exist), letting a cluster's total baseline weight grow with its size instead of staying fixed -- the
same failure this section's baseline exists to avoid at the ridge-regularization level (Section 5.2),
now shown to also afflict a naive density estimate directly if under-resolved.

Bandwidth is `bandwidth_fraction` (default `0.25`) times a *high quantile* (default the 90th
percentile, `spacing_quantile`) of nearest-neighbor distance, not the median. The median fails whenever
near-duplicate or tightly-clustered samples approach or exceed half the dataset: on a
500-uniform-plus-4500-clustered mix, the median nearest-neighbor distance is dominated by the cluster's
own fine internal spacing, collapsing the bandwidth so far that the cluster and the uniform background
receive statistically indistinguishable per-sample weight (`1.001x` ratio measured, versus the roughly
`9-10x` ratio a correct density estimate should show given the cluster outnumbers the background by
that same factor). The 90th percentile is not immune to the identical failure mode once the
concentrated population's share exceeds it (e.g. above roughly 90% concentrated onto one location with
the default quantile); raise `spacing_quantile` for known heavier concentration, or supply a
directly-known noise/duplication scale by pre-processing `directions` if available. This also covers
near-duplicate points differing only by measurement noise (not exactly identical): the quantile is
computed over the whole nearest-neighbor distance distribution regardless of whether the smallest
values are exactly zero or merely noise-small, unlike a filter that only excludes exact zeros.

An *adaptive* (per-sample) bandwidth was tried first, by analogy with Section 4.3's per-sample coverage
radius, and rejected for a different reason: calibrating each sample's own density-estimation bandwidth
to its own local neighbor spacing makes the resulting density estimate approximately scale-invariant by
construction (a point's "density" measured "in units of its own typical neighbor distance" is
necessarily close to the same small constant everywhere, dense or sparse, since the measurement scale
and the thing being measured are tied together) -- it thus loses exactly the information needed to
distinguish a genuinely dense region from a genuinely sparse one, which requires a *shared* reference
scale across the dataset instead.

### 5.4 Adversarial test case

The primary validation case throughout this document is 500 directions spread uniformly (Section 7)
plus 4500 directions distributed within a small (`0.15` radian) cap around one location, with Gaussian
measurement noise (`sigma = 0.3`) and a mild synthetic scale/soft-iron matrix. Averaging fitted
parameters over 25 independent noise realizations (isolating systematic bias from realization-specific
variance) gives a bias-error reduction of roughly `2.7-3.2x` and a shape-error reduction of roughly
`3.2-3.8x` relative to unweighted fitting, consistent across the development of this codebase. A milder,
realistic multi-tilt rotation protocol (Section 10) shows negligible difference between weighted and
unweighted fitting: the correction's benefit scales with how severe the actual sampling imbalance is,
and is not a universal improvement independent of the data.

## 6. Direction estimation and circularity

Building `A` (Section 5) and the target (Section 4) both require *directions*, but only raw,
uncalibrated samples are available at the point weights are computed -- and the ellipsoid calibration
that would convert a raw sample to a direction is exactly what is being estimated. Any way of judging
"how uniformly were directions sampled" from raw data depends, in principle, on the same unknown
calibration: a raw sample only equals `field_magnitude * direction + noise` after undoing the (unknown)
scale/soft-iron matrix `A`. Concretely, if `A = diag(1, 1, 3)` (strong anisotropy) and true directions
are sampled perfectly uniformly, raw samples still bunch up near the compressed axis's poles purely
from the compression, with zero actual sampling unevenness -- so raw-sample density is not the same
thing as angular density, and a local-density measure computed directly on raw (uncalibrated) samples
would be equally subject to this dependency as a moment-matching approach naively assuming raw
normalization already gives directions.

`calibration.weighted_fit_quadric` resolves this the standard way for this class of problem
(iteratively re-weighted least squares, i.e. IRLS): run the existing *unweighted* fit first (no
weights, no assumption about directions whatsoever), then use that fit -- not a raw-normalization guess
-- to map samples to directions (`calibration._estimate_directions`) before computing weights.
Concretely, `sqrtm(Q) @ (s + Q^-1 n)` is proportional to the true direction with a fixed,
direction-independent scalar factor (the same one appearing in `calibrate`'s `A`), so normalizing each
column recovers the direction without needing `d`. Convergence of this scheme is an empirical property
of the specific fixed-point iteration, checked directly (`test_calibration.py`'s exact-recovery and
bias-reduction tests), not assumed from the general IRLS literature. `weighted_fit_quadric`'s
`n_iterations` parameter allows repeating direction-estimate-then-refit beyond the first weighted pass;
one iteration was sufficient in every case tested here.

## 7. Fibonacci sphere point generation

`moments.fibonacci_sphere(n)` places `n` points near-uniformly on `S^2` using the golden-angle
construction: for `i = 0, ..., n-1`,

    z_i = 1 - (2i + 1)/n,          (equivalently inclination_i = arccos(1 - 2(i+0.5)/n))
    theta_i = pi (1 + sqrt(5)) (i + 0.5)     (azimuth, golden-angle increment)

This is a standard low-discrepancy sequence for the sphere (equal-area in inclination via the
arccos transform, near-optimal angular spacing in azimuth via the golden angle) and is used both as the
reference grid for `restricted_moment_matrix`/`coverage_at`/`uncertainty_at`'s spatial binning and, in
the test suite, as a synthetic "well-covered" comparison case.

## 8. Orientation calibration

Ellipsoid calibration (Sections 2-6) maps a sensor's own readings onto its own calibrated unit sphere,
with no reference to any external frame. `orientation.py` establishes that reference from two
independent events.

### 8.1 Zero-tilt fold-in

Given accelerometer samples recorded while the instrument hung plumb, the sensor-frame direction of
true vertical is `zenith = mean(to_unit_vector(samples, calibration))`, renormalized. Rather than
carrying `zenith` as a separate reference compared against at every subsequent reading,
`orientation.zeroing_rotation` returns the rotation `R` with `R @ zenith = [0, 0, 1]`, folded once into
the accelerometer's `a2d` (`apply_zeroing_rotation`, `SensorCalibration(bias, R @ a2d)`); every later
reading's tilt is then simply `arccos(unit_vector[2])` with no reference vector to pass around. Because
accelerometer and magnetometer are rigidly co-mounted, `R` must be applied to *both* sensors'
calibrations identically to keep them in a consistent frame (folding it into only one and not the other
was confirmed, during development, to break heading computation despite `zeroing_rotation` itself
behaving correctly in isolation).

`R` is constructed via Rodrigues' rotation formula, in the closed form for aligning one unit vector to
another (`orientation.rotate(r_from, r_to)`): with `c = r_from x r_to` and `[c]_x` the skew-symmetric
cross-product matrix (`[c]_x v = c x v` for any `v`),

    R = I + [c]_x + [c]_x^2 * (1 - r_from . r_to) / |c|^2,

valid whenever `r_from` and `r_to` are not exactly parallel or antiparallel (`c = 0` otherwise, handled
as the identity when `r_from == r_to` exactly; the antiparallel case is not separately handled and would
divide by zero, since two exactly opposite unit vectors do not determine a unique rotation axis without
additional information). This is verified directly (`R @ r_from == r_to`, `R` orthogonal, `det(R) = 1`)
in `test_orientation.py`.

### 8.2 Heading offset: why not folded in

Heading additionally requires the magnetometer's reading projected onto the *current* horizontal
plane, which changes as the instrument tilts -- unlike zero tilt, there is no single fixed rotation that
absorbs this into a static calibration matrix, since the relevant horizontal plane is different at every
reading. `calibrate_heading_reference` instead returns a scalar offset, applied at read time
(`heading_and_tilt`) rather than folded into `a2d`.

Bearing is computed by `orientation._bearing(field, up, reference_axis)`: project both `field` and
`reference_axis` perpendicular to `up` (removing the vertical component), then take the signed angle
between the results via `atan2` of the cross and dot products, positive counterclockwise around `up`
(right-hand rule). Both `calibrate_heading_reference` (computing the offset from a known-north event)
and `heading_and_tilt` (applying it to arbitrary readings) call the identical function, so the sign
convention only needs to be self-consistent between the two, never independently verified against an
external standard (e.g. NED vs ENU) -- integrating with a system using a different axis convention
requires an explicit check against one known reference orientation.

`calibrate_heading_reference`'s offset is the *circular* mean (`atan2(mean(sin), mean(cos))`) of the
per-sample raw bearing, not the arithmetic mean: angles that straddle the 0/360 degree boundary would
otherwise average toward the wrong side (verified directly with samples at both -2 and +2 degrees,
which must average to 0, not to +/-180).

This module's "heading" is the bearing of a *fixed* sensor axis relative to north, for relating the
sensor's own frame to a compass direction -- a different quantity from a tilt-current-meter's flow
direction (the azimuth of the *tilt itself* relative to vertical, which is undefined at exactly zero
tilt: with zero horizontal tilt component, there is no lean and hence no flow direction to report, which
is correct behavior for that quantity, not a degenerate case of this one). Both use the same
accelerometer-plus-magnetometer fusion underneath; which is wanted depends on whether the instrument
has a meaningful fixed heading of its own or reports drag/flow direction instead.

## 9. Robust and self-assessing calibration

### 9.1 Outlier rejection

`robust_calibration.radial_residuals` gives `(calibrated radius)/field_magnitude - 1` per sample --
approximately zero for a sample that truly lies on the calibrated ellipsoid, regardless of direction,
and directly tied to what the calibration is trying to achieve (unlike a per-axis sigma-clip on raw
X/Y/Z, which has no notion of the ellipsoid model and can accept or reject points for reasons unrelated
to it). `reject_outliers` thresholds this via a median-absolute-deviation (MAD) scale rather than
standard deviation: a handful of genuine outliers inflate a std-based threshold enough to hide
themselves, while the median-based scale does not have this failure mode. `autocalibrate` resolves the
same bootstrap tension as Section 6 (judging outliers needs a calibration; outliers can distort that
calibration) the same way: fit on everything, reject by the fit's own residuals, refit on what remains,
repeat until nothing new is rejected or an iteration limit is reached. On synthetic data with 3% gross
outliers injected, this recovered 58/60 true outliers with 0 false positives among 1940 clean samples,
reducing calibration error by roughly 33x relative to a naive fit on the contaminated data.

### 9.2 Uncertainty quantification

Sample density alone (`coverage_at`) cannot distinguish *why* an error might be large at some direction
-- thin coverage, ordinary sensor noise, or a location where the ellipsoid model itself does not fit
well are three different explanations with different remedies. `uncertainty_at` reports three separate
quantities instead of one:

- `noise_floor`: the global MAD-based scale of `radial_residuals`, an estimate of irreducible per-sample
  noise, independent of location.
- `systematic_z_score`: at each query direction, the local mean residual among nearby samples divided by
  what pure noise predicts for that many samples (`noise_floor / sqrt(n_local)`). Confirmed to separate
  a genuine local model-misfit from ordinary noise clearly: injecting a coherent (non-random) local
  distortion near one pole gave a mean `|z|` of approximately 27 there versus approximately 3 elsewhere.
- `jackknife_spread_rad`: leave-one-region-out refit spread, evaluated on a synthetic reading constructed
  to correspond to each query direction under the full-data fit. This showed only a weak signal in
  testing (a deliberately sparse-but-nonempty region measured roughly 15% higher spread than a dense
  region, not the large contrast expected): a global ellipsoid fit using several hundred to a few
  thousand points is not very sensitive to removing any single region out of a modest number of regions.
  It structurally cannot detect a *completely empty* region either, since removing data that was never
  there changes nothing -- `coverage_at`'s density is the correct tool for "is there any data here at
  all"; `jackknife_spread_rad` is retained as a secondary, exploratory diagnostic pending investigation
  of a leave-out structure (e.g. by time window rather than by direction, or fewer/larger regions) that
  might show a stronger effect.

### 9.3 Sensitivity-weighted moment conditions

Section 4's 55 moment conditions need not matter equally for a specific downstream application. Framing
this precisely: an inclinometer application's actual objective might be stated as

    J_application = sum_i w_i [alpha * d_S2(g_hat_i, g_i)^2 + beta * (psi_hat_i - psi_i)^2] + regularization(...),

for geodesic distance `d_S2` on the sphere, true/estimated gravity direction `g_i`/`g_hat_i`, and
true/estimated azimuth `psi_i`/`psi_hat_i` -- direction error and heading error weighted by their
respective application importance, rather than an abstract equal-weighting of 55 algebraic moment
conditions. Minimizing this directly during calibration is not generally possible: `g_i` and `psi_i` are
exactly what is unknown (that is the calibration problem itself), so a fully faithful version requires
joint latent-variable estimation over all orientations, world vectors, and sensor parameters at once
(e.g. Bayesian MAP or a factor graph over `SO(3) x S^2`) rather than the independent per-sensor
ellipsoid fit used throughout this codebase.

`robust_calibration.moment_condition_sensitivity` implements a minimal alternative covering only the
tilt/direction term (`alpha`): perturb each of the 55 conditions in the achieved `S` in turn, and
measure how much the resulting calibrated direction moves at a reference set of test points. No ground
truth is needed for this because it is a local-sensitivity (Jacobian) question, not an absolute-accuracy
one -- `d_S2(g_hat, g)`'s first-order dependence on a perturbation of `S` is exactly this Jacobian,
regardless of the (unknown) true `g`. The heading/azimuth term (`beta`) would need a joint sensitivity
across both sensors simultaneously (heading depends on both via `orientation._bearing`) and is not
implemented.

Empirically, feeding this into `solve_optimal_weights`'s `condition_weights` did not measurably improve
direction accuracy over uniform condition weights on the adversarial test case of Section 5.4 (three
normalizations tried -- raw, square-root-compressed, log-compressed -- all within the fourth decimal
place of each other and of the uniform-weighting baseline, with no consistent direction of improvement
across trials). A plausible explanation: `solve_optimal_weights` already targets a *reachable* moment
matrix that uniform condition weighting fits closely in this case, leaving little of the
condition-versus-condition trade-off this mechanism is meant to resolve; the mechanism should matter
more when the 55 conditions are in genuine mutual tension (coverage too sparse or too structured to hit
all of them well simultaneously), which was not true of the cases tested here. It is retained as
correctly-implemented infrastructure, not as a verified improvement.

## 10. Test geometry generators

`conftest.py` generates several synthetic angular-coverage patterns used throughout the test suite:

- `fibonacci_sphere`: re-exported from `moments` (Section 7); the near-uniform reference case.
- `dense_sphere_grid`: a midpoint-rule latitude/longitude grid with `dOmega = sin(theta) dtheta dphi`
  weights, for numerically verifying `analytic_moment_matrix` against direct integration independent of
  the closed-form derivation (Section 4.1).
- `clustered_sphere`: `n_uniform` points spread over the whole sphere plus `n_cluster` points scattered
  within a small angular radius of one direction -- the adversarial case of Section 5.4, simulating a
  rotation protocol that spends most of its time in one orientation.
- `belt_geometry`: continuous-azimuth rotation at each of several fixed tilts -- latitude "belts", not a
  uniform sphere, matching a rotate-about-vertical-at-several-tilts protocol.
- `multi_axis_spin_geometry`: for each of several (tilt, azimuth) settings, the device's own spin axis is
  held fixed while it completes a full spin, tracing a small circle of angular radius equal to the tilt
  around that axis (via Rodrigues' rotation-*by-angle* formula -- distinct from `orientation.rotate`'s
  single-pair alignment form -- vectorized over the full spin angle range at once). This differs from
  `belt_geometry` in producing many small circles scattered at discrete locations rather than full
  latitude rings, matching a protocol that holds the rotation axis at a small number of discrete
  settings instead of sweeping azimuth continuously.

## References

1. Qingde Li, J.G. Griffiths, "Least squares ellipsoid specific fitting", Geometric Modeling and
   Processing 2004, pp. 335-340. Source used for the original implementation:
   https://teslabs.com/articles/magnetometer-calibration/
2. J.C. Deville, C.E. Sarndal, "Calibration Estimators in Survey Sampling", Journal of the American
   Statistical Association 87(418), 1992.
3. K. Kanatani, "Statistical Bias of Conic Fitting and Renormalization", IEEE Transactions on Pattern
   Analysis and Machine Intelligence 16(3), 1994.
4. J.R. Magnus, H. Neudecker, *Matrix Differential Calculus with Applications in Statistics and
   Econometrics*, 3rd ed., Wiley, 2007.
5. L. Kish, *Survey Sampling*, Wiley, 1965 (design effect / effective sample size).
6. Closed-form sphere moments: https://en.wikipedia.org/wiki/Solid_harmonics#Moments
