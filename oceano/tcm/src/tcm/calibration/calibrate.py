"""
Li-Griffiths [1] specific ellipsoid fitting, plus the sample-weighting correction from `moments.py` for
calibration rotations that sample the response sphere unevenly.

Getting directions to feed `moments.build_linear_system` is the one place model-dependency is real: a
raw sample s only equals `field_magnitude * direction + noise` after undoing the (unknown) calibration
matrix A, so *any* way of judging "how uniformly were directions sampled" — ours or a plain local-
density measure computed straight on raw samples — depends on that same unknown A. Concretely: if
A = diag(1, 1, 3) (strong anisotropy) and directions are sampled perfectly uniformly, raw samples still
bunch up near the compressed axis's poles purely from the compression, with zero experimental
unevenness — raw-space density is not angular density. There is no way around needing *some* estimate
of A; the question is only how to use one without circularity. `_estimate_directions` resolves it the
standard (IRLS-style) way: run the existing unweighted fit first (no weights, no direction assumption
whatsoever), then use *that* fit — not a raw-normalization guess — to map samples to directions before
computing weights. Convergence of this kind of scheme is an empirical question, checked directly in
`test_calibration.py` rather than assumed.

.. [1] Qingde Li, J.G. Griffiths, "Least squares ellipsoid specific fitting", GMP 2004, pp.335-340.
Source: https://teslabs.com/articles/magnetometer-calibration/
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing import Optional

try:
    from scipy.linalg import sqrtm as _sqrtm  # general Schur-based; used when scipy available
except ImportError:
    def _sqrtm(A: np.ndarray) -> np.ndarray:
        """Matrix square root for symmetric positive-definite *A* via eigendecomposition.

        ``scipy.linalg.sqrtm`` numpy-only fallback for the noh5 distribution.
        Exact for SPD matrices (ellipsoid shape matrix Q).  Symmetrizes *A* first
        to absorb floating-point drift.
        """
        A_sym = (A + A.T) / 2
        w, v = np.linalg.eigh(A_sym)
        return v @ np.diag(np.sqrt(w)) @ v.T

log = logging.getLogger(__name__)



# Semantic aliases mapping domain concepts to strict topologies
Bias = NDArray[np.float64]  # ℝ³ˣ¹
A2D = NDArray[np.float64]  # ℝ³ˣ³
Raw = NDArray[np.float64]  # ℝ³ˣᴺ
Unit = NDArray[np.float64]  # ℝ³ˣᴺ


class SensorCalibration(tuple):
    """Ellipsoid calibration for one sensor.
    Usage example: :func:`calibrate` unpacks natively: ``bias, a2d = calibrate(...)``
    """
    __slots__ = ()
    _fields = ("bias", "a2d")

    def __new__(cls, bias: Bias, a2d: A2D) -> "SensorCalibration":
        """Coerce to ℝ³ˣ¹ and ℝ³ˣ³ topologies."""
        return super().__new__(
            cls,
            (
                np.asarray(bias, dtype=np.float64).reshape(3, 1),
                np.asarray(a2d, dtype=np.float64).reshape(3, 3),
            ),
        )

    @property
    def bias(self) -> Bias:
        return self[0]

    @property
    def a2d(self) -> A2D:
        return self[1]





def to_unit_vector(raw: Raw, calibration: SensorCalibration) -> Unit:
    """Project raw samples onto the calibrated unit sphere.
    :param raw: (3, N) raw sensor samples.
    :param calibration: this sensor's ellipsoid fit (see `calibrate`).
    :return: (3, N) unit vectors.

    Usage: prefer as single place where ``a2d @ (raw - bias)`` is spelled out, so every caller applies it
    identically. Example: :func:`calibrate`
    """
    unit = calibration.a2d @ (raw - calibration.bias)
    return unit / np.linalg.norm(unit, axis=0)


# Li-Griffiths Eq. 15: inv(C) constraint matrix (k=4 quadric)
_C_INV = np.array([                  # inv(C), Eq. 8 (k=4); C itself: [[-1,1,1,0,0,0],[1,-1,1,0,0,0],
    [0, .5, .5, 0, 0, 0],             #                                 [1,1,-1,0,0,0],[0,0,0,-4,0,0],
    [.5, 0, .5, 0, 0, 0],             #                                 [0,0,0,0,-4,0],[0,0,0,0,0,-4]]
    [.5, .5, 0, 0, 0, 0],
    [0, 0, 0, -.25, 0, 0],
    [0, 0, 0, 0, -.25, 0],
    [0, 0, 0, 0, 0, -.25],
])
_M_INDEX = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])  # v_1 → symmetric M, Eq. 16


# --------------------------------------------------------------------------- #
# Core math (numpy-only — testable independently)
# --------------------------------------------------------------------------- #

def _extract_quadric_from_S(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Core of `fit_quadric_form`, taking the (10, 10) moment matrix directly rather than building it
    from data — the entry point `robust_calibration.moment_condition_sensitivity` perturbs through,
    since it needs to ask "what if S had come out slightly different here" without needing raw data."""
    S_11, S_12 = S[:6, :6], S[:6, 6:]
    S_21, S_22 = S[6:, :6], S[6:, 6:]

    try:
        E = _C_INV @ (S_11 - S_12 @ np.linalg.inv(S_22) @ S_21)          # Eq. 15
    except np.linalg.LinAlgError:
        log.exception("S_22 is singular: samples don't constrain all quadric parameters "
                       "(rotations likely didn't cover enough distinct axes)")
        raise

    E_w, E_v = np.linalg.eig(E)
    v_1 = E_v[:, np.argmax(E_w)]
    v_1 *= -1 if v_1[0] < 0 else 1                                    # canonical sign (Eq. 15's v_1[0] > 0)
    v_2 = -np.linalg.inv(S_22) @ S_21 @ v_1                               # Eq. 13

    M = v_1[_M_INDEX]
    n = v_2[:-1, np.newaxis]
    d = v_2[3]
    return M, n, d


def fit_quadric_form(s: np.ndarray,
                      weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate quadric form parameters from a set of points.

    :param s: (3, N) samples (x, y, z rows).
    :param weights: (N,) optional nonnegative sample weights (see `moments.solve_optimal_weights`)
        correcting for uneven angular sampling; None (default) reproduces the original S = D @ D.T.
    :return: M, n, d — quadric form parameters in h.T @ M @ h + h.T @ n + d = 0.
    References
    .. [1] Qingde Li, J.G. Griffiths, "Least squares ellipsoid specific fitting", GMP 2004, pp.335-340.
    Source: https://teslabs.com/articles/magnetometer-calibration/
    """
    from tcm.calibration import moments  # lazy: moments→scipy, not needed by SensorCalibration/to_unit_vector
    D = moments.monomial(s)                                   # (10, N); shared phi definition, moments.py
    S = (D * weights if weights is not None else D) @ D.T            # Eq. 11; weighted => S = D @ W @ D.T
    return _extract_quadric_from_S(S)


def _estimate_directions(s: np.ndarray, Q: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Map centered samples to estimated true (unit) directions through a quadric fit — undoing the
    ellipsoid's own scale/soft-iron distortion via an already-fitted Q, n, rather than assuming raw
    samples are already close to spherical (see module docstring for why that assumption is avoided).

    sqrtm(Q) @ (s + inv(Q) @ n) is proportional to the true direction with a fixed (direction-
    independent) scalar factor — see `calibrate`'s a2d — so normalizing each column recovers it
    without needing `d` at all.

    :param s: (3, N) centered samples (`fit_quadric_form`'s convention).
    :param Q, n: a preceding `fit_quadric_form(s, ...)` fit (weighted or not).
    :return: (3, N) unit vectors.
    """
    unit_ish = np.real(_sqrtm(Q) @ (s + np.linalg.inv(Q) @ n))
    return unit_ish / np.linalg.norm(unit_ish, axis=0)


def weighted_fit_quadric(s: np.ndarray, n_iterations: int = 1,
                          target_directions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    `fit_quadric_form`, weighted to correct for the rotations' uneven angular coverage (see module
    docstring). Iteration 0 is the plain unweighted fit; each subsequent iteration re-estimates
    directions from the previous fit, recomputes weights, and refits — see `test_calibration.py` for
    whether more than one iteration is worth its cost for this data.

    :param s: (3, N) centered samples (same convention as `fit_quadric_form`).
    :param n_iterations: number of weighted refinement passes after the initial unweighted fit.
    :param target_directions: (3, M) directions describing the region the sensor is meant to be
        calibrated *for*; None (default) targets the whole sphere (see
        `moments.build_linear_system`). Pass the calibration data's own directions only when that
        data's coverage genuinely *is* the intended operating envelope (see `calibration.md`).
    :return: same as `fit_quadric_form`.
    """
    from tcm.calibration import moments  # lazy: moments→scipy, not needed by SensorCalibration/to_unit_vector
    target = None if target_directions is None else moments.restricted_moment_matrix(target_directions)
    Q, n, d = fit_quadric_form(s)
    for _ in range(n_iterations):
        directions = _estimate_directions(s, Q, n)
        baseline = moments.local_density_baseline(directions)
        A, b = moments.build_linear_system(directions, target)
        condition_weights = moments.achievable_condition_weights(directions, b)
        weights = moments.solve_optimal_weights(A, b, baseline, condition_weights=condition_weights)
        Q, n, d = fit_quadric_form(s, weights)
    return Q, n, d


def calibrate(
    raw: np.ndarray,
    field_magnitude: float = 1.0,
    weighted: bool = True,
) -> SensorCalibration:
    """Bias + gain matrix for a triaxial sensor from multi-position samples.

    Parameters
    ----------
    raw : ``(3, N)`` raw sensor samples.
    field_magnitude : known reference magnitude (e.g. local IGRF total field
        for a magnetometer, or standard gravity for an accelerometer).
    weighted : use :func:`weighted_fit_quadric` (default) or plain unweighted fit.

    Returns
    -------
    SensorCalibration(bias, a2d) — unpacks as ``(bias, a2d)``:
        bias ``(3, 1)`` raw-space center,
        a2d ``(3, 3)`` such that ``a2d @ (raw - bias)`` maps samples onto
        the calibrated sphere of radius *field_magnitude*.
    """
    center = raw.mean(1, keepdims=True)
    fit = weighted_fit_quadric if weighted else fit_quadric_form
    Q, n, d = fit(raw - center)

    Q_inv = np.linalg.inv(Q)

    bias = center - Q_inv @ n  # combined hard-iron + fit offset = ellipsoid center in raw space

    # some sqrtm() implementations return complex dtype for a near-singular Q; the physical answer is
    # the real part, since Q is symmetric positive (semi-)definite by construction
    a2d = np.real(field_magnitude / np.sqrt(n.T @ Q_inv @ n - d) * _sqrtm(Q))
    log.debug("calibrate: n=%d weighted=%s |bias|=%.4g scale_range=[%.4g, %.4g]",
              raw.shape[1], weighted, np.linalg.norm(bias), *np.sort(np.linalg.eigvalsh(a2d))[[0, -1]])
    return SensorCalibration(bias, a2d)


# --------------------------------------------------------------------------- #
# Wrappers with outlier rejection
# --------------------------------------------------------------------------- #
def reject_outliers_mad(
    data_3d: np.ndarray,
    gain: np.ndarray,
    bias: np.ndarray,
    *,
    outlier_sigma: float = 2.0,
) -> np.ndarray:
    """MAD-based outlier rejection on sphere-distance residuals.

    Parameters
    ----------
    data_3d : ``(3, N)`` raw sensor data.
    gain, bias : from :func:`calibrate`.
    outlier_sigma : MAD multiplier threshold.

    Returns
    -------
    ``(N,)`` boolean mask — ``True`` for inliers.
    """
    calibrated = gain @ (data_3d - bias)
    dist = np.abs(1.0 - np.linalg.norm(calibrated, axis=0))
    med = np.median(dist)
    mad = np.median(np.abs(dist - med))
    threshold = med + outlier_sigma * mad * 1.4826
    return dist < max(threshold, 1e-10)


def calibrate_channel(
    data_3d: np.ndarray,
    *,
    max_iter: int = 5,
    outlier_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ellipsoid with iterative MAD-based outlier rejection.

    The returned ``(gain, bias)`` satisfy::

        gain @ (raw - bias) ≈ unit sphere

    This matches the downstream ``fG(Axyz, Ag, Cg) = Ag @ (Axyz - Cg)``
    convention where ``Cg`` = the raw-space center stored in HDF5 as
    ``//coef//G//C``.

    Parameters
    ----------
    data_3d : ``(3, N)`` raw sensor channels.
    max_iter : iterations for outlier rejection.
    outlier_sigma : MAD multiplier for sphere-distance rejection.

    Returns
    -------
    gain : ``(3, 3)``
    bias : ``(3, 1)``
    """
    pts = data_3d.copy()
    for _ in range(max_iter):
        bias, gain = calibrate(pts, field_magnitude=1.0, weighted=False)
        inlier = reject_outliers_mad(pts, gain, bias, outlier_sigma=outlier_sigma)
        if inlier.all():
            break
        pts = pts[:, inlier]

    # Final fit on inliers
    bias, gain = calibrate(pts, field_magnitude=1.0, weighted=False)
    return gain, bias


def coef2str(gain: np.ndarray, bias: Optional[np.ndarray] = None) -> tuple[str, str]:
    """Human-readable string representation of calibration coefficients.

    Parameters
    ----------
    gain : ``(3, 3)`` calibration gain matrix.
    bias : ``(3, 1)`` calibration bias vector.

    Returns
    -------
    a_str, b_str — formatted strings for gain (× 10⁻⁴) and bias.
    """
    A1e4 = np.round(np.float64(gain) * 1e4, 1)
    a_str = "float64([{}])*1e-4".format(
        ",\n".join([
            "[{}]".format(",".join(str(A1e4[i, j]) for j in range(gain.shape[1])))
            for i in range(gain.shape[0])
        ])
    )
    b_str = (
        'float64([[{}]])'.format(','.join(str(bi) for bi in bias.flat))
        if bias is not None else "None"
    )
    return a_str, b_str
