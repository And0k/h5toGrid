"""
Integration tests for `calibration.py` shows whether the proposed theory gives a practical benefit
- per-trial total error mixes bias with noise-driven variance,
- averaging fitted parameters over many noise draws isolates the systematic bias weighting is meant to remove (the well-documented bias of algebraic conic/quadric fitting under noise — Kanatani 1994, see
`moments.py`'s module docstring) from ordinary trial-to-trial scatter.
"""
import numpy as np
import pytest

from tcm.calibration import calibrate as cal
from tcm.calibration.moments import fibonacci_sphere
from _calibration.conftest import belt_geometry, clustered_sphere

A_TRUE = np.array([[1.05, 0.03, -0.02], [0.03, 0.97, 0.01], [-0.02, 0.01, 1.02]])   # mild soft-iron+misalign
BIAS_TRUE = np.array([[15.], [-8.], [22.]])
FIELD_MAGNITUDE = 50.


def _raw_samples(directions: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Inverts `calibrate`'s model: raw = A^-1 (field_magnitude * direction) + bias + noise."""
    clean = np.linalg.solve(A_TRUE, FIELD_MAGNITUDE * directions) + BIAS_TRUE
    return clean + rng.normal(scale=noise_std, size=clean.shape)


def test_exact_recovery_without_noise_regardless_of_weighting():
    """Sanity check, not TDD item 4: with zero noise every sample satisfies the quadric identity
    exactly, so weights can't change the (exact) answer — see `moments.py`'s module docstring on why
    weighting only matters once noise is present."""
    raw = _raw_samples(fibonacci_sphere(400), noise_std=0., rng=np.random.default_rng(0))
    for weighted in (False, True):
        bias, a2d = cal.calibrate(raw, FIELD_MAGNITUDE, weighted=weighted)
        assert bias == pytest.approx(BIAS_TRUE, abs=1e-9)
        assert a2d == pytest.approx(A_TRUE, abs=1e-9)


def test_weighting_reduces_systematic_bias_under_severe_clustering():
    """The plan doc's own suggested adversarial case: 500 uniform + 4500 crowded near one direction."""
    directions = clustered_sphere(500, 4500, [0, 0, 1], 0.15, np.random.default_rng(123))
    fits = {weighted: [cal.calibrate(_raw_samples(directions, 0.3, np.random.default_rng(seed)),
                                      FIELD_MAGNITUDE, weighted=weighted)
                       for seed in range(25)]
            for weighted in (False, True)}

    def mean_error(field, truth):
        return {w: np.linalg.norm(np.mean([f[field] for f in fits[w]], 0) - truth) for w in (False, True)}
    bias_error, shape_error = mean_error(0, BIAS_TRUE), mean_error(1, A_TRUE)
    assert bias_error[True] < 0.7 * bias_error[False]
    assert shape_error[True] < 0.7 * shape_error[False]


def test_weighting_does_not_hurt_a_realistic_rotation_protocol():
    """The actual protocol (belt_geometry) is far milder than the adversarial case above — there is
    little systematic bias to correct in the first place here, so this only checks weighting doesn't
    make a realistic, already-reasonable protocol appreciably worse."""
    directions = belt_geometry(150, np.array([0, 20, 45, 70, 90, 110, 135, 160, 180]))
    fits = {weighted: [cal.calibrate(_raw_samples(directions, 0.3, np.random.default_rng(seed)),
                                      FIELD_MAGNITUDE, weighted=weighted)
                       for seed in range(20)]
            for weighted in (False, True)}

    def total_error(fit):
        return np.linalg.norm(fit[0] - BIAS_TRUE) + np.linalg.norm(fit[1] - A_TRUE)

    mean_error = {w: np.mean([total_error(f) for f in fits[w]]) for w in (False, True)}
    assert mean_error[True] < 1.25 * mean_error[False]


def test_weighting_does_not_catastrophically_hurt_hemisphere_only_coverage():
    """Regression test for a real finding: targeting the *whole* sphere's moments on directions that
    only reach a hemisphere used to make weighting actively degrade calibration by >20x versus not
    weighting at all (the optimizer chases an unreachable constraint — odd-in-z moments that can
    never be zero when every sample has the same-sign z). `moments.restricted_moment_matrix` fixed
    this by targeting the region actually reached, not the whole sphere; this pins the fix in place."""
    directions = belt_geometry(200, np.linspace(0, 90, 10))                # genuinely one hemisphere only
    fits = {weighted: [cal.calibrate(_raw_samples(directions, 0.3, np.random.default_rng(seed)),
                                      FIELD_MAGNITUDE, weighted=weighted)
                       for seed in range(15)]
            for weighted in (False, True)}

    def total_error(fit):
        return np.linalg.norm(fit[0] - BIAS_TRUE) + np.linalg.norm(fit[1] - A_TRUE)

    mean_error = {w: np.mean([total_error(f) for f in fits[w]]) for w in (False, True)}
    assert mean_error[True] < 2. * mean_error[False]                            # was >20x before the fix
