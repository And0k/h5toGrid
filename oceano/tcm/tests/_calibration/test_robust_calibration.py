"""Tests for `tcm.calibration.robust` — each against a synthetic scenario with a known injected fault.

Ground truth calibration: A_TRUE @ (raw - BIAS_TRUE) ≈ FIELD_MAGNITUDE * direction.
"""
import numpy as np

from tcm.calibration import calibrate as cal
from tcm.calibration import moments
from tcm.calibration import robust as rc
from tcm.calibration.moments import fibonacci_sphere

# --- Shared ground truth (same as test_fitting.py) ---
A_TRUE = np.array([[1.05, 0.03, -0.02], [0.03, 0.97, 0.01], [-0.02, 0.01, 1.02]])
BIAS_TRUE = np.array([[15.], [-8.], [22.]])
FIELD_MAGNITUDE = 50.  # field_magnitude, nT-like units


def _raw_samples(directions, noise_std, rng):
    """Invert the calibration model: raw = A⁻¹(FM·dir) + bias + noise."""
    clean = np.linalg.solve(A_TRUE, FIELD_MAGNITUDE * directions) + BIAS_TRUE
    return clean + rng.normal(scale=noise_std, size=clean.shape)


class TestUncertaintyAt:
    """`systematic_z_score` is tested for real separation (validated); `jackknife_spread_rad` only for
    basic mechanical sanity (shape, finiteness) -- see `uncertainty_at`'s own docstring for why."""

    def test_systematic_z_score_flags_a_coherent_local_distortion(self):
        rng = np.random.default_rng(0)
        directions = fibonacci_sphere(2500)
        raw = _raw_samples(directions, 0.3, rng)
        distorted = directions[2] > np.cos(np.radians(20))
        raw[:, distorted] += directions[:, distorted] * 2.5             # coherent stretch, not noise

        query, unc = rc.uncertainty_at(raw, FIELD_MAGNITUDE, n_regions=15, weighted=False)
        near_pole = query[2] > np.cos(np.radians(15))
        assert np.nanmean(np.abs(unc["systematic_z_score"][near_pole])) > 5 * \
            np.nanmean(np.abs(unc["systematic_z_score"][~near_pole]))

    def test_jackknife_spread_shape_and_finiteness(self):
        rng = np.random.default_rng(1)
        raw = _raw_samples(fibonacci_sphere(800), 0.3, rng)
        query, unc = rc.uncertainty_at(raw, FIELD_MAGNITUDE, n_regions=10, weighted=False)
        assert unc["jackknife_spread_rad"].shape == (query.shape[1],)
        assert np.isfinite(unc["jackknife_spread_rad"]).all()
        assert unc["n_regions_used"] >= 8

class TestOutlierRejection:
    N_SAMPLES = 2000
    N_OUTLIERS = 60
    OUTLIER_SCALE = 25.  # gross — far beyond normal noise

    def test_catches_injected_gross_outliers_with_few_false_positives(self):
        rng = np.random.default_rng(0)
        raw = _raw_samples(fibonacci_sphere(self.N_SAMPLES), 0.3, rng)
        outlier_idx = rng.choice(raw.shape[1], self.N_OUTLIERS, replace=False)
        raw[:, outlier_idx] += rng.normal(scale=self.OUTLIER_SCALE, size=(3, self.N_OUTLIERS))

        calibration, _ = rc.autocalibrate(raw, FIELD_MAGNITUDE)
        keep = rc.reject_outliers(raw, calibration, FIELD_MAGNITUDE)
        detected = set(np.flatnonzero(~keep))
        assert len(detected & set(outlier_idx)) >= 55, "catch most of the 60"
        assert len(detected - set(outlier_idx)) <= 5, "few false positives"

    def test_autocalibrate_recovers_close_to_ground_truth_despite_contamination(self):
        rng = np.random.default_rng(0)
        raw = _raw_samples(fibonacci_sphere(self.N_SAMPLES), 0.3, rng)
        outlier_idx = rng.choice(raw.shape[1], self.N_OUTLIERS, replace=False)
        raw[:, outlier_idx] += rng.normal(scale=self.OUTLIER_SCALE, size=(3, self.N_OUTLIERS))

        naive = cal.calibrate(raw, FIELD_MAGNITUDE)
        robust_cal, history = rc.autocalibrate(raw, FIELD_MAGNITUDE)

        def _error(fit):
            return np.linalg.norm(fit.bias - BIAS_TRUE) + np.linalg.norm(fit.a2d - A_TRUE)

        assert _error(robust_cal) < 0.1 * _error(naive), "≥10× better"
        assert history[-1]["n_rejected_this_round"] == 0, "converged"

    def test_clean_data_rejects_almost_nothing(self):
        rng = np.random.default_rng(1)
        raw = _raw_samples(fibonacci_sphere(1000), 0.3, rng)
        calibration, _ = rc.autocalibrate(raw, FIELD_MAGNITUDE)
        keep = rc.reject_outliers(raw, calibration, FIELD_MAGNITUDE)
        assert keep.mean() > 0.97


# --------------------------------------------------------------------------- #
# Spatial coverage
# --------------------------------------------------------------------------- #
class TestExpectedDirectionError:
    def test_flags_a_deliberately_uncovered_region(self):
        rng = np.random.default_rng(2)
        directions = fibonacci_sphere(3000)
        directions = directions[:, directions[2] > -np.cos(np.radians(30))]     # nothing near the south pole
        raw = _raw_samples(directions, 0.3, rng)

        query, result = rc.expected_direction_error(raw, FIELD_MAGNITUDE, n_regions=10, weighted=False)
        near_gap = query[2] < -np.cos(np.radians(20))
        assert result["total_deg"][near_gap].mean() > result["total_deg"][~near_gap].mean()


class TestCoverageAt:
    def test_flags_a_deliberately_uncovered_region(self):
        rng = np.random.default_rng(2)
        # Remove everything near the south pole (z < -cos(30°))
        directions = fibonacci_sphere(3000)
        directions = directions[:, directions[2] > -np.cos(np.radians(30))]  # nothing near the south pole
        calibration = cal.calibrate(_raw_samples(directions, 0.3, rng), FIELD_MAGNITUDE)

        query, density = rc.coverage_at(_raw_samples(directions, 0.3, rng), calibration)
        near_gap = query[2] < -np.cos(np.radians(20))
        assert density[near_gap].max() == 0., "zero coverage in gap"
        assert density[~near_gap].mean() > 0., "nonzero elsewhere"


# --------------------------------------------------------------------------- #
# Temporal anomaly detection
# --------------------------------------------------------------------------- #

class TestAnomalousTimeWindows:
    def test_flags_only_the_injected_transient_window(self):
        """Two passes over identical directions; only one 40-sample window in pass 2 is corrupted —
        must be flagged despite pass 1 showing the same direction is normally fine."""
        rng = np.random.default_rng(3)
        base_directions = fibonacci_sphere(1500)
        directions = np.hstack([base_directions, base_directions])
        raw = _raw_samples(directions, 0.3, rng)
        corrupt = slice(1500 + 500, 1500 + 540)               # 40 samples corrupted
        raw[:, corrupt] += np.array([[4.], [3.], [-2.]])

        calibration = cal.calibrate(raw, FIELD_MAGNITUDE)
        flagged = rc.anomalous_time_windows(raw, calibration, FIELD_MAGNITUDE, window_size=40, n_direction_bins=150)
        flagged_indices = {f["window_index"] for f in flagged}
        assert (1500 + 500) // 40 in flagged_indices           # corrupted window found
        assert len(flagged_indices) <= 3                       # no broad false-positive sweep

    def test_no_false_positives_on_clean_data(self):
        rng = np.random.default_rng(4)
        raw = _raw_samples(fibonacci_sphere(2000), 0.3, rng)
        calibration = cal.calibrate(raw, FIELD_MAGNITUDE)
        flagged = rc.anomalous_time_windows(raw, calibration, FIELD_MAGNITUDE, window_size=40)
        assert len(flagged) <= 2, "allow rare chance false positive"


# --------------------------------------------------------------------------- #
# Moment condition sensitivity
# --------------------------------------------------------------------------- #

class TestMomentConditionSensitivity:
    def test_full_sphere_conditions_are_near_uniform(self):
        """For uniform coverage, sensitivity should be nonnegative with some active conditions."""
        directions = fibonacci_sphere(500)
        weights = moments.solve_optimal_weights(
            *moments.build_linear_system(directions),
            moments.local_density_baseline(directions),
        )
        sensitivity = rc.moment_condition_sensitivity(directions, weights)
        assert sensitivity.shape == (55,)
        assert (sensitivity >= 0).all()
        assert (sensitivity > 1e-10).sum() >= 20             # at least 20 active conditions

    def test_clustered_data_has_higher_variance_in_sensitivity(self):
        """Clustered coverage should produce more unequal condition sensitivities."""
        from _calibration.conftest import clustered_sphere
        rng = np.random.default_rng(42)
        directions = clustered_sphere(500, 4500, [0, 0, 1], 0.15, rng)
        weights = moments.solve_optimal_weights(
            *moments.build_linear_system(directions),
            moments.local_density_baseline(directions),
        )
        sensitivity = rc.moment_condition_sensitivity(directions, weights)
        # Some conditions should be much more sensitive than others
        assert sensitivity.max() / max(sensitivity.min(), 1e-300) > 10.
