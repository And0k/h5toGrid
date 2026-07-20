"""
Thorough tests for calibration math — both numpy kernels and dataset adapters.

Covers:
* :func:`calibrate_channel` — gain/bias recovery, outlier rejection
* Dataset-level calibration (via numpy extraction, no wrapper needed)
* :func:`bin_avg_3d` — 2-D spherical bin averaging (θ + φ)
* :func:`despike_channels` — per-channel despiking
* Legacy comparison — ``calibrate_channel`` vs ``incl_calibr_hy.calibrate``
* :func:`calibrate_pipeline` — full iterative bin → fit → reject loop
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from tcm.calibration.calibrate import calibrate_channel
from tcm.calibration.spatial_binning import bin_avg_3d, xyz2spherical
from tcm.calibration.filtering import despike_channels
from tcm.calibration.pipeline import calibrate_pipeline, PipelineConfig
from _calibration.conftest import apply_calibration, make_sphere_pts


# --------------------------------------------------------------------------- #
# calibrate_channel — gain/bias recovery
# --------------------------------------------------------------------------- #

@pytest.mark.calibration
class TestCalibrateChannel:
    """``calibrate_channel(data_3d)`` recovers gain + bias from ellipsoid data."""

    def test_recovers_gain_diagonal(self, known_geom):
        """For axis-aligned gain, |diag(recovered)| ≈ |diag(true)|."""
        rng = np.random.default_rng(42)
        sphere = make_sphere_pts(2000, rng)
        pts = apply_calibration(sphere, known_geom["gain"], known_geom["bias"], noise_sigma=0.02, rng=rng)
        gain, _ = calibrate_channel(pts)
        # Eigenvectors may flip sign; compare absolute diagonals
        np.testing.assert_allclose(
            np.sort(np.abs(np.diag(gain))),
            np.sort(np.abs(np.diag(known_geom["gain"]))),
            atol=0.05,
        )

    def test_calibrated_norm_is_unit(self, known_geom):
        """After calibration, ||gain @ raw + bias|| ≈ 1."""
        rng = np.random.default_rng(42)
        sphere = make_sphere_pts(2000, rng)
        pts = apply_calibration(sphere, known_geom["gain"], known_geom["bias"], noise_sigma=0.02, rng=rng)
        gain, bias = calibrate_channel(pts)
        norms = np.linalg.norm(gain @ (pts - bias), axis=0)
        np.testing.assert_allclose(norms.mean(), 1.0, atol=0.01)
        np.testing.assert_array_less(norms.std(), 0.05)

    def test_identity_gain_zero_bias(self):
        """Identity gain + zero bias → recovered gain ≈ I, bias ≈ 0."""
        rng = np.random.default_rng(42)
        sphere = make_sphere_pts(2000, rng)
        pts = apply_calibration(sphere, np.eye(3), np.zeros((3, 1)), noise_sigma=0.01, rng=rng)
        gain, bias = calibrate_channel(pts)
        np.testing.assert_allclose(np.abs(np.diag(gain)), 1.0, atol=0.02)
        np.testing.assert_allclose(bias, 0.0, atol=0.05)

    def test_outlier_rejection(self):
        """Outliers are rejected; calibration still recovers true geometry."""
        rng = np.random.default_rng(99)
        n = 2000
        gain_true = np.diag([1.2, 0.9, 1.0])
        bias_true = np.array([[2.0], [-1.5], [0.5]])
        sphere = make_sphere_pts(n, rng)
        pts = apply_calibration(sphere, gain_true, bias_true, noise_sigma=0.02, rng=rng)
        # Inject 5% extreme outliers
        n_out = n // 20
        idx = rng.choice(n, n_out, replace=False)
        pts[:, idx] = rng.normal(scale=100.0, size=(3, n_out))

        gain, bias = calibrate_channel(pts, max_iter=10, outlier_sigma=2.0)
        norms = np.linalg.norm(gain @ (pts - bias), axis=0)
        # Most inlier norms should be near 1
        inlier_mask = np.abs(norms - 1.0) < 0.5
        np.testing.assert_array_less(n * 0.8, inlier_mask.sum())  # at least 80% inliers recovered

    def test_single_iteration(self):
        """max_iter=1 runs one fit pass — no crash."""
        rng = np.random.default_rng(42)
        sphere = make_sphere_pts(500, rng)
        pts = apply_calibration(sphere, np.eye(3), np.zeros((3, 1)), noise_sigma=0.02, rng=rng)
        gain, bias = calibrate_channel(pts, max_iter=1)
        assert gain.shape == (3, 3)
        assert bias.shape == (3, 1)
        assert np.all(np.isfinite(gain)) and np.all(np.isfinite(bias))

    def test_empty_raises(self):
        """Empty input → exception (not hang)."""
        with pytest.raises(Exception):
            calibrate_channel(np.zeros((3, 0)))


# --------------------------------------------------------------------------- #
# Dataset-level calibration (numpy extraction, no wrapper)
# --------------------------------------------------------------------------- #

@pytest.mark.calibration
class TestDatasetCalibration:
    """Calibration on data extracted from xr.Dataset — no wrapper needed."""

    def test_default_channels(self, sample_magnetometer_ds):
        """Default channels=('Mx','My','Mz') works on sample fixture."""
        data_3d = np.vstack([sample_magnetometer_ds[c].values for c in ("Mx", "My", "Mz")])
        gain, bias = calibrate_channel(data_3d)
        norms = np.linalg.norm(gain @ (data_3d - bias), axis=0)
        np.testing.assert_allclose(norms.mean(), 1.0, atol=0.02)

    def test_custom_channels(self, sample_accelerometer_ds):
        """Channels kwarg selects accelerometer channels."""
        data_3d = np.vstack([sample_accelerometer_ds[c].values for c in ("Ax", "Ay", "Az")])
        gain, bias = calibrate_channel(data_3d)
        np.testing.assert_allclose(np.abs(np.diag(gain)), 1.0, atol=0.05)

    def test_iterative_on_dataset(self, sample_magnetometer_ds):
        """calibrate_channel with more iterations returns well-shaped gain/bias."""
        data_3d = np.vstack([sample_magnetometer_ds[c].values for c in ("Mx", "My", "Mz")])
        gain1, _ = calibrate_channel(data_3d, max_iter=1)
        gain10, _ = calibrate_channel(data_3d, max_iter=10)
        assert np.all(np.isfinite(gain1)) and np.all(np.isfinite(gain10))


# --------------------------------------------------------------------------- #
# bin_avg_3d — 2-D spherical bin averaging (θ + φ)
# --------------------------------------------------------------------------- #

@pytest.mark.calibration
class TestBinAvg3d:
    """``bin_avg_3d`` averages points by (θ, φ) bin on the sphere."""

    def test_nonempty_bins_inside_sphere(self):
        """Bin centres are averages of sphere points — inside by convexity."""
        rng = np.random.default_rng(0)
        pts = make_sphere_pts(10000, rng)
        centers, counts, _, _ = bin_avg_3d(pts, n_bins=50)
        for i in range(len(counts)):
            if counts[i] > 0:
                norm = np.linalg.norm(centers[:, i])
                # Single-point bins ≈ 1.0, multi-point bins < 1.0 (convexity)
                np.testing.assert_array_less(0.3, norm)
                np.testing.assert_array_less(norm, 1.0 + 1e-10)

    def test_output_shapes(self):
        """Correct output shapes regardless of input size."""
        rng = np.random.default_rng(0)
        pts = make_sphere_pts(100, rng)
        centers, counts, _, _ = bin_avg_3d(pts, n_bins=10)
        assert centers.shape[0] == 3
        assert counts.shape[0] == centers.shape[1]

    def test_empty_input(self):
        """Zero-column input → zero centres, zero counts."""
        centers, counts, _, _ = bin_avg_3d(np.zeros((3, 0)), n_bins=36)
        assert centers.shape[0] == 3
        assert counts.sum() == 0

    def test_all_points_covered(self):
        """Sum of counts == number of input points."""
        rng = np.random.default_rng(0)
        pts = make_sphere_pts(1000, rng)
        _, counts, _, _ = bin_avg_3d(pts, n_bins=50)
        np.testing.assert_equal(counts.sum(), 1000)

    def test_spherical_conversion_roundtrip(self):
        """xyz → spherical → xyz roundtrip."""
        rng = np.random.default_rng(42)
        pts = make_sphere_pts(100, rng)
        from tcm.calibration.spatial_binning import spherical2xyz
        rtp = xyz2spherical(pts)
        back = spherical2xyz(rtp)
        np.testing.assert_allclose(back, pts, atol=1e-12)


# --------------------------------------------------------------------------- #
# despike_channels — per-channel filtering
# --------------------------------------------------------------------------- #

@pytest.mark.calibration
class TestDespikeChannels:
    """``despike_channels`` removes spikes from each channel."""

    def test_clean_data_unchanged(self):
        """Clean data with gentle despike → most points kept."""
        rng = np.random.default_rng(42)
        pts = make_sphere_pts(500, rng)
        # Use very gentle offsets — sphere data spans [-1, 1] so rolling std is small
        out, mask = despike_channels(pts, offsets=(10, 10), blocks=(21, 7))
        np.testing.assert_array_less(450, mask.sum())  # at least 90% kept
        assert out.shape[0] == 3

    def test_spikes_removed(self):
        """Extreme spikes are masked out."""
        rng = np.random.default_rng(42)
        pts = make_sphere_pts(500, rng)
        # Inject extreme spikes
        pts[:, 0] = 1000.0
        pts[:, 1] = -1000.0
        _, mask = despike_channels(pts, offsets=(3, 2), blocks=(5, 3))
        # At least the injected spikes should be removed
        np.testing.assert_(
            not mask[0] or not mask[1],
            "Extreme spike not detected by despike",
        )


# --------------------------------------------------------------------------- #
# calibrate_pipeline — full iterative loop
# --------------------------------------------------------------------------- #

@pytest.mark.calibration
class TestCalibratePipeline:
    """``calibrate_pipeline`` — full bin → fit → reject loop."""

    def test_basic_pipeline(self):
        """Pipeline returns well-shaped gain/bias and inlier mask."""
        rng = np.random.default_rng(42)
        n = 2000
        gain_true = np.diag([1.2, 0.9, 1.0])
        bias_true = np.array([[2.0], [-1.5], [0.5]])
        sphere = make_sphere_pts(n, rng)
        pts = apply_calibration(sphere, gain_true, bias_true, noise_sigma=0.02, rng=rng)

        result = calibrate_pipeline(pts, PipelineConfig(robust=False))
        assert result.gain.shape == (3, 3)
        assert result.bias.shape == (3, 1)
        assert result.inlier_mask.shape == (n,)
        np.testing.assert_array_less(0, result.inlier_mask.sum())

    def test_pipeline_calibrated_norm(self):
        """After pipeline, ||gain @ inlier + bias|| ≈ 1."""
        rng = np.random.default_rng(42)
        n = 2000
        gain_true = np.diag([1.2, 0.9, 1.0])
        bias_true = np.array([[2.0], [-1.5], [0.5]])
        sphere = make_sphere_pts(n, rng)
        pts = apply_calibration(sphere, gain_true, bias_true, noise_sigma=0.02, rng=rng)

        result = calibrate_pipeline(pts, PipelineConfig(robust=False))
        inlier = pts[:, result.inlier_mask]
        norms = np.linalg.norm(result.gain @ (inlier - result.bias), axis=0)
        np.testing.assert_allclose(norms.mean(), 1.0, atol=0.05)

    def test_callback_called(self):
        """on_iter callback is invoked at least once."""
        rng = np.random.default_rng(42)
        sphere = make_sphere_pts(500, rng)
        pts = apply_calibration(sphere, np.eye(3), np.zeros((3, 1)), noise_sigma=0.02, rng=rng)

        calls = []
        calibrate_pipeline(
            pts,
            PipelineConfig(robust=False),
            on_iter=lambda step, dc, pct, g, b: calls.append((step, dc, pct)),
        )
        assert len(calls) > 0


# --------------------------------------------------------------------------- #
# Legacy comparison — calibrate_channel vs incl_calibr_hy.calibrate
# --------------------------------------------------------------------------- #

@pytest.mark.calibration
@pytest.mark.comparison
class TestCalibrationComparison:
    """
    ``calibrate_channel`` (eigenvalue-based) vs ``incl_calibr_hy.calibrate``
    (quadric-form-based) on the same synthetic data.

    Both should produce calibrated data with ||.|| ≈ 1.
    """

    @staticmethod
    def _legacy_fit_quadric_form(s: np.ndarray):
        """Reproduce ``incl_calibr_hy.fit_quadric_form`` for comparison."""
        D = np.array([
            s[0] ** 2, s[1] ** 2, s[2] ** 2,
            2 * s[1] * s[2], 2 * s[0] * s[2], 2 * s[0] * s[1],
            2 * s[0], 2 * s[1], 2 * s[2],
            np.ones_like(s[0]),
        ])
        S = np.dot(D, D.T)
        S_11, S_12 = S[:6, :6], S[:6, 6:]
        S_21, S_22 = S[6:, :6], S[6:, 6:]
        c_inv = np.array([
            [0, 0.5, 0.5, 0, 0, 0],
            [0.5, 0, 0.5, 0, 0, 0],
            [0.5, 0.5, 0, 0, 0, 0],
            [0, 0, 0, -0.25, 0, 0],
            [0, 0, 0, 0, -0.25, 0],
            [0, 0, 0, 0, 0, -0.25],
        ])
        E = np.dot(c_inv, S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))
        E_w, E_v = np.linalg.eig(E)
        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0:
            v_1 = -v_1
        v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)
        M = v_1[np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], np.int8)]
        n = v_2[:-1, np.newaxis]
        d = v_2[3]
        return M, n, d

    @staticmethod
    def _legacy_calibrate(raw3d: np.ndarray):
        """Reproduce ``incl_calibr_hy.calibrate`` for comparison."""
        F = np.float64(1)
        mean_Hxyz = np.mean(raw3d, 1)[:, np.newaxis]
        s = np.array(raw3d - mean_Hxyz)
        Q, n, d = TestCalibrationComparison._legacy_fit_quadric_form(s)
        Q_inv = linalg.inv(Q)
        b = -np.dot(Q_inv, n) + mean_Hxyz
        a2d = np.real(
            F / np.sqrt(np.dot(n.T, np.dot(Q_inv, n)) - d) * linalg.sqrtm(Q)
        )
        return a2d, b

    def test_both_calibrate_to_unit_sphere(self):
        """Both pipelines produce calibrated data with ||.|| ≈ 1."""
        rng = np.random.default_rng(42)
        n = 500
        gain_true = np.diag([1.2, 0.9, 1.0])
        bias_true = np.array([[2.0], [-1.5], [0.5]])
        sphere = make_sphere_pts(n, rng)
        pts = apply_calibration(sphere, gain_true, bias_true, noise_sigma=0.02, rng=rng)

        # Legacy: norm_field(raw, a2d, b) = a2d @ (raw - b)
        a2d, b = self._legacy_calibrate(pts)
        cal_legacy = a2d @ (pts - b)
        norms_legacy = np.linalg.norm(cal_legacy, axis=0)

        # New: gain @ (raw - bias)
        gain, bias = calibrate_channel(pts)
        cal_new = gain @ (pts - bias)
        norms_new = np.linalg.norm(cal_new, axis=0)

        # Both should be close to unit sphere
        np.testing.assert_allclose(norms_legacy.mean(), 1.0, atol=0.01)
        np.testing.assert_allclose(norms_new.mean(), 1.0, atol=0.01)
        np.testing.assert_array_less(norms_legacy.std(), 0.05)
        np.testing.assert_array_less(norms_new.std(), 0.05)

    def test_gain_matrices_similar(self):
        """Gain diagonals from both pipelines should have similar magnitudes."""
        rng = np.random.default_rng(42)
        n = 500
        gain_true = np.diag([1.2, 0.9, 1.0])
        bias_true = np.array([[2.0], [-1.5], [0.5]])
        sphere = make_sphere_pts(n, rng)
        pts = apply_calibration(sphere, gain_true, bias_true, noise_sigma=0.01, rng=rng)

        a2d, _ = self._legacy_calibrate(pts)
        gain, _ = calibrate_channel(pts)

        # Sort absolute diagonals for comparison
        np.testing.assert_allclose(
            np.sort(np.abs(np.diag(gain))),
            np.sort(np.abs(np.diag(a2d))),
            atol=0.05,
        )

    def test_identity_calibration(self):
        """Identity gain + zero bias — both pipelines recover ~identity."""
        rng = np.random.default_rng(42)
        n = 500
        sphere = make_sphere_pts(n, rng)
        pts = apply_calibration(sphere, np.eye(3), np.zeros((3, 1)), noise_sigma=0.01, rng=rng)

        a2d, b = self._legacy_calibrate(pts)
        gain, bias = calibrate_channel(pts)

        np.testing.assert_allclose(np.abs(np.diag(a2d)), 1.0, atol=0.03)
        np.testing.assert_allclose(np.abs(np.diag(gain)), 1.0, atol=0.03)
        # Both legacy b and new bias are raw-space centers — near zero
        np.testing.assert_array_less(np.linalg.norm(b), 0.5)
        np.testing.assert_array_less(np.linalg.norm(bias), 0.05)
