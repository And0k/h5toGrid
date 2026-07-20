"""
Tests for `orientation.py`. Ground truth is checked via *relative* consistency (a known additional yaw
around the current local vertical must shift heading by exactly that amount) rather than deriving an
absolute expected heading by hand through several composed rotations — the latter is easy to get
subtly wrong (done twice while developing this); the former only requires one rotation at a time.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tcm.calibration import orientation as ori
from tcm.calibration.calibrate import SensorCalibration, to_unit_vector
from tcm.calibration.orientation import tilt_from_vertical

IDENTITY_CAL = SensorCalibration(bias=np.zeros((3, 1)), a2d=np.eye(3))
UP_WORLD = np.array([0., 0., 1.])
FIELD_WORLD = np.array([np.cos(np.deg2rad(65)), 0., -np.sin(np.deg2rad(65))])   # 65 deg dip


class TestRotate:
    def test_maps_source_to_target(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.normal(size=3)
            a /= np.linalg.norm(a)
            b = rng.normal(size=3)
            b /= np.linalg.norm(b)
            R = ori.rotate(a, b)
            assert R @ a == pytest.approx(b, abs=1e-9)
            assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)      # orthogonal
            assert np.linalg.det(R) == pytest.approx(1.)              # proper rotation, no reflection

    def test_identity_when_already_aligned(self):
        v = np.array([1., 0., 0.])
        assert ori.rotate(v, v) == pytest.approx(np.eye(3))


class TestZeroingRotation:
    def test_folds_to_canonical_up(self):
        """After folding, the zero-tilt reading's direction must land on exactly [0, 0, 1]."""
        rng = np.random.default_rng(1)
        accel_zt = np.array([[3.], [1.], [8.]]) + rng.normal(scale=0.05, size=(3, 20))
        rotation, *stat = ori.zeroing_rotation(accel_zt, IDENTITY_CAL)
        folded = ori.apply_zeroing_rotation(IDENTITY_CAL, rotation)
        canonical = to_unit_vector(accel_zt, folded).mean(1)
        assert canonical / np.linalg.norm(canonical) == pytest.approx([0., 0., 1.], abs=1e-2)

    def test_fold_in_is_exactly_a_fixed_rotation_of_every_reading(self):
        """apply_zeroing_rotation must transform *every* reading by the same rotation, not just the
        zero-tilt samples it was derived from — this is what lets it be folded in once and forgotten."""
        rng = np.random.default_rng(2)
        rotation, *stat = ori.zeroing_rotation(
            rng.normal(size=(3, 10)) + np.array([[3.0], [1.0], [8.0]]), IDENTITY_CAL
        )
        folded = ori.apply_zeroing_rotation(IDENTITY_CAL, rotation)
        raw = rng.normal(size=(3, 5))
        assert to_unit_vector(raw, folded) == pytest.approx(rotation @ to_unit_vector(raw, IDENTITY_CAL))

    def test_tilt_from_vertical_is_zero_at_the_calibration_event(self):
        rng = np.random.default_rng(3)
        accel_zt = np.array([[3.], [1.], [8.]]) + rng.normal(scale=0.02, size=(3, 20))
        folded = ori.apply_zeroing_rotation(IDENTITY_CAL, ori.zeroing_rotation(accel_zt, IDENTITY_CAL)[0])
        assert np.degrees(tilt_from_vertical(to_unit_vector(accel_zt, folded)).mean()) < 1.

class TestHeadingAndTilt:
    def test_heading_tracks_a_known_relative_yaw_exactly(self):
        """Core correctness check: a yaw of psi around the *current* local vertical, applied after the
        north-calibration pose, must shift the reported heading by exactly psi (mod 360 deg)."""
        forward_target = np.array([np.cos(np.deg2rad(20)), 0., np.sin(np.deg2rad(20))])
        r_north = Rotation.align_vectors([[1., 0., 0.]], [forward_target])[0]
        heading_offset = ori.calibrate_heading_reference(
            r_north.apply(FIELD_WORLD)[:, np.newaxis], IDENTITY_CAL,
            r_north.apply(UP_WORLD)[:, np.newaxis], IDENTITY_CAL)

        for yaw_deg in (0, 15, 45, 90, 137, -60):
            r_test = r_north * Rotation.from_euler("z", yaw_deg, degrees=True)
            heading, tilt = ori.heading_and_tilt(
                r_test.apply(FIELD_WORLD)[:, np.newaxis], IDENTITY_CAL,
                r_test.apply(UP_WORLD)[:, np.newaxis], IDENTITY_CAL, heading_offset)
            assert np.degrees(heading[0]) == pytest.approx(yaw_deg % 360, abs=1e-6)
            assert np.degrees(tilt[0]) == pytest.approx(20., abs=1e-6)

    def test_heading_offset_is_zero_when_calibration_event_pointed_exactly_at_north(self):
        r_north = Rotation.align_vectors([[1., 0., 0.]], [[1., 0., 0.]])[0]   # forward already == north
        offset = ori.calibrate_heading_reference(
            r_north.apply(FIELD_WORLD)[:, np.newaxis], IDENTITY_CAL,
            r_north.apply(UP_WORLD)[:, np.newaxis], IDENTITY_CAL)
        assert offset == pytest.approx(0., abs=1e-9)

    def test_circular_mean_handles_wraparound(self):
        """Samples straddling the 0/360 boundary must average to ~0, not ~180 (a plain arithmetic
        mean of angles would get this backwards)."""
        forward = np.array([1., 0., 0.])
        angles_deg = [-2., -1., 0., 1., 2.]
        mags, accels = [], []
        for a in angles_deg:
            r = Rotation.from_euler("z", a, degrees=True)
            mags.append(r.apply(FIELD_WORLD))
            accels.append(r.apply(UP_WORLD))
        offset = ori.calibrate_heading_reference(np.array(mags).T, IDENTITY_CAL,
                                                   np.array(accels).T, IDENTITY_CAL, forward_axis=forward)
        assert np.degrees(offset) == pytest.approx(0., abs=1e-6)
