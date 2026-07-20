"""
Orientation reference calibration: the rotation between the sensor's own axes and a world-aligned
frame, from known reference events — built on top of `calibration.py`'s ellipsoid fit, which only
gets each sensor's own reading onto its own calibrated unit sphere, not aligned to anything external.

Two independent reference events, each usable on its own:
  - zero tilt (`zeroing_rotation`): accelerometer sample(s) from when the instrument hung plumb.
    Folded directly into the accelerometer's a2d (`apply_zeroing_rotation`) rather than carried as a
    separate reference vector, so a fixed mounting offset between the accelerometer's own axes and the
    housing's true vertical does not need to be known, and every later `tilt_from_vertical` reading is
    simply arccos(unit_vector[2]) — no reference to pass around at read time.
  - known north (`calibrate_heading_reference`): magnetometer + *simultaneous* accelerometer sample(s)
    from when a chosen sensor axis was known to point at north, at whatever tilt it happened to be at
    — the accelerometer at that same moment supplies the local horizontal plane, so this event does
    not also need to be level. Unlike zero-tilt, this is *not* folded into a2d: heading needs the
    *current* horizontal plane at read time (both sensors, every reading), and that plane changes as
    the instrument tilts, so no single fixed rotation could absorb it the way `zeroing_rotation` does
    for tilt; it stays a separate scalar offset applied at read time instead (see calibration_wiki.md,
    "Heading offset: why not folded in").

Both calibrations are defined directly from the reference event, not from a declination/dip model, so
whichever "north" the event actually used (true or magnetic) is automatically what `heading_and_tilt`
reports headings relative to afterward — no separate declination correction needed, but also no way
to tell afterward which one was used if that was not tracked at calibration time.

Note on scope: this module's "heading" is the bearing of a *fixed sensor axis* relative to north (for
relating the sensor's own frame to a compass direction) — a different quantity from a tilt-current-
meter's flow direction (the azimuth of the *tilt itself*, undefined at zero tilt by construction,
since "no lean" means no measurable flow direction: e.g. a Gx=Gy=0 reading degenerates any formula
built on the tilt's own horizontal components, correctly, since there is no flow direction to report).
Both use the same accelerometer+magnetometer fusion underneath; which one is wanted depends on whether
the instrument has a meaningful fixed heading of its own, or reports flow/drag direction instead.
"""
import logging

import numpy as np
from typing import Tuple
from tcm.calibration.calibrate import SensorCalibration, to_unit_vector

log = logging.getLogger(__name__)

FORWARD_AXIS = np.array([1., 0., 0.])                     # default heading-reference axis, sensor frame





def rotate(r_from: np.ndarray, r_to: np.ndarray) -> np.ndarray:
    """
    Rotation matrix R aligning unit vectors r_from → r_to (Rodrigues' formula)
    """
    r_f = np.ravel(r_from)
    r_t = np.ravel(r_to)
    if np.array_equal(r_f, r_t):
        return np.eye(3)
    cross = np.cross(r_f, r_t)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew * (1 - r_f @ r_t) / (cross @ cross)


def tilt_from_vertical(Gxyz: np.ndarray) -> np.ndarray:
    """
    Inclination (radians, 0 = plumb)
    :param Gxyz: (3, N) accelerometer samples.
    :return: (N,) inclination in [0, pi].
    """
    return np.arctan2(np.linalg.norm(Gxyz[:-1, :], axis=0), Gxyz[2, :])


def zeroing_rotation(
    accel_at_zero_tilt: np.ndarray, accel_calibration: SensorCalibration
) -> Tuple[np.ndarray, float, float]:
    """
    Rotation canonicalizing "zero tilt" to the sensor Z-axis, from accelerometer sample(s) recorded
    while the instrument hung plumb.

    Usage: fold into a2d once via `apply_zeroing_rotation`, rather than keeping this rotation (
    or a raw zenith vector) around separately.

    :param accel_at_zero_tilt: (3, N) raw accelerometer samples from the zero-tilt event(s); N > 1
        only helps by averaging out noise, all samples should reflect the same true orientation.
    :param accel_calibration: the accelerometer's ellipsoid fit (see `calibration.calibrate`), *before*
        any previous zeroing rotation — folding this twice would double-rotate.
    :return (R, angular_spread):
        R: (3, 3) rotation matrix;
        angular_spread: Standard deviation (degrees) of inclination angles relative to the empirical zenith
        inclination: (degrees) from absolute to the empirical zenith
    """
    unit = to_unit_vector(accel_at_zero_tilt, accel_calibration)
    zenith = (mean_vec := unit.mean(axis=1)) / np.linalg.norm(mean_vec)
    cos_theta = zenith @ unit  # (3,) @ (3, N) -> (N,) <=> (N, 3) @ (3,) -> (N,)
    angular_spread = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)).std()).item()
    incl_rad = np.degrees(tilt_from_vertical(zenith[:, None])).item()
    return (rotate(zenith, np.array([0.0, 0.0, 1.0])), incl_rad, angular_spread)


def apply_zeroing_rotation(calibration: SensorCalibration, rotation: np.ndarray) -> SensorCalibration:
    """Fold a `zeroing_rotation` result into a calibration once, so it needn't be reapplied at read time."""
    return SensorCalibration(calibration.bias, rotation @ calibration.a2d)
def _project_horizontal(vector: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Component of `vector` perpendicular to `up`, normalized — both (3, N)."""
    horizontal = vector - (vector * up).sum(0) * up
    return horizontal / np.linalg.norm(horizontal, axis=0)


def _bearing(field: np.ndarray, up: np.ndarray, reference_axis: np.ndarray) -> np.ndarray:
    """
    Signed angle (radians) from `reference_axis` to `field`, both projected perpendicular to `up` —
    positive = counterclockwise around `up` (right-hand rule). Used identically by
    `calibrate_heading_reference` and `heading_and_tilt`, so this sign convention only has to be
    self-consistent between the two, never externally correct; see module docstring.

    :param field: (3, N) unit vector(s) (typically the calibrated magnetometer reading).
    :param up: (3, N) local vertical unit vector(s) (calibrated accelerometer reading) defining
        "horizontal"; must not be (anti)parallel to `reference_axis`.
    :param reference_axis: (3,) sensor-frame axis bearing is measured from, broadcasting against N.
    :return: (N,) angle in (-pi, pi].
    """
    field_h = _project_horizontal(field, up)
    reference_h = _project_horizontal(np.broadcast_to(reference_axis[:, np.newaxis], up.shape), up)
    return np.arctan2((np.cross(reference_h, field_h, axis=0) * up).sum(0), (reference_h * field_h).sum(0))


def calibrate_heading_reference(mag_at_north: np.ndarray, mag_calibration: SensorCalibration,
                                 accel_at_north: np.ndarray, accel_calibration: SensorCalibration,
                                 forward_axis: np.ndarray = FORWARD_AXIS) -> float:
    """
    Heading offset (radians) between `forward_axis` and (true or magnetic, per whichever was used
    when pointing the instrument — see module docstring) north, from sample(s) recorded while
    `forward_axis` was known to point there.

    :param mag_at_north, accel_at_north: (3, N) raw magnetometer/accelerometer samples recorded
        *simultaneously* during the north-pointing event(s); the accelerometer establishes the local
        horizontal plane at that same moment, so the event need not also be level.
    :param mag_calibration, accel_calibration: each sensor's calibration (`accel_calibration` may or
        may not have a zeroing rotation folded in already — either is fine here, `_bearing` reads
        whatever horizontal plane `up` implies directly from the samples given).
    :param forward_axis: (3,) sensor-frame axis that was pointed at north; must not be (anti)parallel
        to the accelerometer reading at that moment (i.e. the instrument wasn't pointed straight up
        or down while this axis was meant to indicate a heading).
    :return: offset in (-pi, pi], such that heading = _bearing(...) - offset (see `heading_and_tilt`);
        the circular mean over samples if N > 1 (handles wraparound correctly, unlike a plain mean).
    """
    field = to_unit_vector(mag_at_north, mag_calibration)
    up = to_unit_vector(accel_at_north, accel_calibration)
    offset = -_bearing(field, up, forward_axis)          # heading convention: north -> forward, see docstring
    heading_offset = np.arctan2(np.sin(offset).mean(), np.cos(offset).mean())
    log.info("calibrate_heading_reference: n=%d offset=%.2f deg angular_spread=%.3g deg",
              field.shape[1], np.degrees(heading_offset), np.degrees(offset.std()))
    return heading_offset


def heading_and_tilt(mag_raw: np.ndarray, mag_calibration: SensorCalibration, accel_raw: np.ndarray,
                      accel_calibration: SensorCalibration, heading_offset: float,
                      forward_axis: np.ndarray = FORWARD_AXIS) -> tuple[np.ndarray, np.ndarray]:
    """
    Compass heading (radians, counterclockwise from north around vertical) and tilt (radians from
    vertical) for arbitrary samples, using the reference from `calibrate_heading_reference` and an
    accelerometer calibration with a `zeroing_rotation` already folded in.

    :param mag_raw, accel_raw: (3, N) raw samples, recorded simultaneously.
    :param mag_calibration: the magnetometer's ellipsoid fit.
    :param accel_calibration: the accelerometer's fit, with a zeroing rotation folded in (see
        `apply_zeroing_rotation`) — required here, unlike in `calibrate_heading_reference`, since
        tilt is read directly off the Z-component (see `tilt_from_vertical`).
    :param heading_offset: from `calibrate_heading_reference`.
    :param forward_axis: (3,) must match what `calibrate_heading_reference` used.
    :return: heading in [0, 2*pi), tilt in [0, pi].
    """
    field = to_unit_vector(mag_raw, mag_calibration)
    up = to_unit_vector(accel_raw, accel_calibration)
    tilt = np.arccos(np.clip(up[2], -1., 1.))
    heading = (_bearing(field, up, forward_axis) - heading_offset) % (2 * np.pi)
    return heading, tilt


def azimuth_shift(
    mag_raw: np.ndarray, mag_calibration: SensorCalibration,
    accel_raw: np.ndarray, accel_calibration: SensorCalibration,
    forward_axis: np.ndarray = FORWARD_AXIS,
) -> float:
    """Azimuth of magnetic North relative to *forward_axis* (degrees).

    Bearing of the horizontal magnetic field from *forward_axis*, averaged
    over samples via circular mean — the correction such that
    ``true_azimuth = sensor_azimuth + azimuth_shift``.

    Replaces ``calibration.zeroing.find_azimuth_shift`` which required
    velocity/magnitude calculation (``kVabs``).  This version works on
    calibrated **unit vectors** only — no magnitude coefficients needed,
    no rounding-error dependency on inclination-to-magnitude conversion.

    Delegates to :func:`calibrate_heading_reference` (same math, opposite
    sign convention): ``azimuth_shift = -degrees(calibrate_heading_reference(...))``.

    :param mag_raw: (3, N) raw magnetometer samples.
    :param mag_calibration: magnetometer ellipsoid fit.
    :param accel_raw: (3, N) raw accelerometer samples (simultaneous with *mag_raw*).
    :param accel_calibration: accelerometer calibration (with or without zeroing rotation).
    :param forward_axis: (3,) sensor-frame axis; default = x.
    :return: shift in degrees.
    """
    return -np.degrees(calibrate_heading_reference(
        mag_raw, mag_calibration, accel_raw, accel_calibration, forward_axis,
    ))
