# Magnetometer/Accelerometer Calibration — Usage Guide

Scope of this document: what to call and when. Derivations, notation, empirical validation, and design
rationale live in `calibration_wiki.md`; every constraint below links to the section that proves it.
Nothing here should need re-deriving or re-verifying to trust — that work has already been done once,
in one place.

## 1. Module responsibilities

| Module | Responsible for |
|---|---|
| `run.py` | The actual entry point: `run_calibration` loads raw data, despikes, fits, rejects outliers, writes coefficients to HDF5, saves diagnostic plots. Most callers want this and nothing below it. |
| `pipeline.py` | `calibrate_pipeline` — the fit/reject loop `run_calibration` wraps, usable standalone on an in-memory array (no file I/O, no plotting). |
| `filtering.py` | `despike_channels` — per-channel spike removal, run before fitting. |
| `spatial_binning.py` | `bin_avg_3d` — optional point-count reduction for very dense/uneven raw data, ahead of fitting. |
| `calibrate.py` | The ellipsoid fit itself: `calibrate`, `weighted_fit_quadric`, `to_unit_vector`, `SensorCalibration`. The math core everything else builds on. |
| `moments.py` | The sample-weighting scheme `calibrate.py` uses internally to correct for uneven angular coverage. Not normally called directly. |
| `robust.py` | Outlier-rejecting fit (`autocalibrate`), field-data autocalibration (`field_autocalibrate`), and quality diagnostics (`uncertainty_at`, `expected_direction_error`, `coverage_at`, `anomalous_time_windows`). |
| `orientation.py` | Turns a calibrated-but-unaligned reading into actual tilt and compass heading: `zeroing_rotation`, `calibrate_heading_reference`, `heading_and_tilt`. |
| `visualization.py`, `vis_common.py`, `vis_coverage.py` | Diagnostic plots (ellipsoid fit, despiking, sphere-coverage maps). Consumed by `run.py`; not usually called directly. |

## 2. Quick start

Most calibration tasks:

```python
from tcm.calibration.run import run_calibration

coefs = run_calibration(cfg)  # cfg: Hydra-composed config or plain dict — see run.py's own docstring
                               # for every +override; despiking, fitting, rejection, plots, and HDF5
                               # output are all handled internally.
```

Building a calibration from an in-memory array, without the file/config layer:

```python
from tcm.calibration import calibrate as cal
from tcm.calibration import orientation as ori
from tcm.calibration import robust as rc

mag_cal, history = rc.autocalibrate(raw_mag, field_magnitude=52000.)   # nT, local IGRF total field
accel_cal, _ = rc.autocalibrate(raw_accel, field_magnitude=9.81)       # m/s^2, local g

# No dedicated calibration rotation available? Calibrate from operational data instead. This refuses
# explicitly (status != "ok") rather than returning an unreliable fit silently — see Section 4.
result = rc.field_autocalibrate(raw_mag, field_magnitude=52000.)
if result["status"] == "ok":
    mag_cal = result["calibration"]

# Optional: zero-tilt reference, folded into a2d for both sensors at once (rigidly mounted together)
rotation = ori.zeroing_rotation(raw_accel[:, zero_tilt_mask], accel_cal)
accel_cal = ori.apply_zeroing_rotation(accel_cal, rotation)
mag_cal = ori.apply_zeroing_rotation(mag_cal, rotation)

# Optional: north reference
heading_offset = ori.calibrate_heading_reference(
    raw_mag[:, north_mask], mag_cal, raw_accel[:, north_mask], accel_cal)

# Apply to arbitrary samples
heading, tilt = ori.heading_and_tilt(raw_mag, mag_cal, raw_accel, accel_cal, heading_offset)
```

## 3. Calibration procedure

1. **Rotation**: rotate through as many distinct axes as practical, not just one — a single-axis
   rotation under-constrains the fit. *(wiki §2 for the precise conditioning requirement)*
2. **Reference events** (optional): record the time ranges when the instrument (a) hung with zero
   tilt, (b) had a known axis pointed at north/south. The north event does not need its own level
   pause — any tilt is fine, provided the accelerometer is recorded at the same time.
3. **Fit** each sensor separately via `rc.autocalibrate` — magnetometer and accelerometer have
   different `field_magnitude` and different noise behavior; do not combine them.
4. **Check quality** before trusting the result:
   - `autocalibrate`'s returned `history` — `residual_p95` in physical units (multiply by
     `field_magnitude`; `radial_residuals` itself returns `radius/field_magnitude - 1`).
   - `rc.coverage_at(raw, calibration)` — zero-density directions are a coverage hole, not something
     weighting can correct (Section 5).
   - `rc.anomalous_time_windows(raw, calibration, field_magnitude)` — a flagged direction with other,
     low-error time windows nearby points at a transient (interference, shock, temperature step), not
     a geometry problem.
5. **Orientation** (if needed): `zeroing_rotation`/`apply_zeroing_rotation` once for both sensors;
   `calibrate_heading_reference`'s `heading_offset` is kept as a separate constant, not folded into a
   matrix, because it depends on the horizontal plane at read time rather than being fixed.

## 4. Choosing a target region

By default, every fit targets the whole sphere, regardless of what the calibration data covers — this
is a deliberate default, not a limitation to work around. *(wiki §4.4)* Restricting the target is an
opt-in choice, only appropriate when the device genuinely cannot and will not operate outside a known
region (e.g. a seabed instrument whose axis always faces one current direction). Two independent
mechanisms exist; using the wrong one is a common mistake:

- **Changes the fit itself.** `calibrate.weighted_fit_quadric` accepts `target_directions`; the
  convenience wrappers `calibrate.calibrate` and `robust.autocalibrate` do not expose it. Using it
  today means assembling the fit by hand: `center = raw.mean(1, keepdims=True)`, then
  `weighted_fit_quadric(raw - center, target_directions=...)`, then the same `bias`/`a2d` extraction
  steps `calibrate.calibrate` performs internally.
- **Changes only the reported error, not the fit.** `robust.field_autocalibrate` passes the data's own
  achieved directions as `target_directions` to `expected_direction_error` — the fit it calls
  (`autocalibrate`) still targets the whole sphere untouched. For "device faces one direction," this is
  usually what is actually wanted: an honest error estimate for where the device really operates,
  without narrowing what the fit itself optimizes for.

## 5. Operating constraints

Each item states the constraint and what to do about it. Proof, numbers, and derivation are in
`calibration_wiki.md`, linked per item.

- **Weighting corrects uneven angular sampling; it does not fill coverage holes.** A direction with
  zero samples near it stays uncalibratable regardless of weighting — check with `coverage_at`, not by
  inspecting the weighted residual. *(wiki §9.2, contrasting `coverage_at` with `uncertainty_at`)*
- **The fit's target is always the whole sphere unless you opt out (Section 4).** This is chosen
  deliberately so a device needing whole-sphere accuracy is not silently graded against whatever the
  calibration protocol happened to cover. *(wiki §4.4)*
- **Regularization strength is chosen automatically per fit**, not fixed, because a fixed value that
  works for one data geometry can fail for another of the same size. Override only to diagnose a
  specific case. *(wiki §5.2)*
- **Weighted direction estimation is iterative (IRLS), not circular** — an initial unweighted fit
  supplies the directions used to compute weights, then the fit is redone. One iteration suffices for
  moderate anisotropy; a strongly elongated raw ellipsoid may need more. *(wiki §6)*
- **An error in `field_magnitude` only rescales `a2d`** — bias and shape are unaffected. If only
  direction matters, an arbitrary value can be passed and the result's magnitude simply ignored.
- **`field_autocalibrate` refuses rather than fitting on insufficient data** — below `MIN_FIELD_SAMPLES`
  or `MIN_FIELD_DIRECTION_SPREAD_DEG` of orientation spread, it returns `status="insufficient data"`
  instead of a fit that happens to be unreliable. The spread check is on raw sample magnitude, not
  normalized direction — the two are not interchangeable here. *(wiki §9.4)* Its optional quadrupole
  correction (`tilt_reference_cos`) is not yet implemented and raises `NotImplementedError` rather than
  silently doing nothing.
- **Outlier rejection assumes contamination is a minority of the sample.** It degrades as contaminated
  fraction approaches half the data — a property of robust statistics generally, not this
  implementation specifically. *(wiki §9.1)*
- **`heading_and_tilt`'s sign convention is internally consistent, not tied to an external standard**
  (NED/ENU or similar). Verify against one known relative rotation before integrating into a pipeline
  that assumes a specific convention. *(wiki §8.2, and `test_orientation.py` for the verification
  methodology)*
- **Heading is not the same quantity as TCM drift direction.** `heading_and_tilt` reports a fixed
  sensor axis's bearing from north. Flow/drag direction is a different formula that is undefined at
  zero tilt by construction (no tilt, no drift direction to report) — this is correct behavior for that
  formula, not a bug. Both consume the same calibrated readings; do not conflate which one to call.
- **Performance**: the weight solver scales to tens of thousands of samples in well under a second;
  density/coverage diagnostics (`local_density_baseline`, `coverage_at`) are the parts most likely to
  dominate cost at very large `N`. *(wiki §5, §5.3)*

## 6. Tested

See `test_*.py` / `conftest.py` for the suite. Each claim above that is backed by a specific test is
cross-referenced from its wiki section, not repeated here.
