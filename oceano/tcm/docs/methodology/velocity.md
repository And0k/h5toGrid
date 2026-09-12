# Flow velocity computation from accelerometer and magnetometer data

## Converting accelerometer and magnetometer counts into the *gravitational acceleration* and *normalized magnetic field* vectors

<img src="../images/inclin_out_angles.svg" alt="TCM" align="right" width="40%">

From the accelerometer $`U_{Gx}, U_{Gy}, U_{Gz}`$ and magnetometer
$`U_{Bx}, U_{By}, U_{Bz}`$ readings, the projections of the gravitational
acceleration $`G_{x}, G_{y}, G_{z}`$ and of the magnetic field $`B_{x}, B_{y}, B_{z}`$
onto the axes of the sensor-bound coordinate frame are computed. For example,
for the accelerometer:

$$
\mathbf{G} = \mathbf{A} \cdot (\mathbf{U}_G - \mathbf{U}_{G0}),
$$

where

- $`\mathbf{A}`$ — 3×3 matrix of scale and rotation coefficients —
  [`Ag`](../reference/config_reference.md#inputcoefs--calibration-coefficients);
- $`\mathbf{U}_{G0}`$ — 3×1 offset vector in ADC counts — `Cg`;
- $`\mathbf{G} = [G_{x}, G_{y}, G_{z}]^{T}`$ — gravitational acceleration vector, g;
- $`\mathbf{U}_{G} = [U_{Gx}, U_{Gy}, U_{Gz}]^{T}`$ — accelerometer data in ADC counts.

An analogous expression is used for the magnetic field vector $`\mathbf{B}`$
measured by the magnetometer: its offset vector is $`\mathbf{U}_{B0}`$ — `Ch`,
its coefficient matrix is $`\mathbf{M}`$ — `Ah`. Each inclinometer's
accelerometer and magnetometer have their own offset vectors and coefficient
matrices, determined during
[preliminary calibration](../python_developer_guide/calibration.md).

## Vertical alignment ($R_{z}$)

The axes of the sensor's own coordinate frame do not necessarily coincide with
the instrument's vertical. The rotation matrix $R_{z}$ — an orthogonal
$3\times3$ matrix — rotates the calibrated sensor frame so that its Z-axis
coincides with the direction of the gravity force. It is computed during
calibration from accelerometer data recorded while the instrument hung
vertically:

- **From a data interval** (`input.calib.time_ranges_zeroing`): the pipeline
  averages the calibrated accelerometer unit vectors over the window and
  builds the rotation aligning the empirical zenith with $`[0, 0, 1]^{T}`$ via
  Rodrigues' formula — see
  [Calibration Wiki §8.1](calibration_wiki.md#81-zero-tilt-fold-in).
- **From a single vector** (`input.calib.g0xyz`): an uncalibrated
  accelerometer vector measured at known zero tilt is converted to a unit
  vector and rotated to $[0, 0, 1]^{T}$ the same way; it overrides any $R_{z}$
  computed from `time_ranges_zeroing`.

The matrix $R_{z}$ is always applied to the calibration matrices when
computing velocity — to the accelerometer's ($`\mathbf{A}_{g}`$) as well as the
magnetometer's ($`\mathbf{A}_{h}`$) — because the sensors are rigidly mounted
together and must stay in a common coordinate frame:

$$
\mathbf{A}_g' = R_z \cdot \mathbf{A}_g, \qquad
\mathbf{A}_h' = R_z \cdot \mathbf{A}_h.
$$

These rotated matrices are then used in the calibration formulas —
$\mathbf{G} = \mathbf{A}_{g}' \cdot (\mathbf{U}_{G} - \mathbf{U}_{G0})$ and
$\mathbf{B} = \mathbf{A}_{h}' \cdot (\mathbf{U}_{B} - \mathbf{U}_{B0})$ — so that
tilt and azimuth are computed relative to the instrument's vertical rather
than the sensor's original axes. The rotation itself is not stored in the
output: it is folded into the effective calibration matrices before the
physical conversion.

## Azimuth shift ($`\psi_{\text{shift}}`$)

The tilt azimuth $\psi$, computed by formula (2), is given in the sensor's own
coordinate frame. To obtain the geographic direction of the tilt, a constant
shift $\psi_{\text{shift}}$ (coefficient `azimuth_shift_deg`, in degrees) is
subtracted from it:

$$
\psi_{\text{geographic}} = \psi_{\text{shift}} - \psi,
$$

where $\psi$ is the value given by formula (2). The shift sign is chosen to
compensate the magnetometer sign inversion (the default is
`azimuth_shift_deg = 180^\circ`); the corrected direction is then used to
recompute the Cartesian velocity components $u, v$.

$`\psi_{\text{shift}}`$ is **not** folded into a calibration matrix — unlike
$`R_{z}`$, it cannot be absorbed into a static rotation because the relevant
horizontal plane changes as the instrument tilts (each reading has its own).
The shift is computed during calibration, when the instrument is tilted toward
north (`input.calib.time_ranges_azimuth`), and the resulting scalar is then
always applied during processing: it is subtracted from $\psi$. For the
rationale and the circular-mean averaging rule, see
[Calibration Wiki §8.2](calibration_wiki.md#82-heading-offset-why-not-folded-in).

## Tilt and azimuth

The tilt angle and its direction are determined by the [Marsh, 1982] formulas:

$$
\theta = \mathrm{atan2}\left(\sqrt{G_x^2 + G_y^2}, G_z\right), \tag{1}
$$

$$
\psi = \mathrm{atan2}\left((G_x B_y - G_y B_x)\,|\mathbf{G}|,
B_z (G_x^2 + G_y^2) - G_z (G_x B_x + G_y B_y)\right), \tag{2}
$$

where `atan2(y, x)` is the standard function returning the argument (*arg*) of
the complex number $x + iy$.

## Speed magnitude

$$
V = f_v(\theta), \tag{3}
$$

where $f_{v}$ is an experimentally determined polynomial or another increasing
function starting at the origin — the
[`kVabs`](../reference/config_reference.md#inputcoefs--calibration-coefficients)
coefficients.

### Load-function principle

The equilibrium condition of the inclinometer is the equality of the force
projections onto the axis passing along the tangent to the circle of its
rotation:

- $`F_{b} \sin\Theta`$ — projection of the vertically directed buoyancy force
  $`F_{b}`$;
- $`F_{d,\Theta}`$ — projection of the horizontally directed hydrodynamic
  drag:

$$
F_b \sin\Theta = F_{d,\Theta}.
$$

According to the load-function principle,

$$
F_{d,\Theta} \sim V^2 \cdot f(\Theta),
$$

where $f(\Theta)$ is the load function, determined experimentally. For cables
it is often given as a trigonometric series expansion (current calculation
version [`input.coefs.calc_version`](../reference/config_reference.md) =
`trigonometric(incl)`) [Knutson, 1987]:

$$
f(\Theta) = a_0 + \sum_{n=1}^{\infty} \left( a_n \cos n\Theta + b_n \sin n\Theta \right),
$$

where $`a_{0}`$, $`a_{n}`$, $`b_{n}`$ are constant coefficients. Usually the first
five terms are enough, i.e. $n = 2$.

Thus the speed magnitude is sought in the form

$$
V = \sqrt{\dfrac{\sin\Theta}{a_0 + \sum_{n=1}^{\infty}
\left( a_n \cos n\Theta + b_n \sin n\Theta \right)}}.
$$

Using the load-function principle describes the experimental dependence more
accurately than the so-called independence (cross-flow) principle
[Jorge Silva-Leon, 2018].

### Beyond the maximum tilt of the calibration data

The maximum tilt of the calibration data, $`\Theta_{max}`$, is bounded by the
physical characteristics of the inclinometer and carries a large measurement
error: at high speeds the tilt change with increasing speed drops sharply (so,
to compensate, the $V(\Theta)$ curve would have to rise steeply), while the
behavior of the instrument becomes unstable (so, to reduce erroneous values,
the curve must not rise steeply).

Therefore, for speeds near and beyond $`\Theta_{max}`$ a compromise is used: a
linear dependence tangent to the curve at the point $`\Theta_{last}`$. This
value is stored as an additional coefficient of the current calculation
version —
[`kVabs_switch_to_linear`](../reference/config_reference.md#inputcoefs--calibration-coefficients).
It is chosen as the $\Theta$ value just before $`\Theta_{max}`$, close to the
mean tilt observed over several experiments with instruments of the same type.
For $\Theta > 90^\circ$ the instrument's sensitivity to speed changes tends to
zero, yet may remain somewhat higher because of fluctuations of the medium,
the instrument's instability in it, and its noise.

## References

- [Jorge Silva Leon, A.; Cioncolini, F.; Filippone, A. Determination of the normal fluid load on inclined cylinders from optical measurements of the reconfiguration of flexible filaments in flow. Journal of Fluids and Structures, 2018, vol. 76, pp. 488–505.](https://www.sciencedirect.com/science/article/pii/S088997461730419X)
- [Marsh, J. L. Hand-held calculator assists in directional drilling control. Part 2: Petroleum Engineer International, September 1982, pp. 82–88](https://www.osti.gov/biblio/6117940)
- [Knutson, R. K. BASIC Desk-Top Computer Program for the Three-Dimensional Static Configuration of an Extensible Flexible Cable in a Uniform Stream. David W. Taylor Naval Ship Research and Development Center, Report No. DTNSRDC-87/029, August 1987](https://www.comm-tec.com/library/technical_papers/USGS/TWRI_3-A21.pdf)