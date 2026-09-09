# Velocity computation from accelerometer and magnetometer

## Sensor calibration

Raw ADC readings of the accelerometer \(U_{Gx}, U_{Gy}, U_{Gz}\) and of the
magnetometer \(U_{Bx}, U_{By}, U_{Bz}\) are converted to the projections of
gravity \(G_x, G_y, G_z\) and of the magnetic field \(B_x, B_y, B_z\) onto the
axes of the coordinate frame bound to the sensor. For the accelerometer:

$$\mathbf{G} = \mathbf{A} \cdot (\mathbf{U}_G - \mathbf{U}_{G0}),$$

where

- \(\mathbf{A}\) — 3×3 matrix of scale and rotation coefficients —
  [`Ag`](../reference/config_reference.md#inputcoefs--calibration-coefficients);
- \(\mathbf{U}_{G0}\) — 3×1 ADC offset vector — `Cg`;
- \(\mathbf{G} = [G_x, G_y, G_z]^T\) — gravity acceleration vector, g;
- \(\mathbf{U}_G = [U_{Gx}, U_{Gy}, U_{Gz}]^T\) — accelerometer ADC counts.

An analogous expression is used for the magnetic field vector \(\mathbf{B}\)
measured by the magnetometer, with the offset vector \(\mathbf{U}_{B0}\) —
`Ch`, and the coefficient matrix \(\mathbf{M}\) — `Ah`. The offset vectors and
coefficient matrices are determined for the accelerometer and the magnetometer
of every inclinometer during
[preliminary calibration](../python_developer_guide/calibration.md).

## Zeroing rotation (\(R_z\))

The sensor's own axes are not necessarily aligned with the instrument's
vertical. The zeroing rotation \(R_z\) — a \(3\times3\) orthogonal matrix —
rotates the calibrated sensor frame so that its Z-axis coincides with
gravity. It is computed once, before velocity calculation, from
accelerometer samples recorded while the instrument hung plumb:

- **From a data interval** (`input.calib.time_ranges_zeroing`): the pipeline
  averages the calibrated accelerometer unit vectors over the window and
  builds the rotation aligning the empirical zenith with \([0, 0, 1]^T\) via
  Rodrigues' formula — see
  [Calibration Wiki §8.1](calibration_wiki.md#81-zero-tilt-fold-in).
- **From a single vector** (`input.calib.g0xyz`): a raw accelerometer vector
  measured at known zero tilt is converted to a unit vector and rotated to
  \([0, 0, 1]^T\) the same way; it overrides any \(R_z\) from
  `time_ranges_zeroing`.

\(R_z\) is applied to **both** the accelerometer (\(\mathbf{A}_g\)) and the
magnetometer (\(\mathbf{A}_h\)) calibration matrices, because the two sensors
are rigidly co-mounted and must stay in a consistent frame:

\[
\mathbf{A}_g' = R_z \cdot \mathbf{A}_g, \qquad
\mathbf{A}_h' = R_z \cdot \mathbf{A}_h.
\]

These rotated matrices are then used in the formulas above —
\(\mathbf{G} = \mathbf{A}_g' \cdot (\mathbf{U}_G - \mathbf{U}_{G0})\) and
\(\mathbf{B} = \mathbf{A}_h' \cdot (\mathbf{U}_B - \mathbf{U}_{B0})\) — so that
tilt and azimuth are computed with respect to the instrument's vertical
rather than the sensor's raw axes. The rotation itself is not stored in the
output; it is folded into the effective calibration matrices before the
physical conversion.

## Azimuth shift (\(\psi_{\text{shift}}\))

The tilt azimuth \(\psi\) computed by formula (2) is expressed in the sensor's
own frame. To obtain the geographic direction of the tilt, a constant offset
\(\psi_{\text{shift}}\) (coefficient `azimuth_shift_deg`, in degrees) is
subtracted from the sensor-frame azimuth:

\[
\psi_{\text{geographic}} = \psi_{\text{shift}} - \psi,
\]

where \(\psi\) is the value given by formula (2). The sign convention is chosen
so that the offset compensates the magnetometer sign inversion (the default
`azimuth_shift_deg = 180^\circ`); the same offset is then used to recalculate
the Cartesian velocity components \(u, v\) from the corrected direction.

\(\psi_{\text{shift}}\) is **not** folded into a calibration matrix — unlike
\(R_z\), it cannot be absorbed into a static rotation because the relevant
horizontal plane changes as the instrument tilts (every reading has a
different horizontal plane). It is instead computed once from a known-north
event (`input.calib.time_ranges_azimuth`) and applied at read time as a
scalar, subtracted from the raw \(\psi\) of each sample. See
[Calibration Wiki §8.2](calibration_wiki.md#82-heading-offset-why-not-folded-in)
for the rationale and the circular-mean convention used when averaging the
offset over several samples.

## Tilt and azimuth

The tilt angle and its direction are computed after [Marsh, 1982]:

$$\theta = \operatorname{atan2}\left(\sqrt{G_x^2 + G_y^2},\; G_z\right), \tag{1}$$

$$\psi = \operatorname{atan2}\left((G_x B_y - G_y B_x)\,|\mathbf{G}|,\;
B_z (G_x^2 + G_y^2) - G_z (G_x B_x + G_y B_y)\right), \tag{2}$$

where `atan2(y, x)` is the standard function returning the argument (*arg*) of
the complex number \(x + iy\).

## Speed magnitude

$$V = f_v(\theta), \tag{3}$$

where \(f_v\) — a polynomial or other increasing function through the origin,
determined experimentally — the
[`kVabs`](../reference/config_reference.md#inputcoefs--calibration-coefficients)
coefficients.

### Load-function principle

The equilibrium condition of the inclinometer is the equality of the force
projections onto the axis tangent to the circle of its rotation:

- \(F_b \sin\Theta\) — projection of the vertically directed buoyancy force
  \(F_b\);
- \(F_{d,\Theta}\) — projection of the horizontally directed hydrodynamic
  drag:

$$F_b \sin\Theta = F_{d,\Theta}.$$

Following the load-function principle,

$$F_{d,\Theta} \sim V^2 \cdot f(\Theta),$$

where \(f(\Theta)\) is the load function, determined experimentally. For
cables it is often given as a trigonometric series (used in current version: marked as
[`input.coefs.calc_version`](../reference/config_reference.md) = `trigonometric(incl)`). [Knutson, 1987]:

$$f(\Theta) = a_0 + \sum_{n=1}^{\infty} \left( a_n \cos n\Theta + b_n \sin n\Theta \right),$$

where \(a_0\), \(a_n\), \(b_n\) are constant coefficients. Usually the first
five terms are enough, i.e. \(n = 2\).

Thus the speed magnitude is sought in the form

$$V = \sqrt{\dfrac{\sin\Theta}{a_0 + \sum_{n=1}^{\infty}
\left( a_n \cos n\Theta + b_n \sin n\Theta \right)}}.$$

Using the load-function principle describes the experimental dependence more
accurately than the so-called independence (cross-flow) principle
[Jorge Silva-Leon, 2018].

### Beyond the maximum calibrated tilt

The maximum tilt of the calibration data, \(\Theta_{max}\), is limited by the
physical characteristics of the inclinometer and carries a large measurement
error: at high speeds the tilt change per unit of speed drops sharply (so the
\(V(\Theta)\) curve would have to rise steeply to compensate), while the
behavior of the instrument becomes unstable (so, to reduce erroneous values,
the curve must not rise steeply).

Therefore, for speeds near and beyond \(\Theta_{max}\) a compromise is used:
a linear dependence tangent to the curve at the point \(\Theta_{last}\). This
value is stored as an additional calculation coefficient —
[`kVabs_switch_to_linear`](../reference/config_reference.md#inputcoefs--calibration-coefficients).
It is chosen as the value of \(\Theta\) just before \(\Theta_{max}\), close to
the mean tilt observed over several experiments with instruments of the same
type. Note that for \(\Theta > 90^\circ\) the sensitivity of the instrument to
speed changes tends to zero, although it can remain somewhat higher because of
environmental fluctuations, instrument instability in the flow, and its noise.

## References

- Jorge Silva-Leon A., F. Andrea Cioncolini. Determination of the normal
  fluid load on inclined cylinders from optical measurements of the
  reconfiguration of flexible filaments in flow // J. Fluids Struct. 2018.
  Vol. 76. P. 488–505.
- Marsh J. L. Hand-Held Calculator Assists in Directional Drilling Control // Pet Eng Int U. S. 1982.
  Т. 2. С. 12–14.
- Knutson R. K. BASIC Desk-Top Computer Program for the Three-Dimensional Static Configuration of an Extensible Flexible Cable in a Uniform Stream. : David Taylor Naval Ship R&D Center, 1987.
