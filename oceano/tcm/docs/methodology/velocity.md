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
