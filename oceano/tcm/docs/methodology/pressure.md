# Pressure computation from the `P_t` polynomial

How `p`-type probes convert raw pressure counts into physical pressure — the
temperature-compensated 2-D polynomial
[`input.coefs.P_t`](../reference/config_reference.md#inputcoefs--calibration-coefficients).

The polynomial is evaluated as `numpy.polynomial.polynomial.polyval2d(u, t, P_t)`:

$$
P(u, t) = c_{00} + c_{10}\,u + c_{01}\,t + c_{20}\,u^2 + c_{11}\,u\,t + c_{02}\,t^2
$$

where

- $u$ — raw pressure counts (`P` / `P_counts` channel);
- $t$ — temperature (`Temp` channel);
- each `P_t[i][j]` multiplies $`u^{i}\, t^{j}`$; only the six coefficients of
  total degree $\le 2$ are used — the remaining cells of the 3×3 matrix are
  zero.

The computed pressure is as calibrated — referenced to standard atmospheric
pressure $`P_{0} = 10.1325`$ dbar.

## Provenance

Coefficients come from factory calibration polynomial strings
(`A + B * u - C * t - D * u**2 + E * u * t + F * t**2`), dated per calibration
run. [coef_poly2array.py](../../scripts/coef_poly2array.py) parses the string,
packs the coefficients into the 3×3 matrix and writes the `/{tbl}/coef/P_t`
group of the coefficient source (HDF5/NC, or YAML export for per-probe files).

To recover the symbolic formula from a stored matrix:
`sympy.expand(polyval2d(Symbol('u'), Symbol('t'), P_t))`.
