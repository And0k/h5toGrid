import numpy as np
import pytest

from inclinometer.incl_h5clc import *


# ##############################################################################

@pytest.fixture
def incl_rad():
    return np.arange(0, 1.01, 0.1, np.float64)

@pytest.fixture
def coefs():
    return (-49.9880869, 56.97823445, 57.89848781, 8.84274956, -21.66993163, 82.414113)


def rep_if_bad_nj(checkit, replacement):
    return checkit if (any(checkit) and any(np.isfinite(checkit))) else replacement

def f_linear_k_nj(x0, g, g_coefs):
    return min(rep_if_bad_nj(np.diff(g([x0 - 0.01, x0], g_coefs)) / 0.01, 10), 10)

def f_linear_end_nj(g, x, x0, g_coefs):
    g0 = g(x0, g_coefs)
    return np.where(x < x0, g(x, g_coefs), g0 + (x - x0) * f_linear_k_nj(x0, g, g_coefs))

def trigonometric_series_sum_nj(r, coefs):
    return coefs[0] + np.nansum([
        (a * np.cos(nr) + b * np.sin(nr)) for (a, b, nr) in zip(
            coefs[1::2], coefs[2::2], np.arange(1, len(coefs) / 2)[:, None] * r)],
        axis=0)

def v_trig_nj(r, coefs):
    squared = np.sin(r) / trigonometric_series_sum_nj(r, coefs)
    # with np.errstate(invalid='ignore'):  # removes warning of comparison with NaN
    return np.sqrt(squared, where=squared > 0, out=np.zeros_like(squared))


def test_jit_v_trig(incl_rad: np.ndarray, coefs: Sequence):
    max_incl_of_fit = np.radians(coefs[-1])
    coefs = coefs[:-1]
    assert np.allclose(v_trig(incl_rad, coefs), v_trig_nj(incl_rad, coefs))


def test_jit_trigonometric_series_sum(incl_rad: np.ndarray, coefs: Sequence):
    coefs = coefs[:-1]
    assert np.allclose(trigonometric_series_sum(incl_rad, coefs),
                       trigonometric_series_sum_nj(incl_rad, coefs))


# @pytest.mark.parametrize('incl_rad, coefs',
#                         [(np.arange(0, 1.01, 0.1, np.float64),
#                          (-49.9880869, 56.97823445, 57.89848781, 8.84274956, -21.66993163, 82.414113))])
def test_jit_v_abs_from_incl(incl_rad: np.ndarray, coefs: Sequence, calc_version='trigonometric(incl)', max_incl_of_fit_deg=None) -> np.ndarray:
    """
    Vabs = np.polyval(coefs, Gxyz)

    :param incl_rad:
    :param coefs: coefficients.
    Note: for 'trigonometric(incl)' option if not max_incl_of_fit_deg then it is in last coefs element
    :param calc_version: 'polynom(force)' if this str or len(coefs)<=4 else if 'trigonometric(incl)' uses trigonometric_series_sum()
    :param max_incl_of_fit_deg:
    :return:
    """

    def v_abs_from_incl_nj(incl_rad, coefs, calc_version='trigonometric(incl)', max_incl_of_fit_deg=None):
        """
        Vabs = np.polyval(coefs, Gxyz)
        :param incl_rad:
        :param coefs:
        :param calc_version:
        :param max_incl_of_fit_deg:
        :return:
        """
        if calc_version == 'polynom(force)':
            l.warning('Old coefs method polynom(force)')
            force = fIncl_rad2force(incl_rad)
            return fVabs_from_force(force, coefs)

        elif calc_version == 'trigonometric(incl)':
            if max_incl_of_fit_deg:
                max_incl_of_fit = np.radians(max_incl_of_fit_deg)
            else:
                max_incl_of_fit = np.radians(coefs[-1])
                coefs = coefs[:-1]

            with np.errstate(invalid='ignore'):  # removes warning of comparison with NaN
                return f_linear_end_nj(g=v_trig_nj, x=incl_rad, x0=max_incl_of_fit, g_coefs=coefs)

        else:
            raise NotImplementedError(f'Bad calc method {calc_version}', )


    assert np.allclose(v_abs_from_incl(incl_rad, coefs, calc_version='trigonometric(incl)'),
                       v_abs_from_incl_nj(incl_rad, coefs, calc_version='trigonometric(incl)'))
