from numpy import *
#from gsw import O2sol_SP_pt  # O2sol

def do_calc_sbe(V, t, P, S, tim,
                slope= 1.0, Voff = 0, tau20 = 0.0, d_t = -4.64803e-002, d_p = 1.92634e-004, CBA= (0, 0, 0), e= 0):
    """sbe43OxygenTransform
    implementation of the oxygen concentration formula, specified in Seabird Application Note 64 http://www.seabird.com/application_notes/AN64.htm
    measured parameters
    :param V: raw oxygen data, Volts
    :param t:
    :param P:
    :param S:
    :param tim: time, s
    calibration coefficients:
    :param slope: Oxygen slope
    :param Voff: Sensor output offset voltage
    :param tau20: Sensor time constant at 20 deg C and 1 Atm
    :param d_t:
    :param d_p: Compensation coefficient for pressure effect on time constant
    :param CBA: Compensation coefficients for temperature effect on membrane permeability
    :param e: Compensation coefficient for pressure effect on membrane permeability (Atkinson et al, 1996)
    :return: data, array of oxygen concentration values, umol/l
    """

    # calculated values

    data = oxygen_concentration(V, t, P, S, tim, slope, Voff, tau20, d_t, d_p, CBA, e)
    data = hysteresis_on_pres_cor_sbe(data, tim, P, h1 = -0.033, h2 = 5000, h3 = 1450)

    # convert from ml/l to mg/l
    return data * 1.42903

    # # convert from ml/l to umol/l
    # #
    # # Conversion factors from Saunders (1986) :
    # # https://darchive.mblwhoilibrary.org/bitstream/handle/1912/68/WHOI-89-23.pdf?sequence=3
    # # 1ml/l = 43.57 umol/kg (with dens = 1.025 kg/l)
    # # 1ml/l = 44.660 umol/l
    # return data * 44.660
    

def oxygen_concentration(V, t, P, S, tim, slope, Voff, tau20, d_t, d_p, CBA, e):
    """
    Calculates oxygen concentration. Implementation of the oxygen concentration equation
    specified in Seabird Application Note 64.
    :param V: Oxygen voltage data
    :param t: Temperature, degrees celsius
    :param S: Salinity, PSU
    :param P: Pressure, dBars
    :param tim: time, s
    calibration coefficients:
    :param slope: Oxygen slope
    :param Voff: Sensor output offset voltage
    :param tau20: Sensor time constant at 20 deg C and 1 Atm
    :param d_t:
    :param d_p: Compensation coefficient for pressure effect on time constant
    :param CBA: Compensation coefficients for temperature effect on membrane permeability
    :param e: Compensation coefficient for pressure effect on membrane permeability (Atkinson et al, 1996)
    :return oxygen: Oxygen concentration, mL/L
    """
    oxsol = oxygen_solubility(t, S)
    tauTP = tau20 * exp(d_t * (t - 20.0) + d_p * P)
    # Estimate of sensor output change over time
    dVdt = ediff1d(V, to_begin=0.0) / ediff1d(tim, to_begin=0.0)
    K = t + 273.15
    return slope * (V + Voff + tauTP * dVdt) * polyval(append(CBA, 1.0), t) * exp(e * P / K) * oxsol


def ox_saturation_idr(o2_cnt, t, P, o2_cal, t_cal, c1, c_t= - 0.029, c_p= 0.000115):
    """
    Calculation of % saturation

    :param o2_cnt: Oxygen sensor reading in counts
    :param t: Temperature sensor reading in °C
    :param P: Pressure reading in dbar
    :param o2_cal: Oxygen sensor reading in counts during calibration
    :param t_cal: Temperature sensor reading in °C during calibration
    :param c1: Stirring effect and barometric pressure compensation
    :param c_t, c_p: proprietary coefficients are required for the calculation of % saturation to compensate the
IDRONAUT membrane permeability to oxygen due to the temperature and pressure variation respectively
    :return: Saturation, %
    """

    slope= o2_cal / exp(t_cal * c_t) / 100
    return o2_cnt * slope * c1 * exp(c_t * t + c_p * P)


def oxygen_solubility(t, S):
    """
    Alternative: O2sol or O2sol_SP_pt
    Oxygen solubility after Garcia and Gordon (1992)
    This function is an implementation of the Computation of Oxygen Solubility equation, as specified in Seabird Application Note 64, Appendix A.
    :param t: temperature, degrees celsius
    :param S: Salinity, PSU
    :return oxsol: Oxygen solubility     
    """
    a = [3.88767, 0.256847, 4.94457, 4.0501, 3.22014, 2.00907]
    b = [-0.00817083, -0.010341, -0.00737614, -0.00624523]
    c0 = -0.000000488682

    Ts = log((298.15 - t) / (273.15 + t))
    oxsol = exp(polyval(a, Ts) + S * polyval(b, Ts) + c0 * S**2)
    return oxsol
    

def hysteresis_on_pres_cor_sbe(o_in, tim, p, h1, h2, h3):
    """ Correction of Hysteresis induced by High Pressure Effects on Teflon Membrane. Recommended for profiles with depths exceeding 1000 m.
    Algorithm:
    d = (exponential(p[i] / h2) – 1)
    c = exponential (-1 * (tim[i] – tim[i-1]) / h3)
    o[i] = ((o_in[i] + (o[i-1] * c * d)) – (o_in[i-1] * c)) / d
    where
    • i = 1..len(o_in): indexing variable (must be a continuous time series to work; can be performed on bin averaged data)
    • p[i] = pressure (dBar) at index point i
    • tim[i] = time (s) from start of index point i
    • o_in[i] = dissolved DO at index point i
    • d and c are temporary variables used to simplify expression in processing loop.
    • h1 = amplitude of hysteresis correction function. Default = -0.033, range = -0.02 to -0.05 (varies from sensor to
    sensor)
    • h2 = function constant or curvature function for hysteresis. Default = 5000
    • h3 = time constant for hysteresis (s). Default = 1450, range = 1200 to 2000 (varies from sensor to sensor)
    • o[i] = hysteresis-corrected DO at index point i

    Notes:
    • Scan 0 – You cannot calculate o[0] because the algorithm requires information about the previous scan, so skip scan 0 when correcting for hysteresis
    • Scan 1 - When calculating o[1], make the following assumption about values from scan 0: o[0] = o_in[0]

    Reference
    ---------
    SBE 43 Dissolved Oxygen (DO) Sensor – Hysteresis Corrections. Application Note 64-3
    """

    # temporary variables
    c = exp(-ediff1d(tim, to_begin=0.0) / h3)
    d = 1 + h1 * expm1(p / h2)

    # more temporary variables: taking out of a loop all operations we can:
    cd = c * d
    dc_o_in = empty_like(o_in); dc_o_in[1:] = o_in[1:] - (o_in * c)[:-1]

    # calculating o
    o = empty_like(o_in)
    o[0] = o_in[0]
    for i in range(1, len(o_in)):
        o[i] = (o[i - 1] * cd[i] + dc_o_in[i]) / d[i]

    return o


