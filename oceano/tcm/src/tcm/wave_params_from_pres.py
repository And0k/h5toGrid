"""
Wave parameter estimation from raw pressure records.

PHYSICAL BACKGROUND
-------------------
Significant wave height can be estimated in two equivalent ways:

1) Spectral-domain (standard, recommended):
   Hs ≡ Hm0 = 4 * sqrt(m0),
   where m0 is the zeroth spectral moment of the surface elevation spectrum.

2) Time-domain (historical, control):
   H1/3 = mean height of the highest one-third of individual waves
   extracted via zero-upcrossing analysis.

Under the assumptions of a linear, stationary, Gaussian sea state,
Hs ≈ H1/3 within ~10..15% (Rayleigh wave height statistics).

Wave periods:
- Tm-1 (energy period): m(-1)/m0
  Relevant for wave energy transport and engineering applications.
- Spectral moments are sensitive to the frequency integration range;
  therefore, frequency cutoffs must be explicitly documented.

STANDARDS AND GUIDELINES
-----------------------
- ISO 19901-1: Metocean design parameters
- ISO 21650: Wave spectral analysis
- WMO Guide to Wave Measurements
- ITTC Recommended Procedures (Wave Data Processing)
- CF Conventions (NetCDF metadata)

This implementation follows best practice:
- Welch or Multitaper PSD estimation
- Explicit detrending and windowing
- Quality control via logging
- CF-compliant NetCDF output

INPUT
-----
- Pressure time series (pandas.Series with datetime index)
- Sampling interval ~0.2 s
- Optional wind or reanalysis data may be used externally for sea/swell QC

OUTPUT
------
- NetCDF file with CF-compliant variables:
  * sea_surface_wave_significant_height
  * sea_surface_wave_significant_height_time_domain
  * sea_surface_wave_energy_period


"""

# from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import welch, detrend
from utils.logging_config import setup_logging

logger = setup_logging(__name__, log_file_dir="logs")

g_Kaliningrad = 9.80665  # Baltic Sea

def frequency_band_filter(f=None, *args, fmin: float = 0.04, fmax: float = 0.5):
    """
    filter frequency and spectrum arrays within specified band.
    """
    band = (f >= fmin) & (f <= fmax)
    f = f[band]
    return (f, *[s[band] for s in args]) if args else f


def pressure_to_elevation_linear_wave_theory_simple(
    p: np.ndarray,
    rho: float = 1025.0,
    g: float = g_Kaliningrad,
) -> np.ndarray:
    """
    Convert pressure fluctuations to surface elevation (see WMO Guide to Wave Measurements)

    This simplified formulation assumes sensor sufficiently close to the surface => no frequency-dependent
    attenuation correction (linear wave theory). For deeper sensors, frequency-dependent transfer functions
    must be applied, so this function must not be used!
    """
    return p / (rho * g)


def pressure_to_elevation_linear_shallow_simple(
    water_pressure_dbar,
    depth,
    rho=1025.0,
    g=g_Kaliningrad,
):
    """
    Convert pressure record (dBar) to surface elevation for shallow coastal water.
    Uses shallow-water approximation suitable for sensors near seabed.
    For more accurate reconstruction, frequency-domain correction should be used.

    Parameters
    ----------
    water_pressure_dbar: array_like
        Pressure relative to the surface [dBar]
    depth: float
        Total water depth [m]
    rho: float
        Water density [kg m-3]
    g: float
        Gravity [m s-2]

    Returns
    -------
    eta: ndarray
        Estimated surface elevation [m]
    water_column: float
        Sensor depth below surface [m]

    Notes
    -----
    Exact linear wave theory:

        p'(z,f) = rho g η cosh(k(z+h)) / cosh(kh)

    The time-domain approximation used here assumes long waves
    (kh << 1) typical for shallow coastal seas:
        a practical approximation is: η ≈ p'/(ρg) * (h/z)

    where z is the sensor depth below surface

    Validity range
    --------------
    depth: 4..20 m
    sensor height: 0.5..3 m above seabed
    wave periods: 4..10 s

    References
    WMO Guide to Wave Measurements
    ISO 21650: Coastal wave measurements
    """

    p = np.asarray(water_pressure_dbar)

    # mean pressure (convert dBar → Pa) → water column above sensor [m]
    p_mean = np.mean(p)
    water_column = (p_mean * 1e4) / (rho * g)
    if water_column < 0:
        raise ValueError("Computed that sensor is above surface")

    # fluctuations
    p_fluct = p - p_mean

    # hydrostatic surface elevationD
    eta = (p_fluct * 1e4) / (rho * g)

    # shallow-water pressure attenuation correction
    eta *= depth / water_column

    return eta, water_column


def pressure_response_correction(
    freq: np.ndarray,
    water_depth: float,
    sensor_height_above_bed: float,
    g: float = g_Kaliningrad,
) -> np.ndarray:
    """
    Frequency-dependent pressure response correction.

    Parameters
    ----------
    freq : ndarray
        Wave frequencies [Hz].
    water_depth : float
        Total water depth h [m].
    sensor_height_above_bed : float
        Vertical distance between sensor and seabed [m].
        0 ≤ sensor_height_above_bed ≤ water_depth
    g : float, optional
        Gravitational acceleration [m s⁻²].

    Returns
    -------
    ndarray
        Pressure attenuation factor A_p(f).

    References
    ----------
    ISO 21650: Waves and currents measurements
    WMO Guide to Wave Measurements
    Holthuijsen (2007) "Waves in Oceanic and Coastal Waters"

    Linear wave theory:
    p(z,freq) ∝ cosh(k*(z+h)) / cosh(k*h)

    Here:
    - z = -depth + sensor_height_above_bed
    - k obtained from linear dispersion relation

    Reference:
    - WMO Guide to Wave Measurements
    - ISO 21650
    Pressure attenuation factor for a seabed pressure sensor.

    Computes the linear-theory factor describing how the wave-induced
    pressure amplitude at the sensor differs from the pressure amplitude
    at the mean water surface.

    The result is the *pressure attenuation factor*:

        A_p(freq) = cosh(k(z + h)) / cosh(kh)

    where:

        freq : wave frequency [Hz]
        h : water depth [m]
        z : vertical coordinate of the sensor [m]
        k : wavenumber [rad m⁻¹]

    Coordinate system
    -----------------
    The vertical coordinate follows the common convention used in
    ocean wave theory:

        z = 0        → mean water level
        z = -h       → seabed

    If the sensor is mounted above the seabed:

        z = -h + sensor_height_above_bed

    Example:

        water_depth = 12 m
        sensor_height_above_bed = 2 m

        z = -12 + 2 = -10 m

    Physical meaning
    ----------------
    Pressure fluctuations decrease with depth. The attenuation factor
    describes that decay:

        A_p(f) = p'(z,f) / (ρ g η(f))

    where

        p'(z,f) : dynamic pressure fluctuation at sensor depth
        η(f)    : surface elevation spectrum

    To reconstruct surface elevation from pressure measurements,
    the inverse factor is used:

        surface_factor = 1 / A_p

    Validity
    --------
    Linear wave theory, valid when

        wave_height << wavelength
        wave_height << water_depth

    This formulation is standard for processing pressure-based
    wave measurements.
    """

    freq = np.asarray(freq)

    omega = 2 * np.pi * freq

    # initial deep-water estimate
    k = omega**2 / g

    # refine using dispersion relation with numerical safeguards
    for _ in range(3):
        # Avoid overflow in tanh by limiting the argument
        kh = np.minimum(k * water_depth, 700)  # tanh(700) ≈ 1.0
        tanh_kh = np.tanh(kh)

        # Avoid division by zero or very small values
        tanh_kh = np.maximum(tanh_kh, 1e-10)

        k = omega**2 / (g * tanh_kh)

    # sensor vertical coordinate
    # z = -water_depth + sensor_height_above_bed

    # Calculate transfer function
    cosh_kh = np.cosh(k * water_depth)
    cosh_kz = np.cosh(k * sensor_height_above_bed)

    # Avoid division by zero or very small values
    cosh_kz = np.maximum(cosh_kz, 1e-10)

    transfer = cosh_kh / cosh_kz

    return transfer


def calc_psd_welch(
    pressure: np.ndarray,
    fs: float,
    input_units: str = "dBar",
):
    """
    Calculate power spectral density using Welch method.

    Parameters
    ----------
    pressure : array_like
        Pressure time series
    fs : float
        Sampling frequency [Hz]
    input_units : str
        Input pressure units: 'dBar' or 'Pa'

    Returns
    -------
    f : ndarray
        Frequency array [Hz]
    Sp : ndarray
        Power spectral density [Pa²/Hz]
    """
    # Detrend pressure first to remove mean value
    pressure_detrended = detrend(pressure, type="constant")

    # Convert detrended pressure to Pa if needed
    if input_units == "dBar":
        pressure_pa = pressure_detrended * 1e4  # Convert dBar to Pa
    elif input_units == "Pa":
        pressure_pa = pressure_detrended
    else:
        raise ValueError(f"Unsupported input_units: {input_units}. Use 'dBar' or 'Pa'")

    # Calculate PSD from detrended pressure in Pa
    f, Sp = welch(
        pressure_pa,
        fs=fs,
        window="hann",
        nperseg=min(len(pressure_pa), int(fs * 1024)),
        scaling="density",
    )
    return f, Sp


def compute_surface_elevation_spectrum(
    psd,
    transfer,
    rho: float = 1025.0,
    g: float = g_Kaliningrad,
):
    """
    Compute surface elevation spectrum,
    applying frequency-dependent response correction.

    Parameters
    ----------
    psd : array_like
        Pressure power spectral density [Pa²/Hz]
    transfer : array_like
        Frequency-dependent transfer function
    rho : float
        Water density [kg m-3]
    g : float
        Gravity [m s-2]

    Returns
    -------
    Se : ndarray
        Surface elevation power spectral density [m²/Hz]

    Notes
    -----
    From linear wave theory: p'(z,f) = rho g η cosh(k(z+h)) / cosh(kh)
    Therefore: η = p' / (rho g) * cosh(kh) / cosh(k(z+h))
    In spectral domain: S_η(f) = S_p(f) * |K(f)|² / (rho g)²
    where K(f) = cosh(kh) / cosh(k(z+h)) is the transfer function
    """
    # Apply frequency-dependent response correction
    # From linear wave theory: S_η(f) = S_p(f) * |K(f)|² / (rho g)²
    # where K(f) is the transfer function for recovering surface elevation from pressure
    Se = psd * (transfer**2) / (rho * g)**2
    return Se



## Spectral analysis

def spectral_moments(
    freq: np.ndarray,
    psd: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute m0 and m(-1) spectral moments

    Note: frequency range truncation *changes the physical meaning* of the parameters
    and must be documented (ISO 19901-1).
    """

    df = np.mean(np.diff(freq))
    m0 = np.sum(psd * df)
    m_minus1 = np.sum(psd / freq * df)  # m(-1)

    return m0, m_minus1


## Time-domain zero-upcrossing analysis

def h13_from_time_domain(eta: np.ndarray, fs: float) -> float:
    """
    Compute H1/3 from zero-upcrossing analysis.

    This method is sensitive to noise and nonstationarity and is
    therefore used primarily for quality control.
    """
    eta = eta - np.mean(eta)

    crossings = np.where((eta[:-1] <= 0) & (eta[1:] > 0))[0]
    if len(crossings) < 2:
        return np.nan

    heights = []
    for i in range(len(crossings) - 1):
        seg = eta[crossings[i] : crossings[i + 1]]
        heights.append(seg.max() - seg.min())

    heights = np.sort(heights)
    n = len(heights)

    return np.mean(heights[int(2 * n / 3) :])


def wavespectra_metrics(
    f: np.ndarray,
    psd: np.ndarray,
) -> dict:
    """
    Compute wave parameters using wavespectra
    from an externally validated spectrum. Output includes:
    "Hm0": float(spec.hs()),
    "Tm_1": float(spec.tm(-1)),
    "Tm01": float(spec.tm01()),
    "Tm02": float(spec.tm02()),
    "Tp": float(spec.tp()),

    """
    import wavespectra as ws

    ds = xr.Dataset(
        {
            "efth": (("freq",), psd),
        },
        coords={
            "freq": f,
        },
        attrs={"Conventions": "CF-1.10"},
    )

    return ws.SpecDataset(ds)


# ---------------------------------------------------------------------
# Main processing routine
# ---------------------------------------------------------------------
def process_pressure_series(
    pressure: pd.Series,
    depth: float,
    output_nc: Path,
    logfile: Path,
    fmin: float = 0.04,
    fmax: float = 0.5
) -> None:
    """
    End-to-end processing of raw pressure data into wave parameters.

    Steps:
    ШАГ 0 — удаление атмосферного давления
    1) Sampling rate estimation
    2) Pressure → elevation conversion
    3) Spectral moment calculation
    4) Hs, Tm-1 estimation
    5) Time-domain H1/3 for QC
    6) CF-compliant NetCDF output

    Parameters
    ----------
    pressure : array_like
        Water column pressure (pressure relative to the surface) [dBar]
    depth : float
        Total water depth [m]
    """

    logger.info("Starting wave processing")
    dt = pressure.index.to_series().diff().dt.total_seconds().median()
    fs = 1 / dt
    logger.info("Estimated sampling frequency: %.3f Hz", fs)

    # Surface elevation [m] by not precise method if z ≪ 0 (only for QA, Hs будет завышен)

    eta = pressure_to_elevation_linear_shallow_simple(pressure.values, depth)
    logger.info("Converted pressure to surface elevation")

    H13 = h13_from_time_domain(eta, fs)
    logger.info("Time-domain H1/3=%.2f m", H13)

    # Get PSD (todo: all following in windows)
    freq, Sp = calc_psd_welch(pressure=pressure.values, fs=fs, input_units="dBar")

    # Apply frequency band limits:
    # - low cut removes infragravity motions and seiches
    # - high cut removes sensor noise
    f, Sp = frequency_band_filter(freq, Sp, fmin=fmin, fmax=fmax)

    # ---------------------------------------------------------------
    # Apply frequency-dependent pressure response correction
    # ---------------------------------------------------------------
    transfer = pressure_response_correction(
        f,
        water_depth=depth,
        sensor_height_above_bed=2.0,
    )
    S_ws = compute_surface_elevation_spectrum(Sp, transfer)

    # Calculate spectral moments from corrected surface elevation spectrum
    m0, m_minus1 = spectral_moments(f, S_ws)
    Hs = 4.0 * np.sqrt(m0)
    Tm_minus1 = m_minus1 / m0
    logger.info("Spectral results: Hs=%.2f m, Tm-1=%.2f s", Hs, Tm_minus1)

    spec = wavespectra_metrics(f, S_ws)

    logger.info(
        "wavespectra results: "
        "{Hm0=:.2f} m, {Tm_1=:.2f} s,  {Tm01=:.2f} s, {Tm02=:.2f} s, {Tp=:.2f} s".format_map({
            "Hm0": float(spec.hs()),
            "Tm_1": float(spec.tm(-1)),
            "Tm01": float(spec.tm01()),
            "Tm02": float(spec.tm02()),
            "Tp": float(spec.tp()),
        })
    )

    # -----------------------------------------------------------------
    # NetCDF / CF output
    # -----------------------------------------------------------------
    standard_name="sea_surface_wave_significant_height"
    ds = xr.Dataset(
        data_vars={
            standard_name: (
                (),
                Hs,
                dict(
                    units="m",
                    standard_name=standard_name,
                    method="spectral_moment_m0",
                ),
            ),
            f"{standard_name}_time_domain": (
                (),
                H13,
                dict(
                    units="m",
                    standard_name=standard_name,
                    method="zero_upcrossing",
                ),
            ),
            "sea_surface_wave_energy_period": (
                (),
                Tm_minus1,
                dict(
                    units="s",
                    standard_name="sea_surface_wave_energy_period",
                ),
            ),
            # f"{standard_name}_wavespectra": (
            #     (),
            #     ws_metrics["Hm0"],
            #     dict(
            #         units="m",
            #         standard_name=standard_name,
            #         method="wavespectra",
            #         comment="Computed from validated spectrum",
            #     ),
            # ),
            # "sea_surface_wave_mean_period_tm01_wavespectra": (
            #     (),
            #     ws_metrics["Tm01"],
            #     dict(
            #         units="s",
            #         standard_name="sea_surface_wave_mean_period",
            #     ),
            # ),
        },
        attrs=dict(
            Conventions="CF-1.10",
            title="Wave parameters from pressure sensor",
            spectral_low_cutoff_frequency=fmin,
            spectral_high_cutoff_frequency=fmax,
            references=("ISO 19901-1; ISO 21650; WMO Guide to Wave Measurements; ITTC"),
        ),
    )

    ds["hs_wavespectra"] = spec.hs()
    ds["tm01_wavespectra"] = spec.tm01()
    ds["tm02_wavespectra"] = spec.tm02()
    ds["tm_1_wavespectra"] = spec.tm(-1)

    ds.to_netcdf(output_nc)
    logger.info("Saved results to %s", output_nc)
    logger.info("Processing completed successfully")


# ---------------------------------------------------------------------
# Example usage (to be called externally)
# ---------------------------------------------------------------------
# pressure_series = pd.Series(...)
# process_pressure_series(
#     pressure_series,
#     depth=15.0,
#     output_nc=Path("waves.nc"),
#     logfile=Path("waves.log"),
# )
