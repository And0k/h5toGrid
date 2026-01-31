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
Hs ≈ H1/3 within ~10–15% (Rayleigh wave height statistics).

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
- Welch PSD estimation
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

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import welch, detrend


def setup_logger(logfile: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure file and console logging for QC and debugging."""
    logger = logging.getLogger("wave_processing")
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def pressure_to_elevation_linear_wave_theory(
    p: np.ndarray,
    rho: float = 1025.0,
    g: float = 9.81,
) -> np.ndarray:
    """
    Convert pressure fluctuations to surface elevation (see WMO Guide to Wave Measurements)

    This simplified formulation assumes sensor sufficiently close to the surface => no frequency-dependent
    attenuation correction (linear wave theory). For deeper sensors, frequency-dependent transfer functions
    must be applied, so this function must not be used!
    """
    return p / (rho * g)


def pressure_response_correction(
    f: np.ndarray,
    depth: float,
    sensor_height_above_bed: float,
    g: float = 9.81,
) -> np.ndarray:
    """
    Frequency-dependent pressure response correction.

    Linear wave theory:
    p(z,f) ∝ cosh(k*(z+h)) / cosh(k*h)

    Here:
    - z = -depth + sensor_height_above_bed
    - k obtained from linear dispersion relation

    Reference:
    - WMO Guide to Wave Measurements
    - ISO 21650
    """
    omega = 2 * np.pi * f
    k = omega**2 / g  # deep-water initial guess
    for _ in range(3):  # fixed-point refinement (sufficient for 3–15 m)
        k = omega**2 / (g * np.tanh(k * depth))

    z = -depth + sensor_height_above_bed
    transfer = np.cosh(k * (z + depth)) / np.cosh(k * depth)

    return transfer


def compute_surface_elevation_spectrum(
    pressure: np.ndarray,
    fs: float,
    depth: float,
    sensor_height_above_bed: float,
    fmin: float,
    fmax: float,
    rho: float = 1025.0,
    g: float = 9.81,
):
    """
    Compute surface elevation spectrum from raw pressure, using Welch PSD,
    applying frequency-dependent response correction
    """
    pressure = detrend(pressure, type="constant")

    f, Sp = welch(
        pressure,
        fs=fs,
        window="hann",
        nperseg=min(len(pressure), int(fs * 1024)),
        scaling="density",
    )

    band = (f >= fmin) & (f <= fmax)
    f = f[band]
    Sp = Sp[band]

    transfer = pressure_response_correction(f, depth, sensor_height_above_bed)

    Se = Sp / (transfer**2) / (rho * g) ** 2
    return f, Se



## Spectral analysis

def spectral_moments(
    f: np.ndarray,
    S: np.ndarray,
    fmin: float = 0.04,
    fmax: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute m0 and m(-1) spectral moments

    Frequency band limits:
    - low cut removes infragravity motions and seiches
    - high cut removes sensor noise

    Such truncation *changes the physical meaning* of the parameters
    and must be documented (ISO 19901-1).
    """
    band = (f >= fmin) & (f <= fmax)
    f = f[band]
    S = S[band]

    df = np.mean(np.diff(f))

    m0 = np.sum(S * df)
    m_1 = np.sum(S / f * df)  # m(-1)

    return m0, m_1


## Time-domain zero-upcrossing analysis

def time_domain_h13(eta: np.ndarray, fs: float) -> float:
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
    S: np.ndarray,
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
            "efth": (("freq",), S),
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
    fmax: float = 0.5,
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
    """
    logger = setup_logger(logfile)

    logger.info("Starting wave processing")
    dt = pressure.index.to_series().diff().dt.total_seconds().median()
    fs = 1.0 / dt
    logger.info("Estimated sampling frequency: %.3f Hz", fs)
    H13 = time_domain_h13(eta, fs)
    logger.info("Time-domain H1/3=%.2f m", H13)

    # Get PSD in windows


    m0, m_1 = spectral_moments(eta, fs, fmin, fmax)
    Hs = 4.0 * np.sqrt(m0)
    Tm_1 = m_1 / m0

    logger.info("Spectral results: Hs=%.2f m, Tm-1=%.2f s", Hs, Tm_1)



    # ---------------------------------------------------------------
    # Wavespectra-based estimation (secondary, comparison layer)
    # ---------------------------------------------------------------
    f_ws, S_ws = compute_surface_elevation_spectrum(
        pressure.values,
        fs,
        depth=depth,
        sensor_height_above_bed=2.0,
        fmin=fmin,
        fmax=fmax,
    )

    spec = wavespectra_metrics(f_ws, S_ws)

    logger.info(
        "wavespectra results: {Hm0=:.2f} m, {Tm_1=:.2f} s,  {Tm01=:.2f} s, {Tm02=:.2f} s, {Tp=:.2f} s".format_map({
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
                Tm_1,
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
