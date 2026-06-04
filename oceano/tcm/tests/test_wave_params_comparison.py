#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive testing script to compare wave parameter calculation methods.

This script tests and compares:
1. Spectral domain methods (Hs = 4*sqrt(m0))
2. Time domain methods (H1/3 from zero-upcrossing)
3. Different PSD estimation methods (Welch vs Multitaper)
4. Pressure response correction effects

Tests validate:
- Function parameter correctness
- Calculation logic
- Agreement between methods (within expected ranges)
- Edge cases and error handling
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

import tcm.wave_params_from_pres as wave_params_from_pres
from utils.logging_config import setup_logging

logger = setup_logging(__name__, log_file_dir="logs")


class TestWaveParamsComparison:
    """Test suite for wave parameter calculation methods."""

    @pytest.fixture
    def sample_pressure_data(self):
        """Generate synthetic pressure time series for testing (in dBar)."""
        np.random.seed(42)
        fs = 5.0  # 5 Hz sampling frequency
        duration = 300  # 5 minutes
        t = np.arange(0, duration, 1/fs)

        # Create synthetic wave signal: sum of multiple wave components
        # Pressure in dBar (decibars), typical range for shallow water ~10 dBar
        # Component 1: dominant wave
        p1 = 0.1 * np.sin(2 * np.pi * 0.1 * t)  # ~0.1 dBar fluctuations
        # Component 2: secondary wave
        p2 = 0.05 * np.sin(2 * np.pi * 0.15 * t + np.pi/4)
        # Add noise
        noise = 0.005 * np.random.randn(len(t))

        # Base water column pressure (e.g., 10 m depth ~10 dBar)
        pressure = p1 + p2 + noise + 10.0  # 10 dBar base pressure

        # Create pandas Series with datetime index
        index = pd.date_range("2023-01-01", periods=len(t), freq=f"{1/fs*1000}ms")
        return pd.Series(pressure, index=index, name="Pressure")

    @pytest.fixture
    def sample_frequency_array(self):
        """Generate frequency array for testing."""
        return np.linspace(0.01, 0.5, 100)

    @pytest.fixture
    def sample_psd(self, sample_frequency_array):
        """Generate synthetic PSD for testing."""
        f = sample_frequency_array
        # Create synthetic PSD: JONSWAP-like spectrum
        fp = 0.1  # peak frequency
        gamma = 3.3
        alpha = 0.01

        sigma = np.where(f <= fp, 0.07, 0.09)
        r = np.exp(-(f - fp)**2 / (2 * sigma**2 * fp**2))
        psd = alpha * fp**5 / f**5 * np.exp(-1.25 * (fp/f)**4) * gamma**r

        return psd

    def test_pressure_response_correction_parameters(
        self, sample_frequency_array
    ):
        """
        Test that pressure_response_correction accepts correct parameters.

        Validates fix for: Incorrect parameter usage in wave_params_from_pres.py
        """
        depth = 10.0
        sensor_height = 2.0

        # Should not raise error with correct parameters
        transfer = wave_params_from_pres.pressure_response_correction(
            freq=sample_frequency_array,
            water_depth=depth,
            sensor_height_above_bed=sensor_height,
        )

        # Transfer function should be positive and finite
        assert np.all(np.isfinite(transfer)), (
            "Transfer function should be finite"
        )
        assert np.all(transfer > 0), (
            "Transfer function should be positive"
        )
        assert transfer.shape == sample_frequency_array.shape, (
            "Transfer function shape should match frequency array"
        )

    def test_spectral_moments_parameters(self, sample_frequency_array, sample_psd):
        """
        Test that spectral_moments accepts correct parameters.

        Validates fix for: spectral_moments called with fmin, fmax parameters
        which it doesn't accept.
        """
        # Should not raise error with correct parameters (freq, psd only)
        m0, m_minus1 = wave_params_from_pres.spectral_moments(
            freq=sample_frequency_array,
            psd=sample_psd,
        )

        # Moments should be positive and finite
        assert np.isfinite(m0), "m0 should be finite"
        assert np.isfinite(m_minus1), "m_minus1 should be finite"
        assert m0 > 0, "m0 should be positive"
        assert m_minus1 > 0, "m_minus1 should be positive"

    def test_compute_surface_elevation_spectrum_parameters(
        self, sample_psd, sample_frequency_array
    ):
        """
        Test that compute_surface_elevation_spectrum accepts correct parameters.

        Validates fix for: Incorrect parameter order in function call.
        """
        # Create transfer function
        transfer = wave_params_from_pres.pressure_response_correction(
            freq=sample_frequency_array,
            water_depth=10.0,
            sensor_height_above_bed=2.0,
        )

        # Should not raise error with correct parameter order
        surface_psd = wave_params_from_pres.compute_surface_elevation_spectrum(
            psd=sample_psd,
            transfer=transfer,
        )

        # Surface PSD should be finite and non-negative
        assert np.all(np.isfinite(surface_psd)), (
            "Surface PSD should be finite"
        )
        assert np.all(surface_psd >= 0), (
            "Surface PSD should be non-negative"
        )
        assert surface_psd.shape == sample_psd.shape, (
            "Surface PSD shape should match input PSD"
        )

    def test_pressure_to_elevation_linear_shallow_simple_parameters(
        self, sample_pressure_data
    ):
        """
        Test that pressure_to_elevation_linear_shallow_simple accepts correct parameters.

        This function converts pressure in dBar to surface elevation for shallow water.
        """
        depth = 10.0  # Total water depth in meters

        # Should not raise error with correct parameters
        elevation, water_column = wave_params_from_pres.pressure_to_elevation_linear_shallow_simple(
            water_pressure_dbar=sample_pressure_data.values,
            depth=depth,
        )

        # Elevation should be finite
        assert np.all(np.isfinite(elevation)), (
            "Elevation should be finite"
        )
        assert elevation.shape == sample_pressure_data.shape, (
            "Elevation shape should match pressure"
        )
        # Water column should be positive and close to depth
        assert np.isfinite(water_column), "Water column should be finite"
        assert water_column > 0, "Water column should be positive"

    def test_frequency_band_filter(self, sample_frequency_array, sample_psd):
        """Test frequency band filtering functionality."""
        fmin = 0.04
        fmax = 0.3

        f_filtered, psd_filtered = wave_params_from_pres.frequency_band_filter(
            sample_frequency_array, sample_psd, fmin=fmin, fmax=fmax
        )

        # Check that frequencies are within specified range
        assert np.all(f_filtered >= fmin), (
            f"All filtered frequencies should be >= fmin"
        )
        assert np.all(f_filtered <= fmax), (
            f"All filtered frequencies should be <= fmax"
        )
        assert f_filtered.shape == psd_filtered.shape, (
            "Frequency and PSD arrays should have same shape"
        )

    def test_h13_from_time_domain(self, sample_pressure_data):
        """Test H1/3 calculation from time domain."""
        # Convert pressure (dBar) to elevation using shallow water approximation
        depth = 10.0
        elevation, _ = wave_params_from_pres.pressure_to_elevation_linear_shallow_simple(
            water_pressure_dbar=sample_pressure_data.values,
            depth=depth,
        )

        fs = 5.0
        h13 = wave_params_from_pres.h13_from_time_domain(
            eta=elevation,
            fs=fs,
        )

        # H1/3 should be positive and finite
        assert np.isfinite(h13), "H1/3 should be finite"
        assert h13 > 0, "H1/3 should be positive"

    def test_wave_height_agreement(self, sample_pressure_data):
        """
        Test that spectral and time domain methods give similar results.

        For linear Gaussian sea state, Hs (spectral) ≈ H1/3 (time domain)
        within ~10-15% (Rayleigh statistics).

        This test uses the shallow water approximation for both methods.
        """
        depth = 10.0

        # Convert pressure (dBar) to elevation using shallow water approximation
        elevation, water_column = wave_params_from_pres.pressure_to_elevation_linear_shallow_simple(
            water_pressure_dbar=sample_pressure_data.values,
            depth=depth,
        )

        fs = 5.0

        # Time domain H1/3
        h13 = wave_params_from_pres.h13_from_time_domain(
            eta=elevation,
            fs=fs,
        )

        # Spectral domain Hs using shallow water approximation
        # Calculate elevation PSD (in m²/Hz) directly from elevation time series
        # We need to convert elevation (m) to an equivalent pressure in Pa for PSD calculation
        # because calc_psd_welch expects pressure-like values
        # Elevation to pressure conversion: p = rho * g * eta
        rho = 1025.0  # kg/m³
        g = 9.80665  # m/s²
        pressure_from_elevation = elevation * rho * g  # Convert m to Pa

        freq, psd_elevation = wave_params_from_pres.calc_psd_welch(
            pressure=pressure_from_elevation,
            fs=fs,
            input_units="Pa",
        )

        # Convert pressure PSD back to elevation PSD: Se = Sp / (rho * g)²
        psd_elevation = psd_elevation / (rho * g)**2

        f_filtered, psd_filtered = wave_params_from_pres.frequency_band_filter(
            freq, psd_elevation, fmin=0.04, fmax=0.5
        )

        m0, _ = wave_params_from_pres.spectral_moments(
            freq=f_filtered,
            psd=psd_filtered,
        )
        hs = 4.0 * np.sqrt(m0)

        # Check that methods are within reasonable agreement
        # Allow 20% tolerance for synthetic data
        relative_diff = abs(hs - h13) / ((hs + h13) / 2)
        assert relative_diff < 0.20, (
            f"Spectral Hs ({hs:.3f} m) and time domain H1/3 "
            f"({h13:.3f} m) should agree within 20%, "
            f"relative difference: {relative_diff:.1%}"
        )

    def test_spectral_moments_integration(self):
        """
        Test that spectral moments are correctly calculated via integration.

        m0 = integral(S(f) df)
        m(-1) = integral(S(f)/f df)
        """
        # Create simple test case: constant PSD
        f = np.linspace(0.1, 0.5, 100)
        psd = np.ones_like(f) * 0.1  # Constant PSD = 0.1

        m0, m_minus1 = wave_params_from_pres.spectral_moments(freq=f, psd=psd)

        # For constant PSD: m0 = PSD * (fmax - fmin)
        df = np.mean(np.diff(f))
        expected_m0 = 0.1 * (f.max() - f.min())

        # Check m0 calculation (allow 2% tolerance due to numerical integration)
        np.testing.assert_allclose(
            m0, expected_m0, rtol=0.02,
            err_msg="m0 should equal PSD * frequency range"
        )

        # m(-1) should be positive
        assert m_minus1 > 0, "m(-1) should be positive"

    def test_pressure_response_correction_depth_variation(
        self, sample_frequency_array
    ):
        """
        Test that pressure response correction varies correctly with depth.

        The pressure_response_correction function returns the transfer function for
        recovering surface elevation from pressure: K(f) = cosh(kh) / cosh(k(z+h)).
        For a fixed sensor height above bed, deeper water means the sensor
        is deeper below the surface, requiring a larger recovery factor.
        """
        depths = [4.0, 10.0, 20.0]
        transfers = []

        for depth in depths:
            transfer = wave_params_from_pres.pressure_response_correction(
                freq=sample_frequency_array,
                water_depth=depth,
                sensor_height_above_bed=2.0,
            )
            transfers.append(transfer)

        # For fixed sensor height above bed, deeper water should have higher transfer values
        # (sensor is deeper below surface, requiring larger recovery factor)
        for i in range(1, len(depths)):
            assert np.mean(transfers[i]) > np.mean(transfers[i-1]), (
                f"Transfer function should increase with depth for fixed sensor height: "
                f"{depths[i-1]}m -> {depths[i]}m"
            )

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very short signal
        short_signal = np.array([1.0, 2.0, 3.0])
        h13 = wave_params_from_pres.h13_from_time_domain(
            eta=short_signal,
            fs=5.0,
        )
        # Should return NaN for insufficient data
        assert np.isnan(h13), (
            "H1/3 should be NaN for very short signals"
        )

        # Test with constant signal (no variation)
        constant_signal = np.ones(1000)
        h13 = wave_params_from_pres.h13_from_time_domain(
            eta=constant_signal,
            fs=5.0,
        )
        # Should return NaN for constant signal
        assert np.isnan(h13), (
            "H1/3 should be NaN for constant signal"
        )

    def test_wave_period_calculation(self, sample_psd, sample_frequency_array):
        """Test wave period calculation from spectral moments."""
        m0, m_minus1 = wave_params_from_pres.spectral_moments(
            freq=sample_frequency_array,
            psd=sample_psd,
        )

        tm_minus1 = m_minus1 / m0

        # Energy period should be positive and finite
        assert np.isfinite(tm_minus1), "Tm-1 should be finite"
        assert tm_minus1 > 0, "Tm-1 should be positive"

        # For typical wave spectrum, Tm-1 should be between 2-20 seconds
        assert 2.0 < tm_minus1 < 20.0, (
            f"Tm-1 ({tm_minus1:.2f} s) should be in realistic range [2, 20] s"
        )


def run_comparison_test():
    """
    Run comprehensive comparison test and print results.

    This function demonstrates the testing approach without pytest framework.
    """
    print("=" * 70)
    print("Wave Parameters Comparison Test")
    print("=" * 70)

    # Generate test data (pressure in dBar)
    np.random.seed(42)
    fs = 5.0
    duration = 300
    t = np.arange(0, duration, 1/fs)

    # Create synthetic wave signal (pressure in dBar)
    p1 = 0.1 * np.sin(2 * np.pi * 0.1 * t)
    p2 = 0.05 * np.sin(2 * np.pi * 0.15 * t + np.pi/4)
    noise = 0.005 * np.random.randn(len(t))
    pressure = p1 + p2 + noise + 10.0  # 10 dBar base pressure

    # Create pandas Series
    index = pd.date_range("2023-01-01", periods=len(t), freq=f"{1/fs*1000}ms")
    pressure_series = pd.Series(pressure, index=index, name="Pressure")

    depth = 10.0  # Water depth in meters

    print(f"\nGenerated synthetic pressure data:")
    print(f"  Duration: {duration} s ({duration/60:.1f} min)")
    print(f"  Sampling frequency: {fs} Hz")
    print(f"  Number of samples: {len(pressure)}")
    print(f"  Pressure range: {pressure.min():.3f} - {pressure.max():.3f} dBar")
    print(f"  Water depth: {depth} m")

    # Method 1: Time domain H1/3 using shallow water approximation
    elevation, water_column = wave_params_from_pres.pressure_to_elevation_linear_shallow_simple(
        water_pressure_dbar=pressure_series.values,
        depth=depth,
    )
    h13 = wave_params_from_pres.h13_from_time_domain(eta=elevation, fs=fs)
    print(f"\nMethod 1 - Time Domain H1/3 (shallow water):")
    print(f"  H1/3 = {h13:.3f} m")
    print(f"  Computed water column: {water_column:.3f} m")


    # Spectral domain methods

    # Calculate pressure PSD (in Pa²/Hz)
    freq, psd_pressure = wave_params_from_pres.calc_psd_welch(
        pressure=pressure_series.values,
        fs=fs,
        input_units="dBar",
    )

    f_filtered, psd_filtered = wave_params_from_pres.frequency_band_filter(
        freq, psd_pressure, fmin=0.04, fmax=0.5
    )

    sensor_height = 2.0
    transfer = wave_params_from_pres.pressure_response_correction(
        freq=f_filtered,
        water_depth=depth,
        sensor_height_above_bed=sensor_height,
    )

    psd_surface = wave_params_from_pres.compute_surface_elevation_spectrum(
        psd=psd_filtered,
        transfer=transfer,
    )

    # Method 2: Spectral domain Hs (manual calculation) with pressure response correction
    # Use the same corrected surface elevation spectrum as Method 3
    m0, m_minus1 = wave_params_from_pres.spectral_moments(
        freq=f_filtered,
        psd=psd_surface,
    )
    hs_welch = 4.0 * np.sqrt(m0)
    tm_minus1 = m_minus1 / m0

    print(f"\nMethod 2 - Spectral Domain (manual calculation, with pressure response correction):")
    print(f"  Hs = {hs_welch:.3f} m")
    print(f"  Tm-1 = {tm_minus1:.3f} s")
    print(f"  m0 = {m0:.6f} m²")
    print(f"  m(-1) = {m_minus1:.6f} m²·s")

    # Method 3: Spectral wavespectra_metrics with pressure response correction
    spec = wave_params_from_pres.wavespectra_metrics(f_filtered, psd_surface)
    m0_ws = float(spec.momf(0))
    m_minus1_ws = float(spec.momf(-1))
    hs_corrected = 4.0 * np.sqrt(m0_ws)  # Use standard formula Hs = 4*sqrt(m0)
    tm_minus1_corrected = m_minus1_ws / m0_ws
    tm01_corrected = float(spec.tm01())
    tm02_corrected = float(spec.tm02())
    tp_corrected = float(spec.tp())
    hs_ws_native = float(spec.hs())  # Native wavespectra Hs (for comparison)

    print(f"\nMethod 3 - Spectral Domain with Pressure Response Correction (wavespectra):")
    print(f"  Depth: {depth} m")
    print(f"  Sensor height: {sensor_height} m")
    print(f"  Hs (4*sqrt(m0)) = {hs_corrected:.3f} m")
    print(f"  Hs (wavespectra native) = {hs_ws_native:.3f} m")
    print(f"  Tm-1 = {tm_minus1_corrected:.3f} s")
    print(f"  Tm01 = {tm01_corrected:.3f} s")
    print(f"  Tm02 = {tm02_corrected:.3f} s")
    print(f"  Tp = {tp_corrected:.3f} s")
    print(f"  m0 (wavespectra) = {m0_ws:.6f} m²")
    print(f"  m(-1) (wavespectra) = {m_minus1_ws:.6f} m²·s")

    # Comparison
    print(f"\nComparison:")
    print(f"  H1/3 (time domain, shallow water): {h13:.3f} m")
    print(f"  Hs (spectral, manual, with correction): {hs_welch:.3f} m")
    print(f"  Hs (spectral, wavespectra, with correction): {hs_corrected:.3f} m")

    diff_manual = abs(hs_welch - h13) / ((hs_welch + h13) / 2) * 100
    diff_wavespectra = abs(hs_corrected - h13) / ((hs_corrected + h13) / 2) * 100
    diff_methods = abs(hs_corrected - hs_welch) / ((hs_corrected + hs_welch) / 2) * 100

    print(f"\n  Difference (Manual vs H1/3): {diff_manual:.1f}%")
    print(f"  Difference (Wavespectra vs H1/3): {diff_wavespectra:.1f}%")
    print(f"  Difference (Wavespectra vs Manual): {diff_methods:.1f}%")

    # Validation
    print(f"\nValidation:")
    if diff_manual < 20:
        print(f"  ✓ Manual spectral and time domain agree within 20%")
    else:
        print(f"  ✗ Manual spectral and time domain differ by {diff_manual:.1f}% (>20%)")

    if diff_wavespectra < 20:
        print(f"  ✓ Wavespectra and time domain agree within 20%")
    else:
        print(f"  ✗ Wavespectra and time domain differ by {diff_wavespectra:.1f}% (>20%)")

    if diff_methods < 1:
        print(f"  ✓ Manual and wavespectra calculations agree within 1%")
    else:
        print(f"  ✗ Manual and wavespectra calculations differ by {diff_methods:.1f}% (>1%)")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison_test()
