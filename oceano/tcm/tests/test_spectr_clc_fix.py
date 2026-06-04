#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify spectr_clc.py helper functions work correctly.

This test uses typical values from actual wave_params_run.py script to ensure
functions work correctly in production.

Typical values justification:
- length=3600: Data length for wave gauge measurements with 1 Hz sampling
  over 60 minutes (dt_interval = 60 minutes = 3600 seconds).
  This is calculated as: length = dt_interval * fs = 3600 * 1 = 3600 samples.
- bandwidth=0.0089: Bandwidth calculated from wave_params_run.py as
  bandwidth = 32 / dt_interval = 32 / 3600 = 0.0089 Hz.
  The multitaper method requires bandwidth to satisfy:
  1. NW < M/2 where NW is half_nbw = bandwidth * M / (2 * fs)
     For length=3600, fs=1 Hz: half_nbw = 0.0089 * 3600 / 2 = 16.02
     Kmax = 2 * half_nbw = 32.04 < length=3600 ✓
  2. Kmax (number of tapers = 2*half_nbw) must be less than M (data length)
     For bandwidth=0.0089, Kmax ≈ 32 < 3600 ✓
- dt=1.0: Time step of 1 second (1 Hz sampling rate), which is typical for
  wave gauge measurements.
- fmin=0.04, fmax=0.5: Frequency range for wave analysis, covering swell waves
  (0.04-0.1 Hz) to wind waves (0.1-0.5 Hz).
- n_fft=4096: When explicitly passed, this is FFT size used for
  computation. When not passed, it's calculated as max(256, 2**ceil(log2(length)).

The multitaper DPSS (Discrete Prolate Spheroidal Sequences) method requires:
- half_nbw = bandwidth * length / (2 * fs) to be >= 0.5
- Kmax = 2 * half_nbw (number of tapers) to be less than length (data length)
"""

import numpy as np
from pathlib import Path

from tcm.spectr_clc import _get_n_fft, _get_freqs

# Test 1: Verify _get_n_fft function
print("Testing _get_n_fft()...")
# Note: _get_n_fft() returns max(256, power_of_2), so for lengths < 256 it returns 256
assert _get_n_fft(100) == 256, f"Expected 256 for length 100, got {_get_n_fft(100)}"
assert _get_n_fft(256) == 256, f"Expected 256 for length 256, got {_get_n_fft(256)}"
assert _get_n_fft(500) == 512, f"Expected 512 for length 500, got {_get_n_fft(500)}"
assert _get_n_fft(1000) == 1024, f"Expected 1024 for length 1000, got {_get_n_fft(1000)}"
print("✓ _get_n_fft() tests passed")

# Test 2: Verify _get_freqs function
print("\nTesting _get_freqs()...")
n_fft: int = 256
dt: float = 0.5
fmin: float = 0.04
fmax: float = 0.5

freqs, freq_mask = _get_freqs(n_fft, dt, fmin, fmax)
print(f"  n_fft={n_fft}, dt={dt}, fmin={fmin}, fmax={fmax}")
print(f"  Generated {len(freqs)} frequencies")
print(f"  Freq range: {freqs[0]:.4f} to {freqs[-1]:.4f} Hz")
print(f"  freq_mask has {freq_mask.sum()} True values out of {len(freq_mask)}")
assert len(freqs) > 0, "freqs should not be empty"
assert len(freq_mask) == n_fft // 2 + 1, f"freq_mask length should be {n_fft // 2 + 1} for rfftfreq"
assert freq_mask.sum() == len(freqs), "freq_mask.sum() should equal len(freqs)"
print("✓ _get_freqs() tests passed")

# Test 3: Verify psd_mt_params with n_fft parameter
from tcm.spectr_clc import psd_mt_params

print("\nTesting psd_mt_params() with n_fft parameter...")
# Use actual production parameters from wave_params_run.py:
# - dt_interval = 60 minutes = 3600 seconds
# - bandwidth = 32 / dt_interval = 32 / 3600 = 0.0089 Hz
# - fs (sampling frequency) for wave gauges is typically 1 Hz
# - length = dt_interval * fs = 3600 * 1 = 3600 samples
# - n_fft = 2**ceil(log2(length)) = 2**12 = 4096
prm = psd_mt_params(
    length=3600,  # length = dt_interval * fs = 3600 * 1 Hz
    bandwidth=0.0089,  # bandwidth = 32 / dt_interval = 32 / 3600
    low_bias=True,
    adaptive=True,
    dt=1.0,  # 1 Hz sampling frequency
    n_fft=4096  # Explicitly pass n_fft (calculated as 2**ceil(log2(3600)))
)
print(f"  prm['n_fft'] = {prm['n_fft']}")
print(f"  prm['length'] = {prm['length']}")
print(f"  prm['fs'] = {prm['fs']}")
print(f"  Number of tapers: {len(prm['eigvals'])}")
assert prm['n_fft'] == 4096, f"Expected n_fft=4096, got {prm['n_fft']}"
print("✓ psd_mt_params() with n_fft parameter test passed")

# Test 4: Verify psd_mt_params without n_fft parameter (should calculate)
print("\nTesting psd_mt_params() without n_fft parameter (should calculate)...")
prm2 = psd_mt_params(
    length=3600,  # length = dt_interval * fs = 3600 * 1 Hz
    bandwidth=0.0089,  # bandwidth = 32 / dt_interval = 32 / 3600
    low_bias=True,
    adaptive=True,
    dt=1.0,  # 1 Hz sampling frequency
)
print(f"  prm['n_fft'] = {prm2['n_fft']}")
print(f"  prm['length'] = {prm2['length']}")
print(f"  prm['fs'] = {prm2['fs']}")
print(f"  Number of tapers: {len(prm2['eigvals'])}")
# n_fft should be calculated as max(256, 2**ceil(log2(3600))) = 2**12 = 4096
assert prm2['n_fft'] == 4096, f"Expected n_fft=4096, got {prm2['n_fft']}"
print("✓ psd_mt_params() without n_fft parameter test passed")

print("\n" + "="*70)
print("All tests passed! The helper functions are working correctly.")
print("="*70)