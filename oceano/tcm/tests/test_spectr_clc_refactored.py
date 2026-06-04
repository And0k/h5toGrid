#!/usr/bin/env python3
"""
Test script to verify the refactored spectr_clc functions work correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add project to path

from tcm.spectr_clc import _get_n_fft, _get_freqs


def test_get_n_fft():
    """Test _get_n_fft function."""
    print("Testing _get_n_fft()...")

    # Test case 1: Small length
    result = _get_n_fft(100)
    expected = 256  # max(256, 2^ceil(log2(100))) = max(256, 128) = 256
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  ✓ _get_n_fft(100) = {result} (expected {expected})")

    # Test case 2: Medium length
    result = _get_n_fft(180000)
    expected = 262144  # max(256, 2^ceil(log2(180000))) = max(256, 262144) = 262144
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  ✓ _get_n_fft(180000) = {result} (expected {expected})")

    # Test case 3: Large length
    result = _get_n_fft(500000)
    expected = 524288  # max(256, 2^ceil(log2(500000))) = max(256, 524288) = 524288
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  ✓ _get_n_fft(500000) = {result} (expected {expected})")

    print("✓ All _get_n_fft() tests passed!\n")


def test_get_freqs():
    """Test _get_freqs function."""
    print("Testing _get_freqs()...")

    # Test case 1: Typical wave analysis parameters
    n_fft = 262144
    dt = 1.0  # 1 Hz sampling
    fmin = 0.04
    fmax = 0.5

    freqs, freq_mask = _get_freqs(n_fft, dt, fmin, fmax)

    # Check that freqs are within the specified range
    assert freqs[0] >= fmin, f"First frequency {freqs[0]} < fmin {fmin}"
    assert freqs[-1] <= fmax, f"Last frequency {freqs[-1]} > fmax {fmax}"
    print(f"  ✓ Frequency range: {freqs[0]:.4f} - {freqs[-1]:.4f} Hz ({len(freqs)} frequencies)")

    # Check that freq_mask is a boolean array
    assert freq_mask.dtype == bool, f"freq_mask should be boolean, got {freq_mask.dtype}"
    print(f"  ✓ freq_mask is boolean array with {np.sum(freq_mask)} True values")

    # Check that freq_mask correctly filters frequencies
    full_freqs = np.fft.rfftfreq(n_fft, dt)
    assert np.array_equal(freqs, full_freqs[freq_mask]), "freqs should equal full_freqs[freq_mask]"
    print(f"  ✓ freqs correctly matches full_freqs[freq_mask]")

    # Test case 2: Different parameters
    n_fft = 256
    dt = 0.01  # 100 Hz sampling
    fmin = 1.0
    fmax = 10.0

    freqs, freq_mask = _get_freqs(n_fft, dt, fmin, fmax)
    assert freqs[0] >= fmin, f"First frequency {freqs[0]} < fmin {fmin}"
    assert freqs[-1] <= fmax, f"Last frequency {freqs[-1]} > fmax {fmax}"
    print(f"  ✓ Frequency range (test 2): {freqs[0]:.2f} - {freqs[-1]:.2f} Hz ({len(freqs)} frequencies)")

    print("✓ All _get_freqs() tests passed!\n")


def test_integration():
    """Test integration between _get_n_fft and _get_freqs."""
    print("Testing integration between _get_n_fft() and _get_freqs()...")

    # Simulate typical workflow
    length = 180000
    dt = 1.0
    fmin = 0.04
    fmax = 0.5

    # Step 1: Calculate n_fft from length
    n_fft = _get_n_fft(length)
    print(f"  ✓ _get_n_fft({length}) = {n_fft}")

    # Step 2: Calculate frequencies from n_fft
    freqs, freq_mask = _get_freqs(n_fft, dt, fmin, fmax)
    print(f"  ✓ _get_freqs({n_fft}, {dt}, {fmin}, {fmax}) returned {len(freqs)} frequencies")

    # Verify the results make sense
    assert len(freqs) > 0, "Should have at least one frequency"
    assert np.sum(freq_mask) == len(freqs), "freq_mask should have len(freqs) True values"
    print(f"  ✓ Integration test passed!")

    print("✓ Integration test passed!\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing refactored spectr_clc functions")
    print("=" * 60 + "\n")

    try:
        test_get_n_fft()
        test_get_freqs()
        test_integration()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
