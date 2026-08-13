"""Tests for tcm.csv_load.estimate_n_chunks — cheap file-size based chunk estimation.

Verifies:
- Returns 1 when blocksize is falsy (no chunking)
- Returns 0 for empty files
- Returns 1 on unreadable/missing paths (graceful fallback)
- Estimates scale with file size and blocksize
- Header rows (skiprows) are subtracted from the line count
- Large files (> sample_bytes) are correctly extrapolated
"""

from __future__ import annotations

import pytest

from tcm.csv_load import estimate_n_chunks


@pytest.mark.xr
class TestEstimateNChunks:
    """estimate_n_chunks samples first 1 MB, extrapolates total data lines."""

    @pytest.mark.parametrize(
        "blocksize, expected",
        [
            pytest.param(0, 1, id="blocksize-0-returns-1"),
            pytest.param(None, 1, id="blocksize-None-returns-1"),
        ],
    )
    def test_no_chunking_returns_1(self, tmp_path, blocksize, expected):
        """When blocksize is falsy the file is read in one pass."""
        f = tmp_path / "data.txt"
        f.write_text("a,b,c\n" * 100, encoding="utf-8")
        assert estimate_n_chunks(f, blocksize) == expected, (
            f"blocksize={blocksize}: expected {expected}, got {estimate_n_chunks(f, blocksize)}"
        )

    def test_empty_file_returns_0(self, tmp_path):
        """An empty file has no data rows → 0 chunks."""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert estimate_n_chunks(f, 100) == 0, "Empty file should yield 0 chunks"

    def test_missing_file_returns_1(self, tmp_path):
        """OS errors (missing file) fall back to 1."""
        f = tmp_path / "nonexistent.txt"
        assert estimate_n_chunks(f, 100) == 1, "Missing file should fallback to 1"

    @pytest.mark.parametrize(
        "n_lines, blocksize, skiprows, expected",
        [
            pytest.param(100, 50, 0, 2, id="100-lines-bs50-no-skip"),
            pytest.param(100, 30, 0, 4, id="100-lines-bs30-no-skip-ceil"),
            pytest.param(100, 200, 0, 1, id="100-lines-bs200-single-chunk"),
            pytest.param(100, 50, 3, 2, id="100-lines-bs50-skip3"),
            pytest.param(10, 5, 0, 2, id="10-lines-bs5-exact"),
            pytest.param(10, 5, 2, 2, id="10-lines-bs5-skip2-8-data-rows"),
        ],
    )
    def test_estimation_accuracy(self, tmp_path, n_lines, blocksize, skiprows, expected):
        """Estimate is within ±1 chunk of exact for small files (sample covers all)."""
        f = tmp_path / "data.txt"
        f.write_text("a,b,c\n" * n_lines, encoding="utf-8")
        result = estimate_n_chunks(f, blocksize, skiprows)
        assert result == expected, (
            f"n_lines={n_lines}, blocksize={blocksize}, skiprows={skiprows}: "
            f"expected {expected}, got {result}"
        )

    def test_large_file_extrapolation(self, tmp_path):
        """Files > sample_bytes are extrapolated from the sample (within ±1 chunk)."""
        f = tmp_path / "big.txt"
        # 2 MB of uniform 50-byte lines → ~40 000 lines
        line = "x" * 44 + "\n"  # 45 chars + \n = 46 bytes
        n_lines = 45_000
        f.write_text(line * n_lines, encoding="utf-8")
        blocksize = 10_000
        result = estimate_n_chunks(f, blocksize, sample_bytes=500_000)
        # Exact: ceil(45000 / 10000) = 5; allow ±1 from extrapolation error
        assert 4 <= result <= 6, (
            f"Large file estimate {result} outside [4, 6] for {n_lines} lines / bs={blocksize}"
        )

    def test_skiprows_reduces_chunks(self, tmp_path):
        """skiprows > 0 subtracts header lines, reducing chunk count."""
        f = tmp_path / "data.txt"
        f.write_text("header\n" * 5 + "data\n" * 100, encoding="utf-8")
        no_skip = estimate_n_chunks(f, 50, skiprows=0)
        with_skip = estimate_n_chunks(f, 50, skiprows=5)
        assert with_skip < no_skip, f"skiprows=5 should reduce chunks: {with_skip} >= {no_skip}"

    def test_sample_bytes_parameter(self, tmp_path):
        """Custom sample_bytes controls the sample size (smaller → less precise)."""
        f = tmp_path / "data.txt"
        f.write_text("a,b\n" * 1000, encoding="utf-8")
        # Tiny sample (10 bytes ≈ 2 lines) → extrapolation less precise
        result_tiny = estimate_n_chunks(f, 100, sample_bytes=10)
        result_full = estimate_n_chunks(f, 100, sample_bytes=100_000)
        # Both should be close to ceil(1000/100) = 10
        assert 8 <= result_tiny <= 12, f"Tiny sample: {result_tiny} outside [8, 12]"
        assert result_full == 10, f"Full sample: {result_full} != 10"
