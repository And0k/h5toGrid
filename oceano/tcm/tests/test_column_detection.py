"""Tests for column detection, format selection, and header/dtype consistency.

Reproduces three bugs with a 16-column pressure probe file (Inkl P05 format):

1. ``format_parts_select('p')`` is missing ``"TempP"`` — regex captures only
   15 of 16 columns, so correction strips the last column (Bat values).

2. Scoring in ``format_parts_select_raw`` maps header column ``"TP"`` to
   ``"Temp"`` instead of ``"TempP"`` — the 1-char alias ``"T"`` gets an
   unfair ``p_len == len(pattern)`` bonus, defeating the 5-char ``"TempP"``.

3. ``config_text_params`` doesn't guard against duplicate parts — when
   ``"Temp"`` appears twice in the parts list, ``",".join(parts)`` yields
   16 header names but ``format_parts_all.items()`` yields only 15 dtype
   entries → ``ValueError`` in ``init_input_cols``.
"""
from __future__ import annotations

import re

import pytest
from pathlib import Path

from tcm.csv_load import (
    _text_line_regex_from_parts,
    config_text_params,
    format_parts_all,
    format_parts_select,
    format_parts_select_raw,
    init_input_cols,
)


# ── Synthetic raw files ────────────────────────────────────────────────────

_RAW_PRESSURE_16 = (
    "Inkl P05 Firmware 2560_V1.2b\n"
    "Y,M,D,H,M,S,Ax,Ay,Az,Mx,My,Mz,ADC,T,TP,Bat\n"
    "Inkl P05 Firmware 2560_V1.2b\n"
    "Y,M,D,H,M,S,Ax,Ay,Az,Mx,My,Mz,ADC,T,TP,Bat\n"
    "2026,6,25,17,8,50,8192,12256,-6480,-328,-559,40,0,23.43,23.28,9.47\n"
    "2026,6,25,17,9,0,8190,12260,-6475,-330,-560,41,1,23.44,23.29,9.48\n"
)

_RAW_PRESSURE_15 = (
    "Older pressure probe header\n"
    "Y,M,D,H,M,S,Ax,Ay,Az,Mx,My,Mz,ADC,T,Bat\n"
    "2024,3,15,10,30,0,8000,12000,-6500,-300,-500,35,0,22.5,9.8\n"
    "2024,3,15,10,30,1,8010,12010,-6510,-301,-501,36,1,22.6,9.9\n"
)


@pytest.fixture
def pressure_16col_file(tmp_path: Path) -> Path:
    """Raw 16-column pressure probe file (newer format with TempP)."""
    f = tmp_path / "INKL_P05_0_v_trube.TXT"
    f.write_text(_RAW_PRESSURE_16, encoding="utf-8")
    return f


@pytest.fixture
def pressure_15col_file(tmp_path: Path) -> Path:
    """Raw 15-column pressure probe file (older format without TempP)."""
    f = tmp_path / "p_01.txt"
    f.write_text(_RAW_PRESSURE_15, encoding="utf-8")
    return f


@pytest.fixture
def standard_14col_file(tmp_path: Path) -> Path:
    """Standard 14-column inclinometer file with header."""
    f = tmp_path / "i_01.txt"
    f.write_text(
        "some metadata\n"
        "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz,Battery,Temp\n"
        "2024,01,01,00,00,00,123,456,789,-123,-456,-789,95,22.5\n"
        "2024,01,01,00,00,01,124,457,790,-124,-457,-790,96,22.6\n",
        encoding="utf-8",
    )
    return f


# ── Bug 1: format_parts_select('p') missing TempP ────────────────────────


class TestFormatPartsSelectPressure:
    """format_parts_select('p') must include TempP for pressure probes."""

    def test_p_type_includes_tempp(self):
        """Pressure probes have both T (Temp) and TP (TempP) columns."""
        parts = format_parts_select("p")
        assert "TempP" in parts, (
            f"TempP missing from format_parts_select('p'): {parts}"
        )

    def test_p_type_sensor_order_matches_file(self):
        """Sensor parts for 'p' follow actual file column order: P_counts, Temp, TempP, Battery."""
        parts = format_parts_select("p")
        sensor = [p for p in parts if p in format_parts_all
                  and p not in ("yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text)",
                                "Ax,Ay,Az,Mx,My,Mz")]
        assert sensor == ["P_counts", "Temp", "TempP", "Battery"], (
            f"Sensor part order should be [P_counts, Temp, TempP, Battery], got {sensor}"
        )


# ── Bug 2: format_parts_select_raw scoring — "TP" → "TempP" ─────────────


class TestFormatPartsSelectRawPressure:
    """Autodetection must map 'TP' header column to 'TempP', not 'Temp'."""

    def test_detects_tempp_from_tp_header(self, pressure_16col_file: Path):
        """Header column 'TP' must map to format key 'TempP'."""
        parts, header_line, skiprows = format_parts_select_raw(pressure_16col_file)
        assert "TempP" in parts, (
            f"'TP' should map to TempP but got parts={parts}"
        )

    def test_no_duplicate_parts(self, pressure_16col_file: Path):
        """Detection must not produce duplicate format keys."""
        parts, _, _ = format_parts_select_raw(pressure_16col_file)
        assert len(parts) == len(set(parts)), (
            f"Duplicate parts detected: {parts}"
        )

    def test_detects_all_16_columns(self, pressure_16col_file: Path):
        """All 16 columns must be covered: date(6) + Ax-Mz(6) + P_counts(1) + Temp(1) + TempP(1) + Battery(1)."""
        parts, _, _ = format_parts_select_raw(pressure_16col_file)
        total_cols = sum(len(p.split(",")) for p in parts
                        if p in format_parts_all)
        assert total_cols == 16, (
            f"Expected 16 columns from format parts, got {total_cols} from {parts}"
        )

    def test_standard_file_still_works(self, standard_14col_file: Path):
        """Regression: standard 14-column inclinometer detection must still work."""
        parts, _, _ = format_parts_select_raw(standard_14col_file)
        assert "Temp" in parts
        assert "Battery" in parts
        assert "TempP" not in parts  # 14-col file has no TempP

    def test_15col_pressure_excludes_tempp(self, pressure_15col_file: Path):
        """Older 15-col pressure file (no TP column) → autodetection excludes TempP."""
        parts, _, _ = format_parts_select_raw(pressure_15col_file)
        assert "TempP" not in parts, (
            f"15-col file has no TP column but got TempP in parts={parts}"
        )
        assert "Temp" in parts
        assert "Battery" in parts
        assert "P_counts" in parts

    def test_15col_pressure_no_crash(self, pressure_15col_file: Path):
        """15-col pressure file → full chain (autodetect → init_input_cols) succeeds."""
        params = config_text_params("p", pressure_15col_file)
        n_header = len(params["header"].split(","))
        n_dtype = len(params["dtype"])
        assert n_header == n_dtype, (
            f"15-col pressure: header={n_header}, dtype={n_dtype}"
        )
        cfg = {**params, "max_text_width": 1000}
        result = init_input_cols(cfg)
        assert len(result["dtype_raw"].names) == len(result["dtype"].names)


# ── Bug 3: config_text_params header/dtype length mismatch ───────────────


class TestConfigTextParamsPressure:
    """config_text_params must produce matching header and dtype for 16-col files."""

    def test_pressure_16col_header_dtype_lengths_match(self, pressure_16col_file: Path):
        """16-col pressure file: header names count == dtype item count."""
        params = config_text_params("p", pressure_16col_file)
        n_header = len(params["header"].split(","))
        n_dtype = len(params["dtype"])
        assert n_header == n_dtype, (
            f"header has {n_header} names but dtype has {n_dtype} items"
        )

    def test_fallback_p_type_header_dtype_match(self):
        """text_type='p' fallback (no file): header/dtype lengths match."""
        params = config_text_params("p")
        n_header = len(params["header"].split(","))
        n_dtype = len(params["dtype"])
        assert n_header == n_dtype, (
            f"p fallback: header={n_header}, dtype={n_dtype}"
        )


# ── init_input_cols must not crash with pressure params ───────────────────


class TestInitInputColsPressure:
    """init_input_cols must produce valid dtype_raw for pressure probes."""

    def test_pressure_16col_no_crash(self, pressure_16col_file: Path):
        """This was the crash: ValueError on np.dtype names/formats length mismatch."""
        params = config_text_params("p", pressure_16col_file)
        cfg = {**params, "max_text_width": 1000}
        result = init_input_cols(cfg)
        assert len(result["dtype_raw"].names) == len(result["dtype"].names), (
            f"dtype_raw names ({len(result['dtype_raw'].names)}) != "
            f"dtype names ({len(result['dtype'].names)})"
        )

    def test_fallback_p_type_no_crash(self):
        """Fallback (no file) pressure params must also produce valid dtype_raw."""
        params = config_text_params("p")
        cfg = {**params, "max_text_width": 1000}
        result = init_input_cols(cfg)
        assert len(result["dtype_raw"].names) == len(result["dtype"].names)


# ── Regex must capture ALL columns (no stripping) ─────────────────────────


class TestRegexCapturesAllColumns:
    """Regex from format_parts_select must capture all columns in the data line."""

    def test_pressure_regex_captures_16_columns(self):
        """Regex for 'p' must capture all 16 columns (including TempP and Bat)."""
        parts = format_parts_select("p")
        regex = _text_line_regex_from_parts(parts)
        data_line = b"2026,6,25,17,8,50,8192,12256,-6480,-328,-559,40,0,23.43,23.28,9.47"
        match = re.match(regex, data_line)
        assert match, "Regex did not match 16-column pressure data line"
        captured = match.group("use")
        n_captured = captured.count(b",") + 1
        n_data = data_line.count(b",") + 1
        assert n_captured == n_data, (
            f"Regex captured {n_captured} columns but data has {n_data} — "
            f"last {n_data - n_captured} column(s) would be stripped during correction"
        )

    def test_i_type_regex_captures_14_columns(self):
        """Regression: standard i-type regex still captures 14 columns."""
        parts = format_parts_select("i")
        regex = _text_line_regex_from_parts(parts)
        data_line = b"2024,01,01,00,00,00,123,456,789,-123,-456,-789,95,22.5"
        match = re.match(regex, data_line)
        assert match, "Regex did not match 14-column inclinometer data line"
        captured = match.group("use")
        assert captured.count(b",") + 1 == 14

    def test_autodetect_15col_regex_captures_all(self, pressure_15col_file: Path):
        """Autodetected regex for 15-col pressure file captures all 15 columns."""
        params = config_text_params("p", pressure_15col_file)
        regex = params["text_line_regex"]
        data_line = b"2024,3,15,10,30,0,8000,12000,-6500,-300,-500,35,0,22.5,9.8"
        match = re.match(regex, data_line)
        assert match, "Autodetected regex did not match 15-column pressure data"
        captured = match.group("use")
        n_captured = captured.count(b",") + 1
        assert n_captured == 15, (
            f"Autodetected regex captured {n_captured} columns, expected 15"
        )


# ── Header-vs-data column count guard ─────────────────────────────────────


_BOTCHED_16_15 = (
    "Y,M,D,H,M,S,Ax,Ay,Az,Mx,My,Mz,ADC,T,TP,Bat\n"  # header: 16 cols
    "2026,6,25,17,8,50,8192,12256,-6480,-328,-559,40,0,23.43,23.28\n"  # data: 15 cols
    "2026,6,25,17,9,0,8190,12260,-6475,-330,-560,41,1,23.44,23.29\n"
)


@pytest.fixture
def botched_file(tmp_path: Path) -> Path:
    """Previously botched correction: header has 16 columns, data has 15."""
    f = tmp_path / "@i_p5_.TXT"
    f.write_text(_BOTCHED_16_15, encoding="utf-8")
    return f


class TestHeaderDataColumnGuard:
    """format_parts_select_raw must truncate header to match data column count."""

    def test_botched_file_uses_data_column_count(self, botched_file: Path):
        """Header 16 cols, data 15 cols → detection uses 15 → Battery excluded."""
        parts, _, _ = format_parts_select_raw(botched_file)
        total_cols = sum(len(p.split(",")) for p in parts if p in format_parts_all)
        assert total_cols == 15, (
            f"Expected 15 columns (from data), got {total_cols} from parts={parts}"
        )
        assert "Battery" not in parts, (
            f"Battery should be excluded when data has only 15 columns, got parts={parts}"
        )

    def test_botched_file_full_chain_no_crash(self, botched_file: Path):
        """Botched file → config_text_params → init_input_cols → no ValueError."""
        params = config_text_params("p", botched_file)
        n_header = len(params["header"].split(","))
        n_dtype = len(params["dtype"])
        assert n_header == n_dtype, (
            f"Botched file: header={n_header}, dtype={n_dtype}"
        )
        cfg = {**params, "max_text_width": 1000}
        result = init_input_cols(cfg)
        assert len(result["dtype_raw"].names) == len(result["dtype"].names)
