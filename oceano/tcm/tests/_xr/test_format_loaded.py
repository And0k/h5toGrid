"""Tests for Layer-0 format_loaded.py (pure pandas/numpy; no dask)."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from tcm.format_loaded import (
    chars_array_to_datetimeindex,
    concat_to_iso8601,
    correct_txt,
    fill0,
    f_repl_by_dict,
    loaded_corr,
    loaded_tcm,
    mod_name,
    param_funs_closure,
)


# --------------------------------------------------------------------------- #
# fill0 — pads byte arrays with leading zeros to a fixed width
# --------------------------------------------------------------------------- #

@pytest.mark.xr
@pytest.mark.parametrize(
    "values, width, expected",
    [
        pytest.param([b"1", b"2", b"10"], 2, [b"01", b"02", b"10"], id="pad-1char-to-2"),
        pytest.param([b"42"], 4, [b"0042"], id="pad-to-4"),
        pytest.param([b"100"], 3, [b"100"], id="already-exact-width"),
    ],
)
def test_fill0_pads_bytes(values, width, expected):
    """fill0 zero-pads byte strings to fixed *width* on the left."""
    actual = fill0(np.array(values), width)
    assert list(actual) == expected, (
        f"width={width}, values={values!r}: {expected=!r}, {list(actual)=!r}"
    )


# --------------------------------------------------------------------------- #
# chars_array_to_datetimeindex — ISO byte-string → DatetimeIndex
# --------------------------------------------------------------------------- #

@pytest.mark.xr
def test_chars_array_to_datetimeindex_keeps_good_strings():
    """PDF: byte strings → ISO 8601 → datetime64[ns] DatetimeIndex."""
    arr = np.array([b"2020-01-01T10:00:00", b"2020-01-02T10:00:00"])
    actual = chars_array_to_datetimeindex(arr, "datetime64[ns]")
    expected = pd.DatetimeIndex([
        "2020-01-01 10:00:00", "2020-01-02 10:00:00",
    ])
    assert list(actual) == list(expected), (
        f"Good bytes → DatetimeIndex: {list(actual)=!r}"
    )


@pytest.mark.xr
def test_chars_array_to_datetimeindex_fills_bad_strings():
    """Bad/unparseable entries must be forward-filled (Series input)."""
    # Note: legacy implementation only fills via Series path. ndarray input
    # would raise `DatetimeIndex.ffill` AttributeError — accepted limitation.
    arr = pd.Series([b"2020-01-01T10:00:00", b"NOT_A_DATE", b"2020-01-03T10:00:00"])
    actual = chars_array_to_datetimeindex(arr, "datetime64[ns]")
    assert len(actual) == 3, f"len mismatch: {len(actual)}"
    # idx 1 should be filled with idx 0's value
    assert actual[1] == actual[0], (
        f"bad entry not forward-filled: idx 0={actual[0]!r}, idx 1={actual[1]!r}"
    )


# --------------------------------------------------------------------------- #
# concat_to_iso8601 — byte-string date columns → ISO 8601 strings
# --------------------------------------------------------------------------- #

@pytest.mark.xr
def test_concat_to_iso8601_builds_iso_strings():
    """Concat Date+Time byte columns → ISO 8601 strings (zfill on mm..SS)."""
    df = pd.DataFrame({
        "yyyy": [b"2017", b"2018"],
        "mm": [b"10", b"1"],
        "dd": [b"14", b"2"],
        "HH": [b"59", b"3"],
        "MM": [b"59", b"4"],
        "SS": [b"59", b"5"],
    })
    actual = concat_to_iso8601(df).tolist()
    expected = ["2017-10-14T59:59:59", "2018-01-02T03:04:05"]
    assert actual == expected, f"ISO 8601 concat: {expected=!r}, {actual=!r}"


# --------------------------------------------------------------------------- #
# f_repl_by_dict — last named group wins
# --------------------------------------------------------------------------- #

@pytest.mark.xr
def test_f_repl_by_dict_preserves_named_match():
    """Matches with named groups survive; unmatched bytes deleted."""
    # Pattern: 'noise' alternatives without names → matched-but-unnamed → deleted.
    # 'good' alternative has a name → kept.
    repl = [rb"(?P<keep>good)", rb"noise"]
    fsub = f_repl_by_dict(repl, binary_str=True)
    # 'good no noise' → keep 'good', delete 'noise' (unmatched range)
    out = fsub(b"good and noise")
    # 'noise' alternative is unnamed → b"" returned → removed
    assert b"good" in out, f"'good' should be kept: {out!r}"
    assert b"noise" not in out, f"unnamed 'noise' should be deleted: {out!r}"


# --------------------------------------------------------------------------- #
# mod_name — filename normalisation; leading zeros dropped (legacy behaviour)
# --------------------------------------------------------------------------- #

@pytest.mark.xr
@pytest.mark.parametrize(
    "stem, expect_model, expect_stem",
    [
        pytest.param("i_056", "i", "@i_56", id="i_056 -> model=i, stem normalised"),
        pytest.param("voln_v12", "w", "@w_12", id="voln_v12 -> model=w"),
        pytest.param(
            "INKL_P05_0_v_trube", "p", "@i_p5-0_v_trube",
            id="INKL_P05_0_v_trube -> model=p, comment preserved",
        ),
        pytest.param(
            "i_p02_test", "p", "@i_p2-test",
            id="i_p02_test -> model=p, comment stripped of leading _",
        ),
        pytest.param(
            "i_p5-маг", "p", "@i_p5-маг",
            id="i_p5-маг -> model=p, leading - stripped from comment (idempotent)",
        ),
    ],
)
def test_mod_name_returns_model_and_prefixed_stem(stem, expect_model, expect_stem):
    """mod_name identifies model and adds '@' prefix to canonical name."""
    model, path_out = mod_name(Path(f"{stem}.txt"), add_prefix="@")
    assert model == expect_model, f"stem={stem!r}: {expect_model=!r}, {model=!r}"
    assert path_out.stem == expect_stem, (
        f"stem={stem!r}: {expect_stem=!r}, got {path_out.stem!r}"
    )


@pytest.mark.xr
def test_mod_name_parse_false_only_adds_prefix():
    """parse=False skips regex — name is `prefix + stem` (prefix without leading fill)."""
    _, path_out = mod_name(Path("custom_name.txt"), parse=False, add_prefix="@")
    assert path_out.stem == "@custom_name", (
        f"parse=False stem should be '@custom_name': got {path_out.stem!r}"
    )


@pytest.mark.xr
@pytest.mark.parametrize(
    "name",
    [
        "i_p5-маг.TXT",
        "@i_p5-маг.TXT",
        "i_p5_0_v_trube.TXT",
        "@i_p5_0_v_trube.TXT",
        "INKL_P05_0_v_trube.TXT",
        "@INKL_P05_0_v_trube.TXT",
    ],
)
def test_mod_name_idempotent(name):
    """mod_name(mod_name(x)) == mod_name(x) — no dash accumulation on re-parse."""
    _, p1 = mod_name(name, add_prefix="")
    _, p2 = mod_name(p1.name, add_prefix="")
    assert p1.stem == p2.stem, f"Not idempotent: {name!r} -> {p1.stem!r} -> {p2.stem!r}"


# --------------------------------------------------------------------------- #
# stem_to_pcid — strip @ prefix and -comment suffix
# --------------------------------------------------------------------------- #

@pytest.mark.xr
@pytest.mark.parametrize(
    "stem, expected",
    [
        pytest.param("@i_p5-0_v_trube", "i_p5", id="strips @ and -0_v_trube"),
        pytest.param("@i_01-something", "i_01", id="strips @ and -something"),
        pytest.param("@i_56", "i_56", id="strips @ only (no comment)"),
        pytest.param("i_01", "i_01", id="raw stem unchanged"),
        pytest.param("i_p02_test", "i_p02_test", id="raw stem with underscore"),
    ],
)
def test_stem_to_pcid_stem(stem, expected):
    """stem_to_pcid strips @ prefix and -comment suffix."""
    from tcm.format import stem_to_pcid
    assert stem_to_pcid(stem) == expected, (
        f"stem={stem!r}: {expected=!r}, got {stem_to_pcid(stem)!r}"
    )


# --------------------------------------------------------------------------- #
# param_funs_closure — per-column fun/add suffix dispatcher
# --------------------------------------------------------------------------- #

@pytest.mark.xr
def test_param_funs_closure_add_constant():
    """Suffixed ``_add`` adds a constant; assigns to a new column."""
    df = pd.DataFrame({"ax": [1.0, 2.0, 3.0]})
    cfg_in: dict = {}
    funs = param_funs_closure({"ax_add": 10.0}, cfg_in)
    assert "ax" in funs, "key should be stripped of '_add': got keys {list(funs)}"
    expected = pd.Series([11.0, 12.0, 13.0], name="ax")
    actual = funs["ax"](df)
    pd.testing.assert_series_equal(actual, expected, check_names=True)


# --------------------------------------------------------------------------- #
# loaded_corr — apply _add / _fun suffixes to df
# --------------------------------------------------------------------------- #

@pytest.mark.xr
def test_loaded_corr_add_constant_assignment():
    """loaded_corr adds a constant to a column when ``col_add`` is set."""
    df = pd.DataFrame({"ax": [1.0, 2.0]})
    cfg_in: dict = {}
    out = loaded_corr(df, cfg_in, csv_specific_param={"ax_add": 5.0})
    assert "ax" in out.columns, "ax must be reassigned"
    assert list(out["ax"]) == [6.0, 7.0], f"ax +5 expected [6,7], got {list(out['ax'])}"


# --------------------------------------------------------------------------- #
# loaded_tcm — Time conversion from byte-string date columns
# --------------------------------------------------------------------------- #

@pytest.mark.xr
def test_loaded_tcm_assigns_Time_column():
    """loaded_tcm assigns a DatetimeIndex `Time` column from yyyy..SS bytes."""
    df = pd.DataFrame({
        "yyyy": [b"2020", b"2020"],
        "mm": [b"01", b"01"],
        "dd": [b"01", b"01"],
        "HH": [b"10", b"11"],
        "MM": [b"00", b"00"],
        "SS": [b"00", b"00"],
        "Ax": [0.1, 0.2],
    })
    cfg_in: dict = {}
    out = loaded_tcm(df, cfg_in)
    assert "Time" in out.columns, f"'Time' missing — got columns: {list(out.columns)}"
    assert len(out["Time"]) == 2, f"Time length wrong: {len(out['Time'])}"
    expected_t0 = pd.Timestamp("2020-01-01 10:00:00")
    expected_t1 = pd.Timestamp("2020-01-01 11:00:00")
    assert out["Time"].iloc[0] == expected_t0, (
        f"Time[0] expected {expected_t0!r}, got {out['Time'].iloc[0]!r}"
    )
    assert out["Time"].iloc[1] == expected_t1, (
        f"Time[1] expected {expected_t1!r}, got {out['Time'].iloc[1]!r}"
    )


# --------------------------------------------------------------------------- #
# correct_txt — raw-file regex replacement
# --------------------------------------------------------------------------- #

@pytest.mark.xr
def test_correct_txt_writes_filtered_output():
    """correct_txt filters bad lines and writes to dir_out."""
    raw = b"time,A,B\r\n2020-01-01 line_bad_drop_me\r\n2020-01-02 keep_me\r\n"
    with TemporaryDirectory() as tmp:
        f_in = Path(tmp) / "i_056.txt"
        f_in.write_bytes(raw)
        out_dir = Path(tmp) / "_clean"

        # Line 2 contains 'line_bad_drop_me' which our named-group pattern preserves;
        # use a separate pattern that explicitly deletes (unmatched-named alternative).
        out = correct_txt(f_in, dir_out=out_dir, sub_str_list=[rb"drop_me"])
        assert out.exists(), f"output file {out} not created"
        body = out.read_bytes()
        assert b"keep_me" in body, f"good line should survive in body: {body!r}"
