"""Unified time-range reader: bad-line tolerance, formats, archives without extraction."""

import datetime as dt
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from meta_finder import utils_sys
from meta_finder.data_proc_funcs import extract_time_info_from_text_file

BASE = dt.datetime(2024, 3, 1, 0, 0, 0)


def _rows(start, n, step=1, gap_after=(), gap_len=0, sep="\t"):
    """Build timestamped rows with optional gaps: gap_after holds 0-based indices after which time jumps."""
    out, cur = [], start
    for i in range(n):
        out.append(f"{cur.isoformat()}{sep}1{sep}2")
        cur += dt.timedelta(seconds=gap_len if i in gap_after else step)
    return out


def _write(path, lines, encoding="utf-8"):
    Path(path).write_text("\n".join(lines) + "\n", encoding=encoding)
    return Path(path).name


@pytest.mark.parametrize(
    "name,content,kwargs,expected,description",
    [
        (
            "std.tsv",
            ["Time\tAx\tAy", *_rows(BASE, 5)],
            {},
            ("2024-03-01 00:00:00", "2024-03-01 00:00:04", "-", "-"),
            "standard tab file with header and continuous 1s data",
        ),
        (
            "garbage.tsv",
            [f"garbage line {i}" for i in range(12)] + ["", "   ", "Time\tAx", *_rows(BASE, 3)],
            {},
            ("2024-03-01 00:00:00", "2024-03-01 00:00:02", "-", "-"),
            "12 leading garbage lines exceed the old single-attempt start search",
        ),
        (
            "nantail.tsv",
            ["Time\tAx", *_rows(BASE, 4), "2024-03-01T00:00:04\tNaN", "2024-03-01T00:00:05\tNaN", ""],
            {},
            ("2024-03-01 00:00:00", "2024-03-01 00:00:03", "-", "-"),
            "trailing NaN-only rows and blank line excluded from end edge",
        ),
        (
            "comma.csv",
            ["Time,Ax", *_rows(BASE, 3, sep=",")],
            {},
            ("2024-03-01 00:00:00", "2024-03-01 00:00:02", "-", "-"),
            "comma separator autodetect",
        ),
        (
            "space.dat",
            ["Time Ax", *_rows(BASE, 3, sep=" ")],
            {},
            ("2024-03-01 00:00:00", "2024-03-01 00:00:02", "-", "-"),
            "space separator autodetect",
        ),
        (
            "tz.csv",
            ["Time,Ax", "2024-03-01T00:00:00+0200,1", "2024-03-01T00:00:01+0200,2"],
            {},
            ("2024-02-29 22:00:00", "2024-02-29 22:00:01", "-", "-"),
            "+HHMM timezone suffix fixed and normalized to UTC-naive",
        ),
        (
            "zulu.csv",
            ["Time,Ax", "2024-03-01T00:00:00Z,1", "2024-03-01T00:00:01Z,2"],
            {},
            ("2024-03-01 00:00:00", "2024-03-01 00:00:01", "-", "-"),
            "trailing Zulu designator parsed",
        ),
        (
            "serial.tsv",
            ["Time\tAx", "45292.0\t1", "45292.000011574074\t2"],
            {},
            ("2024-01-01 00:00:00", "2024-01-01 00:00:01", "-", "-"),
            "Excel serial timestamps via fallback",
        ),
        (
            "single.tsv",
            ["Time\tAx", "2024-03-01T00:00:00\t1"],
            {},
            ("2024-03-01 00:00:00", "2024-03-01 00:00:00", "-", "-"),
            "single data row yields identical start and end",
        ),
        (
            "hint.csv",
            ["Time,Ax", *_rows(BASE, 3, sep=",")],
            {"sep": ",", "encoding": "cp1251"},
            ("2024-03-01 00:00:00", "2024-03-01 00:00:02", "-", "-"),
            "explicit separator and encoding hints accepted",
        ),
    ],
    ids=["std", "garbage", "nantail", "comma", "space", "tz", "zulu", "serial", "single", "hints"],
)
def test_time_edges_from_content(tmp_path, name, content, kwargs, expected, description):
    """Edges come from timestamp-validated runs, not blank-line checks."""
    _write(tmp_path / name, content)
    actual = extract_time_info_from_text_file(tmp_path, PurePosixPath(name), **kwargs)
    assert actual is not None, f"{description}: expected time info, got None"
    assert tuple(actual) == expected, f"{description}: {tuple(actual)!r} != {expected!r}"


def test_raw_six_column_edges(tmp_path):
    """Raw _raw files parse Year..Second columns split on comma/tab/space."""
    raw_dir = tmp_path / "_raw"
    raw_dir.mkdir()
    description = "raw six-column rows without header"
    _write(raw_dir / "130510.txt", ["2013,05,10,10,00,00,1", "2013,05,10,10,00,05,2"])
    actual = extract_time_info_from_text_file(tmp_path, PurePosixPath("_raw/130510.txt"))
    assert actual is not None, f"{description}: expected time info, got None"
    assert (actual[0], actual[1]) == ("2013-05-10 10:00:00", "2013-05-10 10:00:05"), (
        f"{description}: got {(actual[0], actual[1])!r}"
    )


def test_cp1251_header_decodes(tmp_path):
    """CP1251-encoded non-ASCII header must not break time extraction."""
    description = "cp1251 Russian header with ASCII timestamps"
    lines = ["Время\tAx", "2024-03-01T00:00:00\t1", "2024-03-01T00:00:02\t2"]
    _write(tmp_path / "cp.tsv", lines, encoding="cp1251")
    actual = extract_time_info_from_text_file(tmp_path, PurePosixPath("cp.tsv"))
    assert actual is not None, f"{description}: expected time info, got None"
    assert (actual[0], actual[1]) == ("2024-03-01 00:00:00", "2024-03-01 00:00:02"), (
        f"{description}: got {(actual[0], actual[1])!r}"
    )


def test_inverted_edges_repaired_to_parsed_span(tmp_path):
    """Non-monotonic file (head rows later than tail rows) repairs to min/max of parsed rows."""
    description = "inverted first/last-row edges cannot order the file — full scan spans parsed rows"
    top = [f"{(BASE + dt.timedelta(hours=1, seconds=i)).isoformat()}\t{i}" for i in range(5)]
    bottom = [f"{(BASE + dt.timedelta(seconds=i)).isoformat()}\t{i}" for i in range(5)]
    _write(tmp_path / "wrap.tsv", ["Time\tAx", *top, *bottom])
    actual = extract_time_info_from_text_file(tmp_path, PurePosixPath("wrap.tsv"))
    assert actual is not None, f"{description}: expected time info, got None"
    assert (actual[0], actual[1]) == ("2024-03-01 00:00:00", "2024-03-01 01:00:04"), (
        f"{description}: got {(actual[0], actual[1])!r}"
    )


def test_full_scan_reports_interior_line_numbers(tmp_path):
    """Min/max line numbers pinpoint the wrap point (header=1, top rows 2-6, bottom 7-11)."""
    from meta_finder.data_proc_funcs import _full_span_time_minmax

    description = "interior min/max line numbers"
    top = [f"{(BASE + dt.timedelta(hours=1, seconds=i)).isoformat()}\t{i}" for i in range(5)]
    bottom = [f"{(BASE + dt.timedelta(seconds=i)).isoformat()}\t{i}" for i in range(5)]
    _write(tmp_path / "wrap.tsv", ["Time\tAx", *top, *bottom])
    actual = _full_span_time_minmax(tmp_path, PurePosixPath("wrap.tsv"))
    assert actual == (dt.datetime(2024, 3, 1, 0, 0, 0), dt.datetime(2024, 3, 1, 1, 0, 4), 7, 6), (
        f"{description}: got {actual!r}"
    )


def test_no_valid_timestamps_returns_none(tmp_path):
    """Header plus garbage yields None instead of raising."""
    description = "no parseable timestamps in file"
    _write(tmp_path / "bad.tsv", ["Time\tAx", "garbage", "NaN\tNaN"])
    assert extract_time_info_from_text_file(tmp_path, PurePosixPath("bad.tsv")) is None, description


def test_burst_gaps_detected(tmp_path):
    """Two 600s gaps in 1s data give burst_dt=119 and bursts_t=719 (120-row work spans 119 s)."""
    description = "burst work/gap pattern"
    rows = _rows(BASE, 360, gap_after={119, 239}, gap_len=600)
    _write(tmp_path / "burst.tsv", ["Time\tAx", *rows])
    actual = extract_time_info_from_text_file(tmp_path, PurePosixPath("burst.tsv"), averaging_interval=1)
    assert actual is not None, f"{description}: expected time info, got None"
    end = (BASE + dt.timedelta(seconds=119 + 600 + 119 + 600 + 119)).strftime("%Y-%m-%d %H:%M:%S")
    assert tuple(actual) == ("2024-03-01 00:00:00", end, 119, 719), f"{description}: got {tuple(actual)!r}"


@pytest.mark.skipif(not utils_sys.HAS_LIBARCHIVE, reason="streaming metadata path needs libarchive")
def test_zip_member_without_extraction(tmp_path, monkeypatch):
    """Archive members stream via libarchive: extraction entry points must stay unused."""
    description = "zipped member equals loose result with extraction blocked"

    def _forbidden(*args, **kwargs):
        raise AssertionError("extraction must not run on the metadata path")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extract", _forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", _forbidden)
    name = _write(tmp_path / "arch.tsv", ["Time\tAx", *_rows(BASE, 5)])
    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(tmp_path / name, name)
    expected = extract_time_info_from_text_file(tmp_path, PurePosixPath(name))
    actual = extract_time_info_from_text_file(zpath, PurePosixPath(name))
    assert actual is not None, f"{description}: expected time info, got None"
    assert tuple(actual) == tuple(expected), f"{description}: {tuple(actual)!r} != {tuple(expected)!r}"
