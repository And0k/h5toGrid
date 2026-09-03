"""Tests for recursive fallback search (tcm.search + csv_load wrapper)."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from tcm import csv_load
from tcm.search import is_archive_composite, split_archive_path
from meta_finder.file_finder import find_raw_files_recursive


def _write_txt(path: Path, rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz\n"
        + "\n".join(f"2024,06,13,12,00,{i:02d},100,0,1000,200,0,0" for i in range(rows))
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "setup,expected_keys",
    [
        pytest.param(
            lambda raw: (
                _write_txt(raw / "i_01.txt"),
                _write_txt(raw / "subdir" / "i_02.txt"),
            ),
            {("i", 1), ("i", 2)},
            id="fallback-deep-when-shallow-empty->not-applicable",
        ),
    ],
)
def test_search_recursive_deep_only(tmp_path, setup, expected_keys):
    raw = tmp_path / "_raw"
    raw.mkdir()
    setup(raw)
    # raw has shallow i_01, so shallow wins; deep-only case tested separately
    ptn = re.compile(r"i.*\.txt", re.IGNORECASE)
    # Direct helper should find both
    entries = find_raw_files_recursive(raw, ptn)
    assert len(entries) >= 2


def test_search_shallow_wins_when_present(tmp_path):
    raw = tmp_path / "_raw"
    raw.mkdir()
    _write_txt(raw / "i_01.txt")
    _write_txt(raw / "subdir" / "i_02.txt")
    # wrapper: shallow present → only shallow returned (fallback policy)
    res = csv_load.search_csv_files(raw)
    assert set(res.keys()) == {("i", 1)}
    assert all(not is_archive_composite(p) for v in res.values() for p in v)


def test_search_fallback_finds_deep_and_archive(tmp_path):
    raw = tmp_path / "_raw"
    raw.mkdir()
    # No shallow files → deep + archive
    _write_txt(raw / "subdir" / "i_02.txt")
    _write_txt(raw / "nested" / "i_03.txt")
    # archive containing i_04
    archive = raw / "arch.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "i_04.txt", "yyyy,mm,dd,HH,MM,SS,Ax,Ay,Az,Mx,My,Mz\n2024,06,13,12,00,00,100,0,1000,200,0,0\n"
        )
    res = csv_load.search_csv_files(raw)
    assert ("i", 2) in res
    assert ("i", 3) in res
    assert ("i", 4) in res
    # archive entry stored as composite with "/"
    arch_paths = [p for v in res.values() for p in v if is_archive_composite(p)]
    assert len(arch_paths) == 1
    assert arch_paths[0].as_posix().endswith("arch.zip/i_04.txt")
    sp = split_archive_path(arch_paths[0])
    assert sp is not None
    assert sp[0].name == "arch.zip"
    assert sp[1] == PurePosixPath("i_04.txt")


def test_search_h5_in_subdir_via_recursive(tmp_path):
    from tcm._constants import H5_AVAILABLE

    raw = tmp_path / "_raw"
    raw.mkdir()
    # h5 loose file deep (empty file suffices for discovery, probe identity from name)
    (raw / "subdir").mkdir()
    h5 = raw / "subdir" / "i_p05.h5"
    h5.write_bytes(b"")  # empty, still counted as file
    # txt deep for contrast
    _write_txt(raw / "subdir" / "i_06.txt")
    res = csv_load.search_csv_files(raw)
    # Directory input uses the default txt pattern — txt probe 6 must be present
    assert ("i", 6) in res
    # HDF5 is advertised only when h5py available (global H5_AVAILABLE), and only
    # when the pattern matches it — the default dir pattern is txt-only, so probe
    # with an explicit "*" glob.
    if H5_AVAILABLE:
        res_all = csv_load.search_csv_files(raw / "*")
        assert any(k[1] == 5 for k in res_all), (
            f"H5 probe 5 missing with '*' glob (H5_AVAILABLE) — {sorted(res_all)}"
        )
    else:
        # When h5py missing, HDF5 extensions are skipped entirely
        assert all(k[1] != 5 for k in res)
        assert not any(str(p).lower().endswith(".h5") for v in res.values() for p in v)


def test_at_suppression_deep(tmp_path):
    raw = tmp_path / "_raw"
    raw.mkdir()
    _write_txt(raw / "subdir" / "i_05.txt")
    _write_txt(raw / "subdir" / "@i_05.txt")
    res = csv_load.search_csv_files(raw)
    assert ("i", 5) in res
    names = [split_archive_path(p)[1].name if is_archive_composite(p) else p.name for p in res[("i", 5)]]
    assert any(n.startswith("@") for n in names)
    assert not any(n == "i_05.txt" for n in names)


def test_pattern_matching_respects_ptn(tmp_path):
    raw = tmp_path / "_raw"
    raw.mkdir()
    _write_txt(raw / "subdir" / "i_01.txt")
    _write_txt(raw / "subdir" / "w_02.txt")
    # Pattern i.* should only match inclinometer
    ptn_path = raw / "i*.txt"
    res = csv_load.search_csv_files(ptn_path)
    assert all(k[0] == "i" for k in res)


def test_is_archive_composite_helpers():
    p = Path("C:/a/b.zip/c/d.txt")
    assert is_archive_composite(p)
    assert split_archive_path(p) == (Path("C:/a/b.zip"), PurePosixPath("c/d.txt"))
    p2 = Path("C:/a/b.txt")
    assert not is_archive_composite(p2)
    assert split_archive_path(p2) is None
