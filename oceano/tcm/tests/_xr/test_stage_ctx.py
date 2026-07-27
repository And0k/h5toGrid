"""Tests for tcm.stage_ctx — context-var driven stage tracking.

Verifies:
- Context variable set/get/clear lifecycle
- StageContextFilter: WARNING+ gets [prefix] prepended, INFO/DEBUG stays clean
- Prefix format with conditional display of sub-indexes
- Snapshot tuple return
"""

from __future__ import annotations

import logging

import pytest

from tcm.stage_ctx import (
    StageContextFilter,
    _build_prefix,
    clear,
    set_probe,
    set_stage,
    snapshot,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_ctx():
    """Reset all context vars before and after each test."""
    clear()
    yield
    clear()


@pytest.fixture()
def stage_logger(caplog):
    """Create a logger with StageContextFilter attached for assertion."""
    logger = logging.getLogger("test_stage_ctx")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    filt = StageContextFilter()
    logger.addFilter(filt)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    with caplog.at_level(logging.DEBUG, logger="test_stage_ctx"):
        yield logger


# ── Context variable lifecycle ───────────────────────────────────────────


class TestContextVars:
    """Set/get/clear context var lifecycle."""

    @pytest.mark.parametrize(
        "probe_id, probe_idx, cfg_idx, n_probes, n_cfgs",
        [
            pytest.param("i90", 1, 1, 2, 3, id="first-probe-first-cfg"),
            pytest.param("i67", 2, 2, 2, 3, id="second-probe-second-cfg"),
        ],
    )
    def test_set_probe_stores_values(self, probe_id, probe_idx, cfg_idx, n_probes, n_cfgs):
        set_probe(probe_id, probe_idx, cfg_idx, n_probes, n_cfgs)
        pid, pi, ci, np_, nc, _, _ = snapshot()
        assert pid == probe_id, f"probe_id mismatch: {pid!r} != {probe_id!r}"
        assert pi == probe_idx, f"probe_idx mismatch: {pi} != {probe_idx}"
        assert ci == cfg_idx, f"cfg_idx mismatch: {ci} != {cfg_idx}"
        assert np_ == n_probes, f"n_probes mismatch: {np_} != {n_probes}"
        assert nc == n_cfgs, f"n_cfgs mismatch: {nc} != {n_cfgs}"

    def test_set_stage_stores_values(self):
        set_stage(3, "proc")
        _, _, _, _, _, sn, sname = snapshot()
        assert sn == 3, f"stage_num mismatch: {sn} != 3"
        assert sname == "proc", f"stage_name mismatch: {sname!r} != 'proc'"

    def test_clear_resets_all(self):
        set_probe("i90", 1, 2, 3, 4)
        set_stage(5, "TSV")
        clear()
        result = snapshot()
        assert result == ("", 0, 0, 0, 0, 0, ""), f"clear() did not reset: {result=!r}"


# ── Prefix format ────────────────────────────────────────────────────────


class TestBuildPrefix:
    """Prefix format with conditional sub-index display."""

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            pytest.param(
                dict(
                    probe_id="i90",
                    probe_idx=1,
                    cfg_idx=1,
                    n_probes=1,
                    n_cfgs=1,
                    stage_num=1,
                    stage_name="load",
                ),
                "probe i90 stage 1 load",
                id="single-probe-single-cfg",
            ),
            pytest.param(
                dict(
                    probe_id="i90",
                    probe_idx=1,
                    cfg_idx=1,
                    n_probes=2,
                    n_cfgs=1,
                    stage_num=2,
                    stage_name="coefs",
                ),
                "probe i90 1/2 stage 2 coefs",
                id="multi-probe-single-cfg",
            ),
            pytest.param(
                dict(
                    probe_id="i67",
                    probe_idx=2,
                    cfg_idx=3,
                    n_probes=2,
                    n_cfgs=5,
                    stage_num=4,
                    stage_name="NC 1/2",
                ),
                "probe i67 2.3/2.5 stage 4 NC 1/2",
                id="multi-probe-multi-cfg",
            ),
            pytest.param(
                dict(stage_num=1, stage_name="load"),
                "stage 1 load",
                id="no-probe-just-stage",
            ),
            pytest.param(dict(), "", id="empty-context"),
            pytest.param(
                dict(
                    probe_id="i90",
                    probe_idx=1,
                    cfg_idx=1,
                    n_probes=2,
                    n_cfgs=1,
                    stage_num=0,
                    stage_name="combine",
                ),
                "probe i90 1/2 combine",
                id="combine-stage-num-zero",
            ),
        ],
    )
    def test_prefix_format(self, kwargs, expected):
        set_probe(
            kwargs.get("probe_id", ""),
            kwargs.get("probe_idx", 0),
            kwargs.get("cfg_idx", 0),
            kwargs.get("n_probes", 0),
            kwargs.get("n_cfgs", 0),
        )
        set_stage(kwargs.get("stage_num", 0), kwargs.get("stage_name", ""))
        result = _build_prefix()
        assert result == expected, f"Prefix format: expected {expected!r}, got {result!r}"


# ── Logging filter ───────────────────────────────────────────────────────


class TestStageContextFilter:
    """Filter behaviour: WARNING+ gets prefix, INFO/DEBUG stays clean."""

    def test_info_message_stays_clean(self, stage_logger, caplog):
        set_probe("i90", 1, 1, 2, 1)
        set_stage(1, "load")
        stage_logger.info("Loading data")
        assert "[probe" not in caplog.text, f"INFO should not have prefix: {caplog.text!r}"
        assert "Loading data" in caplog.text

    def test_warning_gets_prefix(self, stage_logger, caplog):
        set_probe("i90", 1, 1, 2, 1)
        set_stage(1, "load")
        stage_logger.warning("Sparse region")
        assert "[probe i90 1/2 stage 1 load]" in caplog.text, f"WARNING should have prefix: {caplog.text!r}"
        assert "Sparse region" in caplog.text

    def test_error_gets_prefix(self, stage_logger, caplog):
        set_probe("i67", 2, 3, 3, 4)
        set_stage(3, "proc")
        stage_logger.error("Calculation failed")
        assert "[probe i67 2.3/3.4 stage 3 proc]" in caplog.text, f"ERROR should have prefix: {caplog.text!r}"

    def test_stage_prefix_attribute_always_set(self, stage_logger, caplog):
        """stage_prefix is set on every record regardless of level."""
        set_probe("i90", 1, 1, 1, 1)
        set_stage(2, "coefs")
        with caplog.at_level(logging.DEBUG):
            stage_logger.debug("debug msg")
        # Find the record
        record = next(r for r in caplog.records if r.getMessage() == "debug msg")
        assert getattr(record, "stage_prefix", None) == "probe i90 stage 2 coefs", (
            f"stage_prefix attribute missing or wrong: {getattr(record, 'stage_prefix', None)!r}"
        )

    def test_no_double_prefix_when_multiple_handlers(self, caplog):
        """Filter with _stage_prefixed guard prevents double [prefix] on multi-handler loggers."""
        logger = logging.getLogger("test_double_prefix")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        filt = StageContextFilter()
        logger.addFilter(filt)
        # Two handlers — both trigger filter()
        for _ in range(2):
            h = logging.StreamHandler()
            h.setLevel(logging.DEBUG)
            logger.addHandler(h)
        set_probe("i90", 1, 1, 1, 1)
        set_stage(1, "load")
        with caplog.at_level(logging.WARNING, logger="test_double_prefix"):
            logger.warning("Test msg")
        # Should have exactly one [prefix], not two
        assert caplog.text.count("[probe i90 stage 1 load]") == 1, (
            f"Double prefix detected: {caplog.text!r}"
        )

    def test_combine_stage_prefix(self, stage_logger, caplog):
        """COMBINE stage (num=0) shows name without 'stage 0' prefix."""
        set_probe("i90", 1, 1, 1, 1)
        set_stage(0, "combine")
        stage_logger.warning("Combining probes")
        assert "[probe i90 combine]" in caplog.text, (
            f"COMBINE prefix wrong: {caplog.text!r}"
        )
