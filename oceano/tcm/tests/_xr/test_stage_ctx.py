"""Tests for tcm.stage_ctx — context-var driven state tracking.

Verifies:
- Context variable set/get/clear lifecycle (including sublevel scope)
- StageContextFilter boundary marks: ## for stage, ### for sublevel
- WARNING+ prefix within a scope (no mark)
- Prefix format with conditional display of sub-indexes
- Snapshot tuple return (8 fields)
- Boundary record attributes (stage_fresh, boundary_msg)
"""

from __future__ import annotations

import logging

import pytest

from tcm.stage_ctx import (
    StageContextFilter,
    _build_prefix,
    _lf,
    clear,
    set_probe,
    set_stage,
    set_sublevel,
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
    # Also attach filter to tcm.stage_ctx (used by set_stage/set_sublevel details)
    _lf.addFilter(filt)
    with caplog.at_level(logging.DEBUG, logger="test_stage_ctx"), caplog.at_level(
        logging.DEBUG, logger="tcm.stage_ctx"
    ):
        yield logger
    _lf.removeFilter(filt)


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
        pid, pi, ci, np_, nc, sn, sname, sub = snapshot()
        assert pid == probe_id, f"probe_id mismatch: {pid!r} != {probe_id!r}"
        assert pi == probe_idx, f"probe_idx mismatch: {pi} != {probe_idx}"
        assert ci == cfg_idx, f"cfg_idx mismatch: {ci} != {cfg_idx}"
        assert np_ == n_probes, f"n_probes mismatch: {np_} != {n_probes}"
        assert nc == n_cfgs, f"n_cfgs mismatch: {nc} != {n_cfgs}"
        assert sn == 0, f"set_probe should clear stage_num: {sn}"
        assert sname == "", f"set_probe should clear stage_name: {sname!r}"
        assert sub == "", f"set_probe should clear sublevel: {sub!r}"

    def test_set_stage_stores_values(self):
        set_stage(3, "proc")
        _, _, _, _, _, sn, sname, sub = snapshot()
        assert sn == 3, f"stage_num mismatch: {sn} != 3"
        assert sname == "proc", f"stage_name mismatch: {sname!r} != 'proc'"
        assert sub == "", f"set_stage should clear sublevel: {sub!r}"

    def test_set_sublevel_stores_values(self):
        set_probe("i90", 1, 1, 1, 1)
        set_stage(1, "load")
        set_sublevel("read")
        _, _, _, _, _, _, _, sub = snapshot()
        assert sub == "read", f"sublevel mismatch: {sub!r} != 'read'"

    def test_clear_resets_all(self):
        set_probe("i90", 1, 2, 3, 4)
        set_stage(5, "TSV")
        set_sublevel("write")
        clear()
        result = snapshot()
        assert result == ("", 0, 0, 0, 0, 0, "", ""), f"clear() did not reset: {result=!r}"


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
                    stage_name="NC",
                ),
                "probe i67 2.3/2.5 stage 4 NC",
                id="multi-probe-multi-cfg",
            ),
            pytest.param(
                dict(stage_num=1, stage_name="load"),
                "",  # no probe_id → empty prefix (probe context required)
                id="no-probe-empty-prefix",
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

    def test_prefix_with_sublevel(self):
        """Sublevel appends '/ name' to prefix."""
        set_probe("i90", 1, 1, 1, 1)
        set_stage(1, "load")
        set_sublevel("read")
        assert _build_prefix() == "probe i90 stage 1 load / read"


# ── Logging filter ───────────────────────────────────────────────────────


class TestStageContextFilter:
    """Filter behaviour: boundary marks + WARNING+ prefix."""

    def test_info_after_stage_gets_mark(self, stage_logger, caplog):
        """First INFO after set_stage gets [## prefix] boundary mark."""
        set_probe("i90", 1, 1, 2, 1)
        set_stage(1, "load")
        stage_logger.info("Loading data")
        assert "[## probe i90 1/2 stage 1 load] Loading data" in caplog.text, (
            f"Boundary mark missing: {caplog.text!r}"
        )

    def test_sublevel_gets_triple_hash(self, stage_logger, caplog):
        """set_sublevel with details emits [### prefix / sub] mark on tcm.stage_ctx logger."""
        set_probe("i90", 1, 1, 1, 1)
        set_stage(1, "load")
        set_sublevel("read", "chunk %d/%d", 1, 3)
        assert "[### probe i90 stage 1 load / read] chunk 1/3" in caplog.text, (
            f"Sublevel mark missing: {caplog.text!r}"
        )

    def test_warning_within_scope_gets_plain_prefix(self, stage_logger, caplog):
        """WARNING within a scope (no fresh mark) gets [prefix] without ##."""
        set_probe("i90", 1, 1, 2, 1)
        set_stage(1, "load")
        # Consume the mark
        stage_logger.info("first record")
        # Now a warning within the same scope
        stage_logger.warning("Sparse region")
        assert "[probe i90 1/2 stage 1 load] Sparse region" in caplog.text, (
            f"WARNING prefix wrong: {caplog.text!r}"
        )

    def test_stage_prefix_attribute_always_set(self, stage_logger, caplog):
        """stage_prefix is set on every record regardless of level."""
        set_probe("i90", 1, 1, 1, 1)
        set_stage(2, "coefs")
        # Consume the boundary mark first
        stage_logger.info("coefs start")
        with caplog.at_level(logging.DEBUG):
            stage_logger.debug("debug msg")
        # Find record by original message (may have been prefixed)
        record = next(r for r in caplog.records if "debug msg" in r.getMessage())
        assert getattr(record, "stage_prefix", None) == "probe i90 stage 2 coefs", (
            f"stage_prefix attribute missing or wrong: {getattr(record, 'stage_prefix', None)!r}"
        )

    def test_boundary_record_attributes(self, stage_logger, caplog):
        """Boundary records have stage_fresh > 0 and boundary_msg set."""
        set_probe("i90", 1, 1, 1, 1)
        set_stage(1, "load")
        stage_logger.info("Loading @i90.TXT")
        record = caplog.records[-1]
        assert getattr(record, "stage_fresh", 0) == 2, (
            f"stage_fresh should be 2 for stage boundary: {getattr(record, 'stage_fresh', 0)}"
        )
        # boundary_msg is the pristine message (before prefix prepend)
        assert getattr(record, "boundary_msg", None) == "Loading @i90.TXT", (
            f"boundary_msg should be pristine: {getattr(record, 'boundary_msg', None)!r}"
        )

    def test_no_double_prefix_when_multiple_handlers(self, caplog):
        """_stage_ctx_done guard prevents double [prefix] on multi-handler loggers."""
        logger = logging.getLogger("test_double_prefix")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        filt = StageContextFilter()
        logger.addFilter(filt)
        for _ in range(2):
            h = logging.StreamHandler()
            h.setLevel(logging.DEBUG)
            logger.addHandler(h)
        set_probe("i90", 1, 1, 1, 1)
        set_stage(1, "load")
        with caplog.at_level(logging.WARNING, logger="test_double_prefix"):
            logger.warning("Test msg")
        assert caplog.text.count("[## probe i90 stage 1 load]") == 1, (
            f"Double prefix detected: {caplog.text!r}"
        )

    def test_combine_stage_prefix(self, stage_logger, caplog):
        """COMBINE stage (num=0) shows name without 'stage 0' prefix."""
        set_probe("i90", 1, 1, 1, 1)
        set_stage(0, "combine")
        stage_logger.warning("Combining probes")
        assert "[## probe i90 combine] Combining probes" in caplog.text, (
            f"COMBINE prefix wrong: {caplog.text!r}"
        )

    def test_set_stage_with_details_emits_record(self, stage_logger, caplog):
        """set_stage() with details emits the boundary record on tcm.stage_ctx logger."""
        set_probe("i90", 1, 1, 1, 1)
        set_stage(1, "load", "Loading %s", "@i90.TXT")
        assert "[## probe i90 stage 1 load] Loading @i90.TXT" in caplog.text, (
            f"set_stage details not logged: {caplog.text!r}"
        )
