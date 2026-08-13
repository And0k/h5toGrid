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
    _cv_work_done,
    _cv_work_total,
    _lf,
    advance,
    clear,
    set_probe,
    set_stage,
    set_stage_plan,
    set_sublevel,
    set_work,
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
    with (
        caplog.at_level(logging.DEBUG, logger="test_stage_ctx"),
        caplog.at_level(logging.DEBUG, logger="tcm.stage_ctx"),
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


# ── Intra-stage sub-step progress ────────────────────────────────────────


class TestAdvance:
    """set_work / advance — generic intra-stage progress for any stage."""

    def test_set_work_stores_total_and_resets_done(self):
        """set_work sets total and resets done counter to 0."""
        set_work(5)
        assert _cv_work_total.get() == 5, "total not stored"
        assert _cv_work_done.get() == 0, "done not reset"

    def test_advance_increments_done(self):
        """Each advance() increments the done counter."""
        set_work(3)
        advance()
        assert _cv_work_done.get() == 1, "done should be 1 after first advance"
        advance()
        assert _cv_work_done.get() == 2, "done should be 2 after second advance"

    def test_advance_noop_when_no_work(self):
        """advance() is a no-op when total is 0 (no set_work called)."""
        set_work(0)
        advance()
        assert _cv_work_done.get() == 0, "done should stay 0 when total=0"

    def test_clear_resets_work_vars(self):
        """clear() resets work_total and work_done."""
        set_work(10)
        advance()
        clear()
        assert _cv_work_total.get() == 0, "total not cleared"
        assert _cv_work_done.get() == 0, "done not cleared"

    @pytest.mark.parametrize(
        "n_work, n_active, expected_fracs",
        [
            pytest.param(4, 4, [6, 12, 19, 25], id="4-work-4-stages"),
            pytest.param(5, 4, [5, 10, 15, 20, 25], id="5-work-4-stages"),
            pytest.param(1, 4, [25], id="1-work-jumps-to-stage-end"),
            pytest.param(3, 3, [11, 22, 33], id="3-work-3-stages"),
        ],
    )
    def test_advance_advances_overall_fractionally(self, n_work, n_active, expected_fracs):
        """advance() moves overall progress proportionally within the current stage.

        With n_active stages and tick_idx=0 (Load), each sub-step moves
        round(done * 100 / n_active / work_total).  For later stages
        tick_base = tick_idx * stage_frac shifts the base.
        """
        from types import SimpleNamespace

        from tcm_gui import progress_bridge as pb

        captured = []
        mock_overall = SimpleNamespace(set=lambda c, t, d: captured.append(("overall", c, t, d)))
        mock_stage = SimpleNamespace(set=lambda c, t, d: captured.append(("stage", c, t, d)))
        mock_bank = SimpleNamespace(inner=lambda cfg, c, t: captured.append(("bank", cfg, c, t)))
        mock_rt = SimpleNamespace(
            progress_overall=mock_overall,
            progress_stage=mock_stage,
            progress_bank=mock_bank,
        )
        old_rt = pb.get_runtime()
        old_cfg = pb.get_cfg()
        pb.set_runtime(mock_rt)
        pb.set_cfg("test_cfg")
        try:
            set_probe("i90", 1, 1, 1, 1, stem_idx=1, n_cfgs_total=1)
            set_stage_plan(n_active)
            set_work(n_work)
            for expected_frac in expected_fracs:
                captured.clear()
                advance()
                overall_ticks = [c for c in captured if c[0] == "overall"]
                assert overall_ticks, f"No overall tick captured for frac={expected_frac}"
                _, cur, tot, _ = overall_ticks[-1]
                assert cur == expected_frac, (
                    f"Expected overall={expected_frac}, got {cur}. n_work={n_work}, n_active={n_active}"
                )
                assert tot == 100, f"Total should be 100 (probe_total), got {tot}"
        finally:
            pb.set_runtime(old_rt)
            pb.set_cfg(old_cfg)

    @pytest.mark.parametrize(
        "tick_idx, n_work, n_active, expected_first",
        [
            pytest.param(2, 2, 4, 62, id="tick-idx-2-first-work-at-62"),
            pytest.param(1, 4, 4, 31, id="tick-idx-1-first-work-at-31"),
            pytest.param(3, 2, 6, 58, id="tick-idx-3-first-work-at-58"),
        ],
    )
    def test_advance_uses_tick_base_for_later_stages(self, tick_idx, n_work, n_active, expected_first):
        """advance() adds tick_idx * stage_frac as base for non-first stages.

        Formula: probe_base + tick_idx * (100/n_active) + done * (100/n_active) / n_work.
        """
        from types import SimpleNamespace

        from tcm_gui import progress_bridge as pb

        captured = []
        mock_rt = SimpleNamespace(
            progress_overall=SimpleNamespace(set=lambda c, t, d: captured.append(c)),
            progress_stage=SimpleNamespace(set=lambda *_: None),
            progress_bank=SimpleNamespace(inner=lambda *_: None, stage_start=lambda *_: None),
        )
        old_rt = pb.get_runtime()
        old_cfg = pb.get_cfg()
        pb.set_runtime(mock_rt)
        pb.set_cfg("test_cfg")
        try:
            set_probe("i90", 1, 1, 1, 1, stem_idx=1, n_cfgs_total=1)
            set_stage_plan(n_active)
            from tcm.stage_ctx import _cv_tick_idx

            _cv_tick_idx.set(tick_idx)
            set_work(n_work)
            captured.clear()
            advance()
            assert captured[-1] == expected_first, (
                f"tick_idx={tick_idx}: expected first advance at {expected_first}, got {captured[-1]}"
            )
        finally:
            pb.set_runtime(old_rt)
            pb.set_cfg(old_cfg)

    def test_advance_updates_progress_bank_inner(self):
        """advance() updates the per-config progress_bank.inner."""
        from types import SimpleNamespace

        from tcm_gui import progress_bridge as pb

        bank_calls = []
        mock_bank = SimpleNamespace(inner=lambda cfg, c, t: bank_calls.append((cfg, c, t)))
        mock_rt = SimpleNamespace(
            progress_overall=SimpleNamespace(set=lambda *_: None),
            progress_stage=SimpleNamespace(set=lambda *_: None),
            progress_bank=mock_bank,
        )
        old_rt = pb.get_runtime()
        old_cfg = pb.get_cfg()
        pb.set_runtime(mock_rt)
        pb.set_cfg("i_01")
        try:
            set_probe("i90", 1, 1, 1, 1, stem_idx=1, n_cfgs_total=1)
            set_stage_plan(4)
            set_work(3)
            advance()
            assert bank_calls == [("i_01", 1, 3)], f"Bank calls: {bank_calls}"
            advance()
            assert bank_calls[-1] == ("i_01", 2, 3), f"Bank calls after 2nd advance: {bank_calls}"
        finally:
            pb.set_runtime(old_rt)
            pb.set_cfg(old_cfg)

    def test_advance_noop_when_no_pb(self):
        """advance() is a no-op when progress_bridge is unavailable."""
        from tcm import stage_ctx as sc

        old_pb = sc._pb
        sc._pb = None
        try:
            set_work(5)
            advance()
            assert _cv_work_done.get() == 0, "Should not increment without _pb"
        finally:
            sc._pb = old_pb


# ── Tick: last-probe progress reaches 100% ──────────────────────────────


def _mock_progress_bridge():
    """Build a mock progress_bridge runtime capturing overall set() calls.

    Returns ``(mock_rt, overall_snapshots)`` where *overall_snapshots* is a list
    that each ``progress_overall.set(cur, tot, desc)`` appends ``(cur, tot)`` to.
    The mock ``progress_overall`` also provides ``snapshot()`` returning the last
    set values — ``progress_bridge.stage_desc`` reads it on each tick boundary.
    The caller is responsible for restoring ``pb.set_runtime`` / ``pb.set_cfg``.
    """
    from types import SimpleNamespace

    overall_snaps: list[tuple[int, int]] = []
    _state = [0, 0, ""]  # cur, tot, desc — mutable closure for snapshot

    def _set(c, t, d):
        _state[0], _state[1], _state[2] = c, t, d
        overall_snaps.append((c, t))

    mock_rt = SimpleNamespace(
        progress_overall=SimpleNamespace(set=_set, snapshot=lambda: tuple(_state)),
        progress_stage=SimpleNamespace(set=lambda *_: None),
        progress_bank=SimpleNamespace(inner=lambda *_: None, stage_start=lambda *_: None),
    )
    return mock_rt, overall_snaps


class TestTickProgressBarCompletion:
    """Verify the overall progress bar reaches 100% on the last probe.

    The stage plan ``n_active`` must equal the count of ticks that actually
    fire: NC for every bin (incl. noAvg), TSV only for bins ≥ dt_min_save.
    When the plan matches actual ticks, the final tick of the last probe
    yields ``frac=100`` and the per-config rail fill completes.
    """

    @staticmethod
    def _simulate_probe(stem_idx: int, n_cfgs: int, n_active: int, n_nc_ticks: int, n_tsv_ticks: int):
        """Drive one probe through LOAD → COEFS → PROC → NC×n_nc → TSV×n_tsv.

        Mirrors ``run_processing`` + ``_process_and_persist`` tick flow:
        tick(LOAD) → tick(COEFS) → tick(PROC) → tick(NC)×n_nc → tick(TSV)×n_tsv.
        Returns the last overall ``cur`` captured.
        """
        from tcm import stage_ctx as sc
        from tcm.states import Stage

        set_probe(f"p{stem_idx}", stem_idx, 1, n_cfgs, 1, stem_idx=stem_idx, n_cfgs_total=n_cfgs)
        set_stage_plan(n_active)
        sc.tick()  # LOAD
        sc.tick()  # COEFS
        sc.tick(Stage.PROC)
        for _ in range(n_nc_ticks):
            sc.tick(Stage.NC)
        for _ in range(n_tsv_ticks):
            sc.tick(Stage.TSV)

    @pytest.mark.parametrize(
        "n_cfgs, n_nc_tick_per_probe, n_tsv_tick_per_probe, n_active_per_probe",
        [
            # noh5: 3 fixed + 0 NC + 4 TSV = 7 active, TSV fires 4 → reaches 100%
            pytest.param(2, 0, 4, 7, id="noh5-2probes-4tsv"),
            # h5 fixed: 3 + 5 NC + 4 TSV = 12 active (noAvg not counted in TSV),
            # TSV fires 4 (noAvg skipped: dt_bin=0 < dt_min_save=1s) → reaches 100%
            pytest.param(2, 5, 4, 12, id="h5-fixed-2probes-5nc-4tsv"),
            # single probe h5 fixed
            pytest.param(1, 5, 4, 12, id="h5-fixed-1probe"),
        ],
    )
    def test_last_probe_reaches_100_percent(
        self, n_cfgs, n_nc_tick_per_probe, n_tsv_tick_per_probe, n_active_per_probe
    ):
        """Last probe's final tick must set overall to probe_base + 100."""
        from tcm_gui import progress_bridge as pb

        mock_rt, snaps = _mock_progress_bridge()
        old_rt, old_cfg = pb.get_runtime(), pb.get_cfg()
        pb.set_runtime(mock_rt), pb.set_cfg("test")
        try:
            for stem_idx in range(1, n_cfgs + 1):
                snaps.clear()
                self._simulate_probe(
                    stem_idx, n_cfgs, n_active_per_probe, n_nc_tick_per_probe, n_tsv_tick_per_probe
                )
                expected_base = (stem_idx - 1) * 100
                final_cur, final_tot = snaps[-1]
                assert final_cur == expected_base + 100, (
                    f"probe {stem_idx}/{n_cfgs}: overall={final_cur} expected {expected_base + 100} "
                    f"(n_active={n_active_per_probe}, nc={n_nc_tick_per_probe}, tsv={n_tsv_tick_per_probe})"
                )
                assert final_tot == n_cfgs * 100, f"total={final_tot} expected {n_cfgs * 100}"
        finally:
            pb.set_runtime(old_rt), pb.set_cfg(old_cfg)

    def test_overcounted_plan_does_not_reach_100(self):
        """Regression guard: an overcounted plan (more stages than ticks) stalls below 100%.

        Proves that matching ``n_active`` to actual tick count is necessary:
        with 13 planned stages but only 12 ticks (noAvg skipped by TSV),
        ``round(12*100/13) = 92`` ≠ 100. The fix prevents this by counting
        only TSV-eligible bins in the stage plan.
        """
        from tcm_gui import progress_bridge as pb

        mock_rt, snaps = _mock_progress_bridge()
        old_rt, old_cfg = pb.get_runtime(), pb.get_cfg()
        pb.set_runtime(mock_rt), pb.set_cfg("test")
        try:
            # 13 active stages, but only 12 ticks fire (noAvg skipped by TSV)
            self._simulate_probe(stem_idx=1, n_cfgs=1, n_active=13, n_nc_ticks=5, n_tsv_ticks=4)
            final_cur, _ = snaps[-1]
            assert final_cur == 92, f"Overcounted plan: expected 92 (round(12*100/13)), got {final_cur}"
            assert final_cur != 100, "Overcounted plan must not reach 100 — fix counts only TSV-eligible bins"
        finally:
            pb.set_runtime(old_rt), pb.set_cfg(old_cfg)
