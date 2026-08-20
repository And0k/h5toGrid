"""Tests for tcm_gui.progress_bridge — GuiTqdm ↔ ProgressState connection.

Verifies the lower (stage) progress bar data path:
  TqdmCallback → GuiTqdm → ProgressState → App._poll_progress → ttk.Progressbar
"""

from __future__ import annotations

import pytest

from tcm_gui.progress_bank import ProgressBank, canon_stage, WEIGHTS
from tcm_gui.progress_bridge import (
    GuiTqdm,
    get_cfg,
    get_runtime,
    get_tqdm_class,
    set_cfg,
    set_runtime,
    set_tqdm_class,
    stage_desc,
)
from tcm_gui.runtime import ProgressState, Runtime

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def rt() -> Runtime:
    """Fresh Runtime with separate progress_stage/progress_overall."""
    return Runtime()


@pytest.fixture(autouse=True)
def _clean_bridge():
    """Reset module-level bridge state before/after each test."""
    set_runtime(None)
    set_tqdm_class(None)
    yield
    set_runtime(None)
    set_tqdm_class(None)


# --------------------------------------------------------------------------- #
# GuiTqdm: instantiation
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestGuiTqdmInit:
    """GuiTqdm must accept total as int (from TqdmCallback._start_state)."""

    def test_init_with_int_total(self, rt):
        """TqdmCallback passes total=<int> — GuiTqdm must not crash."""
        set_runtime(rt)
        bar = GuiTqdm(total=42, desc="test")
        assert bar.total == 42
        assert bar.n == 0
        assert bar.desc == "test"

    def test_init_with_none_total(self, rt):
        """total=None defaults to 0."""
        set_runtime(rt)
        bar = GuiTqdm(total=None)
        assert bar.total == 0

    def test_init_with_zero_total(self, rt):
        """total=0 → self.total = 0 (falsy fallback)."""
        set_runtime(rt)
        bar = GuiTqdm(total=0)
        assert bar.total == 0

    def test_init_without_runtime(self):
        """No runtime set → _ps and _gate are None (no-op mode)."""
        bar = GuiTqdm(total=10)
        assert bar._ps is None
        assert bar._gate is None
        assert bar.total == 10

    def test_init_connects_to_progress_stage(self, rt):
        """With runtime, GuiTqdm._ps points to rt.progress_stage."""
        set_runtime(rt)
        bar = GuiTqdm(total=10)
        assert bar._ps is rt.progress_stage
        assert bar._gate is rt.pause_gate


# --------------------------------------------------------------------------- #
# GuiTqdm: update → ProgressState
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestGuiTqdmUpdate:
    """GuiTqdm.update writes (n, total, desc) to progress_stage."""

    def test_update_increments_n(self, rt):
        set_runtime(rt)
        bar = GuiTqdm(total=10, desc="binning")
        bar.update(1)
        assert bar.n == 1
        bar.update(3)
        assert bar.n == 4

    def test_update_writes_to_progress_stage(self, rt):
        set_runtime(rt)
        bar = GuiTqdm(total=10, desc="NC write")
        bar.update(1)
        cur, tot, desc = rt.progress_stage.snapshot()
        assert cur == 1
        assert tot == 10
        assert desc == "NC write"

    def test_update_accumulates(self, rt):
        set_runtime(rt)
        bar = GuiTqdm(total=100, desc="dask tasks")
        for _ in range(5):
            bar.update(1)
        cur, tot, _desc = rt.progress_stage.snapshot()
        assert cur == 5
        assert tot == 100

    def test_update_without_runtime_is_noop(self):
        """No runtime → update runs without error, no state change."""
        bar = GuiTqdm(total=10)
        bar.update(1)  # must not raise
        assert bar.n == 1

    def test_update_respects_pause_gate(self, rt):
        """update blocks while gate is paused, resumes on gate.resume."""
        import threading
        import time

        set_runtime(rt)
        rt.pause_gate.pause()
        bar = GuiTqdm(total=10)
        reached = threading.Event()

        def _update_in_thread():
            bar.update(1)
            reached.set()

        t = threading.Thread(target=_update_in_thread)
        t.start()
        time.sleep(0.1)
        assert not reached.is_set(), "update should block while paused"
        rt.pause_gate.resume()
        t.join(timeout=2)
        assert reached.is_set(), "update should unblock after resume"


# --------------------------------------------------------------------------- #
# GuiTqdm: close → ProgressState set to completion
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestGuiTqdmClose:
    """GuiTqdm.close sets progress_stage to (total, total, desc)."""

    def test_close_sets_completion(self, rt):
        set_runtime(rt)
        bar = GuiTqdm(total=50, desc="final")
        bar.update(25)
        bar.close()
        cur, tot, desc = rt.progress_stage.snapshot()
        assert cur == 50
        assert tot == 50
        assert desc == "final"

    def test_close_without_runtime_is_noop(self):
        bar = GuiTqdm(total=10)
        bar.close()  # must not raise


# --------------------------------------------------------------------------- #
# GuiTqdm: tqdm compatibility methods
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestGuiTqdmCompat:
    """GuiTqdm satisfies tqdm duck-type for TqdmCallback."""

    def test_context_manager_enter_returns_self(self, rt):
        set_runtime(rt)
        with GuiTqdm(total=10) as bar:
            assert bar is not None
            bar.update(1)
        # __exit__ is a no-op (tqdm compat); close() must be called explicitly

    def test_set_description(self, rt):
        set_runtime(rt)
        bar = GuiTqdm(total=10, desc="initial")
        bar.set_description("updated")
        assert bar.desc == "updated"

    def test_set_postfix_noop(self, rt):
        set_runtime(rt)
        bar = GuiTqdm(total=10)
        bar.set_postfix(loss=0.5)  # must not raise

    def test_set_postfix_str_noop(self, rt):
        set_runtime(rt)
        bar = GuiTqdm(total=10)
        bar.set_postfix_str("loading bins")  # must not raise

    def test_refresh_noop(self, rt):
        set_runtime(rt)
        bar = GuiTqdm(total=10)
        bar.refresh()  # must not raise


# --------------------------------------------------------------------------- #
# GuiTqdm: iterable wrapper (binning loop)
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestGuiTqdmIterable:
    """GuiTqdm wraps an iterable like tqdm — for physical.py binning loop."""

    def test_iterate_with_len(self, rt):
        """GuiTqdm(list, ...) infers total from len and yields items."""
        set_runtime(rt)
        items = [10, 20, 30]
        result = list(GuiTqdm(items, desc="bins"))
        assert result == [10, 20, 30]
        cur, tot, desc = rt.progress_stage.snapshot()
        assert tot == 3
        assert cur == 3
        assert desc == "bins"

    def test_iterate_with_range(self, rt):
        """GuiTqdm(range(...)) works as iterable (range has len)."""
        set_runtime(rt)
        result = list(GuiTqdm(range(5), desc="chunks"))
        assert result == [0, 1, 2, 3, 4]
        cur, tot, _ = rt.progress_stage.snapshot()
        assert tot == 5
        assert cur == 5

    def test_iterate_without_runtime(self):
        """No runtime → iteration still works (just no progress state)."""
        result = list(GuiTqdm([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_iterate_updates_per_item(self, rt):
        """Each yielded item increments progress_stage."""
        set_runtime(rt)
        bar = GuiTqdm([0, 1, 2, 3, 4], desc="bins")
        it = iter(bar)
        next(it)  # item 0
        cur, _, _ = rt.progress_stage.snapshot()
        assert cur == 0  # update happens AFTER yield
        next(it)  # item 1 → update(1)
        cur, _, _ = rt.progress_stage.snapshot()
        assert cur == 1
        next(it)  # item 2 → update(1)
        cur, _, _ = rt.progress_stage.snapshot()
        assert cur == 2

    def test_explicit_total_overrides_len(self, rt):
        """total= overrides len(iterable) when given."""
        set_runtime(rt)
        bar = GuiTqdm([1, 2, 3], total=100, desc="custom")
        assert bar.total == 100


# --------------------------------------------------------------------------- #
# TqdmCallback integration
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestTqdmCallbackIntegration:
    """TqdmCallback(tqdm_class=GuiTqdm) creates GuiTqdm and drives it."""

    def test_tqdm_callback_creates_guitqdm(self, rt, mocker):
        """TqdmCallback with tqdm_class=GuiTqdm instantiates correctly."""
        from tqdm.dask import TqdmCallback

        set_runtime(rt)
        set_tqdm_class(GuiTqdm)

        cb = TqdmCallback(
            desc="test NC",
            leave=False,
            tqdm_class=get_tqdm_class(),
        )
        # Simulate what dask does: _start_state with a fake state
        state = {"ready": [1, 2, 3], "waiting": [], "running": [], "finished": []}
        cb._start_state(None, state)

        assert isinstance(cb.pbar, GuiTqdm)
        assert cb.pbar.total == 3
        assert cb.pbar.desc == "test NC"

    def test_tqdm_callback_posttask_updates_progress(self, rt):
        """TqdmCallback._posttask increments progress_stage."""
        from tqdm.dask import TqdmCallback

        set_runtime(rt)
        cb = TqdmCallback(tqdm_class=GuiTqdm)
        state = {"ready": [1, 2], "waiting": [], "running": [], "finished": []}
        cb._start_state(None, state)

        cb._posttask()
        cur, tot, _ = rt.progress_stage.snapshot()
        assert cur == 1
        assert tot == 2

        cb._posttask()
        cur, tot, _ = rt.progress_stage.snapshot()
        assert cur == 2

    def test_tqdm_callback_finish_sets_completion(self, rt):
        """TqdmCallback._finish closes the bar (progress at total)."""
        from tqdm.dask import TqdmCallback

        set_runtime(rt)
        cb = TqdmCallback(tqdm_class=GuiTqdm, desc="NC write")
        state = {"ready": [1, 2, 3], "waiting": [], "running": [], "finished": []}
        cb._start_state(None, state)
        cb._posttask()

        cb._finish()
        cur, tot, desc = rt.progress_stage.snapshot()
        assert cur == 3
        assert tot == 3
        assert desc == "NC write"


# --------------------------------------------------------------------------- #
# set_runtime / get_runtime
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestModuleLevelRuntime:
    """Module-level _runtime is visible across threads (same module object)."""

    def test_set_get_roundtrip(self):
        rt = Runtime()
        set_runtime(rt)
        assert get_runtime() is rt

    def test_default_is_none(self):
        assert get_runtime() is None

    def test_set_tqdm_class_roundtrip(self):
        set_tqdm_class(GuiTqdm)
        assert get_tqdm_class() is GuiTqdm

    def test_default_tqdm_class_is_none(self):
        assert get_tqdm_class() is None


# --------------------------------------------------------------------------- #
# ProgressState (unit)
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestProgressState:
    """ProgressState.set/snapshot are thread-safe and correct."""

    def test_set_and_snapshot(self):
        ps = ProgressState()
        ps.set(5, 100, "loading")
        cur, tot, desc = ps.snapshot()
        assert cur == 5
        assert tot == 100
        assert desc == "loading"

    def test_snapshot_default(self):
        ps = ProgressState()
        cur, tot, desc = ps.snapshot()
        assert cur == 0
        assert tot == 0
        assert desc == ""

    def test_overwrite(self):
        ps = ProgressState()
        ps.set(1, 10, "a")
        ps.set(2, 20, "b")
        cur, tot, desc = ps.snapshot()
        assert cur == 2
        assert tot == 20
        assert desc == "b"


# --------------------------------------------------------------------------- #
# App._poll_progress data path (without Tk)
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestPollProgressLogic:
    """Verify the _poll_progress decision logic against ProgressState values."""

    def test_stage_bar_updates_when_tot_positive(self, rt):
        """When progress_stage has tot > 0, bar should update."""
        set_runtime(rt)
        bar = GuiTqdm(total=100, desc="dask")
        bar.update(50)

        cur, tot, desc = rt.progress_stage.snapshot()
        assert tot > 0
        assert cur == 50
        assert desc == "dask"

    def test_stage_bar_ignored_when_tot_zero(self, rt):
        """When progress_stage has tot == 0 (no bar), poll should skip update."""
        _cur, tot, _desc = rt.progress_stage.snapshot()
        assert tot == 0
        # _poll_progress would skip: `if tot > 0:` → False

    def test_overall_bar_ticks_progress(self, rt):
        """_stage()-style update on progress_overall is visible in snapshot."""
        rt.progress_overall.set(50, 300, "i_01 proc")
        cur, tot, desc = rt.progress_overall.snapshot()
        assert cur == 50
        assert tot == 300
        assert desc == "i_01 proc"

    def test_stage_inactive_never_touches_status_without_clear_flag(self, rt):
        """When stage=0/0 and no clear flag, _poll_progress does NOT touch _status.

        Covers idle (no run) and post-completion — "Ready" / "Done …" persists.
        """
        rt.progress_stage.set(0, 0, "")
        rt.progress_overall.set(0, 0, "")
        _cur, tot, _desc = rt.progress_stage.snapshot()
        _cur_o, tot_o, _desc_o = rt.progress_overall.snapshot()
        assert tot == 0 and tot_o == 0
        assert not rt.progress_stage.consume_clear(), "no clear signal → status untouched"

    def test_stage_inactive_with_clear_flag_clears_status_once(self, rt):
        """clear_and_reset signals one-shot status wipe; consume_clear fires once.

        Simulates probe boundary: worker calls clear_and_reset(), poll consumes it.
        """
        rt.progress_stage.clear_and_reset()
        cur, tot, desc = rt.progress_stage.snapshot()
        assert (cur, tot, desc) == (0, 0, ""), "clear_and_reset resets to idle"
        assert rt.progress_stage.consume_clear(), "first consume returns True"
        assert not rt.progress_stage.consume_clear(), "second consume returns False (one-shot)"


# --------------------------------------------------------------------------- #
# canon_stage mapping
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestCanonStage:
    """canon_stage maps pipeline Stage enum values to bank canonical stages."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("load", "Load"),
            ("coefs", "Prepare"),
            ("proc", "Processing"),
            ("NC", "Save"),
            ("TSV", "Save"),
            ("combine", "Save"),
            ("Saving results", ""),
            ("Cleanup phase", "Cleanup"),
            ("Finished", "Finished"),
            ("unknown_xyz", ""),
        ],
        ids=[
            "Stage.LOAD",
            "Stage.COEFS via _ALIASES",
            "Stage.PROC",
            "Stage.NC via _ALIASES",
            "Stage.TSV via _ALIASES",
            "Stage.COMBINE via _ALIASES",
            "free-form prefix miss (savi != save)",
            "4-letter prefix",
            "exact match",
            "no match -> empty",
        ],
    )
    def test_canon_stage_maps_correctly(self, text, expected):
        result = canon_stage(text)
        assert result == expected, f"canon_stage({text!r}): expected {expected!r}, got {result!r}"


# --------------------------------------------------------------------------- #
# ProgressBank lifecycle
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestProgressBank:
    """ProgressBank tracks per-config progress through stages → frac."""

    def test_pending_starts_at_zero(self):
        b = ProgressBank()
        b.run_start(["s1"])
        snap = b.snapshot_all()
        state, frac, stage, lvl = snap["s1"]
        assert state == "pending"
        assert frac == 0.0

    def test_running_frac_advances_through_stages(self):
        b = ProgressBank()
        b.run_start(["s1"])

        b.stage_start("s1", canon_stage("load"))
        b.inner("s1", 100, 100)
        snap = b.snapshot_all()
        _, frac_load, _, _ = snap["s1"]
        assert 0 < frac_load < 0.2, "Load stage should be ~10%"

        b.stage_start("s1", canon_stage("proc"))
        b.inner("s1", 100, 100)
        snap = b.snapshot_all()
        _, frac_proc, _, _ = snap["s1"]
        assert frac_proc > frac_load, "Processing > Load"
        assert frac_proc < 1.0, "running < 1.0 until finish"

    def test_finish_sets_done_and_frac_one(self):
        b = ProgressBank()
        b.run_start(["s1"])
        b.stage_start("s1", canon_stage("proc"))
        b.inner("s1", 50, 100)
        b.finish("s1", ok=True)

        snap = b.snapshot_all()
        state, frac, stage, lvl = snap["s1"]
        assert state == "done"
        assert frac == 1.0
        assert stage == "Finished"

    def test_finish_error_preserves_last_frac(self):
        b = ProgressBank()
        b.run_start(["s1"])
        b.stage_start("s1", canon_stage("proc"))
        b.inner("s1", 50, 100)
        b.finish("s1", ok=False)

        snap = b.snapshot_all()
        state, frac, stage, _ = snap["s1"]
        assert state == "error"
        assert 0 < frac < 1.0, "error preserves running frac, not 1.0"
        assert stage != "Finished"

    def test_finish_wrong_key_is_noop(self):
        """finish with a key not in run_start is silently ignored."""
        b = ProgressBank()
        b.run_start(["stem_A"])
        b.stage_start("stem_A", canon_stage("load"))
        b.inner("stem_A", 100, 100)
        b.finish("wrong_key", ok=True)  # should not find stem_A

        snap = b.snapshot_all()
        state, frac, _, _ = snap["stem_A"]
        assert state == "running", "wrong key → still running"
        assert frac < 1.0

    def test_stage_start_resets_inner(self):
        b = ProgressBank()
        b.run_start(["s1"])
        b.stage_start("s1", canon_stage("load"))
        b.inner("s1", 50, 100)
        snap1 = b.snapshot_all()
        _, frac1, _, _ = snap1["s1"]
        assert frac1 > 0

        b.stage_start("s1", canon_stage("coefs"))
        snap2 = b.snapshot_all()
        _, frac2, stage2, _ = snap2["s1"]
        assert stage2 == "Prepare"
        # inner reset → frac should not increase (coefs weight < load+partial inner)
        assert frac2 < frac1 or frac2 < 0.2

    def test_inner_noop_when_pending(self):
        b = ProgressBank()
        b.run_start(["s1"])
        b.inner("s1", 100, 100)  # no stage_start → still pending
        snap = b.snapshot_all()
        state, frac, _, _ = snap["s1"]
        assert state == "pending"
        assert frac == 0.0

    def test_inner_noop_when_cfg_none(self):
        b = ProgressBank()
        b.run_start(["s1"])
        b.inner(None, 100, 100)  # should not raise

    def test_stage_start_after_finish_is_noop(self):
        """Regression: h5 combine re-runs the last config's done cell.

        ``bank.finish(stem)`` fires per-config right after each probe, but the
        post-loop combine stage still carries the last stem's attribution —
        its ``stage_desc("combine")`` used to flip the done cell back to
        running (fill regressed ~1.0 → 0.8), stalling the last config's bar.
        Terminal states must be final.
        """
        b = ProgressBank()
        b.run_start(["s1", "s2"])
        b.stage_start("s2", canon_stage("proc"))
        b.inner("s2", 100, 100)
        b.finish("s2", ok=True)

        # Post-loop phase attributed to the last stem — must not re-run it.
        b.stage_start("s2", canon_stage("combine"))
        b.inner("s2", 1, 4)  # inner is already state-guarded; belt and braces

        state, frac, stage, _ = b.snapshot_all()["s2"]
        assert state == "done", "finish is terminal — combine must not re-run"
        assert frac == 1.0
        assert stage == "Finished"

    def test_multiple_configs_independent(self):
        b = ProgressBank()
        b.run_start(["s1", "s2"])
        b.stage_start("s1", canon_stage("proc"))
        b.inner("s1", 100, 100)
        b.finish("s1", ok=True)

        b.stage_start("s2", canon_stage("load"))
        b.inner("s2", 50, 100)

        snap = b.snapshot_all()
        assert snap["s1"][0] == "done"
        assert snap["s1"][1] == 1.0
        assert snap["s2"][0] == "running"
        assert 0 < snap["s2"][1] < 1.0


# --------------------------------------------------------------------------- #
# set_cfg / get_cfg + stage_desc → ProgressBank integration
# --------------------------------------------------------------------------- #


@pytest.mark.gui
class TestSetCfgBank:
    """set_cfg controls which config receives stage_desc and GuiTqdm ticks."""

    def test_set_cfg_stores_current(self):
        set_cfg("stem_A")
        assert get_cfg() == "stem_A"
        set_cfg(None)
        assert get_cfg() is None

    def test_stage_desc_feeds_bank(self, rt):
        """stage_desc with a known stage → bank.stage_start on _current_cfg."""
        set_runtime(rt)
        set_cfg("stem_A")
        rt.progress_bank.run_start(["stem_A"])

        stage_desc("load")
        snap = rt.progress_bank.snapshot_all()
        state, _, stage, _ = snap["stem_A"]
        assert state == "running"
        assert stage == "Load"

    def test_stage_desc_noop_when_bank_empty(self, rt):
        """stage_desc with empty bank (no run_start) → no crash."""
        set_runtime(rt)
        set_cfg("stem_A")
        stage_desc("proc")  # should not raise

    def test_guitqdm_feeds_bank(self, rt):
        """GuiTqdm.update → bank.inner on _current_cfg."""
        set_runtime(rt)
        set_cfg("stem_A")
        rt.progress_bank.run_start(["stem_A"])
        rt.progress_bank.stage_start("stem_A", "Processing")

        bar = GuiTqdm(total=100, desc="test")
        bar.update(50)

        snap = rt.progress_bank.snapshot_all()
        _, frac, _, _ = snap["stem_A"]
        assert frac > 0, "bank received inner tick"
        # Processing stage: done_before = 20 (Scan+Load+Prepare), cur = 60*0.5 = 30
        # frac = (20 + 30) / 100 = 0.5
        assert abs(frac - 0.5) < 0.01
