"""Tests for tcm_gui.progress_bridge — GuiTqdm ↔ ProgressState connection.

Verifies the lower (stage) progress bar data path:
  TqdmCallback → GuiTqdm → ProgressState → App._poll_progress → ttk.Progressbar
"""
from __future__ import annotations

import pytest

from tcm_gui.progress_bridge import GuiTqdm, get_runtime, get_tqdm_class, set_runtime, set_tqdm_class
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
