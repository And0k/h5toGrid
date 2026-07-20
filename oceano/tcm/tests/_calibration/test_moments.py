"""
Tests for `moments.py`, in the plan doc's own dependency order: each stage's test only trusts stages
already verified above it. See `moments.py`'s module docstring for the underlying derivation.
"""
import numpy as np
import pytest

from tcm.calibration import moments as mm
from tcm.calibration.moments import fibonacci_sphere
from _calibration.conftest import clustered_sphere, dense_sphere_grid


class TestMonomialAndOuterProduct:
    """Foundation both later stages build on — not in the plan's own numbered TDD list, but cheap
    to pin down directly rather than only indirectly through `analytic_moment_matrix`."""

    def test_monomial_matches_definition(self):
        point = np.array([[2.], [3.], [-1.]])
        x, y, z = point[:, 0]
        expected = [x * x, y * y, z * z, 2 * y * z, 2 * x * z, 2 * x * y, 2 * x, 2 * y, 2 * z, 1]
        assert mm.monomial(point)[:, 0] == pytest.approx(expected)

    def test_outer_product_reconstructs_the_full_symmetric_matrix(self):
        phi = mm.monomial(np.array([[2.], [3.], [-1.]]))
        half_vec = mm.outer_product(phi)[:, 0]
        full = np.zeros((10, 10))
        full[mm._TRIU_ROWS, mm._TRIU_COLS] = full[mm._TRIU_COLS, mm._TRIU_ROWS] = half_vec / mm._TRIU_SCALE
        assert full == pytest.approx(np.outer(phi[:, 0], phi[:, 0]))


class TestAnalyticMomentMatrix:
    """Plan doc Sec. TDD, item 1: "the foundation everything else depends on"."""

    def test_matches_dense_numerical_integration(self):
        points, d_omega = dense_sphere_grid()
        numerical = mm.outer_product(mm.monomial(points)) @ d_omega
        assert numerical == pytest.approx(mm.analytic_moment_matrix(), abs=1e-3)

    def test_matches_hand_derived_values_from_the_plan_doc(self):
        b, index = mm.analytic_moment_matrix(), {}
        for k, (r, c) in enumerate(zip(mm._TRIU_ROWS, mm._TRIU_COLS)):
            index[r, c] = k
        def raw(r, c):                                              # undo the sqrt(2) off-diagonal scaling
            return b[index[r, c]] / mm._TRIU_SCALE[index[r, c]]
        assert raw(0, 0) == pytest.approx(4 * np.pi / 5)              # x^2 * x^2 = x^4
        assert raw(0, 1) == pytest.approx(4 * np.pi / 15)             # x^2 * y^2
        assert raw(0, 9) == pytest.approx(4 * np.pi / 3)              # x^2 * 1
        assert raw(9, 9) == pytest.approx(mm.SPHERE_AREA)             # 1 * 1

    def test_cross_terms_vanish(self):
        """The bug found in a draft implementation shared with me: (2yz), (2xz), (2xy) got exponents
        (0,2,2)/(2,0,2)/(2,2,0) instead of (0,1,1)/(1,0,1)/(1,1,0), silently making several entries
        that must be exactly zero (odd in some coordinate) come out nonzero instead."""
        b, index = mm.analytic_moment_matrix(), {}
        for k, (r, c) in enumerate(zip(mm._TRIU_ROWS, mm._TRIU_COLS)):
            index[r, c] = k
        assert b[index[3, 3]] == pytest.approx(4 * np.pi * 4 / 15)    # (2yz)^2 = 4y^2z^2, all-even -> nonzero
        assert b[index[0, 3]] == pytest.approx(0.)                    # x^2 * 2yz — odd in y and z -> 0
        assert b[index[3, 4]] == pytest.approx(0.)               # 2yz * 2xz = 4xyz^2 — odd x,y -> 0


class TestRestrictedMomentMatrix:
    """
    `restricted_moment_matrix` computes what a stated `target` (default: the whole sphere) can
    *actually* deliver, given `directions`' real coverage — `achievable_condition_weights` uses the
    comparison to stop `solve_optimal_weights` chasing conditions that structurally cannot be matched
    (e.g. an odd-in-z moment that must be exactly 0 over S^2 but never can be if every sample has the
    same-sign z), *without* silently substituting a smaller target as the goal (see
    `build_linear_system`'s docstring for why the target itself stays user-controlled, not auto-picked).
    """

    def test_matches_analytic_target_for_full_sphere_coverage(self):
        directions = fibonacci_sphere(3000)
        assert mm.restricted_moment_matrix(directions) == pytest.approx(mm.analytic_moment_matrix(), abs=2e-4)

    def test_achievable_condition_weights_flags_hemisphere_gap_and_recovers_a_good_fit(self):
        from _calibration.conftest import belt_geometry
        directions = belt_geometry(200, np.linspace(0, 90, 10))                # genuinely one hemisphere
        target = mm.analytic_moment_matrix()                                   # the whole sphere, stated explicitly
        A, b = mm.build_linear_system(directions, target)
        condition_weights = mm.achievable_condition_weights(directions, target)
        assert (condition_weights < 0.5).sum() >= 15                           # real gap: several conditions flagged
        w = mm.solve_optimal_weights(A, b, mm.local_density_baseline(directions), condition_weights=condition_weights)
        achievable = condition_weights > 0.5
        residual = np.linalg.norm((A @ w - b)[achievable]) / np.linalg.norm(b[achievable])
        assert residual < 0.05                                                 # good fit on what IS reachable

    def test_robust_to_a_large_near_duplicate_cluster(self):
        """A naive median-nearest-neighbor spacing estimate collapses toward 0 once duplicates are the
        majority, falsely shrinking "covered" to almost nothing even though the underlying data still
        spans the whole sphere — fixed with a per-sample (not one global) spacing estimate. A small,
        understood residual gap remains: the exact-duplicated point's *own* local spacing collapses to
        ~0 too (that is correct — it really is that densely resampled there), which can locally shrink
        the coverage radius credited to reference points whose nearest sample happens to be exactly it,
        slightly under-covering a sliver right at that one location. Near a pole this measurably shifts
        the z-heavy moments (z**4 is near its max there) even though the affected solid angle is tiny
        (~0.3% of the reference grid in this test) — hence the moderate, not tight, tolerance below."""
        base = fibonacci_sphere(300)
        duplicated = np.hstack([base, np.tile(base[:, :1], 500)])
        b_clean = mm.restricted_moment_matrix(base)
        b_duplicated = mm.restricted_moment_matrix(duplicated)
        assert b_duplicated == pytest.approx(b_clean, abs=0.2)


class TestBuildLinearSystem:
    """Plan doc Sec. TDD, item 2."""

    def test_shapes(self):
        A, b = mm.build_linear_system(fibonacci_sphere(7))
        assert A.shape == (55, 7)
        assert b.shape == (55,)

    def test_matches_manual_computation(self):
        directions = np.array([[1., 0.], [0., 1.], [0., 0.]])         # x-hat, y-hat
        A, _ = mm.build_linear_system(directions)
        for col, direction in enumerate(directions.T):
            phi = mm.monomial(direction[:, np.newaxis])[:, 0]
            full = np.outer(phi, phi)
            assert A[:, col] == pytest.approx(full[mm._TRIU_ROWS, mm._TRIU_COLS] * mm._TRIU_SCALE)


class TestSolveOptimalWeights:
    """Plan doc Sec. TDD, item 3."""

    def test_uniform_points_get_near_equal_weights(self):
        directions = fibonacci_sphere(500)
        weights = mm.solve_optimal_weights(*mm.build_linear_system(directions),
                                            mm.local_density_baseline(directions))
        assert weights.std() / weights.mean() < 0.01

    def test_avoids_degenerate_sparsity(self):
        """Plain NNLS on this (55 conditions, N unknowns) system collapses onto a handful of
        points — confirmed while developing this; the ridge-to-baseline term exists to prevent it."""
        directions = fibonacci_sphere(500)
        weights = mm.solve_optimal_weights(*mm.build_linear_system(directions),
                                            mm.local_density_baseline(directions))
        assert (weights > 1e-8).sum() > 490

    def test_duplicated_cluster_total_weight_is_conserved(self):
        """A global-uniform ridge baseline fails this (cluster total grows with duplicate count,
        confirmed empirically) — `local_density_baseline` must depend on local sample geometry."""
        base = fibonacci_sphere(300)
        w_single = mm.solve_optimal_weights(*mm.build_linear_system(base), mm.local_density_baseline(base))
        for n_duplicates in (5, 50, 500):
            duplicated = np.hstack([base, np.tile(base[:, :1], n_duplicates)])
            w = mm.solve_optimal_weights(*mm.build_linear_system(duplicated),
                                          mm.local_density_baseline(duplicated))
            cluster_total = w[0] + w[-n_duplicates:].sum()
            assert cluster_total == pytest.approx(w_single[0], rel=0.1)
            assert np.ptp(w[-n_duplicates:]) < 1e-8                # duplicates interchangeable -> equal w

    def test_severe_clustering_still_dramatically_improves_on_uniform_weights(self):
        rng = np.random.default_rng(0)
        directions = clustered_sphere(500, 4500, [0, 0, 1], 0.15, rng)
        A, b = mm.build_linear_system(directions)
        n = directions.shape[1]
        def resid(w):
            return np.linalg.norm(A @ w - b) / np.linalg.norm(b)
        resid_uniform = resid(np.full(n, mm.SPHERE_AREA / n))
        resid_optimal = resid(mm.solve_optimal_weights(A, b, mm.local_density_baseline(directions)))
        assert resid_optimal < 0.15 * resid_uniform
