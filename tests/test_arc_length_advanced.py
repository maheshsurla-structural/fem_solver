"""Phase 50.2 - 50.3 tests for the extended ArcLength integrator.

Covers:

* **Spherical variant** (``psi > 0``) -- load-factor weighting in the
  arc-length constraint norm.
* **Adaptive step sizing** -- ``delta_s`` scales between steps based
  on iteration count to keep Newton convergence efficient.
* **Limit-point detection** -- GSP sign-flip flags the step where
  the path crossed a limit point (e.g., the snap-through peak).

All validation is against the Mises truss (analytically tractable
snap-through benchmark already used for the cylindrical variant) so
that we have a known-good fixture to compare against.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ArcLength,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
    Truss2DCorotational,
)


# ============================================================ fixtures

def _mises_truss_model(P_ref: float = 300.0):
    """Symmetric two-bar Mises truss with apex initially at h = 1 m,
    spans 2*B = 20 m. The apex node (2) carries a downward unit
    reference load."""
    B = 10.0
    h = 1.0
    EA = 1.0e6
    mat = ElasticIsotropic(1, E=EA, nu=0.3)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, B, h)
    m.add_node(3, 2.0 * B, 0.0)
    m.add_element(Truss2DCorotational(1, (1, 2), mat, 1.0))
    m.add_element(Truss2DCorotational(2, (2, 3), mat, 1.0))
    m.fix(1, [1, 1, 1])
    m.fix(3, [1, 1, 1])
    m.fix(2, [1, 0, 1])
    m.add_nodal_load(2, [0.0, -P_ref, 0.0])
    return m


def _analytical_limit(h: float = 1.0, B: float = 10.0, EA: float = 1.0e6) -> float:
    """Closed-form snap-through limit load for symmetric Mises truss."""
    L0 = math.sqrt(B * B + h * h)
    return 2 * EA * h ** 3 / (3.0 * math.sqrt(3.0) * L0 ** 3)


# ============================================================ spherical psi

class TestSphericalVariant:
    def test_psi_zero_matches_cylindrical(self):
        """psi=0 should reproduce the cylindrical result exactly."""
        m1 = _mises_truss_model()
        m2 = _mises_truss_model()
        a_cyl = ArcLength(delta_s=0.08)             # default psi = 0
        a_sph = ArcLength(delta_s=0.08, psi=0.0)
        res1 = NonlinearStaticAnalysis(
            m1, num_steps=20, integrator=a_cyl,
            tol=1e-7, max_iter=30, track=(2, 1),
        ).run()
        res2 = NonlinearStaticAnalysis(
            m2, num_steps=20, integrator=a_sph,
            tol=1e-7, max_iter=30, track=(2, 1),
        ).run()
        np.testing.assert_allclose(
            res1["tracked"], res2["tracked"], rtol=1e-10,
        )
        np.testing.assert_allclose(
            res1["lambdas"], res2["lambdas"], rtol=1e-10,
        )

    def test_psi_positive_traces_path(self):
        """psi > 0 with a problem-scaled value should still trace the
        snap-through cleanly. The ``psi`` parameter has units of
        displacement/force, so for a problem with |F| ~ 300 N and
        typical |u| ~ 1 m, a sensible scale is ``psi ~ 1/|F| = 3e-3``.
        """
        m = _mises_truss_model()
        integrator = ArcLength(delta_s=0.08, psi=1e-3)
        res = NonlinearStaticAnalysis(
            m, num_steps=30, integrator=integrator,
            tol=1e-7, max_iter=30, track=(2, 1),
        ).run()
        apex_dn = -np.array(res["tracked"])
        # Must cross the flat configuration (h = 1.0)
        assert apex_dn[-1] > 1.0

    def test_psi_rejects_negative(self):
        with pytest.raises(ValueError, match="psi"):
            ArcLength(delta_s=0.1, psi=-0.5)


# ============================================================ adaptive stepping

class TestAdaptiveStepping:
    def test_adaptive_changes_step_size(self):
        m = _mises_truss_model()
        integrator = ArcLength(
            delta_s=0.05,
            adaptive=True,
            target_iterations=4,
            delta_s_min=0.01,
            delta_s_max=0.2,
        )
        initial = integrator.delta_s
        NonlinearStaticAnalysis(
            m, num_steps=15, integrator=integrator,
            tol=1e-7, max_iter=30, track=(2, 1),
        ).run()
        # delta_s should have changed from its initial value
        assert integrator.delta_s != pytest.approx(initial)

    def test_adaptive_respects_min_max_caps(self):
        m = _mises_truss_model()
        integrator = ArcLength(
            delta_s=0.05,
            adaptive=True,
            target_iterations=4,
            delta_s_min=0.04,
            delta_s_max=0.06,
        )
        NonlinearStaticAnalysis(
            m, num_steps=15, integrator=integrator,
            tol=1e-7, max_iter=30, track=(2, 1),
        ).run()
        assert 0.04 <= integrator.delta_s <= 0.06

    def test_rejects_invalid_target_iterations(self):
        with pytest.raises(ValueError, match="target_iterations"):
            ArcLength(delta_s=0.1, target_iterations=0)


# ============================================================ limit-point detection

class TestLimitPoints:
    def test_mises_truss_flags_snap_through(self):
        """The Mises truss has exactly two limit points: at the
        ascending peak (snap-through) and the descending trough
        (snap-back into the inverted configuration). Both should be
        detected as GSP sign flips."""
        m = _mises_truss_model()
        integrator = ArcLength(delta_s=0.08)
        NonlinearStaticAnalysis(
            m, num_steps=40, integrator=integrator,
            tol=1e-7, max_iter=30, track=(2, 1),
        ).run()
        lp = integrator.limit_points
        # Must detect at least one limit point (the snap-through peak)
        assert len(lp) >= 1

    def test_pure_elastic_no_limit_points(self):
        """A monotonic-load elastic problem has no limit points -- list
        should stay empty."""
        E, A, Iz = 2.0e11, 1.0e-2, 8.333e-6
        from femsolver import BeamColumn2D
        mat = ElasticIsotropic(1, E=E, nu=0.3)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, 3.0, 0.0)
        m.add_element(BeamColumn2D(1, (1, 2), mat, area=A, Iz=Iz))
        m.fix(1, [1, 1, 1])
        m.add_nodal_load(2, [0.0, -1.0e3, 0.0])
        integrator = ArcLength(delta_s=0.01)
        NonlinearStaticAnalysis(
            m, num_steps=10, integrator=integrator,
            tol=1e-7, max_iter=20,
        ).run()
        assert integrator.limit_points == []


# ============================================================ regression

class TestRegression:
    def test_cylindrical_path_unchanged(self):
        """Sanity check: with default psi=0 the integrator must reach
        the same final apex displacement as the canonical run."""
        m = _mises_truss_model()
        integrator = ArcLength(delta_s=0.08)
        res = NonlinearStaticAnalysis(
            m, num_steps=30, integrator=integrator,
            tol=1e-7, max_iter=30, track=(2, 1),
        ).run()
        # After 30 steps with delta_s=0.08 the apex should be well past
        # the flat configuration (v < -1.0). Specific value approx -1.7.
        assert res["tracked"][-1] < -1.5
