"""Tests for the ArcLength integrator (cylindrical, psi = 0).

Arc-length control treats both ``u`` and ``lambda`` as unknowns and
enforces ``||u - u_step_start|| = delta_s`` at each step. This is the
canonical method for tracing equilibrium paths past limit points —
snap-through, snap-back, post-buckling.

The Mises (von Mises) truss snap-through is the classical benchmark.
Example 07 in this repo set up the geometry under load control;
example 13 will trace the full curve past the limit point under
arc-length. The tests below verify three things:

1. **Elastic regression** — for a problem load control can solve, arc
   length reaches the same equilibrium states.
2. **Mises truss snap-through** — load control fails at the limit
   point (we already documented this in
   ``examples/07_mises_truss_snapthrough.py``); arc-length traces
   past it to the descending and ascending branches.
3. **Constructor validation** — bad inputs raise errors.
"""
import math

import numpy as np
import pytest

from femsolver import (
    ArcLength,
    BeamColumn2D,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
    Truss2DCorotational,
)


# ====================================================== constructor ==

def test_arc_length_rejects_nonpositive_delta_s():
    with pytest.raises(ValueError):
        ArcLength(delta_s=0.0)
    with pytest.raises(ValueError):
        ArcLength(delta_s=-1.0)


def test_arc_length_rejects_bad_direction():
    with pytest.raises(ValueError):
        ArcLength(delta_s=0.01, initial_direction=0)
    with pytest.raises(ValueError):
        ArcLength(delta_s=0.01, initial_direction=2)


# ====================================================== elastic ==

def test_arc_length_elastic_cantilever_matches_load_control():
    """For a linear elastic problem, arc-length and load control should
    arrive at the same equilibrium states (just with the arc-length
    parameter governing increments rather than load)."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    mat = ElasticIsotropic(1, E=E, nu=0.3)

    def build():
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
        m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
        m.fix(1, [1, 1, 1])
        m.add_nodal_load(2, [0.0, -P, 0.0])
        return m

    # Load control reference
    m_lc = build()
    NonlinearStaticAnalysis(
        m_lc, num_steps=5, dlambda=0.2, tol=1e-8, max_iter=10,
    ).run()
    v_lc = m_lc.node(2).disp[1]

    # Arc length: pick delta_s so we cover the same ground.
    # First-step predictor: dlambda = +delta_s / ||du_p||
    # Choose delta_s = some value, then we take some number of steps to
    # reach lambda = 1.
    m_al = build()
    # Pick delta_s so the analysis advances roughly the same total arc.
    # The norm of the converged ``u`` vector at lambda=1 is
    # ||u_lc||; choose delta_s = ||u_lc|| / num_steps so we end near
    # lambda=1.
    u_lc_full = np.zeros(m_lc.neq)
    for n in m_lc.nodes.values():
        for j in range(n.ndf):
            eq = int(n.eqn[j])
            if eq >= 0:
                u_lc_full[eq] = n.disp[j]
    norm_u_lc = float(np.linalg.norm(u_lc_full))
    integrator = ArcLength(delta_s=norm_u_lc / 5.0)
    res = NonlinearStaticAnalysis(
        m_al, num_steps=5, integrator=integrator, tol=1e-8, max_iter=15,
    ).run()
    v_al = m_al.node(2).disp[1]
    # For a linear problem, arc-length and load control trace the same
    # equilibrium curve; they just parametrise differently. Pick the
    # arc length to land near lambda=1 and verify the tip displacement
    # matches to a reasonable tolerance.
    assert v_al == pytest.approx(v_lc, rel=5e-2)
    assert res["final_lambda"] == pytest.approx(1.0, rel=5e-2)


# ====================================================== Mises truss snap-through

def test_arc_length_traces_mises_truss_past_limit_point():
    """Shallow Mises truss under apex load — load control reaches the
    limit point and stalls; arc length continues past it.

    The structure: two trusses meeting at an apex with a small rise
    ``h``. Vertical compression at the apex stiffens the structure
    until the trusses rotate to horizontal — beyond which it softens
    and eventually loses stability under load control.

    With arc-length we trace the full curve and confirm:
      * the load factor reaches *and then drops below* its initial-
        stiffness path — i.e. we see the descending branch
      * the apex displacement keeps growing monotonically
    """
    B = 10.0
    h = 1.0
    EA = 1.0e6
    P_ref = 300.0       # reference load: just above the analytical limit

    mat = ElasticIsotropic(1, E=EA, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    # Apex at (B, h); supports at (0, 0) and (2B, 0)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, B, h)
    m.add_node(3, 2.0 * B, 0.0)
    m.add_element(Truss2DCorotational(1, (1, 2), mat, 1.0))
    m.add_element(Truss2DCorotational(2, (2, 3), mat, 1.0))
    m.fix(1, [1, 1, 1])
    m.fix(3, [1, 1, 1])
    # Apex constrained horizontally (problem is symmetric) so we have
    # one free DOF: vertical apex motion.
    m.fix(2, [1, 0, 1])    # u, theta fixed; only v free
    m.add_nodal_load(2, [0.0, -P_ref, 0.0])

    # Arc length: choose delta_s small enough to trace the descending
    # branch with decent resolution. Total apex travel ~ 2h = 2.0.
    integrator = ArcLength(delta_s=0.08)
    n_steps = 40
    res = NonlinearStaticAnalysis(
        m, num_steps=n_steps, integrator=integrator,
        tol=1e-6, max_iter=30, track=(2, 1),
    ).run()
    lambdas = np.array(res["lambdas"])
    disps = np.array(res["tracked"])
    forces = lambdas * P_ref

    # Apex must have moved past the flat configuration (v < -h).
    assert disps[-1] < -h, (
        f"apex did not pass through the flat configuration: "
        f"final v = {disps[-1]:.4f}, h = {h}"
    )
    # The load factor must show a peak and then a drop — the limit
    # point. Take the max of the first half and confirm the second
    # half is below the max.
    half = len(lambdas) // 2
    peak_load = forces[:half].max()
    later_min = forces[half:].min()
    assert later_min < peak_load, (
        f"no descending branch: peak={peak_load}, later min={later_min}"
    )
    # Also confirm apex displacement increased monotonically — arc-
    # length traced the whole path without doubling back.
    assert np.all(np.diff(disps) < 0), (
        "apex displacement did not advance monotonically"
    )


def test_arc_length_handles_negative_initial_direction():
    """initial_direction=-1 must produce a downward-load trace
    (opposite of the default +1)."""
    B = 10.0; h = 1.0; EA = 1.0e6
    mat = ElasticIsotropic(1, E=EA, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, B, h); m.add_node(3, 2.0 * B, 0.0)
    m.add_element(Truss2DCorotational(1, (1, 2), mat, 1.0))
    m.add_element(Truss2DCorotational(2, (2, 3), mat, 1.0))
    m.fix(1, [1, 1, 1]); m.fix(3, [1, 1, 1]); m.fix(2, [1, 0, 1])
    m.add_nodal_load(2, [0.0, -1.0, 0.0])
    integrator = ArcLength(delta_s=0.02, initial_direction=-1)
    NonlinearStaticAnalysis(
        m, num_steps=3, integrator=integrator, tol=1e-6, max_iter=15,
        track=(2, 1),
    ).run()
    # With direction = -1, the load factor should go *negative* (the
    # reference load is a downward unit force; going -1 means we
    # actually unload — the apex moves UP).
    assert integrator.lambd < 0.0
    assert m.node(2).disp[1] > 0.0
