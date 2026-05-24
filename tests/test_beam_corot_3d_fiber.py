"""Tests for BeamColumn3DCorotational + FiberSection3D — combined
3-D geometric and material nonlinearity.

Phase 13.5 combines Phase 5.5 (FiberSection3D) with Phase 13
(BeamColumn3DCorotational). Both pieces compose by inheritance —
``BeamColumn3DCorotational`` inherits ``BeamColumn3D``'s stateful-
sections path, which means passing ``section=FiberSection3D(...)`` to
the corotational constructor activates per-IP cloned fiber sections
automatically. The four invariants below pin down the composition.

1. **Stateful-section path activates** — passing a fiber section to
   the corotational constructor produces per-IP fiber clones,
   ``use_numerical_integration = True``, the right ``self.sections``
   shape.
2. **Elastic equivalence at u=0** — for an all-elastic fiber section,
   the corotational + fiber tangent matches the corotational alone
   (to within fiber discretisation error).
3. **Plasticity propagates through commit** — the regression for the
   silent-override bug found in Phase 5.5. With ``commit_state`` chain
   now correctly forwarding to per-IP sections, fiber plastic strain
   persists across the analysis.
4. **Both P-Delta amplification AND yielding emerge** — at moderate
   axial compression (~20 % of cantilever P_cr) combined with lateral
   pushover past first yield, the tip deflection exceeds the
   no-axial-load case AND fibers show plastic strain. This is the
   combined material + geometric signature.
"""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn3D,
    BeamColumn3DCorotational,
    ElasticIsotropic,
    FiberSection3D,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
    UniaxialElastic,
)


# ===================================================== helpers

def _build_3d_fiber_cantilever(*, with_corotational: bool = True,
                                E=2.0e11, sy=400.0e6, b_post=0.05,
                                width_y=0.2, width_z=0.1, L=3.0,
                                n_y=20, n_z=10):
    """Returns (model, element, constants)."""
    G = E / (2.0 * 1.3)            # nu=0.3
    GJ = G * 0.229 * width_y * width_z ** 3      # St. Venant for rectangle
    mat_iso = ElasticIsotropic(1, E=E, nu=0.3)
    mat_u = UniaxialBilinear(E=E, sigma_y=sy, b=b_post)
    sec = FiberSection3D.rectangular(
        width_y=width_y, width_z=width_z, n_y=n_y, n_z=n_z,
        material=mat_u, GJ=GJ,
    )
    m = Model(ndm=3, ndf=6); m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0, 0.0); m.add_node(2, L, 0.0, 0.0)
    if with_corotational:
        elem = BeamColumn3DCorotational(1, (1, 2), mat_iso, section=sec)
    else:
        elem = BeamColumn3D(1, (1, 2), mat_iso, section=sec)
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    cn = dict(E=E, G=G, GJ=GJ, sy=sy, b_post=b_post,
              width_y=width_y, width_z=width_z, L=L,
              Iz=width_z * width_y ** 3 / 12.0,
              Iy=width_y * width_z ** 3 / 12.0,
              Mz_y=sy * width_z * width_y ** 2 / 6.0,
              My_y=sy * width_y * width_z ** 2 / 6.0,
              P_axial_yield=sy * width_y * width_z,
              )
    return m, elem, cn


# ===================================================== construction

def test_corot_3d_with_fiber_activates_stateful_path():
    """Stateful-section path activates correctly when passing a
    FiberSection3D to the corotational element."""
    _, elem, _ = _build_3d_fiber_cantilever()
    assert elem._stateful_sections is True
    assert elem.use_numerical_integration is True
    assert len(elem.sections) == elem.n_int
    # Per-IP independent clones
    for i in range(elem.n_int):
        for j in range(i + 1, elem.n_int):
            assert elem.sections[i] is not elem.sections[j]


def test_corot_3d_with_fiber_K_at_zero_matches_corot_elastic_3d():
    """At u = 0 the K_tangent of a corot+fiber cantilever matches the
    K_tangent of a corot+elastic-section cantilever with equivalent
    gross properties. Fiber discretisation error is the only source
    of discrepancy."""
    # Build the fiber version
    m_fib, e_fib, cn = _build_3d_fiber_cantilever(
        n_y=40, n_z=20,   # fine mesh for low discretisation error
    )
    # Replace UniaxialBilinear with UniaxialElastic (no plasticity for
    # this comparison).
    for sec in e_fib.sections:
        for f in sec.fibers:
            f.material = UniaxialElastic(E=cn["E"])

    # Build the elastic-section reference
    A_ref = cn["width_y"] * cn["width_z"]
    Iz_ref = cn["Iz"]
    Iy_ref = cn["Iy"]
    J_ref = cn["GJ"] / cn["G"]
    mat_iso = ElasticIsotropic(1, E=cn["E"], nu=0.3)
    m_ref = Model(ndm=3, ndf=6); m_ref.add_material(mat_iso)
    m_ref.add_node(1, 0.0, 0.0, 0.0); m_ref.add_node(2, cn["L"], 0.0, 0.0)
    e_ref = BeamColumn3DCorotational(
        1, (1, 2), mat_iso, A_ref, Iy_ref, Iz_ref, J_ref,
    )
    m_ref.add_element(e_ref)

    K_fib = e_fib.K_tangent_global()
    K_ref = e_ref.K_tangent_global()
    # Compare relative — fiber discretisation gives ~0.3 % error on
    # second-moment terms at this fineness.
    rel = np.max(np.abs(K_fib - K_ref)) / np.max(np.abs(K_ref))
    assert rel < 5e-3


# ===================================================== plasticity commit

def test_fiber_3d_plasticity_propagates_through_commit():
    """Regression for the silent-override bug found in Phase 5.5.
    After a pushover past first yield, fiber plastic strain must
    persist in ``eps_p_committed``. The corotational override path
    must invoke the parent BeamColumn3D's stateful commit_state
    (not a stale single-section commit)."""
    m, elem, cn = _build_3d_fiber_cantilever()
    # Push past first yield in pure y-bending
    P_lateral = 1.3 * cn["Mz_y"] / cn["L"]
    m.add_nodal_load(2, [0.0, -P_lateral, 0.0, 0.0, 0.0, 0.0])
    NonlinearStaticAnalysis(
        m, num_steps=20, dlambda=1.0 / 20, tol=1e-5, max_iter=30,
    ).run()
    fixed_sec = elem.sections[0]
    n_yielded = sum(
        1 for f in fixed_sec.fibers if f.material.eps_p_committed != 0.0
    )
    assert n_yielded > 0


def test_fiber_3d_yielding_symmetric_under_pure_bending():
    """Pure y-bending past yield should yield fibers symmetrically:
    equal number on +y (compression) and -y (tension) sides.
    """
    m, elem, cn = _build_3d_fiber_cantilever()
    P_lateral = 1.3 * cn["Mz_y"] / cn["L"]
    m.add_nodal_load(2, [0.0, -P_lateral, 0.0, 0.0, 0.0, 0.0])
    NonlinearStaticAnalysis(
        m, num_steps=20, dlambda=1.0 / 20, tol=1e-5, max_iter=30,
    ).run()
    fixed_sec = elem.sections[0]
    yielded = [f for f in fixed_sec.fibers if f.material.eps_p_committed != 0.0]
    n_plus_y = sum(1 for f in yielded if f.y > 0.0)
    n_minus_y = sum(1 for f in yielded if f.y < 0.0)
    assert n_plus_y == n_minus_y
    assert n_plus_y > 0


# ===================================================== combined nonlinearity

def test_corot_3d_fiber_combines_p_delta_with_plasticity():
    """Apply moderate axial compression (~20 % of cantilever Euler
    load) + lateral pushover just past first yield. The tip
    deflection must exceed the no-axial-load case (P-Delta
    amplification) AND fibers must have yielded (material plasticity).
    """
    E = 2.0e11
    width_y, width_z, L = 0.2, 0.1, 3.0
    Iz = width_z * width_y ** 3 / 12.0
    # Cantilever Euler critical load (effective length 2L)
    P_cr_cantilever = math.pi ** 2 * E * Iz / (4.0 * L ** 2)

    def lateral_disp(axial_ratio: float) -> tuple[float, int]:
        m, elem, cn = _build_3d_fiber_cantilever(L=L)
        P_axial = axial_ratio * P_cr_cantilever
        P_lateral = 1.05 * cn["Mz_y"] / cn["L"]   # just past first yield
        m.add_nodal_load(2, [-P_axial, -P_lateral, 0.0, 0.0, 0.0, 0.0])
        NonlinearStaticAnalysis(
            m, num_steps=20, dlambda=1.0 / 20, tol=1e-5, max_iter=30,
        ).run()
        n_y = sum(
            1 for f in elem.sections[0].fibers
            if f.material.eps_p_committed != 0.0
        )
        return abs(m.node(2).disp[1]), n_y

    # Baseline: no axial
    v0, nyielded0 = lateral_disp(0.0)
    # 20 % of cantilever Euler: moderate P-Delta amplification
    v20, nyielded20 = lateral_disp(0.20)
    # Both should have yielded
    assert nyielded0 > 0
    assert nyielded20 > 0
    # P-Delta makes the deflection larger
    assert v20 > 1.10 * v0, (
        f"P-Delta amplification too small: v0={v0:.4e}, v20={v20:.4e}"
    )


def test_corot_3d_fiber_axial_load_advances_first_yield():
    """M-P interaction: under axial compression, the lateral load at
    which the first fiber yields is *smaller* than the no-axial case.
    Axial compression already loads the +y fibers (under bending) so
    they hit yield sooner.
    """
    width_y, width_z, L = 0.2, 0.1, 3.0
    Iz = width_z * width_y ** 3 / 12.0
    E = 2.0e11
    P_cr_cantilever = math.pi ** 2 * E * Iz / (4.0 * L ** 2)

    def first_yield_state(axial_ratio: float, P_lat_ratio: float) -> int:
        """Run pushover and return number of yielded fibers."""
        m, elem, cn = _build_3d_fiber_cantilever(L=L)
        P_axial = axial_ratio * P_cr_cantilever
        P_lat = P_lat_ratio * cn["Mz_y"] / cn["L"]
        m.add_nodal_load(2, [-P_axial, -P_lat, 0.0, 0.0, 0.0, 0.0])
        NonlinearStaticAnalysis(
            m, num_steps=10, dlambda=1.0 / 10, tol=1e-5, max_iter=30,
        ).run()
        return sum(
            1 for f in elem.sections[0].fibers
            if f.material.eps_p_committed != 0.0
        )

    P_lat_ratio = 1.0      # right at the analytical yield force
    nyielded_no_axial = first_yield_state(0.0, P_lat_ratio)
    nyielded_with_axial = first_yield_state(0.20, P_lat_ratio)
    # With axial compression, more fibers should be past yield at the
    # same lateral force (since axial advances the yield threshold).
    assert nyielded_with_axial > nyielded_no_axial
