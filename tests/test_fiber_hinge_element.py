"""Fiber-hinge plan P9 (G4) — finite-length fiber-hinge beam-column.

The beam-with-hinges element (elastic member + fiber hinge of length ``lp`` at
each end) reproduces the elastic prismatic stiffness for any ``lp``, matches
the distributed force-based element under axial load (P5), and yields under a
lateral pushover with the same base-shear ballpark (P6).
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import (
    ConcreteMander,
    ElasticIsotropic,
    FiberHingeBeamColumn2D,
    FiberSection2D,
    ForceBeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    rc_circular_column_section,
)
from femsolver.analysis.static_integrator import DisplacementControl
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.materials.uniaxial.elastic import UniaxialElastic
from femsolver.sections.analysis import mander_confined_circular

EC, D2, L = 3605.0, 24.0, 100.0


def _rc_section():
    D, cover, fc, eps_c0, Es, fy = 84.0, 2.0, 5.0, 0.002219, 29000.0, 68.0
    conf = mander_confined_circular(
        fco=fc, eps_co=eps_c0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    return rc_circular_column_section(
        diameter=D, cover=cover, core_concrete=conf.to_material(Ec=EC),
        cover_concrete=ConcreteMander(fpc=fc, eps_c0=eps_c0, Ec=EC),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(Es, fy, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)


def _model(elem):
    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 49.0, 0.0)
    m.add_element(elem(mat))
    m.fix(1, [1, 1, 1])
    return m


# --------------------------------------------------- elastic equivalence

def _elastic_K(elem_factory):
    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(elem_factory(mat))
    m.fix(1, [1, 1, 1])
    m.number_dofs()
    return m.elements[1].K_tangent_global()


@pytest.mark.parametrize("lp", [4.0, 8.0, 15.0])
def test_elastic_stiffness_matches_distributed(lp):
    """With an elastic fiber section the hinge element's stiffness equals the
    (exact) distributed force-based element for any hinge length."""
    Kh = _elastic_K(lambda mat: FiberHingeBeamColumn2D(
        1, (1, 2), mat, section=FiberSection2D.circular(
            D2, 8, 24, UniaxialElastic(EC)), lp=lp))
    Kd = _elastic_K(lambda mat: ForceBeamColumn2DCorotational(
        1, (1, 2), mat, section=FiberSection2D.circular(
            D2, 8, 24, UniaxialElastic(EC))))
    assert np.max(np.abs(Kh - Kd)) / np.max(np.abs(Kd)) < 1e-9


def test_rejects_bad_hinge_length():
    sec = FiberSection2D.circular(D2, 8, 24, UniaxialElastic(EC))
    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    with pytest.raises(ValueError):
        FiberHingeBeamColumn2D(1, (1, 2), mat, section=sec, lp=0.0)


# --------------------------------------------------- axial (P5) + pushover (P6)

def _axial_tip_disp(elem_factory):
    m = _model(elem_factory)
    m.fix(2, [0, 1, 1])
    m.add_nodal_load(2, [-2400.0, 0.0, 0.0])
    NonlinearStaticAnalysis(m, num_steps=10, dlambda=0.1,
                            integrator="load_control", track=(2, 0),
                            tol=1e-8).run()
    return m.nodes[2].disp[0]


def test_axial_matches_distributed():
    """Under the 2400 kip axial preload the hinge element and the distributed
    force-based element shorten by the same amount (axial is uniform)."""
    uh = _axial_tip_disp(
        lambda mat: FiberHingeBeamColumn2D(1, (1, 2), mat, section=_rc_section(),
                                           lp=8.0))
    ud = _axial_tip_disp(
        lambda mat: ForceBeamColumn2DCorotational(1, (1, 2), mat,
                                                  section=_rc_section()))
    assert uh == pytest.approx(ud, rel=0.01)


def _pushover(elem_factory, n=22):
    m = _model(elem_factory)
    m.add_nodal_load(2, [0.0, -1.0, 0.0])
    r = NonlinearStaticAnalysis(
        m, num_steps=n, integrator=DisplacementControl(2, 1, -0.006),
        track=(2, 1), tol=1e-6, max_iter=60).run()
    return np.abs(np.array(r["tracked"])), np.abs(np.array(r["lambdas"]))


def test_pushover_yields_and_matches_ballpark():
    """The hinge cantilever pushover yields (sub-linear base-shear curve) and
    its peak base shear is within ~10% of the distributed element."""
    uh, Vh = _pushover(
        lambda mat: FiberHingeBeamColumn2D(1, (1, 2), mat, section=_rc_section(),
                                           lp=8.0))
    ud, Vd = _pushover(
        lambda mat: ForceBeamColumn2DCorotational(1, (1, 2), mat,
                                                  section=_rc_section()))
    assert np.all(np.diff(Vh) > -1.0)                       # monotone-ish rise
    k_init = Vh[0] / uh[0]
    k_late = (Vh[-1] - Vh[-3]) / (uh[-1] - uh[-3])
    assert k_late < 0.5 * k_init                            # yielding
    assert Vh.max() == pytest.approx(Vd.max(), rel=0.10)    # same ballpark
