"""Fiber-hinge plan P5 — axial-load benchmark (Stage 1).

Builds the Caltrans 84 in circular RC column as a two-node force-based fiber
beam-column and validates the axial-load stage of the benchmark (plan §2.6/§7.1):

* the FE model holds the P = 2400 kip preload, with a base reaction and tip
  shortening consistent with the transformed axial stiffness ``EA``;
* the tip strain under the preload reproduces the bare section's axial force
  (element <-> section consistency);
* the section's axial capacity curve (N vs imposed axial strain, ``N = sum
  sigma*A``) peaks near the confined-core strain and softens past it (cover
  spalling + core softening).

Golden Midas/CSI axial numbers (§7.1) are pending the user's export; until then
these are the §7.4 cross-checks.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ConcreteMander,
    ElasticIsotropic,
    ForceBeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    rc_circular_column_section,
)
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.sections.analysis import mander_confined_circular

D, COVER, EC, FC, EPS_C0, ES, FY, L = (
    84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0, 49.0)
P_PRELOAD = 2400.0                       # kip compression


def _section():
    conf = mander_confined_circular(
        fco=FC, eps_co=EPS_C0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    return rc_circular_column_section(
        diameter=D, cover=COVER, core_concrete=conf.to_material(Ec=EC),
        cover_concrete=ConcreteMander(fpc=FC, eps_c0=EPS_C0, Ec=EC),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(ES, FY, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)


def _model(section):
    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(ForceBeamColumn2DCorotational(1, (1, 2), mat, section=section))
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 1])                  # only axial DOF free at the tip
    return m


def _EA_hand():
    As = 56 * 2.25
    core_r = 0.5 * D - COVER
    A_core = math.pi * core_r ** 2 - As
    A_cover = math.pi * ((0.5 * D) ** 2 - core_r ** 2)
    return EC * (A_core + A_cover) + ES * As


def test_preload_reaction_and_stiffness():
    """Load-control to 2400 kip: the base reaction balances the load, the tip
    shortens, and EA = P*L/u matches the transformed section EA within ~1%."""
    m = _model(_section())
    m.add_nodal_load(2, [-P_PRELOAD, 0.0, 0.0])
    res = NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=0.1, integrator="load_control",
        track=(2, 0), tol=1e-8).run()
    assert res["lambdas"][-1] == pytest.approx(1.0, rel=1e-6)
    assert m.nodes[1].reaction[0] == pytest.approx(P_PRELOAD, rel=1e-6)
    u = res["tracked"][-1]
    assert u < 0.0                                  # axial shortening
    EA_model = P_PRELOAD / (-u / L)
    assert EA_model == pytest.approx(_EA_hand(), rel=0.02)
    # 2400 kip is ~8.7% of f'c*Ag -> the working point is small and elastic
    assert abs(u / L) < EPS_C0                       # strain below concrete peak


def test_element_strain_consistent_with_section():
    """The tip axial strain under the preload, applied to the bare section,
    reproduces N = 2400 kip (element and section see the same material)."""
    sec = _section()
    m = _model(sec)
    m.add_nodal_load(2, [-P_PRELOAD, 0.0, 0.0])
    res = NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=0.1, integrator="load_control",
        track=(2, 0), tol=1e-8).run()
    strain = res["tracked"][-1] / L
    fresh = _section()
    s, _ = fresh.get_response(np.array([strain, 0.0]))
    assert -s[0] == pytest.approx(P_PRELOAD, rel=1e-3)   # compression positive


def test_section_axial_capacity_softens():
    """Impose axial strain on the section (N = sum sigma*A): the capacity peaks
    near the confined-core strain, exceeds f'c*Ag from confinement + steel, and
    softens past the peak (cover spalling + core softening)."""
    sec = _section()
    Ag = math.pi * D ** 2 / 4.0
    eps = np.linspace(0.0, -0.02, 200)
    N = []
    for e in eps:
        s, _ = sec.get_response(np.array([e, 0.0]))
        sec.commit_state()
        N.append(-float(s[0]))                      # compression positive
    N = np.array(N)
    ip = int(N.argmax())
    assert N.max() > FC * Ag                         # confinement + steel > f'c*Ag
    assert EPS_C0 < -eps[ip] < 0.013                 # peak near confined strain
    assert N[-1] < N.max()                           # post-peak softening
    # working point: the preload strain carries ~2400 kip
    N_at_wp = np.interp(1.04e-4, -eps, N)
    assert N_at_wp == pytest.approx(P_PRELOAD, rel=0.05)
