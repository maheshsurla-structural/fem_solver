"""Adaptive step-cutting (plan §16 C4).

When a nonlinear step fails to converge, ``NonlinearStaticAnalysis(substep=True)``
halves the increment and retries, subdividing to cover the nominal step, then
grows back — so a run that would abort with ``NotConvergedError`` on a
too-large step completes instead. Opt-in: with ``substep=False`` (default) the
"raise on non-convergence" contract is unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ConcreteMander,
    ElasticIsotropic,
    BeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    rc_circular_column_section,
)
from femsolver.analysis.algorithm import NotConvergedError
from femsolver.analysis.static_integrator import DisplacementControl, LoadControl
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.sections.analysis import mander_confined_circular

L = 49.0


def _column():
    D, cover, Ec, fc, eps_c0, Es, fy = (
        84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0)
    conf = mander_confined_circular(
        fco=fc, eps_co=eps_c0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    sec = rc_circular_column_section(
        diameter=D, cover=cover, core_concrete=conf.to_material(Ec=Ec),
        cover_concrete=ConcreteMander(fpc=fc, eps_c0=eps_c0, Ec=Ec),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(Es, fy, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)
    mat = ElasticIsotropic(1, E=Ec, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(BeamColumn2DCorotational(1, (1, 2), mat, section=sec))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -1.0, 0.0])
    return m


# ------------------------------------------------------ integrator support

def test_integrator_substep_flags():
    assert LoadControl(0.1).supports_substep is True
    assert DisplacementControl(2, 1, -0.01).supports_substep is True
    # a scheduled DisplacementControl cannot be subdivided
    assert DisplacementControl(2, 1, np.array([-0.01, 0.02])).supports_substep \
        is False


def test_load_control_step_scale():
    lc = LoadControl(0.2)
    lc.set_step_scale(0.5)
    lc.new_step()
    assert lc.lambd == pytest.approx(0.1)     # half increment
    lc.revert_step()
    assert lc.lambd == pytest.approx(0.0)


# ------------------------------------------------------ behaviour

def test_default_still_raises_on_nonconvergence():
    """substep=False (default): a too-large step with a small max_iter still
    raises -- the contract is unchanged."""
    m = _column()
    with pytest.raises(NotConvergedError):
        NonlinearStaticAnalysis(
            m, num_steps=1, integrator=DisplacementControl(2, 1, -0.40),
            track=(2, 1), tol=1e-6, max_iter=6).run()


def test_substep_rescues_and_reaches_target():
    """substep=True subdivides the failing step, completes the run, and reaches
    the same tip displacement + base shear as a fine-step reference."""
    ref = NonlinearStaticAnalysis(
        _column(), num_steps=20, integrator=DisplacementControl(2, 1, -0.02),
        track=(2, 1), tol=1e-6, max_iter=25).run()
    r = NonlinearStaticAnalysis(
        _column(), num_steps=1, integrator=DisplacementControl(2, 1, -0.40),
        track=(2, 1), tol=1e-6, max_iter=6, substep=True).run()
    assert abs(r["tracked"][-1]) == pytest.approx(0.40, rel=1e-6)
    assert len(r["tracked"]) > 1                       # it did subdivide
    assert abs(r["lambdas"][-1]) == pytest.approx(abs(ref["lambdas"][-1]),
                                                  rel=0.02)


def test_substep_still_raises_when_hopeless():
    """If even the smallest sub-step can't converge (max_iter far too small),
    substep still raises rather than looping forever."""
    m = _column()
    with pytest.raises(NotConvergedError):
        NonlinearStaticAnalysis(
            m, num_steps=1, integrator=DisplacementControl(2, 1, -0.40),
            track=(2, 1), tol=1e-6, max_iter=1, substep=True,
            max_substep_halvings=3).run()
