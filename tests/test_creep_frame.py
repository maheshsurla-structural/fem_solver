"""C1b tests — step-by-step creep / shrinkage on 2-D frames
(:class:`femsolver.analysis.time_dependent.StepByStepCreepFrame`).

The frame march tracks a per-element axial force (constant along a member ->
**exact**) and an average curvature (**exact for constant-moment members**;
mesh-refine where the moment varies along a member). The validations therefore
exercise the exact regimes:

* **Determinate deflection growth**: a cantilever under a tip moment (constant
  moment -> constant curvature) grows its tip deflection exactly as ``(1+φ)``.
* **Determinate, no force change**: a simply-supported beam's member moment is
  unchanged by creep.
* **Restrained-axial relaxation**: a fully axially restrained member under
  shrinkage develops a tensile force that relaxes with creep (peaks then falls,
  ending well below the un-relaxed elastic value).
* **Homogeneous axial-indeterminate, no redistribution**: two collinear
  members between fixed ends, loaded at the joint, keep their axial forces under
  uniform creep (compatible -> no redistribution).
* **Differential creep redistributes**: when the two members creep at different
  rates, the axial force redistributes toward the stiffer (less-creeping) one.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.time_dependent import StepByStepCreepFrame

E = 30e9
A = 0.09
I = 6.75e-4          # 0.3 x 0.3 rectangle


def _phi(phi_inf=2.0, b=150.0):
    def phi(t, t0):
        dt = t - t0
        return phi_inf * dt / (b + dt) if dt > 0 else 0.0
    return phi


# ============================================================ determinate

def test_cantilever_tip_moment_grows_by_one_plus_phi():
    """Constant-moment cantilever: tip deflection grows exactly as (1+φ)."""
    nel, L, M0 = 6, 6.0, 100e3
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(nel + 1):
        m.add_node(i + 1, i * L / nel, 0.0)
    for i in range(nel):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 1])
    tip = nel + 1
    phi = _phi()

    times = np.array([28, 30, 40, 70, 150, 400, 1000, 3650], float)
    res = StepByStepCreepFrame(m, phi=phi).run(
        times, sustained_loads=lambda mod: mod.add_nodal_load(tip, [0, 0, M0]),
        track=[(tip, 1)])
    d = res.disp[(tip, 1)]
    phiv = np.array([phi(t, 28) for t in times])
    assert np.allclose(d, d[0] * (1.0 + phiv), rtol=1e-6)


def test_determinate_moment_unchanged():
    nel, L, P = 8, 8.0, -80e3
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(nel + 1):
        m.add_node(i + 1, i * L / nel, 0.0)
    for i in range(nel):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 0])
    m.fix(nel + 1, [0, 1, 0])
    mid = nel // 2 + 1
    times = np.array([28, 40, 150, 1000, 3650], float)
    res = StepByStepCreepFrame(m, phi=_phi()).run(
        times, sustained_loads=lambda mod: mod.add_nodal_load(mid, [0, P, 0]))
    m_mid = res.moment[nel // 2]
    assert np.allclose(m_mid, m_mid[0], rtol=1e-6)


# ============================================================ axial relaxation

def test_restrained_shrinkage_relaxes():
    """A fully axially restrained bar: the shrinkage restraint force rises then
    relaxes with creep, ending well below the un-relaxed elastic value."""
    L = 4.0
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0, 0)
    m.add_node(2, L, 0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, I))
    m.fix(1, [1, 1, 1])
    m.fix(2, [1, 1, 0])                       # both ends axially fixed (θ₂ free)

    def sh(t):
        dt = t - 7.0
        return -300e-6 * dt / (120.0 + dt) if dt > 0 else 0.0

    times = np.array([7, 10, 28, 90, 365, 3650, 18250], float)
    res = StepByStepCreepFrame(m, phi=_phi(), shrinkage=sh).run(
        times, sustained_loads=lambda mod: None)
    N = res.axial[1]
    elastic = E * A * abs(sh(times[-1]))      # un-relaxed restraint force
    assert N[1] > 0                            # tensile restraint develops
    assert N[-1] < 0.6 * elastic               # creep relieved most of it
    assert N[-1] < N.max()                     # relaxes after its peak


# ============================================================ axial redistribution

def _series_bar(L=3.0, phis=None):
    """Two collinear members between fixed ends, axial load P at the joint.
    Axially 1x indeterminate — exact for the frame creep march."""
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0, 0)
    m.add_node(2, L, 0)
    m.add_node(3, 2 * L, 0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, I))
    m.add_element(BeamColumn2D(2, (2, 3), mat, A, I))
    m.fix(1, [1, 1, 1])
    m.fix(3, [1, 1, 1])
    m.fix(2, [0, 1, 1])                        # joint free axially only
    return m


def test_homogeneous_axial_no_redistribution():
    m = _series_bar()
    times = np.array([28, 60, 200, 1000, 3650, 18250], float)
    res = StepByStepCreepFrame(m, phi=_phi()).run(
        times, sustained_loads=lambda mod: mod.add_nodal_load(2, [500e3, 0, 0]))
    n1, n2 = res.axial[1], res.axial[2]
    assert np.allclose(n1, n1[0], rtol=1e-6)   # uniform creep -> no shift
    assert np.allclose(n2, n2[0], rtol=1e-6)


def test_differential_axial_creep_redistributes():
    m = _series_bar()
    times = np.array([28, 60, 200, 1000, 3650, 18250], float)
    # member 1 creeps a lot, member 2 barely -> force sheds to member 2
    elem_phi = {1: _phi(phi_inf=3.0), 2: _phi(phi_inf=0.1)}
    res = StepByStepCreepFrame(m, phi=_phi(), element_phi=elem_phi).run(
        times, sustained_loads=lambda mod: mod.add_nodal_load(2, [500e3, 0, 0]))
    n1 = res.axial[1]
    # equilibrium N1 - N2 = P is preserved every step; the split moves in time
    assert abs(n1[-1] - n1[0]) / abs(n1[0]) > 0.05
