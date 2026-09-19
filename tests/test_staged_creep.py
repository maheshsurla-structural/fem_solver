"""C1a tests -- per-element age-based EMM creep in the incremental staged
driver (construction-stage parity plan C1a).

Validations
-----------
* **Backward-compat**: with no creep config the staged run is unchanged
  (elastic), and every applied factor is 1.0.
* **Scalar equivalence**: when every active element shares one age/birth,
  the per-element creep factor equals the uniform EMM factor, so a creep run
  reproduces a ``stiffness_factor`` run at that factor.
* **Determinate creep**: on a simply-supported span the long-term deflection
  is ``delta_inst·(1+phi)`` (pure EMM, chi=1) while the member force is
  unchanged -- creep grows deflection, not force, in a determinate structure.
* **Load age**: a load applied at a later stage creeps less (larger factor).
* **Differential-age redistribution**: in an indeterminate structure, elements
  cast at different ages creep by different amounts, so the internal-force
  distribution differs from the equal-age (uniform-creep) case.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D
from femsolver.elements.truss import Truss2D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.bridges.creep_shrinkage import cebfip_creep_coefficient
from femsolver.bridges.staged_construction import (
    ErectionStage,
    IncrementalStagedAnalysis,
    StagedCreep,
)

F_CM = 38.0e6            # mean concrete strength (Pa) ~ C30/37
E_C = 33.0e9


def _ss_beam(nel=8, L=8.0, A=0.05, I=4e-4, E=E_C):
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(nel + 1):
        m.add_node(i + 1, i * L / nel, 0.0)
    for i in range(nel):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 0])
    m.fix(nel + 1, [0, 1, 0])
    return m, mat


# ============================================================ backward-compat

def test_no_creep_matches_elastic():
    """creep=None -> identical to the pre-C1a elastic staged run."""
    nel, P = 8, -100e3
    m, _ = _ss_beam(nel)
    beam = list(range(1, nel + 1))
    stage = ErectionStage(name="all", add_elements=beam,
                          loads={nel // 2 + 1: [0, P, 0]})
    res = IncrementalStagedAnalysis(m, [stage]).run()
    mid = m.node(nel // 2 + 1).disp[1]

    # closed-form simply-supported point-load deflection PL^3/48EI
    L, I = 8.0, 4e-4
    assert mid == pytest.approx(P * L ** 3 / (48.0 * E_C * I), rel=1e-6)
    # elastic factors are all 1.0
    assert all(abs(f - 1.0) < 1e-15 for f in res.creep_factors[0].values())


# ============================================================ scalar equivalence

def test_uniform_age_equals_scalar_factor():
    """All elements one age -> per-element creep factor == the uniform EMM
    factor, so the creep run matches a stiffness_factor run at that value."""
    nel, P = 8, -100e3
    beam = list(range(1, nel + 1))
    t0, life = 28.0, 10000.0
    phi = cebfip_creep_coefficient(t_days=t0 + life, t0_days=t0,
                                   f_cm=F_CM).phi
    factor = 1.0 / (1.0 + phi)          # chi = 1

    # creep run: elements initially active at age t0, one sustained stage
    m1, _ = _ss_beam(nel)
    creep = StagedCreep(f_cm=F_CM, chi=1.0, initial_age_days=t0,
                        final_time_days=life)
    IncrementalStagedAnalysis(
        m1, [ErectionStage(name="s", loads={nel // 2 + 1: [0, P, 0]})],
        initial_active=beam, creep=creep).run()
    mid_creep = m1.node(nel // 2 + 1).disp[1]

    # scalar run at the same factor
    m2, _ = _ss_beam(nel)
    IncrementalStagedAnalysis(
        m2, [ErectionStage(name="s", add_elements=beam,
                           loads={nel // 2 + 1: [0, P, 0]},
                           stiffness_factor=factor)]).run()
    mid_scalar = m2.node(nel // 2 + 1).disp[1]

    assert mid_creep == pytest.approx(mid_scalar, rel=1e-9)


# ============================================================ determinate span

def test_determinate_deflection_grows_force_constant():
    """delta_long = delta_inst·(1+phi); member force unchanged (determinate)."""
    nel, P = 8, -100e3
    beam = list(range(1, nel + 1))
    t0, life = 7.0, 18250.0
    phi = cebfip_creep_coefficient(t_days=t0 + life, t0_days=t0,
                                   f_cm=F_CM).phi

    m_el, _ = _ss_beam(nel)
    res_el = IncrementalStagedAnalysis(
        m_el, [ErectionStage(name="s", add_elements=beam,
                             loads={nel // 2 + 1: [0, P, 0]})]).run()
    mid_el = m_el.node(nel // 2 + 1).disp[1]

    m_cr, _ = _ss_beam(nel)
    creep = StagedCreep(f_cm=F_CM, chi=1.0, initial_age_days=t0,
                        final_time_days=life)
    res_cr = IncrementalStagedAnalysis(
        m_cr, [ErectionStage(name="s", add_elements=beam,
                             loads={nel // 2 + 1: [0, P, 0]},
                             age_at_activation_days=t0)],
        initial_active=None, creep=creep).run()
    mid_cr = m_cr.node(nel // 2 + 1).disp[1]

    # deflection amplified by (1+phi)
    assert mid_cr == pytest.approx(mid_el * (1.0 + phi), rel=1e-6)
    # member end-force essentially unchanged (creep doesn't redistribute in a
    # determinate structure)
    f_el = res_el.element_forces[1]
    f_cr = res_cr.element_forces[1]
    assert np.allclose(f_cr, f_el, rtol=1e-6, atol=1e-3)


# ============================================================ load age

def test_later_load_creeps_less():
    """A load applied at a later stage has less remaining time to creep, so
    its per-element creep factor is larger (stiffer)."""
    nel, P = 8, -50e3
    beam = list(range(1, nel + 1))
    creep = StagedCreep(f_cm=F_CM, chi=1.0, initial_age_days=28.0,
                        final_time_days=20000.0)
    stages = [
        ErectionStage(name="early", add_elements=beam,
                      loads={nel // 2 + 1: [0, P, 0]}, duration_days=5000.0),
        ErectionStage(name="late", loads={nel // 2 + 1: [0, P, 0]}),
    ]
    res = IncrementalStagedAnalysis(
        m := _ss_beam(nel)[0], stages, creep=creep).run()
    f_early = res.creep_factors[0][1]     # element 1, stage 0
    f_late = res.creep_factors[1][1]      # element 1, stage 1
    assert f_late > f_early               # later load creeps less
    assert 0.0 < f_early < 1.0 and f_late <= 1.0
    _ = m


# ============================================================ redistribution

def test_differential_age_redistributes():
    """A simply-supported beam with a central prop to ground, loaded in one
    stage. When the prop is cast much older (stiffer, creeps less) than the
    beam, it attracts a different share of that load than when the two share
    one age -- differential per-element creep redistributes the increment.

    (C1a redistributes *within* a load increment across parallel paths of
    different age; relaxing an already-applied load through a later-added
    restraint is step-by-step creep -- plan C1b.)
    """
    def _propped(prop_area=0.02):
        mat = ElasticIsotropic(1, E=E_C, nu=0.2, rho=0.0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        for i in range(5):
            m.add_node(i + 1, i * 2.0, 0.0)     # span nodes 1..5, L=8
        m.add_node(6, 4.0, -3.0)                # prop base below midspan
        for i in range(4):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, 0.05, 4e-4))
        m.add_element(Truss2D(100, (3, 6), mat, prop_area))   # central prop
        m.fix(1, [1, 1, 0])                      # simply supported (stable
        m.fix(5, [0, 1, 0])                      #   on its own before prop)
        m.fix(6, [1, 1, 1])
        return m

    beam = [1, 2, 3, 4]
    creep = StagedCreep(f_cm=F_CM, chi=0.8, final_time_days=18250.0)

    # equal age: beam + prop both at 28 d, loaded in one stage
    m_eq = _propped()
    res_eq = IncrementalStagedAnalysis(
        m_eq, [ErectionStage(name="s", add_elements=[100],
                             loads={3: [0, -80e3, 0]},
                             age_at_activation_days=28.0)],
        initial_active=beam,
        creep=StagedCreep(f_cm=F_CM, chi=0.8, initial_age_days=28.0,
                          final_time_days=18250.0)).run()
    prop_eq = float(np.linalg.norm(res_eq.element_forces[100]))

    # differential age: young beam (creeps a lot) + old stiff prop (barely)
    m_df = _propped()
    res_df = IncrementalStagedAnalysis(
        m_df, [ErectionStage(name="s", add_elements=[100],
                             loads={3: [0, -80e3, 0]},
                             age_at_activation_days=3650.0)],   # 10-yr prop
        initial_active=beam,
        creep=StagedCreep(f_cm=F_CM, chi=0.8, initial_age_days=3.0,   # fresh
                          final_time_days=18250.0)).run()
    prop_df = float(np.linalg.norm(res_df.element_forces[100]))

    assert prop_eq > 0.0 and prop_df > 0.0
    # the young beam creeps more -> sheds load to the stiff old prop
    assert prop_df > prop_eq
    assert abs(prop_df - prop_eq) / prop_eq > 1e-3
