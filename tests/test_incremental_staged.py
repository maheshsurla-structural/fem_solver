"""Phase B.3 tests -- incremental staged construction (birth + death).

Exact validations
------------------
* **Consistency**: a single stage with every element born at once and
  one load equals the one-shot ``LinearStaticAnalysis``.
* **Element death (falsework removal)**: a propped beam, loaded then
  with its prop removed, reaches *exactly* the un-propped one-shot
  deflection -- the dying prop's locked-in reaction is released onto
  the beam.
* **Element birth (stress-free)**: an element born in a later stage
  with no subsequent load carries zero force; and forces accumulate
  only over the stages where the element is active.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D
from femsolver.elements.truss import Truss2D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.bridges.staged_construction import (
    ErectionStage,
    IncrementalStagedAnalysis,
)


# ============================================================ fixtures

def _ss_beam(nel=8, L=8.0, A=0.05, I=4e-4, E=30e9):
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


def _propped_beam(nel=8, L=8.0, A=0.05, I=4e-4, E=30e9):
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(nel + 1):
        m.add_node(i + 1, i * L / nel, 0.0)
    gnd = nel + 2
    m.add_node(gnd, L / 2, -3.0)
    for i in range(nel):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.add_element(Truss2D(100, (nel // 2 + 1, gnd), mat, A))   # prop
    m.fix(1, [1, 1, 0])
    m.fix(nel + 1, [0, 1, 0])
    m.fix(gnd, [1, 1, 1])
    return m, mat


# ============================================================ consistency

class TestConsistencyWithOneShot:
    def test_single_stage_matches_oneshot(self):
        nel, L, P = 8, 8.0, -100e3
        m, _ = _ss_beam(nel, L)
        stage = ErectionStage(
            name="all", add_elements=list(range(1, nel + 1)),
            loads={nel // 2 + 1: [0, P, 0]},
        )
        IncrementalStagedAnalysis(m, [stage]).run()
        staged = np.array([m.node(t).disp[d]
                           for t in range(1, nel + 2) for d in range(3)])

        m2, _ = _ss_beam(nel, L)
        m2.add_nodal_load(nel // 2 + 1, [0, P, 0])
        LinearStaticAnalysis(m2).run()
        oneshot = np.array([m2.node(t).disp[d]
                            for t in range(1, nel + 2) for d in range(3)])
        assert np.allclose(staged, oneshot, atol=1e-12)

    def test_load_split_across_two_stages_superposes(self):
        """Two stages on the SAME structure summing to one load ==
        one-shot under the full load (linear superposition)."""
        nel, L = 8, 8.0
        m, _ = _ss_beam(nel, L)
        full = list(range(1, nel + 1))
        stages = [
            ErectionStage(name="s1", add_elements=full,
                          loads={nel // 2 + 1: [0, -60e3, 0]}),
            ErectionStage(name="s2", loads={nel // 2 + 1: [0, -40e3, 0]}),
        ]
        IncrementalStagedAnalysis(m, stages).run()
        staged_mid = m.node(nel // 2 + 1).disp[1]
        m2, _ = _ss_beam(nel, L)
        m2.add_nodal_load(nel // 2 + 1, [0, -100e3, 0])
        LinearStaticAnalysis(m2).run()
        assert staged_mid == pytest.approx(m2.node(nel // 2 + 1).disp[1],
                                            rel=1e-9)


# ============================================================ element death

class TestElementDeath:
    def test_falsework_removal_equals_no_prop(self):
        nel, L, P = 8, 8.0, -100e3
        m, _ = _propped_beam(nel, L)
        beam = list(range(1, nel + 1))
        stages = [
            ErectionStage(name="cast+prop+load",
                          add_elements=beam + [100],
                          loads={nel // 2 + 1: [0, P, 0]}),
            ErectionStage(name="remove prop", remove_elements=[100]),
        ]
        IncrementalStagedAnalysis(m, stages).run()
        mid_staged = m.node(nel // 2 + 1).disp[1]

        m2, _ = _ss_beam(nel, L)
        m2.add_nodal_load(nel // 2 + 1, [0, P, 0])
        LinearStaticAnalysis(m2).run()
        mid_noprop = m2.node(nel // 2 + 1).disp[1]
        assert mid_staged == pytest.approx(mid_noprop, rel=1e-9, abs=1e-12)

    def test_prop_carries_load_then_is_released(self):
        nel, L, P = 8, 8.0, -100e3
        m, _ = _propped_beam(nel, L)
        beam = list(range(1, nel + 1))
        stages = [
            ErectionStage(name="s1", add_elements=beam + [100],
                          loads={nel // 2 + 1: [0, P, 0]}),
            ErectionStage(name="s2", remove_elements=[100]),
        ]
        res = IncrementalStagedAnalysis(m, stages).run()
        hist = res.element_force_history[100]
        assert hist[0] is not None          # prop active + loaded in stage 1
        assert np.linalg.norm(hist[0]) > 0  # carried real force
        assert hist[1] is None              # removed in stage 2
        assert 100 not in res.element_forces

    def test_remove_inactive_raises(self):
        nel = 8
        m, _ = _propped_beam(nel)
        beam = list(range(1, nel + 1))
        stages = [
            ErectionStage(name="s1", add_elements=beam),  # prop never born
            ErectionStage(name="s2", remove_elements=[100]),
        ]
        with pytest.raises(ValueError):
            IncrementalStagedAnalysis(m, stages).run()


# ============================================================ element birth

class TestElementBirth:
    def _cantilever_segments(self, n=4, L=4.0, A=0.05, I=4e-4, E=30e9):
        mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        for i in range(n + 1):
            m.add_node(i + 1, i * L / n, 0.0)
        for i in range(n):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
        m.fix(1, [1, 1, 1])  # fixed base (cantilever)
        return m, mat

    def test_born_with_no_load_carries_zero_force(self):
        """An element added in a later stage with no subsequent load is
        stress-free."""
        n = 4
        m, _ = self._cantilever_segments(n)
        stages = [
            ErectionStage(name="s1", add_elements=[1, 2],
                          loads={3: [0, -50e3, 0]}),
            ErectionStage(name="s2", add_elements=[3, 4]),  # born, no load
        ]
        res = IncrementalStagedAnalysis(m, stages).run()
        # elements 3,4 born in s2 with no load -> zero force
        assert np.linalg.norm(res.element_forces[3]) == pytest.approx(0.0, abs=1e-9)
        assert np.linalg.norm(res.element_forces[4]) == pytest.approx(0.0, abs=1e-9)
        # and they were inactive in stage 1's history
        assert res.element_force_history[3][0] is None
        assert res.element_force_history[4][0] is None

    def test_born_element_ignores_pre_birth_deformation(self):
        """Element born in stage 2 accumulates force only from stage-2
        loads, not from the stage-1 deformation it was 'cast into'."""
        n = 4
        m, _ = self._cantilever_segments(n)
        stages = [
            ErectionStage(name="s1", add_elements=[1, 2],
                          loads={3: [0, -50e3, 0]}),
            ErectionStage(name="s2", add_elements=[3, 4],
                          loads={5: [0, -50e3, 0]}),
        ]
        res = IncrementalStagedAnalysis(m, stages).run()
        # element 4 force == response to stage-2 increment only.
        # Cross-check: rebuild with only stage-2 load on the full
        # 4-element cantilever but starting from the stage-1 state is
        # not one-shot; instead verify element 4's force equals the
        # increment-2 contribution K_4 @ du_2.
        du2 = res.u_increments[1]
        e4 = m.element(4)
        dm = m.element_dof_map(e4)
        du_e = np.array([du2[eq] if eq >= 0 else 0.0 for eq in dm])
        f_expected = e4.K_global() @ du_e
        assert np.allclose(res.element_forces[4], f_expected, atol=1e-6)

    def test_staged_differs_from_oneshot(self):
        """Genuine staging (build outward, load each new tip) gives a
        different result than applying all loads to the full structure
        at once -- staging history matters."""
        n = 4
        m, _ = self._cantilever_segments(n)
        stages = [
            ErectionStage(name="s1", add_elements=[1], loads={2: [0, -20e3, 0]}),
            ErectionStage(name="s2", add_elements=[2], loads={3: [0, -20e3, 0]}),
            ErectionStage(name="s3", add_elements=[3], loads={4: [0, -20e3, 0]}),
            ErectionStage(name="s4", add_elements=[4], loads={5: [0, -20e3, 0]}),
        ]
        res = IncrementalStagedAnalysis(m, stages).run()
        tip_staged = m.node(5).disp[1]

        m2, _ = self._cantilever_segments(n)
        for nd in (2, 3, 4, 5):
            m2.add_nodal_load(nd, [0, -20e3, 0])
        LinearStaticAnalysis(m2).run()
        tip_oneshot = m2.node(5).disp[1]
        # both downward, but not equal (history dependence)
        assert tip_staged < 0 and tip_oneshot < 0
        assert abs(tip_staged - tip_oneshot) > 1e-6


# ============================================================ creep factor + errors

class TestStiffnessFactorAndErrors:
    def test_stiffness_factor_softens(self):
        nel, L, P = 8, 8.0, -100e3
        beam = list(range(1, nel + 1))

        m, _ = _ss_beam(nel, L)
        IncrementalStagedAnalysis(m, [ErectionStage(
            name="stiff", add_elements=beam, loads={nel // 2 + 1: [0, P, 0]},
            stiffness_factor=1.0)]).run()
        d_stiff = m.node(nel // 2 + 1).disp[1]

        m2, _ = _ss_beam(nel, L)
        IncrementalStagedAnalysis(m2, [ErectionStage(
            name="soft", add_elements=beam, loads={nel // 2 + 1: [0, P, 0]},
            stiffness_factor=0.5)]).run()
        d_soft = m2.node(nel // 2 + 1).disp[1]
        # half stiffness -> double deflection
        assert d_soft == pytest.approx(2.0 * d_stiff, rel=1e-9)

    def test_mp_constraints_not_supported(self):
        nel = 8
        m, _ = _ss_beam(nel)
        m.equal_dof(retained=4, constrained=6, dofs=[0])
        with pytest.raises(NotImplementedError):
            IncrementalStagedAnalysis(m, [ErectionStage(name="x")])

    def test_unknown_element_raises(self):
        nel = 8
        m, _ = _ss_beam(nel)
        with pytest.raises(ValueError):
            IncrementalStagedAnalysis(
                m, [ErectionStage(name="x", add_elements=[999])]
            ).run()

    def test_empty_stages_raises(self):
        m, _ = _ss_beam()
        with pytest.raises(ValueError):
            IncrementalStagedAnalysis(m, [])
