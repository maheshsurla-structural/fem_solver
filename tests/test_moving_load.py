"""Phase B.1 tests -- general moving-load / influence-line engine.

Validation strategy
--------------------
1. **Closed-form simple-span ILs** -- midspan-moment triangle (peak
   L/4), linear reaction line. Machine-precision agreement.
2. **Independent solver cross-check** -- the IL ordinate at a node must
   equal the response from a *direct* ``LinearStaticAnalysis`` solve
   with a real unit load at that node (different code path).
3. **Two-span continuous beam** -- the case the closed-form code could
   never do: negative (hogging) moment IL over the interior support,
   zero at all supports, symmetric, peak magnitude 0.0962·L; interior
   reaction IL area = 1.25·L.
4. **Vehicle convolution** -- agreement with the existing
   ``max_response_for_moving_load`` and a structurally-sane AASHTO
   HL-93 envelope.
5. **MP-constraint fallback path** -- engine result matches direct
   solves on a constrained model.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D
from femsolver.io.diagrams import beam_force_diagram
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.bridges.influence import (
    aashto_hl93_truck,
    max_response_for_moving_load,
)
from femsolver.bridges.moving_load import (
    BeamForce,
    Displacement,
    InfluenceLine,
    InfluenceLineEngine,
    Lane,
    Reaction,
    aashto_hl93_envelope,
    lane_load_response,
    moving_load_envelope,
)


# ============================================================ fixtures

def _ss_beam(L=10.0, nel=10, A=0.5, I=0.05, E=30e9):
    """Simply-supported beam: pin at left, roller at right."""
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=2400.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(nel + 1):
        m.add_node(i + 1, i * L / nel, 0.0)
    for i in range(nel):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 0])
    m.fix(nel + 1, [0, 1, 0])
    return m


def _two_span(Lspan=10.0, nelps=10, A=0.5, I=0.05, E=30e9):
    """Two equal spans, supports at x=0, Lspan, 2*Lspan."""
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=2400.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    nnode = 2 * nelps + 1
    for i in range(nnode):
        m.add_node(i + 1, i * Lspan / nelps, 0.0)
    for i in range(nnode - 1):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 0])
    m.fix(nelps + 1, [0, 1, 0])
    m.fix(nnode, [0, 1, 0])
    return m, nnode, nelps


# ============================================================ InfluenceLine object

class TestInfluenceLineObject:
    def test_callable_interpolates(self):
        il = InfluenceLine(stations=[0, 1, 2], values=[0, 2, 0])
        assert il(0.5) == pytest.approx(1.0)
        assert il(1.5) == pytest.approx(1.0)

    def test_zero_outside_lane(self):
        il = InfluenceLine(stations=[0, 1, 2], values=[1, 2, 1])
        assert il(-0.5) == 0.0
        assert il(2.5) == 0.0

    def test_sorts_by_station(self):
        il = InfluenceLine(stations=[2, 0, 1], values=[20, 0, 10])
        assert np.allclose(il.stations, [0, 1, 2])
        assert np.allclose(il.values, [0, 10, 20])

    def test_integrate_positive_only(self):
        # triangle going negative then positive
        il = InfluenceLine(stations=[0, 1, 2], values=[-1, 0, 1])
        pos = il.integrate(sign="positive")
        neg = il.integrate(sign="negative")
        assert pos == pytest.approx(0.5, rel=1e-3)
        assert neg == pytest.approx(-0.5, rel=1e-3)

    def test_needs_two_stations(self):
        with pytest.raises(ValueError):
            InfluenceLine(stations=[0], values=[1])


# ============================================================ simple-span ILs

class TestSimpleSpanInfluenceLines:
    def test_midspan_moment_triangle(self):
        L, nel = 10.0, 10
        m = _ss_beam(L, nel)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        # element nel//2 has node-j at midspan (x = L/2)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nel // 2, component="M", end="j")
        )
        # closed form: triangle, ordinate x/2 (x<=L/2), (L-x)/2 (x>=L/2)
        xq = il.stations
        exact = np.where(xq <= L / 2, xq / 2.0, (L - xq) / 2.0)
        assert np.allclose(np.abs(il.values), exact, atol=1e-9)
        # peak L/4
        assert np.max(np.abs(il.values)) == pytest.approx(L / 4, rel=1e-9)

    def test_left_reaction_linear(self):
        L, nel = 10.0, 10
        m = _ss_beam(L, nel)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        il = eng.influence_line(lane, Reaction(node_tag=1, dof=1))
        xq = il.stations
        exact = (L - xq) / L
        assert np.allclose(il.values, exact, atol=1e-9)
        assert il.values[0] == pytest.approx(1.0)
        assert il.values[-1] == pytest.approx(0.0, abs=1e-12)

    def test_deflection_IL_symmetric(self):
        m = _ss_beam(10.0, 10)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, 12)), load_dof=1)
        il = eng.influence_line(lane, Displacement(node_tag=6, dof=1))
        # Maxwell-Betti: midspan deflection IL is symmetric, peaks at mid
        assert np.allclose(il.values, il.values[::-1], atol=1e-12)
        assert int(np.argmax(np.abs(il.values))) == 5


# ============================================================ independent cross-check

class TestAgainstDirectSolve:
    """The IL ordinate at a node must equal a direct LinearStaticAnalysis
    solve with a real unit load there -- an independent code path."""

    def _direct_moment(self, model, load_node, elem_tag, end_index):
        model.clear_loads()
        model.add_nodal_load(load_node, [0.0, -1.0, 0.0])  # unit downward
        LinearStaticAnalysis(model).run()
        diag = beam_force_diagram(model.element(elem_tag), n_points=3)
        model.clear_loads()
        return diag["M"][end_index]

    def test_ss_moment_matches_direct(self):
        L, nel = 10.0, 10
        m = _ss_beam(L, nel)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nel // 2, component="M", end="j")
        )
        # check several interior nodes
        for node in (3, 4, 6, 8):
            direct = self._direct_moment(_ss_beam(L, nel), node, nel // 2, -1)
            # IL station index of this node:
            idx = node - 1
            assert il.values[idx] == pytest.approx(direct, rel=1e-6, abs=1e-9)

    def test_continuous_moment_matches_direct(self):
        m, nnode, nelps = _two_span()
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nnode + 1)), load_dof=1)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nelps, component="M", end="j")
        )
        for node in (4, 6, 9, 14, 16):
            mm, _, _ = _two_span()
            direct = self._direct_moment(mm, node, nelps, -1)
            assert il.values[node - 1] == pytest.approx(
                direct, rel=1e-6, abs=1e-9
            )


# ============================================================ continuous beam

class TestTwoSpanContinuous:
    def test_interior_support_moment_IL(self):
        Lspan = 10.0
        m, nnode, nelps = _two_span(Lspan=Lspan)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nnode + 1)), load_dof=1)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nelps, component="M", end="j")
        )
        # zero at all three supports
        assert il.values[0] == pytest.approx(0.0, abs=1e-9)
        assert il.values[nelps] == pytest.approx(0.0, abs=1e-9)
        assert il.values[-1] == pytest.approx(0.0, abs=1e-9)
        # negative (hogging) everywhere between supports
        interior = np.delete(il.values, [0, nelps, len(il.values) - 1])
        assert np.all(interior < 1e-9)
        # symmetric about the interior support
        assert np.allclose(il.values, il.values[::-1], atol=1e-6)
        # peak magnitude ~ 0.0962 * Lspan (classic two-equal-span value)
        assert np.max(np.abs(il.values)) == pytest.approx(
            0.0962 * Lspan, rel=0.03
        )

    def test_interior_reaction_IL_area_is_1_25L(self):
        Lspan = 10.0
        m, nnode, nelps = _two_span(Lspan=Lspan)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nnode + 1)), load_dof=1)
        il = eng.influence_line(lane, Reaction(node_tag=nelps + 1, dof=1))
        # peak exactly 1.0 at the interior support
        assert il.max_value == pytest.approx(1.0, abs=1e-9)
        # area under IL = 1.25 * Lspan (R_B for UDL on two equal spans)
        area = il.integrate(sign="all")
        assert area == pytest.approx(1.25 * Lspan, rel=0.01)


# ============================================================ vehicle convolution

class TestVehicleEnvelope:
    def test_matches_existing_convolution(self):
        L, nel = 20.0, 20
        m = _ss_beam(L, nel)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nel // 2, component="M", end="j")
        )
        mine = moving_load_envelope(il, aashto_hl93_truck())
        existing, _ = max_response_for_moving_load(
            moving_load=aashto_hl93_truck(), influence_line=il, L=L
        )
        assert mine["max"] == pytest.approx(existing, rel=1e-6)

    def test_hl93_envelope_structure(self):
        L, nel = 20.0, 20
        m = _ss_beam(L, nel)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nel // 2, component="M", end="j")
        )
        env = aashto_hl93_envelope(il, im=0.33)
        # max sagging moment is positive; vehicular includes 1.33 IM
        assert env["max"] > 0
        assert env["vehicular_max"] > 0
        assert env["lane_max"] > 0
        # total = vehicular + lane
        assert env["max"] == pytest.approx(
            env["vehicular_max"] + env["lane_max"], rel=1e-9
        )
        # truck governs over tandem on a 20 m span
        assert env["governing_max"] == "truck"

    def test_im_increases_effect(self):
        L, nel = 20.0, 20
        m = _ss_beam(L, nel)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nel // 2, component="M", end="j")
        )
        env0 = aashto_hl93_envelope(il, im=0.0, include_lane=False)
        env33 = aashto_hl93_envelope(il, im=0.33, include_lane=False)
        assert env33["max"] == pytest.approx(env0["max"] * 1.33, rel=1e-6)

    def test_lane_load_positive_area(self):
        L, nel = 10.0, 10
        m = _ss_beam(L, nel)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nel // 2, component="M", end="j")
        )
        # midspan-moment IL is all-positive triangle, area = L^2/8... no:
        # area = (1/2)*base*height = (1/2)*L*(L/4) = L^2/8 = 12.5 for L=10
        res = lane_load_response(il, w=1.0)
        assert res["max"] == pytest.approx(L ** 2 / 8.0, rel=1e-3)
        assert res["min"] == pytest.approx(0.0, abs=1e-6)


# ============================================================ MP-constraint fallback

class TestMPConstraintFallback:
    def test_fallback_matches_direct_on_constrained_model(self):
        """Add an MP constraint -> engine uses the per-station
        LinearStaticAnalysis fallback. Its IL must still match direct
        solves on the same (constrained) model."""
        L, nel = 10.0, 10

        def build():
            m = _ss_beam(L, nel)
            # harmless axial tie between two interior nodes -> forces MP path
            m.equal_dof(retained=4, constrained=8, dofs=[0])
            return m

        m = build()
        assert m.mp_constraints  # ensure constraint present
        eng = InfluenceLineEngine(m)
        assert eng._has_mp is True
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        il = eng.influence_line(
            lane, BeamForce(element_tag=nel // 2, component="M", end="j")
        )
        # cross-check at interior nodes via direct solve on a fresh model
        for node in (3, 5, 7):
            mm = build()
            mm.clear_loads()
            mm.add_nodal_load(node, [0.0, -1.0, 0.0])
            LinearStaticAnalysis(mm).run()
            direct = beam_force_diagram(
                mm.element(nel // 2), n_points=3
            )["M"][-1]
            assert il.values[node - 1] == pytest.approx(
                direct, rel=1e-6, abs=1e-9
            )


# ============================================================ Lane + errors

class TestLaneAndErrors:
    def test_auto_stations_from_geometry(self):
        m = _ss_beam(10.0, 10)
        lane = Lane(node_tags=list(range(1, 12)), load_dof=1)
        st = lane.resolve_stations(m)
        assert np.allclose(st, np.linspace(0, 10, 11))

    def test_explicit_stations(self):
        m = _ss_beam(10.0, 10)
        lane = Lane(node_tags=[1, 6, 11], stations=[0, 5, 10], load_dof=1)
        assert np.allclose(lane.resolve_stations(m), [0, 5, 10])

    def test_lane_needs_two_nodes(self):
        with pytest.raises(ValueError):
            Lane(node_tags=[1])

    def test_beam_force_bad_component(self):
        with pytest.raises(ValueError):
            BeamForce(element_tag=1, component="Q")

    def test_multi_response_single_traversal(self):
        """Asking for several responses returns them all consistently."""
        L, nel = 10.0, 10
        m = _ss_beam(L, nel)
        eng = InfluenceLineEngine(m)
        lane = Lane(node_tags=list(range(1, nel + 2)), load_dof=1)
        ils = eng.influence_lines(lane, {
            "M": BeamForce(element_tag=nel // 2, component="M", end="j"),
            "R": Reaction(node_tag=1, dof=1),
            "d": Displacement(node_tag=6, dof=1),
        })
        assert set(ils) == {"M", "R", "d"}
        # consistency: the single-response call gives the same thing
        il_single = eng.influence_line(lane, Reaction(node_tag=1, dof=1))
        assert np.allclose(ils["R"].values, il_single.values)
