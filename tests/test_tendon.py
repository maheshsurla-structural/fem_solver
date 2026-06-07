"""Phase B.4 tests -- Tendon object + equivalent-load apply_to (beams).

Validations
-----------
* **Load balancing**: a sagging parabolic tendon cambers the beam UP and
  its equivalent load cancels a matching downward UDL (8Pa/L²).
* **Straight eccentric tendon**: produces a constant moment exactly P·e.
* **Friction**: P(x) decreases from the jacking end; two-end jacking is
  symmetric; pre-tension is uniform.
* **Secondary (parasitic) moment**: zero in a determinate (single-span)
  beam, non-zero over the interior support of a continuous beam.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D, BeamColumn3D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.io.diagrams import beam_force_diagram
from femsolver.bridges.tendon import (
    Tendon,
    tendon_secondary_forces,
    tendon_secondary_moment,
    tendon_secondary_shear,
)


MAT = ElasticIsotropic(1, E=34e9, nu=0.2, rho=0.0)


def _ss_beam(L=20.0, nel=20, A=0.6, I=0.12):
    m = Model(ndm=2, ndf=3)
    m.add_material(MAT)
    for i in range(nel + 1):
        m.add_node(i + 1, i * L / nel, 0.0)
    for i in range(nel):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), MAT, A, I))
    m.fix(1, [1, 1, 0])
    m.fix(nel + 1, [0, 1, 0])
    return m


def _two_span(Lspan=20.0, nps=20, A=0.6, I=0.12):
    m = Model(ndm=2, ndf=3)
    m.add_material(MAT)
    nn = 2 * nps + 1
    for i in range(nn):
        m.add_node(i + 1, i * Lspan / nps, 0.0)
    for i in range(nn - 1):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), MAT, A, I))
    m.fix(1, [1, 1, 0])
    m.fix(nps + 1, [0, 1, 0])
    m.fix(nn, [0, 1, 0])
    return m, nn, nps


def _parabola(L, nel, a):
    xs = np.array([i * L / nel for i in range(nel + 1)])
    return -4.0 * a * xs * (L - xs) / L ** 2   # 0 at ends, -a at mid (below)


# ============================================================ load balancing

class TestLoadBalancing:
    def test_tendon_only_cambers_up(self):
        L, nel, P, a = 20.0, 20, 2.0e6, 0.3
        m = _ss_beam(L, nel)
        Tendon(nodes=list(range(1, nel + 2)), eccentricity=_parabola(L, nel, a),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m)
        LinearStaticAnalysis(m).run()
        assert m.node(nel // 2 + 1).disp[1] > 0   # up

    def test_balances_matching_udl(self):
        L, nel, P, a = 20.0, 40, 2.0e6, 0.3
        w_bal = 8.0 * P * a / L ** 2
        m = _ss_beam(L, nel)
        Tendon(nodes=list(range(1, nel + 2)), eccentricity=_parabola(L, nel, a),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m)
        for e in m.elements.values():
            e.add_uniform_load(-w_bal)
        LinearStaticAnalysis(m).run()
        d_bal = m.node(nel // 2 + 1).disp[1]
        # reference: UDL alone
        m2 = _ss_beam(L, nel)
        for e in m2.elements.values():
            e.add_uniform_load(-w_bal)
        LinearStaticAnalysis(m2).run()
        d_udl = m2.node(nel // 2 + 1).disp[1]
        assert abs(d_bal) / abs(d_udl) < 0.01   # balanced to within 1%

    def test_finer_mesh_balances_better(self):
        L, P, a = 20.0, 2.0e6, 0.3
        w_bal = 8.0 * P * a / L ** 2

        def residual(nel):
            m = _ss_beam(L, nel)
            Tendon(nodes=list(range(1, nel + 2)),
                   eccentricity=_parabola(L, nel, a),
                   area=0.003, jacking_force=P,
                   effective_force=P).apply_to(m)
            for e in m.elements.values():
                e.add_uniform_load(-w_bal)
            LinearStaticAnalysis(m).run()
            return abs(m.node(nel // 2 + 1).disp[1])

        assert residual(40) < residual(10)


# ============================================================ straight tendon

class TestStraightTendon:
    def test_constant_moment_equals_Pe(self):
        L, nel, P, e0 = 20.0, 20, 2.0e6, -0.2
        m = _ss_beam(L, nel)
        Tendon(nodes=list(range(1, nel + 2)),
               eccentricity=np.full(nel + 1, e0),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m)
        LinearStaticAnalysis(m).run()
        M_mid = beam_force_diagram(m.element(nel // 2 + 1))["M"][0]
        assert M_mid == pytest.approx(P * e0, rel=1e-6)

    def test_primary_moment_is_Pe(self):
        L, nel, P, e0 = 20.0, 10, 1.5e6, -0.15
        m = _ss_beam(L, nel)
        t = Tendon(nodes=list(range(1, nel + 2)),
                   eccentricity=np.full(nel + 1, e0),
                   area=0.003, jacking_force=P, effective_force=P)
        prim = t.primary_moment(m)
        assert prim[5] == pytest.approx(P * e0, rel=1e-9)


# ============================================================ force profile

class TestForceProfile:
    def test_pretension_uniform(self):
        L, nel, P = 20.0, 10, 1.0e6
        m = _ss_beam(L, nel)
        t = Tendon(nodes=list(range(1, nel + 2)),
                   eccentricity=np.zeros(nel + 1), area=0.003,
                   jacking_force=P, tendon_type="pre-tension")
        Pf = t.force_profile(m)
        assert np.allclose(Pf, P)

    def test_friction_decreases_from_jack(self):
        L, nel, P = 20.0, 10, 1.0e6
        m = _ss_beam(L, nel)
        # straight tendon, wobble only (no angle) -> P(x)=P0 exp(-k x)
        t = Tendon(nodes=list(range(1, nel + 2)),
                   eccentricity=np.zeros(nel + 1), area=0.003,
                   jacking_force=P, mu=0.2, wobble_k=0.001,
                   jack_from="start")
        Pf = t.force_profile(m)
        assert Pf[0] == pytest.approx(P, rel=1e-9)
        assert Pf[-1] < Pf[0]
        # closed form: exp(-k L)
        assert Pf[-1] == pytest.approx(P * np.exp(-0.001 * L), rel=1e-6)

    def test_both_ends_symmetric(self):
        L, nel, P = 20.0, 10, 1.0e6
        m = _ss_beam(L, nel)
        t = Tendon(nodes=list(range(1, nel + 2)),
                   eccentricity=np.zeros(nel + 1), area=0.003,
                   jacking_force=P, mu=0.2, wobble_k=0.005,
                   jack_from="both")
        Pf = t.force_profile(m)
        assert np.allclose(Pf, Pf[::-1], rtol=1e-9)
        # min at midspan, max at the jacked ends
        assert Pf[0] == pytest.approx(P, rel=1e-9)
        assert Pf[nel // 2] < Pf[0]

    def test_effective_force_scalar_and_array(self):
        L, nel = 20.0, 10
        m = _ss_beam(L, nel)
        t1 = Tendon(nodes=list(range(1, nel + 2)),
                    eccentricity=np.zeros(nel + 1), area=0.003,
                    jacking_force=1e6, effective_force=8e5)
        assert np.allclose(t1.force_profile(m), 8e5)
        arr = np.linspace(1e6, 9e5, nel + 1)
        t2 = Tendon(nodes=list(range(1, nel + 2)),
                    eccentricity=np.zeros(nel + 1), area=0.003,
                    jacking_force=1e6, effective_force=arr)
        assert np.allclose(t2.force_profile(m), arr)


# ============================================================ secondary moment

class TestSecondaryMoment:
    def test_determinate_has_zero_secondary(self):
        """Single-span beam: total moment == primary, secondary == 0."""
        L, nel, P, e0 = 20.0, 20, 2.0e6, -0.2
        m = _ss_beam(L, nel)
        Tendon(nodes=list(range(1, nel + 2)),
               eccentricity=np.full(nel + 1, e0),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m)
        LinearStaticAnalysis(m).run()
        M_mid = beam_force_diagram(m.element(nel // 2 + 1))["M"][0]
        M_sec = tendon_secondary_moment(total_moment=M_mid, P=P, e=e0)
        assert M_sec == pytest.approx(0.0, abs=1e-3 * abs(P * e0))

    def test_continuous_has_nonzero_secondary(self):
        """Two-span continuous beam with a straight tendon develops a
        non-zero secondary moment over the interior support."""
        Lspan, nps, P, e0 = 20.0, 20, 2.0e6, -0.2
        m, nn, nps = _two_span(Lspan, nps)
        Tendon(nodes=list(range(1, nn + 1)),
               eccentricity=np.full(nn, e0),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m)
        LinearStaticAnalysis(m).run()
        # total moment just left of the interior support (element nps)
        M_total = beam_force_diagram(m.element(nps))["M"][-1]
        M_sec = tendon_secondary_moment(total_moment=M_total, P=P, e=e0)
        assert abs(M_sec) > 0.05 * abs(P * e0)   # genuinely non-zero

    def test_secondary_reactions_determinate_are_zero(self):
        """Single-span (determinate): the self-equilibrated tendon load
        produces zero secondary reactions."""
        L, nel, P, e0 = 20.0, 20, 2.0e6, -0.2
        m = _ss_beam(L, nel)
        t = Tendon(nodes=list(range(1, nel + 2)),
                   eccentricity=np.full(nel + 1, e0),
                   area=0.003, jacking_force=P, effective_force=P)
        reac = tendon_secondary_forces(m, t)
        # vertical secondary reactions ~ 0 at both supports
        assert reac[1][1] == pytest.approx(0.0, abs=1.0)
        assert reac[nel + 1][1] == pytest.approx(0.0, abs=1.0)

    def test_secondary_reactions_continuous(self):
        """Two-span continuous: tendon-only reactions are the secondary
        reactions -- self-equilibrated and matching the secondary moment
        via statics (R_end * Lspan)."""
        Lspan, nps, P, e0 = 20.0, 20, 2.0e6, -0.2
        m, nn, nps = _two_span(Lspan, nps)
        t = Tendon(nodes=list(range(1, nn + 1)),
                   eccentricity=np.full(nn, e0),
                   area=0.003, jacking_force=P, effective_force=P)
        reac = tendon_secondary_forces(m, t)
        supports = [1, nps + 1, nn]
        Ry = [reac[s][1] for s in supports]
        # self-equilibrated
        assert sum(Ry) == pytest.approx(0.0, abs=1.0)
        # interior reaction is non-zero (genuine secondary effect)
        assert abs(Ry[1]) > 1e3
        # secondary moment at the pier from statics == total - P*e
        m.clear_loads()
        t.apply_to(m)
        LinearStaticAnalysis(m).run()
        M_total = beam_force_diagram(m.element(nps))["M"][-1]
        M_sec_split = tendon_secondary_moment(total_moment=M_total, P=P, e=e0)
        M_sec_statics = Ry[0] * Lspan      # end reaction * span
        assert M_sec_statics == pytest.approx(M_sec_split, rel=1e-3)

    def test_secondary_shear_helper(self):
        # straight tendon (slope 0): all total shear is secondary
        assert tendon_secondary_shear(total_shear=500.0, P=2e6, slope=0.0) == 500.0
        # primary shear = P*slope is subtracted
        assert tendon_secondary_shear(
            total_shear=500.0, P=2e6, slope=1e-4
        ) == pytest.approx(500.0 - 2e6 * 1e-4)


# ============================================================ integration + errors

class TestApplyAndErrors:
    def test_factor_scales(self):
        L, nel, P, e0 = 20.0, 10, 2.0e6, -0.2
        m1 = _ss_beam(L, nel)
        Tendon(nodes=list(range(1, nel + 2)), eccentricity=np.full(nel + 1, e0),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m1, factor=1.0)
        LinearStaticAnalysis(m1).run()
        d1 = m1.node(nel // 2 + 1).disp[1]
        m2 = _ss_beam(L, nel)
        Tendon(nodes=list(range(1, nel + 2)), eccentricity=np.full(nel + 1, e0),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m2, factor=0.5)
        LinearStaticAnalysis(m2).run()
        d2 = m2.node(nel // 2 + 1).disp[1]
        assert d2 == pytest.approx(0.5 * d1, rel=1e-9)

    def test_load_pattern_applies(self):
        L, nel, P, e0 = 20.0, 10, 2.0e6, -0.2
        m = _ss_beam(L, nel)
        t = Tendon(nodes=list(range(1, nel + 2)), eccentricity=np.full(nel + 1, e0),
                   area=0.003, jacking_force=P, effective_force=P)
        pat = t.load_pattern()
        pat.apply_to(m, factor=1.0)
        LinearStaticAnalysis(m).run()
        assert m.node(nel // 2 + 1).disp[1] != 0.0

    def test_too_few_nodes_raises(self):
        with pytest.raises(ValueError):
            Tendon(nodes=[1], eccentricity=[0.0], area=0.003, jacking_force=1e6)

    def test_ecc_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            Tendon(nodes=[1, 2, 3], eccentricity=[0.0, 0.0], area=0.003,
                   jacking_force=1e6)

    def test_unsupported_host_raises_notimplemented(self):
        # 3-D truss model (ndm=3, ndf=3) is neither a 2-D nor 3-D frame
        m = Model(ndm=3, ndf=3)
        m.add_material(MAT)
        m.add_node(1, 0, 0, 0)
        m.add_node(2, 1, 0, 0)
        t = Tendon(nodes=[1, 2], eccentricity=[0.0, 0.0], area=0.003,
                   jacking_force=1e6, effective_force=1e6)
        with pytest.raises(NotImplementedError):
            t.apply_to(m)

    def test_unconnected_nodes_raise(self):
        L, nel, P = 20.0, 10, 1e6
        m = _ss_beam(L, nel)
        # nodes 1 and 5 are not directly joined by one element
        t = Tendon(nodes=[1, 5], eccentricity=[0.0, 0.0], area=0.003,
                   jacking_force=P, effective_force=P)
        with pytest.raises(ValueError):
            t.apply_to(m)


# ============================================================ 3-D beam tendons

def _ss_beam_3d(L=20.0, nel=20, A=0.6, Iy=0.12, Iz=0.12, J=0.05):
    """Simply-supported 3-D beam along global x (local ey=+y, ez=+z)."""
    m = Model(ndm=3, ndf=6)
    m.add_material(MAT)
    for i in range(nel + 1):
        m.add_node(i + 1, i * L / nel, 0.0, 0.0)
    for i in range(nel):
        m.add_element(BeamColumn3D(i + 1, (i + 1, i + 2), MAT, A, Iy, Iz, J))
    # pin at left (x,y,z + torsion), roller at right (y,z + torsion)
    m.fix(1, [1, 1, 1, 1, 0, 0])
    m.fix(nel + 1, [0, 1, 1, 1, 0, 0])
    return m


class TestTendon3D:
    def test_ey_matches_2d(self):
        """A 3-D tendon eccentric in local y reproduces the 2-D x-y
        deflection exactly."""
        L, nel, P, a = 20.0, 20, 2.0e6, 0.3
        ecc = _parabola(L, nel, a)
        # 2-D reference
        m2 = _ss_beam(L, nel)
        Tendon(nodes=list(range(1, nel + 2)), eccentricity=ecc, area=0.003,
               jacking_force=P, effective_force=P).apply_to(m2)
        LinearStaticAnalysis(m2).run()
        d2 = m2.node(nel // 2 + 1).disp[1]
        # 3-D, e_y
        m3 = _ss_beam_3d(L, nel)
        Tendon(nodes=list(range(1, nel + 2)), eccentricity=ecc, area=0.003,
               jacking_force=P, effective_force=P).apply_to(m3)
        LinearStaticAnalysis(m3).run()
        d3 = m3.node(nel // 2 + 1).disp[1]
        assert d3 == pytest.approx(d2, rel=1e-9)

    def test_ez_vertical_camber(self):
        """A tendon eccentric in local z cambers the beam up in z, with
        the same magnitude as the equivalent y-plane case."""
        L, nel, P, a = 20.0, 20, 2.0e6, 0.3
        ecc = _parabola(L, nel, a)
        m2 = _ss_beam(L, nel)
        Tendon(nodes=list(range(1, nel + 2)), eccentricity=ecc, area=0.003,
               jacking_force=P, effective_force=P).apply_to(m2)
        LinearStaticAnalysis(m2).run()
        d2 = m2.node(nel // 2 + 1).disp[1]
        m3 = _ss_beam_3d(L, nel)
        Tendon(nodes=list(range(1, nel + 2)),
               eccentricity=np.zeros(nel + 1), eccentricity_z=ecc,
               area=0.003, jacking_force=P, effective_force=P).apply_to(m3)
        LinearStaticAnalysis(m3).run()
        dz = m3.node(nel // 2 + 1).disp[2]
        assert dz == pytest.approx(d2, rel=1e-9)
        assert dz > 0   # up

    def test_straight_ez_gives_My_equal_Pe(self):
        L, nel, P, e0 = 20.0, 20, 2.0e6, -0.2
        m = _ss_beam_3d(L, nel)
        Tendon(nodes=list(range(1, nel + 2)),
               eccentricity=np.zeros(nel + 1),
               eccentricity_z=np.full(nel + 1, e0),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m)
        LinearStaticAnalysis(m).run()
        # local end-force magnitude == |P*e| (the signed deflection is
        # validated by test_ez_vertical_camber)
        My_i = m.element(nel // 2 + 1).end_forces_local[4]
        assert abs(My_i) == pytest.approx(abs(P * e0), rel=1e-6)

    def test_straight_ey_gives_Mz_equal_Pe(self):
        L, nel, P, e0 = 20.0, 20, 2.0e6, -0.2
        m = _ss_beam_3d(L, nel)
        Tendon(nodes=list(range(1, nel + 2)),
               eccentricity=np.full(nel + 1, e0),
               area=0.003, jacking_force=P, effective_force=P).apply_to(m)
        LinearStaticAnalysis(m).run()
        # |M_z| == |P*e|; the x-y plane physics is validated exactly by
        # test_ey_matches_2d. (BeamColumn3D reports local M_z and M_y with
        # opposite sign conventions, the standard right-hand-rule quirk.)
        Mz_i = m.element(nel // 2 + 1).end_forces_local[5]
        assert abs(Mz_i) == pytest.approx(abs(P * e0), rel=1e-6)

    def test_biaxial_is_superposition(self):
        """A tendon eccentric in both y and z deflects in y like the
        y-only case and in z like the z-only case (uncoupled)."""
        L, nel, P, a = 20.0, 20, 2.0e6, 0.3
        ecc = _parabola(L, nel, a)
        nodes = list(range(1, nel + 2))
        # y-only
        my = _ss_beam_3d(L, nel)
        Tendon(nodes=nodes, eccentricity=ecc, area=0.003,
               jacking_force=P, effective_force=P).apply_to(my)
        LinearStaticAnalysis(my).run()
        dy_only = my.node(nel // 2 + 1).disp[1]
        # z-only
        mz = _ss_beam_3d(L, nel)
        Tendon(nodes=nodes, eccentricity=np.zeros(nel + 1),
               eccentricity_z=ecc, area=0.003, jacking_force=P,
               effective_force=P).apply_to(mz)
        LinearStaticAnalysis(mz).run()
        dz_only = mz.node(nel // 2 + 1).disp[2]
        # both
        mb = _ss_beam_3d(L, nel)
        Tendon(nodes=nodes, eccentricity=ecc, eccentricity_z=ecc,
               area=0.003, jacking_force=P, effective_force=P).apply_to(mb)
        LinearStaticAnalysis(mb).run()
        assert mb.node(nel // 2 + 1).disp[1] == pytest.approx(dy_only, rel=1e-9)
        assert mb.node(nel // 2 + 1).disp[2] == pytest.approx(dz_only, rel=1e-9)

    def test_eccentricity_z_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            Tendon(nodes=[1, 2, 3], eccentricity=[0.0, 0.0, 0.0],
                   eccentricity_z=[0.0, 0.0], area=0.003, jacking_force=1e6)
