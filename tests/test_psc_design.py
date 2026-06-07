"""Phase B.8 tests -- PSC bridge limit-state design checks.

Covers the fibre-stress engine, AASHTO LRFD + EN 1992 stress limits,
the decompression check, the ULS factored-moment demand, and an
end-to-end chain: tendon -> secondary moment -> PSC limit-state check.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver.design.psc import (
    PscSection,
    aashto_beta1,
    aashto_flexure_capacity,
    aashto_service_check,
    aashto_service_limits,
    aashto_transfer_check,
    aashto_transfer_limits,
    check_psc_stresses,
    ec2_decompression_check,
    ec2_f_ctm,
    ec2_flexure_capacity,
    ec2_service_check,
    ec2_service_limits,
    ec2_transfer_limits,
    psc_extreme_fiber_stresses,
    psc_factored_moment,
    psc_flexure_check,
)

_KSI = 6.894757e6


# ============================================================ section + stresses

class TestSectionAndStresses:
    def test_rectangular_props(self):
        s = PscSection.rectangular(0.4, 1.0)
        assert s.A == pytest.approx(0.4)
        assert s.I == pytest.approx(0.4 * 1.0 ** 3 / 12.0)
        assert s.S_top == pytest.approx(s.I / 0.5)
        assert s.S_bot == pytest.approx(s.I / 0.5)

    def test_service_fiber_stress_hand_calc(self):
        s = PscSection.rectangular(0.4, 1.0)
        r = psc_extreme_fiber_stresses(s, P=2e6, e=0.3, M=400e3)
        assert r.f_top == pytest.approx(2.0e6, rel=1e-9)
        assert r.f_bot == pytest.approx(8.0e6, rel=1e-9)

    def test_transfer_top_tension(self):
        s = PscSection.rectangular(0.4, 1.0)
        r = psc_extreme_fiber_stresses(s, P=2.2e6, e=0.3, M=100e3)
        assert r.f_top == pytest.approx(-2.9e6, rel=1e-9)   # tension
        assert r.f_bot == pytest.approx(13.9e6, rel=1e-9)

    def test_bad_section_raises(self):
        with pytest.raises(ValueError):
            PscSection(A=-1, I=1, y_top=1, y_bot=1)


# ============================================================ AASHTO limits

class TestAashtoLimits:
    def test_transfer_compression(self):
        comp, _ = aashto_transfer_limits(28e6)
        assert comp == pytest.approx(0.60 * 28e6)

    def test_transfer_tension_no_bond_vs_bonded(self):
        _, t_no = aashto_transfer_limits(28e6, bonded_reinforcement=False)
        _, t_bond = aashto_transfer_limits(28e6, bonded_reinforcement=True)
        # bonded allows more tension
        assert t_bond > t_no
        # √-in-ksi form
        assert t_no == pytest.approx(
            min(0.0948 * math.sqrt(28e6 / _KSI) * _KSI, 0.2 * _KSI), rel=1e-9
        )
        assert t_bond == pytest.approx(
            0.24 * math.sqrt(28e6 / _KSI) * _KSI, rel=1e-9
        )

    def test_transfer_tension_cap(self):
        # high f_ci -> the no-bond limit is capped at 0.2 ksi
        _, t = aashto_transfer_limits(80e6, bonded_reinforcement=False)
        assert t == pytest.approx(0.2 * _KSI, rel=1e-9)

    def test_service_compression_transient_toggle(self):
        c_tr, _ = aashto_service_limits(40e6, transient=True)
        c_perm, _ = aashto_service_limits(40e6, transient=False)
        assert c_tr == pytest.approx(0.60 * 40e6)
        assert c_perm == pytest.approx(0.45 * 40e6)

    def test_service_class_T_more_tension_than_U(self):
        _, tU = aashto_service_limits(40e6, prestress_class="U")
        _, tT = aashto_service_limits(40e6, prestress_class="T")
        assert tT == pytest.approx(2.0 * tU, rel=1e-9)   # 0.38 vs 0.19

    def test_bad_class_raises(self):
        with pytest.raises(ValueError):
            aashto_service_limits(40e6, prestress_class="C")


# ============================================================ EN1992 limits

class TestEc2Limits:
    def test_f_ctm_low_grade(self):
        assert ec2_f_ctm(40e6) == pytest.approx(0.30 * 40 ** (2 / 3) * 1e6, rel=1e-9)

    def test_f_ctm_high_grade(self):
        # f_ck = 60 MPa -> log formula
        f_cm = 60 + 8
        expected = 2.12 * math.log(1 + f_cm / 10.0) * 1e6
        assert ec2_f_ctm(60e6) == pytest.approx(expected, rel=1e-9)

    def test_transfer_limits(self):
        comp, tens = ec2_transfer_limits(30e6)
        assert comp == pytest.approx(0.60 * 30e6)
        assert tens == 0.0   # no tension unless allow_tension
        _, tens2 = ec2_transfer_limits(30e6, allow_tension=True)
        assert tens2 == pytest.approx(ec2_f_ctm(30e6))

    def test_service_combinations(self):
        c_char, _ = ec2_service_limits(40e6, combination="characteristic")
        c_qp, _ = ec2_service_limits(40e6, combination="quasi-permanent")
        assert c_char == pytest.approx(0.60 * 40e6)
        assert c_qp == pytest.approx(0.45 * 40e6)

    def test_bad_combination_raises(self):
        with pytest.raises(ValueError):
            ec2_service_limits(40e6, combination="frequent")

    def test_decompression_passes_when_all_compression(self):
        s = PscSection.rectangular(0.4, 1.0)
        # moderate eccentricity + moment: both fibres stay in compression
        # f_top = 5 - 4.5 + 3 = 3.5 MPa; f_bot = 5 + 4.5 - 3 = 6.5 MPa
        chk = ec2_decompression_check(s, P=2e6, e=0.15, M_external=200e3)
        assert chk.f_top > 0 and chk.f_bot > 0
        assert chk.passes

    def test_decompression_fails_with_tension(self):
        s = PscSection.rectangular(0.4, 1.0)
        # large sagging moment -> bottom goes into tension
        chk = ec2_decompression_check(s, P=1e6, e=0.1, M_external=900e3)
        assert chk.f_bot < 0      # tension at bottom
        assert not chk.passes


# ============================================================ secondary consumption

class TestSecondaryConsumption:
    def test_secondary_moment_changes_stresses(self):
        s = PscSection.rectangular(0.4, 1.0)
        base = aashto_service_check(s, P=2e6, e=0.3, M_external=300e3,
                                     M_secondary=0.0, f_c=40e6)
        withsec = aashto_service_check(s, P=2e6, e=0.3, M_external=300e3,
                                        M_secondary=150e3, f_c=40e6)
        # a +M_sec adds compression to the top fibre, relieves the bottom
        assert withsec.f_top > base.f_top
        assert withsec.f_bot < base.f_bot

    def test_check_uses_external_plus_secondary(self):
        s = PscSection.rectangular(0.4, 1.0)
        # M_external + M_secondary should equal a single combined M
        a = check_psc_stresses(s, P=2e6, e=0.3, M_external=300e3,
                               M_secondary=150e3, comp_limit=24e6,
                               tens_limit=3e6)
        b = check_psc_stresses(s, P=2e6, e=0.3, M_external=450e3,
                               M_secondary=0.0, comp_limit=24e6,
                               tens_limit=3e6)
        assert a.f_top == pytest.approx(b.f_top)
        assert a.f_bot == pytest.approx(b.f_bot)


# ============================================================ ULS demand

class TestUlsDemand:
    def test_factored_moment_includes_secondary_at_unity(self):
        Mu = psc_factored_moment(
            factored_external=1.25 * 500e3 + 1.75 * 300e3,
            M_secondary=150e3,
        )
        assert Mu == pytest.approx(1.25 * 500e3 + 1.75 * 300e3 + 150e3)

    def test_secondary_factor_override(self):
        Mu = psc_factored_moment(factored_external=1000e3,
                                  M_secondary=150e3, gamma_secondary=1.2)
        assert Mu == pytest.approx(1000e3 + 1.2 * 150e3)


# ============================================================ end-to-end chain

class TestAutoRebarPerimeter:
    def _rect_section(self, b=0.4, h=0.8):
        from femsolver.sections.parametric import rectangular_section
        from femsolver.materials.elastic import ElasticIsotropic
        conc = ElasticIsotropic(1, E=30e9, nu=0.2, rho=2400.0)
        return rectangular_section(b=b, h=h, material=conc)

    def _voided_section(self):
        from femsolver.sections.parametric import custom_polygon_section
        from femsolver.materials.elastic import ElasticIsotropic
        conc = ElasticIsotropic(1, E=30e9, nu=0.2, rho=2400.0)
        outline = [(-1.0, -0.8), (1.0, -0.8), (1.0, 0.4), (0.4, 0.8), (-1.0, 0.8)]
        hole = [(-0.6, -0.4), (0.4, -0.4), (0.4, 0.3), (-0.6, 0.3)]
        return custom_polygon_section(outline=outline, holes=[hole], material=conc)

    def test_count_and_area(self):
        from femsolver.sections import ReinforcementLayout
        rl = ReinforcementLayout.from_perimeter(
            self._rect_section(), n_bars=8, bar_area=0.0005, cover=0.05)
        assert rl.n_bars == 8
        assert rl.total_area == pytest.approx(8 * 0.0005)

    def test_bars_inside_at_cover(self):
        from shapely.geometry import Point
        from femsolver.sections import ReinforcementLayout
        sec = self._rect_section(0.4, 0.8)
        rl = ReinforcementLayout.from_perimeter(
            sec, n_bars=12, bar_area=0.0005, cover=0.05)
        poly = sec.geometry.polygon
        for b in rl.bars:
            p = Point(b.z, b.y)
            assert poly.covers(p)                              # inside material
            assert poly.exterior.distance(p) == pytest.approx(0.05, abs=1e-6)

    def test_arbitrary_polygon_and_holes(self):
        from shapely.geometry import Point, Polygon
        from femsolver.sections import ReinforcementLayout
        sec = self._voided_section()
        rl = ReinforcementLayout.from_perimeter(
            sec, n_bars=12, bar_area=0.0008, cover=0.06,
            include_holes=True, n_bars_per_hole=8)
        assert rl.n_bars == 20                                  # 12 + 8
        poly = sec.geometry.polygon
        # every bar lies in the material (outside the hole, inside outline)
        for b in rl.bars:
            assert poly.covers(Point(b.z, b.y))

    def test_cover_too_large_raises(self):
        from femsolver.sections import ReinforcementLayout
        with pytest.raises(ValueError):
            ReinforcementLayout.from_perimeter(
                self._rect_section(0.4, 0.8), n_bars=8,
                bar_area=0.0005, cover=0.5)   # > half-width -> vanishes


class TestPscFromSection:
    def _rect_section(self, b=0.4, h=1.0):
        from femsolver.sections.parametric import rectangular_section
        from femsolver.materials.elastic import ElasticIsotropic
        conc = ElasticIsotropic(1, E=30e9, nu=0.2, rho=2400.0)
        return rectangular_section(b=b, h=h, material=conc)

    def test_rectangular_matches_manual(self):
        from femsolver.design.psc import PscSection
        sec = self._rect_section(0.4, 1.0)
        psc = PscSection.from_section(sec, axis="z")
        ref = PscSection.rectangular(0.4, 1.0)
        assert psc.A == pytest.approx(ref.A)
        assert psc.I == pytest.approx(ref.I)
        assert psc.y_top == pytest.approx(ref.y_top)
        assert psc.y_bot == pytest.approx(ref.y_bot)

    def test_asymmetric_section_fibers(self):
        from femsolver.sections.parametric import t_section
        from femsolver.materials.elastic import ElasticIsotropic
        from femsolver.design.psc import PscSection
        conc = ElasticIsotropic(1, E=30e9, nu=0.2, rho=2400.0)
        # a T is vertically asymmetric -> y_top != y_bot
        sec = t_section(h=1.0, b=0.8, t_f=0.2, t_w=0.2, material=conc)
        psc = PscSection.from_section(sec, axis="z")
        assert psc.y_top != pytest.approx(psc.y_bot)
        # fibres span the full depth
        assert psc.y_top + psc.y_bot == pytest.approx(1.0, rel=1e-6)

    def test_axis_y_uses_Iyy(self):
        from femsolver.design.psc import PscSection
        sec = self._rect_section(0.4, 1.0)
        psc_y = PscSection.from_section(sec, axis="y")
        # bending about y -> I_yy = h*b^3/12, fibres at +/- b/2
        assert psc_y.I == pytest.approx(1.0 * 0.4 ** 3 / 12.0, rel=1e-6)
        assert psc_y.y_top == pytest.approx(0.2, rel=1e-6)

    def test_bad_axis_raises(self):
        from femsolver.design.psc import PscSection
        with pytest.raises(ValueError):
            PscSection.from_section(self._rect_section(), axis="x")

    def test_from_section_feeds_service_check(self):
        from femsolver.design.psc import PscSection, ec2_service_check
        psc = PscSection.from_section(self._rect_section(0.4, 1.0), axis="z")
        chk = ec2_service_check(psc, P=2e6, e=0.3, M_external=200e3,
                                M_secondary=50e3, f_ck=40e6)
        assert math.isfinite(chk.f_top) and math.isfinite(chk.f_bot)


class TestUlsFlexureCapacity:
    def test_aashto_beta1(self):
        assert aashto_beta1(28e6) == pytest.approx(0.85)
        assert aashto_beta1(40e6) == pytest.approx(0.85 - 0.05 * 12 / 7, rel=1e-9)
        assert aashto_beta1(80e6) == pytest.approx(0.65)   # floor

    def test_aashto_rectangular_hand_calc(self):
        cap = aashto_flexure_capacity(
            A_ps=1.4e-3, f_pu=1860e6, f_py=1674e6, d_p=0.85, b=0.4, f_c=40e6)
        assert cap.c == pytest.approx(0.2314, abs=2e-4)
        assert cap.f_ps == pytest.approx(1718e6, rel=2e-3)
        assert cap.M_n == pytest.approx(1832e3, rel=2e-3)
        assert cap.phi == pytest.approx(1.0)
        assert cap.controlled == "tension"
        assert cap.section_type == "rectangular"

    def test_mild_steel_increases_Mn(self):
        base = aashto_flexure_capacity(
            A_ps=1.4e-3, f_pu=1860e6, f_py=1674e6, d_p=0.85, b=0.4, f_c=40e6)
        withrebar = aashto_flexure_capacity(
            A_ps=1.4e-3, f_pu=1860e6, f_py=1674e6, d_p=0.85, b=0.4, f_c=40e6,
            A_s=1.5e-3, f_y=500e6, d_s=0.8)
        assert withrebar.M_n > base.M_n

    def test_phi_reduces_for_over_reinforced(self):
        # heavy steel, shallow depth -> large c/d -> not tension-controlled
        cap = aashto_flexure_capacity(
            A_ps=8e-3, f_pu=1860e6, f_py=1674e6, d_p=0.5, b=0.3, f_c=35e6)
        assert cap.phi < 1.0
        assert cap.controlled in ("transition", "compression")

    def test_flanged_path(self):
        # wide thin flange, narrow web, lots of steel -> a > h_f -> flanged
        cap = aashto_flexure_capacity(
            A_ps=1.2e-2, f_pu=1860e6, f_py=1674e6, d_p=1.5,
            b=2.0, f_c=40e6, b_w=0.3, h_f=0.10)
        assert cap.section_type == "flanged"
        assert cap.a > 0.10

    def test_ec2_rectangular(self):
        ec = ec2_flexure_capacity(
            A_p=1.4e-3, f_p01k=1600e6, d_p=0.85, b=0.4, f_ck=40e6)
        assert ec.f_ps == pytest.approx(1600e6 / 1.15, rel=1e-9)   # f_pd
        f_cd = 1.0 * 40e6 / 1.5
        x = (1.4e-3 * 1600e6 / 1.15) / (f_cd * 0.4 * 0.8)
        assert ec.c == pytest.approx(x, rel=1e-9)
        assert ec.phi == 1.0   # gammas inside

    def test_flexure_check_consumes_secondary(self):
        cap = aashto_flexure_capacity(
            A_ps=1.4e-3, f_pu=1860e6, f_py=1674e6, d_p=0.85, b=0.4, f_c=40e6)
        Mu = psc_factored_moment(
            factored_external=1.25 * 600e3 + 1.75 * 450e3, M_secondary=200e3)
        chk = psc_flexure_check(M_u=Mu, capacity=cap)
        assert chk.phi_M_n == pytest.approx(cap.phi_M_n)
        assert chk.DCR == pytest.approx(abs(Mu) / cap.phi_M_n, rel=1e-9)
        # dropping the secondary lowers the demand
        Mu0 = psc_factored_moment(
            factored_external=1.25 * 600e3 + 1.75 * 450e3, M_secondary=0.0)
        chk0 = psc_flexure_check(M_u=Mu0, capacity=cap)
        assert chk0.DCR < chk.DCR

    def test_flexure_check_fails_when_overloaded(self):
        cap = aashto_flexure_capacity(
            A_ps=1.4e-3, f_pu=1860e6, f_py=1674e6, d_p=0.85, b=0.4, f_c=40e6)
        chk = psc_flexure_check(M_u=5000e3, capacity=cap)
        assert not chk.passes
        assert chk.DCR > 1.0

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError):
            aashto_flexure_capacity(A_ps=-1, f_pu=1860e6, f_py=1674e6,
                                    d_p=0.85, b=0.4, f_c=40e6)
        with pytest.raises(ValueError):
            ec2_flexure_capacity(A_p=1e-3, f_p01k=1600e6, d_p=0.85, b=-1,
                                 f_ck=40e6)


class TestEndToEndTendonToLimitState:
    def test_tendon_secondary_feeds_psc_check(self):
        """Build a 2-span continuous PT beam, extract the secondary
        moment from the tendon, and feed it into a PSC service check --
        the full analysis -> secondary -> limit-state pipeline."""
        from femsolver.core.model import Model
        from femsolver.elements.beam import BeamColumn2D
        from femsolver.materials.elastic import ElasticIsotropic
        from femsolver.analysis.linear_static import LinearStaticAnalysis
        from femsolver.results.diagrams import beam_force_diagram
        from femsolver.bridges.tendon import Tendon, tendon_secondary_moment

        Lspan, nps, P, e0 = 20.0, 20, 2.0e6, -0.2
        A, I = 0.4, 0.4 * 1.0 ** 3 / 12.0
        mat = ElasticIsotropic(1, E=34e9, nu=0.2, rho=0.0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        nn = 2 * nps + 1
        for i in range(nn):
            m.add_node(i + 1, i * Lspan / nps, 0.0)
        for i in range(nn - 1):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
        m.fix(1, [1, 1, 0]); m.fix(nps + 1, [0, 1, 0]); m.fix(nn, [0, 1, 0])

        # straight tendon at e0 below the centroid (e in the Tendon's
        # sign convention is local +y; below = negative)
        t = Tendon(nodes=list(range(1, nn + 1)), eccentricity=np.full(nn, e0),
                   area=0.003, jacking_force=P, effective_force=P)
        t.apply_to(m)
        LinearStaticAnalysis(m).run()

        # total prestress moment at the pier, then secondary
        M_total = beam_force_diagram(m.element(nps))["M"][-1]
        # tendon eccentricity magnitude below centroid for the PSC section:
        e_psc = abs(e0)
        M_sec = tendon_secondary_moment(total_moment=M_total, P=P, e=e0)
        assert abs(M_sec) > 1e3        # genuinely non-zero secondary

        # PSC service check at the pier consuming the secondary moment,
        # under some external service moment.
        sec = PscSection.rectangular(0.4, 1.0)
        chk = ec2_service_check(
            sec, P=P, e=e_psc, M_external=250e3, M_secondary=M_sec,
            f_ck=40e6, combination="characteristic",
        )
        # the check ran and produced finite stresses incorporating M_sec
        assert math.isfinite(chk.f_top) and math.isfinite(chk.f_bot)
        # and dropping the secondary changes the result
        chk0 = ec2_service_check(
            sec, P=P, e=e_psc, M_external=250e3, M_secondary=0.0,
            f_ck=40e6, combination="characteristic",
        )
        assert chk.f_top != pytest.approx(chk0.f_top)
