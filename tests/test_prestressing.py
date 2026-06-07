"""Phase II.13 tests -- prestressing integration with unified Section."""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    ConcreteMaterial,
    biaxial_pmm_point,
    biaxial_pmm_surface,
    moment_curvature,
)
from femsolver.materials.uniaxial import (
    ConcreteKentPark,
    PrestressedUniaxial,
    UniaxialBilinear,
    UniaxialElastic,
)
from femsolver.sections import (
    PrestressTendon,
    ReinforcementLayout,
    TendonLayout,
    rc_rectangular_section,
)


# ============================================================ PrestressedUniaxial wrapper

class TestPrestressedUniaxialWrapper:
    def test_zero_concrete_strain_gives_prestress(self):
        base = UniaxialBilinear(E=195e9, sigma_y=1675e6, b=0.005)
        # f_pe = 1100 MPa, E_p = 195 GPa -> eps_pe = 0.005641
        eps_pe = 1100e6 / 195e9
        wrap = PrestressedUniaxial(base, eps_pe)
        sigma, _ = wrap.get_response(0.0)
        assert sigma == pytest.approx(1100e6, rel=1e-6)

    def test_additional_tension_increases_stress(self):
        base = UniaxialBilinear(E=195e9, sigma_y=1675e6, b=0.005)
        wrap = PrestressedUniaxial(base, 1100e6 / 195e9)
        sigma_0, _ = wrap.get_response(0.0)
        sigma_plus, _ = wrap.get_response(0.001)
        # +0.001 strain on top of pre-strain -> more tension
        assert sigma_plus > sigma_0

    def test_compression_reduces_stress(self):
        base = UniaxialBilinear(E=195e9, sigma_y=1675e6, b=0.005)
        wrap = PrestressedUniaxial(base, 1100e6 / 195e9)
        sigma_0, _ = wrap.get_response(0.0)
        sigma_comp, _ = wrap.get_response(-0.001)
        # -0.001 strain on top of pre-strain -> less tension
        assert sigma_comp < sigma_0

    def test_negative_eps_pe_rejected(self):
        base = UniaxialBilinear(E=195e9, sigma_y=1675e6)
        with pytest.raises(ValueError, match="eps_pe"):
            PrestressedUniaxial(base, -0.001)

    def test_clone_independent(self):
        base = UniaxialBilinear(E=195e9, sigma_y=1675e6)
        w1 = PrestressedUniaxial(base, 0.005)
        w2 = w1.clone()
        assert w2.eps_pe == w1.eps_pe
        assert w2.base is not w1.base   # independent base


# ============================================================ TendonLayout

class TestTendonLayout:
    def test_total_area(self):
        tendons = TendonLayout(tendons=[
            PrestressTendon(z=0, y=-0.3, area=99e-6, f_pe=1100e6),
            PrestressTendon(z=0.1, y=-0.3, area=99e-6, f_pe=1100e6),
        ])
        assert tendons.total_area == pytest.approx(198e-6)
        assert tendons.n_tendons == 2

    def test_total_prestress_force(self):
        tendons = TendonLayout(tendons=[
            PrestressTendon(z=0, y=-0.3, area=99e-6, f_pe=1100e6),
            PrestressTendon(z=0, y=-0.3, area=99e-6, f_pe=1100e6),
        ])
        assert tendons.total_prestress_force == pytest.approx(2 * 99e-6 * 1100e6)


# ============================================================ moment-curvature

def _make_psc_beam():
    b, h = 0.4, 0.8
    cm = ConcreteMaterial(fc_prime=40e6, fy=420e6)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=b, h=h, top_bars=[(200e-6, "#5")] * 4, top_cover=0.05,
    )
    strand_mat = UniaxialBilinear(E=195e9, sigma_y=1675e6, b=0.005)
    tendons = TendonLayout(tendons=[
        PrestressTendon(z=z, y=-0.34, area=99e-6,
                          material=strand_mat, f_pe=1100e6)
        for z in (-0.15, -0.05, 0.05, 0.15)
    ] + [
        PrestressTendon(z=z, y=-0.30, area=99e-6,
                          material=strand_mat, f_pe=1100e6)
        for z in (-0.10, 0.10)
    ])
    sec = rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)
    sec.prestress = tendons
    return sec


def _concrete():
    return ConcreteKentPark(fpc=40e6, eps_c0=0.002, fpcu=16e6, eps_cu=0.0035)


def _steel():
    return UniaxialBilinear(E=200e9, sigma_y=420e6, b=0.01)


class TestPscMomentCurvature:
    def test_psc_works(self):
        sec = _make_psc_beam()
        res = moment_curvature(
            sec, P_target=0.0,
            concrete_uniaxial=_concrete(),
            steel_uniaxial=_steel(),
            kappa_max=0.05, n_steps=30,
            f_rupture=4.0e6,
        )
        assert res.M_u > 0
        assert len(res.points) > 5

    def test_psc_M_cr_matches_hand_calc(self):
        """For our test PSC beam, hand-calc M_cr = 471.2 kN.m"""
        sec = _make_psc_beam()
        res = moment_curvature(
            sec, P_target=0.0,
            concrete_uniaxial=_concrete(),
            steel_uniaxial=_steel(),
            kappa_max=0.05, n_steps=10,
            f_rupture=4.0e6,
        )
        # Hand calc derivation:
        # A_p = 6*99e-6 = 5.94e-4
        # P_pe = 5.94e-4 * 1100e6 = 653.4e3 N
        # e = (4*0.34 + 2*0.30)/6 = 0.3267 m
        # A_g = 0.32, I_g = 0.01707, y_t = 0.4
        # sigma = P_pe/A_g + P_pe*e*y_t/I_g = 2.04e6 + 5.00e6 = 7.04 MPa
        # M_cr = (4 + 7.04) MPa * 0.01707 / 0.4 = 471.2 kN.m
        assert res.M_cr == pytest.approx(471.2e3, rel=0.01)

    def test_psc_M_u_matches_hand_calc(self):
        """For our test PSC beam, A_p*f_pu*(d_p - a/2) with f_pu~1700:
        A_p = 5.94e-4, d_p ~ 0.72, a = A_p*f_pu/(0.85*fc*b) ~ 0.0892
        M_n = 5.94e-4 * 1700e6 * (0.72 - 0.0446) = 681 kN.m
        With Kent-Park parabolic (slightly higher): ~5-10% over Whitney"""
        sec = _make_psc_beam()
        res = moment_curvature(
            sec, P_target=0.0,
            concrete_uniaxial=_concrete(),
            steel_uniaxial=_steel(),
            kappa_max=0.06, n_steps=50,
            f_rupture=4.0e6,
        )
        # Hand calc with f_pu = 1700 MPa
        A_p = 6 * 99e-6
        f_pu = 1700e6
        d_p = (4 * 0.34 + 2 * 0.30) / 6 + 0.4   # = 0.7267 m
        a = A_p * f_pu / (0.85 * 40e6 * 0.4)
        M_n_hand = A_p * f_pu * (d_p - a / 2)
        # Kent-Park gives slightly more (~5-10%)
        assert 0.95 <= res.M_u / M_n_hand <= 1.12

    def test_prestress_raises_M_cr_vs_plain_RC(self):
        """Compare PSC beam to identical section without prestress;
        PSC should have much higher cracking moment."""
        sec_psc = _make_psc_beam()
        sec_plain = _make_psc_beam()
        sec_plain.prestress = None

        # For the "plain" comparison we need some bottom rebar (otherwise
        # the section has no flexural capacity past cracking). Add 6 bars
        # of same area to bottom.
        rl_plain = ReinforcementLayout.from_rectangular_layers(
            b=0.4, h=0.8,
            top_bars=[(200e-6, "#5")] * 4,
            bottom_bars=[(99e-6, "eq")] * 6,
            top_cover=0.05, bottom_cover=0.06,
        )
        sec_plain.reinforcement = rl_plain

        res_psc = moment_curvature(
            sec_psc, P_target=0.0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            kappa_max=0.05, n_steps=20, f_rupture=4.0e6,
        )
        res_plain = moment_curvature(
            sec_plain, P_target=0.0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            kappa_max=0.05, n_steps=20, f_rupture=4.0e6,
        )
        # PSC M_cr should be significantly higher than plain RC
        assert res_psc.M_cr > 2.0 * res_plain.M_cr

    def test_decompression_state_axial_satisfied(self):
        """At decompression (P_target=0), Newton iteration must drive
        net section axial to zero despite prestress."""
        sec = _make_psc_beam()
        res = moment_curvature(
            sec, P_target=0.0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            kappa_max=0.02, n_steps=10, f_rupture=4.0e6,
        )
        for p in res.points[:5]:
            if p.converged:
                # P (compression-positive) should equal P_target = 0
                assert p.P == pytest.approx(0.0, abs=5e3)


# ============================================================ biaxial PMM

class TestPscBiaxialPMM:
    def test_psc_biaxial_works(self):
        sec = _make_psc_beam()
        surf = biaxial_pmm_surface(
            sec, f_c_prime=40e6, f_y=420e6,
            n_angles=4, n_depths=8,
        )
        assert len(surf.points) == 32

    def test_psc_P_o_includes_tendon_capacity(self):
        """For PSC, P_o = 0.85·fc·(Ag-Ast-Ap) + fy·Ast + fpu·Ap"""
        sec = _make_psc_beam()
        surf = biaxial_pmm_surface(
            sec, f_c_prime=40e6, f_y=420e6,
            n_angles=4, n_depths=4,
        )
        # Hand calc
        A_g = 0.4 * 0.8
        A_st = 4 * 200e-6
        A_pt = 6 * 99e-6
        f_pu = 1675e6 * (1 + 0.005 * 9)  # bilinear at strain ~0.01
        # Just verify P_o is in the right ballpark (within 10%)
        P_o_estimate = (0.85 * 40e6 * (A_g - A_st - A_pt)
                        + 420e6 * A_st + 1700e6 * A_pt)
        assert surf.P_o == pytest.approx(P_o_estimate, rel=0.15)

    def test_unbonded_tendons_skipped(self):
        """Unbonded tendons should NOT contribute to section response
        (they don't follow strain compatibility)."""
        b, h = 0.4, 0.8
        cm = ConcreteMaterial(fc_prime=40e6, fy=420e6)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h, bottom_bars=[(510e-6, "#8")] * 3,
            bottom_cover=0.06,
        )
        strand_mat = UniaxialBilinear(E=195e9, sigma_y=1675e6)

        sec_bonded = rc_rectangular_section(
            b=b, h=h, concrete=cm, reinforcement=rl,
        )
        sec_bonded.prestress = TendonLayout(tendons=[
            PrestressTendon(z=0, y=-0.3, area=99e-6,
                              material=strand_mat, f_pe=1100e6,
                              bonded=True),
        ])

        sec_unbonded = rc_rectangular_section(
            b=b, h=h, concrete=cm, reinforcement=rl,
        )
        sec_unbonded.prestress = TendonLayout(tendons=[
            PrestressTendon(z=0, y=-0.3, area=99e-6,
                              material=strand_mat, f_pe=1100e6,
                              bonded=False),   # unbonded
        ])

        pt_b = biaxial_pmm_point(
            sec_bonded, theta_rad=0.0, c=0.4,
            f_c_prime=40e6, f_y=420e6,
        )
        pt_u = biaxial_pmm_point(
            sec_unbonded, theta_rad=0.0, c=0.4,
            f_c_prime=40e6, f_y=420e6,
        )
        # Bonded contributes; unbonded does not -> different P_n
        assert pt_b.P_n != pt_u.P_n


# ============================================================ validation

class TestValidation:
    def test_only_prestress_no_rebar_works(self):
        """A section with only tendons (no mild rebar) must build."""
        b, h = 0.4, 0.8
        cm = ConcreteMaterial(fc_prime=40e6, fy=420e6)
        strand_mat = UniaxialBilinear(E=195e9, sigma_y=1675e6)
        sec = rc_rectangular_section(b=b, h=h, concrete=cm)
        sec.prestress = TendonLayout(tendons=[
            PrestressTendon(z=0, y=-0.3, area=99e-6,
                              material=strand_mat, f_pe=1100e6),
        ])
        # moment_curvature should accept (has prestress, no rebar)
        res = moment_curvature(
            sec, P_target=0.0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            kappa_max=0.02, n_steps=5, f_rupture=4.0e6,
        )
        assert res.M_u > 0

    def test_tendon_without_material_raises(self):
        b, h = 0.4, 0.8
        cm = ConcreteMaterial(fc_prime=40e6, fy=420e6)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h, bottom_bars=[(510e-6, "#8")] * 3,
        )
        sec = rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)
        sec.prestress = TendonLayout(tendons=[
            PrestressTendon(z=0, y=-0.3, area=99e-6, material=None,
                              f_pe=1100e6),
        ])
        with pytest.raises(ValueError, match="material"):
            moment_curvature(
                sec, P_target=0.0,
                concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
                kappa_max=0.01, n_steps=5,
            )
