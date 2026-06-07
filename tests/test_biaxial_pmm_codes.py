"""Phase II.11 tests -- biaxial P-M-M for EC2 and IS 456."""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    ConcreteMaterial,
    StressBlockParams,
    aci_params,
    biaxial_pmm_point,
    biaxial_pmm_point_ec2,
    biaxial_pmm_point_is456,
    biaxial_pmm_surface,
    biaxial_pmm_surface_ec2,
    biaxial_pmm_surface_is456,
    ec2_params,
    is456_params,
)
from femsolver.sections import (
    ReinforcementLayout,
    rc_rectangular_section,
)


def _make_rc(b=0.4, h=0.6, n_top=4, n_bot=4):
    cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=b, h=h,
        bottom_bars=[(510e-6, "#8")] * n_bot,
        top_bars=[(510e-6, "#8")] * n_top,
        bottom_cover=0.05, top_cover=0.05,
    )
    return rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)


# ============================================================ StressBlockParams

class TestParamsFactories:
    def test_aci_params(self):
        p = aci_params(f_c_prime=30e6, f_y=420e6)
        assert p.code == "ACI318"
        assert p.sigma_block == pytest.approx(0.85 * 30e6)
        assert p.eps_cu == pytest.approx(0.003)
        assert p.f_yd == pytest.approx(420e6)
        assert p.apply_phi_table is True

    def test_ec2_params_normal_strength(self):
        # f_ck = 30 MPa: eta=1.0, lambda=0.80, eps_cu3=0.0035
        p = ec2_params(f_ck=30e6, f_yk=500e6)
        assert p.code == "EC2"
        assert p.beta_1 == pytest.approx(0.80)
        # sigma_block = 1.0 * 0.85 * 30 / 1.5 = 17 MPa
        assert p.sigma_block == pytest.approx(17e6, rel=1e-9)
        assert p.eps_cu == pytest.approx(0.0035)
        assert p.f_yd == pytest.approx(500e6 / 1.15)
        assert p.apply_phi_table is False

    def test_ec2_params_high_strength(self):
        # f_ck = 60 MPa: eta < 1, lambda < 0.8
        p = ec2_params(f_ck=60e6, f_yk=500e6)
        assert p.beta_1 < 0.80
        # sigma_block = eta * 0.85 * 60 / 1.5 with eta < 1
        assert p.sigma_block < 0.85 * 60e6 / 1.5

    def test_is456_params(self):
        # f_ck = 30 MPa: sigma_block = 0.36*30/0.84 = 12.857 MPa
        p = is456_params(f_ck=30e6, f_y=415e6)
        assert p.code == "IS456"
        assert p.beta_1 == pytest.approx(0.84)
        assert p.sigma_block == pytest.approx(0.36 * 30e6 / 0.84, rel=1e-9)
        # f_yd = 415 / 1.15 = 0.87 * 415
        assert p.f_yd == pytest.approx(415e6 / 1.15)
        assert p.apply_phi_table is False


# ============================================================ EC2 vs hand calc

class TestEC2HandCalc:
    def test_singly_reinforced_M_Rd(self):
        """EC2 hand calc for a singly-reinforced beam:
        b=300, h=600, d=540, As=1000 mm^2, f_ck=30 MPa, f_yk=500 MPa.

        f_yd = 500/1.15 = 434.78 MPa
        sigma_block = 1.0 * 0.85 * 30/1.5 = 17 MPa
        a (block depth) = (f_yd * A_s) / (sigma_block * b)
                        = 434780 / (17e6 * 0.3) = 0.0853 m
        M_Rd = f_yd * A_s * (d - a/2)
             = 434780 * (0.540 - 0.0427)
             = 216.3 kN.m
        """
        b, h = 0.3, 0.6
        cm = ConcreteMaterial(fc_prime=30e6, fy=500e6)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h,
            bottom_bars=[(1000e-6 / 4, "eq")] * 4,
            top_bars=[],
            bottom_cover=0.06,    # d = 0.54
        )
        sec = rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)

        # Hand calc
        f_yd = 500e6 / 1.15
        sigma_b = 17e6
        A_s = 1000e-6
        d = 0.54
        a = (f_yd * A_s) / (sigma_b * b)
        # In EC2, c = a / lambda where lambda = 0.8
        c_handcalc = a / 0.8
        M_Rd_hand = f_yd * A_s * (d - a / 2)

        # Our EC2 calc at the c that gives pure flexure
        p = biaxial_pmm_point_ec2(
            sec, theta_rad=0.0, c=c_handcalc,
            f_ck=30e6, f_yk=500e6,
        )
        # At this c, the section should be nearly at zero axial
        assert abs(p.P_n) < 50e3  # within 50 kN of zero
        assert p.M_nz == pytest.approx(M_Rd_hand, rel=0.05)


# ============================================================ IS 456 vs hand calc

class TestIS456HandCalc:
    def test_singly_reinforced_M_u(self):
        """IS 456 hand calc for a singly-reinforced beam:
        b=230, h=500, d=460, A_s=1000 mm^2, f_ck=25 MPa, f_y=415 MPa.

        x_u = (0.87 * f_y * A_s) / (0.36 * f_ck * b)
            = (0.87 * 415e6 * 1000e-6) / (0.36 * 25e6 * 0.23)
            = 174.3 mm
        M_u = 0.87 * f_y * A_s * (d - 0.42 * x_u)
            = 360,825 * (0.460 - 0.0732)
            = 139.6 kN.m
        """
        b, h = 0.23, 0.50
        cm = ConcreteMaterial(fc_prime=25e6, fy=415e6)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h,
            bottom_bars=[(500e-6, "eq")] * 2,  # A_s = 1000 mm^2
            top_bars=[],
            bottom_cover=0.04,
        )
        sec = rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)

        x_u = 0.1743   # from hand calc
        # In IS 456, c = x_u (NA depth from extreme compression fiber)
        p = biaxial_pmm_point_is456(
            sec, theta_rad=0.0, c=x_u,
            f_ck=25e6, f_y=415e6,
        )
        assert abs(p.P_n) < 30e3
        assert p.M_nz == pytest.approx(139.6e3, rel=0.05)


# ============================================================ code comparisons

class TestCodeComparison:
    def test_aci_vs_ec2_vs_is456_at_same_section(self):
        """Same section, three codes -> ACI > EC2 > IS 456 nominal
        capacities (partial safety factors built into EC2/IS)."""
        sec = _make_rc()
        c = 0.3
        p_aci = biaxial_pmm_point(sec, 0.0, c, f_c_prime=30e6, f_y=420e6)
        p_ec2 = biaxial_pmm_point_ec2(sec, 0.0, c, f_ck=30e6, f_yk=500e6)
        p_is = biaxial_pmm_point_is456(sec, 0.0, c, f_ck=30e6, f_y=415e6)
        # ACI nominal > EC2 > IS 456 (because EC2/IS bake in gamma factors)
        assert p_aci.P_n > p_ec2.P_n
        assert p_ec2.P_n > p_is.P_n
        # ACI applies separate phi
        assert p_aci.phi < 1.0
        # EC2 and IS use phi=1 (factors built into stress block)
        assert p_ec2.phi == 1.0
        assert p_is.phi == 1.0


# ============================================================ surfaces

class TestEC2Surface:
    def test_builds_surface(self):
        sec = _make_rc()
        surf = biaxial_pmm_surface_ec2(
            sec, f_ck=30e6, f_yk=500e6,
            n_angles=8, n_depths=8,
        )
        assert len(surf.points) == 64

    def test_P_o_uses_gamma_factors(self):
        sec = _make_rc()
        surf = biaxial_pmm_surface_ec2(
            sec, f_ck=30e6, f_yk=500e6,
            n_angles=4, n_depths=4,
        )
        # P_o = 17 MPa * (Ag - Ast) + (500/1.15) MPa * Ast
        A_g = 0.4 * 0.6
        A_st = 8 * 510e-6
        sigma_b = 17e6
        f_yd = 500e6 / 1.15
        P_o_expected = sigma_b * (A_g - A_st) + f_yd * A_st
        assert surf.P_o == pytest.approx(P_o_expected, rel=1e-3)

    def test_phi_is_one_throughout(self):
        """EC2 partial safety factors are baked in; phi = 1 everywhere."""
        sec = _make_rc()
        surf = biaxial_pmm_surface_ec2(
            sec, f_ck=30e6, f_yk=500e6,
            n_angles=4, n_depths=4,
        )
        for p in surf.points:
            assert p.phi == 1.0

    def test_rejects_invalid_strength(self):
        sec = _make_rc()
        with pytest.raises(ValueError, match="positive"):
            biaxial_pmm_surface_ec2(sec, f_ck=-1, f_yk=500e6)


class TestIS456Surface:
    def test_builds_surface(self):
        sec = _make_rc()
        surf = biaxial_pmm_surface_is456(
            sec, f_ck=30e6, f_y=415e6,
            n_angles=8, n_depths=8,
        )
        assert len(surf.points) == 64

    def test_P_o_uses_partial_factors(self):
        sec = _make_rc()
        surf = biaxial_pmm_surface_is456(
            sec, f_ck=30e6, f_y=415e6,
            n_angles=4, n_depths=4,
        )
        # sigma_block = 0.36*fck/0.84 = 0.4286*fck
        # f_yd = 415/1.15
        sigma_b = 0.36 * 30e6 / 0.84
        f_yd = 415e6 / 1.15
        A_g = 0.4 * 0.6
        A_st = 8 * 510e-6
        P_o_expected = sigma_b * (A_g - A_st) + f_yd * A_st
        assert surf.P_o == pytest.approx(P_o_expected, rel=1e-3)

    def test_phi_is_one_throughout(self):
        sec = _make_rc()
        surf = biaxial_pmm_surface_is456(
            sec, f_ck=30e6, f_y=415e6,
            n_angles=4, n_depths=4,
        )
        for p in surf.points:
            assert p.phi == 1.0

    def test_rejects_invalid_strength(self):
        sec = _make_rc()
        with pytest.raises(ValueError, match="positive"):
            biaxial_pmm_surface_is456(sec, f_ck=30e6, f_y=-1)


# ============================================================ backward compat

class TestAciBackwardCompat:
    """The existing ACI biaxial_pmm_surface signature must remain
    unchanged. Phase II.10 tests should still pass."""

    def test_aci_surface_still_works(self):
        sec = _make_rc()
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6,
            n_angles=8, n_depths=8,
        )
        assert len(surf.points) == 64

    def test_aci_point_still_works(self):
        sec = _make_rc()
        p = biaxial_pmm_point(sec, 0.0, 0.3, f_c_prime=30e6, f_y=420e6)
        # Should match the pre-refactor ACI value
        assert p.P_n == pytest.approx(2495e3, rel=0.01)
        assert p.M_nz == pytest.approx(861e3, rel=0.01)
