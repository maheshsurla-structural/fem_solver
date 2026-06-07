"""Phase II.14 tests -- cracked transformed section + Branson I_e."""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    ConcreteMaterial,
    CrackedElasticConcrete,
    CrackedSectionProperties,
    branson_I_e,
    cracked_section_properties,
    ec2_mean_curvature,
)
from femsolver.sections import (
    ReinforcementLayout,
    rc_rectangular_section,
)


# ============================================================ CrackedElasticConcrete

class TestCrackedElasticConcrete:
    def test_compression_linear(self):
        c = CrackedElasticConcrete(E=25e9)
        sigma, Et = c.get_response(-0.001)
        assert sigma == pytest.approx(-25e6, rel=1e-12)
        assert Et == pytest.approx(25e9, rel=1e-12)

    def test_tension_zero(self):
        c = CrackedElasticConcrete(E=25e9)
        sigma, Et = c.get_response(0.001)
        assert sigma == 0.0
        assert Et == 0.0

    def test_zero_strain(self):
        c = CrackedElasticConcrete(E=25e9)
        sigma, Et = c.get_response(0.0)
        assert sigma == 0.0
        assert Et == 25e9

    def test_rejects_invalid_E(self):
        with pytest.raises(ValueError):
            CrackedElasticConcrete(E=-1)

    def test_clone_independent(self):
        c1 = CrackedElasticConcrete(E=25e9)
        c2 = c1.clone()
        assert c2.E == c1.E
        assert c2 is not c1


# ============================================================ hand-calc verification

class TestHandCalc:
    """Singly-reinforced beam, classical hand calculation:
    b=300, h=600, d=560, A_s=1500 mm^2 (3 #8)
    f_c'=30 MPa, E_c = 4700*sqrt(30) = 25742 MPa
    n = E_s/E_c = 7.77

    NA from top: b*x^2/2 = n*A_s*(d - x)
        -> 150*x^2 + 11.65e-3 * x - 6.524e-3 = 0
        -> x = 173.3 mm

    I_cr = b*x^3/3 + n*A_s*(d - x)^2
         = 0.3 * 0.173^3 / 3 + 7.77 * 1500e-6 * (0.560 - 0.173)^2
         = 0.000520 + 0.001743
         = 2.263e-3 m^4
    """
    def setup_method(self):
        b, h = 0.3, 0.6
        A_s = 1500e-6
        cover = 0.04
        n_bars = 3
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h,
            bottom_bars=[(A_s / n_bars, "#8")] * n_bars,
            bottom_cover=cover,
        )
        self.sec = rc_rectangular_section(
            b=b, h=h, concrete=cm, reinforcement=rl,
        )
        self.A_s = A_s
        self.b = b
        self.h = h
        self.d = h - cover

    def _hand_calc(self):
        E_c = 4700 * math.sqrt(30) * 1e6
        E_s = 200e9
        n = E_s / E_c
        # b*x^2/2 = n*A_s*(d-x)
        a_q = self.b / 2
        b_q = n * self.A_s
        c_q = -n * self.A_s * self.d
        x = (-b_q + math.sqrt(b_q**2 - 4 * a_q * c_q)) / (2 * a_q)
        I_cr = self.b * x**3 / 3 + n * self.A_s * (self.d - x)**2
        return x, I_cr, E_c

    def test_NA_depth_matches_hand_calc(self):
        x_hand, _, _ = self._hand_calc()
        xs = cracked_section_properties(self.sec, P=0, M_z=100e3,
                                          n_z=8, n_y=40)
        assert xs.neutral_axis_depth_from_top == pytest.approx(x_hand, rel=0.005)

    def test_I_cr_matches_hand_calc(self):
        _, I_cr_hand, _ = self._hand_calc()
        xs = cracked_section_properties(self.sec, P=0, M_z=100e3,
                                          n_z=8, n_y=40)
        assert xs.I_cr_z == pytest.approx(I_cr_hand, rel=0.005)

    def test_I_cr_independent_of_applied_M(self):
        """For linear-elastic cracked section, I_cr should not depend
        on the applied moment (within numerical noise)."""
        xs_low = cracked_section_properties(self.sec, P=0, M_z=80e3,
                                              n_z=8, n_y=40)
        xs_high = cracked_section_properties(self.sec, P=0, M_z=300e3,
                                              n_z=8, n_y=40)
        assert xs_high.I_cr_z == pytest.approx(xs_low.I_cr_z, rel=0.01)


# ============================================================ extreme fibre stresses

class TestExtremeStresses:
    def setup_method(self):
        b, h = 0.3, 0.6
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h,
            bottom_bars=[(510e-6, "#8")] * 3,
            bottom_cover=0.04,
        )
        self.sec = rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)

    def test_compression_at_top_under_sagging_M(self):
        xs = cracked_section_properties(self.sec, P=0, M_z=100e3,
                                          n_z=8, n_y=40)
        assert xs.extreme_compression_strain < 0
        assert xs.extreme_compression_stress < 0

    def test_steel_tension_under_sagging_M(self):
        xs = cracked_section_properties(self.sec, P=0, M_z=100e3,
                                          n_z=8, n_y=40)
        assert xs.max_steel_tensile_strain > 0
        assert xs.max_steel_tensile_stress > 0
        # Sigma_s = E_s * eps_s
        assert xs.max_steel_tensile_stress == pytest.approx(
            xs.E_s * xs.max_steel_tensile_strain, rel=1e-6,
        )

    def test_axial_compression_increases_NA_depth(self):
        """Axial compression pushes the NA down (more compression
        zone)."""
        xs_M = cracked_section_properties(self.sec, P=0, M_z=100e3,
                                            n_z=8, n_y=40)
        xs_PM = cracked_section_properties(self.sec, P=200e3, M_z=100e3,
                                             n_z=8, n_y=40)
        assert xs_PM.neutral_axis_depth_from_top > xs_M.neutral_axis_depth_from_top


# ============================================================ Branson I_e

class TestBranson:
    def test_below_cracking_returns_I_g(self):
        I_g = 5.4e-3
        I_cr = 2.26e-3
        M_cr = 60e3
        # M_a < M_cr
        assert branson_I_e(I_g, I_cr, M_cr, M_a=40e3) == pytest.approx(I_g)
        # M_a = M_cr boundary
        assert branson_I_e(I_g, I_cr, M_cr, M_a=M_cr) == pytest.approx(I_g)

    def test_formula_correct(self):
        I_g = 5.4e-3
        I_cr = 2.26e-3
        M_cr = 60e3
        M_a = 100e3
        ratio = M_cr / M_a
        expected = I_cr + (I_g - I_cr) * ratio**3
        assert branson_I_e(I_g, I_cr, M_cr, M_a) == pytest.approx(expected, rel=1e-12)

    def test_high_M_asymptotes_to_I_cr(self):
        I_g = 5.4e-3
        I_cr = 2.26e-3
        M_cr = 60e3
        # M_a >> M_cr -> I_e ~ I_cr
        I_e = branson_I_e(I_g, I_cr, M_cr, M_a=1000e3)
        assert I_e == pytest.approx(I_cr, abs=(I_g - I_cr) * 0.001)

    def test_monotonically_decreasing(self):
        I_g = 5.4e-3
        I_cr = 2.26e-3
        M_cr = 60e3
        Ms = [100e3, 200e3, 500e3, 1000e3]
        I_es = [branson_I_e(I_g, I_cr, M_cr, M) for M in Ms]
        # As M_a increases, I_e should decrease (more cracking)
        for i in range(len(I_es) - 1):
            assert I_es[i + 1] <= I_es[i]

    def test_capped_at_I_g(self):
        # Edge case: I_cr > I_g would be unphysical, but formula
        # should still cap at I_g
        I_e = branson_I_e(I_g=5e-3, I_cr=6e-3, M_cr=60e3, M_a=100e3)
        assert I_e <= 5e-3 + 1e-12   # capped at I_g

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            branson_I_e(I_g=-1, I_cr=1, M_cr=1, M_a=1)


# ============================================================ EC2 mean curvature

class TestEC2MeanCurvature:
    def test_below_cracking_returns_uncracked(self):
        k = ec2_mean_curvature(kappa_uncracked=0.001, kappa_cracked=0.005,
                                M_cr=100e3, M_a=80e3)
        assert k == pytest.approx(0.001)

    def test_at_cracking_returns_uncracked(self):
        k = ec2_mean_curvature(kappa_uncracked=0.001, kappa_cracked=0.005,
                                M_cr=100e3, M_a=100e3)
        assert k == pytest.approx(0.001)

    def test_well_above_cracking_approaches_cracked(self):
        """For M_a >> M_cr, zeta -> 1, kappa_mean -> kappa_cracked."""
        k = ec2_mean_curvature(kappa_uncracked=0.001, kappa_cracked=0.005,
                                M_cr=50e3, M_a=1000e3)
        # zeta = 1 - 1*(50/1000)^2 = 0.9975
        assert k == pytest.approx(0.9975 * 0.005 + 0.0025 * 0.001, rel=1e-9)

    def test_beta_long_term_increases_mean_curvature(self):
        """EC2: sustained loading uses beta=0.5 to model the gradual
        loss of tension stiffening. ``zeta = 1 - beta*(M_cr/M_a)^2``,
        so smaller beta -> larger zeta -> more weight on the cracked
        (less-stiff) curvature -> larger mean curvature -> larger
        long-term deflection. Verify the math:
            beta=1.0: zeta = 1 - 1*(0.5)^2 = 0.75
            beta=0.5: zeta = 1 - 0.5*(0.5)^2 = 0.875
        """
        k_short = ec2_mean_curvature(0.001, 0.005, M_cr=50e3, M_a=100e3, beta=1.0)
        k_long = ec2_mean_curvature(0.001, 0.005, M_cr=50e3, M_a=100e3, beta=0.5)
        # Long-term beta=0.5 gives larger zeta -> larger mean curvature
        assert k_long > k_short
        # Hand calc
        assert k_short == pytest.approx(0.75 * 0.005 + 0.25 * 0.001)
        assert k_long == pytest.approx(0.875 * 0.005 + 0.125 * 0.001)


# ============================================================ PSC support

class TestPSCCrackedSection:
    """For a PSC section, the cracked-section calc should work but
    note that cracking is delayed by prestress (M_cr is larger)."""
    def test_psc_cracked_section_builds(self):
        from femsolver.sections.section import PrestressTendon
        from femsolver.sections import TendonLayout
        from femsolver.materials.uniaxial import UniaxialBilinear
        b, h = 0.4, 0.8
        cm = ConcreteMaterial(fc_prime=40e6, fy=420e6)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h, top_bars=[(200e-6, "#5")] * 4, top_cover=0.05,
        )
        strand_mat = UniaxialBilinear(E=195e9, sigma_y=1675e6)
        tendons = TendonLayout(tendons=[
            PrestressTendon(z=0, y=-0.34, area=6 * 99e-6,
                              material=strand_mat, f_pe=1100e6),
        ])
        sec = rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)
        sec.prestress = tendons
        # Should solve without error
        xs = cracked_section_properties(
            sec, P=0, M_z=500e3, n_z=6, n_y=30,
        )
        assert xs.I_cr_z is not None
        assert xs.field.converged


# ============================================================ validation

class TestValidation:
    def test_missing_E_c_raises(self):
        from femsolver.sections import rc_rectangular_section, ReinforcementLayout
        b, h = 0.3, 0.6
        # No concrete material attached -> can't read E_c
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h, bottom_bars=[(510e-6, "#8")] * 3, bottom_cover=0.04,
        )
        sec = rc_rectangular_section(b=b, h=h, concrete=None, reinforcement=rl)
        with pytest.raises(ValueError, match="E_c"):
            cracked_section_properties(sec, P=0, M_z=50e3)

    def test_explicit_E_c_works_without_material(self):
        from femsolver.sections import rc_rectangular_section, ReinforcementLayout
        b, h = 0.3, 0.6
        rl = ReinforcementLayout.from_rectangular_layers(
            b=b, h=h, bottom_bars=[(510e-6, "#8")] * 3, bottom_cover=0.04,
        )
        sec = rc_rectangular_section(b=b, h=h, concrete=None, reinforcement=rl)
        xs = cracked_section_properties(
            sec, P=0, M_z=100e3, E_c=25e9, n_z=8, n_y=40,
        )
        assert xs.E_c == 25e9
        assert xs.I_cr_z is not None
