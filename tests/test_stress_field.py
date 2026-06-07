"""Phase II.15 tests -- stress / strain field query + SVG overlay."""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    ConcreteMaterial,
    SectionStressField,
    stress_at_point,
    stress_field,
    stress_field_to_svg,
)
from femsolver.materials.uniaxial import (
    ConcreteKentPark,
    UniaxialBilinear,
    UniaxialElastic,
)
from femsolver.sections import (
    PrestressTendon,
    ReinforcementLayout,
    TendonLayout,
    rc_rectangular_section,
)


def _make_rc():
    b, h = 0.4, 0.6
    cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=b, h=h,
        bottom_bars=[(510e-6, "#8")] * 3,
        top_bars=[(285e-6, "#6")] * 2,
        bottom_cover=0.04, top_cover=0.04,
    )
    return rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)


def _concrete():
    return ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=12e6, eps_cu=0.005)


def _steel():
    return UniaxialBilinear(E=200e9, sigma_y=420e6, b=0.01)


# ============================================================ basic solving

class TestStressFieldBasic:
    def test_zero_loading_zero_field(self):
        sec = _make_rc()
        sf = stress_field(
            sec, P=0.0, M_z=0.0, M_y=0.0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=8, n_y=12,
        )
        assert sf.converged
        # All strains and stresses should be near zero
        assert abs(sf.eps_0) < 1e-6
        assert abs(sf.kappa_z) < 1e-6
        assert abs(sf.kappa_y) < 1e-6
        for f in sf.fibers:
            assert abs(f.eps) < 1e-6

    def test_pure_axial_compression(self):
        """Uniform axial compression -> uniform strain, zero curvature."""
        sec = _make_rc()
        sf = stress_field(
            sec, P=1000e3, M_z=0.0, M_y=0.0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=8, n_y=12,
        )
        assert sf.converged
        # eps_0 should be negative (compression in tension-positive)
        assert sf.eps_0 < 0
        # curvatures near zero
        assert abs(sf.kappa_z) < 1e-4
        assert abs(sf.kappa_y) < 1e-4

    def test_pure_bending_about_z(self):
        sec = _make_rc()
        sf = stress_field(
            sec, P=0.0, M_z=150e3, M_y=0.0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=8, n_y=20,
        )
        assert sf.converged
        # Top fiber compressed, bottom in tension (sagging)
        eps_top = sf.extreme_compression_strain()
        eps_bot = sf.extreme_tension_strain()
        assert eps_top < 0
        assert eps_bot > 0

    def test_biaxial_bending(self):
        """Loading about both axes -> both curvatures non-zero."""
        sec = _make_rc()
        sf = stress_field(
            sec, P=0.0, M_z=150e3, M_y=80e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=12, n_y=12,
        )
        assert sf.converged
        assert abs(sf.kappa_z) > 1e-5
        assert abs(sf.kappa_y) > 1e-5


# ============================================================ plane sections

class TestPlaneSections:
    def test_linear_strain_distribution(self):
        """Plane sections kinematic: strain varies linearly with (z, y)."""
        sec = _make_rc()
        sf = stress_field(
            sec, P=0.0, M_z=200e3, M_y=0.0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=20,
        )
        # Check: eps(z=0, y=h/2) - eps(z=0, y=-h/2) = -h * kappa_z
        eps_top = sf.strain_at(0, 0.3)
        eps_bot = sf.strain_at(0, -0.3)
        assert eps_top - eps_bot == pytest.approx(
            -0.6 * sf.kappa_z, rel=1e-6,
        )


# ============================================================ stress_at_point

class TestStressAtPoint:
    def test_returns_pair(self):
        sec = _make_rc()
        eps, sigma = stress_at_point(
            sec, z=0, y=0.25,
            P=0, M_z=150e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
        )
        assert isinstance(eps, float)
        assert isinstance(sigma, float)
        # Top region under positive M_z -> compression
        assert eps < 0
        assert sigma <= 0   # compression-side

    def test_consistent_with_stress_field(self):
        sec = _make_rc()
        sf = stress_field(
            sec, P=0, M_z=100e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=8, n_y=20,
        )
        eps_from_field = sf.strain_at(0.05, 0.20)
        eps_direct, _ = stress_at_point(
            sec, 0.05, 0.20,
            P=0, M_z=100e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=8, n_y=20,
        )
        # Should be equal (same underlying solve)
        assert eps_from_field == pytest.approx(eps_direct, rel=1e-6)


# ============================================================ cracking

class TestCracking:
    def test_low_moment_few_cracks(self):
        sec = _make_rc()
        sf_low = stress_field(
            sec, P=0, M_z=30e3,    # below cracking
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=20,
        )
        # Low moment: most bottom fibers should be below cracking strain
        n_cracked_low = len(sf_low.cracked_fibers(eps_crack=1.5e-4))
        assert n_cracked_low < 30   # some, but not many

    def test_high_moment_many_cracks(self):
        sec = _make_rc()
        sf_high = stress_field(
            sec, P=0, M_z=300e3,    # well above cracking
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=20,
        )
        n_cracked_high = len(sf_high.cracked_fibers(eps_crack=1.5e-4))
        assert n_cracked_high > 20

    def test_axial_compression_reduces_cracks(self):
        """Adding axial compression delays cracking -> fewer cracked
        fibers at same M."""
        sec = _make_rc()
        sf_M_only = stress_field(
            sec, P=0, M_z=200e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=20,
        )
        sf_with_P = stress_field(
            sec, P=500e3, M_z=200e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=20,
        )
        n_cracked_M = len(sf_M_only.cracked_fibers(1.5e-4))
        n_cracked_PM = len(sf_with_P.cracked_fibers(1.5e-4))
        assert n_cracked_PM < n_cracked_M


# ============================================================ prestress

class TestStressFieldPSC:
    def test_psc_initial_state_no_external_load(self):
        """PSC section at zero external load + zero external moment:
        - Tendon itself in tension (positive eps, sigma)
        - Concrete AT TENDON LEVEL is in compression (transferred force)
        - Concrete far from tendon (e.g. top fiber) may be in tension
          due to the prestress moment (eccentric tendon causes hogging
          moment equivalent that puts top in tension)
        - Net axial = 0 (because P_target = 0)"""
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

        sf = stress_field(
            sec, P=0, M_z=0,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=8, n_y=20,
        )
        # Tendon must be in tension (sigma > 0). Note: FiberState.eps
        # stores the kinematic concrete-side strain at the fiber
        # location; the strand's actual strain has the pre-strain
        # offset added internally by PrestressedUniaxial. So sigma is
        # what we check.
        tendons_in_field = [f for f in sf.fibers if f.kind == "tendon"]
        assert len(tendons_in_field) == 1
        assert tendons_in_field[0].sigma > 0

        # Concrete at tendon level (y near -0.34) must be in compression
        concrete_near_tendon = [
            f for f in sf.fibers
            if f.kind == "concrete" and -0.4 <= f.y <= -0.25
        ]
        assert len(concrete_near_tendon) > 0
        # All concrete near the tendon should be compressed
        compressed_near = sum(1 for f in concrete_near_tendon if f.eps < 0)
        assert compressed_near == len(concrete_near_tendon)


# ============================================================ SVG

class TestStressFieldSvg:
    def test_returns_svg_string(self):
        sec = _make_rc()
        sf = stress_field(
            sec, P=0, M_z=100e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=12,
        )
        svg = stress_field_to_svg(sec, sf)
        assert svg.startswith("<svg")
        assert svg.rstrip().endswith("</svg>")

    def test_svg_includes_fiber_rectangles(self):
        sec = _make_rc()
        sf = stress_field(
            sec, P=0, M_z=100e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=12,
        )
        svg = stress_field_to_svg(sec, sf)
        # Many fiber cells -> many rect elements
        assert svg.count("<rect") >= 30

    def test_svg_shows_cracked_count_at_high_M(self):
        sec = _make_rc()
        sf = stress_field(
            sec, P=0, M_z=300e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=12,
        )
        svg = stress_field_to_svg(sec, sf)
        assert "cracked" in svg

    def test_custom_title(self):
        sec = _make_rc()
        sf = stress_field(
            sec, P=0, M_z=100e3,
            concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
            n_z=4, n_y=12,
        )
        svg = stress_field_to_svg(sec, sf, title="My Beam Loading")
        assert "My Beam Loading" in svg


# ============================================================ validation

class TestValidation:
    def test_invalid_P_convention_raises(self):
        sec = _make_rc()
        with pytest.raises(ValueError, match="P_convention"):
            stress_field(
                sec, P=0, M_z=100e3,
                concrete_uniaxial=_concrete(), steel_uniaxial=_steel(),
                P_convention="bogus",
            )
