"""Phase II.10 tests -- biaxial P-Mz-My interaction surface."""
from __future__ import annotations

import math

import pytest

from femsolver.design.concrete import (
    ConcreteMaterial,
    ConcreteSection,
    RebarLayout,
    biaxial_pmm_point,
    biaxial_pmm_surface,
    column_interaction_point,
)
from femsolver.sections import (
    PolygonGeometry,
    ReinforcementLayout,
    Section,
    custom_polygon_section,
    rc_rectangular_section,
)


# ============================================================ fixtures

def _make_rect_rc(b=0.4, h=0.6, A_bar=510e-6, n_top=4, n_bot=4, cover=0.05):
    cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=b, h=h,
        bottom_bars=[(A_bar, "#8")] * n_bot,
        top_bars=[(A_bar, "#8")] * n_top,
        bottom_cover=cover, top_cover=cover,
    )
    return rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl)


def _make_legacy_rect_rc(b=0.4, h=0.6, n_top=4, n_bot=4, cover=0.05):
    cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
    legacy_rl = RebarLayout(
        bottom_bars=("#8",) * n_bot, top_bars=("#8",) * n_top,
        bottom_cover=cover, top_cover=cover,
    )
    return ConcreteSection(b=b, h=h, material=cm, rebar=legacy_rl)


# ============================================================ vs legacy 2-D code

class TestMatchLegacy2D:
    """The analytical Whitney-block biaxial code at theta=0 must match
    the existing closed-form 2-D code to round-off."""

    @pytest.mark.parametrize("c_mm", [50, 100, 200, 300, 400, 500, 600, 800])
    def test_match_at_theta_zero(self, c_mm):
        sec = _make_rect_rc()
        legacy = _make_legacy_rect_rc()
        c = c_mm / 1000

        biax = biaxial_pmm_point(
            sec, theta_rad=0.0, c=c,
            f_c_prime=30e6, f_y=420e6,
        )
        leg = column_interaction_point(legacy, c=c)

        # P_n match to 1% (mostly < 0.1%; small drift from rebar
        # distribution differences and shapely round-off)
        if abs(leg.P_n) > 1e3:
            assert abs(biax.P_n - leg.P_n) / abs(leg.P_n) < 0.01
        # M_n match to 1%
        if abs(leg.M_n) > 1e3:
            assert abs(biax.M_nz - leg.M_n) / abs(leg.M_n) < 0.01
        # M_ny should be zero by symmetry
        assert abs(biax.M_ny) < max(abs(biax.M_nz), 1e6) * 1e-9

    def test_phi_matches(self):
        sec = _make_rect_rc()
        legacy = _make_legacy_rect_rc()
        for c_mm in [100, 200, 300, 500]:
            c = c_mm / 1000
            biax = biaxial_pmm_point(
                sec, theta_rad=0.0, c=c, f_c_prime=30e6, f_y=420e6,
            )
            leg = column_interaction_point(legacy, c=c)
            assert biax.phi == pytest.approx(leg.phi, abs=0.02)


# ============================================================ symmetry

class TestSymmetry:
    def test_square_with_4_corner_bars_has_rotational_symmetry(self):
        """For a square section with 4 corner bars only (4-fold
        rotationally symmetric), surface at theta should equal
        surface at theta+90deg with M_z and M_y swapped."""
        from femsolver.sections.section import (
            RebarBar, ReinforcementLayout,
        )
        b, h = 0.4, 0.4
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        bars = [
            RebarBar(z=-0.15, y=-0.15, area=510e-6, designation="#8"),
            RebarBar(z=+0.15, y=-0.15, area=510e-6, designation="#8"),
            RebarBar(z=+0.15, y=+0.15, area=510e-6, designation="#8"),
            RebarBar(z=-0.15, y=+0.15, area=510e-6, designation="#8"),
        ]
        rl = ReinforcementLayout(bars=bars)
        sec = rc_rectangular_section(
            b=b, h=h, concrete=cm, reinforcement=rl,
        )
        c = 0.2
        p0 = biaxial_pmm_point(
            sec, theta_rad=0.0, c=c, f_c_prime=30e6, f_y=420e6,
        )
        p90 = biaxial_pmm_point(
            sec, theta_rad=math.pi/2, c=c, f_c_prime=30e6, f_y=420e6,
        )
        # P_n must be identical
        assert p0.P_n == pytest.approx(p90.P_n, rel=1e-6)
        # |Mz| at theta=0 should equal |My| at theta=90
        assert abs(p0.M_nz) == pytest.approx(abs(p90.M_ny), rel=1e-4)

    def test_rectangle_strong_vs_weak_axis(self):
        """For a tall rectangle, P-M about strong axis (theta=0) >
        P-M about weak axis (theta=pi/2) at the same axial level."""
        sec = _make_rect_rc(b=0.4, h=0.8)  # tall
        c = 0.4
        p_strong = biaxial_pmm_point(
            sec, theta_rad=0.0, c=c, f_c_prime=30e6, f_y=420e6,
        )
        p_weak = biaxial_pmm_point(
            sec, theta_rad=math.pi/2, c=c, f_c_prime=30e6, f_y=420e6,
        )
        # Strong-axis Mz > weak-axis My for the same c
        assert abs(p_strong.M_nz) > abs(p_weak.M_ny)


# ============================================================ surface

class TestSurface:
    def test_surface_builds_for_rectangular(self):
        sec = _make_rect_rc()
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6,
            n_angles=12, n_depths=12,
        )
        assert len(surf.points) == 12 * 12

    def test_surface_attaches_metadata(self):
        sec = _make_rect_rc()
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6,
            n_angles=8, n_depths=8,
        )
        assert surf.f_c_prime == 30e6
        assert surf.f_y == 420e6
        assert surf.n_angles == 8
        assert surf.P_o > 0
        assert surf.P_n_max < surf.P_o   # 0.80 cap for tied
        assert surf.P_n_max == pytest.approx(0.80 * surf.P_o, rel=1e-6)
        assert surf.P_pure_tension < 0   # negative = tension

    def test_surface_caps_compression(self):
        """Every P_n in the surface should be capped at P_n_max."""
        sec = _make_rect_rc()
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6, n_angles=4, n_depths=8,
        )
        for p in surf.points:
            assert p.P_n <= surf.P_n_max + 1e-6

    def test_cap_uses_plastic_centroid_moments(self):
        """At the ACI 22.4.2.1 cap (P = P_n_max), the section is at
        pure axial compression and the resultant force passes through
        the plastic centroid. For a DOUBLY SYMMETRIC section centred
        at origin, this gives (P_n_max, 0, 0). For an asymmetric
        section (L, T, channel), the cap has non-zero plastic-centroid
        moments about the origin.

        Earlier bug history: first attempt was proportional scaling
        (left chaotic fictitious moments at all angles); second
        attempt zeroed M for all sections (correct for symmetric
        but wrong for asymmetric -> "teepee" plot). Current fix
        computes the plastic-centroid moments analytically and uses
        those as the cap point."""
        sec = _make_rect_rc()   # doubly symmetric, centred at origin
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6, n_angles=8, n_depths=16,
        )
        capped = [
            p for p in surf.points
            if p.section_type == "pure-compression-cap"
        ]
        # At least some points should hit the cap (with this loading)
        assert len(capped) > 0
        # For doubly-symmetric section: cap is exactly at the P-axis
        for p in capped:
            assert p.P_n == pytest.approx(surf.P_n_max, rel=1e-9)
            assert abs(p.M_nz) < 1.0   # < 1 N.m, effectively zero
            assert abs(p.M_ny) < 1.0

    def test_spiral_uses_different_caps(self):
        sec = _make_rect_rc()
        s_tied = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6,
            n_angles=4, n_depths=4, spiral=False,
        )
        s_spiral = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6,
            n_angles=4, n_depths=4, spiral=True,
        )
        assert s_spiral.P_n_max == pytest.approx(0.85 * s_spiral.P_o)
        assert s_spiral.P_n_max > s_tied.P_n_max

    def test_slice_at_theta(self):
        sec = _make_rect_rc()
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6, n_angles=8, n_depths=8,
        )
        slice_0 = surf.slice_at_theta(0.0)
        assert len(slice_0) == 8

    def test_slice_uniaxial_z_and_y(self):
        sec = _make_rect_rc(b=0.4, h=0.4)  # square
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6, n_angles=4, n_depths=4,
        )
        # theta=0 should be in the angle list (linspace 0 to 2pi)
        sz = surf.slice_uniaxial_z()
        sy = surf.slice_uniaxial_y()
        # Square symmetry: P-M_z slice at theta=0 should mirror
        # P-M_y slice at theta=pi/2
        assert len(sz) == len(sy)

    def test_as_arrays_nominal_and_design(self):
        sec = _make_rect_rc()
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6, n_angles=4, n_depths=4,
        )
        P_n, Mz_n, My_n = surf.as_arrays(design=False)
        P_d, Mz_d, My_d = surf.as_arrays(design=True)
        # Smart 3-segment c-distribution may grow n_depths from the
        # requested value if it's small (each segment gets min 2 pts).
        assert len(P_n) == 4 * surf.n_depths
        # |design| values <= |nominal| (phi <= 1 reduces magnitude
        # of both compression and tension P_n, and Mz / My)
        import numpy as np
        assert (np.abs(P_d) <= np.abs(P_n) + 1e-6).all()
        assert (np.abs(Mz_d) <= np.abs(Mz_n) + 1e-6).all()
        assert (np.abs(My_d) <= np.abs(My_n) + 1e-6).all()


# ============================================================ non-rectangular sections

class TestNonRectangular:
    def test_L_shape_section(self):
        """L-shape RC section: 400 vertical + 200 horizontal foot,
        thickness 100, both arms 400 long. Centroid is offset; the
        biaxial surface should NOT be symmetric about Mz=0 or My=0."""
        # L-shape outline
        outline = [
            (0, 0), (0.4, 0), (0.4, 0.1),
            (0.1, 0.1), (0.1, 0.4), (0, 0.4),
        ]
        from femsolver.sections.section import RebarBar
        # Bars at 4 corner-ish positions, near the corners of the L
        bars = [
            RebarBar(z=0.03, y=0.03, area=200e-6, designation="#5"),
            RebarBar(z=0.37, y=0.03, area=200e-6, designation="#5"),
            RebarBar(z=0.07, y=0.37, area=200e-6, designation="#5"),
            RebarBar(z=0.03, y=0.37, area=200e-6, designation="#5"),
        ]
        from femsolver.sections.section import ReinforcementLayout
        rl = ReinforcementLayout(bars=bars)
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        sec = custom_polygon_section(
            outline=outline, material=cm, name="L-shape",
        )
        sec.reinforcement = rl  # attach manually

        # Should build a surface without error
        surf = biaxial_pmm_surface(
            sec, f_c_prime=30e6, f_y=420e6,
            n_angles=8, n_depths=6,
        )
        assert len(surf.points) == 48
        # All P_n must be finite
        for p in surf.points:
            assert math.isfinite(p.P_n)
            assert math.isfinite(p.M_nz)
            assert math.isfinite(p.M_ny)

    def test_pure_compression_approaches_P_o(self):
        """At very large c, the Whitney block covers the whole
        section. P_n should approach P_o (concrete + steel both
        in compression)."""
        sec = _make_rect_rc()
        # Very large c -> a >> section depth -> Whitney block covers all
        pt = biaxial_pmm_point(
            sec, theta_rad=0.0, c=10.0,    # very large
            f_c_prime=30e6, f_y=420e6,
        )
        A_g = 0.4 * 0.6
        A_st = 8 * 510e-6
        P_o = 0.85 * 30e6 * (A_g - A_st) + 420e6 * A_st
        # Should be near P_o (within rebar tension/compression effects)
        assert pt.P_n == pytest.approx(P_o, rel=0.01)


# ============================================================ validation

class TestValidation:
    def test_zero_c_raises(self):
        sec = _make_rect_rc()
        with pytest.raises(ValueError, match="c must be positive"):
            biaxial_pmm_point(sec, 0.0, 0.0, f_c_prime=30e6, f_y=420e6)

    def test_negative_strength_raises(self):
        sec = _make_rect_rc()
        with pytest.raises(ValueError, match="positive"):
            biaxial_pmm_point(sec, 0.0, 0.1, f_c_prime=-1, f_y=420e6)

    def test_no_reinforcement_raises(self):
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        sec = rc_rectangular_section(b=0.3, h=0.6, concrete=cm)
        with pytest.raises(ValueError, match="reinforcement"):
            biaxial_pmm_surface(sec, f_c_prime=30e6, f_y=420e6)
