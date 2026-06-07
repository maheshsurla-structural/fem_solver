"""Phase II.3 tests -- parametric section primitives.

Verifies that every factory in
:mod:`femsolver.sections.parametric.factory` produces a unified
:class:`Section` whose gross properties match the textbook closed-form
formulas (Roark, Boresi & Schmidt, AISC Design Guide 9).
"""
from __future__ import annotations

import math

import pytest

from femsolver.sections import (
    Section,
    angle_section,
    channel_section,
    circular_section,
    hollow_circular_section,
    hollow_rect_section,
    i_section,
    rectangular_section,
    t_section,
)


class _DummySteel:
    E = 200e9
    nu = 0.3
    density = 7850.0


# ============================================================ rectangle

class TestRectangle:
    def test_returns_unified_section(self):
        sec = rectangular_section(b=0.3, h=0.6)
        assert isinstance(sec, Section)
        assert sec.family == "rect"
        assert "R 300x600" == sec.name

    def test_area_closed_form(self):
        sec = rectangular_section(b=0.3, h=0.6)
        assert sec.area == pytest.approx(0.18, rel=1e-12)

    def test_I_zz_bh3_over_12(self):
        sec = rectangular_section(b=0.3, h=0.6)
        assert sec.I_zz == pytest.approx(0.3 * 0.6**3 / 12.0, rel=1e-12)
        assert sec.I_yy == pytest.approx(0.6 * 0.3**3 / 12.0, rel=1e-12)

    def test_Z_bh2_over_4(self):
        sec = rectangular_section(b=0.3, h=0.6)
        assert sec.geometry.Z_zz == pytest.approx(0.3 * 0.6**2 / 4.0, rel=1e-12)
        assert sec.geometry.Z_yy == pytest.approx(0.6 * 0.3**2 / 4.0, rel=1e-12)

    def test_torsion_constant_thin_strip(self):
        """For a thin strip (h >> b), J -> (1/3) b^3 h."""
        b, h = 0.01, 0.5
        sec = rectangular_section(b=b, h=h)
        # Roark formula gives 99% of (1/3) b^3 h for h/b = 50
        J_thin_limit = b**3 * h / 3.0
        assert sec.J == pytest.approx(J_thin_limit, rel=0.05)

    def test_torsion_constant_square(self):
        """For a square b = h, Roark gives J ~ 0.141 * b^4."""
        sec = rectangular_section(b=0.1, h=0.1)
        # Roark: J = a*b^3 * (1/3 - 0.21*1*(1 - 1/12)) = b^4 * (1/3 - 0.1925) = 0.1408 b^4
        assert sec.J == pytest.approx(0.1408 * 0.1**4, rel=0.02)

    def test_with_material(self):
        sec = rectangular_section(b=0.3, h=0.6, material=_DummySteel())
        es = sec.elastic_section_3d()
        assert es.EA == pytest.approx(200e9 * 0.18, rel=1e-12)

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            rectangular_section(b=-0.1, h=0.5)


# ============================================================ I-section

class TestISection:
    """Verify against W14x90 (depth 355.6, flange 368.6, tf 18.04, tw 11.18 mm).
    AISC table values (without fillets) should agree to ~2%."""

    def setup_method(self):
        self.sec = i_section(
            h=0.3556, b=0.3686, t_f=0.01804, t_w=0.01118,
        )

    def test_area_within_2pct_aisc(self):
        # AISC: A = 26.5 in^2 = 17097 mm^2 = 0.017097 m^2
        # Our polygon (no fillets) gives A = 0.016871
        assert self.sec.area == pytest.approx(0.01687, rel=0.01)

    def test_I_zz_within_2pct_aisc(self):
        # AISC: Ix = 999 in^4 = 4.158e8 mm^4 = 4.158e-4 m^4
        # Our (no fillets): 4.096e-4
        assert self.sec.I_zz == pytest.approx(4.096e-4, rel=0.01)

    def test_I_yy_within_2pct_aisc(self):
        # AISC: Iy = 362 in^4 = 1.507e8 mm^4 = 1.507e-4 m^4
        # Our (no fillets): 1.506e-4
        assert self.sec.I_yy == pytest.approx(1.506e-4, rel=0.01)

    def test_J_thin_walled_approx(self):
        # J = (2 b tf^3 + (h - tf) tw^3) / 3
        # = (2*0.3686*0.01804^3 + (0.3556-0.01804)*0.01118^3) / 3
        # Expected ~1.6e-6 m^4
        expected = (
            2 * 0.3686 * 0.01804**3
            + (0.3556 - 0.01804) * 0.01118**3
        ) / 3.0
        assert self.sec.J == pytest.approx(expected, rel=1e-10)

    def test_Z_zz_closed_form(self):
        h, b, tf, tw = 0.3556, 0.3686, 0.01804, 0.01118
        expected = b * tf * (h - tf) + tw * (h - 2 * tf)**2 / 4
        assert self.sec.geometry.Z_zz == pytest.approx(expected, rel=1e-12)

    def test_centroid_at_origin_by_symmetry(self):
        cz, cy = self.sec.centroid
        assert cz == pytest.approx(0.0, abs=1e-10)
        assert cy == pytest.approx(0.0, abs=1e-10)

    def test_I_yy_smaller_than_I_zz(self):
        assert self.sec.I_yy < self.sec.I_zz

    def test_name_format(self):
        assert self.sec.name == "I 356x369x18.04x11.18"

    def test_with_material_produces_elastic_3d(self):
        sec = i_section(
            h=0.3556, b=0.3686, t_f=0.01804, t_w=0.01118,
            material=_DummySteel(),
        )
        es = sec.elastic_section_3d()
        # GJ should be in the engineering range (~10^5)
        assert es.GJ > 0
        # And EIz should match
        assert es.EIz == pytest.approx(200e9 * sec.I_zz, rel=1e-10)

    def test_rejects_thin_web(self):
        with pytest.raises(ValueError):
            i_section(h=0.4, b=0.18, t_f=0.014, t_w=0.0)

    def test_rejects_too_thick_flange(self):
        with pytest.raises(ValueError):
            i_section(h=0.04, b=0.18, t_f=0.030, t_w=0.0085)  # 2tf > h


# ============================================================ T-section

class TestTSection:
    def test_area_two_rectangles(self):
        # h=300, b=200, tf=20, tw=10 -- A = b*tf + (h-tf)*tw = 4000 + 2800
        sec = t_section(h=0.3, b=0.2, t_f=0.02, t_w=0.01)
        assert sec.area == pytest.approx(0.2*0.02 + 0.28*0.01, rel=1e-12)

    def test_centroid_offset_toward_flange(self):
        sec = t_section(h=0.3, b=0.2, t_f=0.02, t_w=0.01)
        cz, cy = sec.centroid
        assert cz == pytest.approx(0.0, abs=1e-10)  # symmetric about z
        # Centroid above origin (toward the flange at the top)
        assert cy > 0


# ============================================================ Channel

class TestChannel:
    def test_area_two_flanges_plus_web(self):
        sec = channel_section(h=0.3, b=0.08, t_f=0.012, t_w=0.008)
        # A = 2*(b*tf) + (h-2*tf)*tw = 2*0.00096 + 0.276*0.008
        expected = 2 * 0.08 * 0.012 + (0.3 - 2*0.012) * 0.008
        assert sec.area == pytest.approx(expected, rel=1e-12)

    def test_centroid_offset_toward_web(self):
        sec = channel_section(h=0.3, b=0.08, t_f=0.012, t_w=0.008)
        cz, _ = sec.centroid
        # Web at z=0..tw, flanges extend to z=b. Centroid offset toward z=0
        assert 0 < cz < 0.04  # within the half of [0, b]


# ============================================================ Angle

class TestAngle:
    def test_area_two_rectangles_minus_overlap(self):
        # L 100x100x10: A = a*t + b*t - t^2 = 100*10 + 100*10 - 100 = 1900 mm^2
        sec = angle_section(a=0.1, b=0.1, t=0.01)
        assert sec.area == pytest.approx(0.0019, rel=1e-12)

    def test_centroid_offset_toward_corner(self):
        sec = angle_section(a=0.1, b=0.1, t=0.01)
        cz, cy = sec.centroid
        # For equal-leg angle, centroid at (28.7, 28.7) mm from corner
        # (textbook: x_bar = (a^2 t / 2 + (a-t) t^2 / 2) / A ... 28.7 mm)
        assert cz == pytest.approx(0.02868, rel=1e-3)
        assert cy == pytest.approx(0.02868, rel=1e-3)

    def test_unequal_leg(self):
        sec = angle_section(a=0.15, b=0.1, t=0.01)
        assert sec.area == pytest.approx(0.15*0.01 + 0.09*0.01, rel=1e-12)

    def test_rejects_too_thick(self):
        with pytest.raises(ValueError):
            angle_section(a=0.05, b=0.05, t=0.06)


# ============================================================ Hollow rectangle

class TestHollowRect:
    def test_area_outer_minus_inner(self):
        sec = hollow_rect_section(b=0.2, h=0.1, t=0.006)
        # A = b*h - (b-2t)*(h-2t) = 0.02 - 0.188*0.088
        expected = 0.2 * 0.1 - 0.188 * 0.088
        assert sec.area == pytest.approx(expected, rel=1e-12)

    def test_J_bredt(self):
        """For RHS 200x100x6: J via Bredt = 4 A_m^2 t / s.
        A_m = (200-6)(100-6) = 194*94 = 18236 mm^2 (mean enclosed)
        s = 2*(194 + 94) = 576 mm
        J = 4 * 18236^2 * 6 / 576 = 4 * 332,551,696 * 6 / 576 mm^4 = 13,855,488 mm^4
        """
        sec = hollow_rect_section(b=0.2, h=0.1, t=0.006)
        A_m = 0.194 * 0.094
        s = 2.0 * (0.194 + 0.094)
        expected = 4 * A_m**2 * 0.006 / s
        assert sec.J == pytest.approx(expected, rel=1e-12)

    def test_I_zz_outer_minus_inner(self):
        sec = hollow_rect_section(b=0.2, h=0.1, t=0.006)
        b_i = 0.2 - 2*0.006
        h_i = 0.1 - 2*0.006
        expected = (0.2 * 0.1**3 - b_i * h_i**3) / 12.0
        assert sec.I_zz == pytest.approx(expected, rel=1e-10)


# ============================================================ Circular

class TestCircular:
    def test_area_exact(self):
        sec = circular_section(D=0.5)
        assert sec.area == pytest.approx(math.pi * 0.5**2 / 4, rel=1e-12)

    def test_I_zz_eq_I_yy_by_symmetry(self):
        sec = circular_section(D=0.5)
        assert sec.I_zz == pytest.approx(sec.I_yy, rel=1e-12)

    def test_I_zz_pi_D4_over_64(self):
        sec = circular_section(D=0.5)
        assert sec.I_zz == pytest.approx(math.pi * 0.5**4 / 64, rel=1e-12)

    def test_J_pi_D4_over_32(self):
        sec = circular_section(D=0.5)
        assert sec.J == pytest.approx(math.pi * 0.5**4 / 32, rel=1e-12)

    def test_J_equals_2_I_for_circle(self):
        """Polar moment J = 2 I for a circle."""
        sec = circular_section(D=0.5)
        assert sec.J == pytest.approx(2 * sec.I_zz, rel=1e-12)

    def test_Z_D3_over_6(self):
        sec = circular_section(D=0.5)
        assert sec.geometry.Z_zz == pytest.approx(0.5**3 / 6, rel=1e-12)

    def test_polygon_n_sides_does_not_affect_closed_form(self):
        s16 = circular_section(D=0.5, n_sides=16)
        s128 = circular_section(D=0.5, n_sides=128)
        # Both override .area with the exact closed-form
        assert s16.area == pytest.approx(s128.area, rel=1e-12)
        assert s16.I_zz == pytest.approx(s128.I_zz, rel=1e-12)


# ============================================================ Hollow circular

class TestHollowCircular:
    def test_area_annulus(self):
        sec = hollow_circular_section(D=0.5, t=0.01)
        d = 0.5 - 2 * 0.01
        expected = math.pi * (0.5**2 - d**2) / 4
        assert sec.area == pytest.approx(expected, rel=1e-12)

    def test_I_annulus(self):
        sec = hollow_circular_section(D=0.5, t=0.01)
        d = 0.48
        expected = math.pi * (0.5**4 - d**4) / 64
        assert sec.I_zz == pytest.approx(expected, rel=1e-12)

    def test_J_annulus(self):
        sec = hollow_circular_section(D=0.5, t=0.01)
        d = 0.48
        expected = math.pi * (0.5**4 - d**4) / 32
        assert sec.J == pytest.approx(expected, rel=1e-12)

    def test_rejects_too_thick(self):
        with pytest.raises(ValueError):
            hollow_circular_section(D=0.05, t=0.05)


# ============================================================ Cross-cutting

class TestCrossCutting:
    def test_every_section_has_name_and_family(self):
        secs = [
            rectangular_section(b=0.3, h=0.6),
            i_section(h=0.4, b=0.2, t_f=0.015, t_w=0.01),
            t_section(h=0.3, b=0.2, t_f=0.02, t_w=0.01),
            channel_section(h=0.3, b=0.08, t_f=0.012, t_w=0.008),
            angle_section(a=0.1, b=0.1, t=0.01),
            hollow_rect_section(b=0.2, h=0.1, t=0.006),
            circular_section(D=0.5),
            hollow_circular_section(D=0.5, t=0.01),
        ]
        for s in secs:
            assert s.name, f"section has empty name: {s!r}"
            assert s.family, f"section has empty family: {s!r}"

    def test_every_section_has_positive_A_and_I(self):
        secs = [
            rectangular_section(b=0.3, h=0.6),
            i_section(h=0.4, b=0.2, t_f=0.015, t_w=0.01),
            t_section(h=0.3, b=0.2, t_f=0.02, t_w=0.01),
            channel_section(h=0.3, b=0.08, t_f=0.012, t_w=0.008),
            angle_section(a=0.1, b=0.1, t=0.01),
            hollow_rect_section(b=0.2, h=0.1, t=0.006),
            circular_section(D=0.5),
            hollow_circular_section(D=0.5, t=0.01),
        ]
        for s in secs:
            assert s.area > 0
            assert s.I_zz > 0
            assert s.I_yy > 0
            assert s.J > 0

    def test_elastic_3d_adapter_uses_real_J(self):
        """The J override on parametric geometries should propagate to
        ElasticSection3D.GJ -- so we can't get the placeholder zero."""
        sec = i_section(
            h=0.4, b=0.18, t_f=0.014, t_w=0.0085, material=_DummySteel(),
        )
        es = sec.elastic_section_3d()
        # GJ should be much larger than the 1e-30 floor
        assert es.GJ > 1e3
        # And should equal G * J
        assert es.GJ == pytest.approx(
            (200e9 / (2 * 1.3)) * sec.J, rel=1e-10
        )
