"""Phase II.2 tests -- Section ABC + Geometry + SectionLibrary foundation.

These tests cover the Theme II.2 deliverables:

* polygon geometric primitives (Shoelace, centroid, second moments)
* :class:`PolygonGeometry` -- rectangle, hollow rectangle, custom polygon
* :class:`Section` -- composition, gross-property pass-through, elastic
  adapter
* :class:`SectionLibrary` -- registry register / get / list-family
"""
from __future__ import annotations

import math

import pytest

from femsolver.sections import (
    Geometry,
    MaterialZone,
    PolygonGeometry,
    Section,
    SectionLibrary,
    polygon_centroid,
    polygon_second_moments,
    shoelace_area,
)


# ============================================================ low-level helpers

class TestLowLevelHelpers:
    def test_shoelace_rectangle(self):
        # Unit square CCW
        v = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert shoelace_area(v) == pytest.approx(1.0)

    def test_shoelace_cw_negative(self):
        v = [(0, 0), (0, 1), (1, 1), (1, 0)]
        assert shoelace_area(v) == pytest.approx(-1.0)

    def test_centroid_offset_rectangle(self):
        # 2x4 rectangle centered at (5, 7)
        v = [(4, 5), (6, 5), (6, 9), (4, 9)]
        cz, cy = polygon_centroid(v)
        assert cz == pytest.approx(5.0)
        assert cy == pytest.approx(7.0)

    def test_second_moments_rectangle(self):
        """For an a x b rectangle centred at origin:
        I_zz = a * b^3 / 12, I_yy = a^3 * b / 12 about centroid."""
        a, b = 2.0, 4.0  # width-z, height-y
        v = [(-a/2, -b/2), (a/2, -b/2), (a/2, b/2), (-a/2, b/2)]
        I_yy_0, I_zz_0, I_yz_0 = polygon_second_moments(v)
        # centroid at origin -> moments about origin == moments about centroid
        assert I_zz_0 == pytest.approx(a * b**3 / 12.0)
        assert I_yy_0 == pytest.approx(a**3 * b / 12.0)
        assert I_yz_0 == pytest.approx(0.0, abs=1e-10)


# ============================================================ PolygonGeometry

class TestPolygonGeometryRectangle:
    def test_area_matches_closed_form(self):
        g = PolygonGeometry.rectangle(width=0.3, height=0.6)
        assert g.area == pytest.approx(0.18, rel=1e-12)

    def test_centroid_at_origin(self):
        g = PolygonGeometry.rectangle(width=0.3, height=0.6)
        cz, cy = g.centroid
        assert cz == pytest.approx(0.0, abs=1e-12)
        assert cy == pytest.approx(0.0, abs=1e-12)

    def test_I_zz_matches_closed_form(self):
        # 300 wide x 600 deep: I_zz = b * h^3 / 12 = 0.3 * 0.6^3 / 12 = 0.0054
        g = PolygonGeometry.rectangle(width=0.3, height=0.6)
        assert g.I_zz == pytest.approx(0.3 * 0.6**3 / 12.0, rel=1e-12)
        assert g.I_yy == pytest.approx(0.6 * 0.3**3 / 12.0, rel=1e-12)
        assert g.I_yz == pytest.approx(0.0, abs=1e-12)

    def test_extreme_fibres(self):
        g = PolygonGeometry.rectangle(width=0.3, height=0.6)
        assert g.c_top == pytest.approx(0.3, rel=1e-12)
        assert g.c_bot == pytest.approx(0.3, rel=1e-12)
        assert g.depth == pytest.approx(0.6, rel=1e-12)
        assert g.width == pytest.approx(0.3, rel=1e-12)

    def test_section_modulus(self):
        # S_zz = I_zz / c_top = 0.0054 / 0.3 = 0.018
        g = PolygonGeometry.rectangle(width=0.3, height=0.6)
        assert g.S_zz_top == pytest.approx(0.018, rel=1e-12)

    def test_plastic_modulus_zz_rectangle(self):
        """For a rectangle: Z_zz = b * h^2 / 4."""
        b, h = 0.3, 0.6
        g = PolygonGeometry.rectangle(width=b, height=h)
        Z_expected = b * h**2 / 4
        # Bisection-based Z is only good to a few digits; rectangle is exact
        assert g.Z_zz == pytest.approx(Z_expected, rel=1e-6)


class TestPolygonGeometryHollowRectangle:
    def test_area_outer_minus_inner(self):
        g = PolygonGeometry.hollow_rectangle(0.3, 0.6, 0.2, 0.5)
        # 0.3*0.6 - 0.2*0.5 = 0.18 - 0.10 = 0.08
        assert g.area == pytest.approx(0.08, rel=1e-12)

    def test_I_zz_outer_minus_inner(self):
        # I_zz = b_o * h_o^3 / 12 - b_i * h_i^3 / 12
        g = PolygonGeometry.hollow_rectangle(0.3, 0.6, 0.2, 0.5)
        expected = (0.3 * 0.6**3 - 0.2 * 0.5**3) / 12.0
        assert g.I_zz == pytest.approx(expected, rel=1e-10)

    def test_centroid_at_origin(self):
        g = PolygonGeometry.hollow_rectangle(0.3, 0.6, 0.2, 0.5)
        cz, cy = g.centroid
        assert cz == pytest.approx(0.0, abs=1e-12)
        assert cy == pytest.approx(0.0, abs=1e-12)


class TestPolygonGeometryCustom:
    def test_l_section_basic(self):
        """L-shape made of two rectangles: 100x200 vertical + 100x60 horizontal.
        Total area: 100*200 + 100*60 - 0 (no overlap) but we describe as one
        polygon."""
        # Vertices in mm, then scale to m
        s = 0.001
        v = [
            (0, 0), (200 * s, 0), (200 * s, 60 * s),
            (100 * s, 60 * s), (100 * s, 200 * s), (0, 200 * s),
        ]
        g = PolygonGeometry(v)
        # area: 200*60 + 100*(200-60) = 12000 + 14000 = 26000 mm^2 = 0.026 m^2
        assert g.area == pytest.approx(0.026, rel=1e-10)

    def test_invalid_polygon_raises(self):
        with pytest.raises(ValueError):
            PolygonGeometry([(0, 0), (1, 0)])  # only 2 vertices

    def test_self_intersecting_raises(self):
        with pytest.raises(ValueError):
            # bowtie
            PolygonGeometry([(0, 0), (1, 1), (1, 0), (0, 1)])

    def test_translate_preserves_area_and_I(self):
        g0 = PolygonGeometry.rectangle(width=0.2, height=0.4)
        g1 = g0.translate(dz=10.0, dy=20.0)
        assert g1.area == pytest.approx(g0.area, rel=1e-12)
        assert g1.I_zz == pytest.approx(g0.I_zz, rel=1e-10)
        assert g1.centroid == pytest.approx((10.0, 20.0), rel=1e-10)


# ============================================================ Section

class _DummyMaterial:
    """Minimal material stub: only carries E and density."""
    def __init__(self, E=200e9, nu=0.3, density=7850.0):
        self.E = E
        self.nu = nu
        self.density = density


class TestSection:
    def test_basic_construction(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        mat = _DummyMaterial(E=30e9, density=2400.0)
        sec = Section(
            geometry=g, zones=[MaterialZone(material=mat)],
            name="B1 300x600", family="rect",
        )
        assert sec.name == "B1 300x600"
        assert sec.family == "rect"
        assert sec.area == pytest.approx(0.18, rel=1e-12)
        assert sec.I_zz == pytest.approx(0.0054, rel=1e-12)

    def test_centroid_pass_through(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        sec = Section(geometry=g)
        cz, cy = sec.centroid
        assert cz == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_primary_material_first_zone(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        mat1 = _DummyMaterial(E=30e9)
        mat2 = _DummyMaterial(E=200e9)
        sec = Section(
            geometry=g,
            zones=[MaterialZone(material=mat1), MaterialZone(material=mat2)],
        )
        assert sec.primary_material is mat1

    def test_elastic_section_2d_adapter(self):
        """Section -> ElasticSection2D produces the right EA, EIz."""
        from femsolver.sections import ElasticSection2D

        g = PolygonGeometry.rectangle(0.3, 0.6)
        mat = _DummyMaterial(E=30e9)
        sec = Section(geometry=g, zones=[MaterialZone(material=mat)])
        es = sec.elastic_section_2d()
        assert isinstance(es, ElasticSection2D)
        assert es.EA == pytest.approx(30e9 * 0.18, rel=1e-12)
        assert es.EIz == pytest.approx(30e9 * 0.0054, rel=1e-12)

    def test_elastic_section_3d_adapter(self):
        from femsolver.sections import ElasticSection3D

        g = PolygonGeometry.rectangle(0.3, 0.6)
        mat = _DummyMaterial(E=30e9, nu=0.2)
        sec = Section(geometry=g, zones=[MaterialZone(material=mat)])
        # Provide an explicit J via geometry override-free path -- J defaults
        # to zero in Geometry base; the adapter clamps to >= 1e-30 so it
        # accepts. We only test that EA / EI / G are correct here.
        es = sec.elastic_section_3d()
        assert isinstance(es, ElasticSection3D)
        assert es.E == pytest.approx(30e9)
        assert es.G == pytest.approx(30e9 / (2 * 1.2))   # nu = 0.2

    def test_adapter_raises_without_material(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        sec = Section(geometry=g)
        with pytest.raises(ValueError):
            sec.elastic_section_2d()

    def test_weight_per_length(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        mat = _DummyMaterial(density=2400.0)  # concrete
        sec = Section(geometry=g, zones=[MaterialZone(material=mat)])
        # 0.18 * 2400 * 9.81 = 4237.92 N/m
        assert sec.weight_per_length() == pytest.approx(4237.92, rel=1e-3)

    def test_to_dict_round_trip(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        sec = Section(
            geometry=g, name="B1", family="rect", catalogue_ref=None,
        )
        d = sec.to_dict()
        assert d["name"] == "B1"
        assert d["family"] == "rect"
        assert d["area"] == pytest.approx(0.18, rel=1e-12)


# ============================================================ SectionLibrary

class TestSectionLibrary:
    def test_register_and_get(self):
        lib = SectionLibrary()
        g = PolygonGeometry.rectangle(0.3, 0.6)
        sec = Section(geometry=g, name="B1 300x600", family="rect")
        lib.register(sec)
        assert "B1 300x600" in lib
        assert lib["B1 300x600"] is sec
        assert lib.get("B1 300x600") is sec

    def test_missing_raises(self):
        lib = SectionLibrary()
        with pytest.raises(KeyError):
            lib.get("nonexistent")

    def test_duplicate_raises_unless_overwrite(self):
        lib = SectionLibrary()
        g = PolygonGeometry.rectangle(0.3, 0.6)
        sec1 = Section(geometry=g, name="B1", family="rect")
        sec2 = Section(geometry=g, name="B1", family="rect")
        lib.register(sec1)
        with pytest.raises(ValueError):
            lib.register(sec2)
        lib.register(sec2, overwrite=True)
        assert lib.get("B1") is sec2

    def test_empty_name_rejected(self):
        lib = SectionLibrary()
        sec = Section(geometry=PolygonGeometry.rectangle(0.3, 0.6))
        with pytest.raises(ValueError):
            lib.register(sec)

    def test_list_family(self):
        lib = SectionLibrary()
        for h in (0.3, 0.4, 0.5):
            g = PolygonGeometry.rectangle(0.2, h)
            lib.register(Section(geometry=g, name=f"rect_{h}", family="rect"))
        # Add one I-section family
        g = PolygonGeometry.rectangle(0.2, 0.4)
        lib.register(Section(geometry=g, name="W14x90", family="I",
                              catalogue_ref="W14x90"))
        assert sorted(lib.list_family("rect")) == [
            "rect_0.3", "rect_0.4", "rect_0.5",
        ]
        assert lib.list_family("I") == ["W14x90"]
        assert set(lib.families()) == {"rect", "I"}

    def test_global_instance_singleton(self):
        SectionLibrary.reset_global()
        lib1 = SectionLibrary.global_instance()
        lib2 = SectionLibrary.global_instance()
        assert lib1 is lib2

    def test_len(self):
        lib = SectionLibrary()
        assert len(lib) == 0
        g = PolygonGeometry.rectangle(0.3, 0.6)
        lib.register(Section(geometry=g, name="A", family="rect"))
        lib.register(Section(geometry=g, name="B", family="rect"))
        assert len(lib) == 2
