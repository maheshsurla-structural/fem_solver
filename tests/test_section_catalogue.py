"""Phase II.4 tests -- unified AISC / Eurocode / Indian catalogues.

Verifies that:

* :func:`aisc_section`, :func:`eurocode_section`, :func:`indian_section`
  return :class:`Section` instances whose gross properties match the
  underlying catalogue dictionaries EXACTLY (to round-off).
* The unified-mapping convention is correct (catalogue strong axis ->
  unified ``I_zz``, weak axis -> ``I_yy``).
* The geometry polygon outline is built from the right parametric
  primitive per family (I, channel, angle).
* :meth:`SectionLibrary.aisc/eurocode/indian` populate every section
  from the underlying tables and cache the library.
* Elastic-section adapter on a catalogued section uses the
  catalogue-exact ``J`` (not the placeholder zero).
"""
from __future__ import annotations

import pytest

from femsolver.sections import (
    Section,
    SectionLibrary,
    aisc_section,
    eurocode_section,
    indian_section,
    load_aisc_library,
    load_eurocode_library,
    load_indian_library,
)


class _DummySteel:
    E = 200e9
    nu = 0.3
    density = 7850.0


# ============================================================ AISC

class TestAisc:
    def test_W14x90_matches_raw_table(self):
        from femsolver.design.steel.sections import get_section
        ss = get_section("W14x90")
        sec = aisc_section("W14x90")
        assert sec.area == pytest.approx(ss.A, rel=1e-12)
        # AISC convention: Ix is strong-axis (-> unified I_zz)
        assert sec.I_zz == pytest.approx(ss.Ix, rel=1e-12)
        assert sec.I_yy == pytest.approx(ss.Iy, rel=1e-12)
        assert sec.J == pytest.approx(ss.J, rel=1e-12)
        assert sec.geometry.Z_zz == pytest.approx(ss.Zx, rel=1e-12)
        assert sec.geometry.Z_yy == pytest.approx(ss.Zy, rel=1e-12)

    def test_W14x90_identity_fields(self):
        sec = aisc_section("W14x90")
        assert sec.name == "W14x90"
        assert sec.family == "W"
        assert sec.catalogue_ref == "W14x90"

    def test_W14x90_uses_AISC_J_not_thin_walled(self):
        """The catalogued J includes fillet contributions; it must be
        larger than the thin-walled-open approximation we'd get from a
        parametric primitive."""
        from femsolver.design.steel.sections import get_section
        from femsolver.sections.parametric import i_section
        ss = get_section("W14x90")
        parametric = i_section(h=ss.d, b=ss.bf, t_f=ss.tf, t_w=ss.tw)
        catalogued = aisc_section("W14x90")
        assert catalogued.J > parametric.J

    def test_with_material_drives_elastic_3d(self):
        sec = aisc_section("W14x90", material=_DummySteel())
        es = sec.elastic_section_3d()
        assert es.EA == pytest.approx(200e9 * sec.area, rel=1e-12)
        # GJ should be in the engineering range (W14x90 J ~ 1.69e-6)
        assert es.GJ > 1e3

    def test_unknown_designation_raises(self):
        with pytest.raises(KeyError):
            aisc_section("W99x999")

    def test_polygon_outline_present(self):
        sec = aisc_section("W14x90")
        poly = sec.geometry.polygon
        # Polygon should have non-zero area (matches the bounding box,
        # but our catalogued geometry overrides A separately)
        assert poly.area > 0


class TestAiscLibrary:
    def test_load_populates_all_W_shapes(self):
        from femsolver.design.steel.sections import _DATABASE
        lib = load_aisc_library(force_reload=True)
        assert len(lib) == len(_DATABASE)
        assert "W14x90" in lib
        assert "W36x150" in lib

    def test_library_is_cached(self):
        lib1 = load_aisc_library()
        lib2 = load_aisc_library()
        assert lib1 is lib2

    def test_section_library_aisc_classmethod(self):
        # SectionLibrary.aisc() is a thin alias around load_aisc_library
        lib = SectionLibrary.aisc()
        assert "W14x90" in lib

    def test_all_aisc_have_W_family(self):
        lib = load_aisc_library()
        for name in lib.list_all():
            sec = lib[name]
            assert sec.family == "W"
            assert sec.catalogue_ref == name


# ============================================================ Eurocode

class TestEurocode:
    def test_IPE300_matches_raw_table(self):
        from femsolver.data.sections_ec import EC_IPE
        sp = EC_IPE["IPE 300"]
        sec = eurocode_section("IPE 300")
        # EC table is in mm/mm^2/mm^4 etc -- convert to SI:
        assert sec.area == pytest.approx(sp.A * 1e-6, rel=1e-12)
        # EC convention: I_y is strong -> unified I_zz
        assert sec.I_zz == pytest.approx(sp.I_y * 1e-12, rel=1e-12)
        assert sec.I_yy == pytest.approx(sp.I_z * 1e-12, rel=1e-12)

    def test_HEA200_matches_raw_table(self):
        from femsolver.data.sections_ec import EC_HEA
        sp = EC_HEA["HEA 200"]
        sec = eurocode_section("HEA 200")
        assert sec.area == pytest.approx(sp.A * 1e-6, rel=1e-12)

    def test_IPE300_identity(self):
        sec = eurocode_section("IPE 300")
        assert sec.name == "IPE 300"
        assert sec.family == "IPE"
        assert sec.catalogue_ref == "IPE 300"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            eurocode_section("IPE 999")

    def test_load_library(self):
        from femsolver.data.sections_ec import EC_HEA, EC_HEB, EC_IPE
        lib = load_eurocode_library(force_reload=True)
        expected = len(EC_IPE) + len(EC_HEA) + len(EC_HEB)
        assert len(lib) == expected
        assert set(lib.families()) >= {"IPE", "HEA", "HEB"}


# ============================================================ Indian

class TestIndian:
    def test_ISMB400_matches_raw_table(self):
        from femsolver.data.sections_is import IS_ISMB
        sp = IS_ISMB["ISMB 400"]
        sec = indian_section("ISMB 400")
        assert sec.area == pytest.approx(sp.A * 1e-6, rel=1e-12)
        assert sec.I_zz == pytest.approx(sp.I_y * 1e-12, rel=1e-12)
        assert sec.family == "ISMB"

    def test_ISMC100_is_channel_family(self):
        sec = indian_section("ISMC 100")
        assert sec.family == "ISMC"
        from femsolver.sections.parametric import ChannelGeometry
        # The base geometry should be a Channel (catalogued wraps it)
        assert isinstance(sec.geometry._base, ChannelGeometry)

    def test_ISA50x50x6_is_angle_family(self):
        sec = indian_section("ISA 50x50x6")
        assert sec.family == "ISA"
        from femsolver.sections.parametric import AngleGeometry
        assert isinstance(sec.geometry._base, AngleGeometry)

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            indian_section("ISMB 9999")

    def test_load_library_has_all_three_families(self):
        from femsolver.data.sections_is import IS_ISA, IS_ISMB, IS_ISMC
        lib = load_indian_library(force_reload=True)
        expected = len(IS_ISMB) + len(IS_ISMC) + len(IS_ISA)
        assert len(lib) == expected
        assert set(lib.families()) == {"ISMB", "ISMC", "ISA"}


# ============================================================ Convention check

class TestStrongAxisConvention:
    """All three catalogues should map their strong-axis I onto unified
    I_zz consistently. This test pins the convention."""

    def test_aisc_strong_axis_is_I_zz(self):
        sec = aisc_section("W14x90")
        assert sec.I_zz > sec.I_yy  # strong > weak

    def test_eurocode_strong_axis_is_I_zz(self):
        sec = eurocode_section("IPE 300")
        assert sec.I_zz > sec.I_yy

    def test_indian_strong_axis_is_I_zz(self):
        sec = indian_section("ISMB 400")
        assert sec.I_zz > sec.I_yy


# ============================================================ Cross-cutting

class TestCrossCutting:
    def test_three_catalogues_independent_caches(self):
        a = load_aisc_library()
        e = load_eurocode_library()
        i = load_indian_library()
        assert a is not e
        assert e is not i
        # No name collisions across families
        a_names = set(a.list_all())
        e_names = set(e.list_all())
        i_names = set(i.list_all())
        assert not (a_names & e_names)
        assert not (e_names & i_names)

    def test_catalogue_section_drives_elastic(self):
        """Catalogued section with material should produce a working
        ElasticSection3D whose GJ is non-zero (the J override is
        what makes this work; without it the parametric geometry
        would give the placeholder 1e-30 floor)."""
        sec = aisc_section("W14x90", material=_DummySteel())
        es = sec.elastic_section_3d()
        assert es.GJ > 1e3   # engineering-range, not floor
