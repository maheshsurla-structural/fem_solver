"""Phase II.7 tests -- consumer migration to unified Section.

Verifies the migration helpers added in II.7:

* :func:`composite_girder_deck_section` -- bridge composite as
  unified Section
* :meth:`ConcreteSection.to_unified` -- legacy RC dataclass -> unified
  (inverse of :meth:`Section.as_aci_concrete_section`)
* :meth:`SteelSection.to_unified` -- legacy AISC dataclass -> unified
  (inverse of :meth:`Section.as_aisc_section`)

All legacy paths must continue to work unchanged -- these tests cover
only the new migration helpers and their round-trip integrity.
"""
from __future__ import annotations

import pytest

from femsolver.bridges.composite_section import (
    composite_girder_deck,
    composite_girder_deck_section,
)
from femsolver.design.concrete.section import (
    ConcreteMaterial,
    ConcreteSection,
    RebarLayout,
)
from femsolver.design.steel.sections import get_section
from femsolver.sections import Section


# ============================================================ bridge composite

class TestCompositeGirderDeckSection:
    def test_returns_section(self):
        sec = composite_girder_deck_section(
            girder_width=0.6, girder_height=1.5,
            deck_width=3.0, deck_thickness=0.25,
            girder_material=ConcreteMaterial(fc_prime=50e6, fy=420e6),
            deck_material=ConcreteMaterial(fc_prime=30e6, fy=420e6),
        )
        assert isinstance(sec, Section)
        assert sec.family == "composite_girder_deck"

    def test_two_material_zones(self):
        sec = composite_girder_deck_section(
            girder_width=0.6, girder_height=1.5,
            deck_width=3.0, deck_thickness=0.25,
            girder_material="GIRDER",
            deck_material="DECK",
        )
        assert len(sec.zones) == 2
        zone_names = {z.name for z in sec.zones}
        assert zone_names == {"girder", "deck"}

    def test_area_equals_two_rectangles(self):
        """Without modular-ratio transformation, gross area is the
        polygon union -- not the transformed-section A_t."""
        sec = composite_girder_deck_section(
            girder_width=0.6, girder_height=1.5,
            deck_width=3.0, deck_thickness=0.25,
            girder_material="G", deck_material="D",
        )
        # Two stacked rectangles touch but don't overlap: union area = sum
        expected = 0.6 * 1.5 + 3.0 * 0.25
        assert sec.area == pytest.approx(expected, rel=1e-10)

    def test_geometry_overall_height(self):
        sec = composite_girder_deck_section(
            girder_width=0.6, girder_height=1.5,
            deck_width=3.0, deck_thickness=0.25,
            girder_material="G", deck_material="D",
        )
        # Total height: 1.5 (girder) + 0.25 (deck) = 1.75
        assert sec.geometry.depth == pytest.approx(1.75, rel=1e-10)

    def test_legacy_function_still_works(self):
        """The legacy `composite_girder_deck` must continue to
        operate unchanged (backward compatibility)."""
        props = composite_girder_deck(
            girder_area=0.6 * 1.5, girder_I=0.6 * 1.5**3 / 12,
            girder_y_centroid=0.75, girder_height=1.5,
            deck_width=3.0, deck_thickness=0.25,
            E_girder=35e9, E_deck=30e9,
        )
        # Modular-ratio transformation: deck transformed to girder
        # concrete with n = 30/35 ~ 0.857
        assert 0.8 < props.n < 0.9
        assert props.A_t > 0


# ============================================================ ConcreteSection.to_unified

class TestConcreteSectionToUnified:
    def setup_method(self):
        self.cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        self.rl = RebarLayout(
            bottom_bars=("#8", "#8", "#8"),
            top_bars=("#6", "#6"),
            bottom_cover=0.040, top_cover=0.040,
            stirrup_designation="#3", stirrup_spacing=0.150,
        )
        self.cs = ConcreteSection(b=0.3, h=0.6, material=self.cm, rebar=self.rl)

    def test_returns_unified_section(self):
        uni = self.cs.to_unified()
        assert isinstance(uni, Section)
        assert uni.family == "rect"
        assert uni.area == pytest.approx(0.18, rel=1e-12)

    def test_reinforcement_carried_through(self):
        uni = self.cs.to_unified()
        assert uni.reinforcement is not None
        assert uni.reinforcement.n_bars == 5  # 3 bottom + 2 top

    def test_round_trip_to_legacy(self):
        """to_unified -> as_aci_concrete_section should give back the
        same b, h, bar counts, and covers."""
        uni = self.cs.to_unified()
        back = uni.as_aci_concrete_section()
        assert back.b == pytest.approx(self.cs.b, rel=1e-12)
        assert back.h == pytest.approx(self.cs.h, rel=1e-12)
        assert back.rebar.bottom_bars == self.cs.rebar.bottom_bars
        assert back.rebar.top_bars == self.cs.rebar.top_bars
        assert back.rebar.bottom_cover == pytest.approx(
            self.cs.rebar.bottom_cover, rel=1e-9,
        )
        assert back.rebar.top_cover == pytest.approx(
            self.cs.rebar.top_cover, rel=1e-9,
        )

    def test_concrete_material_preserved(self):
        uni = self.cs.to_unified()
        assert uni.primary_material is self.cm

    def test_stirrup_params_carried(self):
        uni = self.cs.to_unified()
        assert uni.reinforcement.stirrup_designation == "#3"
        assert uni.reinforcement.stirrup_spacing == pytest.approx(0.150)


# ============================================================ SteelSection.to_unified

class TestSteelSectionToUnified:
    def test_returns_unified_section(self):
        ss = get_section("W14x90")
        uni = ss.to_unified()
        assert isinstance(uni, Section)
        assert uni.catalogue_ref == "W14x90"
        assert uni.family == "W"

    def test_gross_properties_match(self):
        ss = get_section("W14x90")
        uni = ss.to_unified()
        assert uni.area == pytest.approx(ss.A, rel=1e-12)
        # AISC Ix -> unified I_zz (strong axis)
        assert uni.I_zz == pytest.approx(ss.Ix, rel=1e-12)
        assert uni.I_yy == pytest.approx(ss.Iy, rel=1e-12)
        assert uni.J == pytest.approx(ss.J, rel=1e-12)

    def test_round_trip_to_legacy(self):
        ss = get_section("W14x90")
        uni = ss.to_unified()
        back = uni.as_aisc_section()
        assert back.designation == ss.designation
        assert back.A == pytest.approx(ss.A, rel=1e-12)

    def test_with_material_drives_elastic_3d(self):
        class _Steel:
            E = 200e9
            nu = 0.3
            density = 7850.0
        ss = get_section("W14x90")
        uni = ss.to_unified(material=_Steel())
        es = uni.elastic_section_3d()
        assert es.EA == pytest.approx(200e9 * ss.A, rel=1e-12)


# ============================================================ Documentation check

class TestMigrationDocs:
    """Pin the deprecation notes added in II.7 so we know they don't
    silently disappear."""

    def test_aisc_doc_mentions_unified(self):
        from femsolver.design.steel import sections as aisc_mod
        assert "II.7 migration" in (aisc_mod.__doc__ or "")
        assert "SectionLibrary.aisc" in (aisc_mod.__doc__ or "")

    def test_eurocode_doc_mentions_unified(self):
        from femsolver.data import sections_ec
        assert "II.7 migration" in (sections_ec.__doc__ or "")
        assert "SectionLibrary.eurocode" in (sections_ec.__doc__ or "")

    def test_indian_doc_mentions_unified(self):
        from femsolver.data import sections_is
        assert "II.7 migration" in (sections_is.__doc__ or "")
        assert "SectionLibrary.indian" in (sections_is.__doc__ or "")
