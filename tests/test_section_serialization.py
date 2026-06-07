"""Phase II.8 tests -- JSON round-trip + SVG + SectionReport."""
from __future__ import annotations

import json

import pytest

from femsolver.design.concrete.section import ConcreteMaterial
from femsolver.sections import (
    ReinforcementLayout,
    aisc_section,
    build_section_report,
    custom_polygon_section,
    eurocode_section,
    hollow_rect_section,
    i_section,
    indian_section,
    rc_rectangular_section,
    rectangular_section,
    section_from_dict,
    section_from_json,
    section_to_dict,
    section_to_json,
    section_to_svg,
)


# ============================================================ JSON: catalogued sections

class TestJsonAISC:
    def test_round_trip_preserves_gross_props(self):
        sec = aisc_section("W14x90")
        text = section_to_json(sec)
        sec2 = section_from_json(text)
        assert sec2.area == pytest.approx(sec.area, rel=1e-12)
        assert sec2.I_zz == pytest.approx(sec.I_zz, rel=1e-12)
        assert sec2.I_yy == pytest.approx(sec.I_yy, rel=1e-12)
        assert sec2.J == pytest.approx(sec.J, rel=1e-12)

    def test_round_trip_preserves_identity(self):
        sec = aisc_section("W14x90")
        sec2 = section_from_json(sec.to_json())
        assert sec2.name == "W14x90"
        assert sec2.family == "W"
        assert sec2.catalogue_ref == "W14x90"

    def test_dict_is_valid_json(self):
        sec = aisc_section("W14x90")
        d = section_to_dict(sec)
        assert isinstance(d, dict)
        # Must serialize via the stdlib json module
        json.dumps(d)

    def test_version_tag_present(self):
        sec = aisc_section("W14x90")
        d = section_to_dict(sec)
        assert d["femsolver_section_version"] == 1


class TestJsonEurocode:
    def test_IPE300_round_trip(self):
        sec = eurocode_section("IPE 300")
        sec2 = section_from_json(sec.to_json())
        assert sec2.area == pytest.approx(sec.area, rel=1e-12)
        assert sec2.family == "IPE"

    def test_indian_round_trip(self):
        sec = indian_section("ISMB 400")
        sec2 = section_from_json(sec.to_json())
        assert sec2.area == pytest.approx(sec.area, rel=1e-12)
        assert sec2.family == "ISMB"


# ============================================================ JSON: parametric

class TestJsonParametric:
    def test_rectangular(self):
        sec = rectangular_section(b=0.3, h=0.6, name="B1")
        sec2 = section_from_json(sec.to_json())
        assert sec2.area == pytest.approx(0.18, rel=1e-12)
        assert sec2.name == "B1"
        assert sec2.family == "rect"

    def test_hollow_rectangle_holes_preserved(self):
        sec = hollow_rect_section(b=0.2, h=0.1, t=0.006)
        sec2 = section_from_json(sec.to_json())
        # Hollow area: 0.2*0.1 - 0.188*0.088 = 0.0034...
        expected = 0.2 * 0.1 - 0.188 * 0.088
        assert sec2.area == pytest.approx(expected, rel=1e-10)

    def test_custom_polygon_round_trip(self):
        sec = custom_polygon_section(
            outline=[(0, 0), (0.2, 0), (0.2, 0.06),
                      (0.1, 0.06), (0.1, 0.2), (0, 0.2)],
            name="L-bracket",
        )
        sec2 = section_from_json(sec.to_json())
        assert sec2.area == pytest.approx(0.026, rel=1e-10)
        assert sec2.name == "L-bracket"


# ============================================================ JSON: RC sections

class TestJsonRC:
    def test_rebar_positions_preserved(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
            top_bars=[(285e-6, "#6")] * 2,
        )
        sec = rc_rectangular_section(b=0.3, h=0.6, reinforcement=rl)
        sec2 = section_from_json(sec.to_json())
        assert sec2.reinforcement is not None
        assert sec2.reinforcement.n_bars == 5
        # Bar positions preserved
        z_positions = sorted(b.z for b in sec2.reinforcement.bars if b.y < 0)
        assert z_positions[0] == pytest.approx(-0.11, rel=1e-3)

    def test_concrete_material_reconstructed_via_dataclass(self):
        """ConcreteMaterial is a dataclass with simple fields, so the
        decoder should reconstruct it from the JSON args."""
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        sec = rc_rectangular_section(b=0.3, h=0.6, concrete=cm)
        sec2 = section_from_json(sec.to_json())
        mat2 = sec2.primary_material
        assert mat2 is not None
        assert mat2.fc_prime == pytest.approx(30e6)
        assert mat2.fy == pytest.approx(420e6)

    def test_stirrup_params_preserved(self):
        rl = ReinforcementLayout(
            bars=[],
            stirrup_designation="#4",
            stirrup_spacing=0.10,
            stirrup_legs=4,
        )
        sec = rc_rectangular_section(b=0.3, h=0.6, reinforcement=rl)
        sec2 = section_from_json(sec.to_json())
        assert sec2.reinforcement.stirrup_designation == "#4"
        assert sec2.reinforcement.stirrup_spacing == pytest.approx(0.10)
        assert sec2.reinforcement.stirrup_legs == 4


class TestJsonUnknownMaterial:
    def test_unknown_material_returns_none_gracefully(self):
        """If the material class can't be imported on decode, the
        material slot should be None and the section should still
        load."""
        sec = rectangular_section(b=0.3, h=0.6)
        # Fabricate a JSON payload with a bogus class path
        d = section_to_dict(sec)
        d["zones"] = [{
            "name": "unknown",
            "material": {"class": "nonexistent.module.NoSuchClass"},
        }]
        sec2 = section_from_dict(d)
        assert sec2.zones[0].material is None
        assert sec2.area == pytest.approx(0.18, rel=1e-10)

    def test_version_mismatch_raises(self):
        with pytest.raises(ValueError, match="version"):
            section_from_dict({"femsolver_section_version": 99})


# ============================================================ SVG

class TestSvg:
    def test_returns_svg_string(self):
        sec = rectangular_section(b=0.3, h=0.6)
        svg = sec.to_svg()
        assert svg.startswith("<svg")
        assert svg.rstrip().endswith("</svg>")

    def test_contains_polygon_path(self):
        sec = rectangular_section(b=0.3, h=0.6)
        svg = sec.to_svg()
        assert "<path" in svg

    def test_with_rebar_contains_circles(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6, bottom_bars=[(510e-6, "#8")] * 3,
        )
        sec = rc_rectangular_section(b=0.3, h=0.6, reinforcement=rl)
        svg = sec.to_svg()
        assert "<circle" in svg
        # Three bars -> at least 3 circles
        assert svg.count("<circle") >= 3

    def test_no_rebar_for_steel_section(self):
        sec = aisc_section("W14x90")
        svg = sec.to_svg()
        assert "<circle" not in svg  # no rebar for a steel I

    def test_section_name_in_svg(self):
        sec = rectangular_section(b=0.3, h=0.6, name="MyBeam")
        svg = sec.to_svg()
        assert "MyBeam" in svg

    def test_svg_escapes_special_chars(self):
        sec = rectangular_section(b=0.3, h=0.6, name="A&B<test>")
        svg = sec.to_svg()
        # Special chars must be HTML-escaped
        assert "A&amp;B&lt;test&gt;" in svg

    def test_hollow_section_uses_evenodd_fill(self):
        sec = hollow_rect_section(b=0.2, h=0.1, t=0.006)
        svg = sec.to_svg()
        assert "evenodd" in svg

    def test_dimensions_annotated(self):
        sec = rectangular_section(b=0.3, h=0.6)
        svg = sec.to_svg(show_dimensions=True)
        assert "300 mm" in svg or "600 mm" in svg


# ============================================================ SectionReport

class TestSectionReport:
    def test_basic_fields(self):
        sec = aisc_section("W14x90")
        rep = build_section_report(sec)
        assert rep.name == "W14x90"
        assert rep.family == "W"
        assert rep.catalogue_ref == "W14x90"
        assert rep.area_mm2 == pytest.approx(sec.area * 1e6, rel=1e-10)

    def test_engineering_units(self):
        sec = rectangular_section(b=0.3, h=0.6)
        rep = build_section_report(sec)
        # 300 x 600 mm rectangle
        assert rep.depth_mm == pytest.approx(600, rel=1e-10)
        assert rep.width_mm == pytest.approx(300, rel=1e-10)
        # I_zz in mm^4: b·h³/12 = 0.3·0.6³/12 m^4 = 5.4e-3 m^4 = 5.4e9 mm^4
        assert rep.I_zz_mm4 == pytest.approx(5.4e9, rel=1e-9)

    def test_rebar_rows_populated(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
            top_bars=[(285e-6, "#6")] * 2,
        )
        sec = rc_rectangular_section(b=0.3, h=0.6, reinforcement=rl)
        rep = build_section_report(sec)
        assert len(rep.rebar_rows) == 5
        # Total A_s in mm^2
        assert rep.total_rebar_area_mm2 == pytest.approx(
            (3 * 510 + 2 * 285), rel=1e-10
        )

    def test_html_contains_section_name(self):
        sec = rectangular_section(b=0.3, h=0.6, name="B1")
        rep = build_section_report(sec)
        html = rep.to_html()
        assert "<h3>B1" in html
        # Properties table appears
        assert "section-props" in html
        # SVG embedded
        assert "<svg" in html

    def test_html_includes_rebar_table_when_present(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
        )
        sec = rc_rectangular_section(b=0.3, h=0.6, reinforcement=rl)
        html = sec.section_report().to_html()
        assert "section-rebar" in html
        assert "#8" in html

    def test_html_skips_rebar_table_when_absent(self):
        sec = aisc_section("W14x90")
        html = sec.section_report().to_html()
        assert "section-rebar" not in html

    def test_stirrup_info_present(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
            stirrup_designation="#4",
            stirrup_spacing=0.10,
        )
        sec = rc_rectangular_section(b=0.3, h=0.6, reinforcement=rl)
        rep = build_section_report(sec)
        assert "#4" in rep.stirrup_info
        assert "100" in rep.stirrup_info or "0.1" in rep.stirrup_info
