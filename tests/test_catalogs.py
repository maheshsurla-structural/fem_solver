"""Phase 56 tests -- Theme Z engineering catalogues."""
from __future__ import annotations

import pytest

from femsolver.catalogs import (
    auto_select_ec_section,
    bolt_lookup,
    concrete_grade,
    eurocode_section,
    indian_section,
    list_bolt_grades,
    list_eurocode_sections,
    list_indian_sections,
    rebar_grade,
    steel_grade,
)


# ============================================================ EC sections

class TestEurocodeSections:
    def test_ipe_300_handbook_values(self):
        s = eurocode_section("IPE 300")
        assert s.mass == pytest.approx(42.2)
        assert s.h == 300
        assert s.b == 150
        assert s.W_pl_y == pytest.approx(628e3, rel=1e-3)
        assert s.I_y == pytest.approx(8360e4, rel=1e-3)

    def test_hea_200_handbook_values(self):
        s = eurocode_section("HEA 200")
        assert s.mass == pytest.approx(42.3)
        assert s.h == 190
        assert s.b == 200

    def test_case_and_space_tolerant(self):
        s1 = eurocode_section("IPE 300")
        s2 = eurocode_section("ipe300")
        s3 = eurocode_section("IPE300")
        assert s1.name == s2.name == s3.name == "IPE 300"

    def test_unknown_section_raises(self):
        with pytest.raises(KeyError, match="unknown"):
            eurocode_section("IPE 999")

    def test_list_families(self):
        assert "IPE 300" in list_eurocode_sections("IPE")
        assert "HEA 200" in list_eurocode_sections("HEA")
        assert "HEB 300" in list_eurocode_sections("HEB")
        with pytest.raises(ValueError):
            list_eurocode_sections("UB")

    def test_auto_select_lightest(self):
        # Demand 500 cm^3 -> IPE 270 (484) too small, IPE 300 (628) OK
        sel = auto_select_ec_section(W_pl_required=500e3, family="IPE")
        assert sel.name == "IPE 300"

    def test_auto_select_no_solution(self):
        with pytest.raises(ValueError, match="no IPE section"):
            auto_select_ec_section(W_pl_required=1e10, family="IPE")

    def test_auto_select_minimise_depth(self):
        sel = auto_select_ec_section(
            W_pl_required=300e3, family="IPE", minimise="depth",
        )
        # IPE 240 (367 cm^3) qualifies; shallow alternative would be HEA-like
        # but family is fixed -> shortest IPE that qualifies = IPE 240
        assert sel.h == 240


# ============================================================ IS sections

class TestIndianSections:
    def test_ismb_300_handbook_values(self):
        s = indian_section("ISMB 300")
        assert s.mass == pytest.approx(44.2)
        assert s.h == 300
        # Plastic modulus approximated 1.14x elastic; ISMB 300 W_el ~ 573.6 cm^3
        # -> W_pl ~ 653.9 cm^3
        assert s.W_pl_y == pytest.approx(1.14 * 573.6e3, rel=1e-3)

    def test_ismc_200_data(self):
        s = indian_section("ISMC 200")
        assert s.family == "ISMC"
        assert s.h == 200

    def test_isa_75_data(self):
        s = indian_section("ISA 75x75x8")
        assert s.family == "ISA"
        assert s.mass == pytest.approx(8.86)

    def test_list_families(self):
        assert "ISMB 300" in list_indian_sections("ISMB")
        assert "ISMC 200" in list_indian_sections("ISMC")
        assert "ISA 100x100x8" in list_indian_sections("ISA")


# ============================================================ bolts

class TestBolts:
    def test_m20_8_8(self):
        b = bolt_lookup("8.8", 20)
        assert b.grade == "8.8"
        assert b.d_mm == 20
        assert b.A_t == pytest.approx(245.0)
        assert b.f_ub == pytest.approx(800e6)

    def test_a325_3_4_inch(self):
        # ~22 mm is closest standard ISO value to 3/4" (19.05 mm)
        b = bolt_lookup("A325", 22)
        assert b.grade == "A325"
        assert b.f_ub == pytest.approx(830e6)

    def test_unknown_grade(self):
        with pytest.raises(ValueError, match="unknown bolt grade"):
            bolt_lookup("12.5", 20)

    def test_unknown_diameter(self):
        with pytest.raises(ValueError, match="not in standard"):
            bolt_lookup("8.8", 15)

    def test_list_grades(self):
        g = list_bolt_grades()
        assert "8.8" in g and "A325" in g

    def test_higher_grade_higher_strength(self):
        b1 = bolt_lookup("8.8", 20)
        b2 = bolt_lookup("10.9", 20)
        assert b2.f_ub > b1.f_ub


# ============================================================ materials

class TestMaterials:
    def test_c30_ec2_E_cm(self):
        c = concrete_grade("C30")
        # EC2 E_cm = 22 (f_cm/10)^0.3 in GPa with f_cm = f_ck + 8
        # = 22 * (38/10)^0.3 GPa = 22 * 1.486 = 32.7 GPa
        assert c.E_cm == pytest.approx(32.7e9, rel=0.01)
        # f_ctm = 0.30 * 30^(2/3) = 0.30 * 9.66 = 2.90 MPa
        assert c.f_ctm == pytest.approx(2.90e6, rel=0.01)

    def test_m30_matches_C30_fck(self):
        # Different naming but same f_ck (Indian M-grade is cube
        # strength; this library uses it directly as f_ck)
        c1 = concrete_grade("C30")
        c2 = concrete_grade("M30")
        assert c1.f_ck == c2.f_ck

    def test_aci_psi_lookup(self):
        c = concrete_grade("4000 psi")
        # 4000 psi = 27.6 MPa
        assert c.f_ck == pytest.approx(27.6e6, rel=0.01)

    def test_s355_handbook(self):
        s = steel_grade("S355")
        assert s.f_y == 355e6
        assert s.f_u == 510e6
        assert s.E == 200e9

    def test_fe410_indian(self):
        s = steel_grade("Fe410")
        assert s.f_y == 250e6
        assert s.f_u == 410e6

    def test_rebar_b500(self):
        r = rebar_grade("B500")
        assert r.f_yk == 500e6
        assert r.E_s == 200e9

    def test_rebar_grade60(self):
        r = rebar_grade("Grade 60")
        # ASTM A615 Grade 60: f_yk = 60 ksi = 414 MPa
        assert r.f_yk == pytest.approx(414e6, rel=0.01)

    def test_unknown_grade_raises(self):
        with pytest.raises(KeyError):
            concrete_grade("C999")
        with pytest.raises(KeyError):
            steel_grade("S999")
        with pytest.raises(KeyError):
            rebar_grade("Fe999")
