"""Tests for the standardized Section-Designer verification suite."""
from __future__ import annotations

import math

import pytest

from femsolver.benchmarks.section_designer import (
    ALL_SECTION_BUILDERS,
    VerificationItem,
    all_verification_items,
    compare_items,
    export_csv,
    format_console,
    items_for_case,
    load_gsd_csv,
    to_markdown,
)


# ============================================================ section builders

class TestSectionBuilders:
    def test_all_build(self):
        assert len(ALL_SECTION_BUILDERS) == 5
        for build in ALL_SECTION_BUILDERS:
            case = build()
            assert case.section is not None
            assert case.f_c_prime > 0
            assert case.f_y > 0

    def test_rebar_or_prestress_present(self):
        for build in ALL_SECTION_BUILDERS:
            case = build()
            has_rebar = (case.section.reinforcement is not None
                         and case.section.reinforcement.bars)
            has_pt = (case.section.prestress is not None
                      and case.section.prestress.tendons)
            assert has_rebar or has_pt

    def test_sections_recentred(self):
        """Every section's centroid is at the origin (GSD convention)."""
        for build in ALL_SECTION_BUILDERS:
            case = build()
            cz, cy = case.section.geometry.centroid
            assert cz == pytest.approx(0.0, abs=1e-6)
            assert cy == pytest.approx(0.0, abs=1e-6)


# ============================================================ items

class TestItems:
    def test_all_items_finite(self):
        items = all_verification_items()
        assert len(items) > 40
        for it in items:
            assert math.isfinite(it.computed), f"{it.section_id} {it.quantity}"

    def test_every_section_has_props_and_analysis(self):
        for build in ALL_SECTION_BUILDERS:
            items = items_for_case(build())
            quantities = {it.quantity for it in items}
            assert "Gross area A_g" in quantities
            # each section has at least one code-based analysis quantity
            codes = {it.code for it in items}
            assert codes - {"-"}


# ============================================================ hand-calc anchors

class TestHandCalcAnchors:
    def test_S1_area_and_squash(self):
        items = items_for_case(ALL_SECTION_BUILDERS[0]())   # S1 rect column
        by = {(it.code, it.quantity): it for it in items}
        # A_g = 400 x 600 = 240,000 mm^2
        assert by[("-", "Gross area A_g")].computed == pytest.approx(
            240000.0, rel=1e-6)
        # AASHTO P_o = 0.85*30*(240000-3927) + 500*3927  (N) -> kN
        Ast = 8 * math.pi / 4 * 25.0 ** 2      # mm^2
        P_o = (0.85 * 30 * (240000 - Ast) + 500 * Ast) / 1e3  # kN
        got = by[("AASHTO LRFD 2024", "P_o squash (nominal)")].computed
        assert got == pytest.approx(P_o, rel=2e-3)

    def test_S3_cracking_moment(self):
        # S3 beam: M_cr = f_r * I_g / y_t, f_r = 0.62 sqrt(30) MPa
        items = items_for_case(ALL_SECTION_BUILDERS[2]())   # S3 beam
        by = {it.quantity: it for it in items}
        f_r = 0.62 * math.sqrt(30.0)          # MPa
        I_g = 300.0 * 600.0 ** 3 / 12.0        # mm^4
        M_cr = f_r * I_g / 300.0 / 1e6         # N.mm -> kN.m
        assert by["Cracking moment M_cr"].computed == pytest.approx(
            M_cr, rel=0.02)


# ============================================================ compare + IO

class TestCompareAndIO:
    def _one(self, gsd=None):
        return VerificationItem(
            "S9", "test", "-", "q", "kN", computed=100.0,
            tol_pct=2.0, gsd=gsd)

    def test_compare_pass(self):
        it = self._one(gsd=101.0)          # 1% off, tol 2% -> pass
        compare_items([it])
        assert it.passed is True
        assert it.diff_pct == pytest.approx(-0.9901, abs=1e-3)

    def test_compare_fail(self):
        it = self._one(gsd=110.0)          # 9% off -> fail
        compare_items([it])
        assert it.passed is False

    def test_compare_pending(self):
        it = self._one(gsd=None)
        compare_items([it])
        assert it.passed is None
        assert it.diff_pct is None

    def test_csv_roundtrip(self, tmp_path):
        items = all_verification_items()
        path = tmp_path / "v.csv"
        export_csv(items, str(path))
        assert path.exists()
        # Fill one value equal to computed, one badly off, reload
        text = path.read_text().splitlines()
        header = text[0].split(",")
        gi = header.index("midas_gsd")
        # locate the first data row for S1 area and set gsd = computed
        target = items[0]
        rows = text[1:]
        new_rows = [text[0]]
        for r in rows:
            cells = r.split(",")
            if (cells[0] == target.section_id
                    and cells[3] == target.quantity):
                cells[gi] = f"{target.computed:.6g}"
            new_rows.append(",".join(cells))
        path.write_text("\n".join(new_rows))

        items2 = all_verification_items()
        load_gsd_csv(items2, str(path))
        matched = [it for it in items2
                   if it.section_id == target.section_id
                   and it.quantity == target.quantity][0]
        assert matched.gsd is not None
        assert matched.passed is True

    def test_reporters_render(self):
        items = all_verification_items()
        md = to_markdown(items)
        con = format_console(items)
        assert "Midas GSD" in md
        assert "femsolver" in con
