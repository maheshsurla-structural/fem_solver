"""Phase 55 tests -- Theme Y deliverables (calc sheet, DXF, BOM, QA)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from femsolver.deliverables import (
    BomLine, BomReport,
    CalcCheck, CalcInput, CalcOutput, CalcSection, CalcSheet,
    DxfDocument, QaReport, QaWarning,
    bom_concrete_frame, bom_rebar, bom_steel_frame,
    render_calc_sheet_html, render_calc_sheet_pdf,
    run_qa_checks, write_model_plan_dxf,
)


# ============================================================ calc-sheet

class TestCalcCheckLogic:
    def test_passes_when_dcr_below_one(self):
        c = CalcCheck(name="x", demand=80.0, capacity=100.0, units="kN")
        assert c.passes
        assert c.calculated_dcr == pytest.approx(0.8)

    def test_fails_when_dcr_above_one(self):
        c = CalcCheck(name="x", demand=120.0, capacity=100.0, units="kN")
        assert not c.passes
        assert c.calculated_dcr == pytest.approx(1.2)

    def test_explicit_dcr_overrides(self):
        c = CalcCheck(name="x", demand=80.0, capacity=100.0, dcr=0.95)
        assert c.calculated_dcr == pytest.approx(0.95)


class TestCalcSheetHTML:
    def _make_sheet(self):
        sh = CalcSheet(project="Test", member="B1", code="ACI",
                         designer="X")
        sec = CalcSection(title="Flexure")
        sec.inputs.append(CalcInput("f_c", 30e6, "Pa"))
        sec.outputs.append(CalcOutput("phi*Mn", 220e3, "N.m",
                                        formula="phi*A_s*f_y*(d-a/2)"))
        sec.checks.append(CalcCheck("M", 180e3, 220e3, units="N.m"))
        sh.sections.append(sec)
        return sh

    def test_html_contains_member_name(self):
        h = render_calc_sheet_html(self._make_sheet())
        assert "B1" in h
        assert "Flexure" in h
        assert "PASS" in h

    def test_html_shows_fail_when_dcr_high(self):
        sh = CalcSheet(project="T", member="M", code="C", designer="D")
        sec = CalcSection(title="x")
        sec.checks.append(CalcCheck("x", 2.0, 1.0))
        sh.sections.append(sec)
        h = render_calc_sheet_html(sh)
        assert "FAIL" in h

    def test_pdf_renders(self, tmp_path):
        pytest.importorskip("reportlab")
        sh = self._make_sheet()
        out = tmp_path / "calc.pdf"
        render_calc_sheet_pdf(sh, str(out))
        assert out.exists()
        assert out.stat().st_size > 1000   # non-trivial PDF


# ============================================================ DXF

class TestDxf:
    def test_empty_document_writes(self, tmp_path):
        doc = DxfDocument()
        out = tmp_path / "empty.dxf"
        doc.write(str(out))
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "SECTION" in text
        assert "EOF" in text

    def test_line_entity_written(self, tmp_path):
        doc = DxfDocument()
        doc.add_line((0, 0), (10, 0))
        out = tmp_path / "line.dxf"
        doc.write(str(out))
        text = out.read_text(encoding="utf-8")
        assert "LINE" in text

    def test_layer_table_writes(self, tmp_path):
        doc = DxfDocument()
        doc.add_layer("BEAMS", color=5)
        doc.add_layer("COLUMNS", color=3)
        out = tmp_path / "layers.dxf"
        doc.write(str(out))
        text = out.read_text(encoding="utf-8")
        assert "BEAMS" in text and "COLUMNS" in text

    def test_polyline_validates(self):
        with pytest.raises(ValueError, match="at least 2"):
            DxfDocument().add_polyline([(0, 0)])

    def test_color_range_validates(self):
        doc = DxfDocument()
        with pytest.raises(ValueError, match="color"):
            doc.add_layer("X", color=300)

    def test_model_plan_export(self, tmp_path):
        from femsolver import BeamColumn2D, ElasticIsotropic, Model
        mat = ElasticIsotropic(1, E=30e9, nu=0.2, rho=2400)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0, 0); m.add_node(2, 6, 0)
        m.add_element(BeamColumn2D(1, (1, 2), mat, area=0.01, Iz=1e-4))
        m.fix(1, [1, 1, 1])
        out = tmp_path / "model.dxf"
        write_model_plan_dxf(m, str(out))
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "LINE" in text and "CIRCLE" in text


# ============================================================ BOM

class TestBom:
    def test_rebar_standard_diameter(self):
        # #20 bar at 0.617 wait, table says 20 mm -> 2.466 kg/m
        rep = bom_rebar([(20, 1.0, 1)])
        assert rep.total_for("kg") == pytest.approx(2.466)

    def test_rebar_count_scales(self):
        rep = bom_rebar([(12, 2.0, 5)])
        expected = 0.888 * 2.0 * 5
        assert rep.total_for("kg") == pytest.approx(expected)

    def test_concrete_frame_volume(self):
        rep = bom_concrete_frame([("B1", 6.0, 0.09)])
        # volume = 6.0 * 0.09 = 0.54
        conc = [l for l in rep.lines if l.item == "concrete"]
        assert sum(l.quantity for l in conc) == pytest.approx(0.54)

    def test_steel_total_weight(self):
        # 4 m long, 1e-2 m^2 -> 4*1e-2*7850 = 314 kg
        rep = bom_steel_frame([("W14x53", 4.0, 1.0e-2)])
        assert rep.total_for("kg") == pytest.approx(314.0)

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            bom_rebar([(20, -1, 1)])
        with pytest.raises(ValueError):
            bom_concrete_frame([("X", -1, 0.1)])

    def test_summary_aggregates_by_item(self):
        rep = bom_steel_frame([("A", 4, 1e-2), ("B", 6, 1e-2)])
        s = rep.summary()
        assert "steel" in s
        # Total = (4 + 6) * 1e-2 * 7850 = 785 kg
        assert s["steel"][0] == pytest.approx(785.0)


# ============================================================ QA

class TestQaChecks:
    def _make_model(self, with_orphan=False, with_fixity=True,
                     duplicate_elem=False):
        from femsolver import BeamColumn2D, ElasticIsotropic, Model
        mat = ElasticIsotropic(1, E=30e9, nu=0.2, rho=2400)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        m.add_node(1, 0, 0); m.add_node(2, 6, 0)
        if with_orphan:
            m.add_node(99, 100, 100)
        m.add_element(BeamColumn2D(1, (1, 2), mat, area=0.01, Iz=1e-4))
        if duplicate_elem:
            m.add_element(BeamColumn2D(2, (1, 2), mat, area=0.01, Iz=1e-4))
        if with_fixity:
            m.fix(1, [1, 1, 1])
        return m

    def test_clean_model_no_errors(self):
        m = self._make_model()
        rep = run_qa_checks(m)
        assert not rep.has_errors

    def test_no_fixity_is_error(self):
        m = self._make_model(with_fixity=False)
        rep = run_qa_checks(m)
        assert rep.has_errors
        assert any("fixities" in e.message for e in rep.errors)

    def test_orphan_node_warning(self):
        m = self._make_model(with_orphan=True)
        rep = run_qa_checks(m)
        warns = [w for w in rep.warnings if "orphan" in w.message.lower()]
        assert len(warns) == 1
        assert 99 in warns[0].affected

    def test_duplicate_element_warning(self):
        m = self._make_model(duplicate_elem=True)
        rep = run_qa_checks(m)
        dups = [w for w in rep.warnings
                if "duplicate" in w.message.lower()]
        assert len(dups) == 1

    def test_inventory_info(self):
        m = self._make_model()
        rep = run_qa_checks(m)
        infos = [w for w in rep.warnings if w.category == "INFO"]
        assert any("2 nodes" in i.message for i in infos)

    def test_str_renders_categories(self):
        m = self._make_model(with_orphan=True, with_fixity=False)
        rep = run_qa_checks(m)
        s = str(rep)
        assert "[ERROR]" in s
        assert "[WARNING]" in s
        assert "[INFO]" in s
