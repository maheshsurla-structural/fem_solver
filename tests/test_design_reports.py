"""Phase 33 tests -- HTML + CSV design reports.
"""
from __future__ import annotations

import csv
import io
import re

import pytest

from femsolver.design import (
    MemberReportEntry,
    from_beam_design_result,
    from_column_design_result,
    from_steel_member_check,
    make_csv_summary,
    make_html_report,
    write_csv_summary,
    write_html_report,
)
from femsolver.design.concrete import (
    BeamDesignDemand,
    ColumnDesignDemand,
    ConcreteMaterial,
    RcMemberDesigner,
)
from femsolver.design.steel import (
    SteelMemberDemand,
    SteelMemberDesigner,
    astm_a992,
)


# ============================================================ fixtures

def _aci_beam_entry():
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    demand = BeamDesignDemand(M_u_positive=180e3, V_u=100e3)
    res = RcMemberDesigner.design_beam(
        b=0.30, h=0.55, material=mat, demand=demand, cover=0.050,
    )
    return from_beam_design_result("BeamX", res, demand=demand)


def _aci_column_entry():
    mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    demand = ColumnDesignDemand(P_u=600e3, M_u=80e3)
    res = RcMemberDesigner.design_column(
        b=0.40, h=0.40, material=mat, demand=demand, cover=0.060,
    )
    return from_column_design_result("ColX", res, demand=demand)


def _steel_entry():
    demand = SteelMemberDemand(
        P_u=300e3, M_ux=150e3, V_u=50e3,
    )
    res = SteelMemberDesigner.auto_size(
        material=astm_a992(), demand=demand,
        L=3.5, L_b=3.5, C_b=1.0,
    )
    return from_steel_member_check("SteelX", res.best, demand=demand)


# ============================================================ entry construction

def test_beam_entry_built_from_design_result():
    entry = _aci_beam_entry()
    assert isinstance(entry, MemberReportEntry)
    assert entry.member_tag == "BeamX"
    assert "RC beam" in entry.member_type
    assert "mm" in entry.section_summary
    assert "MPa" in entry.material_summary
    assert "ACI 318-19" in " ".join(entry.code_refs)


def test_column_entry_built_from_design_result():
    entry = _aci_column_entry()
    assert entry.member_tag == "ColX"
    assert "RC column" in entry.member_type
    assert "rho" in entry.section_summary
    # P-M interaction DCR present
    assert "PM interaction" in entry.dcrs


def test_steel_entry_built_from_member_check():
    entry = _steel_entry()
    assert entry.member_tag == "SteelX"
    assert entry.member_type.startswith("Steel W")
    assert "AISC 360-22" in " ".join(entry.code_refs)


def test_entry_pass_flag_consistent_with_DCR():
    entry = _aci_beam_entry()
    assert entry.passes == (entry.governing_dcr <= 1.0)


def test_failed_beam_design_yields_fail_entry():
    """If the design failed, the entry should be marked failed."""
    from femsolver.design.concrete import (
        BeamDesignResult,
    )
    # Construct a fake failed result
    failed = BeamDesignResult(
        section=None, flexure_positive=None, flexure_negative=None,
        shear=None, success=False,
        notes="could not satisfy demand",
    )
    entry = from_beam_design_result("BadBeam", failed)
    assert not entry.passes
    assert entry.governing_dcr == float("inf")
    assert "DESIGN FAILED" in entry.member_type


# ============================================================ HTML

def test_html_report_contains_member_tags():
    entries = [_aci_beam_entry(), _aci_column_entry(), _steel_entry()]
    html = make_html_report(entries)
    assert "BeamX" in html
    assert "ColX" in html
    assert "SteelX" in html


def test_html_report_summary_counts_pass_fail():
    entries = [_aci_beam_entry(), _aci_column_entry()]
    html = make_html_report(entries)
    # Summary line mentions counts
    n_total = len(entries)
    assert f"{n_total}</b>" in html


def test_html_report_styling_self_contained():
    """HTML output must include the <style> block so the file is
    self-contained without external CSS."""
    html = make_html_report([_aci_beam_entry()])
    assert "<style>" in html
    assert "</style>" in html


def test_html_report_has_status_marker_per_member():
    html = make_html_report([_aci_beam_entry()])
    assert "PASS" in html or "FAIL" in html


def test_html_report_includes_DCR_per_check():
    """DCR values should appear in HTML."""
    entry = _aci_beam_entry()
    html = make_html_report([entry])
    # Find a DCR value in the output (format: "X.XXX")
    assert re.search(r"\d+\.\d{3}", html) is not None


def test_html_report_includes_code_references():
    html = make_html_report([_aci_beam_entry()])
    assert "ACI 318-19" in html


def test_html_report_includes_governing_DCR_in_header():
    entry = _aci_beam_entry()
    html = make_html_report([entry])
    assert "governing DCR" in html


def test_write_html_to_file(tmp_path):
    path = tmp_path / "report.html"
    write_html_report(
        [_aci_beam_entry(), _aci_column_entry()],
        str(path),
        title="Test Report",
    )
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Test Report" in content
    assert "BeamX" in content


# ============================================================ CSV

def test_csv_summary_has_correct_columns():
    csv_str = make_csv_summary([_aci_beam_entry()])
    reader = csv.reader(io.StringIO(csv_str))
    header = next(reader)
    expected_cols = {
        "member_tag", "member_type", "section_summary",
        "material_summary", "governing_dcr", "governing_check",
        "passes", "notes",
    }
    assert set(header) >= expected_cols


def test_csv_summary_one_row_per_member():
    entries = [_aci_beam_entry(), _aci_column_entry(), _steel_entry()]
    csv_str = make_csv_summary(entries)
    reader = csv.reader(io.StringIO(csv_str))
    rows = list(reader)
    # 1 header + 3 data rows
    assert len(rows) == 1 + 3


def test_csv_summary_values_match_entries():
    entry = _aci_column_entry()
    csv_str = make_csv_summary([entry])
    rows = list(csv.reader(io.StringIO(csv_str)))
    header, data = rows[0], rows[1]
    row_dict = dict(zip(header, data))
    assert row_dict["member_tag"] == entry.member_tag
    assert row_dict["passes"] == ("PASS" if entry.passes else "FAIL")
    assert float(row_dict["governing_dcr"]) == pytest.approx(
        entry.governing_dcr, abs=1e-4,
    )


def test_write_csv_to_file(tmp_path):
    path = tmp_path / "summary.csv"
    write_csv_summary([_aci_beam_entry(), _aci_column_entry()], str(path))
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "member_tag" in content
    assert "BeamX" in content


# ============================================================ standalone entry

def test_entry_with_no_demands_or_capacities_still_renders():
    """A bare entry with just metadata still produces valid HTML."""
    entry = MemberReportEntry(
        member_tag="Bare",
        member_type="Test",
        section_summary="N/A",
        material_summary="",
        governing_dcr=0.5,
        passes=True,
    )
    html = make_html_report([entry])
    assert "Bare" in html


def test_entry_with_notes_renders_warning_block():
    entry = MemberReportEntry(
        member_tag="WithNotes",
        member_type="Test",
        section_summary="",
        material_summary="",
        governing_dcr=1.2,
        passes=False,
        notes="this is a critical warning",
    )
    html = make_html_report([entry])
    assert "critical warning" in html
