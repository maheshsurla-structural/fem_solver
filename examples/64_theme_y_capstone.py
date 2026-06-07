"""Theme Y capstone -- full deliverable bundle for a small RC frame.

Runs a 2-bay, 1-storey RC frame through linear-static analysis and
emits the complete commercial-grade deliverable set:

1. **PDF + HTML calc sheet** for the most-stressed beam (formulas,
   inputs, outputs, pass/fail stamps).
2. **DXF plan view** with labelled nodes and elements.
3. **BOM** -- concrete volume, rebar tonnage, formwork area.
4. **QA report** showing the model passes basic sanity checks.

All artefacts land in ``./theme_y_deliverables/``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
)
from femsolver.results import (
    BomReport,
    CalcCheck,
    CalcInput,
    CalcOutput,
    CalcSection,
    CalcSheet,
    bom_concrete_frame,
    render_calc_sheet_html,
    render_calc_sheet_pdf,
    run_qa_checks,
    write_model_plan_dxf,
)


def main():
    print("=" * 78)
    print("Theme Y capstone -- complete deliverable bundle for an RC frame")
    print("=" * 78)
    os.makedirs("theme_y_deliverables", exist_ok=True)

    # ============================ build the frame ============================
    mat = ElasticIsotropic(1, E=30e9, nu=0.20, rho=2400.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    # 2-bay frame: columns at x=0, 6, 12 m; roof beams between
    H_storey = 4.0
    for i, x in enumerate([0.0, 6.0, 12.0]):
        m.add_node(i + 1, x, 0.0)
        m.add_node(i + 4, x, H_storey)
    # Columns: nodes 1->4, 2->5, 3->6 (400x400 mm)
    A_col = 0.4 * 0.4
    I_col = 0.4 ** 4 / 12.0
    m.add_element(BeamColumn2D(1, (1, 4), mat, area=A_col, Iz=I_col))
    m.add_element(BeamColumn2D(2, (2, 5), mat, area=A_col, Iz=I_col))
    m.add_element(BeamColumn2D(3, (3, 6), mat, area=A_col, Iz=I_col))
    # Beams: 4->5, 5->6 (300x500 mm)
    A_bm = 0.3 * 0.5
    I_bm = 0.3 * 0.5 ** 3 / 12.0
    m.add_element(BeamColumn2D(4, (4, 5), mat, area=A_bm, Iz=I_bm))
    m.add_element(BeamColumn2D(5, (5, 6), mat, area=A_bm, Iz=I_bm))
    # Supports
    for tag in (1, 2, 3):
        m.fix(tag, [1, 1, 1])
    # Distributed gravity: equivalent UDL of 30 kN/m on each roof beam
    # -> half goes to each end node
    for tag in (4, 5, 6):
        m.add_nodal_load(tag, [0.0, -30e3 * 3.0, 0.0])

    LinearStaticAnalysis(m).run()
    print(f"  Frame: 2-bay portal, columns 400x400, beams 300x500 mm")
    print(f"  Max midspan deflection: "
          f"{abs(m.node(5).disp[1])*1e3:.2f} mm")

    # ============================ QA ============================
    qa = run_qa_checks(m)
    print()
    print("  --- QA report ---")
    for line in str(qa).splitlines():
        print(f"  {line}")
    qa_path = "theme_y_deliverables/qa_report.txt"
    with open(qa_path, "w", encoding="utf-8") as fh:
        fh.write(str(qa))
    print(f"  QA written: {qa_path}")

    # ============================ BOM ============================
    members = [
        (f"Col-{tag}", H_storey, A_col) for tag in (1, 2, 3)
    ] + [
        (f"Beam-{tag}", 6.0, A_bm) for tag in (4, 5)
    ]
    bom = bom_concrete_frame(members, rebar_kg_per_m3=90.0)
    print()
    print("  --- BOM ---")
    print(f"  {'item':<10s} {'qty':>10s} {'unit':>5s}")
    for ll in bom.lines:
        print(f"  {ll.item:<10s} {ll.quantity:>10.2f} {ll.unit:>5s}")
    bom_path = "theme_y_deliverables/bom.txt"
    with open(bom_path, "w", encoding="utf-8") as fh:
        fh.write("item\tquantity\tunit\tdescription\n")
        for ll in bom.lines:
            fh.write(f"{ll.item}\t{ll.quantity:.3f}\t{ll.unit}\t"
                     f"{ll.description}\n")
    print(f"  BOM written: {bom_path}")

    # ============================ DXF ============================
    dxf_path = "theme_y_deliverables/frame.dxf"
    write_model_plan_dxf(m, dxf_path, text_height=0.15)
    print()
    print(f"  DXF written: {dxf_path} "
          f"({os.path.getsize(dxf_path)} bytes)")

    # ============================ Calc sheet for the critical beam ============
    # Beam between nodes 5 and 6 (the right-bay beam)
    L_beam = 6.0
    w_ud = 30e3
    M_max = w_ud * L_beam ** 2 / 8.0       # simply-supported equivalent
    # Capacity per ACI 318-19 22.2 (singly reinforced, simplified)
    # Use rho = 0.012, f_y = 420 MPa
    f_c = 30e6
    f_y = 420e6
    b = 0.300; d = 0.450
    rho = 0.012
    A_s = rho * b * d
    a = A_s * f_y / (0.85 * f_c * b)
    phi = 0.90
    M_n = A_s * f_y * (d - a / 2.0)
    phi_Mn = phi * M_n

    sheet = CalcSheet(
        project="Theme Y Demo Frame",
        member="Beam B-RB (interior, span 6.0 m)",
        code="ACI 318-19",
        designer="femsolver",
    )
    sec = CalcSection(
        title="Flexure check (positive moment)",
        narrative="Mid-span positive moment check for the interior "
                  "roof beam under factored gravity loading.",
    )
    sec.inputs += [
        CalcInput("f_c'", f_c, "Pa", "Concrete compressive strength"),
        CalcInput("f_y", f_y, "Pa", "Rebar yield strength"),
        CalcInput("b", b, "m", "Beam width"),
        CalcInput("d", d, "m", "Effective depth"),
        CalcInput("rho", rho, "", "Tension steel ratio"),
        CalcInput("w_u", w_ud, "N/m", "Factored line load"),
        CalcInput("L", L_beam, "m", "Span"),
    ]
    sec.outputs += [
        CalcOutput("A_s", A_s, "m^2",
                    formula="A_s = rho * b * d",
                    description="Tension steel area"),
        CalcOutput("a", a, "m",
                    formula="a = A_s * f_y / (0.85 * f_c' * b)",
                    description="Equivalent stress-block depth"),
        CalcOutput("phi*M_n", phi_Mn, "N.m",
                    formula="phi * A_s * f_y * (d - a/2)",
                    description="Design moment capacity"),
        CalcOutput("M_u", M_max, "N.m",
                    formula="w_u * L^2 / 8",
                    description="Factored midspan moment"),
    ]
    sec.checks.append(CalcCheck(
        name="Flexure (positive moment)",
        demand=M_max, capacity=phi_Mn, units="N.m",
        code_clause="ACI 318-19 22.2.2",
    ))
    sheet.sections.append(sec)

    html_path = "theme_y_deliverables/beam_calc.html"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_calc_sheet_html(sheet))
    print(f"  Calc HTML written: {html_path}")

    pdf_path = "theme_y_deliverables/beam_calc.pdf"
    try:
        render_calc_sheet_pdf(sheet, pdf_path)
        print(f"  Calc PDF written:  {pdf_path} "
              f"({os.path.getsize(pdf_path)} bytes)")
    except ImportError:
        print("  Calc PDF skipped (reportlab not installed)")

    print()
    print("Theme Y capstone DONE.")
    print(f"  All deliverables in: theme_y_deliverables/")


if __name__ == "__main__":
    sys.exit(main())
