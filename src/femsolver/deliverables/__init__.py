"""Production deliverables: calc sheets, DXF drawings, BOM, QA.

Modules
-------
* :mod:`calc_sheet`     -- structured engineering calc sheet with HTML
  + optional PDF render (via reportlab).
* :mod:`dxf`            -- native minimal DXF writer (no third-party
  dependency).
* :mod:`bom`            -- bill-of-materials / quantity takeoff.
* :mod:`qa`             -- model QA / sanity warnings.
"""
from femsolver.deliverables.calc_sheet import (
    CalcCheck,
    CalcInput,
    CalcOutput,
    CalcSheet,
    CalcSection,
    render_calc_sheet_html,
    render_calc_sheet_pdf,
)
from femsolver.deliverables.dxf import (
    DxfDocument,
    write_model_plan_dxf,
)
from femsolver.deliverables.bom import (
    BomLine,
    BomReport,
    bom_concrete_frame,
    bom_steel_frame,
    bom_rebar,
)
from femsolver.deliverables.qa import (
    QaWarning,
    QaReport,
    run_qa_checks,
)

__all__ = [
    "CalcCheck", "CalcInput", "CalcOutput",
    "CalcSheet", "CalcSection",
    "render_calc_sheet_html", "render_calc_sheet_pdf",
    "DxfDocument", "write_model_plan_dxf",
    "BomLine", "BomReport",
    "bom_concrete_frame", "bom_steel_frame", "bom_rebar",
    "QaWarning", "QaReport", "run_qa_checks",
]
