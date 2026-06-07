"""Structured engineering calc sheet -- HTML + optional PDF.

A *calc sheet* is the deliverable an engineer signs and stamps. It
documents:

* **Header** -- project, member, code clause, designer.
* **Inputs** -- the geometric, material, and load values used.
* **Formulas** -- the design equations applied (as math text).
* **Outputs** -- computed quantities with units.
* **Checks** -- pass / fail / N.A. stamps with margins (DCR).

Rendering paths
---------------
* :func:`render_calc_sheet_html` -- self-contained HTML with inline
  CSS, no dependencies beyond Python stdlib.
* :func:`render_calc_sheet_pdf`  -- requires ``reportlab`` (lazy
  import); raises a clear ``ImportError`` if not installed.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence


# ============================================================ data

@dataclass
class CalcInput:
    """One input row in the calc sheet."""
    symbol: str         # e.g., "f'_c", "L"
    value: float
    units: str = ""
    description: str = ""


@dataclass
class CalcOutput:
    """One output row -- typically a code-formula result."""
    symbol: str
    value: float
    units: str = ""
    formula: str = ""           # human-readable formula
    description: str = ""


@dataclass
class CalcCheck:
    """One pass/fail check with a demand-to-capacity ratio."""
    name: str
    demand: float
    capacity: float
    dcr: float | None = None        # explicit override if not demand/capacity
    units: str = ""
    code_clause: str = ""
    note: str = ""

    @property
    def calculated_dcr(self) -> float:
        if self.dcr is not None:
            return float(self.dcr)
        if self.capacity == 0:
            return float("inf")
        return float(self.demand / self.capacity)

    @property
    def passes(self) -> bool:
        return self.calculated_dcr <= 1.0


@dataclass
class CalcSection:
    """Named section of a calc sheet (inputs / formulas / outputs / checks)."""
    title: str
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    narrative: str = ""             # free text block


@dataclass
class CalcSheet:
    """Full engineering calc sheet."""
    project: str
    member: str
    code: str = ""                   # e.g., "ACI 318-19"
    designer: str = ""
    date: str = ""
    sections: list = field(default_factory=list)

    def __post_init__(self):
        if not self.date:
            self.date = datetime.now(datetime.now().astimezone().tzinfo).strftime("%Y-%m-%d")


# ============================================================ HTML

_HTML_CSS = """
<style>
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #222;
    margin: 30px;
    line-height: 1.45;
}
h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 16px; color: #336; margin-top: 24px; }
h3 { font-size: 14px; color: #336; margin-bottom: 4px; }
table { border-collapse: collapse; margin: 8px 0; }
table.kv td { padding: 3px 12px; }
table.kv td.sym { font-style: italic; font-family: 'Cambria Math', 'Times New Roman', serif; }
table.checks { border: 1px solid #888; width: 100%; }
table.checks th, table.checks td { padding: 4px 8px; border: 1px solid #aaa; }
.pass { background: #d8f3d8; }
.fail { background: #f7c8c8; }
.stamp { font-weight: bold; }
.header-box { background: #f0f0f6; padding: 12px; border-left: 4px solid #336; }
.formula { font-family: 'Cambria Math', 'Times New Roman', serif; color: #114; padding-left: 12px; }
.code-clause { color: #666; font-size: 11px; }
</style>
""".strip()


def _fmt_num(v: float, units: str = "") -> str:
    if abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-3):
        body = f"{v:.4e}"
    else:
        body = f"{v:.4g}"
    return f"{body} {units}".rstrip()


def _render_section_html(s: CalcSection) -> str:
    parts = [f"<h2>{html.escape(s.title)}</h2>"]
    if s.narrative:
        parts.append(f"<p>{html.escape(s.narrative)}</p>")
    if s.inputs:
        parts.append("<h3>Inputs</h3><table class='kv'>")
        for it in s.inputs:
            parts.append(
                f"<tr><td class='sym'>{html.escape(it.symbol)}</td>"
                f"<td>=</td><td>{_fmt_num(it.value, it.units)}</td>"
                f"<td>{html.escape(it.description)}</td></tr>"
            )
        parts.append("</table>")
    if s.outputs:
        parts.append("<h3>Calculations</h3><table class='kv'>")
        for o in s.outputs:
            row = (
                f"<tr><td class='sym'>{html.escape(o.symbol)}</td>"
                f"<td>=</td><td>{_fmt_num(o.value, o.units)}</td>"
                f"<td>{html.escape(o.description)}</td></tr>"
            )
            parts.append(row)
            if o.formula:
                parts.append(
                    f"<tr><td colspan='4' class='formula'>"
                    f"{html.escape(o.formula)}</td></tr>"
                )
        parts.append("</table>")
    if s.checks:
        parts.append("<h3>Checks</h3><table class='checks'><tr>"
                     "<th>Check</th><th>Demand</th><th>Capacity</th>"
                     "<th>DCR</th><th>Status</th><th>Clause</th></tr>")
        for c in s.checks:
            klass = "pass" if c.passes else "fail"
            stamp = "PASS" if c.passes else "FAIL"
            parts.append(
                f"<tr class='{klass}'>"
                f"<td>{html.escape(c.name)}</td>"
                f"<td>{_fmt_num(c.demand, c.units)}</td>"
                f"<td>{_fmt_num(c.capacity, c.units)}</td>"
                f"<td>{c.calculated_dcr:.3f}</td>"
                f"<td class='stamp'>{stamp}</td>"
                f"<td class='code-clause'>{html.escape(c.code_clause)}</td>"
                f"</tr>"
            )
        parts.append("</table>")
    return "\n".join(parts)


def render_calc_sheet_html(
    sheet: CalcSheet, *, title_extra: str = "",
) -> str:
    """Render a :class:`CalcSheet` as a self-contained HTML string."""
    title = f"{sheet.member} - {sheet.code}"
    if title_extra:
        title = f"{title} - {title_extra}"
    body_sections = "\n".join(
        _render_section_html(s) for s in sheet.sections
    )
    summary_stamp = ""
    all_checks = [c for s in sheet.sections for c in s.checks]
    if all_checks:
        if all(c.passes for c in all_checks):
            summary_stamp = "<p class='stamp pass'>ALL CHECKS PASS</p>"
        else:
            n_fail = sum(1 for c in all_checks if not c.passes)
            summary_stamp = (
                f"<p class='stamp fail'>{n_fail} CHECK(S) FAIL</p>"
            )
    return f"""<!DOCTYPE html>
<html>
<head><meta charset='utf-8'><title>{html.escape(title)}</title>
{_HTML_CSS}
</head>
<body>
<div class='header-box'>
<h1>{html.escape(sheet.project)} - {html.escape(sheet.member)}</h1>
<table class='kv'>
<tr><td>Code:</td><td>{html.escape(sheet.code)}</td></tr>
<tr><td>Designer:</td><td>{html.escape(sheet.designer)}</td></tr>
<tr><td>Date:</td><td>{html.escape(sheet.date)}</td></tr>
</table>
{summary_stamp}
</div>
{body_sections}
</body></html>"""


# ============================================================ PDF (reportlab)

def render_calc_sheet_pdf(sheet: CalcSheet, path: str) -> None:
    """Render to PDF via reportlab. Raises ``ImportError`` if reportlab
    is not installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "reportlab is required for render_calc_sheet_pdf. "
            "Install with: pip install reportlab"
        ) from exc

    styles = getSampleStyleSheet()
    title_st = styles["Title"]
    body_st = styles["BodyText"]
    h2_st = ParagraphStyle(
        "h2", parent=styles["Heading2"], textColor=colors.HexColor("#336"),
    )
    h3_st = ParagraphStyle(
        "h3", parent=styles["Heading3"], textColor=colors.HexColor("#336"),
    )
    formula_st = ParagraphStyle(
        "formula", parent=body_st, fontName="Times-Italic",
        leftIndent=20, textColor=colors.HexColor("#114"),
    )
    elements = []
    elements.append(Paragraph(
        f"{sheet.project} - {sheet.member}", title_st,
    ))
    elements.append(Paragraph(
        f"Code: {sheet.code} | Designer: {sheet.designer} | {sheet.date}",
        body_st,
    ))
    elements.append(Spacer(1, 0.4 * cm))

    all_checks = [c for s in sheet.sections for c in s.checks]
    if all_checks and all(c.passes for c in all_checks):
        elements.append(Paragraph(
            "<b>ALL CHECKS PASS</b>",
            ParagraphStyle("ok", parent=body_st,
                           backColor=colors.HexColor("#d8f3d8")),
        ))
    elif all_checks:
        n_fail = sum(1 for c in all_checks if not c.passes)
        elements.append(Paragraph(
            f"<b>{n_fail} CHECK(S) FAIL</b>",
            ParagraphStyle("ng", parent=body_st,
                           backColor=colors.HexColor("#f7c8c8")),
        ))
    elements.append(Spacer(1, 0.4 * cm))

    for s in sheet.sections:
        elements.append(Paragraph(s.title, h2_st))
        if s.narrative:
            elements.append(Paragraph(s.narrative, body_st))
        if s.inputs:
            elements.append(Paragraph("Inputs", h3_st))
            rows = [["Symbol", "Value", "Description"]]
            for it in s.inputs:
                rows.append([it.symbol, _fmt_num(it.value, it.units),
                             it.description])
            t = Table(rows, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eee")),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 0.2 * cm))
        if s.outputs:
            elements.append(Paragraph("Calculations", h3_st))
            for o in s.outputs:
                if o.formula:
                    elements.append(Paragraph(o.formula, formula_st))
                elements.append(Paragraph(
                    f"<b>{o.symbol}</b> = {_fmt_num(o.value, o.units)} "
                    f"<font color='#666'>({o.description})</font>",
                    body_st,
                ))
        if s.checks:
            elements.append(Paragraph("Checks", h3_st))
            rows = [["Check", "Demand", "Capacity", "DCR", "Status", "Clause"]]
            row_styles = []
            for i, c in enumerate(s.checks, 1):
                rows.append([c.name,
                             _fmt_num(c.demand, c.units),
                             _fmt_num(c.capacity, c.units),
                             f"{c.calculated_dcr:.3f}",
                             "PASS" if c.passes else "FAIL",
                             c.code_clause])
                bg = colors.HexColor("#d8f3d8") if c.passes \
                    else colors.HexColor("#f7c8c8")
                row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))
            t = Table(rows, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eee")),
            ] + row_styles))
            elements.append(t)
            elements.append(Spacer(1, 0.2 * cm))
        elements.append(Spacer(1, 0.3 * cm))

    doc = SimpleDocTemplate(
        path, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    doc.build(elements)
