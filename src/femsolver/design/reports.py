"""Design reports -- HTML per-member + CSV summary.

The deliverable layer for Phase 29 (ACI 318 concrete) and Phase 30
(AISC 360 steel) designs. Consumes any of the result dataclasses from
those phases (``BeamDesignResult``, ``ColumnDesignResult``,
``SteelMemberCheck``), converts them into a code-neutral
:class:`MemberReportEntry`, and emits:

* a **self-contained HTML report** (no external CSS / JS) with one
  card per member containing demand, capacity, DCR, code clauses,
  and pass/fail status, suitable for review attachment;
* a flat **CSV summary** with one row per member for whole-frame
  spreadsheet review.

No third-party templating engine -- plain string concatenation,
following the no-extra-dependencies house style.
"""
from __future__ import annotations

import io
import csv
from dataclasses import dataclass, field

try:
    # Make adapters opt-in: the reports module itself only depends on
    # the dataclasses, but if the design modules are present we can
    # provide convenient from_* helpers.
    from femsolver.design.concrete import (
        BeamDesignResult,
        ColumnDesignDemand,
        ColumnDesignResult,
    )
    _CONCRETE_AVAILABLE = True
except ImportError:
    _CONCRETE_AVAILABLE = False

try:
    from femsolver.design.steel import (
        SteelMemberCheck,
        SteelMemberDemand,
    )
    _STEEL_AVAILABLE = True
except ImportError:
    _STEEL_AVAILABLE = False


# ============================================================ entry

@dataclass
class MemberReportEntry:
    """Code-neutral per-member design entry, ready for reporting.

    Attributes
    ----------
    member_tag : str
        Identifier for the member (e.g., ``"Beam L1-B1"``,
        ``"Col S2-C3"``).
    member_type : str
        Free text describing the member type (e.g.,
        ``"RC beam (ACI 318-19)"``, ``"Steel W14x90 (AISC 360-22)"``).
    section_summary : str
        Geometric / section summary (e.g., ``"300 x 550 mm, 4 #7 bot
        + 3 #7 top, #3 stirrups @ 150 mm"``).
    material_summary : str
        Material grades summary (e.g., ``"fc' = 28 MPa, fy = 420 MPa"``).
    demands : dict[str, str]
        Map of demand label -> formatted value string. Examples:
        ``{"M_u": "200 kN·m", "V_u": "120 kN", "P_u": "+400 kN"}``.
    capacities : dict[str, str]
        Same format for capacities (``φM_n``, ``φV_n``, ``φP_n``).
    dcrs : dict[str, float]
        Per-check DCR numerical values.
    governing_dcr : float
        Worst (largest) DCR across all checks.
    governing_check : str
        Label of which check produced ``governing_dcr``.
    passes : bool
        ``governing_dcr <= 1.0``.
    code_refs : list[str]
        Citation strings (e.g., ``"ACI 318-19 §22.2"``).
    notes : str
        Free-text design notes / warnings.
    """

    member_tag: str
    member_type: str
    section_summary: str
    material_summary: str
    demands: dict = field(default_factory=dict)
    capacities: dict = field(default_factory=dict)
    dcrs: dict = field(default_factory=dict)
    governing_dcr: float = 0.0
    governing_check: str = ""
    passes: bool = True
    code_refs: list = field(default_factory=list)
    notes: str = ""


# ============================================================ adapters

def from_beam_design_result(
    member_tag: str, result, *, demand=None,
) -> MemberReportEntry:
    """Convert a Phase 29 :class:`BeamDesignResult` into a report entry."""
    if not _CONCRETE_AVAILABLE:
        raise RuntimeError("femsolver.design.concrete is not importable")
    if not result.success or result.section is None:
        return MemberReportEntry(
            member_tag=member_tag,
            member_type="RC beam (ACI 318-19) -- DESIGN FAILED",
            section_summary="(no valid section)",
            material_summary="",
            governing_dcr=float("inf"),
            governing_check="design",
            passes=False,
            notes=result.notes,
        )
    s = result.section
    section_summary = (
        f"{s.b * 1000:.0f} x {s.h * 1000:.0f} mm; "
        f"bot {', '.join(s.rebar.bottom_bars) or '(none)'}; "
        f"top {', '.join(s.rebar.top_bars) or '(none)'}; "
        f"{s.rebar.stirrup_designation} stirrups @ "
        f"{s.rebar.stirrup_spacing * 1000:.0f} mm"
    )
    material_summary = (
        f"fc' = {s.material.fc_prime / 1e6:.0f} MPa, "
        f"fy = {s.material.fy / 1e6:.0f} MPa"
    )
    demands_d: dict[str, str] = {}
    capacities_d: dict[str, str] = {}
    dcrs_d: dict[str, float] = {}
    if demand is not None:
        if abs(demand.M_u_positive) > 0:
            demands_d["M_u+ (sag)"] = f"{demand.M_u_positive / 1e3:+.1f} kN.m"
            capacities_d["phi*M_n+"] = f"{result.flexure_positive.phi_M_n / 1e3:+.1f} kN.m"
            dcrs_d["M+"] = (abs(demand.M_u_positive) /
                              result.flexure_positive.phi_M_n
                              if result.flexure_positive.phi_M_n > 0 else 0.0)
        if abs(demand.M_u_negative) > 0:
            demands_d["M_u- (hog)"] = f"{demand.M_u_negative / 1e3:+.1f} kN.m"
            if result.flexure_negative is not None:
                capacities_d["phi*M_n-"] = (
                    f"{result.flexure_negative.phi_M_n / 1e3:+.1f} kN.m"
                )
                dcrs_d["M-"] = (abs(demand.M_u_negative) /
                                  result.flexure_negative.phi_M_n
                                  if result.flexure_negative.phi_M_n > 0
                                  else 0.0)
        if abs(demand.V_u) > 0:
            demands_d["V_u"] = f"{demand.V_u / 1e3:.1f} kN"
            capacities_d["phi*V_n"] = f"{result.shear.phi_V_n / 1e3:.1f} kN"
            dcrs_d["V"] = (abs(demand.V_u) / result.shear.phi_V_n
                              if result.shear.phi_V_n > 0 else 0.0)
    governing_dcr = max(dcrs_d.values()) if dcrs_d else 0.0
    governing_check = (max(dcrs_d, key=dcrs_d.get) if dcrs_d else "none")
    return MemberReportEntry(
        member_tag=member_tag,
        member_type="RC beam (ACI 318-19)",
        section_summary=section_summary,
        material_summary=material_summary,
        demands=demands_d,
        capacities=capacities_d,
        dcrs=dcrs_d,
        governing_dcr=governing_dcr,
        governing_check=governing_check,
        passes=governing_dcr <= 1.0,
        code_refs=[
            "ACI 318-19 §22.2 (flexure)",
            "ACI 318-19 §22.5 (shear)",
            "ACI 318-19 §9.6.1 (As_min)",
            "ACI 318-19 §9.7.6.2 (s_max)",
        ],
        notes=result.notes,
    )


def from_column_design_result(
    member_tag: str, result, demand,
) -> MemberReportEntry:
    """Convert a Phase 29 :class:`ColumnDesignResult` into a report entry."""
    if not _CONCRETE_AVAILABLE:
        raise RuntimeError("femsolver.design.concrete is not importable")
    if not result.success or result.section is None:
        return MemberReportEntry(
            member_tag=member_tag,
            member_type="RC column (ACI 318-19) -- DESIGN FAILED",
            section_summary="(no valid section)",
            material_summary="",
            governing_dcr=float("inf"),
            governing_check="design",
            passes=False,
            notes=result.notes,
        )
    s = result.section
    n_long = len(s.rebar.top_bars) + len(s.rebar.bottom_bars)
    section_summary = (
        f"{s.b * 1000:.0f} x {s.h * 1000:.0f} mm; "
        f"{n_long} long ({s.rebar.top_bars[0] if s.rebar.top_bars else '?'}); "
        f"rho = {result.rho * 100:.2f}%"
    )
    material_summary = (
        f"fc' = {s.material.fc_prime / 1e6:.0f} MPa, "
        f"fy = {s.material.fy / 1e6:.0f} MPa"
    )
    demands_d = {
        "P_u (compr)": f"{demand.P_u / 1e3:+.1f} kN",
        "M_u": f"{demand.M_u / 1e3:+.1f} kN.m",
    }
    phi_M_n_at_P_u = result.interaction_surface.phi_M_n_at_P_u(demand.P_u)
    capacities_d = {
        "phi*M_n @ P_u": f"{phi_M_n_at_P_u / 1e3:.1f} kN.m",
        "P_n_max (cap)": f"{result.interaction_surface.P_n_max / 1e3:.1f} kN",
    }
    dcrs_d = {"PM interaction": result.dcr}
    return MemberReportEntry(
        member_tag=member_tag,
        member_type="RC column (ACI 318-19)",
        section_summary=section_summary,
        material_summary=material_summary,
        demands=demands_d,
        capacities=capacities_d,
        dcrs=dcrs_d,
        governing_dcr=result.dcr,
        governing_check="PM interaction",
        passes=result.success and result.dcr <= 1.0,
        code_refs=[
            "ACI 318-19 §22.4 (axial-flexure interaction)",
            "ACI 318-19 §10.6.1.1 (ρ limits 1-8%)",
        ],
        notes=result.notes,
    )


def from_steel_member_check(
    member_tag: str, check, demand=None,
) -> MemberReportEntry:
    """Convert a Phase 30 :class:`SteelMemberCheck` into a report entry."""
    if not _STEEL_AVAILABLE:
        raise RuntimeError("femsolver.design.steel is not importable")
    s = check.section
    section_summary = (
        f"{s.designation}, A = {s.A * 1e4:.1f} cm² "
        f"({s.weight_per_length / 9.81:.1f} kg/m)"
    )
    material_summary = ""        # SteelMaterial not currently exposed on check
    demands_d: dict[str, str] = {}
    capacities_d: dict[str, str] = {}
    dcrs_d: dict[str, float] = {}
    if demand is not None:
        if abs(demand.P_u) > 0:
            demands_d["P_u"] = f"{demand.P_u / 1e3:+.1f} kN"
        if abs(demand.M_ux) > 0:
            demands_d["M_ux"] = f"{demand.M_ux / 1e3:+.1f} kN.m"
        if abs(demand.M_uy) > 0:
            demands_d["M_uy"] = f"{demand.M_uy / 1e3:+.1f} kN.m"
        if abs(demand.V_u) > 0:
            demands_d["V_u"] = f"{demand.V_u / 1e3:.1f} kN"
    if check.combined is not None:
        capacities_d["P_c (axial)"] = f"{check.combined.P_c / 1e3:.1f} kN"
        capacities_d["M_cx"] = f"{check.combined.M_cx / 1e3:.1f} kN.m"
        capacities_d["M_cy"] = f"{check.combined.M_cy / 1e3:.1f} kN.m"
        dcrs_d["Combined (H1)"] = check.combined.DCR
    if check.shear is not None:
        capacities_d["phi*V_n"] = f"{check.shear.phi_V_n / 1e3:.1f} kN"
        if demand is not None and abs(demand.V_u) > 0:
            dcrs_d["Shear (G)"] = abs(demand.V_u) / check.shear.phi_V_n
    return MemberReportEntry(
        member_tag=member_tag,
        member_type=f"Steel {s.designation} (AISC 360-22)",
        section_summary=section_summary,
        material_summary=material_summary,
        demands=demands_d,
        capacities=capacities_d,
        dcrs=dcrs_d,
        governing_dcr=check.governing_DCR,
        governing_check=check.governing_limit_state,
        passes=check.passes,
        code_refs=[
            "AISC 360-22 Ch. E (compression)",
            "AISC 360-22 Ch. F (flexure / LTB)",
            "AISC 360-22 Ch. G (shear)",
            "AISC 360-22 Ch. H (combined forces)",
        ],
        notes=check.notes,
    )


# ============================================================ HTML

_HTML_STYLE = """
<style>
  body { font-family: -apple-system, system-ui, sans-serif;
          max-width: 1100px; margin: 20px auto; padding: 0 12px;
          color: #222; }
  h1 { border-bottom: 2px solid #444; padding-bottom: 6px; }
  .member { border: 1px solid #ccc; border-radius: 6px;
             margin: 14px 0; padding: 12px 16px;
             background: #fafafa; }
  .member.pass { border-left: 6px solid #2ca02c; }
  .member.fail { border-left: 6px solid #d62728; }
  .header { display: flex; justify-content: space-between;
             align-items: baseline; flex-wrap: wrap; }
  .tag { font-size: 1.15em; font-weight: 600; }
  .status { font-weight: 700; }
  .status.pass { color: #1e7a1e; }
  .status.fail { color: #b1131c; }
  .meta { color: #555; font-size: 0.95em; margin: 4px 0; }
  table { border-collapse: collapse; width: 100%;
           margin-top: 8px; font-size: 0.9em; }
  th, td { text-align: left; padding: 4px 8px;
            border-bottom: 1px solid #eee; }
  th { background: #efefef; }
  .dcr-bar { background: #eee; border-radius: 3px;
              height: 14px; width: 200px; display: inline-block;
              vertical-align: middle; overflow: hidden; }
  .dcr-fill { height: 100%; background: #2ca02c; }
  .dcr-fill.warn { background: #ffb02e; }
  .dcr-fill.fail { background: #d62728; }
  .refs { font-size: 0.85em; color: #666; margin-top: 6px; }
  .notes { background: #fff8e0; padding: 4px 8px;
            border-left: 3px solid #d4a017; margin-top: 6px;
            font-size: 0.9em; }
</style>
"""


def _dcr_bar_html(dcr: float) -> str:
    pct = min(dcr, 1.5) / 1.5 * 100.0
    if dcr > 1.0:
        cls = "fail"
    elif dcr > 0.85:
        cls = "warn"
    else:
        cls = ""
    return (
        f'<span class="dcr-bar"><span class="dcr-fill {cls}" '
        f'style="width:{pct:.1f}%"></span></span> {dcr:.3f}'
    )


def _entry_html(entry: MemberReportEntry) -> str:
    pass_cls = "pass" if entry.passes else "fail"
    status_text = "PASS" if entry.passes else "FAIL"
    parts: list[str] = []
    parts.append(f'<div class="member {pass_cls}">')
    parts.append('<div class="header">')
    parts.append(f'<div class="tag">{entry.member_tag} &mdash; '
                  f'{entry.member_type}</div>')
    parts.append(f'<div class="status {pass_cls}">{status_text} '
                  f'(governing DCR = {entry.governing_dcr:.3f})</div>')
    parts.append('</div>')
    parts.append(f'<div class="meta">Section: {entry.section_summary}</div>')
    if entry.material_summary:
        parts.append(f'<div class="meta">Material: {entry.material_summary}</div>')
    if entry.demands or entry.capacities:
        parts.append('<table>')
        parts.append('<tr><th>Demand</th><th>Capacity</th>'
                       '<th>Check</th><th>DCR</th></tr>')
        # Pair demand/capacity rows; show DCRs separately if more
        labels = list(set(entry.demands) | set(entry.capacities))
        labels.sort()
        for lbl in labels:
            d = entry.demands.get(lbl, "")
            c = entry.capacities.get(lbl, "")
            parts.append(f'<tr><td>{lbl}: {d}</td>'
                          f'<td>{c}</td><td></td><td></td></tr>')
        for chk, dcr in entry.dcrs.items():
            parts.append(f'<tr><td></td><td></td>'
                          f'<td>{chk}</td><td>{_dcr_bar_html(dcr)}</td></tr>')
        parts.append('</table>')
    if entry.code_refs:
        parts.append('<div class="refs">Code references: ' +
                       ', '.join(entry.code_refs) + '</div>')
    if entry.notes:
        parts.append(f'<div class="notes">{entry.notes}</div>')
    parts.append('</div>')
    return '\n'.join(parts)


def make_html_report(
    entries,
    *,
    title: str = "Member Design Report",
) -> str:
    """Generate a self-contained HTML design report."""
    head = (
        '<!DOCTYPE html>\n<html><head>'
        f'<meta charset="utf-8"><title>{title}</title>'
        f'{_HTML_STYLE}</head><body>'
    )
    body = [f'<h1>{title}</h1>']
    # Summary
    n_total = len(entries)
    n_pass = sum(1 for e in entries if e.passes)
    n_fail = n_total - n_pass
    body.append(
        f'<p><b>{n_total}</b> member(s), '
        f'<span style="color:#1e7a1e"><b>{n_pass} pass</b></span>, '
        f'<span style="color:#b1131c"><b>{n_fail} fail</b></span></p>'
    )
    for e in entries:
        body.append(_entry_html(e))
    tail = '</body></html>'
    return head + '\n' + '\n'.join(body) + '\n' + tail


def write_html_report(
    entries, path: str, *, title: str = "Member Design Report",
) -> None:
    """Write the HTML report to ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(make_html_report(entries, title=title))


# ============================================================ CSV

def make_csv_summary(entries) -> str:
    """Generate a CSV string with one row per member."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "member_tag", "member_type",
        "section_summary", "material_summary",
        "governing_dcr", "governing_check",
        "passes", "notes",
    ])
    for e in entries:
        writer.writerow([
            e.member_tag, e.member_type,
            e.section_summary, e.material_summary,
            f"{e.governing_dcr:.4f}", e.governing_check,
            "PASS" if e.passes else "FAIL",
            e.notes,
        ])
    return buf.getvalue()


def write_csv_summary(entries, path: str) -> None:
    """Write the CSV summary to ``path``."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(make_csv_summary(entries))
