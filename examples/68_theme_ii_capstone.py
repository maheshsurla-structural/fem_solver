"""Theme II (Section Designer) capstone.

Demonstrates the single-source-of-truth ``Section`` flow on 10 different
section types. Every section is built ONCE and flows through:

* gross-property query (A, I_zz, I_yy, J)
* elastic-section adapter -> beam element
* (where applicable) fiber-section adapter -> nonlinear analysis
* (where applicable) design-code adapter -> AISC/EC/IS/ACI legacy
* JSON round-trip -> save/load preserves identity + geometry
* SVG sketch -> visual deliverable
* Section report -> HTML / PDF / calc-sheet

This is the closing demonstration that Theme II's promise is met:
**every section in the system comes from one place**.
"""
from __future__ import annotations

import json
import sys

from femsolver.bridges.composite_section import (
    composite_girder_deck_section,
)
from femsolver.design.concrete.section import ConcreteMaterial
from femsolver.materials.uniaxial import UniaxialBilinear, UniaxialElastic
from femsolver.sections import (
    PolygonGeometry,
    ReinforcementLayout,
    aisc_section,
    circular_section,
    custom_polygon_section,
    eurocode_section,
    hollow_rect_section,
    i_section,
    indian_section,
    rc_rectangular_section,
    rectangular_section,
    section_from_json,
    subtract_polygons,
)


SEP = "=" * 78


def header(title: str) -> None:
    print()
    print(SEP)
    print(f" {title}")
    print(SEP)


def section_row(sec, kind: str = "") -> tuple:
    """One-line summary: (name, family, A_cm2, I_zz_cm4, J_cm4, weight_kg_m)."""
    A_cm2 = sec.area * 1e4
    I_zz_cm4 = sec.I_zz * 1e8
    J_cm4 = sec.J * 1e8
    w_kg_m = sec.weight_per_length() / 9.81
    return (sec.name, sec.family, kind, A_cm2, I_zz_cm4, J_cm4, w_kg_m)


def main() -> None:
    header("Theme II (Section Designer) capstone -- one source of truth")
    print("\nDemonstrating the unified Section flow for 10 section types.\n")

    # Common steel material for elastic adapters
    class _Steel:
        E = 200e9
        nu = 0.3
        density = 7850.0
    steel = _Steel()

    rows = []
    json_sizes = []

    # 1 -- Parametric rectangle
    s1 = rectangular_section(b=0.3, h=0.6, material=steel, name="B1 300x600")
    rows.append(section_row(s1, "parametric"))
    json_sizes.append(len(s1.to_json()))

    # 2 -- Parametric I-section (custom dimensions)
    s2 = i_section(
        h=0.4, b=0.2, t_f=0.015, t_w=0.009, material=steel,
        name="Custom I 400x200",
    )
    rows.append(section_row(s2, "parametric"))
    json_sizes.append(len(s2.to_json()))

    # 3 -- AISC catalogued
    s3 = aisc_section("W14x90", material=steel)
    rows.append(section_row(s3, "catalogue:AISC"))
    json_sizes.append(len(s3.to_json()))

    # 4 -- Eurocode catalogued
    s4 = eurocode_section("IPE 300", material=steel)
    rows.append(section_row(s4, "catalogue:EC"))
    json_sizes.append(len(s4.to_json()))

    # 5 -- Indian catalogued
    s5 = indian_section("ISMB 400", material=steel)
    rows.append(section_row(s5, "catalogue:IS"))
    json_sizes.append(len(s5.to_json()))

    # 6 -- Custom polygon (L-bracket)
    s6 = custom_polygon_section(
        outline=[
            (0, 0), (0.2, 0), (0.2, 0.06),
            (0.1, 0.06), (0.1, 0.2), (0, 0.2),
        ],
        material=steel,
        name="L-bracket 200x200x60x100",
    )
    rows.append(section_row(s6, "custom polygon"))
    json_sizes.append(len(s6.to_json()))

    # 7 -- Hollow rect via Boolean subtraction
    outer = PolygonGeometry.rectangle(0.2, 0.3)
    inner = PolygonGeometry.rectangle(0.18, 0.28)
    s7 = custom_polygon_section(
        geometry=subtract_polygons(outer, inner),
        material=steel,
        name="HSS 200x300x10 (subtract)",
    )
    rows.append(section_row(s7, "Boolean subtract"))
    json_sizes.append(len(s7.to_json()))

    # 8 -- Parametric hollow rect for comparison
    s8 = hollow_rect_section(b=0.2, h=0.3, t=0.01, material=steel,
                              name="HSS 200x300x10 (parametric)")
    rows.append(section_row(s8, "parametric hollow"))
    json_sizes.append(len(s8.to_json()))

    # 9 -- RC rectangular with rebar
    cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
    rebar_steel = UniaxialBilinear(E=200e9, sigma_y=420e6, b=0.01)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=0.3, h=0.6,
        bottom_bars=[(510e-6, "#8")] * 3,
        top_bars=[(285e-6, "#6")] * 2,
        steel_material=rebar_steel,
    )
    s9 = rc_rectangular_section(
        b=0.3, h=0.6, concrete=cm, reinforcement=rl, name="B1 RC 300x600",
    )
    rows.append(section_row(s9, "RC w/ rebar"))
    json_sizes.append(len(s9.to_json()))

    # 10 -- Bridge composite girder + deck
    s10 = composite_girder_deck_section(
        girder_width=0.6, girder_height=1.5,
        deck_width=3.0, deck_thickness=0.25,
        girder_material=ConcreteMaterial(fc_prime=50e6, fy=420e6),
        deck_material=ConcreteMaterial(fc_prime=30e6, fy=420e6),
        name="PSC girder + 3000x250 deck",
    )
    rows.append(section_row(s10, "bridge composite"))
    json_sizes.append(len(s10.to_json()))

    # 11 (bonus) -- Circular pile
    s11 = circular_section(D=0.6, material=cm, name="Pile D=600")
    rows.append(section_row(s11, "parametric circular"))
    json_sizes.append(len(s11.to_json()))

    # ---------------------------------------------------- print summary
    print(f"{'Section':32s} {'Family':16s} {'Kind':22s} "
          f"{'A':>8s} {'Izz':>10s} {'J':>10s} {'kg/m':>7s}")
    print("-" * 110)
    for (name, family, kind, A, Izz, J, w) in rows:
        print(f"{name:32s} {family:16s} {kind:22s} "
              f"{A:8.1f} {Izz:10.1f} {J:10.2f} {w:7.1f}")

    # ---------------------------------------------------- adapter flow
    header("Adapter flow -- each section flows through analysis + design + I/O")

    print("\nElastic adapter (every section -> ElasticSection3D):")
    for sec, label in [
        (s1, "param rect"),
        (s3, "AISC W14x90"),
        (s4, "IPE 300"),
        (s11, "Pile D=600"),
    ]:
        es = sec.elastic_section_3d()
        print(f"  {label:18s} EA={es.EA:8.2e} GJ={es.GJ:8.2e} EIz={es.EIz:8.2e}")

    print("\nFiber adapter (auto-discretization for nonlinear analysis):")
    mat = UniaxialElastic(E=30e9)
    fs1 = s1.fiber_section_2d(material=mat, n_z=4, n_y=10)
    print(f"  param rect (4x10):  {len(fs1.fibers):3d} fibers, A={fs1.gross_area*1e4:.1f} cm^2")
    fs9 = s9.fiber_section_2d(material=mat, n_z=4, n_y=10)
    print(f"  RC w/ rebar:        {len(fs9.fibers):3d} fibers (40 concrete + 5 rebar)")
    fs7 = s7.fiber_section_2d(material=mat, n_z=20, n_y=20)
    A_expected = 0.2 * 0.3 - 0.18 * 0.28
    print(f"  hollow via subtract: {len(fs7.fibers):3d} fibers, "
          f"A={fs7.gross_area*1e4:.2f} cm^2 (expected {A_expected*1e4:.2f})")

    print("\nDesign adapter (catalogue / AISC / EC / IS round-trip to legacy):")
    print(f"  W14x90.as_aisc_section()  -> "
          f"{s3.as_aisc_section().designation}, "
          f"A={s3.as_aisc_section().A*1e4:.2f} cm^2")
    print(f"  IPE 300.as_eurocode_section() -> "
          f"{s4.as_eurocode_section().name}, family={s4.as_eurocode_section().family}")
    print(f"  ISMB 400.as_indian_section()  -> "
          f"{s5.as_indian_section().name}")
    print(f"  RC.as_aci_concrete_section() -> "
          f"b={s9.as_aci_concrete_section().b}, h={s9.as_aci_concrete_section().h}, "
          f"bottom={s9.as_aci_concrete_section().rebar.bottom_bars}, "
          f"top={s9.as_aci_concrete_section().rebar.top_bars}")

    # ---------------------------------------------------- JSON round-trip
    header("JSON round-trip -- save / load preserves identity")

    print(f"\n{'Section':32s} {'JSON size':>10s}  A round-trip exact?")
    print("-" * 65)
    secs_all = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
    for sec, sz in zip(secs_all, json_sizes):
        sec_loaded = section_from_json(sec.to_json())
        match = abs(sec_loaded.area - sec.area) / max(sec.area, 1e-30) < 1e-12
        mark = "yes" if match else f"NO ({sec_loaded.area:.4g} vs {sec.area:.4g})"
        print(f"  {sec.name:32s} {sz:>9d} B  {mark}")

    # ---------------------------------------------------- SVG + report
    header("SVG sketch + section report -- one-line API")

    print()
    print(f"  s1.to_svg()       -> {len(s1.to_svg()):d} chars (standalone SVG)")
    print(f"  s9.to_svg()       -> {len(s9.to_svg()):d} chars (includes 5 rebar dots)")
    print(f"  s3.section_report().to_html() -> "
          f"{len(s3.section_report().to_html()):d} chars (HTML fragment)")

    # ---------------------------------------------------- finale
    header("Theme II closed -- 8 of 9 caveats from the original audit "
            "now Production")
    print()
    print("Every section type in the codebase now comes from one place:")
    print(f"  - 8 parametric primitives (rect, I, T, channel, angle, hollow_rect,")
    print(f"    circular, hollow_circular)")
    print(f"  - 3 catalogues unified (AISC 45 + EC 56 + IS 32 = 133 named sections)")
    print(f"  - Custom polygons + Boolean ops (union, subtract)")
    print(f"  - RC + bridge composite composition helpers")
    print(f"  - Lazy adapters to legacy ElasticSection / FiberSection")
    print(f"  - Lazy adapters to design dataclasses (AISC / EC / IS / ACI)")
    print(f"  - JSON round-trip, SVG sketcher, HTML section report")
    print()
    print("See docs/source/claims_matrix.md and docs/source/phase_ii_complete.md")
    print("for the full audit trail. Next: Phase C (vendor V&V) or Phase D")
    print("(new material classes -- timber, masonry, glass, aluminum, CFS).")


if __name__ == "__main__":
    sys.exit(main())
