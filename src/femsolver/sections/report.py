"""One-page section report (Theme II.8).

A :class:`SectionReport` is a structured snapshot of a section's gross
properties, geometry, materials, and reinforcement, ready for
inclusion in HTML / PDF / calc-sheet outputs. It pairs naturally with
:func:`femsolver.sections.visualization.section_to_svg` (which
provides the visual sketch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from femsolver.sections.section import Section


@dataclass
class RebarRow:
    """One row in the rebar table of a section report."""
    designation: str
    z_mm: float
    y_mm: float
    area_mm2: float


@dataclass
class SectionReport:
    """One-page snapshot of a :class:`Section`.

    Designed to be easy to feed into HTML / PDF / calc-sheet builders.
    Numerical fields use engineering-display units (mm, mm^2, mm^4,
    mm^3 -- NOT raw SI). The :attr:`svg` field is the section sketch.
    """
    name: str
    family: str
    catalogue_ref: str | None

    # Gross geometric properties (engineering units)
    area_mm2: float
    depth_mm: float
    width_mm: float
    centroid_z_mm: float
    centroid_y_mm: float
    I_zz_mm4: float
    I_yy_mm4: float
    J_mm4: float
    Z_zz_mm3: float
    Z_yy_mm3: float
    S_zz_top_mm3: float
    S_zz_bot_mm3: float

    # Weight / fabrication
    weight_per_length_N_per_m: float
    paint_area_per_length_m2_per_m: float

    # Composition
    materials: list[str] = field(default_factory=list)
    rebar_rows: list[RebarRow] = field(default_factory=list)
    total_rebar_area_mm2: float = 0.0
    stirrup_info: str = ""

    # Visual
    svg: str = ""

    # ----------------------------------------------------- HTML
    def to_html(self) -> str:
        """Render this report as a small HTML fragment.

        Not a complete HTML document -- suitable for embedding in a
        larger report's body. CSS-friendly classes are applied so the
        host page can style it.
        """
        rebar_table = ""
        if self.rebar_rows:
            rows = "".join(
                f"<tr><td>{r.designation}</td>"
                f"<td>{r.z_mm:+.1f}</td><td>{r.y_mm:+.1f}</td>"
                f"<td>{r.area_mm2:.0f}</td></tr>"
                for r in self.rebar_rows
            )
            rebar_table = (
                '<h4>Reinforcement</h4>'
                '<table class="section-rebar"><thead><tr>'
                '<th>Bar</th><th>z (mm)</th><th>y (mm)</th>'
                '<th>A (mm²)</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>'
                f'<p>Total A_s = {self.total_rebar_area_mm2:.0f} mm²; '
                f'{self.stirrup_info}</p>'
            )
        ref = f' <span class="catref">[{self.catalogue_ref}]</span>' \
            if self.catalogue_ref else ""
        return (
            f'<div class="section-report">'
            f'<h3>{self.name}{ref}</h3>'
            f'<div class="section-sketch">{self.svg}</div>'
            f'<table class="section-props">'
            f'<tr><th>Family</th><td>{self.family}</td></tr>'
            f'<tr><th>Area</th><td>{self.area_mm2:.0f} mm²</td></tr>'
            f'<tr><th>Depth × Width</th>'
            f'<td>{self.depth_mm:.0f} × {self.width_mm:.0f} mm</td></tr>'
            f'<tr><th>I_zz (strong)</th>'
            f'<td>{self.I_zz_mm4:.3e} mm⁴</td></tr>'
            f'<tr><th>I_yy (weak)</th>'
            f'<td>{self.I_yy_mm4:.3e} mm⁴</td></tr>'
            f'<tr><th>J (torsion)</th>'
            f'<td>{self.J_mm4:.3e} mm⁴</td></tr>'
            f'<tr><th>Z_zz / Z_yy</th>'
            f'<td>{self.Z_zz_mm3:.0f} / {self.Z_yy_mm3:.0f} mm³</td></tr>'
            f'<tr><th>S_zz top/bot</th>'
            f'<td>{self.S_zz_top_mm3:.0f} / {self.S_zz_bot_mm3:.0f} mm³</td></tr>'
            f'<tr><th>Weight</th>'
            f'<td>{self.weight_per_length_N_per_m / 9.81:.1f} kg/m</td></tr>'
            f'<tr><th>Paint area</th>'
            f'<td>{self.paint_area_per_length_m2_per_m:.3f} m²/m</td></tr>'
            f'</table>'
            f'{rebar_table}'
            f'</div>'
        )


def build_section_report(section: Section, *, svg_width: int = 320) -> SectionReport:
    """Build a :class:`SectionReport` from a :class:`Section`."""
    from femsolver.sections.visualization import section_to_svg

    cz, cy = section.centroid
    g = section.geometry
    materials = []
    for z in section.zones:
        if z.material is not None:
            cls_name = type(z.material).__name__
            zone_label = f"{z.name} ({cls_name})" if z.name else cls_name
            materials.append(zone_label)

    rebar_rows: list[RebarRow] = []
    total_rebar = 0.0
    stirrup_info = ""
    if section.reinforcement and section.reinforcement.bars:
        for bar in section.reinforcement.bars:
            rebar_rows.append(RebarRow(
                designation=bar.designation,
                z_mm=bar.z * 1000,
                y_mm=bar.y * 1000,
                area_mm2=bar.area * 1e6,
            ))
            total_rebar += bar.area
        rl = section.reinforcement
        stirrup_info = (
            f"stirrups {rl.stirrup_designation} @ "
            f"{rl.stirrup_spacing*1000:.0f} mm, {rl.stirrup_legs} legs"
        )

    return SectionReport(
        name=section.name,
        family=section.family,
        catalogue_ref=section.catalogue_ref,
        area_mm2=g.area * 1e6,
        depth_mm=g.depth * 1000,
        width_mm=g.width * 1000,
        centroid_z_mm=cz * 1000,
        centroid_y_mm=cy * 1000,
        I_zz_mm4=g.I_zz * 1e12,
        I_yy_mm4=g.I_yy * 1e12,
        J_mm4=g.J * 1e12,
        Z_zz_mm3=g.Z_zz * 1e9,
        Z_yy_mm3=g.Z_yy * 1e9,
        S_zz_top_mm3=g.S_zz_top * 1e9,
        S_zz_bot_mm3=g.S_zz_bot * 1e9,
        weight_per_length_N_per_m=section.weight_per_length(),
        paint_area_per_length_m2_per_m=section.paint_area_per_length(),
        materials=materials,
        rebar_rows=rebar_rows,
        total_rebar_area_mm2=total_rebar * 1e6,
        stirrup_info=stirrup_info,
        svg=section_to_svg(section, width_px=svg_width),
    )
