"""SVG sketcher for :class:`Section` (Theme II.8).

Produces a compact, self-contained SVG string suitable for embedding
in HTML reports, the eventual GUI section picker, and design
calc-sheets.

Output features
---------------
* Polygon outline (with holes drawn as cut-outs)
* Centroid cross-mark
* Rebar bars as filled circles (radius ~ sqrt(A/pi))
* Optional dimension annotations (depth + width)
* Right-handed coordinate system: +z right, +y up

The SVG is plain text -- no external assets, no JavaScript -- so it
can be safely written to disk or piped into a PDF generator.
"""
from __future__ import annotations

import math
from typing import Optional

from femsolver.sections.section import Section


_DEFAULT_W = 320     # pixels
_DEFAULT_PAD = 30    # pixel border


def section_to_svg(
    section: Section,
    *,
    width_px: int = _DEFAULT_W,
    pad_px: int = _DEFAULT_PAD,
    show_centroid: bool = True,
    show_rebar: bool = True,
    show_dimensions: bool = True,
    fill: str = "#cfe8ff",
    stroke: str = "#1f4f73",
    rebar_color: str = "#b03030",
) -> str:
    """Render a :class:`Section` as a standalone SVG string.

    Parameters
    ----------
    section : Section
        The section to render.
    width_px : int
        SVG width in pixels. Height is computed to preserve aspect
        ratio.
    pad_px : int
        Border padding in pixels.
    show_centroid : bool
        Draw a small cross at the section centroid.
    show_rebar : bool
        Draw rebar bars as filled circles.
    show_dimensions : bool
        Annotate overall depth and width.
    fill, stroke, rebar_color : str
        SVG color strings.
    """
    polygon = section.geometry.polygon
    minz, miny, maxz, maxy = polygon.bounds
    geo_w = maxz - minz
    geo_h = maxy - miny
    if geo_w <= 0 or geo_h <= 0:
        raise ValueError("section has degenerate bounding box")

    # Scale to fit (width_px - 2*pad_px) horizontally
    inner_w = width_px - 2 * pad_px
    inner_h = inner_w * (geo_h / geo_w)
    height_px = int(inner_h + 2 * pad_px)
    scale = inner_w / geo_w  # pixels per metre

    def _x(z: float) -> float:
        return pad_px + (z - minz) * scale

    def _y(y: float) -> float:
        # SVG y grows downward; invert
        return pad_px + (maxy - y) * scale

    # Build SVG
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">'
    )
    parts.append(
        f'<g font-family="Helvetica, sans-serif" '
        f'font-size="11" fill="#222">'
    )

    # Polygon outline (using SVG path so holes work via fill-rule)
    ext = list(polygon.exterior.coords)
    if len(ext) > 1 and ext[0] == ext[-1]:
        ext = ext[:-1]
    path_d_parts: list[str] = []
    if ext:
        path_d_parts.append(f"M {_x(ext[0][0]):.2f},{_y(ext[0][1]):.2f}")
        for (z, y) in ext[1:]:
            path_d_parts.append(f"L {_x(z):.2f},{_y(y):.2f}")
        path_d_parts.append("Z")
    for ring in polygon.interiors:
        h = list(ring.coords)
        if len(h) > 1 and h[0] == h[-1]:
            h = h[:-1]
        if not h:
            continue
        path_d_parts.append(f"M {_x(h[0][0]):.2f},{_y(h[0][1]):.2f}")
        for (z, y) in h[1:]:
            path_d_parts.append(f"L {_x(z):.2f},{_y(y):.2f}")
        path_d_parts.append("Z")
    parts.append(
        f'<path d="{" ".join(path_d_parts)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" '
        f'fill-rule="evenodd"/>'
    )

    # Centroid cross-mark
    if show_centroid:
        cz, cy = section.centroid
        px, py = _x(cz), _y(cy)
        parts.append(
            f'<g stroke="{stroke}" stroke-width="1">'
            f'<line x1="{px-5:.1f}" y1="{py:.1f}" '
            f'x2="{px+5:.1f}" y2="{py:.1f}"/>'
            f'<line x1="{px:.1f}" y1="{py-5:.1f}" '
            f'x2="{px:.1f}" y2="{py+5:.1f}"/>'
            f'</g>'
        )

    # Rebar dots
    if show_rebar and section.reinforcement and section.reinforcement.bars:
        for bar in section.reinforcement.bars:
            r_m = math.sqrt(bar.area / math.pi)
            r_px = max(2.0, r_m * scale)
            parts.append(
                f'<circle cx="{_x(bar.z):.2f}" cy="{_y(bar.y):.2f}" '
                f'r="{r_px:.2f}" fill="{rebar_color}" '
                f'stroke="#400" stroke-width="0.5"/>'
            )

    # Dimensions
    if show_dimensions:
        depth_mm = geo_h * 1000
        width_mm = geo_w * 1000
        # Width annotation below
        parts.append(
            f'<text x="{width_px/2:.1f}" y="{height_px - 6:.1f}" '
            f'text-anchor="middle" font-size="10">'
            f'{width_mm:.0f} mm</text>'
        )
        # Depth annotation on the right
        parts.append(
            f'<text x="{width_px - 4:.1f}" y="{height_px/2:.1f}" '
            f'text-anchor="end" font-size="10" '
            f'transform="rotate(-90 {width_px-4:.1f},{height_px/2:.1f})">'
            f'{depth_mm:.0f} mm</text>'
        )

    # Title (section name)
    if section.name:
        parts.append(
            f'<text x="{pad_px:.1f}" y="{pad_px - 8:.1f}" '
            f'font-weight="bold" font-size="11">'
            f'{_escape(section.name)}'
            f'</text>'
        )

    parts.append('</g></svg>')
    return "\n".join(parts)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )
