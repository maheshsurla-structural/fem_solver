"""Phase II.5 tests -- custom polygon sections + Boolean ops.

Verifies:
* :func:`custom_polygon_section` accepts both raw vertices and pre-built
  :class:`PolygonGeometry`
* Boolean ops produce correct geometry:
  - subtract_polygons(outer_rect, inner_rect) == hollow_rect
  - union_polygons of two overlapping rectangles
  - subtract_polygons on multiple holes
* Disjoint Boolean results raise (single connected section invariant)
* Custom polygons can drive elastic adapter when material attached
"""
from __future__ import annotations

import pytest

from femsolver.sections import (
    PolygonGeometry,
    Section,
    custom_polygon_section,
    hollow_rect_section,
    rectangular_section,
    subtract_polygons,
    union_polygons,
)


class _DummySteel:
    E = 200e9
    nu = 0.3
    density = 7850.0


# ============================================================ factory

class TestCustomPolygonFactory:
    def test_from_outline(self):
        """L-shape via raw vertices."""
        # 200x60 horizontal + 100x140 vertical stem, in mm scaled to m
        sec = custom_polygon_section(
            outline=[
                (0, 0), (0.2, 0), (0.2, 0.06),
                (0.1, 0.06), (0.1, 0.2), (0, 0.2),
            ],
            material=_DummySteel(),
        )
        assert isinstance(sec, Section)
        assert sec.family == "polygon"
        # A = 200*60 + 100*(200-60) = 12000 + 14000 = 26000 mm^2 = 0.026 m^2
        assert sec.area == pytest.approx(0.026, rel=1e-10)

    def test_from_geometry(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        sec = custom_polygon_section(geometry=g, name="B1")
        assert sec.area == pytest.approx(0.18, rel=1e-12)
        assert sec.name == "B1"

    def test_rejects_both_inputs(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        with pytest.raises(ValueError, match="exactly one"):
            custom_polygon_section(outline=[(0, 0), (1, 0), (1, 1)], geometry=g)

    def test_rejects_neither_input(self):
        with pytest.raises(ValueError, match="exactly one"):
            custom_polygon_section()

    def test_rejects_holes_with_geometry(self):
        g = PolygonGeometry.rectangle(0.3, 0.6)
        with pytest.raises(ValueError, match="holes"):
            custom_polygon_section(
                geometry=g, holes=[[(0, 0), (0.1, 0), (0.1, 0.1)]],
            )

    def test_with_holes(self):
        sec = custom_polygon_section(
            outline=[(-0.2, -0.1), (0.2, -0.1), (0.2, 0.1), (-0.2, 0.1)],
            holes=[[(-0.1, -0.05), (0.1, -0.05), (0.1, 0.05), (-0.1, 0.05)]],
        )
        # 0.4*0.2 - 0.2*0.1 = 0.08 - 0.02 = 0.06
        assert sec.area == pytest.approx(0.06, rel=1e-10)

    def test_elastic_adapter_works(self):
        sec = custom_polygon_section(
            outline=[(-0.15, -0.3), (0.15, -0.3), (0.15, 0.3), (-0.15, 0.3)],
            material=_DummySteel(),
        )
        es = sec.elastic_section_3d()
        # EA = 200e9 * 0.18
        assert es.EA == pytest.approx(200e9 * 0.18, rel=1e-12)


# ============================================================ Boolean: subtract

class TestSubtract:
    def test_subtract_inner_rect_equals_hollow_rect(self):
        """subtract_polygons(outer, inner) should give the same gross
        properties as the parametric hollow_rect_section."""
        outer = PolygonGeometry.rectangle(0.2, 0.1)
        inner = PolygonGeometry.rectangle(0.188, 0.088)  # 6 mm wall
        composed = subtract_polygons(outer, inner)
        parametric = hollow_rect_section(b=0.2, h=0.1, t=0.006)
        assert composed.area == pytest.approx(parametric.area, rel=1e-10)
        assert composed.I_zz == pytest.approx(parametric.I_zz, rel=1e-8)

    def test_subtract_single_hole(self):
        base = PolygonGeometry.rectangle(0.4, 0.4)
        hole = PolygonGeometry.rectangle(0.1, 0.1)  # centred
        g = subtract_polygons(base, hole)
        assert g.area == pytest.approx(0.4 * 0.4 - 0.1 * 0.1, rel=1e-12)

    def test_subtract_multiple_holes(self):
        base = PolygonGeometry.rectangle(0.4, 0.4)
        h1 = PolygonGeometry.rectangle(0.05, 0.05, center=(0.1, 0.1))
        h2 = PolygonGeometry.rectangle(0.05, 0.05, center=(-0.1, 0.1))
        h3 = PolygonGeometry.rectangle(0.05, 0.05, center=(0.1, -0.1))
        h4 = PolygonGeometry.rectangle(0.05, 0.05, center=(-0.1, -0.1))
        g = subtract_polygons(base, [h1, h2, h3, h4])
        expected = 0.4 * 0.4 - 4 * (0.05 * 0.05)
        assert g.area == pytest.approx(expected, rel=1e-12)

    def test_subtract_into_section(self):
        outer = PolygonGeometry.rectangle(0.5, 0.3)
        hole = PolygonGeometry.rectangle(0.4, 0.2)
        g = subtract_polygons(outer, hole)
        sec = custom_polygon_section(
            geometry=g, name="HSS-like", material=_DummySteel(),
        )
        assert sec.area == pytest.approx(0.5 * 0.3 - 0.4 * 0.2, rel=1e-12)
        # Adapter works on a hole-containing section
        es = sec.elastic_section_3d()
        assert es.EA > 0

    def test_subtract_empty_holes_returns_base(self):
        base = PolygonGeometry.rectangle(0.3, 0.6)
        g = subtract_polygons(base, [])
        assert g is base

    def test_subtract_disjoint_raises(self):
        """Subtract a hole bigger than the base -> empty result, raises."""
        base = PolygonGeometry.rectangle(0.1, 0.1)
        hole = PolygonGeometry.rectangle(0.3, 0.3)   # fully contains base
        with pytest.raises(ValueError, match="empty"):
            subtract_polygons(base, hole)

    def test_subtract_split_disjoint_raises(self):
        """Subtract a centred bar that splits the base in two -> raises."""
        base = PolygonGeometry.rectangle(0.4, 0.1)
        # vertical bar that bisects: 0.05 wide, full height + slight overhang
        splitter = PolygonGeometry.rectangle(0.05, 0.2)
        with pytest.raises(ValueError, match="MultiPolygon"):
            subtract_polygons(base, splitter)


# ============================================================ Boolean: union

class TestUnion:
    def test_union_two_overlapping_rects_T_shape(self):
        """Union of a horizontal flange and a vertical web (overlapping).

        Flange: 300x50 at y=0.175..0.225, web: 50x400 at y=-0.2..0.2.
        These overlap in a 50x50 mm patch (z=-0.025..0.025, y=0.175..0.2).
        """
        flange = PolygonGeometry.rectangle(0.3, 0.05, center=(0, 0.200))
        web = PolygonGeometry.rectangle(0.05, 0.4, center=(0, 0.0))
        g = union_polygons(flange, web)
        expected = 0.3 * 0.05 + 0.05 * 0.4 - 0.05 * 0.025
        assert g.area == pytest.approx(expected, rel=1e-10)

    def test_union_two_touching_rects(self):
        """Two rectangles sharing an edge -> union is a bigger rect."""
        a = PolygonGeometry.rectangle(0.2, 0.1, center=(-0.1, 0))
        b = PolygonGeometry.rectangle(0.2, 0.1, center=(0.1, 0))
        g = union_polygons(a, b)
        # Combined rect: 0.4 x 0.1
        assert g.area == pytest.approx(0.4 * 0.1, rel=1e-12)

    def test_union_disjoint_raises(self):
        a = PolygonGeometry.rectangle(0.1, 0.1, center=(-1, 0))
        b = PolygonGeometry.rectangle(0.1, 0.1, center=(+1, 0))
        with pytest.raises(ValueError, match="MultiPolygon"):
            union_polygons(a, b)

    def test_union_needs_two_inputs(self):
        a = PolygonGeometry.rectangle(0.1, 0.1)
        with pytest.raises(ValueError, match="at least two"):
            union_polygons(a)

    def test_union_three_pieces_into_z_section(self):
        """Compose a Z-shape from three overlapping rectangles."""
        bot = PolygonGeometry.rectangle(0.2, 0.02, center=(0.05, -0.09))
        web = PolygonGeometry.rectangle(0.02, 0.2,  center=(0, 0))
        top = PolygonGeometry.rectangle(0.2, 0.02, center=(-0.05, 0.09))
        g = union_polygons(bot, web, top)
        # Each rect: 0.2 * 0.02 = 0.004; web: 0.02 * 0.2 = 0.004
        # Overlaps: bot/web at (z=-0.01..0.01, y=-0.1..-0.08) = 0.02*0.02 = 0.0004
        # Same for top/web at top
        expected = 3 * 0.004 - 2 * 0.0004
        assert g.area == pytest.approx(expected, rel=1e-10)


# ============================================================ Engineering scenario

class TestEngineeringComposition:
    def test_steel_plate_with_bolt_holes(self):
        """A 400x200 mm plate with 4 bolt holes (24mm dia approximated as
        rectangle for simplicity). Used in connection design."""
        plate = PolygonGeometry.rectangle(0.4, 0.2)
        d = 0.024
        # 4 holes near the corners (50mm from each edge)
        holes = [
            PolygonGeometry.rectangle(d, d, center=( 0.15,  0.075)),
            PolygonGeometry.rectangle(d, d, center=(-0.15,  0.075)),
            PolygonGeometry.rectangle(d, d, center=( 0.15, -0.075)),
            PolygonGeometry.rectangle(d, d, center=(-0.15, -0.075)),
        ]
        g = subtract_polygons(plate, holes)
        net_A = 0.4 * 0.2 - 4 * d * d
        assert g.area == pytest.approx(net_A, rel=1e-10)
        # Should drive an elastic section
        sec = custom_polygon_section(geometry=g, material=_DummySteel())
        es = sec.elastic_section_3d()
        assert es.EA == pytest.approx(200e9 * net_A, rel=1e-10)

    def test_box_section_via_subtract_then_section(self):
        """Build a 200x300x10 box section by subtraction; verify gross
        properties match the parametric hollow_rect_section."""
        outer = PolygonGeometry.rectangle(0.2, 0.3)
        inner = PolygonGeometry.rectangle(0.18, 0.28)
        g = subtract_polygons(outer, inner)
        sec_composed = custom_polygon_section(
            geometry=g, material=_DummySteel(),
        )
        sec_parametric = hollow_rect_section(
            b=0.2, h=0.3, t=0.01, material=_DummySteel(),
        )
        assert sec_composed.area == pytest.approx(sec_parametric.area, rel=1e-10)
        assert sec_composed.I_zz == pytest.approx(
            sec_parametric.I_zz, rel=1e-8,
        )
