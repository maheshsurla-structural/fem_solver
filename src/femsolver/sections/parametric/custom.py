"""Custom-polygon sections + polygon Boolean operations (Theme II.5).

``PolygonGeometry`` (delivered in II.2) already supports arbitrary
polygons with optional holes. This module wraps it in two
user-facing affordances:

1. :func:`custom_polygon_section` -- one-shot factory that takes
   either raw vertices or a pre-built :class:`PolygonGeometry` and
   returns a unified :class:`Section`. Right next to the parametric
   primitive factories.
2. :func:`union_polygons` / :func:`subtract_polygons` -- shapely-
   backed Boolean composition so a user can build complex sections
   by combining simple pieces:

   .. code-block:: python

       outer = PolygonGeometry.rectangle(0.4, 0.4)
       h1 = PolygonGeometry.rectangle(0.1, 0.1, center=(0.1, 0.1))
       h2 = PolygonGeometry.rectangle(0.1, 0.1, center=(-0.1, 0.1))
       g  = subtract_polygons(outer, [h1, h2])
       sec = custom_polygon_section(geometry=g, material=steel)

The result is always a single :class:`PolygonGeometry` (possibly
with interior rings). Disjoint multi-polygon results raise --
femsolver's :class:`Section` represents one connected cross-section.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from shapely.ops import unary_union

from femsolver.sections.geometry.polygon import PolygonGeometry
from femsolver.sections.section import MaterialZone, Section


Vertex = tuple[float, float]


def custom_polygon_section(
    *,
    outline: Optional[Sequence[Vertex]] = None,
    holes: Optional[list[Sequence[Vertex]]] = None,
    geometry: Optional[PolygonGeometry] = None,
    material: Any = None,
    name: Optional[str] = None,
) -> Section:
    """Build a :class:`Section` from a user-defined polygon.

    Two calling forms (exactly one required):

    * Pass ``outline`` (and optionally ``holes``) as lists of ``(z, y)``
      vertex tuples; the function constructs a :class:`PolygonGeometry`
      internally.
    * Pass an already-constructed ``geometry`` (e.g. the result of
      :func:`union_polygons` or :func:`subtract_polygons`).

    Parameters
    ----------
    outline : sequence of (z, y), optional
        Outer-boundary vertices.
    holes : list of vertex sequences, optional
        Internal hole loops (each one a polygon).
    geometry : PolygonGeometry, optional
        Pre-built geometry; mutually exclusive with ``outline``.
    material : optional
        Material reference; attached as a single :class:`MaterialZone`.
    name : str, optional
        Section name; defaults to ``"polygon"``.
    """
    have_outline = outline is not None
    have_geometry = geometry is not None
    if have_outline == have_geometry:
        raise ValueError(
            "pass exactly one of `outline` or `geometry`"
        )
    if have_outline:
        geometry = PolygonGeometry(outline, holes=holes)
    elif holes is not None:
        raise ValueError(
            "`holes` is only valid when passing `outline`; merge into "
            "`geometry` first"
        )

    zones = []
    if material is not None:
        zones.append(MaterialZone(material=material, name="custom"))
    return Section(
        geometry=geometry,
        zones=zones,
        name=name or "polygon",
        family="polygon",
    )


# ============================================================ Boolean ops

def _shapely_polygon_to_geometry(result) -> PolygonGeometry:
    """Convert a shapely Polygon result to a :class:`PolygonGeometry`,
    handling exterior closing vertex and interior rings."""
    if result.is_empty:
        raise ValueError("Boolean operation produced an empty polygon")
    if result.geom_type != "Polygon":
        # Could be MultiPolygon (disjoint pieces) or GeometryCollection.
        raise ValueError(
            f"Boolean operation produced {result.geom_type!r}; "
            f"femsolver Section represents one connected cross-section. "
            f"If you need disjoint pieces, build separate Sections."
        )
    # Drop the duplicate closing vertex that shapely appends.
    ext = list(result.exterior.coords)
    if len(ext) > 1 and ext[0] == ext[-1]:
        ext = ext[:-1]
    holes: list[list[Vertex]] = []
    for ring in result.interiors:
        h = list(ring.coords)
        if len(h) > 1 and h[0] == h[-1]:
            h = h[:-1]
        holes.append(h)
    return PolygonGeometry(ext, holes=holes if holes else None)


def union_polygons(*geometries: PolygonGeometry) -> PolygonGeometry:
    """Union (logical OR) of two or more polygons.

    Returns a single :class:`PolygonGeometry`. Raises if the union is
    disjoint (i.e. the inputs don't overlap or touch).
    """
    if len(geometries) < 2:
        raise ValueError("union_polygons needs at least two inputs")
    result = unary_union([g.polygon for g in geometries])
    return _shapely_polygon_to_geometry(result)


def subtract_polygons(
    base: PolygonGeometry,
    holes: Iterable[PolygonGeometry] | PolygonGeometry,
) -> PolygonGeometry:
    """Subtract one or more polygons from a base polygon.

    The holes become internal rings (or modify the exterior if they
    intersect the boundary). Common uses:

    * Hollow box: ``subtract_polygons(outer_rect, [inner_rect])``
    * Steel-plate with bolt holes: subtract many small circles from a
      large rectangle
    * Composite girder with cope: subtract a small corner rectangle

    Returns a single :class:`PolygonGeometry`. Raises if the subtract
    splits the base into disjoint pieces.
    """
    if isinstance(holes, PolygonGeometry):
        hole_list = [holes]
    else:
        hole_list = list(holes)
    if not hole_list:
        return base
    hole_union = unary_union([h.polygon for h in hole_list])
    result = base.polygon.difference(hole_union)
    return _shapely_polygon_to_geometry(result)
