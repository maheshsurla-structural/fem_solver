"""Pure-geometry layer for the Section Designer (Theme II.2+).

A ``Geometry`` is *shape only* -- no material, no analysis, no
reinforcement. It carries:

* a polygon outline (with optional holes), via ``shapely``
* gross geometric properties (A, centroid, I_yy, I_zz, J approximation,
  S, Z, ρ_paint area)
* visualization helpers (to_polygon, to_svg)

Geometry is decoupled from :class:`~femsolver.sections.section.Section`
so the same shape can be composed with different materials, different
reinforcement, or different analysis strategies.

Public API
----------
* :class:`Geometry` -- abstract base
* :class:`PolygonGeometry` -- general polygon (with optional holes),
  shapely-backed
* :func:`shoelace_area`, :func:`polygon_centroid`,
  :func:`polygon_second_moments` -- low-level helpers (also work on
  raw vertex tuples, no shapely needed)
"""
from femsolver.sections.geometry.base import Geometry
from femsolver.sections.geometry.polygon import (
    PolygonGeometry,
    polygon_centroid,
    polygon_second_moments,
    shoelace_area,
)

__all__ = [
    "Geometry",
    "PolygonGeometry",
    "shoelace_area",
    "polygon_centroid",
    "polygon_second_moments",
]
