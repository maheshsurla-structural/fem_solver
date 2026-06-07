"""Parametric section primitives (Theme II.3).

This module is the user-facing factory layer for *standard parametric
section shapes*. Every factory function returns a unified
:class:`~femsolver.sections.section.Section` whose underlying
:class:`~femsolver.sections.geometry.Geometry` is a specialised
subclass with **closed-form** geometric properties (not the
polygon-bisection fallback used by :class:`PolygonGeometry`).

Public factories
----------------
* :func:`rectangular_section` -- solid rectangle (b x h)
* :func:`i_section` -- symmetric I-shape (W, IPE, HEA, ISMB families)
* :func:`t_section` -- inverted-T (asymmetric)
* :func:`channel_section` -- C-shape (asymmetric about z)
* :func:`angle_section` -- L-shape (asymmetric about both axes)
* :func:`hollow_rect_section` -- RHS / SHS (rectangular tube)
* :func:`circular_section` -- solid round bar
* :func:`hollow_circular_section` -- CHS (round tube)

Each factory takes parametric dimensions and an optional ``material``.
The closed-form torsion constant ``J`` and (where applicable) plastic
modulus ``Z`` are computed from the standard textbook formulas (Roark
7th ed, Boresi & Schmidt 6th ed, AISC Design Guide 9 for torsion).
"""
from femsolver.sections.parametric.geometry import (
    AngleGeometry,
    ChannelGeometry,
    CircularGeometry,
    HollowCircularGeometry,
    HollowRectGeometry,
    ISectionGeometry,
    RectangularGeometry,
    TSectionGeometry,
)
from femsolver.sections.parametric.custom import (
    custom_polygon_section,
    subtract_polygons,
    union_polygons,
)
from femsolver.sections.parametric.factory import (
    angle_section,
    channel_section,
    circular_section,
    hollow_circular_section,
    hollow_rect_section,
    i_section,
    rc_rectangular_section,
    rectangular_section,
    t_section,
)

__all__ = [
    # factories
    "rectangular_section",
    "i_section",
    "t_section",
    "channel_section",
    "angle_section",
    "hollow_rect_section",
    "circular_section",
    "hollow_circular_section",
    "rc_rectangular_section",
    "custom_polygon_section",
    "union_polygons",
    "subtract_polygons",
    # geometry classes
    "RectangularGeometry",
    "ISectionGeometry",
    "TSectionGeometry",
    "ChannelGeometry",
    "AngleGeometry",
    "HollowRectGeometry",
    "CircularGeometry",
    "HollowCircularGeometry",
]
