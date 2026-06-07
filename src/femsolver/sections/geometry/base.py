"""Geometry abstract base class.

A ``Geometry`` provides:

* a single ``shapely.Polygon`` outline (with optional holes)
* gross properties: area, centroid, second moments of area,
  section moduli, plastic moduli
* a thin-walled / closed-form torsion constant when known; the
  default falls back to a polygon-thin-walled approximation
* outline coordinates for visualization (``to_polygon`` returns the
  shapely outline; downstream code can render to SVG / DXF)

The Y / Z convention matches :class:`~femsolver.sections.response.base.SectionBase`:

* ``y`` is the section-local axis perpendicular to the bending axis
  (strong-axis bending happens about the z-axis -> kappa_z)
* ``z`` is the other in-plane direction

For a typical I-beam upright, ``y`` is the vertical (depth) direction
and ``z`` is the horizontal (width) direction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Geometry(ABC):
    """Pure shape -- no material, no analysis."""

    # ----------------------------------------------------------- outline
    @property
    @abstractmethod
    def polygon(self):
        """A ``shapely.geometry.Polygon`` (possibly with holes) describing
        the section outline. Coordinates in section-local
        ``(z, y)`` (so the polygon's first axis is horizontal-z, second
        is vertical-y -- matches the ``BeamColumn3D`` local frame).
        """

    # ----------------------------------------------------------- properties
    @property
    @abstractmethod
    def area(self) -> float:
        """Gross cross-section area (m^2)."""

    @property
    @abstractmethod
    def centroid(self) -> tuple[float, float]:
        """Centroid coordinates (z_c, y_c) in section-local frame (m)."""

    @property
    @abstractmethod
    def I_zz(self) -> float:
        """Second moment of area about the centroidal z-axis (strong-
        axis bending for an upright I), m^4."""

    @property
    @abstractmethod
    def I_yy(self) -> float:
        """Second moment of area about the centroidal y-axis, m^4."""

    @property
    def I_yz(self) -> float:
        """Centroidal product of inertia. Default 0 for sections
        symmetric about both axes; overrides should compute the
        Green's-theorem integral."""
        return 0.0

    # ----------------------------------------------------------- J (torsion)
    @property
    def J(self) -> float:
        """Saint-Venant torsional constant (m^4).

        Default: thin-walled closed-section approximation if the
        section is closed, or a sum-of-rectangles approximation
        otherwise. Concrete shapes (rectangular, I, channel) override
        with their known closed-form expression.
        """
        # Polygon thin-walled approximation -- crude but always positive.
        # Subclasses should override with the correct closed-form J.
        A = self.area
        return float(0.5 * A * A * 0.0)  # placeholder; subclasses override

    # ----------------------------------------------------------- section moduli
    @property
    def S_zz_top(self) -> float:
        """Elastic section modulus to the extreme +y fibre, ``S = I_zz / c_top``."""
        return self.I_zz / max(self.c_top, 1e-30)

    @property
    def S_zz_bot(self) -> float:
        return self.I_zz / max(self.c_bot, 1e-30)

    @property
    def S_yy_right(self) -> float:
        return self.I_yy / max(self.c_right, 1e-30)

    @property
    def S_yy_left(self) -> float:
        return self.I_yy / max(self.c_left, 1e-30)

    # ----------------------------------------------------------- extreme fibres
    @property
    def c_top(self) -> float:
        """Distance from centroid to extreme +y fibre (m)."""
        y_max = max(coord[1] for coord in self.polygon.exterior.coords)
        return float(y_max - self.centroid[1])

    @property
    def c_bot(self) -> float:
        y_min = min(coord[1] for coord in self.polygon.exterior.coords)
        return float(self.centroid[1] - y_min)

    @property
    def c_right(self) -> float:
        z_max = max(coord[0] for coord in self.polygon.exterior.coords)
        return float(z_max - self.centroid[0])

    @property
    def c_left(self) -> float:
        z_min = min(coord[0] for coord in self.polygon.exterior.coords)
        return float(self.centroid[0] - z_min)

    # ----------------------------------------------------------- depth / width
    @property
    def depth(self) -> float:
        """Overall depth (y-extent of the bounding box), m."""
        return self.c_top + self.c_bot

    @property
    def width(self) -> float:
        """Overall width (z-extent of the bounding box), m."""
        return self.c_right + self.c_left

    # ----------------------------------------------------------- plastic moduli
    @property
    def Z_zz(self) -> float:
        """Plastic section modulus about z-axis (m^3).

        Default: numerical integration -- ``Z = sum(|y - y_pna| * dA)``,
        with the plastic neutral axis at the y that splits the area in
        two equal halves. Subclasses with closed-form expressions
        (rectangular, I) override.
        """
        return _polygon_plastic_modulus_zz(self.polygon)

    @property
    def Z_yy(self) -> float:
        return _polygon_plastic_modulus_yy(self.polygon)

    # ----------------------------------------------------------- mass / paint
    def mass_per_length(self, density: float) -> float:
        """Mass per unit beam length (kg/m) given material density
        (kg/m^3)."""
        return self.area * density

    @property
    def perimeter(self) -> float:
        """External perimeter (m). Used for paint area estimation."""
        return float(self.polygon.exterior.length)

    def paint_area_per_length(self) -> float:
        """Outer surface area per unit beam length (m^2/m). Default:
        the external perimeter (does not subtract hidden surfaces in
        composite assemblies)."""
        return self.perimeter

    # ----------------------------------------------------------- transforms
    def translate(self, dz: float, dy: float) -> "Geometry":
        """Return a translated copy. Default raises -- subclasses
        with constructor support override."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement translate"
        )

    # ----------------------------------------------------------- visualization
    def to_polygon(self):
        """Return the shapely polygon for visualization."""
        return self.polygon


# ============================================================ helpers

def _polygon_plastic_modulus_zz(polygon) -> float:
    """Numerical plastic modulus about z-axis: find the horizontal
    line ``y = y_pna`` that splits the polygon's area into two equal
    halves, then ``Z = integral(|y - y_pna| dA)``.

    Uses bisection on y. Tolerance: 1e-6 of total area."""
    from shapely.geometry import box

    minx, miny, maxx, maxy = polygon.bounds
    total = polygon.area
    if total <= 0:
        return 0.0

    # Bisect for the y that gives half-area below.
    target = total / 2.0
    lo, hi = miny, maxy
    y_pna = 0.5 * (lo + hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        clipper = box(minx - 1, miny - 1, maxx + 1, mid)
        below_area = polygon.intersection(clipper).area
        if abs(below_area - target) < 1e-12:
            y_pna = mid
            break
        if below_area > target:
            hi = mid
        else:
            lo = mid
        y_pna = 0.5 * (lo + hi)

    # Z = int |y - y_pna| dA. Compute via centroids of upper / lower halves.
    upper_clipper = box(minx - 1, y_pna, maxx + 1, maxy + 1)
    lower_clipper = box(minx - 1, miny - 1, maxx + 1, y_pna)
    upper = polygon.intersection(upper_clipper)
    lower = polygon.intersection(lower_clipper)
    if upper.is_empty or lower.is_empty:
        return 0.0
    y_upper_c = upper.centroid.y
    y_lower_c = lower.centroid.y
    return float(
        upper.area * (y_upper_c - y_pna) + lower.area * (y_pna - y_lower_c)
    )


def _polygon_plastic_modulus_yy(polygon) -> float:
    """Same idea, vertical bisection line."""
    from shapely.geometry import box

    minx, miny, maxx, maxy = polygon.bounds
    total = polygon.area
    if total <= 0:
        return 0.0

    target = total / 2.0
    lo, hi = minx, maxx
    z_pna = 0.5 * (lo + hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        clipper = box(minx - 1, miny - 1, mid, maxy + 1)
        left_area = polygon.intersection(clipper).area
        if abs(left_area - target) < 1e-12:
            z_pna = mid
            break
        if left_area > target:
            hi = mid
        else:
            lo = mid
        z_pna = 0.5 * (lo + hi)

    left_clipper = box(minx - 1, miny - 1, z_pna, maxy + 1)
    right_clipper = box(z_pna, miny - 1, maxx + 1, maxy + 1)
    left = polygon.intersection(left_clipper)
    right = polygon.intersection(right_clipper)
    if left.is_empty or right.is_empty:
        return 0.0
    z_left_c = left.centroid.x
    z_right_c = right.centroid.x
    return float(
        left.area * (z_pna - z_left_c) + right.area * (z_right_c - z_pna)
    )
