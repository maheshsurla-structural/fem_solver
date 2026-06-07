"""General polygon geometry, shapely-backed.

:class:`PolygonGeometry` is the workhorse for arbitrary section
outlines: rectangles built from corners, custom shapes from CAD,
I-beams built from triangulated outlines, etc. It supports holes
(via shapely's ``Polygon(exterior, holes=[...])``) so hollow
sections and box girders fit into the same model.

The class computes all gross properties analytically via Green's
theorem (Shoelace + closed-form integrals for I and product of
inertia). These match shapely's ``area`` and ``centroid`` to round-
off but are computed explicitly so users can verify against hand
calculations.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from shapely.geometry import Polygon

from femsolver.sections.geometry.base import Geometry


Vertex = tuple[float, float]


# ============================================================ low-level helpers

def shoelace_area(vertices: Sequence[Vertex]) -> float:
    """Signed area of a polygon (positive when CCW).

    Vertices are ``(z, y)`` tuples. Implements the Shoelace formula:

        A = 0.5 * sum_{i} (z_i * y_{i+1} - z_{i+1} * y_i)
    """
    if len(vertices) < 3:
        return 0.0
    s = 0.0
    n = len(vertices)
    for i in range(n):
        z1, y1 = vertices[i]
        z2, y2 = vertices[(i + 1) % n]
        s += z1 * y2 - z2 * y1
    return 0.5 * s


def polygon_centroid(vertices: Sequence[Vertex]) -> tuple[float, float]:
    """Centroid of a simple polygon via Green's theorem."""
    A_signed = shoelace_area(vertices)
    if abs(A_signed) < 1e-30:
        return 0.0, 0.0
    cz = 0.0
    cy = 0.0
    n = len(vertices)
    for i in range(n):
        z1, y1 = vertices[i]
        z2, y2 = vertices[(i + 1) % n]
        cross = z1 * y2 - z2 * y1
        cz += (z1 + z2) * cross
        cy += (y1 + y2) * cross
    factor = 1.0 / (6.0 * A_signed)
    return cz * factor, cy * factor


def polygon_second_moments(
    vertices: Sequence[Vertex],
) -> tuple[float, float, float]:
    """Second moments of area about the **origin** via Green's theorem.

    Returns ``(I_yy_0, I_zz_0, I_yz_0)`` -- still about the origin, not
    yet centroid-shifted. The caller (or :class:`PolygonGeometry`)
    applies the parallel-axis correction to get centroidal values.

    Standard formulas:

        I_zz = (1/12) * sum (y1^2 + y1 y2 + y2^2) * cross
        I_yy = (1/12) * sum (z1^2 + z1 z2 + z2^2) * cross
        I_yz = (1/24) * sum (z1 y2 + 2 z1 y1 + 2 z2 y2 + z2 y1) * cross

    where ``cross = z1 y2 - z2 y1``.
    """
    I_zz_0 = 0.0
    I_yy_0 = 0.0
    I_yz_0 = 0.0
    n = len(vertices)
    for i in range(n):
        z1, y1 = vertices[i]
        z2, y2 = vertices[(i + 1) % n]
        cross = z1 * y2 - z2 * y1
        I_zz_0 += (y1 * y1 + y1 * y2 + y2 * y2) * cross
        I_yy_0 += (z1 * z1 + z1 * z2 + z2 * z2) * cross
        I_yz_0 += (z1 * y2 + 2 * z1 * y1 + 2 * z2 * y2 + z2 * y1) * cross
    return I_yy_0 / 12.0, I_zz_0 / 12.0, I_yz_0 / 24.0


# ============================================================ PolygonGeometry

class PolygonGeometry(Geometry):
    """General polygon section, with optional internal holes.

    Parameters
    ----------
    exterior : Sequence[(z, y)]
        Vertices of the outer boundary. Order does not matter -- the
        class enforces CCW exterior internally.
    holes : list[Sequence[(z, y)]] | None
        Inner-boundary loops for holes. Each hole must be inside the
        exterior. CW hole order is enforced internally.
    """

    def __init__(
        self,
        exterior: Sequence[Vertex],
        holes: list[Sequence[Vertex]] | None = None,
    ):
        if len(exterior) < 3:
            raise ValueError("polygon needs at least 3 vertices")
        # Enforce CCW exterior, CW holes (shapely convention is the
        # opposite of CW/CCW depending on version; we orient explicitly).
        ext = list(exterior)
        if shoelace_area(ext) < 0:
            ext = list(reversed(ext))
        hole_loops: list[list[Vertex]] = []
        for hole in (holes or []):
            h = list(hole)
            if shoelace_area(h) > 0:
                h = list(reversed(h))
            hole_loops.append(h)
        self._polygon = Polygon(ext, holes=hole_loops)
        if not self._polygon.is_valid:
            raise ValueError(
                "polygon is not valid (self-intersecting or degenerate)"
            )
        self._ext = ext
        self._holes = hole_loops
        self._cache: dict = {}

    # --------------------------------------------------- outline / props
    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def area(self) -> float:
        if "area" not in self._cache:
            A_ext = abs(shoelace_area(self._ext))
            A_holes = sum(abs(shoelace_area(h)) for h in self._holes)
            self._cache["area"] = float(A_ext - A_holes)
        return self._cache["area"]

    @property
    def centroid(self) -> tuple[float, float]:
        if "centroid" not in self._cache:
            # Area-weighted: exterior - holes. Force CCW orientation
            # on every loop so polygon_centroid returns the correct
            # signs (the constructor stores holes as CW for shapely).
            A_ext = abs(shoelace_area(self._ext))
            cz_ext, cy_ext = polygon_centroid(self._ext)
            num_z = cz_ext * A_ext
            num_y = cy_ext * A_ext
            for h in self._holes:
                h_ccw = h if shoelace_area(h) > 0 else list(reversed(h))
                A_h = abs(shoelace_area(h_ccw))
                cz_h, cy_h = polygon_centroid(h_ccw)
                num_z -= cz_h * A_h
                num_y -= cy_h * A_h
            A = self.area
            self._cache["centroid"] = (float(num_z / A), float(num_y / A))
        return self._cache["centroid"]

    def _moments_about_origin(self) -> tuple[float, float, float]:
        if "moments0" not in self._cache:
            Iyy, Izz, Iyz = polygon_second_moments(self._ext)
            for h in self._holes:
                # Force CCW so the moment-integral sign is positive;
                # then subtract.
                h_ccw = h if shoelace_area(h) > 0 else list(reversed(h))
                Iyy_h, Izz_h, Iyz_h = polygon_second_moments(h_ccw)
                Iyy -= Iyy_h
                Izz -= Izz_h
                Iyz -= Iyz_h
            self._cache["moments0"] = (Iyy, Izz, Iyz)
        return self._cache["moments0"]

    @property
    def I_zz(self) -> float:
        Iyy_0, Izz_0, _ = self._moments_about_origin()
        cz, cy = self.centroid
        # Parallel-axis correction: I_zz = I_zz_0 - A * y_c^2
        return float(Izz_0 - self.area * cy * cy)

    @property
    def I_yy(self) -> float:
        Iyy_0, _, _ = self._moments_about_origin()
        cz, cy = self.centroid
        return float(Iyy_0 - self.area * cz * cz)

    @property
    def I_yz(self) -> float:
        _, _, Iyz_0 = self._moments_about_origin()
        cz, cy = self.centroid
        return float(Iyz_0 - self.area * cz * cy)

    # ------------------------------------------------------ transforms
    def translate(self, dz: float, dy: float) -> "PolygonGeometry":
        ext = [(z + dz, y + dy) for (z, y) in self._ext]
        holes = [
            [(z + dz, y + dy) for (z, y) in h] for h in self._holes
        ]
        return PolygonGeometry(ext, holes=holes)

    # ------------------------------------------------------ factories
    @classmethod
    def rectangle(
        cls,
        width: float,
        height: float,
        *,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> "PolygonGeometry":
        """Centered rectangle of (``width``, ``height``) in (z, y)."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        cz, cy = center
        zl, zr = cz - width / 2, cz + width / 2
        yb, yt = cy - height / 2, cy + height / 2
        return cls([(zl, yb), (zr, yb), (zr, yt), (zl, yt)])

    @classmethod
    def hollow_rectangle(
        cls,
        outer_width: float, outer_height: float,
        inner_width: float, inner_height: float,
        *,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> "PolygonGeometry":
        """Centered hollow rectangle (rectangular tube)."""
        if not (outer_width > inner_width > 0):
            raise ValueError("require outer_width > inner_width > 0")
        if not (outer_height > inner_height > 0):
            raise ValueError("require outer_height > inner_height > 0")
        cz, cy = center
        ext_zl, ext_zr = cz - outer_width / 2, cz + outer_width / 2
        ext_yb, ext_yt = cy - outer_height / 2, cy + outer_height / 2
        int_zl, int_zr = cz - inner_width / 2, cz + inner_width / 2
        int_yb, int_yt = cy - inner_height / 2, cy + inner_height / 2
        ext = [(ext_zl, ext_yb), (ext_zr, ext_yb),
               (ext_zr, ext_yt), (ext_zl, ext_yt)]
        hole = [(int_zl, int_yb), (int_zr, int_yb),
                (int_zr, int_yt), (int_zl, int_yt)]
        return cls(ext, holes=[hole])

    def __repr__(self) -> str:
        return (
            f"PolygonGeometry(area={self.area:.4g}, "
            f"I_zz={self.I_zz:.4g}, I_yy={self.I_yy:.4g})"
        )
