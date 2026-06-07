"""Parametric :class:`Geometry` subclasses with closed-form properties.

Each class:

* takes parametric dimensions in its constructor
* builds the underlying polygon outline (passed to
  :class:`PolygonGeometry`)
* overrides ``J`` with the textbook closed-form torsion constant
* overrides ``Z_zz`` / ``Z_yy`` with the closed-form plastic moduli
  where they exist (rectangles, I-shapes, circles, tubes)
* exposes the parametric dimensions as plain attributes for downstream
  introspection (design codes, BOM, reports)

References
----------
* Roark's Formulas for Stress and Strain, 7th ed. -- torsion of
  prismatic bars (Table 10.7 for non-circular sections)
* Boresi & Schmidt 6th ed. -- St-Venant torsion of thin-walled open
  and closed sections
* AISC Design Guide 9 -- Torsional Analysis of Structural Steel
  Members
"""
from __future__ import annotations

import math

from femsolver.sections.geometry.polygon import PolygonGeometry


# ============================================================ Rectangle

class RectangularGeometry(PolygonGeometry):
    """Solid rectangle of ``width`` (z) by ``height`` (y), centred at
    origin.

    Closed-form overrides:
        J = a·b^3·(1/3 - 0.21·(b/a)·(1 - (b/a)^4 / 12))   (Roark 10.7.1)
            where a = max(width,height), b = min(width,height)
        Z_zz = width·height^2 / 4   (Boresi & Schmidt, plastic modulus)
        Z_yy = height·width^2 / 4
    """
    def __init__(self, width: float, height: float):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.b = float(width)    # z-extent
        self.h = float(height)   # y-extent
        super().__init__([
            (-self.b/2, -self.h/2), (self.b/2, -self.h/2),
            (self.b/2,  self.h/2), (-self.b/2,  self.h/2),
        ])

    @property
    def J(self) -> float:
        """Roark Table 10.7 Case 1 (closed form for solid rectangle)."""
        a = max(self.b, self.h)
        b = min(self.b, self.h)
        ratio = b / a
        return float(a * b**3 * (1.0/3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0)))

    @property
    def Z_zz(self) -> float:
        return float(self.b * self.h**2 / 4.0)

    @property
    def Z_yy(self) -> float:
        return float(self.h * self.b**2 / 4.0)


# ============================================================ I-section

class ISectionGeometry(PolygonGeometry):
    """Doubly-symmetric I-shape: depth ``h``, flange width ``b``,
    flange thickness ``t_f``, web thickness ``t_w``.

    Coordinates: centroid at origin, web aligned with z = 0, flanges
    centred about y = +/-(h/2 - t_f/2).

    Closed-form J: open thin-walled approximation
        J ~ (2 b t_f^3 + (h - t_f) t_w^3) / 3        (Boresi & Schmidt 7.18)

    Closed-form Z_zz:
        Z_zz = b t_f (h - t_f) + t_w (h - 2 t_f)^2 / 4
    """
    def __init__(self, h: float, b: float, t_f: float, t_w: float):
        if h <= 0 or b <= 0 or t_f <= 0 or t_w <= 0:
            raise ValueError("all dimensions must be positive")
        if h <= 2 * t_f:
            raise ValueError("h must exceed 2 * t_f")
        if b <= t_w:
            raise ValueError("b must exceed t_w")
        self.h = float(h)
        self.b = float(b)
        self.t_f = float(t_f)
        self.t_w = float(t_w)

        # I-shape outline (10 vertices, CCW, starting bottom-left
        # of bottom flange):
        b2 = self.b / 2
        h2 = self.h / 2
        tw2 = self.t_w / 2
        h_web_half = (self.h - 2 * self.t_f) / 2
        outline = [
            (-b2,           -h2),
            ( b2,           -h2),
            ( b2,           -h2 + self.t_f),
            ( tw2,          -h2 + self.t_f),
            ( tw2,           h_web_half),
            ( b2,            h_web_half),
            ( b2,            h2),
            (-b2,            h2),
            (-b2,            h_web_half),
            (-tw2,           h_web_half),
            (-tw2,          -h2 + self.t_f),
            (-b2,           -h2 + self.t_f),
        ]
        super().__init__(outline)

    @property
    def J(self) -> float:
        return float((2 * self.b * self.t_f**3 + (self.h - self.t_f) * self.t_w**3) / 3.0)

    @property
    def I_zz(self) -> float:
        """Closed form: (b h^3 - (b - t_w)(h - 2 t_f)^3) / 12."""
        return float((self.b * self.h**3
                      - (self.b - self.t_w) * (self.h - 2 * self.t_f)**3) / 12.0)

    @property
    def I_yy(self) -> float:
        """2 (t_f b^3 / 12) + (h - 2 t_f) t_w^3 / 12."""
        return float(2 * (self.t_f * self.b**3 / 12.0)
                     + (self.h - 2 * self.t_f) * self.t_w**3 / 12.0)

    @property
    def Z_zz(self) -> float:
        """b t_f (h - t_f) + t_w (h - 2 t_f)^2 / 4."""
        return float(
            self.b * self.t_f * (self.h - self.t_f)
            + self.t_w * (self.h - 2 * self.t_f)**2 / 4.0
        )

    @property
    def Z_yy(self) -> float:
        """2 (t_f b^2 / 4) + (h - 2 t_f) t_w^2 / 4."""
        return float(
            2 * (self.t_f * self.b**2 / 4.0)
            + (self.h - 2 * self.t_f) * self.t_w**2 / 4.0
        )


# ============================================================ T-section

class TSectionGeometry(PolygonGeometry):
    """T-shape (flange at top): depth ``h`` (vertical), flange width
    ``b``, flange thickness ``t_f``, web thickness ``t_w``.

    The centroid is offset in y (toward the flange). We position the
    section so the outline is geometrically natural (bottom of stem
    at y = -h/2), then let :class:`PolygonGeometry` compute the
    centroid via Green's theorem.
    """
    def __init__(self, h: float, b: float, t_f: float, t_w: float):
        if h <= 0 or b <= 0 or t_f <= 0 or t_w <= 0:
            raise ValueError("all dimensions must be positive")
        if h <= t_f:
            raise ValueError("h must exceed t_f")
        if b <= t_w:
            raise ValueError("b must exceed t_w")
        self.h = float(h)
        self.b = float(b)
        self.t_f = float(t_f)
        self.t_w = float(t_w)

        b2 = self.b / 2
        tw2 = self.t_w / 2
        y_bot = -self.h / 2
        y_flange = self.h / 2 - self.t_f
        y_top = self.h / 2
        outline = [
            (-tw2,  y_bot),
            ( tw2,  y_bot),
            ( tw2,  y_flange),
            ( b2,   y_flange),
            ( b2,   y_top),
            (-b2,   y_top),
            (-b2,   y_flange),
            (-tw2,  y_flange),
        ]
        super().__init__(outline)

    @property
    def J(self) -> float:
        """Open thin-walled: (b t_f^3 + (h - t_f) t_w^3) / 3."""
        return float(
            (self.b * self.t_f**3 + (self.h - self.t_f) * self.t_w**3) / 3.0
        )


# ============================================================ Channel

class ChannelGeometry(PolygonGeometry):
    """C-shape (channel): depth ``h``, flange width ``b``, flange
    thickness ``t_f``, web thickness ``t_w``.

    The web is on the left (z < 0) face. The centroid is offset in z
    toward the web. Position so the left face of the web is at z = 0.
    """
    def __init__(self, h: float, b: float, t_f: float, t_w: float):
        if h <= 0 or b <= 0 or t_f <= 0 or t_w <= 0:
            raise ValueError("all dimensions must be positive")
        if h <= 2 * t_f:
            raise ValueError("h must exceed 2 * t_f")
        if b <= t_w:
            raise ValueError("b must exceed t_w")
        self.h = float(h)
        self.b = float(b)
        self.t_f = float(t_f)
        self.t_w = float(t_w)

        h2 = self.h / 2
        outline = [
            (0,         -h2),
            (self.b,    -h2),
            (self.b,    -h2 + self.t_f),
            (self.t_w,  -h2 + self.t_f),
            (self.t_w,   h2 - self.t_f),
            (self.b,     h2 - self.t_f),
            (self.b,     h2),
            (0,          h2),
        ]
        super().__init__(outline)

    @property
    def J(self) -> float:
        """Open thin-walled."""
        return float(
            (2 * self.b * self.t_f**3 + (self.h - 2 * self.t_f) * self.t_w**3) / 3.0
        )


# ============================================================ Angle (L)

class AngleGeometry(PolygonGeometry):
    """L-shape (single angle): leg dimensions ``a`` (vertical) and
    ``b`` (horizontal), thickness ``t``.

    The corner is at the origin. The centroid is offset in both z and
    y (toward the corner).
    """
    def __init__(self, a: float, b: float, t: float):
        if a <= 0 or b <= 0 or t <= 0:
            raise ValueError("a, b, t must be positive")
        if t >= min(a, b):
            raise ValueError("t must be less than min(a, b)")
        self.a = float(a)
        self.b = float(b)
        self.t = float(t)
        outline = [
            (0, 0), (self.b, 0), (self.b, self.t),
            (self.t, self.t), (self.t, self.a), (0, self.a),
        ]
        super().__init__(outline)

    @property
    def J(self) -> float:
        """Open thin-walled: (a + b) t^3 / 3."""
        return float((self.a + self.b) * self.t**3 / 3.0)


# ============================================================ Hollow rectangle

class HollowRectGeometry(PolygonGeometry):
    """Rectangular hollow section (RHS / SHS).

    Parameters
    ----------
    width, height : outer dimensions (m)
    t : wall thickness, uniform on all four sides (m)
    """
    def __init__(self, width: float, height: float, t: float):
        if width <= 0 or height <= 0 or t <= 0:
            raise ValueError("width, height, t must be positive")
        if t * 2 >= min(width, height):
            raise ValueError(
                "wall thickness too large (2t must be < min(width,height))"
            )
        self.b = float(width)
        self.h = float(height)
        self.t = float(t)
        # outer rectangle, CCW
        bo, ho = self.b / 2, self.h / 2
        ext = [(-bo, -ho), (bo, -ho), (bo, ho), (-bo, ho)]
        # inner rectangle (hole)
        bi = self.b / 2 - self.t
        hi = self.h / 2 - self.t
        hole = [(-bi, -hi), (bi, -hi), (bi, hi), (-bi, hi)]
        super().__init__(ext, holes=[hole])

    @property
    def J(self) -> float:
        """Bredt's formula for closed thin-walled tube:
            J = 4 A_m^2 t / s
        where A_m = (b - t)(h - t) is the enclosed area at the mean
        wall line, and s = 2((b - t) + (h - t)) is the perimeter at
        the mean wall line.
        """
        b_m = self.b - self.t
        h_m = self.h - self.t
        A_m = b_m * h_m
        s = 2.0 * (b_m + h_m)
        return float(4.0 * A_m**2 * self.t / s)

    @property
    def I_zz(self) -> float:
        """Closed form: (b_o h_o^3 - b_i h_i^3) / 12."""
        b_i = self.b - 2 * self.t
        h_i = self.h - 2 * self.t
        return float((self.b * self.h**3 - b_i * h_i**3) / 12.0)

    @property
    def I_yy(self) -> float:
        b_i = self.b - 2 * self.t
        h_i = self.h - 2 * self.t
        return float((self.h * self.b**3 - h_i * b_i**3) / 12.0)


# ============================================================ Circular

def _approx_circle(D: float, n: int = 64, center: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    """Approximate a circle of diameter D by an n-gon (CCW)."""
    r = D / 2
    cz, cy = center
    return [
        (cz + r * math.cos(2 * math.pi * i / n),
         cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


class CircularGeometry(PolygonGeometry):
    """Solid circular cross-section of diameter ``D``.

    Polygon outline is an n-gon (default 64 sides) used for visual
    rendering; gross properties override with the exact closed-form
    values so the polygon discretisation has no effect on numerics.
    """
    def __init__(self, D: float, *, n_sides: int = 64):
        if D <= 0:
            raise ValueError("D must be positive")
        if n_sides < 16:
            raise ValueError("n_sides must be >= 16 for reasonable rendering")
        self.D = float(D)
        super().__init__(_approx_circle(D, n_sides))

    @property
    def area(self) -> float:
        return float(math.pi * self.D**2 / 4.0)

    @property
    def I_zz(self) -> float:
        return float(math.pi * self.D**4 / 64.0)

    @property
    def I_yy(self) -> float:
        return float(math.pi * self.D**4 / 64.0)

    @property
    def I_yz(self) -> float:
        return 0.0

    @property
    def J(self) -> float:
        """Solid circle: J = pi D^4 / 32 (polar moment of inertia)."""
        return float(math.pi * self.D**4 / 32.0)

    @property
    def Z_zz(self) -> float:
        """Plastic modulus of a solid circle: D^3 / 6."""
        return float(self.D**3 / 6.0)

    @property
    def Z_yy(self) -> float:
        return float(self.D**3 / 6.0)


class HollowCircularGeometry(PolygonGeometry):
    """Hollow circular cross-section (CHS): outer diameter ``D``,
    wall thickness ``t``.
    """
    def __init__(self, D: float, t: float, *, n_sides: int = 64):
        if D <= 0 or t <= 0:
            raise ValueError("D and t must be positive")
        if t * 2 >= D:
            raise ValueError("wall thickness too large (2t must be < D)")
        if n_sides < 16:
            raise ValueError("n_sides must be >= 16")
        self.D = float(D)
        self.t = float(t)
        self.d = self.D - 2 * self.t          # inner diameter
        ext = _approx_circle(D, n_sides)
        # inner ring also CCW (constructor will flip to CW for shapely hole)
        hole = _approx_circle(self.d, n_sides)
        super().__init__(ext, holes=[hole])

    @property
    def area(self) -> float:
        return float(math.pi * (self.D**2 - self.d**2) / 4.0)

    @property
    def I_zz(self) -> float:
        return float(math.pi * (self.D**4 - self.d**4) / 64.0)

    @property
    def I_yy(self) -> float:
        return float(math.pi * (self.D**4 - self.d**4) / 64.0)

    @property
    def J(self) -> float:
        return float(math.pi * (self.D**4 - self.d**4) / 32.0)

    @property
    def Z_zz(self) -> float:
        """Plastic modulus of an annulus: (D^3 - d^3) / 6."""
        return float((self.D**3 - self.d**3) / 6.0)

    @property
    def Z_yy(self) -> float:
        return float((self.D**3 - self.d**3) / 6.0)
