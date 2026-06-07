"""Factory functions that wrap a parametric :class:`Geometry` into a
unified :class:`Section`.

Each factory takes parametric dimensions and an optional ``material``.
The returned ``Section`` is ready for analysis (via
``elastic_section_2d/3d``), design (via ``as_aci_section()`` etc.,
landing in II.6), reports, and BOM.

Naming convention for auto-generated section names:
* rectangle:    ``"R 300x600"``  (b in mm x h in mm)
* I-shape:      ``"I 400x180x14x8.5"``  (h x b x tf x tw)
* T-shape:      ``"T 300x150x12x8"``
* channel:      ``"C 300x80x12x8"``
* angle:        ``"L 100x100x10"``  (a x b x t)
* hollow rect:  ``"RHS 200x100x6"``
* circular:     ``"D 500"``
* hollow circ:  ``"CHS 500x10"``  (D x t)
"""
from __future__ import annotations

from typing import Any, Optional

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
from femsolver.sections.section import (
    MaterialZone,
    ReinforcementLayout,
    Section,
)


def _mm(x: float) -> int:
    """Format a meter dimension as a millimetre integer for naming."""
    return int(round(x * 1000))


def _build(
    geometry,
    *,
    family: str,
    name: str,
    material: Any,
    catalogue_ref: Optional[str] = None,
) -> Section:
    zones = []
    if material is not None:
        zones.append(MaterialZone(material=material, name=family))
    return Section(
        geometry=geometry,
        zones=zones,
        name=name,
        family=family,
        catalogue_ref=catalogue_ref,
    )


# ============================================================ factories

def rectangular_section(
    *,
    b: float,
    h: float,
    material: Any = None,
    name: Optional[str] = None,
) -> Section:
    """Solid rectangle of width ``b`` x height ``h`` (m)."""
    g = RectangularGeometry(width=b, height=h)
    return _build(
        g, family="rect",
        name=name or f"R {_mm(b)}x{_mm(h)}",
        material=material,
    )


def i_section(
    *,
    h: float, b: float, t_f: float, t_w: float,
    material: Any = None,
    name: Optional[str] = None,
    catalogue_ref: Optional[str] = None,
) -> Section:
    """Doubly-symmetric I-shape."""
    g = ISectionGeometry(h=h, b=b, t_f=t_f, t_w=t_w)
    return _build(
        g, family="I",
        name=name or f"I {_mm(h)}x{_mm(b)}x{t_f*1000:g}x{t_w*1000:g}",
        material=material,
        catalogue_ref=catalogue_ref,
    )


def t_section(
    *,
    h: float, b: float, t_f: float, t_w: float,
    material: Any = None,
    name: Optional[str] = None,
) -> Section:
    g = TSectionGeometry(h=h, b=b, t_f=t_f, t_w=t_w)
    return _build(
        g, family="T",
        name=name or f"T {_mm(h)}x{_mm(b)}x{t_f*1000:g}x{t_w*1000:g}",
        material=material,
    )


def channel_section(
    *,
    h: float, b: float, t_f: float, t_w: float,
    material: Any = None,
    name: Optional[str] = None,
) -> Section:
    g = ChannelGeometry(h=h, b=b, t_f=t_f, t_w=t_w)
    return _build(
        g, family="channel",
        name=name or f"C {_mm(h)}x{_mm(b)}x{t_f*1000:g}x{t_w*1000:g}",
        material=material,
    )


def angle_section(
    *,
    a: float, b: float, t: float,
    material: Any = None,
    name: Optional[str] = None,
) -> Section:
    g = AngleGeometry(a=a, b=b, t=t)
    return _build(
        g, family="angle",
        name=name or f"L {_mm(a)}x{_mm(b)}x{t*1000:g}",
        material=material,
    )


def hollow_rect_section(
    *,
    b: float, h: float, t: float,
    material: Any = None,
    name: Optional[str] = None,
) -> Section:
    g = HollowRectGeometry(width=b, height=h, t=t)
    return _build(
        g, family="hollow_rect",
        name=name or f"RHS {_mm(b)}x{_mm(h)}x{t*1000:g}",
        material=material,
    )


def circular_section(
    *,
    D: float,
    material: Any = None,
    name: Optional[str] = None,
    n_sides: int = 64,
) -> Section:
    g = CircularGeometry(D=D, n_sides=n_sides)
    return _build(
        g, family="circular",
        name=name or f"D {_mm(D)}",
        material=material,
    )


def hollow_circular_section(
    *,
    D: float, t: float,
    material: Any = None,
    name: Optional[str] = None,
    n_sides: int = 64,
) -> Section:
    g = HollowCircularGeometry(D=D, t=t, n_sides=n_sides)
    return _build(
        g, family="hollow_circular",
        name=name or f"CHS {_mm(D)}x{t*1000:g}",
        material=material,
    )


# ============================================================ RC composition (II.6)

def rc_rectangular_section(
    *,
    b: float,
    h: float,
    concrete: Any = None,
    reinforcement: Optional[ReinforcementLayout] = None,
    name: Optional[str] = None,
) -> Section:
    """Build a unified RC section: rectangular concrete with rebar layout.

    Parameters
    ----------
    b, h : float
        Section width (z) and depth (y), m.
    concrete : Material
        Concrete material reference (typically a
        :class:`femsolver.design.concrete.section.ConcreteMaterial`).
    reinforcement : ReinforcementLayout, optional
        Rebar layout. Use
        :meth:`ReinforcementLayout.from_rectangular_layers` to build
        from layer descriptions.
    name : str, optional
        Section name. Defaults to ``"RC {b_mm}x{h_mm}"``.

    Example
    -------
    >>> from femsolver.sections import (
    ...     rc_rectangular_section, ReinforcementLayout,
    ... )
    >>> rebar = ReinforcementLayout.from_rectangular_layers(
    ...     b=0.3, h=0.6,
    ...     bottom_bars=[(510e-6, "#8"), (510e-6, "#8"), (510e-6, "#8")],
    ...     top_bars=[(285e-6, "#6"), (285e-6, "#6")],
    ...     bottom_cover=0.040, top_cover=0.040,
    ... )
    >>> sec = rc_rectangular_section(
    ...     b=0.3, h=0.6, concrete=c30, reinforcement=rebar, name="B1",
    ... )
    """
    g = RectangularGeometry(width=b, height=h)
    zones = []
    if concrete is not None:
        zones.append(MaterialZone(material=concrete, name="concrete"))
    return Section(
        geometry=g, zones=zones,
        name=name or f"RC {_mm(b)}x{_mm(h)}",
        family="rect",
        reinforcement=reinforcement,
    )
