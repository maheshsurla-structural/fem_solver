"""Composite girder + deck cross-section properties.

A common bridge cross-section is a precast / cast-in-place concrete
girder topped with a cast-in-place RC deck of different concrete
strength (and hence different modulus). The composite section
properties are obtained by transforming the deck into "equivalent
girder concrete" using the modular ratio ``n = E_deck / E_girder``,
then computing area, centroid, moment of inertia, and section moduli
of the transformed compound section about its own centroid.

For prestressed concrete, the same transformation is applied with
``n_p = E_p / E_c`` for the prestressing strand, treated as concentrated
at the strand centroid. Concrete fiber stresses at the top/bottom of
the composite section under combined dead, live, prestress, and
secondary loads use the transformed section properties.

This module is deliberately analytical (closed-form composite of two
rectangles + strand point) for clarity. For arbitrary shapes, build
a :class:`~femsolver.sections.response.fiber.FiberSection2D` with appropriate
fiber materials at each location.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femsolver.bridges._utils import require_positive


@dataclass
class CompositeSectionProps:
    """Transformed-section properties of a girder + deck.

    Geometry assumed with the GIRDER at the bottom and the DECK on top.

    Coordinate convention: y is measured from the bottom of the girder
    upward. The composite centroid is reported as y_bar.

    Attributes
    ----------
    A_g : float
        Girder area (m^2).
    A_d_trans : float
        Deck area, transformed to girder-concrete-equivalent (m^2).
    A_t : float
        Total transformed area (m^2).
    y_bar : float
        Centroid of the composite section (m, from girder bottom).
    I_t : float
        Composite moment of inertia about the centroid (m^4).
    S_t_top : float
        Section modulus to the top of the deck (m^3, positive).
    S_t_bot : float
        Section modulus to the bottom of the girder (m^3, positive).
    n : float
        Modular ratio E_deck / E_girder.
    girder_height : float
        Girder depth (m).
    deck_thickness : float
        Deck thickness (m).
    total_height : float
        ``girder_height + deck_thickness`` (m).
    """

    A_g: float
    A_d_trans: float
    A_t: float
    y_bar: float
    I_t: float
    S_t_top: float
    S_t_bot: float
    n: float
    girder_height: float
    deck_thickness: float
    total_height: float


def composite_girder_deck(
    *,
    girder_area: float, girder_I: float,
    girder_y_centroid: float, girder_height: float,
    deck_width: float, deck_thickness: float,
    E_girder: float, E_deck: float,
) -> CompositeSectionProps:
    """Transformed composite-section properties for a girder + deck.

    The girder is described by its area, moment of inertia about its
    own centroid, and the location of its centroid (``girder_y_centroid``,
    measured from the girder bottom). The deck is assumed rectangular
    of ``deck_width`` x ``deck_thickness``, placed centrally on top of
    the girder.

    Parameters
    ----------
    girder_area : float (m^2)
    girder_I : float (m^4)
    girder_y_centroid : float (m, from girder bottom)
    girder_height : float (m)
    deck_width : float (m)
    deck_thickness : float (m)
    E_girder : float (Pa)
    E_deck : float (Pa)
    """
    require_positive(
        girder_area=girder_area, girder_I=girder_I,
        girder_y_centroid=girder_y_centroid,
        girder_height=girder_height,
        deck_width=deck_width, deck_thickness=deck_thickness,
        E_girder=E_girder, E_deck=E_deck,
    )
    n = E_deck / E_girder
    A_d_trans = n * deck_width * deck_thickness
    A_t = girder_area + A_d_trans

    # Centroid (about girder bottom)
    y_g = girder_y_centroid
    y_d = girder_height + deck_thickness / 2.0
    y_bar = (girder_area * y_g + A_d_trans * y_d) / A_t

    # Moment of inertia about composite centroid (parallel-axis)
    I_g_centred = girder_I + girder_area * (y_g - y_bar) ** 2
    I_d_self = (deck_width * deck_thickness ** 3 / 12.0) * n
    I_d_centred = I_d_self + A_d_trans * (y_d - y_bar) ** 2
    I_t = I_g_centred + I_d_centred

    # Distances to extreme fibres
    c_top = (girder_height + deck_thickness) - y_bar
    c_bot = y_bar
    S_t_top = I_t / c_top
    S_t_bot = I_t / c_bot

    return CompositeSectionProps(
        A_g=float(girder_area),
        A_d_trans=float(A_d_trans),
        A_t=float(A_t),
        y_bar=float(y_bar),
        I_t=float(I_t),
        S_t_top=float(S_t_top),
        S_t_bot=float(S_t_bot),
        n=float(n),
        girder_height=float(girder_height),
        deck_thickness=float(deck_thickness),
        total_height=float(girder_height + deck_thickness),
    )


# ============================================================ unified Section (II.7)

def composite_girder_deck_section(
    *,
    girder_width: float,
    girder_height: float,
    deck_width: float,
    deck_thickness: float,
    girder_material,
    deck_material,
    name: str = "composite_girder_deck",
):
    """Build a unified :class:`~femsolver.sections.Section` for a
    composite girder + deck cross-section.

    Theme II.7 migration helper. The returned ``Section`` has two
    :class:`~femsolver.sections.MaterialZone` entries -- one for the
    girder, one for the deck -- and the geometry is the union of
    the two rectangular polygons. The girder is positioned with its
    centroid at y = 0 (the natural location for a Bernoulli beam
    integration), and the deck sits on top.

    For backward-compatibility with the existing closed-form
    transformed-section workflow, see :func:`composite_girder_deck`,
    which is unchanged. This function is purely additive.

    Parameters
    ----------
    girder_width, girder_height : float
        Girder rectangle dimensions (m). The girder is taken as a
        simple rectangle; for I-shaped girders, build the geometry
        via :func:`femsolver.sections.parametric.i_section` and
        compose manually.
    deck_width, deck_thickness : float
        Deck slab dimensions (m).
    girder_material, deck_material : material references
        Materials assigned to each zone. Typically two different
        concrete grades.
    name : str
        Section name.

    Notes
    -----
    The unified Section's gross properties come from the polygon
    union, NOT the transformed-section formulas. For prestressed
    bridge engineering use the legacy :func:`composite_girder_deck`
    which carries the modular ratio.
    """
    from femsolver.sections import (
        MaterialZone,
        PolygonGeometry,
        Section,
        union_polygons,
    )

    # Position: girder centroid at origin, deck on top
    girder_geom = PolygonGeometry.rectangle(
        width=girder_width, height=girder_height,
    )
    deck_geom = PolygonGeometry.rectangle(
        width=deck_width, height=deck_thickness,
        center=(0.0, (girder_height + deck_thickness) / 2.0),
    )
    composite_geom = union_polygons(girder_geom, deck_geom)
    return Section(
        geometry=composite_geom,
        zones=[
            MaterialZone(
                material=girder_material, geometry=girder_geom,
                name="girder",
            ),
            MaterialZone(
                material=deck_material, geometry=deck_geom,
                name="deck",
            ),
        ],
        name=name,
        family="composite_girder_deck",
    )


@dataclass
class CompositeFiberStress:
    """Concrete fiber stresses in a composite section under combined
    axial + moment.

    The stresses are signed (compression negative under usual bridge
    convention where positive moment = sagging, positive axial =
    tension), and reported at four locations: top of deck, top of
    girder (= bottom of deck), bottom of girder, and centre of gravity
    of strand.
    """

    sigma_top_deck: float
    sigma_top_girder: float
    sigma_bot_girder: float
    sigma_at_strand: float


def composite_fiber_stresses(
    *,
    props: CompositeSectionProps,
    P: float, M: float,
    strand_y_from_bottom: float,
) -> CompositeFiberStress:
    """Compute concrete fiber stresses at four key fibers.

    The stresses are in **girder-concrete units** at the top of the
    deck the stress is divided by ``n`` to give the actual deck
    stress.

    Parameters
    ----------
    P : float
        Axial force on the composite section (N, positive tension).
    M : float
        Moment about the composite centroid (N·m, positive sagging).
    strand_y_from_bottom : float
        Distance from girder bottom to strand centroid (m).

    Notes
    -----
    Section geometry (heights) is read from ``props``; the strand
    location is supplied per-call since one section may host several
    tendons / cable groups at different elevations.
    """
    y_bar = props.y_bar
    inv_A_t = 1.0 / props.A_t
    M_over_I = M / props.I_t
    # Distances from composite centroid (positive upward); concrete
    # stress = P/A - M·y/I (positive tension under our M-sagging sign).
    y_top_deck = props.total_height - y_bar
    y_top_girder = props.girder_height - y_bar
    y_bot_girder = -y_bar
    y_strand = strand_y_from_bottom - y_bar
    # Single vectorised evaluation across the four fibres.
    ys = np.array([y_top_deck, y_top_girder, y_bot_girder, y_strand])
    sigma = P * inv_A_t - M_over_I * ys
    # Top of the deck reports the actual (softer) deck-concrete stress.
    return CompositeFiberStress(
        sigma_top_deck=float(sigma[0] / props.n),
        sigma_top_girder=float(sigma[1]),
        sigma_bot_girder=float(sigma[2]),
        sigma_at_strand=float(sigma[3]),
    )
