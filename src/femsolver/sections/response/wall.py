"""Wall fiber sections with confined boundary elements.

Reinforced-concrete shear walls in tall-building practice are
analysed with fiber sections whose layout differs from a beam in
three ways:

1. **Two concrete materials** -- a *confined* concrete at the two
   ends (the boundary elements, where transverse hoops + cross-ties
   raise the concrete's f_c and ultimate strain) and an *unconfined*
   concrete in the middle (web).
2. **Smeared vertical reinforcement** -- the vertical bars are
   represented as a uniform steel "shadow" over each region with
   reinforcement ratio rho. Each concrete fiber's area is reduced
   by ``(1 - rho)`` and a paired steel fiber of the same y-position
   carries the ``rho`` fraction.
3. **Boundary-element + web split** -- the cross-section is split
   into three regions: a left boundary of length ``L_be``, a web of
   length ``L_w - 2 L_be``, and a right boundary of length ``L_be``.
   Each region uses its own concrete and rho values.

The function :func:`wall_section_2d` builds this layout and returns a
standard :class:`FiberSection2D`. Such a section drives an ordinary
:class:`BeamColumn2D` (vertically oriented), giving the macro-fiber
wall element familiar from PERFORM-3D / OpenSees ``Wall``.

For 3D wall cross-sections (T, L, U, C, I shapes) see the
:func:`t_wall_section_3d`, :func:`l_wall_section_3d`,
:func:`u_wall_section_3d`, and :func:`i_wall_section_3d` factories.

References
----------
* Wallace, J.W. (1995) "Seismic design of RC structural walls. Part I:
  New code format." *J. Struct. Eng.*, 121(1), 75-87.
* Orakcal, K. & Wallace, J.W. (2006) "Flexural modeling of reinforced
  concrete walls -- experimental verification." *ACI Struct. J.*,
  103(2), 196-206.
* ACI 318-19 §18.10 "Special structural walls".
"""
from __future__ import annotations

from dataclasses import dataclass

from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.sections.response.fiber import (
    Fiber,
    FiberSection2D,
    FiberSection3D,
)


# ============================================================ wall region

@dataclass
class WallRegion:
    """One vertical region of a wall cross-section (boundary element or web).

    Attributes
    ----------
    y_start, y_end : float
        Region extent along the section's y-axis (m). ``y_start < y_end``.
    concrete : UniaxialMaterial
        Concrete material for this region (confined for boundary
        elements, unconfined for web).
    rebar_material : UniaxialMaterial
        Vertical reinforcement material.
    rho : float
        Vertical reinforcement ratio (steel area / gross area), in
        ``[0, 0.10]``. Boundary elements typically 0.01-0.06 per
        ACI 318 §18.10.6; web reinforcement minimum 0.0025.
    n_fibers : int
        Number of concrete fibers in this region (steel fibers
        are added at the same y-positions).
    """

    y_start: float
    y_end: float
    concrete: UniaxialMaterial
    rebar_material: UniaxialMaterial
    rho: float
    n_fibers: int


def _build_region_fibers(
    region: WallRegion,
    thickness: float,
) -> list[Fiber]:
    """Discretize ``region`` into concrete + steel fiber pairs.

    Each fiber strip has total area ``thickness * dy``. The concrete
    fiber takes ``(1 - rho)`` of that area; the steel fiber takes
    ``rho`` of that area, sharing the same y-coordinate.
    """
    if region.n_fibers < 1:
        raise ValueError(
            f"n_fibers must be >= 1, got {region.n_fibers}"
        )
    if region.y_end <= region.y_start:
        raise ValueError(
            f"region span empty: y_start={region.y_start} >= "
            f"y_end={region.y_end}"
        )
    if not (0.0 <= region.rho <= 0.10):
        raise ValueError(
            f"reinforcement ratio rho must be in [0, 0.10], got "
            f"{region.rho}"
        )

    L_region = region.y_end - region.y_start
    dy = L_region / region.n_fibers
    fibers: list[Fiber] = []
    for i in range(region.n_fibers):
        y = region.y_start + (i + 0.5) * dy
        strip_area = thickness * dy
        # Concrete and steel share the strip area in proportion (1-rho) / rho
        # Use shared materials with separate clones so each fiber has
        # independent history.
        if region.rho > 0.0:
            fibers.append(Fiber(
                y=y, z=0.0,
                area=strip_area * (1.0 - region.rho),
                material=region.concrete.clone(),
            ))
            fibers.append(Fiber(
                y=y, z=0.0,
                area=strip_area * region.rho,
                material=region.rebar_material.clone(),
            ))
        else:
            fibers.append(Fiber(
                y=y, z=0.0,
                area=strip_area,
                material=region.concrete.clone(),
            ))
    return fibers


# ============================================================ 2D wall

def wall_section_2d(
    *,
    L_w: float,
    t_w: float,
    L_be: float,
    web_concrete: UniaxialMaterial,
    boundary_concrete: UniaxialMaterial,
    rebar_material: UniaxialMaterial,
    web_rho: float = 0.0025,
    boundary_rho: float = 0.02,
    n_web_fibers: int = 20,
    n_be_fibers: int = 6,
) -> FiberSection2D:
    """Build a 2D wall fiber section with confined boundary elements.

    Cross-section layout (looking down the wall axis)::

        |<-L_be->|<------- L_w - 2 L_be -------->|<-L_be->|
        +========+================================+========+
        | confined |        web (unconfined)      | confined |
        +========+================================+========+
                <----------- L_w ----------->
                       (in y-direction)

    All fibers have the same z (= 0); the wall thickness ``t_w`` is
    multiplied into each fiber's strip area.

    Parameters
    ----------
    L_w : float
        Total wall length (m, in the section's y-direction).
    t_w : float
        Wall thickness (m, normal to the wall plane).
    L_be : float
        Boundary-element length at EACH end (m). Must satisfy
        ``2 L_be < L_w``.
    web_concrete : UniaxialMaterial
        Unconfined concrete (e.g. :class:`ConcreteKentPark`).
    boundary_concrete : UniaxialMaterial
        Confined concrete (e.g. :class:`ConcreteMander`).
    rebar_material : UniaxialMaterial
        Vertical reinforcement material (e.g.
        :class:`UniaxialMenegottoPinto`).
    web_rho : float, default 0.0025
        Web vertical reinforcement ratio (ACI 318 §11.6 minimum).
    boundary_rho : float, default 0.02
        Boundary-element vertical reinforcement ratio.
    n_web_fibers : int, default 20
    n_be_fibers : int, default 6

    Returns
    -------
    FiberSection2D
        Centred on the wall centroid so ``y in [-L_w/2, +L_w/2]``.
    """
    if L_w <= 0.0:
        raise ValueError(f"L_w must be positive, got {L_w}")
    if t_w <= 0.0:
        raise ValueError(f"t_w must be positive, got {t_w}")
    if L_be <= 0.0:
        raise ValueError(f"L_be must be positive, got {L_be}")
    if 2.0 * L_be >= L_w:
        raise ValueError(
            f"2 L_be ({2*L_be}) must be < L_w ({L_w})"
        )

    half = L_w / 2.0
    left_be = WallRegion(
        y_start=-half, y_end=-half + L_be,
        concrete=boundary_concrete,
        rebar_material=rebar_material,
        rho=boundary_rho,
        n_fibers=n_be_fibers,
    )
    web = WallRegion(
        y_start=-half + L_be, y_end=half - L_be,
        concrete=web_concrete,
        rebar_material=rebar_material,
        rho=web_rho,
        n_fibers=n_web_fibers,
    )
    right_be = WallRegion(
        y_start=half - L_be, y_end=half,
        concrete=boundary_concrete,
        rebar_material=rebar_material,
        rho=boundary_rho,
        n_fibers=n_be_fibers,
    )

    fibers: list[Fiber] = []
    for region in (left_be, web, right_be):
        fibers.extend(_build_region_fibers(region, thickness=t_w))
    return FiberSection2D(fibers)


# ============================================================ 3D wall builders

def _rect_fibers_3d(
    *,
    y_centre: float,
    z_centre: float,
    width_y: float,
    width_z: float,
    n_y: int,
    n_z: int,
    concrete: UniaxialMaterial,
    rebar_material: UniaxialMaterial,
    rho: float,
) -> list[Fiber]:
    """Discretize a rectangle into ``n_y x n_z`` smeared concrete+steel fibers.

    The rectangle is centred at (y_centre, z_centre) with side lengths
    ``width_y, width_z`` along (y, z). Each cell carries a concrete
    fiber + (if rho > 0) a steel fiber at the same (y, z).
    """
    if n_y < 1 or n_z < 1:
        raise ValueError(f"n_y, n_z must be >= 1, got {(n_y, n_z)}")
    if width_y <= 0.0 or width_z <= 0.0:
        raise ValueError("widths must be positive")
    dy = width_y / n_y
    dz = width_z / n_z
    fibers: list[Fiber] = []
    for i in range(n_y):
        y = y_centre - 0.5 * width_y + (i + 0.5) * dy
        for j in range(n_z):
            z = z_centre - 0.5 * width_z + (j + 0.5) * dz
            cell_area = dy * dz
            if rho > 0.0:
                fibers.append(Fiber(
                    y=y, z=z,
                    area=cell_area * (1.0 - rho),
                    material=concrete.clone(),
                ))
                fibers.append(Fiber(
                    y=y, z=z,
                    area=cell_area * rho,
                    material=rebar_material.clone(),
                ))
            else:
                fibers.append(Fiber(
                    y=y, z=z, area=cell_area,
                    material=concrete.clone(),
                ))
    return fibers


def t_wall_section_3d(
    *,
    web_length: float, web_thickness: float,
    flange_length: float, flange_thickness: float,
    L_be: float,
    web_concrete: UniaxialMaterial,
    boundary_concrete: UniaxialMaterial,
    rebar_material: UniaxialMaterial,
    web_rho: float = 0.0025,
    boundary_rho: float = 0.02,
    n_web_y: int = 16, n_web_z: int = 2,
    n_flange_y: int = 12, n_flange_z: int = 2,
    n_be_y: int = 4, n_be_z: int = 2,
    GJ: float = 1.0e8,
) -> FiberSection3D:
    """T-shaped reinforced-concrete wall section.

    Layout (in the section y-z plane, looking down the wall axis)::

        +-------------------------+ <- flange
        |  unconfined concrete    |
        +---+-------------+-------+
            |             |
            |     web     |
            |  unconfined |
            |             |
            +=============+    <- bottom (boundary element)
                  |
              confined

    The flange runs along the y-axis (its length is ``flange_length``,
    thickness ``flange_thickness``). The web extends in the
    z-direction with the boundary element at its far end.

    Parameters
    ----------
    web_length, web_thickness : float
        Web dimensions (z-direction length, y-direction thickness).
    flange_length, flange_thickness : float
        Flange dimensions (y-direction length, z-direction thickness).
    L_be : float
        Boundary-element length at the web tip (m).
    web_concrete, boundary_concrete, rebar_material : UniaxialMaterial
    web_rho, boundary_rho : float
    n_web_y, n_web_z, n_flange_y, n_flange_z, n_be_y, n_be_z : int
        Discretisation parameters per region.
    GJ : float
        Torsional rigidity (St-Venant approximation).

    Returns
    -------
    FiberSection3D
    """
    if L_be >= web_length:
        raise ValueError(
            f"L_be ({L_be}) must be < web_length ({web_length})"
        )

    # Flange: centred at (y=0, z=+web_length/2), spans (-flange_length/2,
    # +flange_length/2) in y and (web_length/2, web_length/2 + flange_thickness)
    # in z. Simplification: flange contains both unconfined concrete +
    # smeared reinforcement; no separate confined region at flange ends.
    fibers: list[Fiber] = []
    flange_z_centre = web_length / 2.0 + flange_thickness / 2.0
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0, z_centre=flange_z_centre,
        width_y=flange_length, width_z=flange_thickness,
        n_y=n_flange_y, n_z=n_flange_z,
        concrete=web_concrete, rebar_material=rebar_material,
        rho=web_rho,
    ))

    # Web (unconfined part, in z-direction from 0 to web_length - L_be)
    web_unconf_length = web_length - L_be
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0,
        z_centre=-web_length / 2.0 + web_unconf_length / 2.0,
        width_y=web_thickness, width_z=web_unconf_length,
        n_y=n_web_y, n_z=n_web_z,
        concrete=web_concrete, rebar_material=rebar_material,
        rho=web_rho,
    ))

    # Boundary element at the web tip (confined concrete)
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0,
        z_centre=-web_length / 2.0 + web_unconf_length + L_be / 2.0,
        width_y=web_thickness, width_z=L_be,
        n_y=n_be_y, n_z=n_be_z,
        concrete=boundary_concrete, rebar_material=rebar_material,
        rho=boundary_rho,
    ))

    return FiberSection3D(fibers, GJ=GJ)


def l_wall_section_3d(
    *,
    leg1_length: float, leg1_thickness: float,
    leg2_length: float, leg2_thickness: float,
    L_be: float,
    web_concrete: UniaxialMaterial,
    boundary_concrete: UniaxialMaterial,
    rebar_material: UniaxialMaterial,
    web_rho: float = 0.0025,
    boundary_rho: float = 0.02,
    n_leg_y: int = 12, n_leg_z: int = 2,
    n_be_y: int = 4, n_be_z: int = 2,
    GJ: float = 1.0e8,
) -> FiberSection3D:
    """L-shaped wall section (two perpendicular legs).

    Leg 1 along the +y axis, leg 2 along the +z axis, joined at the
    corner. A boundary element is placed at the FREE end of each leg.

    Parameters
    ----------
    leg1_length, leg1_thickness : float
        Leg 1 (along y) dimensions.
    leg2_length, leg2_thickness : float
        Leg 2 (along z) dimensions.
    L_be : float
        Boundary-element length at each leg's free end.
    """
    if L_be >= leg1_length or L_be >= leg2_length:
        raise ValueError("L_be must be < both leg lengths")

    fibers: list[Fiber] = []

    # Leg 1: along +y, from y=0 to y=leg1_length, z centred on 0
    leg1_unconf = leg1_length - L_be
    fibers.extend(_rect_fibers_3d(
        y_centre=leg1_unconf / 2.0, z_centre=0.0,
        width_y=leg1_unconf, width_z=leg1_thickness,
        n_y=n_leg_y, n_z=n_leg_z,
        concrete=web_concrete, rebar_material=rebar_material,
        rho=web_rho,
    ))
    fibers.extend(_rect_fibers_3d(
        y_centre=leg1_unconf + L_be / 2.0, z_centre=0.0,
        width_y=L_be, width_z=leg1_thickness,
        n_y=n_be_y, n_z=n_be_z,
        concrete=boundary_concrete, rebar_material=rebar_material,
        rho=boundary_rho,
    ))

    # Leg 2: along +z, from z=leg2_thickness/2 upward (avoid double-count
    # at the corner). For simplicity, span [leg1_thickness/2, leg2_length].
    leg2_unconf = leg2_length - L_be
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0,
        z_centre=leg1_thickness / 2.0 + leg2_unconf / 2.0,
        width_y=leg2_thickness,
        width_z=leg2_unconf,
        n_y=n_leg_z, n_z=n_leg_y,
        concrete=web_concrete, rebar_material=rebar_material,
        rho=web_rho,
    ))
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0,
        z_centre=leg1_thickness / 2.0 + leg2_unconf + L_be / 2.0,
        width_y=leg2_thickness,
        width_z=L_be,
        n_y=n_leg_z, n_z=n_be_y,
        concrete=boundary_concrete, rebar_material=rebar_material,
        rho=boundary_rho,
    ))

    return FiberSection3D(fibers, GJ=GJ)


def i_wall_section_3d(
    *,
    web_length: float, web_thickness: float,
    flange_length: float, flange_thickness: float,
    L_be: float,
    web_concrete: UniaxialMaterial,
    boundary_concrete: UniaxialMaterial,
    rebar_material: UniaxialMaterial,
    web_rho: float = 0.0025,
    boundary_rho: float = 0.02,
    n_web_y: int = 4, n_web_z: int = 12,
    n_flange_y: int = 8, n_flange_z: int = 2,
    GJ: float = 1.0e8,
) -> FiberSection3D:
    """I-shaped wall section: web with flanges (acting as boundary elements)
    at both ends.

    The two flanges act as confined boundary elements at each end of
    the web. This is the canonical 'barbell' shear wall of seismic
    codes.

    Layout (y-z plane, looking down the wall axis)::

       +============+         flange (top, confined)
       +============+
             ||
             ||  web (unconfined)
             ||
       +============+         flange (bot, confined)
       +============+
    """
    fibers: list[Fiber] = []

    # Top flange (confined)
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0,
        z_centre=web_length / 2.0 + flange_thickness / 2.0,
        width_y=flange_length, width_z=flange_thickness,
        n_y=n_flange_y, n_z=n_flange_z,
        concrete=boundary_concrete, rebar_material=rebar_material,
        rho=boundary_rho,
    ))
    # Web (unconfined)
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0, z_centre=0.0,
        width_y=web_thickness, width_z=web_length,
        n_y=n_web_y, n_z=n_web_z,
        concrete=web_concrete, rebar_material=rebar_material,
        rho=web_rho,
    ))
    # Bottom flange (confined)
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0,
        z_centre=-web_length / 2.0 - flange_thickness / 2.0,
        width_y=flange_length, width_z=flange_thickness,
        n_y=n_flange_y, n_z=n_flange_z,
        concrete=boundary_concrete, rebar_material=rebar_material,
        rho=boundary_rho,
    ))

    return FiberSection3D(fibers, GJ=GJ)


def u_wall_section_3d(
    *,
    web_length: float, web_thickness: float,
    flange_length: float, flange_thickness: float,
    L_be: float,
    web_concrete: UniaxialMaterial,
    boundary_concrete: UniaxialMaterial,
    rebar_material: UniaxialMaterial,
    web_rho: float = 0.0025,
    boundary_rho: float = 0.02,
    n_web_y: int = 4, n_web_z: int = 12,
    n_flange_y: int = 8, n_flange_z: int = 2,
    n_be_y: int = 4, n_be_z: int = 2,
    GJ: float = 1.0e8,
) -> FiberSection3D:
    """U-shaped (channel) wall section: web with one flange returning
    on both sides.

    Layout::

       +========+              flange returns at the open end
       +        +
       +   web  +
       +        +
       +========+
    """
    fibers: list[Fiber] = []

    # Web (unconfined) -- runs from -web_length/2 to +web_length/2 in z
    fibers.extend(_rect_fibers_3d(
        y_centre=0.0, z_centre=0.0,
        width_y=web_thickness, width_z=web_length,
        n_y=n_web_y, n_z=n_web_z,
        concrete=web_concrete, rebar_material=rebar_material,
        rho=web_rho,
    ))

    # Top flange (perpendicular to web at +z end)
    fibers.extend(_rect_fibers_3d(
        y_centre=flange_length / 2.0,
        z_centre=web_length / 2.0 + flange_thickness / 2.0,
        width_y=flange_length, width_z=flange_thickness,
        n_y=n_flange_y, n_z=n_flange_z,
        concrete=boundary_concrete, rebar_material=rebar_material,
        rho=boundary_rho,
    ))
    # Bottom flange (perpendicular to web at -z end)
    fibers.extend(_rect_fibers_3d(
        y_centre=flange_length / 2.0,
        z_centre=-web_length / 2.0 - flange_thickness / 2.0,
        width_y=flange_length, width_z=flange_thickness,
        n_y=n_flange_y, n_z=n_flange_z,
        concrete=boundary_concrete, rebar_material=rebar_material,
        rho=boundary_rho,
    ))

    return FiberSection3D(fibers, GJ=GJ)
